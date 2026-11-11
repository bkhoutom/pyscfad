# Copyright 2026 The PySCFAD Authors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Host DF-CCSD and generalized lambda response, with resident fragment ERIs.

Only the callback boundary uses JAX. Numerical kernels use PySCF, NumPy and
BLAS, including the response action and the final integral adjoint. No PySCF
AO2MO builder or HDF5 lambda intermediates are used.
"""
import copy
from types import SimpleNamespace

import jax
import numpy as np
from pyscf import gto, lib
from pyscf.cc import ccsd, dfccsd
from pyscf.lib import logger
from threadpoolctl import threadpool_limits
from pyscfad.tools import resource_profile

from . import _native_ccsd_lambda as response
from . import _native_ccsd_eris as adjoint
from . import _native_kernels

ERI_NAMES = ('fock', 'mo_energy', 'oooo', 'ovoo', 'ovov', 'oovv',
             'ovvo', 'ovvv', 'Lov', 'Lvv')
BAR_NAMES = ('fock', 'mo_energy', 'oooo', 'ovoo', 'ovov', 'oovv',
             'ovvo', 'Lov', 'Lvv')


class _ResidentERIs(dfccsd._ChemistsERIs):
    """Use PySCF's packed-integral ladder when its RAM cache is available."""

    def _contract_vvvv_t2(self, mycc, t2, direct=False, out=None, verbose=None):
        assert not direct
        if self.vvvv is None:
            return super()._contract_vvvv_t2(mycc, t2, direct, out, verbose)
        result = _native_kernels.contract_vvvv(mycc, self.vvvv, t2, out)
        if result is not None:
            return result
        return ccsd._contract_vvvv_t2(mycc, self.mol, self.vvvv, t2, out, verbose)


def _maybe_cache_vvvv(mycc, eris):
    """Match the legacy 60%-of-remaining-total-memory cache policy.

    PySCF needs one float64 pair matrix; the JAX ladder uses two. The cache
    belongs to this host solve, so it is reused across all iterations and
    freed with the host ERIs. Backward builds it once for lambda and the
    integral adjoint, without retaining another quartic tensor on the AD tape.
    """
    eris.vvvv = None
    if mycc.vvvv_direct:
        return
    npair = eris.vvL.shape[0]
    cache_mb = eris.vvL.dtype.itemsize * npair**2 / 1e6
    remaining_mb = max(0., mycc.max_memory-lib.current_memory()[0])
    if cache_mb > .6 * remaining_mb:
        logger.info(mycc, 'Native CCSD vvvv cache needs %.1f MB; using DF tiles', cache_mb)
        return
    with resource_profile.section('ccsd.vvvv_cache', backend='numpy', cache_mb=cache_mb):
        eris.vvvv = eris.vvL @ eris.vvL.T
    logger.info(mycc, 'Native CCSD vvvv cache: %.1f MB in RAM', cache_mb)


def enabled(mycc):
    from pyscfad import config
    return (config.dfccsd_native and getattr(mycc, 'native_cpu', False)
            and config.dfccsd_custom_response
            and not mycc.dcsd and not mycc.cc2 and not mycc.direct)


def settings(mycc):
    """Capture scalar settings before entering a JAX callback worker thread."""
    from pyscfad import config
    # CC inherits the total process budget from the SCF object. Never use
    # mycc.with_df.max_memory here: that is a separate DF workspace budget.
    names = ('max_memory', 'max_cycle', 'conv_tol', 'conv_tol_normt',
             'iterative_damping', 'diis_space', 'diis_start_cycle',
             'diis_start_energy_diff', 'level_shift', 'verbose', 'vvvv_direct')
    values = {name: getattr(mycc, name) for name in names}
    values.update(diis=bool(mycc.diis), threads=lib.num_threads(),
                  stdout=mycc.stdout, cpp_kernels=config.dfccsd_cpp)
    return values


def host_objects(options, arrays, nocc):
    """Reuse local Fock/ERI tensors and optionally cache the virtual ladder."""
    arrays = tuple(np.asarray(x) for x in arrays)
    if any(x.dtype != np.float64 for x in arrays):
        raise TypeError('Native DF-CCSD requires real float64 tensors')
    mol = gto.Mole()
    mol.verbose, mol.stdout = options['verbose'], options['stdout']
    mol.max_memory = options['max_memory']
    nmo = arrays[0].shape[0]
    mf = SimpleNamespace(mol=mol, verbose=mol.verbose, stdout=mol.stdout,
                         max_memory=options['max_memory'],
                         mo_coeff=np.eye(nmo),
                         mo_occ=np.r_[np.full(nocc, 2.), np.zeros(nmo-nocc)],
                         e_tot=0., chkfile=None, with_df=SimpleNamespace())
    cc = dfccsd.RCCSD(mf)
    for name, value in options.items():
        if name not in ('threads', 'stdout'):
            setattr(cc, name, value)
    cc.incore_complete = True
    cc.diis_file = None
    cc.async_io = False
    cc.dcsd = False
    from pyscfad import config
    cc.cpp_kernels = options.get('cpp_kernels', config.dfccsd_cpp)
    cc.vvvv_direct = options.get('vvvv_direct', False)
    cc._nocc, cc._nmo = nocc, nmo
    eris = _ResidentERIs(mol)
    for name, value in zip(ERI_NAMES, arrays):
        setattr(eris, name, value)
    eris.nocc, eris.naux = nocc, eris.Lvv.shape[0]
    eris.vvL = np.ascontiguousarray(eris.Lvv.T)
    eris.e_hf = 0.
    _maybe_cache_vvvv(cc, eris)
    return cc, eris


def pack_cotangent(a, b):
    """Adjoint of symmetric composite-index unpacking, including diagonals."""
    o, v = a.shape
    mat = b.transpose(0, 2, 1, 3).reshape(o*v, o*v)
    rows, cols = np.tril_indices(o*v)
    values = mat[rows, cols] + mat[cols, rows]
    values[rows == cols] *= .5
    return np.concatenate((a.ravel(), values))


def metric(a, b):
    """Restricted spin metric from conventional lambda to packed covectors."""
    return pack_cotangent(2*a, 2*b-b.swapaxes(2, 3))


def inverse_metric(vector, nocc, nvir):
    a, b = ccsd.vector_to_amplitudes(vector, nocc+nvir, nocc)
    mat = b.transpose(0, 2, 1, 3).reshape(nocc*nvir, -1).copy()
    mat *= .5
    rows = np.diag_indices(nocc*nvir)
    mat[rows] *= 2
    b = mat.reshape(nocc, nvir, nocc, nvir).transpose(0, 2, 1, 3)
    return .5*a, (2*b+b.swapaxes(2, 3))/3


def response_action(cc, eris, t1, t2):
    """Construct J_update.T using the conventional, affine lambda update.

    For converged amplitudes the spin metric M and denominator D give
    J_update.T x = M D [K(D^-1 M^-1 x) - K(0)]. Subtracting K(0)
    removes the ordinary energy source; the caller supplies the full source
    from projected CC energy and triples. No Jacobian matrix is assembled.
    """
    o, v = t1.shape
    d1 = eris.mo_energy[:o, None] - eris.mo_energy[None, o:] - cc.level_shift
    d2 = d1[:, None, :, None] + d1[None, :, None, :]
    imds = response.make_intermediates(cc, t1, t2, eris)
    zero = response.update_lambda(cc, t1, t2, np.zeros_like(t1),
                                  np.zeros_like(t2), eris, imds)

    def apply(vector):
        a, b = inverse_metric(vector, o, v)
        a, b = response.update_lambda(cc, t1, t2, a/d1, b/d2, eris, imds)
        return metric(d1*(a-zero[0]), d2*(b-zero[1]))
    return apply


def energy_amplitude_cotangent(eris, t1, bar_e):
    o = t1.shape[0]
    g = 2*eris.ovov.transpose(0, 2, 1, 3) - eris.ovov.transpose(0, 2, 3, 1)
    a = (2*eris.fock[:o, o:] + lib.einsum('ijab,jb->ia', g, t1)
         + lib.einsum('ijab,ia->jb', g, t1))
    return pack_cotangent(bar_e*a, bar_e*g)


def solve_response(cc, eris, t1, t2, bar_e, bar_t1, bar_t2):
    action = response_action(cc, eris, t1, t2)
    source = (energy_amplitude_cotangent(eris, t1, bar_e)
              + pack_cotangent(bar_t1, bar_t2))
    vector = ccsd.amplitudes_to_vector(t1, t2)
    diis = lib.diis.DIIS(cc, incore=True) if cc.diis else None
    if diis is not None:
        diis.space = cc.diis_space
    converged = False
    for step in range(cc.max_cycle):
        new = action(vector) + source
        norm = np.linalg.norm(new-vector)
        vector = new
        if (diis is not None and step >= cc.diis_start_cycle
                and cc.diis_start_energy_diff > 0):
            vector = diis.update(vector)
        logger.info(cc, 'native response cycle = %d  norm(lambda) = %.6g', step+1, norm)
        if norm < cc.conv_tol_normt:
            converged = True
            break
    if not converged:
        logger.warn(cc, 'Native DF-CCSD response did not converge')
    return converged, vector


def integral_cotangent(cc, eris, t1, t2, bar_e, vector):
    a, b = adjoint._amplitudes_to_vector_cotangent(vector, *t1.shape)
    bars = adjoint._dfccsd_update_amps_eris_cotangent(cc, eris, t1, t2, a, b)
    fock, ovov = adjoint._energy_eris_cotangent(eris, t1, t2, bar_e)
    bars['fock'] += fock
    bars['ovov'] = ovov
    return tuple(np.asarray(bars[name]) for name in BAR_NAMES)


def _call(fn, shapes, *args):
    if any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(args)):
        return jax.pure_callback(fn, shapes, *args, vmap_method='sequential')
    return jax.tree.map(jax.numpy.asarray, fn(*args))


def forward(mycc, eris, t1, t2):
    if t1 is None or t2 is None:
        from .dfccsd import _init_amps_no_side_effect
        guess1, guess2 = _init_amps_no_side_effect(mycc, eris)
        t1 = guess1 if t1 is None else t1
        t2 = guess2 if t2 is None else t2
    options = settings(mycc)
    shapes = (jax.ShapeDtypeStruct((), np.bool_),
              jax.ShapeDtypeStruct((), t1.dtype),
              jax.ShapeDtypeStruct(t1.shape, t1.dtype),
              jax.ShapeDtypeStruct(t2.shape, t2.dtype))

    def run(*arrays):
        # OpenMP's thread count is thread-local; callbacks can run on workers.
        with lib.with_omp_threads(options['threads']), threadpool_limits(options['threads']):
            cc, host = host_objects(options, arrays[:-2], t1.shape[0])
            out = ccsd.kernel(cc, host, np.asarray(arrays[-2]), np.asarray(arrays[-1]),
                              max_cycle=cc.max_cycle, tol=cc.conv_tol,
                              tolnormt=cc.conv_tol_normt, verbose=cc.verbose)
            return tuple(np.asarray(x) for x in out)
    return _call(run, shapes, *(getattr(eris, n) for n in ERI_NAMES), t1, t2)


def backward(mycc, eris, t1, t2, bar_e, bar_t1, bar_t2):
    options = settings(mycc)
    shapes = tuple(jax.ShapeDtypeStruct(getattr(eris, n).shape,
                                      getattr(eris, n).dtype) for n in BAR_NAMES)

    def run(*arrays):
        with lib.with_omp_threads(options['threads']), threadpool_limits(options['threads']):
            cc, host = host_objects(options, arrays[:-5], t1.shape[0])
            a, b, be, ba, bb = map(np.asarray, arrays[-5:])
            with resource_profile.section('ccsd_bwd.response_lambda',
                                          backend='numpy', nocc=a.shape[0], nvir=a.shape[1]):
                _, vector = solve_response(cc, host, a, b, be, ba, bb)
            with resource_profile.section('ccsd_bwd.lagrangian_gradient',
                                          backend='numpy', nocc=a.shape[0], nvir=a.shape[1]):
                return integral_cotangent(cc, host, a, b, be, vector)
    bars = _call(run, shapes, *(getattr(eris, n) for n in ERI_NAMES),
                 t1, t2, bar_e, bar_t1, bar_t2)
    template = copy.copy(eris)
    template._vvvv_pair = template.ovvv = None
    result = jax.tree.map(lambda x: None if x is None else jax.numpy.zeros_like(x), template)
    for name, value in zip(BAR_NAMES, bars):
        setattr(result, name, value)
    return result
