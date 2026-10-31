# Copyright 2021-2025 Xing Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
import numpy
import jax
from jax import custom_vjp
from jax.custom_derivatives import SymbolicZero
from pyscf import __config__
from pyscf import lib as pyscf_lib
from pyscf.lib import direct_sum, current_memory
#from pyscf.mp.mp2 import _ChemistsERIs
from pyscfad import config
from pyscfad import numpy as np
from pyscfad.ops import is_array, vmap
from pyscfad.lib import logger
from pyscfad.ao2mo import _ao2mo
from pyscfad.df import incore as df_incore
from pyscfad.df import addons as df_addons
from pyscfad.df import _cderi_vjp
from pyscfad.mp import mp2
from pyscfad.tools import resource_profile

try:
    from pyscfadlib import libmp_vjp
except ImportError:
    libmp_vjp = None

try:
    libmp = pyscf_lib.load_library('libmp')
except OSError:
    # Older supported PySCF releases have no native MP2 library.
    libmp = None

WITH_T2 = getattr(__config__, 'mp_dfmp2_with_t2', True)

def _contract(Lov, mo_energy, nocc, nvir, with_t2=True):
    def body(Lv, Lov, ea, eia):
        gi = np.einsum('la,ljb->jab', Lv, Lov)
        t2i = gi / (eia[:,:,None] + ea[None,None,:])
        ei = np.einsum('jab,jab', t2i, gi) * 2 - np.einsum('jab,jba', t2i, gi)
        return ei, t2i

    Lov = Lov.reshape((-1, nocc, nvir))
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    e, t2 = vmap(body, in_axes=(1,None,0,None),
                 signature='(l,a),(b)->(),(j,a,b)')(Lov, Lov, eia, eia)
    if not with_t2:
        t2 = None
    emp2 = e.sum().real
    return emp2, t2

@partial(custom_vjp, nondiff_argnums=(2,3,4))
def _contract_opt(Lov, mo_energy, nocc, nvir, with_t2=True):
    if with_t2:
        t2 = numpy.empty((nocc,nocc,nvir,nvir))
    else:
        t2 = None
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    emp2 = 0
    for i in range(nocc):
        buf = numpy.dot(Lov[:,i*nvir:(i+1)*nvir].T,
                        Lov).reshape((nvir,nocc,nvir))
        gi = buf.transpose(1,0,2)
        t2i = gi / direct_sum('jb+a->jba', eia, eia[i])
        emp2 += numpy.einsum('jab,jab', t2i, gi) * 2
        emp2 -= numpy.einsum('jab,jba', t2i, gi)
        if with_t2:
            t2[i] = t2i
    return emp2, t2

def _contract_opt_fwd(Lov, mo_energy, nocc, nvir, with_t2):
    emp2, t2 = _contract_opt(Lov, mo_energy, nocc, nvir, with_t2)
    return (emp2, t2), (Lov, mo_energy, t2)

def _contract_opt_bwd(nocc, nvir, with_t2,
                      res, ybar):
    Lov, mo_energy, t2 = res
    emp2_bar = ybar[0]
    if with_t2:
        t2 = numpy.asarray(t2)
        t2_bar = numpy.asarray(ybar[1])

    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    Lov_bar = numpy.zeros_like(Lov)
    mo_energy_bar = numpy.zeros_like(mo_energy)
    for i in range(nocc):
        ejab = direct_sum('jb+a->jba', eia, eia[i])
        if with_t2:
            t2i = t2[i]
            gi = t2i * ejab
        else:
            gi = numpy.dot(Lov[:,i*nvir:(i+1)*nvir].T,
                           Lov).reshape((nvir,nocc,nvir))
            gi = gi.transpose(1,0,2)
            t2i = gi / ejab

        buf = emp2_bar * t2i
        gi_bar = 2 * buf - buf.transpose(0,2,1)
        buf = emp2_bar * gi
        t2i_bar = 2 * buf - buf.transpose(0,2,1)
        if with_t2:
            t2i_bar += t2_bar[i]
        gi_bar += t2i_bar / ejab

        buf = gi_bar.transpose(1,0,2).reshape(nvir,-1)
        Lov_bar += numpy.dot(Lov[:,i*nvir:(i+1)*nvir], buf)
        Lov_bar[:,i*nvir:(i+1)*nvir] += numpy.dot(Lov, buf.T)
        gi_bar = None

        buf = -gi * t2i_bar / (ejab * ejab)
        ejab = gi = t2i_bar = None
        mo_energy_bar[i] += numpy.sum(buf)
        mo_energy_bar[:nocc] += numpy.sum(buf.reshape(nocc,-1), axis=-1)
        mo_energy_bar[nocc:] -= numpy.sum(buf.reshape(-1,nvir), axis=0)
        mo_energy_bar[nocc:] -= numpy.sum(buf.transpose(1,0,2).reshape(nvir,-1), axis=-1)
        buf = None
    return (Lov_bar, mo_energy_bar)

_contract_opt.defvjp(_contract_opt_fwd, _contract_opt_bwd)

_PROFILE_REVERSE_MODE = ContextVar('dfmp2_profile_reverse_mode', default=False)


@contextmanager
def profile_reverse_mode():
    """Opt in to separate contraction forward/reverse execution profiling.

    Use around eager reverse-mode ``jax.grad``, ``jax.value_and_grad``, or
    ``jax.vjp`` evaluation.  The resource-profile environment flag must also
    be enabled.  This explicit scope is needed because JAX's ``linearize``
    shares the eager reverse-mode tracer, but cannot use a custom-VJP boundary.
    General resource profiling retains the original scan's full AD behavior.
    """
    token = _PROFILE_REVERSE_MODE.set(True)
    try:
        yield
    finally:
        _PROFILE_REVERSE_MODE.reset(token)


def _profile_eager_arrays(tree):
    """Get completed primal arrays, or reject staged/nested traces.

    Eager JAX linearization uses LinearizeTracer with concrete primals.
    Other tracer kinds (including older JAX AD tracers) conservatively keep the
    original calculation path: a host tracing interval is not execution time.
    Only one reverse-AD layer is accepted, so nested AD remains untouched.
    """
    arrays = []
    for value in jax.tree_util.tree_leaves(tree):
        if isinstance(value, jax.core.Tracer):
            if type(value).__name__ == 'LinearizeTracer':
                value = value.primal
            if isinstance(value, jax.core.Tracer):
                return None
        if hasattr(value, 'block_until_ready'):
            arrays.append(value)
    return arrays


def _profile_call(phase, fn, *args):
    if not resource_profile.enabled():
        return fn(*args)
    arrays = _profile_eager_arrays(args)
    if arrays is None:
        return fn(*args)
    # Exclude pending input work; include completion of every result leaf.
    jax.block_until_ready(arrays)
    before = resource_profile.start()
    try:
        result = fn(*args)
        output_arrays = _profile_eager_arrays(result)
        if output_arrays is not None:
            jax.block_until_ready(output_arrays)
        else:
            # A closure can contain a staged value even when args are concrete.
            # Discard this tracing interval and release its optional sampler.
            if before is not None and before.sampler is not None:
                before.sampler.stop(resource_profile.snapshot().rss_mib)
            before = None
        return result
    finally:
        resource_profile.finish(phase, before)


def _contract_scan(Lov, mo_energy, nocc, nvir, with_t2=True):
    if not resource_profile.enabled():
        return _contract_scan_impl(Lov, mo_energy, nocc, nvir, with_t2)
    if (_PROFILE_REVERSE_MODE.get()
            and _profile_eager_arrays((Lov, mo_energy)) is not None):
        return _contract_scan_profiled(Lov, mo_energy, nocc, nvir, with_t2)
    return _profile_call(
        'dfmp2.energy_forward', _contract_scan_impl,
        Lov, mo_energy, nocc, nvir, with_t2,
    )


@partial(custom_vjp, nondiff_argnums=(2, 3, 4))
def _contract_scan_profiled(Lov, mo_energy, nocc, nvir, with_t2):
    return _profile_call(
        'dfmp2.energy_forward', _contract_scan_impl,
        Lov, mo_energy, nocc, nvir, with_t2,
    )


def _contract_scan_profiled_fwd(Lov, mo_energy, nocc, nvir, with_t2):
    # Differentiate the identical checkpointed scan.  Keeping its pullback as
    # the residual avoids an extra contraction and retains its rematerialization
    # strategy; the boundary only separates completed forward and reverse work.
    fn = partial(_contract_scan_impl, nocc=nocc, nvir=nvir, with_t2=with_t2)
    return _profile_call('dfmp2.energy_forward', jax.vjp, fn, Lov, mo_energy)


def _contract_scan_profiled_bwd(nocc, nvir, with_t2, pullback, ybar):
    return _profile_call('dfmp2.energy_cotangents', pullback, ybar)


_contract_scan_profiled.defvjp(
    _contract_scan_profiled_fwd, _contract_scan_profiled_bwd,
)


def _contract_scan_impl(Lov, mo_energy, nocc, nvir, with_t2=True):
    if with_t2:
        t2 = numpy.empty((nocc,nocc,nvir,nvir))
    else:
        t2 = None
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    @jax.checkpoint
    def _fn(emp2, x):
        La, ea = x
        gi = np.dot(La.T, Lov).reshape((nvir,nocc,nvir))
        gi = gi.transpose(1,0,2)
        t2i = gi / (eia[:,:,None] + ea[None,None,:])
        emp2 += np.einsum('jab,jab', t2i, gi) * 2
        emp2 -= np.einsum('jab,jba', t2i, gi)
        if not with_t2:
            t2i = None
        return emp2, t2i

    emp2, t2 = jax.lax.scan(_fn, 0., (Lov.reshape((-1,nocc,nvir)).transpose(1,0,2), eia))
    if not with_t2:
        t2 = None
    return emp2, t2


def _to_iaL(Lov, nocc, nvir):
    # (nocc, nvir, naux), the layout of MP2_contract_d
    return pyscf_lib.transpose(numpy.asarray(Lov)).reshape(nocc, nvir, -1)

def _ed_ex(iaL, mo_energy, nocc, nvir, t2=None):
    '''(ed, ex) of MP2_contract_d, and the amplitudes written to t2,
    (nocc, nocc, nvir, nvir) in C order, if given.'''
    naux = iaL.shape[-1]
    e_occ = mo_energy[:nocc]
    e_vir = mo_energy[nocc:]
    moeoo = numpy.asarray(e_occ[:,None] + e_occ, order='C')
    moevv = numpy.asarray(e_vir[:,None] + e_vir, order='C')
    ed = numpy.zeros(1)
    ex = numpy.zeros(1)
    libmp.MP2_contract_d(
        ed.ctypes.data_as(ctypes.c_void_p),
        ex.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(1),
        iaL.ctypes.data_as(ctypes.c_void_p),
        iaL.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(0), ctypes.c_int(0),
        ctypes.c_int(nocc), ctypes.c_int(nocc),
        ctypes.c_int(nocc), ctypes.c_int(nvir), ctypes.c_int(naux),
        moeoo.ctypes.data_as(ctypes.c_void_p),
        moevv.ctypes.data_as(ctypes.c_void_p),
        _c_ptr(t2), ctypes.c_int(0),
    )
    return ed[0], ex[0]

def _c_ptr(a):
    if a is None:
        return pyscf_lib.c_null_ptr()
    return a.ctypes.data_as(ctypes.c_void_p)

def _contract_d_vjp(iaL, mo_energy, nocc, nvir, ed_bar, ex_bar,
                    iaL_bar=None, e_bar=None, with_energy=False,
                    t2=None, t2_bar=None):
    '''VJP of (ed, ex) of MP2_contract_d for the cotangents (ed_bar, ex_bar),
    and of the amplitudes for t2_bar if given, added to iaL_bar and e_bar if
    given. If with_energy, also returns ed and ex, and writes the amplitudes
    to t2 if given, computed in the same pass (MP2_contract_d_energy_vjp).
    '''
    naux = iaL.shape[-1]
    if iaL_bar is None:
        iaL_bar = numpy.zeros_like(iaL)
    if e_bar is None:
        e_bar = numpy.zeros_like(mo_energy)
    e_vir_bar = e_bar[nocc:]
    args = (
        iaL_bar.ctypes.data_as(ctypes.c_void_p),
        e_bar.ctypes.data_as(ctypes.c_void_p),
        e_vir_bar.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(ed_bar),
        ctypes.c_double(ex_bar),
        iaL.ctypes.data_as(ctypes.c_void_p),
        mo_energy.ctypes.data_as(ctypes.c_void_p),
        mo_energy[nocc:].ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nocc), ctypes.c_int(nvir), ctypes.c_int(naux),
    )
    if with_energy or t2 is not None or t2_bar is not None:
        e = numpy.zeros(2)
        if t2_bar is not None:
            t2_bar = numpy.asarray(t2_bar, dtype=numpy.float64, order='C')
            assert t2_bar.shape == (nocc, nocc, nvir, nvir)
        libmp_vjp.MP2_contract_d_energy_vjp(_c_ptr(e), _c_ptr(t2), _c_ptr(t2_bar),
                                            *args)
        if with_energy:
            return e[0], e[1], iaL_bar, e_bar
    else:
        libmp_vjp.MP2_contract_d_vjp(*args)
    return iaL_bar, e_bar


_NATIVE_REVERSE_MODE = ContextVar('dfmp2_native_reverse_mode', default=False)


@contextmanager
def _native_reverse_mode():
    """Enable native fusion in an audited eager first-order reverse call.

    Only use around grad, value_and_grad, or vjp, never around linearize.
    Linearize uses the same tracer as reverse AD and cannot be distinguished
    reliably. Outside this private opt-in, all traced calls keep generic AD.
    """
    token = _NATIVE_REVERSE_MODE.set(True)
    try:
        yield
    finally:
        _NATIVE_REVERSE_MODE.reset(token)


def _native_lov_eligible(Lov, mo_energy, nocc, nvir):
    if (libmp is None or not hasattr(libmp, 'MP2_contract_d')
            or libmp_vjp is None
            or not hasattr(libmp_vjp, 'MP2_contract_d_vjp')
            or not hasattr(libmp_vjp, 'MP2_contract_d_energy_vjp')
            or nocc <= 0 or nvir <= 0):
        return False
    for value in (Lov, mo_energy):
        if getattr(value, 'dtype', None) != numpy.dtype(numpy.float64):
            return False
        if isinstance(value, jax.core.Tracer):
            if not (_NATIVE_REVERSE_MODE.get()
                    and type(value).__name__ == 'LinearizeTracer'):
                return False
            if isinstance(value.primal, jax.core.Tracer):
                return False
    return (Lov.ndim == 2 and Lov.shape[0] > 0
            and Lov.shape[1] == nocc*nvir
            and mo_energy.shape == (nocc+nvir,))



def _native_lov_memory_fits(mp, Lov, mo_energy, nocc, nvir, with_t2):
    """Check native additions after the existing transform has produced Lov.

    Keep the transform's own memory guard. Native contraction adds a layout
    copy and OpenMP workspace; reverse mode also retains cached derivatives
    and may need writable seed corrections plus the outgoing Lov cotangent.
    Falling back leaves the original scan and its memory policy available.
    """
    if getattr(getattr(mp, 'mol', None), 'incore_anyway', False):
        return True
    max_memory = getattr(mp, 'max_memory', float('inf'))
    reverse = any(isinstance(x, jax.core.Tracer) for x in (Lov, mo_energy))
    values = tuple(x.primal if isinstance(x, jax.core.Tracer) else x
                   for x in (Lov, mo_energy))
    # Native execution must wait for these primals anyway. Account resident
    # transform outputs before inspecting RSS, including async JAX outputs.
    jax.block_until_ready(values)
    nthreads = max(1, int(pyscf_lib.num_threads()))
    nmo = nocc + nvir
    factor_copies = 4 if reverse else 1
    extra_doubles = factor_copies * Lov.size
    extra_doubles += 3*nvir*nvir*nthreads  # private v/t/g per native worker
    extra_doubles += nocc*nocc + nvir*nvir  # primal energy denominators
    extra_doubles += nmo*(nthreads + 3)  # energy buffers/reductions
    if with_t2:
        # Output plus reverse seed/layout allowance; no T2-sized cache is kept.
        extra_doubles += (3 if reverse else 1)*(nocc*nvir)**2
    extra_bytes = 8*extra_doubles
    extra_bytes += 4*nocc*(nocc + 1) + 64*nocc  # pair indices/lock allowance
    return current_memory()[0] + extra_bytes/1e6 < max_memory


@partial(custom_vjp, nondiff_argnums=(2, 3, 4))
def _contract_d_lov(Lov, mo_energy, nocc, nvir, with_t2=False):
    """Native (ed, ex[, t2]) after the existing AO-to-MO transform.

    The forward reverse-AD rule also computes the total-energy seed (2, 1)
    in the same pair pass. The backward rule scales it and adds any remaining
    energy/T2 seed. Retain only the iaL layout, not both iaL and Lov.
    """
    iaL = _to_iaL(Lov, nocc, nvir)
    mo_energy = numpy.asarray(mo_energy, order='C')
    if iaL.dtype != numpy.float64 or mo_energy.dtype != numpy.float64:
        raise TypeError('native MP2 requires float64 factors and energies')
    t2 = numpy.empty((nocc, nocc, nvir, nvir)) if with_t2 else None
    ed, ex = _profile_call(
        'dfmp2.energy_forward', _ed_ex, iaL, mo_energy, nocc, nvir, t2)
    return (ed, ex, t2) if with_t2 else (ed, ex)


def _contract_d_lov_fwd(Lov, mo_energy, nocc, nvir, with_t2):
    iaL = _to_iaL(Lov.value, nocc, nvir)
    mo_energy = numpy.asarray(mo_energy.value, order='C')
    if iaL.dtype != numpy.float64 or mo_energy.dtype != numpy.float64:
        raise TypeError('native MP2 requires float64 factors and energies')
    t2 = numpy.empty((nocc, nocc, nvir, nvir)) if with_t2 else None
    compute = partial(_contract_d_vjp, with_energy=True, t2=t2)
    ed, ex, iaL_bar, e_bar = _profile_call(
        'dfmp2.energy_vjp_fused', compute,
        iaL, mo_energy, nocc, nvir, 2., 1.)
    out = (ed, ex, t2) if with_t2 else (ed, ex)
    return out, (iaL, mo_energy, iaL_bar, e_bar)


def _contract_d_lov_bwd(nocc, nvir, with_t2, res, ybar):
    iaL, mo_energy, total_iaL_bar, total_e_bar = res
    ed_bar, ex_bar = (0. if isinstance(y, SymbolicZero) else float(y)
                      for y in ybar[:2])
    t2_bar = None
    if with_t2 and not isinstance(ybar[2], SymbolicZero):
        t2_bar = ybar[2]
    if ed_bar == 0 and ex_bar == 0 and t2_bar is None:
        return None, None

    # (ed_bar, ex_bar) = ex_bar*(2, 1) + (ed_bar-2*ex_bar, 0).
    correction = ed_bar - 2*ex_bar
    if correction != 0 or t2_bar is not None:
        # Native kernels add to their output: never mutate residuals, since
        # the same pullback may be reused with different seeds.
        iaL_bar = total_iaL_bar * ex_bar
        e_bar = total_e_bar * ex_bar
        compute = partial(_contract_d_vjp, t2_bar=t2_bar)
        iaL_bar, e_bar = _profile_call(
            'dfmp2.energy_cotangents', compute, iaL, mo_energy, nocc, nvir,
            correction, 0., iaL_bar, e_bar)
        scale = 1.
    else:
        iaL_bar, e_bar = total_iaL_bar, total_e_bar
        scale = ex_bar
    naux = iaL_bar.shape[-1]
    Lov_bar = pyscf_lib.transpose(iaL_bar.reshape(nocc*nvir, naux))
    if scale != 1:
        Lov_bar *= scale
        e_bar = e_bar * scale
    return Lov_bar, e_bar


_contract_d_lov.defvjp(
    _contract_d_lov_fwd, _contract_d_lov_bwd, symbolic_zeros=True)


@partial(custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def _outcore_nr_e2(mol, auxmol, mo_coeff, cderi_source, max_memory,
                   ijslice, aosym):
    del mol, auxmol
    if numpy.dtype(mo_coeff.dtype) != numpy.dtype(numpy.float64):
        # Preserve the generic backend (including its non-float64 support).
        # The streamed PySCF C transform below is intentionally float64-only.
        with df_addons.load(cderi_source, 'j3c') as eri1:
            if not is_array(eri1):
                eri1 = numpy.asarray(eri1)
            return _ao2mo.nr_e2_gen(
                eri1, mo_coeff, ijslice, aosym=aosym
            )
    # Keep only the transformed occupied-virtual factor in memory.  Loading
    # the packed AO CDERI here defeats the purpose of attaching an out-of-core
    # source and becomes the forward memory limit before the MP2 contraction
    # itself.  This helper reads auxiliary slabs and writes them directly into
    # the (naux, nocc*nvir) result.
    return _cderi_vjp._stream_nr_e2_from_cderi_source(
        cderi_source, mo_coeff, ijslice, max_memory, aosym=aosym
    )


def _outcore_nr_e2_fwd(mol, auxmol, mo_coeff, cderi_source, max_memory,
                       ijslice, aosym):
    out = _outcore_nr_e2(mol, auxmol, mo_coeff, cderi_source, max_memory,
                         ijslice, aosym)
    return out, (mol, auxmol, mo_coeff)


def _outcore_nr_e2_bwd(cderi_source, max_memory, ijslice, aosym, res, ybar):
    return _profile_call(
        'dfmp2.input_pullback', _outcore_nr_e2_bwd_impl,
        cderi_source, max_memory, ijslice, aosym, res, ybar,
    )


def _outcore_nr_e2_bwd_impl(cderi_source, max_memory, ijslice, aosym, res, ybar):
    mol, auxmol, mo_coeff = res
    effective_memory = max(max_memory, 4096)
    use_direct_vjp = (
        numpy.dtype(mo_coeff.dtype) == numpy.dtype(numpy.float64)
        and numpy.dtype(ybar.dtype) == numpy.dtype(numpy.float64)
    )
    if use_direct_vjp:
        try:
            # The canonical transform uses the complete AO basis, so its two
            # reverse contributions can both be evaluated without constructing
            # the packed (naux, nao_pair) CDERI cotangent.  Stream auxiliary rows
            # for the MO-coefficient cotangent, then contract the same Lov
            # cotangent directly with the three-centre integral derivatives.
            #
            # Besides avoiding cderi_bar, this also avoids loading the complete
            # primal CDERI merely to differentiate the AO-to-MO transform.  The
            # fallback below is retained for unsupported symmetry/dtype cases.
            mo_coeff_bar = (
                _cderi_vjp.nr_e2_mo_coeff_vjp_from_cderi_source(
                    cderi_source,
                    mo_coeff,
                    ybar,
                    ijslice,
                    aosym=aosym,
                    max_memory=effective_memory,
                )
            )
            mol_bar, auxmol_bar = (
                _cderi_vjp.cholesky_eri_vjp_from_mo_coeff_ybar(
                    mol,
                    auxmol,
                    cderi_source,
                    mo_coeff,
                    ybar,
                    ijslice,
                    effective_memory,
                    int3c=mol._add_suffix('int3c2e'),
                    int2c=mol._add_suffix('int2c2e'),
                    aosym='s2ij',
                )
            )
            return mol_bar, auxmol_bar, mo_coeff_bar
        except NotImplementedError:
            pass

    try:
        if use_direct_vjp:
            cderi_bar, mo_coeff_bar = (
                _cderi_vjp.nr_e2_vjp_from_cderi_source(
                    cderi_source, mo_coeff, ybar, ijslice, aosym=aosym
                )
            )
        else:
            # The optimized global nr_e2 switch is float64-only.  Keep the
            # legacy generic dtype support independent of that global setting
            # in both the primal and its fallback pullback.
            with df_addons.load(cderi_source, 'j3c') as eri1:
                if not is_array(eri1):
                    eri1 = numpy.asarray(eri1)
                cderi = np.asarray(eri1)

            def generic_transform(cderi_, mo_coeff_):
                return _ao2mo.nr_e2_gen(
                    cderi_, mo_coeff_, ijslice, aosym=aosym
                )

            _, generic_pullback = jax.vjp(
                generic_transform, cderi, mo_coeff
            )
            cderi_bar, mo_coeff_bar = generic_pullback(ybar)
        mol_bar, auxmol_bar = _cderi_vjp.cholesky_eri_vjp_from_cderi_source(
            mol, auxmol, cderi_source, cderi_bar, effective_memory,
            int3c=mol._add_suffix('int3c2e'),
            int2c=mol._add_suffix('int2c2e'),
            aosym='s2ij',
        )
        return mol_bar, auxmol_bar, mo_coeff_bar
    except NotImplementedError:
        pass

    def fn(mol_, auxmol_, mo_coeff_):
        cderi = df_incore.cholesky_eri(
            mol_,
            auxmol=auxmol_,
            int3c=mol_._add_suffix('int3c2e'),
            int2c=mol_._add_suffix('int2c2e'),
            max_memory=max(max_memory, 4096),
            verbose=0,
        )
        return _ao2mo.nr_e2(cderi, mo_coeff_, ijslice, aosym=aosym)

    _, pullback = jax.vjp(fn, mol, auxmol, mo_coeff)
    return pullback(ybar)


_outcore_nr_e2.defvjp(_outcore_nr_e2_fwd, _outcore_nr_e2_bwd)


def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
           verbose=None):
    if mo_energy is not None or mo_coeff is not None:
        assert (mp.frozen == 0 or mp.frozen is None)

    if eris is None:
        eris = mp.ao2mo(mo_coeff)
    if mo_energy is None:
        mo_energy = eris.mo_energy
    if mo_coeff is None:
        mo_coeff = eris.mo_coeff

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    #naux = mp.with_df.get_naoaux()

    Lov = _profile_call(
        'dfmp2.ao2mo', mp.loop_ao2mo, mo_coeff, nocc, with_t2,
    )

    if (config.moleintor_opt
            and _native_lov_eligible(Lov, mo_energy, nocc, nvir)
            and _native_lov_memory_fits(mp, Lov, mo_energy, nocc, nvir, with_t2)):
        out = _contract_d_lov(Lov, mo_energy, nocc, nvir, with_t2)
        return 2*out[0] + out[1], out[2] if with_t2 else None

    if config.moleintor_opt:
        #emp2, t2 = _contract_opt(Lov, mo_energy, nocc, nvir, with_t2)
        emp2, t2 = _contract_scan(Lov, mo_energy, nocc, nvir, with_t2)
    else:
        emp2, t2 = _contract(Lov, mo_energy, nocc, nvir, with_t2)
    return emp2, t2

class MP2(mp2.MP2):
    _dynamic_attr = _keys = {'with_df'}

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        super().__init__(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            raise KeyError('The mean-field object has no density fitting.')

    def ao2mo(self, mo_coeff=None):
        eris = mp2._ChemistsERIs()
        eris._common_init_(self, mo_coeff)
        return eris

    def loop_ao2mo(self, mo_coeff, nocc, with_t2=WITH_T2):
        # NOTE return the whole 3c integral for now
        nao, nmo = mo_coeff.shape
        nvir = nmo - nocc
        ijslice = (0, nocc, nocc, nmo)

        with_df = self.with_df
        naux = with_df.get_naoaux()
        mem_incore = (naux*nocc*nvir + nocc*nvir*nvir*2) * 8 / 1e6
        if with_t2:
            mem_incore += (nocc*nvir)**2 * 8 / 1e6
        mem_now = current_memory()[0]
        if (mem_incore + mem_now < self.max_memory) or self.mol.incore_anyway:
            get_cderi = getattr(with_df, '_get_cderi_source', None)
            cderi = get_cderi() if get_cderi is not None else with_df._cderi
            has_outcore_cderi = (
                hasattr(with_df, '_has_outcore_cderi_placeholder')
                and with_df._has_outcore_cderi_placeholder()
            )
            if has_outcore_cderi:
                if with_df.auxmol is None:
                    with_df.auxmol = df_addons.make_auxmol(
                        with_df.mol, with_df.auxbasis
                    )
                return _outcore_nr_e2(
                    with_df.mol, with_df.auxmol, mo_coeff, cderi,
                    with_df.max_memory, ijslice, 's2'
                )
            with df_addons.load(cderi, 'j3c') as eri1:
                if not is_array(eri1):
                    eri1 = numpy.asarray(eri1)
                Lov = _ao2mo.nr_e2(eri1, mo_coeff, ijslice, aosym='s2')
            return Lov
        else:
            raise RuntimeError(f'{mem_incore+mem_now} MB of memory is needed.')

    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        if self.verbose >= logger.WARN:
            self.check_sanity()

        self.dump_flags()

        self.e_hf = self.get_e_hf(mo_coeff=mo_coeff)

        if eris is None:
            eris = self.ao2mo(mo_coeff)

        if self._scf.converged:
            self.e_corr, self.t2 = self.init_amps(mo_energy, mo_coeff, eris, with_t2)
        else:
            raise NotImplementedError

        # TODO SCS-MP2
        self.e_corr_ss = 0
        self.e_corr_os = 0
        self._finalize()
        return self.e_corr, self.t2

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

del WITH_T2
