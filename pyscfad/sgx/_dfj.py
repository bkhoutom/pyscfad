# Copyright 2026 The PySCFAD Authors
# Licensed under the Apache License, Version 2.0.
"""Integral-direct bilinear DF-J geometry response, without three-index panels.

For t(D)_P = sum_ij (ij|P) D_ij and rho(D) = M^-1 t(D), differentiate
    <H,J(D)> = t(H).T M^-1 t(D).
Contract both densities in the native OpenMP shell driver. Only auxiliary
vectors leave that driver; neither an AO-pair panel nor a derivative J matrix
is formed. The forward J metric factorization is reused.
"""
import numpy as np
import scipy.linalg
from pyscf import gto, lib
from pyscf.df import df_jk
from pyscf.scf import _vhf
from pyscfad.lib._threading import fixed_threads


@fixed_threads()
def density_response(obj, dm, tol=1e-13):
    """Apply the Coulomb density pullback without density-based screening.

    Krylov basis vectors and the final solution can have very different
    magnitudes. Absolute density cutoffs make their J actions inconsistent,
    even when SGX exchange uses static FP64 screening. Retain the existing
    integral metadata and metric; disable only the two J density prescreens
    for this pullback and restore the forward setting afterward.
    """
    if obj._vjopt is None:
        df_jk.get_j(obj, np.zeros_like(dm), hermi=1, direct_scf_tol=tol)
    with lib.temporary_env(obj._vjopt, direct_scf_tol=0.):
        return df_jk.get_j(obj, dm, hermi=1, direct_scf_tol=tol)


def _contract(opt, dms, intor, aosym, slices, comp):
    mol = opt.fakemol
    name = mol._add_suffix(intor)
    cache = getattr(opt, '_sgx_response_cintopts', None)
    if cache is None:
        cache = opt._sgx_response_cintopts = {}
    if name not in cache:
        cache[name] = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, name)
    # No density-dependent screening of derivative integrals. This matches
    # the reference DF gradient and permits arbitrary signed response seeds.
    values = _vhf.nr_direct_drv(
        name, aosym, ['ji->s1kl'] * len(dms), dms, comp,
        mol._atm, mol._bas, mol._env, cintopt=cache[name],
        shls_slice=slices, optimize_sr=False)
    return [v[..., 0] for v in values]


def _metric_tiles(auxmol, first, last, workspace):
    """Tile both metric axes, without a hidden minimum of 20 auxiliaries."""
    loc = auxmol.ao_loc_nr()
    cap = min(int(workspace), 32 * 1024**2)
    largest = int(np.max(np.diff(loc), initial=0))
    if cap < 24 * largest**2:
        raise MemoryError('DF-J response workspace cannot hold one metric derivative shell pair')
    # Atom rows are already short. Split even those if a tiny budget requires it.
    row_target = max(largest, int(np.sqrt(cap / 24)))
    i = first
    while i < last:
        ie = i + 1
        while ie < last and loc[ie + 1] - loc[i] <= row_target:
            ie += 1
        col_target = max(largest, cap // (24 * int(loc[ie] - loc[i])))
        j = 0
        while j < auxmol.nbas:
            je = j + 1
            while je < auxmol.nbas and loc[je + 1] - loc[j] <= col_target:
                je += 1
            yield i, ie, j, je, int(loc[i]), int(loc[ie]), int(loc[j]), int(loc[je])
            j = je
        i = ie


def geometry(obj, dm, bar):
    """Return d <bar,J(dm)> / dR for symmetric real AO matrices."""
    return geometry_many(obj, [(dm, bar)])


@fixed_threads()
def geometry_many(obj, contributions):
    """Contract every bilinear seed while each derivative shell tile is live."""
    contributions = [(d, b) for d, b in contributions if np.any(d) and np.any(b)]
    if not contributions:
        return np.zeros((obj.mol.natm, 3))
    if obj._vjopt is None:
        # Normally initialized by the primal J build. Also support direct
        # callers without duplicating PySCF's Coulomb setup implementation.
        df_jk.get_j(obj, np.zeros_like(contributions[0][0]), hermi=1)
    opt = obj._vjopt
    mol, aux = obj.mol, obj.auxmol
    nbas, nauxbas = mol.nbas, aux.nbas
    tail = (nbas + nauxbas, nbas + nauxbas + 1)
    dms = [x for pair in contributions for x in pair]
    partner = np.arange(len(dms)) ^ 1
    rhs = np.stack(_contract(opt, dms, 'int3c2e', 's2ij',
        (0, nbas, 0, nbas, nbas, nbas + nauxbas, *tail), 1))[:, 0]
    if opt.j2c_type == 'cd':
        rho = scipy.linalg.cho_solve(opt.j2c, rhs.T, check_finite=False).T
    else:
        # Keep PySCF's nonsingular indefinite-metric fallback, but factor it
        # only once for all response seeds at this geometry.
        if not hasattr(opt, '_sgx_metric_lu'):
            opt._sgx_metric_lu = scipy.linalg.lu_factor(opt.j2c, check_finite=False)
        rho = scipy.linalg.lu_solve(opt._sgx_metric_lu, rhs.T, check_finite=False).T
    result = np.zeros((mol.natm, 3))
    for atom, (s0, s1, p, q) in enumerate(mol.aoslice_by_atom()):
        if s0 == s1:
            continue
        v = _contract(opt, [d[:, p:q] for d in dms], 'int3c2e_ip1', 's1',
            (s0, s1, 0, nbas, nbas, nbas + nauxbas, *tail), 3)
        result[atom] -= 2 * np.einsum('sxp,sp->x', v, rho[partner])
    for atom, (s0, s1, p, q) in enumerate(aux.aoslice_by_atom()):
        if s0 == s1:
            continue
        v = _contract(opt, dms, 'int3c2e_ip2', 's2ij',
            (0, nbas, 0, nbas, nbas + s0, nbas + s1, *tail), 3)
        result[atom] -= np.einsum('sxp,sp->x', v, rho[partner, p:q])
        for i, ie, j, je, p0, p1, q0, q1 in _metric_tiles(
                aux, s0, s1, obj._response_workspace_bytes):
            derivative = aux.intor('int2c2e_ip1', comp=3, shls_slice=(i, ie, j, je))
            result[atom] += np.einsum('xpq,sp,sq->x', derivative,
                                     rho[:, p0:p1], rho[partner, q0:q1])
            del derivative
    return result
