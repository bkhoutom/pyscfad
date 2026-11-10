# Copyright 2026 The PySCFAD Authors
# Licensed under the Apache License, Version 2.0.
"""Screened direct DF-J forward, density response and bilinear geometry.

For t(D)_P = sum_ij (ij|P) D_ij and rho(D) = M^-1 t(D), differentiate
    <H,J(D)> = t(H).T M^-1 t(D).
Contract both densities in the native OpenMP shell driver. Only auxiliary
vectors leave that driver; neither an AO-pair panel nor a derivative J matrix
is formed. The forward J metric factorization is reused. Response masks use
fixed Schwarz bounds, so signed/tiny Krylov seeds do not change selection;
geometry uses derivative norms rather than reusing the value-only bounds.
"""
import ctypes
import time

import numpy as np
import scipy.linalg
from pyscf import gto, lib
from pyscf.df import df_jk
from pyscf.scf import _vhf
from pyscfad.lib._threading import fixed_threads


@fixed_threads()
def get_j(obj, dm, hermi=1, tol=1e-13):
    """Forward J with PySCF's Schwarz and density screens at the current tol.

    PySCF stores the cutoff on the cached optimizer only at its creation.
    Keep later calls consistent with their requested integral tolerance.
    """
    if obj._vjopt is None:
        value = df_jk.get_j(obj, dm, hermi=hermi, direct_scf_tol=tol)
        obj._vjopt._sgx_q_cond_tol = tol
        return value
    _ensure_value_bounds(obj._vjopt, tol)
    with lib.temporary_env(obj._vjopt, direct_scf_tol=tol):
        return df_jk.get_j(obj, dm, hermi=hermi, direct_scf_tol=tol)


def _initialize(obj, dm, tol):
    if obj._vjopt is None:
        get_j(obj, np.zeros_like(dm), tol=tol)
    obj._vjopt._sgx_response_auxmol = obj.auxmol
    _ensure_value_bounds(obj._vjopt, tol)
    return obj._vjopt


def _ensure_value_bounds(opt, tol):
    built_tol = getattr(opt, '_sgx_q_cond_tol', opt.direct_scf_tol)
    if tol < built_tol:
        # A tighter request also needs bounds evaluated at that tolerance;
        # changing only the prescreen cutoff cannot recover truncated norms.
        aux_bounds = opt.q_cond[opt.mol.nbas**2:].copy()
        with lib.temporary_env(opt, direct_scf_tol=tol):
            opt.init_cvhf_direct(opt.mol, 'int2e', 'CVHFnr_int2e_q_cond')
        opt.q_cond = np.concatenate((opt.q_cond.ravel(), aux_bounds))
        opt._sgx_schwarz_opts = {}
    opt._sgx_q_cond_tol = min(tol, built_tol)


def _screen_optimizer(opt, intor, tol):
    """Density-independent Schwarz bounds for values or either derivative.

    AO derivatives use sqrt((d_i j|d_i j)); auxiliary derivatives use
    sqrt((d_P|d_P)). Value bounds cannot safely screen derivative integrals.
    Separate optimizers leave the forward density screens and cutoff intact.
    """
    cache = getattr(opt, '_sgx_schwarz_opts', None)
    if cache is None:
        cache = opt._sgx_schwarz_opts = {}
    key = (intor, float(tol))
    if key in cache:
        return cache[key]
    mol, aux = opt.mol, opt._sgx_response_auxmol
    nbas = mol.nbas
    qao = opt.q_cond[:nbas**2].reshape(nbas, nbas)
    qaux = opt.q_cond[nbas**2:]
    if intor == 'int3c2e_ip1':
        qao = np.empty((nbas, nbas))
        loc = mol.ao_loc_nr()
        with mol.with_integral_screen(tol**2):
            _vhf.libcvhf.CVHFnr_int2e_pp_q_cond(
                getattr(_vhf.libcvhf, mol._add_suffix('int2e_ip1ip2')),
                lib.c_null_ptr(), qao.ctypes, loc.ctypes,
                mol._atm.ctypes, ctypes.c_int(mol.natm), mol._bas.ctypes,
                ctypes.c_int(nbas), mol._env.ctypes)
    elif intor == 'int3c2e_ip2':
        qaux = np.empty(aux.nbas)
        with aux.with_integral_screen(tol**2):
            for shell in range(aux.nbas):
                block = aux.intor('int2c2e_ip1ip2', comp=9,
                                  shls_slice=(shell, shell+1, shell, shell+1))
                qaux[shell] = np.sqrt(np.max(np.abs(
                    np.diagonal(block[[0, 4, 8]], axis1=1, axis2=2))))
    elif intor != 'int3c2e':
        raise ValueError(f'No DF-J Schwarz bounds for {intor}')
    screened = _vhf._VHFOpt(mol, prescreen='CVHFnr3c2e_schwarz_cond',
                             direct_scf_tol=tol)
    screened.q_cond = np.concatenate((qao.ravel(), qaux))
    # The generic driver requires a non-null dm_cond; the Schwarz-only
    # callback never reads it. No density magnitude enters this selection.
    screened.dm_cond = np.ones(1)
    pair_bounds = qao.ravel() if intor == 'int3c2e_ip1' else qao[np.tril_indices(nbas)]
    ordered = np.sort(pair_bounds)
    cutoffs = np.divide(tol, qaux, out=np.full_like(qaux, np.inf), where=qaux > 0)
    retained = int(np.sum(len(ordered)-np.searchsorted(ordered, cutoffs, side='right')))
    screened._sgx_screening_counts = dict(
        candidate_shell_triples=int(len(ordered)*aux.nbas),
        retained_shell_triples=retained,
        skipped_shell_triples=int(len(ordered)*aux.nbas)-retained)
    cache[key] = screened
    return screened


def _solve_metric(opt, rhs):
    if opt.j2c_type == 'cd':
        return scipy.linalg.cho_solve(opt.j2c, rhs.T, check_finite=False).T
    if not hasattr(opt, '_sgx_metric_lu'):
        opt._sgx_metric_lu = scipy.linalg.lu_factor(opt.j2c, check_finite=False)
    return scipy.linalg.lu_solve(opt._sgx_metric_lu, rhs.T, check_finite=False).T


@fixed_threads()
def density_response(obj, dm, tol=1e-13):
    """Linear J response with the same fixed Schwarz mask in both passes."""
    opt = _initialize(obj, dm, tol)
    shape = np.shape(dm)
    dms = np.asarray(dm).reshape(-1, obj.mol.nao, obj.mol.nao)
    nbas, naux = obj.mol.nbas, obj.auxmol.nbas
    slices = (0, nbas, 0, nbas, nbas, nbas+naux, nbas+naux, nbas+naux+1)
    rhs = np.stack(_contract(opt, dms, 'int3c2e', 's2ij', slices, 1, tol=tol))[:, 0]
    rho = _solve_metric(opt, rhs)[:, None, :]
    mol = opt.fakemol
    screen = _screen_optimizer(opt, 'int3c2e', tol)
    values = _vhf.nr_direct_drv(mol._add_suffix('int3c2e'), 's2ij',
        ['lk->s2ij']*len(dms), rho, 1, mol._atm, mol._bas, mol._env,
        cvhfopt=None if screen is None else screen._this,
        cintopt=opt._sgx_response_cintopts[mol._add_suffix('int3c2e')],
        shls_slice=slices, optimize_sr=False)
    return np.stack([lib.hermi_triu(v[0], 1, inplace=True) for v in values]).reshape(shape)


def _contract(opt, dms, intor, aosym, slices, comp, *, tol=None):
    mol = opt.fakemol
    name = mol._add_suffix(intor)
    cache = getattr(opt, '_sgx_response_cintopts', None)
    if cache is None:
        cache = opt._sgx_response_cintopts = {}
    if name not in cache:
        cache[name] = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, name)
    screen = _screen_optimizer(opt, intor, opt.direct_scf_tol if tol is None else tol)
    values = _vhf.nr_direct_drv(
        name, aosym, ['ji->s1kl'] * len(dms), dms, comp,
        mol._atm, mol._bas, mol._env, cintopt=cache[name],
        cvhfopt=None if screen is None else screen._this,
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
    tol = obj._pjs_data._itol if obj._pjs_data is not None else 1e-13
    opt = _initialize(obj, contributions[0][0], tol)
    mol, aux = obj.mol, obj.auxmol
    nbas, nauxbas = mol.nbas, aux.nbas
    tail = (nbas + nauxbas, nbas + nauxbas + 1)
    dms = [x for pair in contributions for x in pair]
    partner = np.arange(len(dms)) ^ 1
    started = time.perf_counter()
    rhs = np.stack(_contract(opt, dms, 'int3c2e', 's2ij',
        (0, nbas, 0, nbas, nbas, nbas + nauxbas, *tail), 1, tol=tol))[:, 0]
    rhs_seconds = time.perf_counter()-started
    tick = time.perf_counter()
    rho = _solve_metric(opt, rhs)
    metric_solve_seconds = time.perf_counter()-tick
    result = np.zeros((mol.natm, 3))
    tick = time.perf_counter()
    for atom, (s0, s1, p, q) in enumerate(mol.aoslice_by_atom()):
        if s0 == s1:
            continue
        v = _contract(opt, [d[:, p:q] for d in dms], 'int3c2e_ip1', 's1',
            (s0, s1, 0, nbas, nbas, nbas + nauxbas, *tail), 3, tol=tol)
        result[atom] -= 2 * np.einsum('sxp,sp->x', v, rho[partner])
    ao_derivative_seconds = time.perf_counter()-tick
    auxiliary_derivative_seconds = metric_derivative_seconds = 0.
    for atom, (s0, s1, p, q) in enumerate(aux.aoslice_by_atom()):
        if s0 == s1:
            continue
        tick = time.perf_counter()
        v = _contract(opt, dms, 'int3c2e_ip2', 's2ij',
            (0, nbas, 0, nbas, nbas + s0, nbas + s1, *tail), 3, tol=tol)
        result[atom] -= np.einsum('sxp,sp->x', v, rho[partner, p:q])
        auxiliary_derivative_seconds += time.perf_counter()-tick
        tick = time.perf_counter()
        for i, ie, j, je, p0, p1, q0, q1 in _metric_tiles(
                aux, s0, s1, obj._response_workspace_bytes):
            derivative = aux.intor('int2c2e_ip1', comp=3, shls_slice=(i, ie, j, je))
            result[atom] += np.einsum('xpq,sp,sq->x', derivative,
                                     rho[:, p0:p1], rho[partner, q0:q1])
            del derivative
        metric_derivative_seconds += time.perf_counter()-tick
    obj._dfj_geometry_stats = dict(seed_pairs=len(contributions),
        ao_shells=nbas, auxiliary_shells=nauxbas,
        integral_tolerance=tol,
        screening={name: dict(screen._sgx_screening_counts)
                   for name in ('int3c2e', 'int3c2e_ip1', 'int3c2e_ip2')
                   if (screen := _screen_optimizer(opt, name, tol)) is not None},
        rhs_seconds=rhs_seconds, metric_solve_seconds=metric_solve_seconds,
        ao_derivative_seconds=ao_derivative_seconds,
        auxiliary_derivative_seconds=auxiliary_derivative_seconds,
        metric_derivative_seconds=metric_derivative_seconds,
        total_seconds=time.perf_counter()-started)
    return result
