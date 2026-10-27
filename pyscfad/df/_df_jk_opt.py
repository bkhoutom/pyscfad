# Copyright 2021-2025 Xing Zhang
# Portions copyright 2021-2026 The PySCFAD Authors
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

from collections import OrderedDict
from functools import partial
import ctypes
import tempfile
import os
import time
import h5py
import numpy
from scipy.linalg.lapack import dpstrf
import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.custom_derivatives import CustomVJPPrimal, SymbolicZero
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from pyscf import lib
from pyscf.lib import logger
from pyscf.df import df_jk as pyscf_df_jk
from pyscfadlib import libcvhf_vjp as libvhf
from pyscfad.df import incore as df_incore
from pyscfad.df import _cderi_vjp
from pyscfad._src.implicit_diff import is_implicit_diff_solve_matvec

libao2mo = lib.load_library('libao2mo')
_FAST_EXCHANGE_DM_DATA = OrderedDict()
_DF_VK_VJP_LOWRANK = getattr(libvhf, 'df_vk_vjp_lowrank', None)
_DF_VK_DM_VJP = getattr(libvhf, 'df_vk_dm_vjp', None)
if _DF_VK_DM_VJP is not None:
    _DF_VK_DM_VJP.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    _DF_VK_DM_VJP.restype = None


def _profile_enabled():
    value = os.environ.get('PYSCFAD_PROFILE_BACKWARD_PHASES')
    return value is not None and value.strip().lower() in ('1', 'true', 'yes', 'on')


def _profile_msg(msg):
    if _profile_enabled():
        print(f'[profile][df_jk_opt] {msg}', flush=True)


def _restore_s1_jax(cderi, nao):
    npair = nao * (nao + 1) // 2
    if cderi.ndim == 3 and cderi.shape[-1] == nao:
        return cderi
    if cderi.ndim == 2 and cderi.shape[-1] == nao**2:
        return cderi.reshape(-1, nao, nao)
    if cderi.ndim == 2 and cderi.shape[-1] == npair:
        rows, cols = numpy.tril_indices(nao)
        rows = jnp.asarray(rows)
        cols = jnp.asarray(cols)
        out = jnp.zeros((cderi.shape[0], nao, nao), dtype=cderi.dtype)
        out = out.at[:, rows, cols].set(cderi)
        return out.at[:, cols, rows].set(cderi)
    raise RuntimeError(f'cderi shape {cderi.shape} incompatible with nao {nao}')


def _get_jk_gen_jax(dfobj, dm, hermi=1, with_j=True, with_k=True,
                    direct_scf_tol=1e-13):
    del hermi, direct_scf_tol
    nao = dfobj.mol.nao
    dms = dm.reshape(-1, nao, nao)
    Lpq = _restore_s1_jax(dfobj._cderi, nao)

    vj = vk = jnp.zeros_like(dms)
    if with_j:
        tmp = jnp.einsum('Lpq,xpq->xL', Lpq, dms)
        vj = jnp.einsum('Lpq,xL->xpq', Lpq, tmp)
    if with_k:
        tmp = jnp.einsum('Lij,xjk->xLik', Lpq, dms)
        vk = jnp.einsum('xLik,Lkj->xij', tmp, Lpq)
    return vj.reshape(dm.shape), vk.reshape(dm.shape)


def _fast_exchange_key(dfobj):
    token = getattr(dfobj, '_fast_exchange_cache_token', None)
    if token is not None:
        return ('dfobj_token', token)
    cderi = getattr(dfobj, '_cderi', None)
    if cderi is not None:
        return ('cderi', id(cderi), tuple(cderi.shape))
    return ('dfobj', id(dfobj))


def _fast_exchange_cache_limit():
    """Maximum number of eager SCF orbital frames retained process-wide."""
    try:
        return max(int(os.environ.get(
            'PYSCFAD_SCF_FAST_EXCHANGE_CACHE_SIZE', '8'
        )), 1)
    except ValueError:
        return 8


def _fast_exchange_cache_get(dfobj):
    key = _fast_exchange_key(dfobj)
    data = _FAST_EXCHANGE_DM_DATA.get(key)
    if data is not None:
        _FAST_EXCHANGE_DM_DATA.move_to_end(key)
    return data


def set_fast_exchange_dm_data(dfobj, mo_coeff, mo_occ, s1e=None):
    """Cache converged-orbital data used by the fast DF exchange paths."""
    # This cache is an eager execution optimization.  Never retain tracers in
    # process-global state during an explicitly traced SCF replay.
    if _has_tracer(mo_coeff, mo_occ, s1e):
        return
    mo_coeff = numpy.asarray(mo_coeff)
    mo_occ = numpy.asarray(mo_occ)
    data = {
        'mo_coeff': mo_coeff,
        'mo_occ': mo_occ,
        'dm_ref': None,
        'occ_coeff': None,
        'overlap_occ_coeff': None,
    }
    if (
        s1e is not None
        and mo_coeff.ndim == 2
        and mo_occ.ndim == 1
        and mo_coeff.shape[1] == mo_occ.size
    ):
        occupied = mo_occ > 0
        if numpy.any(occupied) and numpy.any(~occupied):
            occ_coeff = numpy.asarray(mo_coeff[:, occupied], order='F')
            occupied_weights = numpy.asarray(mo_occ[occupied])
            data['dm_ref'] = numpy.asarray(
                (occ_coeff * occupied_weights) @ occ_coeff.conj().T
            )
            overlap_occ_coeff = numpy.asarray(
                numpy.asarray(s1e) @ occ_coeff, order='F'
            )
            data['occ_coeff'] = occ_coeff
            data['overlap_occ_coeff'] = overlap_occ_coeff
    key = _fast_exchange_key(dfobj)
    _FAST_EXCHANGE_DM_DATA.pop(key, None)
    _FAST_EXCHANGE_DM_DATA[key] = data
    while len(_FAST_EXCHANGE_DM_DATA) > _fast_exchange_cache_limit():
        _FAST_EXCHANGE_DM_DATA.popitem(last=False)


def _tag_dm_for_fast_exchange(dfobj, dm):
    if getattr(dm, 'mo_coeff', None) is not None:
        return dm

    tagged = _fast_exchange_cache_get(dfobj)
    if tagged is None:
        return dm
    mo_coeff = tagged['mo_coeff']
    mo_occ = tagged['mo_occ']
    dm_ref = tagged.get('dm_ref')

    dm = numpy.asarray(dm)
    if (
        dm_ref is None
        or dm.shape != dm_ref.shape
        or not numpy.allclose(dm, dm_ref, atol=1e-10, rtol=1e-9)
    ):
        return dm
    return lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)


def _density_factors(dm, hermi=1):
    """Return validated real factors ``D[k] = X[k] @ X[k].T``, or None.

    The tuple follows flattened leading density dimensions. Each factor is
    float64 and Fortran contiguous, with occupations absorbed as sqrt weights.
    Valid orbital tags take priority; otherwise use upstream PySCFAD's pivoted
    Cholesky factorization (db0acafb, The PySCFAD Authors, Apache-2.0).
    Unsupported densities and any failed reconstruction retain the dense path.
    """
    dms = numpy.asarray(dm)
    if (hermi != 1 or dms.ndim < 2 or dms.shape[-2] != dms.shape[-1]
            or dms.dtype.kind != 'f' or not numpy.all(numpy.isfinite(dms))):
        return None
    nao = dms.shape[-1]
    if nao == 0:
        return None
    dms = numpy.asarray(dms, dtype=numpy.float64).reshape(-1, nao, nao)
    if len(dms) == 0:
        return None
    # Scale with the density, including near-zero densities; a fixed absolute
    # tolerance could accept stale tags or indefinite matrices at small scale.
    scales = numpy.max(numpy.abs(dms), axis=(1, 2))
    tolerances = 10 * nao * numpy.finfo(numpy.float64).eps * scales
    if numpy.any(numpy.max(numpy.abs(dms - dms.transpose(0, 2, 1)),
                           axis=(1, 2)) > tolerances):
        return None

    def valid(xs):
        return all(numpy.max(numpy.abs(d - x @ x.T)) <= tol
                   for d, x, tol in zip(dms, xs, tolerances))

    coeff = getattr(dm, 'mo_coeff', None)
    occ = getattr(dm, 'mo_occ', None)
    if coeff is not None and occ is not None:
        try:
            coeff, occ = numpy.asarray(coeff), numpy.asarray(occ)
        except (TypeError, ValueError):
            # Malformed optional tags must not invalidate the actual density.
            coeff = occ = None
    if coeff is not None and occ is not None:
        if (coeff.ndim >= 2 and coeff.shape[-2] == nao
                and coeff.dtype.kind == 'f' and occ.dtype.kind in 'fi'
                and numpy.all(numpy.isfinite(coeff))
                and numpy.all(numpy.isfinite(occ)) and numpy.all(occ >= 0)
                and coeff.shape[-1] > 0
                and coeff.size == len(dms) * nao * coeff.shape[-1]
                and occ.size == len(dms) * coeff.shape[-1]):
            coeff = coeff.reshape(len(dms), nao, -1)
            occ = occ.reshape(len(dms), -1)
            xs = tuple(numpy.asarray(c[:, o > 0] * numpy.sqrt(o[o > 0]),
                                     dtype=numpy.float64, order='F')
                       for c, o in zip(coeff, occ))
            if valid(xs):
                return xs

    xs = []
    for d in dms:
        c, piv, rank, info = dpstrf(d, lower=1)
        if info < 0:
            return None
        x = numpy.zeros((nao, rank), order='F')
        x[piv - 1] = numpy.tril(c[:, :rank])
        xs.append(x)
    return tuple(xs) if valid(xs) else None


def _tag_dm(dm, hermi=1):
    """Give PySCF validated factors, stripping stale tags on dense fallback."""
    xs = _density_factors(dm, hermi)
    dms = numpy.asarray(dm)
    if xs is None:
        return dms
    nao = dms.shape[-1]
    # PySCF reshapes tags with an inferred dimension, which needs at least
    # one (unoccupied) column even for an identically zero density.
    rank = max(1, max(x.shape[1] for x in xs))
    coeff = numpy.zeros((len(xs), nao, rank))
    occ = numpy.zeros((len(xs), rank))
    for k, x in enumerate(xs):
        coeff[k, :, :x.shape[1]] = x
        occ[k, :x.shape[1]] = 1
    return lib.tag_array(dms, mo_coeff=coeff, mo_occ=occ)


def _implicit_lowrank_df_k_enabled():
    value = os.environ.get('PYSCFAD_SCF_IMPLICIT_LOWRANK_DF_K', '1')
    return value.strip().lower() not in ('0', 'false', 'no', 'off')


def _implicit_lowrank_df_k_tolerance():
    try:
        return max(float(os.environ.get(
            'PYSCFAD_SCF_IMPLICIT_LOWRANK_DF_K_TOL', '1e-8'
        )), 0.0)
    except ValueError:
        return 1e-8


def _implicit_lowrank_exchange_factors(dfobj, vj_bar, vk_bar, hermi,
                                       with_j, with_k):
    """Recover ``vk_bar = R Co.T + Co R.T`` for an RHF response matvec.

    The fixed-point map returns an occupied projector.  Its Fock cotangent
    therefore has only occupied--virtual blocks, even though generic AD has
    materialized it as a dense AO matrix.  Recovering the thin factor here
    lets the DF exchange transpose use forward-like occupied scaling without
    changing the AO-density Krylov variable or any post-solve orbital VJP.
    """
    if not (
        _implicit_lowrank_df_k_enabled()
        and hermi == 1
        and with_j
        and with_k
    ):
        return None

    data = _fast_exchange_cache_get(dfobj)
    if data is None:
        _profile_msg(
            'implicit low-rank DF-K fallback: occupied-orbital cache miss '
            f'key={_fast_exchange_key(dfobj)!r}'
        )
        return None
    occ_coeff = data.get('occ_coeff')
    overlap_occ_coeff = data.get('overlap_occ_coeff')
    mo_occ = data.get('mo_occ')
    if occ_coeff is None or overlap_occ_coeff is None:
        _profile_msg(
            'implicit low-rank DF-K fallback: occupied factors unavailable'
        )
        return None
    if (
        mo_occ.ndim != 1
        or numpy.count_nonzero(mo_occ > 0) != occ_coeff.shape[1]
        or not numpy.allclose(mo_occ[mo_occ > 0], 2.0)
        or not numpy.allclose(mo_occ[mo_occ <= 0], 0.0)
    ):
        _profile_msg(
            'implicit low-rank DF-K fallback: occupations are not closed-shell'
        )
        return None

    vj_bar = numpy.asarray(vj_bar)
    vk_bar = numpy.asarray(vk_bar)
    if vj_bar.shape[0] != 1 or vk_bar.shape[0] != 1:
        _profile_msg(
            'implicit low-rank DF-K fallback: batched density cotangent'
        )
        return None
    if (
        vj_bar.ndim != 3
        or vk_bar.ndim != 3
        or vj_bar.shape != vk_bar.shape
        or vj_bar.shape[1] != vj_bar.shape[2]
        or occ_coeff.ndim != 2
        or overlap_occ_coeff.shape != occ_coeff.shape
        or occ_coeff.shape[0] != vk_bar.shape[-1]
    ):
        _profile_msg(
            'implicit low-rank DF-K fallback: incompatible factor shapes'
        )
        return None
    if any(numpy.iscomplexobj(array) for array in (
            occ_coeff, overlap_occ_coeff, vj_bar, vk_bar)):
        _profile_msg(
            'implicit low-rank DF-K fallback: complex factors'
        )
        return None
    if not all(numpy.all(numpy.isfinite(array)) for array in (
            occ_coeff, overlap_occ_coeff, vj_bar, vk_bar)):
        _profile_msg(
            'implicit low-rank DF-K fallback: non-finite factors'
        )
        return None

    # In RHF get_veff = J - K/2.  This relation makes the route specific to
    # the SCF fixed-point response rather than to arbitrary implicit solves
    # that happen to contain a DF get_jk call.
    scale = max(float(numpy.linalg.norm(vk_bar[0])), numpy.finfo(float).tiny)
    relation_error = float(
        numpy.linalg.norm(vk_bar[0] + 0.5 * vj_bar[0]) / scale
    )
    tol = _implicit_lowrank_df_k_tolerance()
    if relation_error > tol:
        _profile_msg(
            'implicit low-rank DF-K fallback: J/K cotangent relation '
            f'error={relation_error:.3e}'
        )
        return None

    symmetry_error = max(
        float(numpy.linalg.norm(vj_bar[0] - vj_bar[0].T) / scale),
        float(numpy.linalg.norm(vk_bar[0] - vk_bar[0].T) / scale),
    )
    if symmetry_error > tol:
        _profile_msg(
            'implicit low-rank DF-K fallback: cotangent symmetry error='
            f'{symmetry_error:.3e}'
        )
        return None

    vk_symmetric = 0.5 * (vk_bar[0] + vk_bar[0].T)
    response_coeff = vk_symmetric @ overlap_occ_coeff
    occupied_block = overlap_occ_coeff.T @ response_coeff
    response_coeff = numpy.asarray(
        response_coeff - 0.5 * occ_coeff @ occupied_block, order='F'
    )
    reconstructed = (
        response_coeff @ occ_coeff.T
        + occ_coeff @ response_coeff.T
    )
    residual = float(
        numpy.linalg.norm(vk_symmetric - reconstructed) / scale
    )
    if residual > tol:
        _profile_msg(
            'implicit low-rank DF-K fallback: occupied--virtual residual='
            f'{residual:.3e}'
        )
        return None

    _profile_msg(
        'implicit low-rank DF-K factors '
        f'rank={occ_coeff.shape[1]} residual={residual:.3e}'
    )
    return occ_coeff, response_coeff


def _df_vk_dm_vjp_lowrank(dm_bar, eri1, occ_coeff, response_coeff,
                          buf_occ, buf_response):
    """Apply the DF exchange transpose from thin occupied/response factors."""
    naux, _ = eri1.shape
    nao, nocc = occ_coeff.shape
    fmmm = libao2mo.AO2MOmmm_bra_nr_s2
    fdrv = libao2mo.AO2MOnr_e2_drv
    ftrans = libao2mo.AO2MOtranse2_nr_s2
    null = lib.c_null_ptr()
    rargs = (
        ctypes.c_int(nao),
        (ctypes.c_int * 4)(0, nocc, 0, nao),
        null,
        ctypes.c_int(0),
    )

    occ_block = buf_occ[:naux * nocc]
    response_block = buf_response[:naux * nocc]
    fdrv(
        ftrans, fmmm,
        occ_block.ctypes.data_as(ctypes.c_void_p),
        eri1.ctypes.data_as(ctypes.c_void_p),
        occ_coeff.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(naux), *rargs,
    )
    fdrv(
        ftrans, fmmm,
        response_block.ctypes.data_as(ctypes.c_void_p),
        eri1.ctypes.data_as(ctypes.c_void_p),
        response_coeff.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(naux), *rargs,
    )
    cross = lib.dot(occ_block.T, response_block)
    dm_bar += cross
    dm_bar += cross.T


def _cderi_mol_aux_vjp(dfobj, eri_bar):
    auxmol = dfobj.auxmol
    if auxmol is None:
        return None, None

    get_cderi = getattr(dfobj, '_get_cderi_source', None)
    cderi_source = (
        get_cderi() if get_cderi is not None else getattr(dfobj, '_cderi', None)
    )
    try:
        mol_bar, auxmol_bar = _cderi_vjp.cholesky_eri_vjp_from_cderi_source(
            dfobj.mol,
            auxmol,
            cderi_source,
            eri_bar,
            max(dfobj.max_memory, 4096),
            int3c=dfobj.mol._add_suffix('int3c2e'),
            int2c=dfobj.mol._add_suffix('int2c2e'),
            aosym='s2ij',
        )
        mol_bar_leaves = tree_flatten(mol_bar)[0]
        auxmol_bar_leaves = tree_flatten(auxmol_bar)[0]
        mol_bar = mol_bar_leaves[0] if mol_bar_leaves else None
        auxmol_bar = auxmol_bar_leaves[0] if auxmol_bar_leaves else None
        return mol_bar, auxmol_bar
    except NotImplementedError:
        pass

    def build_cderi(mol, auxmol):
        max_memory = max(dfobj.max_memory, 4096)
        return df_incore.cholesky_eri(
            mol,
            auxmol=auxmol,
            int3c=mol._add_suffix('int3c2e'),
            int2c=mol._add_suffix('int2c2e'),
            max_memory=max_memory,
            verbose=0,
        )

    _, pullback = jax.vjp(build_cderi, dfobj.mol, auxmol)
    mol_bar, auxmol_bar = pullback(eri_bar)
    mol_bar = tree_flatten(mol_bar)[0][0]
    auxmol_bar = tree_flatten(auxmol_bar)[0][0]
    return mol_bar, auxmol_bar


def _cderi_mol_aux_vjp_from_block_fn(dfobj, eri_bar_block_fn, *,
                                      int3c_block_vjp=None,
                                      max_pair_block=None):
    auxmol = dfobj.auxmol
    if auxmol is None:
        return None, None

    get_cderi = getattr(dfobj, '_get_cderi_source', None)
    cderi_source = (
        get_cderi() if get_cderi is not None else getattr(dfobj, '_cderi', None)
    )
    mol_bar, auxmol_bar = _cderi_vjp.cholesky_eri_vjp_from_cderi_block_fn(
        dfobj.mol,
        auxmol,
        cderi_source,
        eri_bar_block_fn,
        max(dfobj.max_memory, 4096),
        int3c=dfobj.mol._add_suffix('int3c2e'),
        int2c=dfobj.mol._add_suffix('int2c2e'),
        aosym='s2ij',
        int3c_block_vjp=int3c_block_vjp,
        max_pair_block=max_pair_block,
    )
    mol_bar_leaves = tree_flatten(mol_bar)[0]
    auxmol_bar_leaves = tree_flatten(auxmol_bar)[0]
    mol_bar = mol_bar_leaves[0] if mol_bar_leaves else None
    auxmol_bar = auxmol_bar_leaves[0] if auxmol_bar_leaves else None
    return mol_bar, auxmol_bar


def _has_tracer(*xs):
    leaves = []
    for x in xs:
        if x is None:
            continue
        x_leaves, _ = tree_flatten(x)
        leaves.extend(x_leaves)
    return any(isinstance(x, jax.core.Tracer) for x in leaves)


@partial(custom_vjp, nondiff_argnums=(2,3,4,5))
def get_jk(dfobj, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-13):
    dm_forward = dm
    if with_k:
        dm_forward = _tag_dm(_tag_dm_for_fast_exchange(dfobj, dm), hermi)
    return pyscf_df_jk.get_jk(dfobj, dm_forward, hermi=hermi,
                              with_j=with_j, with_k=with_k,
                              direct_scf_tol=direct_scf_tol)


def _unwrap_jk_primals(dfobj, dm):
    """Capture independent density, physical-factor and geometry activity.

    Adapted from upstream PySCFAD db0acafb (Apache-2.0). Unlike the upstream
    resident-only rule, an inactive disk placeholder does not freeze geometry.
    """
    def active(value):
        return any(isinstance(x, CustomVJPPrimal) and x.perturbed
                   for x in tree_flatten(value)[0])
    activity = (active(dm), active(dfobj._cderi),
                active(dfobj.mol), active(dfobj.auxmol))
    unwrap = lambda x: x.value if isinstance(x, CustomVJPPrimal) else x
    return tree_map(unwrap, dfobj), unwrap(dm), activity


def _jk_residual(res):
    if len(res) == 2:
        # Compatibility for direct callers predating symbolic activity.
        return res[0], res[1], (True, True, True, True)
    return res


def _active_output_bar(bar, enabled):
    return bar if enabled and not isinstance(bar, SymbolicZero) else None


def get_jk_fwd(dfobj, dm, hermi, with_j, with_k, direct_scf_tol):
    dfobj, dm, activity = _unwrap_jk_primals(dfobj, dm)
    vj, vk = get_jk(dfobj, dm, hermi, with_j, with_k, direct_scf_tol)
    return (vj, vk), (dfobj, dm, activity)


def get_jk_bwd(hermi, with_j, with_k, direct_scf_tol,
               res, ybar, *, coordinate_vjp=None):
    t_bwd = time.perf_counter()
    _profile_msg(f'get_jk_bwd start hermi={hermi} with_j={with_j} with_k={with_k}')
    dfobj, dm, activity = _jk_residual(res)
    dm_active, eri_active, mol_active, aux_active = activity
    vj_bar = _active_output_bar(ybar[0], with_j)
    vk_bar = _active_output_bar(ybar[1], with_k)
    solution_only = is_implicit_diff_solve_matvec()
    if vj_bar is None and vk_bar is None:
        return None, None
    if _has_tracer(dfobj, dm, vj_bar, vk_bar):
        if (not solution_only and (mol_active or aux_active)
                and dfobj.auxmol is not None
                and getattr(dfobj._cderi, 'shape', None) == (0, 0)):
            raise NotImplementedError(
                'traced DF geometry with a disk/source CDERI placeholder '
                'is not supported')
        # The generic tensor algebra depends on resident factors and density.
        # Differentiate only active inputs, matching the eager rule's absent
        # factor/geometry bars instead of returning materialized inactive bars.
        factor_active = eri_active and not solution_only
        if not (factor_active or dm_active):
            return None, None
        primals = ((dfobj._cderi,) if factor_active else ())
        primals += (dm,) if dm_active else ()
        def fn(*active):
            dfobj_ = dfobj
            if factor_active:
                dfobj_ = tree_map(lambda x: x, dfobj)
                dfobj_._cderi = active[0]
            dm_ = active[-1] if dm_active else dm
            return _get_jk_gen_jax(
                dfobj_, dm_, hermi=hermi, with_j=with_j, with_k=with_k,
                direct_scf_tol=direct_scf_tol)
        _, vjp = jax.vjp(fn, *primals)
        zeros = jnp.zeros_like(dm)
        bars = vjp((zeros if vj_bar is None else vj_bar,
                    zeros if vk_bar is None else vk_bar))
        df_bar = None
        if factor_active:
            df_bar = tree_map(lambda x: None, dfobj)
            df_bar._cderi = bars[0]
        return df_bar, bars[-1] if dm_active else None

    log = logger.new_logger(dfobj)
    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    nao_pair = nao * (nao + 1) // 2
    dms = dms.reshape(-1, nao, nao)
    nset = len(dms)
    naoaux = dfobj.get_naoaux()
    eri_bar_shape = (naoaux, nao_pair)
    physical_cderi = (hasattr(dfobj._cderi, 'shape')
                      and tuple(dfobj._cderi.shape) == eri_bar_shape)
    return_cderi_bar = physical_cderi and eri_active and not solution_only
    geometry_active = (not physical_cderi and (mol_active or aux_active)
                       and dfobj.auxmol is not None and not solution_only)
    need_eri = return_cderi_bar or geometry_active
    if not (dm_active or need_eri):
        return None, None
    if vj_bar is not None:
        vj_bar = numpy.asarray(vj_bar).reshape(-1, nao, nao)
        idx = numpy.arange(nao)
        vj_bar_tril = lib.pack_tril(vj_bar + vj_bar.transpose(0, 2, 1))
        vj_bar_tril[:, idx*(idx+1)//2+idx] *= .5
        if need_eri:
            dmtril = lib.pack_tril(dms + dms.transpose(0, 2, 1))
            dmtril[:, idx*(idx+1)//2+idx] *= .5
    vk_array = (None if vk_bar is None else
                numpy.asarray(vk_bar).reshape(-1, nao, nao))
    response_factors = None
    if solution_only and dm_active and vj_bar is not None and vk_array is not None:
        response_factors = _implicit_lowrank_exchange_factors(
            dfobj, vj_bar, vk_array, hermi, with_j, with_k)
    xs = (_density_factors(_tag_dm_for_fast_exchange(dfobj, dm), hermi)
          if need_eri and vk_array is not None and _DF_VK_VJP_LOWRANK is not None
          else None)
    vk_f = ([numpy.asarray(x, order='F') for x in vk_array]
            if vk_array is not None else None)
    symmetric_vk = (numpy.asarray(vk_array + vk_array.transpose(0, 2, 1), order='C')
                    if xs is not None else None)
    dms = [numpy.asarray(x, order='F') for x in dms]
    max_memory = dfobj.max_memory - lib.current_memory()[0]
    blksize = max(4, int(min(dfobj.blockdim, max_memory*.3e6/8/nao**2)))
    # Dense fallback computes both bars in one native call. Its unused outputs
    # are bounded block-local scratch, never global factor storage.
    dense_k = vk_array is not None and (
        (need_eri and xs is None)
        or (dm_active and response_factors is None and _DF_VK_DM_VJP is None))
    dms_bar = ([numpy.zeros((nao, nao), order='F') for _ in range(nset)]
               if dm_active or dense_k else None)
    buf = numpy.empty((blksize, nao, nao)) if dense_k else None
    if response_factors is not None:
        occ_coeff, response_coeff = response_factors
        buf_occ = numpy.empty((blksize * occ_coeff.shape[1], nao))
        buf_response = numpy.empty_like(buf_occ)
    stream_eri_bar = geometry_active
    eri_bar = numpy.zeros(eri_bar_shape) if return_cderi_bar else None
    eri_bar_file = eri_bar_path = eri_bar_h5 = None
    p1 = 0
    try:
        if stream_eri_bar:
            fd, eri_bar_path = tempfile.mkstemp(
                suffix='.h5', prefix='pyscfad_dfjk_eri_bar_', dir=lib.param.TMPDIR)
            os.close(fd)
            eri_bar_file = h5py.File(eri_bar_path, 'w')
            eri_bar_h5 = eri_bar_file.create_dataset(
                'eri_bar_T', shape=(nao_pair, naoaux), dtype=numpy.float64,
                chunks=(min(nao_pair, 1024), min(naoaux, max(blksize, 16))))
        t_loop = time.perf_counter()
        for eri1 in dfobj.loop(blksize):
            naux, pair = eri1.shape
            if pair != nao_pair:
                raise RuntimeError(f'DF block has pair dimension {pair}, expected {nao_pair}.')
            p0, p1 = p1, p1 + naux
            eri_blk = (eri_bar[p0:p1] if return_cderi_bar else
                       numpy.zeros((naux, nao_pair)) if need_eri or dense_k else None)
            if vj_bar is not None:
                rho_bar = vj_bar_tril @ eri1.T
                if dm_active:
                    # Keep each native output independent and Fortran ordered.
                    dense_dm_bar = lib.unpack_tril(rho_bar @ eri1)
                    for k in range(nset):
                        dms_bar[k] += dense_dm_bar[k]
                if need_eri:
                    eri_blk += (dmtril @ eri1.T).T @ vj_bar_tril
                    eri_blk += rho_bar.T @ dmtril
            if vk_array is not None:
                for k in range(nset):
                    if dense_k:
                        buf1 = buf[:naux]
                        libao2mo.AO2MOnr_e2_drv(
                            libao2mo.AO2MOtranse2_nr_s2, libao2mo.AO2MOmmm_bra_nr_s2,
                            buf1.ctypes.data_as(ctypes.c_void_p),
                            eri1.ctypes.data_as(ctypes.c_void_p),
                            dms[k].ctypes.data_as(ctypes.c_void_p), ctypes.c_int(naux),
                            ctypes.c_int(nao), (ctypes.c_int*4)(0, nao, 0, nao),
                            lib.c_null_ptr(), ctypes.c_int(0))
                        libvhf.df_vk_vjp(
                            eri_blk.ctypes.data_as(ctypes.c_void_p),
                            dms_bar[k].ctypes.data_as(ctypes.c_void_p),
                            vk_f[k].ctypes.data_as(ctypes.c_void_p),
                            buf1.ctypes.data_as(ctypes.c_void_p),
                            eri1.ctypes.data_as(ctypes.c_void_p),
                            dms[k].ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(naux), ctypes.c_int(nao))
                    else:
                        if need_eri and xs[k].shape[1]:
                            _DF_VK_VJP_LOWRANK(
                                eri_blk.ctypes.data_as(ctypes.c_void_p),
                                symmetric_vk[k].ctypes.data_as(ctypes.c_void_p),
                                xs[k].ctypes.data_as(ctypes.c_void_p),
                                eri1.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(naux), ctypes.c_int(nao),
                                ctypes.c_int(xs[k].shape[1]))
                        if dm_active:
                            if response_factors is not None:
                                _df_vk_dm_vjp_lowrank(dms_bar[k], eri1, occ_coeff,
                                                     response_coeff, buf_occ, buf_response)
                            else:
                                _DF_VK_DM_VJP(
                                    dms_bar[k].ctypes.data_as(ctypes.c_void_p),
                                    vk_f[k].ctypes.data_as(ctypes.c_void_p),
                                    eri1.ctypes.data_as(ctypes.c_void_p),
                                    ctypes.c_int(naux), ctypes.c_int(nao))
            if stream_eri_bar:
                eri_bar_h5[:, p0:p1] = eri_blk.T
        _profile_msg(f'get_jk_bwd DF block loop done naux={p1} blksize={blksize} '
                     f'{time.perf_counter() - t_loop:.2f} s')
        dm_bar = numpy.asarray(dms_bar).reshape(dm_shape) if dm_active else None
        if not need_eri:
            log.timer('get_jk_bwd')
            if solution_only:
                _profile_msg('get_jk_bwd density-only implicit-solve matvec done '
                             f'{time.perf_counter() - t_bwd:.2f} s')
            return None, dm_bar
        df_bar = tree_map(lambda x: None, dfobj)
        if return_cderi_bar:
            df_bar._cderi = eri_bar
        if geometry_active:
            t_cderi = time.perf_counter()
            _profile_msg('get_jk_bwd cderi mol/aux vjp start')
            eri_bar_file.flush()
            def eri_bar_block_fn(q0, q1):
                return numpy.asarray(eri_bar_h5[q0:q1, :]).T
            pullback = (_cderi_mol_aux_vjp_from_block_fn
                        if coordinate_vjp is None else coordinate_vjp)
            mol_bar, aux_bar = pullback(dfobj, eri_bar_block_fn)
            _profile_msg('get_jk_bwd cderi mol/aux vjp done '
                         f'{time.perf_counter() - t_cderi:.2f} s')
            if mol_active:
                _, mol_tree = tree_flatten(dfobj.mol)
                df_bar.mol = tree_unflatten(mol_tree, [mol_bar])
            if aux_active:
                _, aux_tree = tree_flatten(dfobj.auxmol)
                df_bar.auxmol = tree_unflatten(aux_tree, [aux_bar])
    finally:
        if eri_bar_file is not None:
            eri_bar_file.close()
        if eri_bar_path is not None:
            try:
                os.remove(eri_bar_path)
            except OSError:
                pass
    log.timer('get_jk_bwd')
    _profile_msg(f'get_jk_bwd done {time.perf_counter() - t_bwd:.2f} s')
    return df_bar, dm_bar


get_jk.defvjp(get_jk_fwd, get_jk_bwd, symbolic_zeros=True)
