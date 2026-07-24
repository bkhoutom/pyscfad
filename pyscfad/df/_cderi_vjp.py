# Copyright 2026 The PySCFAD Authors
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

"""Memory-conscious VJP helpers for out-of-core molecular DF CDERI."""

from functools import lru_cache
import ctypes
import os
import time

import numpy
import scipy.linalg
import jax
from jax import scipy as jax_scipy
from jax.tree_util import tree_flatten, tree_map, tree_unflatten
from pyscf import __config__
from pyscf import lib as pyscf_lib
from pyscf.ao2mo import _ao2mo as pyscf_ao2mo
from pyscf.ao2mo.outcore import balance_partition
from pyscf.df.outcore import _guess_shell_ranges

from pyscfad import numpy as np
from pyscfad.ao2mo import _ao2mo
from pyscfad.df import _int3c_cross_opt
from pyscfad.df import addons as df_addons
from pyscfad.tools import resource_profile
try:
    from pyscfadlib import libao2mo_vjp
except (ImportError, OSError):
    libao2mo_vjp = None


_INT3C_VJP_TARGET_MB = 256.0
_NR_E2_VJP_BLOCK_MB = 256.0
_CDERI_BAR_PAIR_BLOCK_MB = 256.0
_INT3C_MO_VJP_BLOCK_MB = 512.0
_NR_E2_CDERI_BAR_NATIVE = (
    libao2mo_vjp is not None
    and hasattr(libao2mo_vjp, 'AO2MOnr_e2_cderi_bar_project_omp')
)
if _NR_E2_CDERI_BAR_NATIVE:
    libao2mo_vjp.AO2MOnr_e2_cderi_bar_project_omp.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    libao2mo_vjp.AO2MOnr_e2_cderi_bar_project_omp.restype = None


def _profile_enabled():
    value = os.environ.get('PYSCFAD_PROFILE_BACKWARD_PHASES')
    return value is not None and value.strip().lower() in ('1', 'true', 'yes', 'on')


def _profile_msg(msg):
    if _profile_enabled():
        print(f'[profile][df.cderi_vjp] {msg}', flush=True)


@lru_cache(maxsize=16)
def _tril_indices(nao):
    return numpy.tril_indices(nao)


def _tree_add(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return tree_map(lambda a, b: a + b, x, y)


def nr_e2_vjp_from_cderi_source(cderi_source, mo_coeff, ybar,
                                orbs_slice, aosym='s2', mosym='s1'):
    """Return CDERI and MO-coefficient cotangents using stored CDERI."""
    if cderi_source is None:
        raise NotImplementedError('Missing CDERI source for nr_e2 VJP.')
    with df_addons.load(cderi_source, 'j3c') as eri1:
        if not hasattr(eri1, 'shape'):
            raise NotImplementedError('Unsupported CDERI source for nr_e2 VJP.')
        cderi = numpy.asarray(eri1)

    def fn(cderi_, mo_coeff_):
        return _ao2mo.nr_e2(cderi_, mo_coeff_, orbs_slice,
                            aosym=aosym, mosym=mosym)

    _, pullback = jax.vjp(fn, np.asarray(cderi), mo_coeff)
    return pullback(ybar)


def _nr_e2_vjp_block_mb():
    return _NR_E2_VJP_BLOCK_MB


def _cderi_bar_pair_block_mb():
    return _CDERI_BAR_PAIR_BLOCK_MB


def _int3c_mo_vjp_block_mb():
    return _INT3C_MO_VJP_BLOCK_MB


def _use_native_cderi_bar_project(ybar, mok_rows, mol_cols):
    return (
        _NR_E2_CDERI_BAR_NATIVE
        and ybar.dtype == numpy.float64
        and mok_rows.dtype == numpy.float64
        and mol_cols.dtype == numpy.float64
    )


def _nr_e2_cderi_bar_project_native(ybar, mok_rows, mol_cols, blksize):
    naux, kc, lc = ybar.shape
    npos = mok_rows.shape[0]
    y2 = numpy.asarray(
        ybar.transpose(0, 2, 1).reshape(naux * lc, kc),
        order='C',
        dtype=numpy.double,
    )
    mok_rows = numpy.asarray(mok_rows, order='C', dtype=numpy.double)
    mol_cols = numpy.asarray(mol_cols, order='C', dtype=numpy.double)
    out = numpy.empty((naux, npos), order='C', dtype=numpy.double)
    drv = libao2mo_vjp.AO2MOnr_e2_cderi_bar_project_omp
    drv(
        out.ctypes.data_as(ctypes.c_void_p),
        y2.ctypes.data_as(ctypes.c_void_p),
        mok_rows.ctypes.data_as(ctypes.c_void_p),
        mol_cols.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(naux),
        ctypes.c_int(kc),
        ctypes.c_int(lc),
        ctypes.c_int(npos),
        ctypes.c_int(blksize),
    )
    return out


def _nr_e2_cderi_bar_project(ybar, mok_rows, mol_cols):
    """Project ``ybar[Lij]`` onto pair-specific MO rows.

    The direct expression ``einsum('Lij,pi,pj->Lp', ...)`` often dispatches to
    NumPy's generic ``c_einsum`` loop, which is single-threaded.  Splitting the
    contraction makes the dominant ``i`` contraction a GEMM-shaped
    ``numpy.dot`` so threaded BLAS can do the heavy work.
    """
    naux, kc, lc = ybar.shape
    npos = mok_rows.shape[0]
    if npos == 0:
        return numpy.zeros((naux, 0), dtype=ybar.dtype)

    target_bytes = _cderi_bar_pair_block_mb() * 1e6
    itemsize = numpy.dtype(ybar.dtype).itemsize
    blksize = max(int(target_bytes / max(naux * lc * itemsize, 1)), 1)
    blksize = min(blksize, npos)

    if _use_native_cderi_bar_project(ybar, mok_rows, mol_cols):
        return _nr_e2_cderi_bar_project_native(
            ybar, mok_rows, mol_cols, blksize
        )

    y2 = numpy.asarray(ybar.transpose(0, 2, 1).reshape(naux * lc, kc), order='C')
    out = numpy.empty((naux, npos), dtype=ybar.dtype)
    for p0 in range(0, npos, blksize):
        p1 = min(p0 + blksize, npos)
        tmp = numpy.dot(y2, numpy.asarray(mok_rows[p0:p1], order='C').T)
        tmp = tmp.reshape(naux, lc, p1 - p0)
        out[:, p0:p1] = numpy.sum(
            tmp * mol_cols[p0:p1].T[None, :, :],
            axis=1,
        )
    return out


def _df_jk_style_blockdim(max_memory, row_width):
    blockdim = getattr(__config__, 'df_df_DF_blockdim', 240)
    if max_memory is None:
        return int(blockdim)
    max_memory = max(float(max_memory) - pyscf_lib.current_memory()[0], 1.0)
    return max(4, int(min(blockdim, max_memory*.3e6/8/max(row_width, 1))))


def nr_e2_mo_coeff_vjp_from_cderi_source(cderi_source, mo_coeff, ybar,
                                         orbs_slice, aosym='s2',
                                         mosym='s1', pair_idx=None,
                                         max_memory=None):
    """Return only the MO-coefficient cotangent, streaming CDERI rows."""
    if cderi_source is None:
        raise NotImplementedError('Missing CDERI source for nr_e2 VJP.')
    if aosym not in ('s2', 's2ij'):
        raise NotImplementedError(f'Only packed s2 CDERI is supported, got {aosym}.')

    resource_start = resource_profile.start()
    mo_coeff_bar = None
    ybar = numpy.asarray(jax.device_get(ybar))
    with df_addons.load(cderi_source, 'j3c') as eri1:
        if not hasattr(eri1, 'shape'):
            raise NotImplementedError('Unsupported CDERI source for nr_e2 VJP.')
        naux = int(eri1.shape[0])
        if pair_idx is None:
            npair = int(eri1.shape[1])
        else:
            pair_idx = numpy.asarray(pair_idx, dtype=numpy.int64)
            if (
                pair_idx.size == int(eri1.shape[1])
                and (pair_idx.size == 0 or (
                    pair_idx[0] == 0
                    and pair_idx[-1] == pair_idx.size - 1
                    and numpy.all(numpy.diff(pair_idx) == 1)
                ))
            ):
                pair_idx = None
                npair = int(eri1.shape[1])
            else:
                npair = int(pair_idx.size)

        blksize = max(
            1,
            min(naux, _df_jk_style_blockdim(max_memory, npair + ybar.shape[1])),
        )

        for p0 in range(0, naux, blksize):
            p1 = min(p0 + blksize, naux)
            if pair_idx is None:
                cderi = numpy.asarray(eri1[p0:p1])
            else:
                cderi = numpy.asarray(eri1[p0:p1, pair_idx])

            mo_coeff_bar_blk = _ao2mo.nr_e2_mo_coeff_vjp(
                cderi, mo_coeff, ybar[p0:p1], orbs_slice,
                aosym=aosym, mosym=mosym,
            )
            mo_coeff_bar = _tree_add(mo_coeff_bar, mo_coeff_bar_blk)
            cderi = mo_coeff_bar_blk = None
    resource_profile.finish(
        'df_vjp.mo_coeff_stream',
        resource_start,
        naux=naux,
        pair_count=npair,
        aux_block=blksize,
        blocks=(naux + blksize - 1) // blksize,
        ybar_mib=resource_profile.estimated_array_mib(ybar),
        configured_memory_mib=max_memory,
        pyscf_current_memory_mib=pyscf_lib.current_memory()[0],
    )
    return mo_coeff_bar


def nr_e2_cderi_bar_packed_block(mo_coeff, ybar, orbs_slice, pair_positions):
    """Build packed-CDERI cotangents for selected packed AO-pair positions.

    ``pair_positions`` are packed lower-triangular indices in the AO basis of
    ``mo_coeff``.  The returned array has shape ``(naux, len(pair_positions))``.
    """
    pair_positions = numpy.asarray(pair_positions, dtype=numpy.int64).ravel()
    naux = ybar.shape[0]
    if pair_positions.size == 0:
        return numpy.zeros((naux, 0), dtype=numpy.asarray(ybar).dtype)

    mo = numpy.asarray(jax.device_get(mo_coeff))
    ybar = numpy.asarray(jax.device_get(ybar))
    nao = mo.shape[0]
    k0, k1, l0, l1 = orbs_slice
    kc = k1 - k0
    lc = l1 - l0
    ybar = ybar.reshape(naux, kc, lc)

    rows_all, cols_all = _tril_indices(nao)
    rows = rows_all[pair_positions]
    cols = cols_all[pair_positions]

    mok_rows = mo[rows, k0:k1]
    mol_cols = mo[cols, l0:l1]
    out = _nr_e2_cderi_bar_project(ybar, mok_rows, mol_cols)

    offdiag = rows != cols
    if numpy.any(offdiag):
        mok_cols = mo[cols[offdiag], k0:k1]
        mol_rows = mo[rows[offdiag], l0:l1]
        out[:, offdiag] += _nr_e2_cderi_bar_project(
            ybar, mok_cols, mol_rows
        )
    return out


def cholesky_eri_vjp_from_cderi_source(mol, auxmol, cderi_source, cderi_bar,
                                       max_memory, int3c=None, int2c=None,
                                       aosym='s2ij'):
    """Back-propagate through Cholesky CDERI using saved CDERI blocks.

    This implements the reverse of the Cholesky-whitening step

        cderi = solve(chol(j2c), int3c)

    without rebuilding the full differentiable ``cderi`` primal.  The stored
    CDERI supplies the blockwise primal ``cderi`` needed for the triangular
    solve VJP; derivative integral VJPs are evaluated one AO-pair block at a
    time.
    """
    t_total = time.perf_counter()
    _profile_msg('cholesky_eri_vjp_from_cderi_source start')
    if aosym not in ('s2', 's2ij'):
        raise NotImplementedError(f'Only packed s2 CDERI is supported, got {aosym}.')
    if cderi_source is None:
        raise NotImplementedError('Missing CDERI source for cholesky_eri VJP.')

    if int3c is None:
        int3c = mol._add_suffix('int3c2e')
    if int2c is None:
        int2c = mol._add_suffix('int2c2e')

    cderi_bar = numpy.asarray(jax.device_get(cderi_bar))
    naux, nao_pair = cderi_bar.shape
    nao = mol.nao
    if nao_pair != nao * (nao + 1) // 2:
        raise NotImplementedError('Only packed s2 CDERI cotangents are supported.')

    t = time.perf_counter()
    j2c = auxmol.intor(int2c, hermi=1)
    j2c_np = numpy.asarray(jax.device_get(j2c))
    try:
        low = scipy.linalg.cholesky(j2c_np, lower=True, check_finite=False)
    except scipy.linalg.LinAlgError as err:
        raise NotImplementedError('2c metric Cholesky fallback is not implemented.') from err

    naoaux = low.shape[0]
    if low.shape[0] != low.shape[1] or naux != naoaux:
        raise NotImplementedError(
            'Linear-dependent auxiliary metric fallback is not implemented.'
        )
    _profile_msg(
        'cholesky_eri_vjp_from_cderi_source metric setup done '
        f'{time.perf_counter() - t:.2f} s'
    )

    max_words = max(max_memory, 0) * 1e6 / 8 - low.size - naux * nao_pair
    mem_buflen = max(int(max_words / max(naoaux, 1) / 2), 8)
    target_buflen = max(int(_INT3C_VJP_TARGET_MB * 1e6 / 8 / max(naoaux, 1) / 3), 8)
    buflen = min(nao_pair, mem_buflen, target_buflen)
    shranges = _guess_shell_ranges(mol, buflen, aosym)

    mol_bar = None
    auxmol_bar = None
    low_bar = numpy.zeros_like(low)

    t = time.perf_counter()
    nblocks = 0
    with df_addons.load(cderi_source, 'j3c') as feri:
        p1 = 0
        for sh_range in shranges:
            bstart, bend, _ = sh_range
            shls_slice = (
                bstart, bend,
                0, mol.nbas,
                mol.nbas, mol.nbas + auxmol.nbas,
            )

            p0, p1 = p1, p1 + sh_range[2]
            cderi_bar_blk = cderi_bar[:, p0:p1]
            if not numpy.any(cderi_bar_blk):
                continue
            nblocks += 1

            cderi_blk = numpy.asarray(feri[:, p0:p1])
            ints_bar = scipy.linalg.solve_triangular(
                low.T, cderi_bar_blk, lower=False, check_finite=False
            )
            low_bar -= ints_bar @ cderi_blk.T

            def int3c_block(mol_, auxmol_):
                ints = _int3c_cross_opt.int3c_cross(
                    mol_, auxmol_, intor=int3c, comp=1,
                    aosym='s2ij', shls_slice=shls_slice
                )
                return ints.reshape((-1, naoaux)).T

            _, int3c_pullback = jax.vjp(int3c_block, mol, auxmol)
            mol_blk_bar, auxmol_blk_bar = int3c_pullback(np.asarray(ints_bar))
            mol_bar = _tree_add(mol_bar, mol_blk_bar)
            auxmol_bar = _tree_add(auxmol_bar, auxmol_blk_bar)
            cderi_blk = ints_bar = mol_blk_bar = auxmol_blk_bar = None
    _profile_msg(
        'cholesky_eri_vjp_from_cderi_source int3c block loop done '
        f'nblocks={nblocks} {time.perf_counter() - t:.2f} s'
    )

    if p1 != nao_pair:
        raise RuntimeError('CDERI VJP shell ranges did not cover all AO pairs.')

    def metric_cholesky(auxmol_):
        return jax_scipy.linalg.cholesky(auxmol_.intor(int2c, hermi=1), lower=True)

    t = time.perf_counter()
    _, chol_pullback = jax.vjp(metric_cholesky, auxmol)
    aux_metric_bar = chol_pullback(np.asarray(numpy.tril(low_bar)))[0]
    auxmol_bar = _tree_add(auxmol_bar, aux_metric_bar)
    _profile_msg(
        'cholesky_eri_vjp_from_cderi_source metric cholesky pullback done '
        f'{time.perf_counter() - t:.2f} s'
    )
    _profile_msg(
        'cholesky_eri_vjp_from_cderi_source done '
        f'{time.perf_counter() - t_total:.2f} s'
    )
    return mol_bar, auxmol_bar


def cholesky_eri_vjp_from_cderi_block_fn(mol, auxmol, cderi_source,
                                         cderi_bar_block_fn, max_memory,
                                         int3c=None, int2c=None,
                                         aosym='s2ij'):
    """Back-propagate through Cholesky CDERI from AO-pair cotangent blocks."""
    t_total = time.perf_counter()
    resource_total = resource_profile.start()
    _profile_msg('cholesky_eri_vjp_from_cderi_block_fn start')
    if aosym not in ('s2', 's2ij'):
        raise NotImplementedError(f'Only packed s2 CDERI is supported, got {aosym}.')
    if cderi_source is None:
        raise NotImplementedError('Missing CDERI source for cholesky_eri VJP.')

    if int3c is None:
        int3c = mol._add_suffix('int3c2e')
    if int2c is None:
        int2c = mol._add_suffix('int2c2e')

    nao = mol.nao
    nao_pair = nao * (nao + 1) // 2

    t = time.perf_counter()
    resource_phase = resource_profile.start()
    j2c = auxmol.intor(int2c, hermi=1)
    j2c_np = numpy.asarray(jax.device_get(j2c))
    try:
        low = scipy.linalg.cholesky(j2c_np, lower=True, check_finite=False)
    except scipy.linalg.LinAlgError as err:
        raise NotImplementedError('2c metric Cholesky fallback is not implemented.') from err

    naoaux = low.shape[0]
    if low.shape[0] != low.shape[1]:
        raise NotImplementedError(
            'Linear-dependent auxiliary metric fallback is not implemented.'
        )
    _profile_msg(
        'cholesky_eri_vjp_from_cderi_block_fn metric setup done '
        f'{time.perf_counter() - t:.2f} s'
    )
    resource_profile.finish(
        'df_vjp.metric_cholesky_setup',
        resource_phase,
        nao=nao,
        naux=naoaux,
        metric_mib=resource_profile.estimated_array_mib(j2c_np, low),
    )

    max_words = max(max_memory, 0) * 1e6 / 8 - low.size
    mem_buflen = max(int(max_words / max(naoaux, 1) / 3), 8)
    target_buflen = max(int(_INT3C_VJP_TARGET_MB * 1e6 / 8 / max(naoaux, 1) / 3), 8)
    buflen = min(nao_pair, mem_buflen, target_buflen)
    shranges = _guess_shell_ranges(mol, buflen, aosym)

    mol_bar = None
    auxmol_bar = None
    low_bar = numpy.zeros_like(low)

    t = time.perf_counter()
    resource_phase = resource_profile.start()
    nblocks = 0
    with df_addons.load(cderi_source, 'j3c') as feri:
        if int(feri.shape[0]) != naoaux or int(feri.shape[1]) != nao_pair:
            raise NotImplementedError('CDERI source shape does not match mol/auxmol.')

        p1 = 0
        for sh_range in shranges:
            bstart, bend, _ = sh_range
            shls_slice = (
                bstart, bend,
                0, mol.nbas,
                mol.nbas, mol.nbas + auxmol.nbas,
            )

            p0, p1 = p1, p1 + sh_range[2]
            cderi_bar_blk = numpy.asarray(cderi_bar_block_fn(p0, p1))
            if cderi_bar_blk.shape != (naoaux, p1 - p0):
                raise RuntimeError(
                    'CDERI cotangent block has shape '
                    f'{cderi_bar_blk.shape}, expected {(naoaux, p1 - p0)}.'
            )
            if not numpy.any(cderi_bar_blk):
                continue
            nblocks += 1

            cderi_blk = numpy.asarray(feri[:, p0:p1])

            ints_bar = scipy.linalg.solve_triangular(
                low.T, cderi_bar_blk, lower=False, check_finite=False
            )
            low_bar -= ints_bar @ cderi_blk.T

            def int3c_block(mol_, auxmol_):
                ints = _int3c_cross_opt.int3c_cross(
                    mol_, auxmol_, intor=int3c, comp=1,
                    aosym='s2ij', shls_slice=shls_slice
                )
                return ints.reshape((-1, naoaux)).T

            _, int3c_pullback = jax.vjp(int3c_block, mol, auxmol)
            mol_blk_bar, auxmol_blk_bar = int3c_pullback(np.asarray(ints_bar))
            mol_bar = _tree_add(mol_bar, mol_blk_bar)
            auxmol_bar = _tree_add(auxmol_bar, auxmol_blk_bar)
            cderi_bar_blk = cderi_blk = ints_bar = None
            mol_blk_bar = auxmol_blk_bar = None
    _profile_msg(
        'cholesky_eri_vjp_from_cderi_block_fn int3c block loop done '
        f'nblocks={nblocks} {time.perf_counter() - t:.2f} s'
    )
    resource_profile.finish(
        'df_vjp.int3c_derivative_block_loop',
        resource_phase,
        nao=nao,
        naux=naoaux,
        ao_pairs=nao_pair,
        shell_blocks=nblocks,
        target_pair_block=buflen,
        est_three_block_mib=(
            3.0 * naoaux * buflen * 8.0 / 1024.0**2
        ),
        configured_memory_mib=max_memory,
        pyscf_current_memory_mib=pyscf_lib.current_memory()[0],
    )

    if p1 != nao_pair:
        raise RuntimeError('CDERI VJP shell ranges did not cover all AO pairs.')

    def metric_cholesky(auxmol_):
        return jax_scipy.linalg.cholesky(auxmol_.intor(int2c, hermi=1), lower=True)

    t = time.perf_counter()
    resource_phase = resource_profile.start()
    _, chol_pullback = jax.vjp(metric_cholesky, auxmol)
    aux_metric_bar = chol_pullback(np.asarray(numpy.tril(low_bar)))[0]
    auxmol_bar = _tree_add(auxmol_bar, aux_metric_bar)
    _profile_msg(
        'cholesky_eri_vjp_from_cderi_block_fn metric cholesky pullback done '
        f'{time.perf_counter() - t:.2f} s'
    )
    resource_profile.finish(
        'df_vjp.metric_cholesky_pullback',
        resource_phase,
        naux=naoaux,
    )
    _profile_msg(
        'cholesky_eri_vjp_from_cderi_block_fn done '
        f'{time.perf_counter() - t_total:.2f} s'
    )
    resource_profile.finish(
        'df_vjp.cholesky_total',
        resource_total,
        nao=nao,
        naux=naoaux,
        shell_blocks=nblocks,
    )
    return mol_bar, auxmol_bar


def _coords_tree_like(obj, coords_bar):
    leaves, tree = tree_flatten(obj)
    if len(leaves) != 1:
        raise NotImplementedError(
            'Direct int3c-MO VJP currently supports coordinate derivatives only.'
        )
    return tree_unflatten(tree, [coords_bar])


def _ao_to_atom_coords_bar(mol, ao_bar):
    coords_bar = numpy.zeros((mol.natm, 3), dtype=ao_bar.dtype)
    for ia, (p0, p1) in enumerate(mol.aoslice_by_atom()[:, 2:4]):
        coords_bar[ia] = ao_bar[p0:p1].sum(axis=0)
    return coords_bar


def _stream_nr_e2_from_cderi_source(cderi_source, mo_coeff, orbs_slice,
                                    max_memory, aosym='s2'):
    with df_addons.load(cderi_source, 'j3c') as eri1:
        if not hasattr(eri1, 'shape'):
            raise NotImplementedError('Unsupported CDERI source for streamed nr_e2.')
        naux = int(eri1.shape[0])
        npair = int(eri1.shape[1])
        target_bytes = max(_int3c_mo_vjp_block_mb(), 1.0) * 1024.0**2
        row_bytes = max(npair, 1) * numpy.dtype(numpy.float64).itemsize
        blksize = max(1, min(naux, int(target_bytes // row_bytes)))

        out = None
        for p0 in range(0, naux, blksize):
            p1 = min(p0 + blksize, naux)
            cderi = numpy.asarray(eri1[p0:p1])
            block = pyscf_ao2mo.nr_e2(
                cderi, mo_coeff, orbs_slice, aosym=aosym, mosym='s1'
            )
            if out is None:
                out = numpy.empty((naux, block.shape[1]), dtype=block.dtype)
            out[p0:p1] = block
    if out is None:
        return numpy.empty((0, 0), dtype=mo_coeff.dtype)
    return out


def _int3c_mo_deriv_coords_vjp(mol, auxmol, mo_coeff, z,
                               orbs_slice, int3c='int3c2e',
                               aosym='s2ij'):
    if aosym not in ('s2', 's2ij'):
        raise NotImplementedError(f'Only packed s2 CDERI is supported, got {aosym}.')
    if mo_coeff.shape[0] != mol.nao:
        raise NotImplementedError(
            'Direct int3c-MO VJP currently requires the full AO basis.'
        )

    nao = mol.nao
    naux = auxmol.nao
    nbas = mol.nbas
    nauxbas = auxmol.nbas
    npair = nao * (nao + 1) // 2
    k0, k1, l0, l1 = orbs_slice
    kc = k1 - k0
    lc = l1 - l0
    kl_count = kc * lc
    mo_k = numpy.asarray(mo_coeff[:, k0:k1], order='F')
    mo_l = numpy.asarray(mo_coeff[:, l0:l1], order='F')
    z = numpy.asarray(z).reshape(naux, kc, lc)
    z_flat = z.reshape(naux, kl_count)

    target_words = _int3c_mo_vjp_block_mb() * 1024.0**2 / 8
    words_per_aux = (
        3 * nao * nao
        + 4 * nao * max(kc, 1)
        + 4 * nao * max(lc, 1)
        + 3 * npair
        + 3 * kl_count
    )
    blksize = max(1, min(naux, int(target_words // max(words_per_aux, 1))))
    aux_loc = auxmol.ao_loc
    aux_ranges = balance_partition(aux_loc, blksize)

    int3c_ip1 = int3c.replace('int3c2e', 'int3c2e_ip1')
    int3c_ip2 = int3c.replace('int3c2e', 'int3c2e_ip2')
    mol_ao_bar = numpy.zeros((nao, 3), dtype=z.dtype)
    aux_ao_bar = numpy.zeros((naux, 3), dtype=z.dtype)

    for shl0, shl1, _ in aux_ranges:
        p0, p1 = aux_loc[shl0], aux_loc[shl1]
        shls_slice = (0, nbas, 0, nbas, nbas + shl0, nbas + shl1)
        z_blk = z[p0:p1]

        # AO-center derivative.  Keep the MO transform factorized as
        # d(mu nu|P) C_mu,k C_nu,l, and accumulate AO-center rows before
        # summing to atoms.
        ints = _int3c_cross_opt.int3c_cross(
            mol, auxmol, intor=int3c_ip1, comp=3, aosym='s1',
            shls_slice=shls_slice,
        )
        ints = numpy.asarray(ints)
        intbuf = pyscf_lib.einsum('xuvp,vl->xupl', ints, mo_l)
        dm2buf = pyscf_lib.einsum('uk,pkl->upl', mo_k, z_blk)
        mol_ao_bar -= numpy.einsum('upl,xupl->ux', dm2buf, intbuf)
        intbuf = dm2buf = None

        intbuf = pyscf_lib.einsum('xuvp,vk->xupk', ints, mo_k)
        dm2buf = pyscf_lib.einsum('ul,pkl->upk', mo_l, z_blk)
        mol_ao_bar -= numpy.einsum('upk,xupk->ux', dm2buf, intbuf)
        ints = intbuf = dm2buf = None

        # Auxiliary-center derivative.  The derivative three-center rows are
        # batched as 3*naux_block ordinary packed AO rows, transformed by the
        # same nr_e2 C kernel as the forward AO2MO, then dotted with Z.
        ints = _int3c_cross_opt.int3c_cross(
            mol, auxmol, intor=int3c_ip2, comp=3, aosym='s2ij',
            shls_slice=shls_slice,
        )
        ints = numpy.ascontiguousarray(
            numpy.asarray(ints).transpose(0, 2, 1).reshape(3 * (p1 - p0), npair)
        )
        ints_mo = pyscf_ao2mo.nr_e2(
            ints, mo_coeff, orbs_slice, aosym='s2', mosym='s1'
        ).reshape(3, p1 - p0, kl_count)
        aux_ao_bar[p0:p1] -= numpy.einsum(
            'pm,xpm->px', z_flat[p0:p1], ints_mo
        )
        ints = ints_mo = None

    mol_bar = _coords_tree_like(mol, _ao_to_atom_coords_bar(mol, mol_ao_bar))
    auxmol_bar = _coords_tree_like(
        auxmol, _ao_to_atom_coords_bar(auxmol, aux_ao_bar)
    )
    return mol_bar, auxmol_bar


def cholesky_eri_vjp_from_mo_coeff_ybar(mol, auxmol, cderi_source,
                                        mo_coeff, ybar, orbs_slice,
                                        max_memory, int3c=None, int2c=None,
                                        aosym='s2ij'):
    """Back-propagate through CDERI using MO-basis int3c derivatives.

    This avoids materializing the large packed AO-pair cotangent
    ``cderi_bar[naux, nao_pair]``.  It is currently limited to the full AO
    basis; localized AO domains should keep using the block-function fallback.
    """
    t_total = time.perf_counter()
    _profile_msg(
        'cholesky_eri_vjp_from_mo_coeff_ybar start '
        f'orbs_slice={orbs_slice}'
    )
    if aosym not in ('s2', 's2ij'):
        raise NotImplementedError(f'Only packed s2 CDERI is supported, got {aosym}.')
    if cderi_source is None:
        raise NotImplementedError('Missing CDERI source for cholesky_eri VJP.')
    mo_coeff = numpy.asarray(jax.device_get(mo_coeff))
    if mo_coeff.shape[0] != mol.nao:
        raise NotImplementedError(
            'Direct int3c-MO VJP currently requires the full AO basis.'
        )

    if int3c is None:
        int3c = mol._add_suffix('int3c2e')
    if int2c is None:
        int2c = mol._add_suffix('int2c2e')

    ybar = numpy.asarray(jax.device_get(ybar))
    naux = auxmol.nao
    k0, k1, l0, l1 = orbs_slice
    kl_count = (k1 - k0) * (l1 - l0)
    ybar = ybar.reshape(naux, kl_count)

    with df_addons.load(cderi_source, 'j3c') as feri:
        if not hasattr(feri, 'shape'):
            raise NotImplementedError('Unsupported CDERI source for cholesky_eri VJP.')
        nao_pair = mol.nao * (mol.nao + 1) // 2
        if int(feri.shape[0]) != naux or int(feri.shape[1]) != nao_pair:
            raise NotImplementedError('CDERI source shape does not match mol/auxmol.')

    t = time.perf_counter()
    j2c = auxmol.intor(int2c, hermi=1)
    j2c_np = numpy.asarray(jax.device_get(j2c))
    try:
        low = scipy.linalg.cholesky(j2c_np, lower=True, check_finite=False)
    except scipy.linalg.LinAlgError as err:
        raise NotImplementedError('2c metric Cholesky fallback is not implemented.') from err
    if low.shape[0] != low.shape[1] or low.shape[0] != naux:
        raise NotImplementedError(
            'Linear-dependent auxiliary metric fallback is not implemented.'
        )

    z = scipy.linalg.solve_triangular(
        low.T, ybar, lower=False, check_finite=False
    )
    _profile_msg(
        'cholesky_eri_vjp_from_mo_coeff_ybar metric/z setup done '
        f'{time.perf_counter() - t:.2f} s'
    )

    t = time.perf_counter()
    y = _stream_nr_e2_from_cderi_source(
        cderi_source, mo_coeff, orbs_slice, max_memory, aosym='s2'
    )
    low_bar = -numpy.dot(z, y.T)
    _profile_msg(
        'cholesky_eri_vjp_from_mo_coeff_ybar stream y/low_bar done '
        f'{time.perf_counter() - t:.2f} s'
    )

    t = time.perf_counter()
    mol_bar, auxmol_bar = _int3c_mo_deriv_coords_vjp(
        mol, auxmol, mo_coeff, z, orbs_slice, int3c=int3c, aosym=aosym
    )
    _profile_msg(
        'cholesky_eri_vjp_from_mo_coeff_ybar int3c_mo_deriv done '
        f'{time.perf_counter() - t:.2f} s'
    )

    def metric_cholesky(auxmol_):
        return jax_scipy.linalg.cholesky(auxmol_.intor(int2c, hermi=1), lower=True)

    t = time.perf_counter()
    _, chol_pullback = jax.vjp(metric_cholesky, auxmol)
    aux_metric_bar = chol_pullback(np.asarray(numpy.tril(low_bar)))[0]
    auxmol_bar = _tree_add(auxmol_bar, aux_metric_bar)
    _profile_msg(
        'cholesky_eri_vjp_from_mo_coeff_ybar metric cholesky pullback done '
        f'{time.perf_counter() - t:.2f} s'
    )
    _profile_msg(
        'cholesky_eri_vjp_from_mo_coeff_ybar done '
        f'{time.perf_counter() - t_total:.2f} s'
    )
    return mol_bar, auxmol_bar
