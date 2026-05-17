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

import numpy
import scipy.linalg
import jax
from jax import scipy as jax_scipy
from jax.tree_util import tree_map
from pyscf.df.outcore import _guess_shell_ranges

from pyscfad import numpy as np
from pyscfad.ao2mo import _ao2mo
from pyscfad.df import _int3c_cross_opt
from pyscfad.df import addons as df_addons


_INT3C_VJP_TARGET_MB = 256.0


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
    import os
    try:
        return max(float(os.environ.get('PYSCFAD_DF_NR_E2_VJP_BLOCK_MB', 256.0)), 1.0)
    except ValueError:
        return 256.0


def nr_e2_mo_coeff_vjp_from_cderi_source(cderi_source, mo_coeff, ybar,
                                         orbs_slice, aosym='s2',
                                         mosym='s1', pair_idx=None):
    """Return only the MO-coefficient cotangent, streaming CDERI rows."""
    if cderi_source is None:
        raise NotImplementedError('Missing CDERI source for nr_e2 VJP.')
    if aosym not in ('s2', 's2ij'):
        raise NotImplementedError(f'Only packed s2 CDERI is supported, got {aosym}.')

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

        target_bytes = _nr_e2_vjp_block_mb() * 1024.0**2
        row_bytes = max(npair + ybar.shape[1], 1) * numpy.dtype(numpy.float64).itemsize
        blksize = max(1, min(naux, int(target_bytes // row_bytes)))

        for p0 in range(0, naux, blksize):
            p1 = min(p0 + blksize, naux)
            if pair_idx is None:
                cderi = numpy.asarray(eri1[p0:p1])
            else:
                cderi = numpy.asarray(eri1[p0:p1, pair_idx])

            def fn(cderi_, mo_coeff_):
                return _ao2mo.nr_e2(cderi_, mo_coeff_, orbs_slice,
                                    aosym=aosym, mosym=mosym)

            _, pullback = jax.vjp(fn, np.asarray(cderi), mo_coeff)
            _, mo_coeff_bar_blk = pullback(np.asarray(ybar[p0:p1]))
            mo_coeff_bar = _tree_add(mo_coeff_bar, mo_coeff_bar_blk)
            cderi = mo_coeff_bar_blk = None
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
    out = numpy.einsum('Lij,pi,pj->Lp', ybar, mok_rows, mol_cols,
                       optimize=True)

    offdiag = rows != cols
    if numpy.any(offdiag):
        mok_cols = mo[cols[offdiag], k0:k1]
        mol_rows = mo[rows[offdiag], l0:l1]
        out[:, offdiag] += numpy.einsum(
            'Lij,pi,pj->Lp', ybar, mok_cols, mol_rows, optimize=True
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

    max_words = max(max_memory, 0) * 1e6 / 8 - low.size - naux * nao_pair
    mem_buflen = max(int(max_words / max(naoaux, 1) / 2), 8)
    target_buflen = max(int(_INT3C_VJP_TARGET_MB * 1e6 / 8 / max(naoaux, 1) / 3), 8)
    buflen = min(nao_pair, mem_buflen, target_buflen)
    shranges = _guess_shell_ranges(mol, buflen, aosym)

    mol_bar = None
    auxmol_bar = None
    low_bar = numpy.zeros_like(low)

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

    if p1 != nao_pair:
        raise RuntimeError('CDERI VJP shell ranges did not cover all AO pairs.')

    def metric_cholesky(auxmol_):
        return jax_scipy.linalg.cholesky(auxmol_.intor(int2c, hermi=1), lower=True)

    _, chol_pullback = jax.vjp(metric_cholesky, auxmol)
    aux_metric_bar = chol_pullback(np.asarray(numpy.tril(low_bar)))[0]
    auxmol_bar = _tree_add(auxmol_bar, aux_metric_bar)
    return mol_bar, auxmol_bar


def cholesky_eri_vjp_from_cderi_block_fn(mol, auxmol, cderi_source,
                                         cderi_bar_block_fn, max_memory,
                                         int3c=None, int2c=None,
                                         aosym='s2ij'):
    """Back-propagate through Cholesky CDERI from AO-pair cotangent blocks."""
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

    max_words = max(max_memory, 0) * 1e6 / 8 - low.size
    mem_buflen = max(int(max_words / max(naoaux, 1) / 3), 8)
    target_buflen = max(int(_INT3C_VJP_TARGET_MB * 1e6 / 8 / max(naoaux, 1) / 3), 8)
    buflen = min(nao_pair, mem_buflen, target_buflen)
    shranges = _guess_shell_ranges(mol, buflen, aosym)

    mol_bar = None
    auxmol_bar = None
    low_bar = numpy.zeros_like(low)

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

    if p1 != nao_pair:
        raise RuntimeError('CDERI VJP shell ranges did not cover all AO pairs.')

    def metric_cholesky(auxmol_):
        return jax_scipy.linalg.cholesky(auxmol_.intor(int2c, hermi=1), lower=True)

    _, chol_pullback = jax.vjp(metric_cholesky, auxmol)
    aux_metric_bar = chol_pullback(np.asarray(numpy.tril(low_bar)))[0]
    auxmol_bar = _tree_add(auxmol_bar, aux_metric_bar)
    return mol_bar, auxmol_bar
