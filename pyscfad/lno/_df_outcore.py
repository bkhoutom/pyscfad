# Copyright 2023-2026 The PySCFAD Authors
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

"""Blocked transformations and reverse passes for an existing CDERI source."""

import os
from functools import partial, reduce
from math import isqrt
import numpy
import jax
from pyscfad import numpy as np
from pyscfad.ao2mo import _ao2mo
from pyscfad.df import addons as df_addons
from pyscfad.df import incore as df_incore
from pyscfad.df import _cderi_vjp
from ._profile import (
    _vjp_progress_section,
)


def _global_pair_indices_for_local_ao(ao_idx, nao):
    ao_idx = numpy.asarray(ao_idx, dtype=numpy.int64).ravel()
    rows, cols = numpy.tril_indices(ao_idx.size)
    grows = ao_idx[rows]
    gcols = ao_idx[cols]
    if numpy.any(grows < gcols):
        raise RuntimeError('Local AO indices must be sorted for packed s2 mapping.')
    idx = grows * (grows + 1) // 2 + gcols
    npair = nao * (nao + 1) // 2
    if idx.size and (idx.min() < 0 or idx.max() >= npair):
        raise RuntimeError('Local AO pair index exceeds global packed CDERI size.')
    return idx


def _embed_local_mo_coeff_from_pair_idx(mo_coeff, pair_idx, nao):
    """Embed a complete local packed-pair domain into the global AO basis."""
    coeff = numpy.asarray(jax.device_get(mo_coeff))
    pair_idx = numpy.asarray(pair_idx, dtype=numpy.int64)
    if coeff.ndim != 2 or pair_idx.ndim != 1:
        raise ValueError('Expected a coefficient matrix and a one-dimensional pair map.')
    nlocal = coeff.shape[0]
    if nlocal > nao or pair_idx.size != nlocal * (nlocal + 1) // 2:
        raise ValueError('pair_idx must contain the complete local AO-pair space.')
    if pair_idx.size and (pair_idx.min() < 0 or pair_idx.max() >= nao * (nao + 1) // 2):
        raise ValueError('pair_idx is outside the global packed-pair space.')

    rows = numpy.arange(nlocal, dtype=numpy.int64)
    diagonal = pair_idx[rows * (rows + 3) // 2]
    ao_idx = numpy.asarray([(isqrt(8 * int(p) + 1) - 1) // 2 for p in diagonal],
                           dtype=numpy.int64)
    if numpy.any(numpy.diff(ao_idx) <= 0):
        raise ValueError('Local AO labels must be strictly increasing.')
    # Validate all pairs, not just the diagonal. Row-wise comparison avoids
    # allocating additional O(nlocal**2) index arrays for large AO domains.
    for i, ao in enumerate(ao_idx):
        p0, p1 = i * (i + 1) // 2, (i + 1) * (i + 2) // 2
        if not numpy.array_equal(pair_idx[p0:p1], ao * (ao + 1) // 2 + ao_idx[:i + 1]):
            raise ValueError('pair_idx is not an induced local AO-domain map.')

    coeff_full = numpy.zeros((nao, coeff.shape[1]), dtype=coeff.dtype)
    coeff_full[ao_idx] = coeff
    return coeff_full


def _outcore_nr_e2_block_mb():
    try:
        return max(float(os.environ.get('PYSCFAD_LNO_OUTCORE_NR_E2_BLOCK_MB', 256.0)), 1.0)
    except ValueError:
        return 256.0


def _outcore_nr_e2_from_source_blocked(cderi_source, mo_coeff, orbs_slice,
                                       aosym='s2', mosym='s1', pair_idx=None):
    """Transform out-of-core CDERI without materializing the full CDERI block."""
    out = None
    with df_addons.load(cderi_source, 'j3c') as eri1:
        if not hasattr(eri1, 'shape'):
            raise NotImplementedError('Unsupported CDERI source for blocked nr_e2.')

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

        target_bytes = _outcore_nr_e2_block_mb() * 1024.0**2
        row_bytes = max(npair, 1) * numpy.dtype(numpy.float64).itemsize
        blksize = max(1, min(naux, int(target_bytes // row_bytes)))

        for p0 in range(0, naux, blksize):
            p1 = min(p0 + blksize, naux)
            if pair_idx is None:
                cderi = numpy.asarray(eri1[p0:p1])
            else:
                cderi = numpy.asarray(eri1[p0:p1, pair_idx])
            block = numpy.asarray(
                _ao2mo.nr_e2(
                    cderi, mo_coeff, orbs_slice, aosym=aosym, mosym=mosym
                )
            )
            if out is None:
                out = numpy.empty((naux,) + block.shape[1:], dtype=block.dtype)
            out[p0:p1] = block
            cderi = block = None
    if out is None:
        return np.empty((0,), dtype=mo_coeff.dtype)
    return np.asarray(out)


def _select_sorted_pair_positions(pair_idx, p0, p1):
    pair_idx = numpy.asarray(pair_idx, dtype=numpy.int64)
    if pair_idx.size == 0:
        return numpy.zeros(0, dtype=numpy.int64)
    if pair_idx.size == 1 or numpy.all(numpy.diff(pair_idx) >= 0):
        i0 = numpy.searchsorted(pair_idx, p0, side='left')
        i1 = numpy.searchsorted(pair_idx, p1, side='left')
        return numpy.arange(i0, i1, dtype=numpy.int64)
    return numpy.nonzero((pair_idx >= p0) & (pair_idx < p1))[0]


def _nr_e2_global_cderi_bar_block(mo_coeff, ybar, orbs_slice, pair_positions,
                                  p0, p1):
    block = numpy.zeros((ybar.shape[0], p1 - p0), dtype=numpy.asarray(ybar).dtype)
    pair_positions = numpy.asarray(pair_positions, dtype=numpy.int64)
    if pair_positions.size == 0:
        return block
    cderi_bar = _cderi_vjp.nr_e2_cderi_bar_packed_block(
        mo_coeff, ybar, orbs_slice, pair_positions
    )
    block[:, pair_positions - p0] = cderi_bar
    return block


def _nr_e2_local_cderi_bar_block(mo_coeff, ybar, orbs_slice, pair_idx, p0, p1):
    block = numpy.zeros((ybar.shape[0], p1 - p0), dtype=numpy.asarray(ybar).dtype)
    local_positions = _select_sorted_pair_positions(pair_idx, p0, p1)
    if local_positions.size == 0:
        return block
    cderi_bar = _cderi_vjp.nr_e2_cderi_bar_packed_block(
        mo_coeff, ybar, orbs_slice, local_positions
    )
    global_positions = numpy.asarray(pair_idx, dtype=numpy.int64)[local_positions]
    block[:, global_positions - p0] = cderi_bar
    return block


def _nr_e2_local_cderi_bar_disk_block(cderi_bar_h5, pair_idx, p0, p1):
    """Read local packed cotangents into one global AO-pair block."""
    pair_idx = numpy.asarray(pair_idx, dtype=numpy.int64).ravel()
    naux = int(cderi_bar_h5.shape[0])
    block = numpy.zeros((naux, p1 - p0), dtype=cderi_bar_h5.dtype)
    if pair_idx.size == 0:
        return block
    if int(cderi_bar_h5.shape[1]) != pair_idx.size:
        raise RuntimeError(
            'Disk CDERI cotangent pair dimension does not match local pair map.'
        )
    if pair_idx.size > 1 and numpy.any(numpy.diff(pair_idx) <= 0):
        raise RuntimeError('Local-to-global AO-pair map must be strictly increasing.')

    i0 = int(numpy.searchsorted(pair_idx, p0, side='left'))
    i1 = int(numpy.searchsorted(pair_idx, p1, side='left'))
    if i0 == i1:
        return block
    global_positions = pair_idx[i0:i1]
    block[:, global_positions - p0] = numpy.asarray(cderi_bar_h5[:, i0:i1])
    return block


def _is_full_global_pair_idx(pair_idx, nao):
    pair_idx = numpy.asarray(pair_idx, dtype=numpy.int64).ravel()
    npair = nao * (nao + 1) // 2
    return (
        pair_idx.size == npair
        and (npair == 0 or (
            pair_idx[0] == 0
            and pair_idx[-1] == npair - 1
            and numpy.all(numpy.diff(pair_idx) == 1)
        ))
    )


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7))
def _outcore_local_nr_e2_from_global_cderi(mol, auxmol, mo_coeff, cderi_source,
                                           max_memory, orbs_slice, aosym,
                                           pair_idx):
    del mol, auxmol, max_memory
    if aosym not in ('s2', 's2ij'):
        raise NotImplementedError
    pair_idx = numpy.asarray(pair_idx, dtype=numpy.int64)
    return _outcore_nr_e2_from_source_blocked(
        cderi_source, mo_coeff, orbs_slice, aosym='s2', pair_idx=pair_idx
    )


def _outcore_local_nr_e2_from_global_cderi_fwd(mol, auxmol, mo_coeff,
                                               cderi_source, max_memory,
                                               orbs_slice, aosym, pair_idx):
    out = _outcore_local_nr_e2_from_global_cderi(
        mol, auxmol, mo_coeff, cderi_source, max_memory, orbs_slice, aosym,
        pair_idx,
    )
    return out, (mol, auxmol, mo_coeff)


def _outcore_local_nr_e2_from_global_cderi_bwd(cderi_source, max_memory,
                                               orbs_slice, aosym, pair_idx,
                                               res, ybar):
    mol, auxmol, mo_coeff = res
    pair_idx = numpy.asarray(pair_idx, dtype=numpy.int64)
    full_pair_idx = _is_full_global_pair_idx(pair_idx, mol.nao)
    coeff_full = None if full_pair_idx else _embed_local_mo_coeff_from_pair_idx(
        mo_coeff, pair_idx, mol.nao
    )
    if ybar.size == 0:
        return jax.tree_util.tree_map(np.zeros_like, (mol, auxmol, mo_coeff))
    try:
        with _vjp_progress_section('fragment DF AO2MO MO-coeff backward'):
            mo_coeff_bar = _cderi_vjp.nr_e2_mo_coeff_vjp_from_cderi_source(
                cderi_source, mo_coeff, ybar, orbs_slice, aosym='s2',
                pair_idx=pair_idx, max_memory=max_memory,
            )

        ybar_np = numpy.asarray(jax.device_get(ybar))
        if full_pair_idx:
            try:
                with _vjp_progress_section('fragment DF integral derivative backward'):
                    mol_bar, auxmol_bar = _cderi_vjp.cholesky_eri_vjp_from_mo_coeff_ybar(
                        mol,
                        auxmol,
                        cderi_source,
                        mo_coeff,
                        ybar_np,
                        orbs_slice,
                        max(max_memory, 4096),
                        int3c=mol._add_suffix('int3c2e'),
                        int2c=mol._add_suffix('int2c2e'),
                        aosym='s2ij',
                    )
            except NotImplementedError:
                mol_bar = auxmol_bar = None
            if mol_bar is not None:
                return mol_bar, auxmol_bar, mo_coeff_bar
            # Preserve the established full-AO fallback.  The disk path below
            # is specifically for a local AO domain and assumes its compact
            # pair dimension maps into a larger global packed dimension.
            with _vjp_progress_section('fragment DF integral derivative backward'):
                mol_bar, auxmol_bar = _cderi_vjp.cholesky_eri_vjp_from_cderi_block_fn(
                    mol,
                    auxmol,
                    cderi_source,
                    lambda p0, p1: _nr_e2_local_cderi_bar_block(
                        mo_coeff, ybar_np, orbs_slice, pair_idx, p0, p1
                    ),
                    max(max_memory, 4096),
                    int3c=mol._add_suffix('int3c2e'),
                    int2c=mol._add_suffix('int2c2e'),
                    aosym='s2ij',
                )
            return mol_bar, auxmol_bar, mo_coeff_bar
        coordinate_only = all(
            obj.coords is not None and all(
                getattr(obj, name, None) is None for name in ('exp', 'ctr_coeff', 'r0')
            ) for obj in (mol, auxmol)
        )
        if coordinate_only and coeff_full.dtype == numpy.float64 and ybar_np.dtype == numpy.float64:
            _cderi_vjp._profile_msg(
                'partial-domain strategy=embedded_mo '
                f'local_nao={mo_coeff.shape[0]} global_nao={mol.nao} '
                f'naux={auxmol.nao} orbs_slice={orbs_slice}'
            )
            try:
                with _vjp_progress_section('fragment DF integral derivative backward'):
                    mol_bar, auxmol_bar = _cderi_vjp.cholesky_eri_vjp_from_mo_coeff_ybar(
                        mol, auxmol, cderi_source, coeff_full, ybar_np, orbs_slice,
                        max(max_memory, 4096), int3c=mol._add_suffix('int3c2e'),
                        int2c=mol._add_suffix('int2c2e'), aosym='s2ij',
                    )
            except NotImplementedError as err:
                _cderi_vjp._profile_msg(f'partial-domain strategy=disk fallback: {err}')
            else:
                return mol_bar, auxmol_bar, mo_coeff_bar
        else:
            _cderi_vjp._profile_msg(
                'partial-domain strategy=disk fallback: '
                'embedded MO derivative requires real float64 coordinate-only inputs'
            )
        del coeff_full
        with _vjp_progress_section('fragment DF integral derivative backward'):
            # Build Bbar by auxiliary slabs with the dense two-GEMM native
            # kernel.  The tiled HDF layout is then read by AO-pair slabs,
            # which is the access order required by the L^{-T} solve below.
            with _cderi_vjp.nr_e2_cderi_bar_packed_disk(
                    mo_coeff, ybar_np, orbs_slice) as cderi_bar_h5:
                mol_bar, auxmol_bar = _cderi_vjp.cholesky_eri_vjp_from_cderi_block_fn(
                    mol,
                    auxmol,
                    cderi_source,
                    lambda p0, p1: _nr_e2_local_cderi_bar_disk_block(
                        cderi_bar_h5, pair_idx, p0, p1
                    ),
                    max(max_memory, 4096),
                    int3c=mol._add_suffix('int3c2e'),
                    int2c=mol._add_suffix('int2c2e'),
                    aosym='s2ij',
                )
        return mol_bar, auxmol_bar, mo_coeff_bar
    except NotImplementedError:
        pass

    def full_fn(mol_, auxmol_, mo_coeff_):
        cderi = df_incore.cholesky_eri(
            mol_,
            auxmol=auxmol_,
            int3c=mol_._add_suffix('int3c2e'),
            int2c=mol_._add_suffix('int2c2e'),
            max_memory=max(max_memory, 4096),
            verbose=0,
        )
        cderi = cderi[:, pair_idx]
        return _ao2mo.nr_e2(cderi, mo_coeff_, orbs_slice, aosym='s2')

    _, pullback = jax.vjp(full_fn, mol, auxmol, mo_coeff)
    return pullback(ybar)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def _outcore_nr_e2(mol, auxmol, mo_coeff, cderi_source, max_memory,
                   orbs_slice, aosym):
    del mol, auxmol, max_memory
    return _outcore_nr_e2_from_source_blocked(
        cderi_source, mo_coeff, orbs_slice, aosym=aosym
    )


def _outcore_nr_e2_fwd(mol, auxmol, mo_coeff, cderi_source, max_memory,
                       orbs_slice, aosym):
    out = _outcore_nr_e2(mol, auxmol, mo_coeff, cderi_source, max_memory,
                         orbs_slice, aosym)
    return out, (mol, auxmol, mo_coeff)


def _outcore_nr_e2_bwd(cderi_source, max_memory, orbs_slice, aosym, res, ybar):
    mol, auxmol, mo_coeff = res
    try:
        with _vjp_progress_section('global DF AO2MO MO-coeff backward'):
            mo_coeff_bar = _cderi_vjp.nr_e2_mo_coeff_vjp_from_cderi_source(
                cderi_source, mo_coeff, ybar, orbs_slice, aosym=aosym,
                max_memory=max_memory,
            )

        ybar_np = numpy.asarray(jax.device_get(ybar))
        try:
            with _vjp_progress_section('global DF integral derivative backward'):
                mol_bar, auxmol_bar = _cderi_vjp.cholesky_eri_vjp_from_mo_coeff_ybar(
                    mol,
                    auxmol,
                    cderi_source,
                    mo_coeff,
                    ybar_np,
                    orbs_slice,
                    max(max_memory, 4096),
                    int3c=mol._add_suffix('int3c2e'),
                    int2c=mol._add_suffix('int2c2e'),
                    aosym='s2ij',
                )
        except NotImplementedError:
            with _vjp_progress_section('global DF integral derivative backward'):
                mol_bar, auxmol_bar = _cderi_vjp.cholesky_eri_vjp_from_cderi_block_fn(
                    mol,
                    auxmol,
                    cderi_source,
                    lambda p0, p1: _nr_e2_global_cderi_bar_block(
                        mo_coeff, ybar_np, orbs_slice,
                        numpy.arange(p0, p1, dtype=numpy.int64), p0, p1
                    ),
                    max(max_memory, 4096),
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
        return _ao2mo.nr_e2(cderi, mo_coeff_, orbs_slice, aosym=aosym)

    _, pullback = jax.vjp(fn, mol, auxmol, mo_coeff)
    return pullback(ybar)


_outcore_local_nr_e2_from_global_cderi.defvjp(
    _outcore_local_nr_e2_from_global_cderi_fwd,
    _outcore_local_nr_e2_from_global_cderi_bwd,
)

_outcore_nr_e2.defvjp(_outcore_nr_e2_fwd, _outcore_nr_e2_bwd)
