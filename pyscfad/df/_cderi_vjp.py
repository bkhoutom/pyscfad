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

from contextlib import contextmanager
from functools import lru_cache
import contextvars
import ctypes
import json
import os
import tempfile
import time

import h5py
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
_CDERI_BAR_AUX_BLOCK_MB = 256.0
_CDERI_BAR_HDF_CHUNK_MB = 8.0
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

_NR_E2_CDERI_BAR_AUX_NATIVE = (
    libao2mo_vjp is not None
    and hasattr(libao2mo_vjp, 'AO2MOnr_e2_cderi_bar_pack_aux_block')
)
if _NR_E2_CDERI_BAR_AUX_NATIVE:
    libao2mo_vjp.AO2MOnr_e2_cderi_bar_pack_aux_block.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    libao2mo_vjp.AO2MOnr_e2_cderi_bar_pack_aux_block.restype = ctypes.c_int


def _profile_enabled():
    value = os.environ.get('PYSCFAD_PROFILE_BACKWARD_PHASES')
    return value is not None and value.strip().lower() in ('1', 'true', 'yes', 'on')


def _profile_msg(msg):
    if _profile_enabled():
        print(f'[profile][df.cderi_vjp] {msg}', flush=True)


_DF_VJP_DETAILED_TIMING = contextvars.ContextVar(
    'pyscfad_df_vjp_detailed_timing', default=None
)

_DF_VJP_TIMING_STAGES = (
    ('metric_integral_initial_cholesky', 'complete_response', False, None),
    ('ao_projection_total', 'outer_block_loop', False, None),
    ('ao_projection_kernel', 'outer_block_loop', True,
     'ao_projection_total'),
    ('zero_check', 'outer_block_loop', False, None),
    ('cderi_read', 'outer_block_loop', False, None),
    ('triangular_solve', 'outer_block_loop', False, None),
    ('low_bar_gemm', 'outer_block_loop', False, None),
    ('int3c_primal_vjp_setup', 'outer_block_loop', False, None),
    ('int3c_pullback', 'outer_block_loop', False, None),
    ('gradient_accumulation', 'outer_block_loop', False, None),
    ('unattributed_overhead', 'outer_block_loop', False, None),
    ('outer_block_total', 'outer_block_loop', False, None),
    ('metric_vjp_primal_creation', 'complete_response', False, None),
    ('metric_cholesky_pullback', 'complete_response', False, None),
    ('complete_response', 'complete_response', False, None),
)

_DF_VJP_OUTER_CHILDREN = (
    'ao_projection_total',
    'zero_check',
    'cderi_read',
    'triangular_solve',
    'low_bar_gemm',
    'int3c_primal_vjp_setup',
    'int3c_pullback',
    'gradient_accumulation',
)


def _new_df_vjp_detailed_timing():
    """Allocate detailed timing state only for an enabled profile run."""
    return {
        'samples': {name: [] for name, _, _, _ in _DF_VJP_TIMING_STAGES},
        'counters': {
            'candidate_shell_blocks': 0,
            'processed_nonzero_blocks': 0,
            'skipped_zero_blocks': 0,
            'global_pair_positions_encountered': 0,
            'local_ao_pair_positions_projected': 0,
            'projection_kernel_calls': 0,
            'projection_kernel_pair_positions': 0,
        },
        'projection': {
            'primary': {'calls': 0, 'pair_positions': 0},
            'exchanged_offdiagonal': {'calls': 0, 'pair_positions': 0},
            'native': {'calls': 0, 'pair_positions': 0},
            'numpy': {'calls': 0, 'pair_positions': 0},
        },
        'projection_order': {
            'selection_policy': 'smaller_trailing_mo_dimension',
            'selected_calls': {'as_given': 0, 'transposed': 0},
            'layouts': {},
        },
        'active_block_index': None,
    }


def _detailed_timing_start():
    return time.perf_counter(), time.process_time()


def _detailed_timing_elapsed(start):
    wall = max(time.perf_counter() - start[0], 0.0)
    cpu = max(time.process_time() - start[1], 0.0)
    return wall, cpu


def _record_detailed_timing(detail, stage, elapsed, block_index=None):
    detail['samples'][stage].append(
        (float(elapsed[0]), float(elapsed[1]), block_index)
    )


def _record_projection_kernel(detail, elapsed, role, backend, pair_positions):
    block_index = detail['active_block_index']
    _record_detailed_timing(
        detail, 'ao_projection_kernel', elapsed, block_index=block_index
    )
    pair_positions = int(pair_positions)
    detail['counters']['projection_kernel_calls'] += 1
    detail['counters']['projection_kernel_pair_positions'] += pair_positions
    detail['projection'][role]['calls'] += 1
    detail['projection'][role]['pair_positions'] += pair_positions
    detail['projection'][backend]['calls'] += 1
    detail['projection'][backend]['pair_positions'] += pair_positions


def _finish_outer_block_timing(detail, start, child_elapsed, block_index):
    outer_elapsed = _detailed_timing_elapsed(start)
    _record_detailed_timing(
        detail, 'outer_block_total', outer_elapsed, block_index=block_index
    )
    child_wall = sum(value[0] for value in child_elapsed.values())
    child_cpu = sum(value[1] for value in child_elapsed.values())
    overhead = (
        outer_elapsed[0] - child_wall,
        outer_elapsed[1] - child_cpu,
    )
    _record_detailed_timing(
        detail, 'unattributed_overhead', overhead, block_index=block_index
    )


def _summarize_detailed_timing_samples(samples):
    if not samples:
        return {
            'call_count': 0,
            'wall_total_s': 0.0,
            'cpu_total_s': 0.0,
            'mean_wall_s': None,
            'median_wall_s': None,
            'p95_wall_s': None,
            'max_wall_s': None,
            'mean_cpu_s': None,
            'median_cpu_s': None,
            'p95_cpu_s': None,
            'max_cpu_s': None,
            'first_block_index': None,
            'first_block_wall_s': None,
            'first_block_cpu_s': None,
            'effective_cpu_cores': None,
        }

    wall = numpy.asarray([sample[0] for sample in samples], dtype=float)
    cpu = numpy.asarray([sample[1] for sample in samples], dtype=float)
    first_block_index = samples[0][2]
    if first_block_index is None:
        first_wall = float(wall[0])
        first_cpu = float(cpu[0])
    else:
        first_samples = [
            sample for sample in samples if sample[2] == first_block_index
        ]
        first_wall = float(sum(sample[0] for sample in first_samples))
        first_cpu = float(sum(sample[1] for sample in first_samples))
    wall_total = float(wall.sum())
    cpu_total = float(cpu.sum())
    return {
        'call_count': int(wall.size),
        'wall_total_s': wall_total,
        'cpu_total_s': cpu_total,
        'mean_wall_s': float(wall.mean()),
        'median_wall_s': float(numpy.median(wall)),
        'p95_wall_s': float(numpy.percentile(wall, 95)),
        'max_wall_s': float(wall.max()),
        'mean_cpu_s': float(cpu.mean()),
        'median_cpu_s': float(numpy.median(cpu)),
        'p95_cpu_s': float(numpy.percentile(cpu, 95)),
        'max_cpu_s': float(cpu.max()),
        'first_block_index': (
            None if first_block_index is None else int(first_block_index)
        ),
        'first_block_wall_s': first_wall,
        'first_block_cpu_s': first_cpu,
        'effective_cpu_cores': (
            cpu_total / wall_total if wall_total > 0.0 else None
        ),
    }


def _emit_df_vjp_detailed_timing(detail):
    outer_wall = sum(
        sample[0] for sample in detail['samples']['outer_block_total']
    )
    complete_wall = sum(
        sample[0] for sample in detail['samples']['complete_response']
    )
    denominator = {
        'outer_block_loop': float(outer_wall),
        'complete_response': float(complete_wall),
    }
    stages = {}
    for name, scope, nested, parent in _DF_VJP_TIMING_STAGES:
        summary = _summarize_detailed_timing_samples(detail['samples'][name])
        scope_wall = denominator[scope]
        summary.update({
            'scope': scope,
            'nested': nested,
            'parent': parent,
            'wall_percent_of_scope': (
                100.0 * summary['wall_total_s'] / scope_wall
                if scope_wall > 0.0 else 0.0
            ),
        })
        if nested:
            parent_wall = sum(
                sample[0] for sample in detail['samples'][parent]
            )
            summary['wall_percent_of_parent'] = (
                100.0 * summary['wall_total_s'] / parent_wall
                if parent_wall > 0.0 else 0.0
            )
        stages[name] = summary

    child_wall = sum(
        stages[name]['wall_total_s'] for name in _DF_VJP_OUTER_CHILDREN
    ) + stages['unattributed_overhead']['wall_total_s']
    child_cpu = sum(
        stages[name]['cpu_total_s'] for name in _DF_VJP_OUTER_CHILDREN
    ) + stages['unattributed_overhead']['cpu_total_s']
    payload = {
        'schema_version': 1,
        'scope': 'cholesky_eri_vjp_from_cderi_block_fn',
        'clock_notes': {
            'wall': 'time.perf_counter; synchronous CPU path; no explicit JAX sync',
            'cpu': 'time.process_time; CPU/wall is effective core utilization',
        },
        'denominator_wall_s': denominator,
        'counters': detail['counters'],
        'projection_kernel_breakdown': detail['projection'],
        'projection_contraction_order': detail['projection_order'],
        'counter_notes': {
            'local_ao_pair_positions_projected': (
                'sum of selected packed pair positions passed to the projection; '
                'exact local-position count for the LNO local block callback'
            ),
            'projection_kernel_pair_positions': (
                'primary plus exchanged-offdiagonal kernel positions'
            ),
        },
        'reconciliation': {
            'nested_ao_projection_kernel_excluded': True,
            'unattributed_overhead_is_signed': True,
            'outer_children_plus_overhead_wall_s': float(child_wall),
            'outer_children_plus_overhead_cpu_s': float(child_cpu),
            'outer_block_total_wall_s': stages['outer_block_total']['wall_total_s'],
            'outer_block_total_cpu_s': stages['outer_block_total']['cpu_total_s'],
            'wall_residual_s': float(
                stages['outer_block_total']['wall_total_s'] - child_wall
            ),
            'cpu_residual_s': float(
                stages['outer_block_total']['cpu_total_s'] - child_cpu
            ),
        },
        'stages': stages,
    }

    counters = detail['counters']
    print(
        '[df-vjp-detailed-timing] '
        f"candidate_blocks={counters['candidate_shell_blocks']} "
        f"processed_nonzero={counters['processed_nonzero_blocks']} "
        f"skipped_zero={counters['skipped_zero_blocks']} "
        f"global_pair_positions={counters['global_pair_positions_encountered']} "
        f"local_pair_positions={counters['local_ao_pair_positions_projected']} "
        f"projection_calls={counters['projection_kernel_calls']}",
        flush=True,
    )
    for name, _, nested, parent in _DF_VJP_TIMING_STAGES:
        stage = stages[name]
        nested_text = f' nested=true parent={parent}' if nested else ' nested=false'
        print(
            '[df-vjp-detailed-timing] '
            f"stage={name}{nested_text} calls={stage['call_count']} "
            f"wall_s={stage['wall_total_s']:.6f} "
            f"cpu_s={stage['cpu_total_s']:.6f} "
            f"wall_pct_{stage['scope']}={stage['wall_percent_of_scope']:.3f} "
            f"mean_wall_s={stage['mean_wall_s']} "
            f"median_wall_s={stage['median_wall_s']} "
            f"p95_wall_s={stage['p95_wall_s']} "
            f"max_wall_s={stage['max_wall_s']} "
            f"first_block_wall_s={stage['first_block_wall_s']} "
            f"effective_cpu_cores={stage['effective_cpu_cores']}",
            flush=True,
        )
    print(
        '[df-vjp-timing-json] '
        + json.dumps(payload, sort_keys=True, separators=(',', ':'),
                     allow_nan=False),
        flush=True,
    )


@lru_cache(maxsize=16)
def _tril_indices(nao):
    return numpy.tril_indices(nao)


def _tree_add(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return tree_map(lambda a, b: a + b, x, y)


def _zero_strict_upper_inplace(matrix):
    """Zero a square matrix's strict upper triangle without index arrays."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('matrix must be square')
    for row in range(matrix.shape[0] - 1):
        matrix[row, row + 1:] = 0
    return matrix


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


def _cderi_bar_aux_block_mb():
    value = os.environ.get('PYSCFAD_DF_CDERI_BAR_AUX_BLOCK_MB')
    if value is None:
        return _CDERI_BAR_AUX_BLOCK_MB
    try:
        return max(float(value), 1.0)
    except ValueError:
        return _CDERI_BAR_AUX_BLOCK_MB


def _cderi_bar_hdf_chunk_mb():
    value = os.environ.get('PYSCFAD_DF_CDERI_BAR_HDF_CHUNK_MB')
    if value is None:
        return _CDERI_BAR_HDF_CHUNK_MB
    try:
        return max(float(value), 1.0)
    except ValueError:
        return _CDERI_BAR_HDF_CHUNK_MB


def _select_ao_projection_order(ybar, mok_rows, mol_cols):
    """Contract the larger MO dimension first without changing the result."""
    if ybar.ndim != 3:
        raise ValueError(f'ybar must be rank 3, got shape {ybar.shape}.')
    _, kc, lc = ybar.shape
    if mok_rows.shape != (mol_cols.shape[0], kc):
        raise ValueError(
            f'mok_rows has shape {mok_rows.shape}; expected '
            f'{(mol_cols.shape[0], kc)}.'
        )
    if mol_cols.shape[1] != lc:
        raise ValueError(
            f'mol_cols has shape {mol_cols.shape}; expected second dimension {lc}.'
        )

    if lc > kc:
        return (
            ybar.transpose(0, 2, 1),
            mol_cols,
            mok_rows,
            'transposed',
            kc,
            lc,
        )
    return ybar, mok_rows, mol_cols, 'as_given', kc, lc


def _record_ao_projection_layout(selected, original_kc, original_lc,
                                 effective_kc, effective_lc, npos, blksize):
    detail = _DF_VJP_DETAILED_TIMING.get()
    if detail is None:
        return
    order_detail = detail['projection_order']
    order_detail['selected_calls'][selected] += 1
    key = (
        f'{original_kc}x{original_lc}->{effective_kc}x{effective_lc}:'
        f'{selected}:block={blksize}'
    )
    layout = order_detail['layouts'].setdefault(key, {
        'selection_policy': order_detail['selection_policy'],
        'selected_order': selected,
        'original_kc': int(original_kc),
        'original_lc': int(original_lc),
        'effective_kc': int(effective_kc),
        'effective_lc': int(effective_lc),
        'python_pair_block': int(blksize),
        'native_pair_block_per_thread_estimate': int(max(
            (blksize + max(pyscf_lib.num_threads(), 1) - 1)
            // max(pyscf_lib.num_threads(), 1),
            1,
        )),
        'calls': 0,
        'pair_positions': 0,
        'npos_min': int(npos),
        'npos_max': int(npos),
    })
    layout['calls'] += 1
    layout['pair_positions'] += int(npos)
    layout['npos_min'] = min(layout['npos_min'], int(npos))
    layout['npos_max'] = max(layout['npos_max'], int(npos))


def _int3c_mo_vjp_block_mb():
    """Memory target for direct MO-basis three-center derivative work."""
    value = os.environ.get('PYSCFAD_DF_INT3C_MO_VJP_BLOCK_MB')
    if value is None:
        return _INT3C_MO_VJP_BLOCK_MB
    try:
        return max(float(value), 1.0)
    except ValueError:
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
    (ybar, mok_rows, mol_cols, selected,
     original_kc, original_lc) = _select_ao_projection_order(
         ybar, mok_rows, mol_cols
     )
    naux, kc, lc = ybar.shape
    npos = mok_rows.shape[0]
    if npos == 0 or kc == 0 or lc == 0:
        return numpy.zeros((naux, npos), dtype=ybar.dtype)

    target_bytes = _cderi_bar_pair_block_mb() * 1e6
    itemsize = numpy.dtype(ybar.dtype).itemsize
    blksize = max(int(target_bytes / max(naux * lc * itemsize, 1)), 1)
    blksize = min(blksize, npos)
    _record_ao_projection_layout(
        selected, original_kc, original_lc,
        kc, lc, npos, blksize,
    )

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


def _use_native_cderi_bar_aux_block(ybar, mo_k, mo_l):
    return (
        _NR_E2_CDERI_BAR_AUX_NATIVE
        and ybar.dtype == numpy.float64
        and mo_k.dtype == numpy.float64
        and mo_l.dtype == numpy.float64
    )


def _nr_e2_cderi_bar_packed_aux_block_native(ybar, mo_k, mo_l):
    """Two-GEMM packed AO projection for one contiguous auxiliary slab."""
    naux, kc, lc = ybar.shape
    nao = mo_k.shape[0]
    npair = nao * (nao + 1) // 2
    y2 = numpy.asarray(
        ybar.transpose(0, 2, 1).reshape(naux * lc, kc),
        order='C',
        dtype=numpy.double,
    )
    mo_k = numpy.asarray(mo_k, order='C', dtype=numpy.double)
    mo_l = numpy.asarray(mo_l, order='C', dtype=numpy.double)
    out = numpy.empty((naux, npair), order='C', dtype=numpy.double)
    status = libao2mo_vjp.AO2MOnr_e2_cderi_bar_pack_aux_block(
        out.ctypes.data_as(ctypes.c_void_p),
        y2.ctypes.data_as(ctypes.c_void_p),
        mo_k.ctypes.data_as(ctypes.c_void_p),
        mo_l.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(naux),
        ctypes.c_int(nao),
        ctypes.c_int(kc),
        ctypes.c_int(lc),
    )
    if status != 0:
        raise MemoryError('Native packed AO-projection workspace allocation failed.')
    return out


def _nr_e2_cderi_bar_packed_aux_block_numpy(ybar, mo_k, mo_l):
    """NumPy fallback for the two-GEMM auxiliary-slab projection."""
    naux, kc, lc = ybar.shape
    nao = mo_k.shape[0]
    y2 = numpy.asarray(ybar.transpose(0, 2, 1).reshape(naux * lc, kc), order='C')
    tmp = numpy.dot(y2, numpy.asarray(mo_k, order='C').T)
    tmp = tmp.reshape(naux, lc, nao)

    rows, cols = _tril_indices(nao)
    offdiag = rows != cols
    out = numpy.empty((naux, rows.size), dtype=numpy.result_type(
        ybar.dtype, mo_k.dtype, mo_l.dtype
    ))
    mo_l_t = numpy.asarray(mo_l, order='C').T
    for p in range(naux):
        mat = numpy.dot(tmp[p].T, mo_l_t)
        out[p] = mat[rows, cols]
        out[p, offdiag] += mat[cols[offdiag], rows[offdiag]]
    return out


def _prepare_nr_e2_cderi_bar_aux_projection(
        mo_coeff, ybar, orbs_slice):
    """Validate the projection and prepare reusable coefficient blocks."""
    mo = numpy.asarray(jax.device_get(mo_coeff))
    ybar = numpy.asarray(jax.device_get(ybar))
    if mo.ndim != 2:
        raise ValueError(f'mo_coeff must be rank 2, got shape {mo.shape}.')

    k0, k1, l0, l1 = map(int, orbs_slice)
    if not (0 <= k0 <= k1 <= mo.shape[1] and 0 <= l0 <= l1 <= mo.shape[1]):
        raise ValueError(
            f'orbs_slice {orbs_slice} is incompatible with mo_coeff shape {mo.shape}.'
        )
    kc = k1 - k0
    lc = l1 - l0
    naux = int(ybar.shape[0])
    if ybar.size != naux * kc * lc:
        raise ValueError(
            f'ybar has shape {ybar.shape}; expected {naux * kc * lc} values.'
        )
    ybar = ybar.reshape(naux, kc, lc)
    mo_k = mo[:, k0:k1]
    mo_l = mo[:, l0:l1]
    ybar, mo_k, mo_l, _, _, _ = _select_ao_projection_order(
        ybar, mo_k, mo_l
    )
    return (
        ybar,
        numpy.asarray(mo_k, order='C'),
        numpy.asarray(mo_l, order='C'),
    )


def _nr_e2_cderi_bar_packed_aux_block_prepared(ybar, mo_k, mo_l):
    """Project one auxiliary slab using prepared coefficient blocks."""
    naux = int(ybar.shape[0])
    nao = int(mo_k.shape[0])
    npair = nao * (nao + 1) // 2
    if naux == 0 or nao == 0 or ybar.shape[1] == 0 or ybar.shape[2] == 0:
        return numpy.zeros((naux, npair), dtype=numpy.result_type(
            ybar.dtype, mo_k.dtype, mo_l.dtype
        ))
    if _use_native_cderi_bar_aux_block(ybar, mo_k, mo_l):
        return _nr_e2_cderi_bar_packed_aux_block_native(ybar, mo_k, mo_l)
    return _nr_e2_cderi_bar_packed_aux_block_numpy(ybar, mo_k, mo_l)


def nr_e2_cderi_bar_packed_aux_block(mo_coeff, ybar, orbs_slice):
    """Build every packed AO-pair cotangent for one auxiliary slab.

    Unlike :func:`nr_e2_cderi_bar_packed_block`, which evaluates selected AO
    pairs independently, this routine forms the complete AO matrix with two
    sequential GEMMs and packs it afterward.  Its cost for a slab of ``b``
    auxiliary functions is

    ``O(b * (nao*kc*lc + nao**2*min(kc,lc)))``

    because the smaller trailing MO dimension is always selected.
    """
    ybar, mo_k, mo_l = _prepare_nr_e2_cderi_bar_aux_projection(
        mo_coeff, ybar, orbs_slice
    )
    return _nr_e2_cderi_bar_packed_aux_block_prepared(ybar, mo_k, mo_l)


def _nr_e2_cderi_bar_aux_blksize(mo_coeff, ybar, orbs_slice):
    """Choose an auxiliary slab size for the packed two-GEMM projection."""
    naux = int(ybar.shape[0])
    if naux == 0:
        return 1
    nao = int(mo_coeff.shape[0])
    k0, k1, l0, l1 = map(int, orbs_slice)
    kc = k1 - k0
    lc = l1 - l0
    effective_lc = min(kc, lc)
    npair = nao * (nao + 1) // 2
    itemsize = numpy.dtype(numpy.result_type(mo_coeff.dtype, ybar.dtype)).itemsize
    target_bytes = _cderi_bar_aux_block_mb() * 1024.0**2
    # Per auxiliary row, account for the packed output, the first-GEMM
    # result, and the contiguous y2 view required by BLAS.  The second GEMM
    # needs one dense AO matrix per active OpenMP thread; the native kernel
    # caps that thread count at the number of rows in the slab.
    bytes_per_aux = max(
        (npair + nao * effective_lc + kc * lc) * itemsize,
        1,
    )
    matrix_bytes = nao * nao * itemsize
    max_threads = max(int(pyscf_lib.num_threads()), 1)
    coeff_bytes = nao * (kc + lc) * itemsize
    target_bytes = max(
        target_bytes - coeff_bytes,
        bytes_per_aux + matrix_bytes,
    )

    small_region = int(target_bytes // max(bytes_per_aux + matrix_bytes, 1))
    if naux <= max_threads or small_region < max_threads:
        return max(1, min(naux, small_region))

    available = target_bytes - max_threads * matrix_bytes
    large_region = int(available // bytes_per_aux)
    return max(1, min(naux, large_region))


@contextmanager
def nr_e2_cderi_bar_packed_disk(mo_coeff, ybar, orbs_slice,
                                directory=None, aux_blksize=None):
    """Store a packed AO cotangent using aux writes and pair-oriented reads.

    The HDF5 dataset has the natural ``(naux, nao_pair)`` shape and two-
    dimensional chunks.  Auxiliary slabs are generated by
    :func:`nr_e2_cderi_bar_packed_aux_block` and written directly as row slabs.
    Consumers can subsequently read contiguous AO-pair column slabs without
    materializing the complete array.  This layout avoids a full-slab
    transpose copy on write while supporting both access directions.
    """
    mo = numpy.asarray(jax.device_get(mo_coeff))
    ybar = numpy.asarray(jax.device_get(ybar))
    if ybar.ndim == 0:
        raise ValueError('ybar must have an auxiliary dimension.')
    naux = int(ybar.shape[0])
    nao = int(mo.shape[0])
    npair = nao * (nao + 1) // 2
    if aux_blksize is None:
        aux_blksize = _nr_e2_cderi_bar_aux_blksize(
            mo, ybar, orbs_slice
        )
    aux_blksize = max(1, min(max(naux, 1), int(aux_blksize)))
    ybar_project, mo_k, mo_l = _prepare_nr_e2_cderi_bar_aux_projection(
        mo, ybar, orbs_slice
    )

    if directory is None:
        directory = (
            os.environ.get('PYSCFAD_DF_CDERI_BAR_DISK_DIR')
            or pyscf_lib.param.TMPDIR
        )
    fd, path = tempfile.mkstemp(
        suffix='.h5', prefix='pyscfad_cderi_bar_', dir=directory
    )
    os.close(fd)
    resource_start = None
    resource_finished = False
    h5file = None
    try:
        resource_start = resource_profile.start()
        h5file = h5py.File(path, 'w')
        create_kwargs = {}
        if npair > 0 and naux > 0:
            aux_chunk = min(naux, aux_blksize)
            chunk_target = _cderi_bar_hdf_chunk_mb() * 1024.0**2
            itemsize = numpy.dtype(numpy.result_type(mo.dtype, ybar.dtype)).itemsize
            pair_chunk = max(1, int(chunk_target // max(aux_chunk * itemsize, 1)))
            pair_chunk = min(npair, 1024, pair_chunk)
            create_kwargs['chunks'] = (aux_chunk, pair_chunk)
        dataset = h5file.create_dataset(
            'cderi_bar',
            shape=(naux, npair),
            dtype=numpy.result_type(mo.dtype, ybar.dtype),
            **create_kwargs,
        )

        for p0 in range(0, naux, aux_blksize):
            p1 = min(p0 + aux_blksize, naux)
            block = _nr_e2_cderi_bar_packed_aux_block_prepared(
                ybar_project[p0:p1], mo_k, mo_l
            )
            dataset[p0:p1, :] = block
            block = None
        h5file.flush()
        resource_profile.finish(
            'df_vjp.cderi_bar_disk_build',
            resource_start,
            naux=naux,
            local_nao=nao,
            local_ao_pairs=npair,
            aux_block=aux_blksize,
            hdf_chunks=dataset.chunks,
            disk_mib=(npair * naux * dataset.dtype.itemsize / 1024.0**2),
        )
        resource_finished = True
        yield dataset
    finally:
        try:
            if resource_start is not None and not resource_finished:
                resource_profile.finish(
                    'df_vjp.cderi_bar_disk_build_incomplete',
                    resource_start,
                    naux=naux,
                    local_nao=nao,
                    local_ao_pairs=npair,
                    aux_block=aux_blksize,
                )
        finally:
            try:
                if h5file is not None:
                    h5file.close()
            finally:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass


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

    detail = _DF_VJP_DETAILED_TIMING.get()
    if detail is not None:
        detail['counters']['local_ao_pair_positions_projected'] += int(
            pair_positions.size
        )

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
    if detail is None:
        out = _nr_e2_cderi_bar_project(ybar, mok_rows, mol_cols)
    else:
        backend = (
            'native' if _use_native_cderi_bar_project(
                ybar, mok_rows, mol_cols
            ) else 'numpy'
        )
        timing_start = _detailed_timing_start()
        out = _nr_e2_cderi_bar_project(ybar, mok_rows, mol_cols)
        _record_projection_kernel(
            detail,
            _detailed_timing_elapsed(timing_start),
            'primary',
            backend,
            pair_positions.size,
        )

    offdiag = rows != cols
    if numpy.any(offdiag):
        mok_cols = mo[cols[offdiag], k0:k1]
        mol_rows = mo[rows[offdiag], l0:l1]
        if detail is None:
            out[:, offdiag] += _nr_e2_cderi_bar_project(
                ybar, mok_cols, mol_rows
            )
        else:
            backend = (
                'native' if _use_native_cderi_bar_project(
                    ybar, mok_cols, mol_rows
                ) else 'numpy'
            )
            timing_start = _detailed_timing_start()
            exchanged = _nr_e2_cderi_bar_project(
                ybar, mok_cols, mol_rows
            )
            _record_projection_kernel(
                detail,
                _detailed_timing_elapsed(timing_start),
                'exchanged_offdiagonal',
                backend,
                mok_cols.shape[0],
            )
            out[:, offdiag] += exchanged
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
    del j2c, j2c_np
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
    del cderi_bar, low

    def metric_cholesky(auxmol_):
        return jax_scipy.linalg.cholesky(auxmol_.intor(int2c, hermi=1), lower=True)

    t = time.perf_counter()
    _, chol_pullback = jax.vjp(metric_cholesky, auxmol)
    _zero_strict_upper_inplace(low_bar)
    aux_metric_bar = chol_pullback(np.asarray(low_bar))[0]
    del chol_pullback, low_bar
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


def _int3c_coordinate_vjp_block(mol, auxmol, ints_bar, *, int3c,
                                shls_slice, naoaux):
    """Return the coordinate VJP for one packed AO-pair shell block.

    Keeping this operation separate from the Cholesky-whitening algebra lets
    the latter stream blocks on one process while independent three-centre
    integral derivatives are evaluated by MPI workers.  ``ints_bar`` is the
    cotangent after the :math:`L^{-T}` triangular solve and therefore has
    shape ``(naoaux, packed_pair_block)``.
    """

    ints_bar = numpy.asarray(ints_bar)
    naoaux = int(naoaux)
    if ints_bar.ndim != 2 or ints_bar.shape[0] != naoaux:
        raise ValueError(
            'three-centre integral cotangent has shape '
            f'{ints_bar.shape}, expected ({naoaux}, packed_pair_block)'
        )

    def int3c_block(mol_, auxmol_):
        ints = _int3c_cross_opt.int3c_cross(
            mol_, auxmol_, intor=int3c, comp=1,
            aosym='s2ij', shls_slice=tuple(shls_slice)
        )
        return ints.reshape((-1, naoaux)).T

    _, int3c_pullback = jax.vjp(int3c_block, mol, auxmol)
    return int3c_pullback(np.asarray(ints_bar))


def cholesky_eri_vjp_from_cderi_block_fn(mol, auxmol, cderi_source,
                                         cderi_bar_block_fn, max_memory,
                                         int3c=None, int2c=None,
                                         aosym='s2ij',
                                         int3c_block_vjp=None,
                                         max_pair_block=None):
    """Back-propagate through Cholesky CDERI from AO-pair cotangent blocks."""
    t_total = time.perf_counter()
    detail = _new_df_vjp_detailed_timing() if _profile_enabled() else None
    complete_timing_start = (
        (t_total, time.process_time()) if detail is not None else None
    )
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
    metric_timing_start = (
        _detailed_timing_start() if detail is not None else None
    )
    resource_phase = resource_profile.start()
    j2c = auxmol.intor(int2c, hermi=1)
    j2c_np = numpy.asarray(jax.device_get(j2c))
    try:
        low = scipy.linalg.cholesky(j2c_np, lower=True, check_finite=False)
    except scipy.linalg.LinAlgError as err:
        raise NotImplementedError('2c metric Cholesky fallback is not implemented.') from err
    if detail is not None:
        _record_detailed_timing(
            detail,
            'metric_integral_initial_cholesky',
            _detailed_timing_elapsed(metric_timing_start),
        )

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
    del j2c, j2c_np

    max_words = max(max_memory, 0) * 1e6 / 8 - low.size
    mem_buflen = max(int(max_words / max(naoaux, 1) / 3), 8)
    target_buflen = max(int(_INT3C_VJP_TARGET_MB * 1e6 / 8 / max(naoaux, 1) / 3), 8)
    buflen = min(nao_pair, mem_buflen, target_buflen)
    if max_pair_block is not None:
        max_pair_block = int(max_pair_block)
        if max_pair_block <= 0:
            raise ValueError('max_pair_block must be positive')
        buflen = min(buflen, max(max_pair_block, 8))
    shranges = _guess_shell_ranges(mol, buflen, aosym)

    mol_bar = None
    auxmol_bar = None
    low_bar = numpy.zeros_like(low)

    t = time.perf_counter()
    resource_phase = resource_profile.start()
    nblocks = 0
    detail_token = (
        _DF_VJP_DETAILED_TIMING.set(detail) if detail is not None else None
    )
    try:
        with df_addons.load(cderi_source, 'j3c') as feri:
            if int(feri.shape[0]) != naoaux or int(feri.shape[1]) != nao_pair:
                raise NotImplementedError('CDERI source shape does not match mol/auxmol.')

            p1 = 0
            for block_index, sh_range in enumerate(shranges):
                if detail is not None:
                    detail['active_block_index'] = block_index
                    detail['counters']['candidate_shell_blocks'] += 1
                    outer_timing_start = _detailed_timing_start()
                    child_elapsed = {}

                bstart, bend, _ = sh_range
                shls_slice = (
                    bstart, bend,
                    0, mol.nbas,
                    mol.nbas, mol.nbas + auxmol.nbas,
                )

                p0, p1 = p1, p1 + sh_range[2]
                if detail is not None:
                    detail['counters']['global_pair_positions_encountered'] += (
                        p1 - p0
                    )
                    stage_timing_start = _detailed_timing_start()
                cderi_bar_blk = numpy.asarray(cderi_bar_block_fn(p0, p1))
                if detail is not None:
                    elapsed = _detailed_timing_elapsed(stage_timing_start)
                    child_elapsed['ao_projection_total'] = elapsed
                    _record_detailed_timing(
                        detail, 'ao_projection_total', elapsed,
                        block_index=block_index,
                    )
                if cderi_bar_blk.shape != (naoaux, p1 - p0):
                    raise RuntimeError(
                        'CDERI cotangent block has shape '
                        f'{cderi_bar_blk.shape}, expected {(naoaux, p1 - p0)}.'
                    )

                if detail is not None:
                    stage_timing_start = _detailed_timing_start()
                has_nonzero = numpy.any(cderi_bar_blk)
                if detail is not None:
                    elapsed = _detailed_timing_elapsed(stage_timing_start)
                    child_elapsed['zero_check'] = elapsed
                    _record_detailed_timing(
                        detail, 'zero_check', elapsed,
                        block_index=block_index,
                    )
                if not has_nonzero:
                    if detail is not None:
                        detail['counters']['skipped_zero_blocks'] += 1
                        _finish_outer_block_timing(
                            detail, outer_timing_start, child_elapsed,
                            block_index,
                        )
                        detail['active_block_index'] = None
                    continue
                nblocks += 1
                if detail is not None:
                    detail['counters']['processed_nonzero_blocks'] += 1

                if detail is not None:
                    stage_timing_start = _detailed_timing_start()
                cderi_blk = numpy.asarray(feri[:, p0:p1])
                if detail is not None:
                    elapsed = _detailed_timing_elapsed(stage_timing_start)
                    child_elapsed['cderi_read'] = elapsed
                    _record_detailed_timing(
                        detail, 'cderi_read', elapsed,
                        block_index=block_index,
                    )

                if detail is not None:
                    stage_timing_start = _detailed_timing_start()
                ints_bar = scipy.linalg.solve_triangular(
                    low.T, cderi_bar_blk, lower=False, check_finite=False
                )
                if detail is not None:
                    elapsed = _detailed_timing_elapsed(stage_timing_start)
                    child_elapsed['triangular_solve'] = elapsed
                    _record_detailed_timing(
                        detail, 'triangular_solve', elapsed,
                        block_index=block_index,
                    )

                if detail is not None:
                    stage_timing_start = _detailed_timing_start()
                low_bar -= ints_bar @ cderi_blk.T
                if detail is not None:
                    elapsed = _detailed_timing_elapsed(stage_timing_start)
                    child_elapsed['low_bar_gemm'] = elapsed
                    _record_detailed_timing(
                        detail, 'low_bar_gemm', elapsed,
                        block_index=block_index,
                    )

                if detail is not None:
                    stage_timing_start = _detailed_timing_start()
                int3c_pullback = None
                if int3c_block_vjp is None:
                    def int3c_block(mol_, auxmol_):
                        ints = _int3c_cross_opt.int3c_cross(
                            mol_, auxmol_, intor=int3c, comp=1,
                            aosym='s2ij', shls_slice=shls_slice
                        )
                        return ints.reshape((-1, naoaux)).T

                    _, int3c_pullback = jax.vjp(int3c_block, mol, auxmol)
                if detail is not None:
                    elapsed = _detailed_timing_elapsed(stage_timing_start)
                    child_elapsed['int3c_primal_vjp_setup'] = elapsed
                    _record_detailed_timing(
                        detail, 'int3c_primal_vjp_setup', elapsed,
                        block_index=block_index,
                    )

                if detail is not None:
                    stage_timing_start = _detailed_timing_start()
                if int3c_block_vjp is None:
                    mol_blk_bar, auxmol_blk_bar = int3c_pullback(
                        np.asarray(ints_bar)
                    )
                else:
                    mol_blk_bar, auxmol_blk_bar = int3c_block_vjp(
                        block_index, shls_slice, ints_bar
                    )
                if detail is not None:
                    elapsed = _detailed_timing_elapsed(stage_timing_start)
                    child_elapsed['int3c_pullback'] = elapsed
                    _record_detailed_timing(
                        detail, 'int3c_pullback', elapsed,
                        block_index=block_index,
                    )

                if detail is not None:
                    stage_timing_start = _detailed_timing_start()
                mol_bar = _tree_add(mol_bar, mol_blk_bar)
                auxmol_bar = _tree_add(auxmol_bar, auxmol_blk_bar)
                if detail is not None:
                    elapsed = _detailed_timing_elapsed(stage_timing_start)
                    child_elapsed['gradient_accumulation'] = elapsed
                    _record_detailed_timing(
                        detail, 'gradient_accumulation', elapsed,
                        block_index=block_index,
                    )
                cderi_bar_blk = cderi_blk = ints_bar = None
                mol_blk_bar = auxmol_blk_bar = None
                if detail is not None:
                    _finish_outer_block_timing(
                        detail, outer_timing_start, child_elapsed, block_index
                    )
                    detail['active_block_index'] = None
    finally:
        if detail is not None:
            detail['active_block_index'] = None
            _DF_VJP_DETAILED_TIMING.reset(detail_token)
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
    del low

    def metric_cholesky(auxmol_):
        return jax_scipy.linalg.cholesky(auxmol_.intor(int2c, hermi=1), lower=True)

    t = time.perf_counter()
    resource_phase = resource_profile.start()
    metric_vjp_timing_start = (
        _detailed_timing_start() if detail is not None else None
    )
    _, chol_pullback = jax.vjp(metric_cholesky, auxmol)
    if detail is not None:
        _record_detailed_timing(
            detail,
            'metric_vjp_primal_creation',
            _detailed_timing_elapsed(metric_vjp_timing_start),
        )
        metric_pullback_timing_start = _detailed_timing_start()
    _zero_strict_upper_inplace(low_bar)
    aux_metric_bar = chol_pullback(np.asarray(low_bar))[0]
    del chol_pullback, low_bar
    auxmol_bar = _tree_add(auxmol_bar, aux_metric_bar)
    if detail is not None:
        _record_detailed_timing(
            detail,
            'metric_cholesky_pullback',
            _detailed_timing_elapsed(metric_pullback_timing_start),
        )
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
    if detail is not None:
        _record_detailed_timing(
            detail,
            'complete_response',
            _detailed_timing_elapsed(complete_timing_start),
        )
        _emit_df_vjp_detailed_timing(detail)
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


def _int3c_ip1_mo_density_contractions(ints, mo_k, mo_l, z_blk):
    """Contract one ``int3c_ip1`` block for both coordinate centres.

    For each auxiliary function, form

    ``D[p,u,v] = sum(k,l) mo_k[u,k] * z[p,k,l] * mo_l[v,l]``.

    The two AO-centre terms use ``D + D.T``.  Translational invariance of each
    three-centre Coulomb integral gives

    ``d/dR_aux = -(d/dR_AO1 + d/dR_AO2)``.

    Retaining the auxiliary index until the last reduction therefore yields
    both the AO-centre and auxiliary-centre contractions from ``int3c_ip1``.
    This avoids a separate ``int3c_ip2`` integral build and AO-to-MO
    transformation.  Building ``D`` in the order whose AO-square GEMM carries
    the smaller of ``k`` and ``l`` remains important for local MP2, where the
    virtual dimension is normally much larger than the occupied dimension.

    No conjugation is introduced here: this is algebraically identical to the
    two contractions it replaces and therefore preserves their real/complex
    convention.
    """
    kc = int(mo_k.shape[1])
    lc = int(mo_l.shape[1])
    if kc <= lc:
        # Contract the large l space before forming the AO-square object, so
        # the latter contraction is over k.
        tmp = pyscf_lib.einsum('pkl,vl->pkv', z_blk, mo_l)
        density = pyscf_lib.einsum('uk,pkv->puv', mo_k, tmp)
    else:
        # The transposed order makes the AO-square contraction run over l.
        tmp = pyscf_lib.einsum('uk,pkl->pul', mo_k, z_blk)
        density = pyscf_lib.einsum('pul,vl->puv', tmp, mo_l)
    tmp = None

    density = density + density.swapaxes(1, 2)
    per_aux_ao = numpy.einsum('puv,xuvp->pux', density, ints)
    return per_aux_ao.sum(axis=0), per_aux_ao.sum(axis=1)


def _int3c_ip1_mo_density_contraction(ints, mo_k, mo_l, z_blk):
    """Compatibility wrapper returning only the AO-centre contraction."""
    return _int3c_ip1_mo_density_contractions(
        ints, mo_k, mo_l, z_blk
    )[0]


def _iter_auxiliary_subranges(p0, p1, max_rows):
    """Yield disjoint bounded AO ranges for an oversized single shell."""
    max_rows = int(max_rows)
    if max_rows <= 0:
        raise ValueError('z_aux_block_max_rows must be positive')
    for read0 in range(int(p0), int(p1), max_rows):
        yield read0, min(read0 + max_rows, int(p1))


def _int3c_mo_deriv_coords_vjp_from_z_reader(
        mol, auxmol, mo_coeff, read_z_aux_block, orbs_slice,
        int3c='int3c2e', aosym='s2ij', block_memory_mb=None,
        z_dtype=None, z_aux_block_max_rows=None):
    """Coordinate pullback with bounded z reads per auxiliary-shell block."""
    if aosym not in ('s2', 's2ij'):
        raise NotImplementedError(f'Only packed s2 CDERI is supported, got {aosym}.')
    if mo_coeff.shape[0] != mol.nao:
        raise NotImplementedError(
            'Direct int3c-MO VJP currently requires the full AO basis.'
        )

    nao = mol.nao
    naux = auxmol.nao
    nbas = mol.nbas
    k0, k1, l0, l1 = orbs_slice
    kc = k1 - k0
    lc = l1 - l0
    mo_k = numpy.asarray(mo_coeff[:, k0:k1], order='F')
    mo_l = numpy.asarray(mo_coeff[:, l0:l1], order='F')
    if kc == 0 or lc == 0:
        if z_dtype is None:
            z_dtype = numpy.asarray(mo_coeff).dtype
        mol_ao_bar = numpy.zeros((nao, 3), dtype=z_dtype)
        aux_ao_bar = numpy.zeros((naux, 3), dtype=z_dtype)
        return (
            _coords_tree_like(
                mol, _ao_to_atom_coords_bar(mol, mol_ao_bar)
            ),
            _coords_tree_like(
                auxmol, _ao_to_atom_coords_bar(auxmol, aux_ao_bar)
            ),
        )

    if block_memory_mb is None:
        block_memory_mb = _int3c_mo_vjp_block_mb()
    else:
        block_memory_mb = max(float(block_memory_mb), 1.0)
    target_words = block_memory_mb * 1024.0**2 / 8
    words_per_aux = (
        # AO-centre phase: the three derivative components plus the density
        # and its symmetrized replacement can briefly coexist during D+D.T.
        # The much smaller per-auxiliary AO contraction is retained only long
        # enough to reduce its AO and auxiliary centre indices.
        5 * nao * nao
        + nao * max(min(kc, lc), 1)
        + 3 * nao
    )
    blksize = max(1, min(naux, int(target_words // max(words_per_aux, 1))))
    if z_aux_block_max_rows is not None:
        z_aux_block_max_rows = int(z_aux_block_max_rows)
        if z_aux_block_max_rows <= 0:
            raise ValueError('z_aux_block_max_rows must be positive')
        blksize = min(blksize, z_aux_block_max_rows)
    aux_loc = auxmol.ao_loc
    aux_ranges = balance_partition(aux_loc, blksize)

    int3c_ip1 = int3c.replace('int3c2e', 'int3c2e_ip1')
    mol_ao_bar = None
    aux_ao_bar = None

    def contract_aux_range(read0, read1, ints):
        nonlocal mol_ao_bar, aux_ao_bar
        z_blk = numpy.asarray(read_z_aux_block(read0, read1))
        expected_shape = (read1 - read0, kc * lc)
        if z_blk.shape != expected_shape:
            raise ValueError(
                'z auxiliary block has incompatible shape: '
                f'got {z_blk.shape}, expected {expected_shape}'
            )
        z_blk = z_blk.reshape(read1 - read0, kc, lc)
        if mol_ao_bar is None:
            mol_ao_bar = numpy.zeros((nao, 3), dtype=z_blk.dtype)
            aux_ao_bar = numpy.zeros((naux, 3), dtype=z_blk.dtype)
        mol_contraction, aux_contraction = \
            _int3c_ip1_mo_density_contractions(
                ints, mo_k, mo_l, z_blk
            )
        mol_ao_bar -= mol_contraction
        # The libcint ``ip1`` convention carries the same leading minus sign
        # as the former explicit ``ip2`` path.  Translational invariance
        # reverses the auxiliary-centre contribution.
        aux_ao_bar[read0:read1] += aux_contraction

    for shl0, shl1, _ in aux_ranges:
        p0, p1 = int(aux_loc[shl0]), int(aux_loc[shl1])
        shls_slice = (0, nbas, 0, nbas, nbas + shl0, nbas + shl1)
        ints = _int3c_cross_opt.int3c_cross(
            mol, auxmol, intor=int3c_ip1, comp=3, aosym='s1',
            shls_slice=shls_slice,
        )
        ints = numpy.asarray(ints)
        if z_aux_block_max_rows is None or p1 - p0 <= z_aux_block_max_rows:
            contract_aux_range(p0, p1, ints)
        else:
            # Shell granularity can exceed the requested row cap.  The shell
            # integral is still generated once, then sliced into disjoint
            # logical blocks paired one-for-one with bounded z reads.
            for read0, read1 in _iter_auxiliary_subranges(
                    p0, p1, z_aux_block_max_rows):
                int0 = read0 - p0
                int1 = read1 - p0
                contract_aux_range(
                    read0, read1, ints[..., int0:int1]
                )
        ints = None

    if mol_ao_bar is None:
        if z_dtype is None:
            z_dtype = numpy.asarray(mo_coeff).dtype
        mol_ao_bar = numpy.zeros((nao, 3), dtype=z_dtype)
        aux_ao_bar = numpy.zeros((naux, 3), dtype=z_dtype)

    mol_bar = _coords_tree_like(mol, _ao_to_atom_coords_bar(mol, mol_ao_bar))
    auxmol_bar = _coords_tree_like(
        auxmol, _ao_to_atom_coords_bar(auxmol, aux_ao_bar)
    )
    return mol_bar, auxmol_bar


def _int3c_mo_deriv_coords_vjp(mol, auxmol, mo_coeff, z,
                               orbs_slice, int3c='int3c2e',
                               aosym='s2ij', block_memory_mb=None,
                               z_aux_block_max_rows=None):
    """Coordinate pullback from a full ``z`` array or auxiliary-block reader."""
    if callable(z):
        read_z_aux_block = z
        z_dtype = None
    else:
        k0, k1, l0, l1 = orbs_slice
        z_array = numpy.asarray(z).reshape(
            auxmol.nao, (k1 - k0) * (l1 - l0)
        )
        read_z_aux_block = lambda p0, p1: z_array[p0:p1, :]
        z_dtype = z_array.dtype
    return _int3c_mo_deriv_coords_vjp_from_z_reader(
        mol,
        auxmol,
        mo_coeff,
        read_z_aux_block,
        orbs_slice,
        int3c=int3c,
        aosym=aosym,
        block_memory_mb=block_memory_mb,
        z_dtype=z_dtype,
        z_aux_block_max_rows=z_aux_block_max_rows,
    )


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
    del j2c, j2c_np, low, ybar
    _profile_msg(
        'cholesky_eri_vjp_from_mo_coeff_ybar metric/z setup done '
        f'{time.perf_counter() - t:.2f} s'
    )

    t = time.perf_counter()
    y = _stream_nr_e2_from_cderi_source(
        cderi_source, mo_coeff, orbs_slice, max_memory, aosym='s2'
    )
    low_bar = -numpy.dot(z, y.T)
    del y
    _profile_msg(
        'cholesky_eri_vjp_from_mo_coeff_ybar stream y/low_bar done '
        f'{time.perf_counter() - t:.2f} s'
    )

    t = time.perf_counter()
    mol_bar, auxmol_bar = _int3c_mo_deriv_coords_vjp(
        mol, auxmol, mo_coeff, z, orbs_slice, int3c=int3c, aosym=aosym
    )
    del z
    _profile_msg(
        'cholesky_eri_vjp_from_mo_coeff_ybar int3c_mo_deriv done '
        f'{time.perf_counter() - t:.2f} s'
    )

    def metric_cholesky(auxmol_):
        return jax_scipy.linalg.cholesky(auxmol_.intor(int2c, hermi=1), lower=True)

    t = time.perf_counter()
    _, chol_pullback = jax.vjp(metric_cholesky, auxmol)
    _zero_strict_upper_inplace(low_bar)
    aux_metric_bar = chol_pullback(np.asarray(low_bar))[0]
    del chol_pullback, low_bar
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
