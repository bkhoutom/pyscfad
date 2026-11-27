"""Deterministic MP2/D/W adapter: SciPy LAPACK and the RCCSD C++ BLAS runtime.

Only small replay metadata survives forward. The delayed pullback rebuilds
amplitudes and uses the full matrix-exponential Frechet derivative, including
within degenerate eigenspaces. Armadillo is not imported by this module.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from pyscfad.cc._native_kernels import _ccsd_kernels as cpp
from .controls import normalize_matrix_controls
from .protocol import matrix_request_fingerprint, validate_full_inputs

matrix_kernel_version = 'scipy_blas_deterministic_w_v1'
checkpoint_matrix_version = 'src_code_rdm_v2:427459d9b91fc70f:standalone-6-d-w'


@dataclass(frozen=True)
class Replay:
    targets: tuple
    controls: dict
    offsets: np.ndarray
    fingerprint: str
    native_version: str = matrix_kernel_version
    matrix_kind: str = "W"


def available():
    return cpp is not None and hasattr(cpp, 'deterministic_w')


def _request(foo, fvv, factors, targets, controls, offsets=None, matrix_kind="W"):
    from .backend import _matrix_targets, _matrix_aux_offsets
    inputs = dict(foo=foo, fvv=fvv, B=factors)
    validate_full_inputs(inputs)
    settings = normalize_matrix_controls(controls, matrix_kind)
    if settings['mode'] != 'deterministic':
        raise ValueError('native W supports deterministic controls only')
    targets = _matrix_targets(dict(target_indices=targets), foo.shape[0])
    offsets = _matrix_aux_offsets(factors.shape[0], offsets, settings['auxiliary_group_size'])
    fingerprint = matrix_request_fingerprint(inputs, dict(target_indices=targets), settings, offsets, matrix_kind=matrix_kind)
    return Replay(targets, settings, offsets, fingerprint, matrix_kind=matrix_kind)


def _check_replay(foo, fvv, factors, replay):
    if not isinstance(replay, Replay) or replay.native_version != matrix_kernel_version:
        raise ValueError('native W replay version/type changed')
    current = _request(foo, fvv, factors, replay.targets, replay.controls, replay.offsets, replay.matrix_kind)
    if current.fingerprint != replay.fingerprint:
        raise ValueError('native W replay inputs or controls changed')


def _run(foo, fvv, factors, replay, seeds=None, *, per_target=False, sequential=False, energy_only=False,
         occupied_rotation=None, rotation_pullback=None):
    if not available():
        raise ImportError('rebuild pyscfad/cc/cpp for deterministic MP2/D/W')
    threads = replay.controls['threads']
    started = perf_counter()
    # Like CCSD, set the native OpenMP budget and restore the caller's state.
    # LAPACK runs outside the outer team; kernel GEMMs are serial within it.
    with threadpool_limits(threads, user_api='openmp'):
        with threadpool_limits(threads, user_api='blas'):
            eo, uo = eigh(foo, check_finite=False, driver='evd')
            ev, uv = eigh(fvv, check_finite=False, driver='evd')
        if not len(eo) or not len(ev) or ev[0] <= eo[-1]:
            raise ValueError('STC-MP2 requires a positive occupied-to-virtual gap')
        eig_seconds = perf_counter() - started
        if seeds is None:
            eb = ob = vb = np.empty(0)
        else:
            energy, occupied, virtuals = seeds
            nt = len(replay.targets)
            eb, ob, vb = (np.asarray(a, dtype=np.float64) for a in (energy, occupied, virtuals))
            if (eb.shape != (nt,) or ob.shape != (nt, *foo.shape)
                    or vb.shape != (nt, *fvv.shape)
                    or any(not np.isfinite(a).all() for a in (eb, ob, vb))):
                raise ValueError('invalid native W cotangent shape or values')
        arrays = (factors, uo, uv, eo, ev, replay.controls['laplace_roots'],
                  replay.controls['laplace_weights'])
        arrays = tuple(np.require(a, dtype=np.float64, requirements=['C', 'A']) for a in arrays)
        sequential_options = {}
        if occupied_rotation is not None:
            sequential_options = dict(
                occupied_rotation=np.require(occupied_rotation, dtype=np.float64, requirements=['C', 'A']),
                rotation_pullback=rotation_pullback)
        with threadpool_limits(1, user_api='blas'):
            result = cpp.deterministic_w(*arrays, replay.targets,
                *(np.require(a, requirements=['C', 'A']) for a in (eb, ob, vb)),
                per_target, sequential, replay.matrix_kind == "D", energy_only,
                **sequential_options)
    result['timings']['eigensolve_seconds'] = eig_seconds
    result['timings']['adapter_total_seconds'] = perf_counter() - started
    return result


def _result(raw, index, replay, *, timings=True):
    occupied, virtual = raw['occupied'][index], raw['virtual'][index]
    diagnostics = dict(mode='deterministic', matrix_kind=replay.matrix_kind, target_indices=list(replay.targets),
        global_seed=replay.controls['global_seed'], threads=replay.controls['threads'],
        openmp_dynamic=raw['openmp_dynamic'],
        native_version=matrix_kernel_version, effective_options=dict(occupied={}, virtual={}),
        energy_standard_error=0., occupied_frobenius_se=0., virtual_frobenius_se=0.,
        occupied_standard_error=np.zeros_like(occupied), virtual_standard_error=np.zeros_like(virtual),
        occupied_target_met=True, virtual_target_met=True, pilot_draws=0, production_draws=0,
        native_timings=raw['timings'] if timings else {})
    return dict(energy=np.float64(raw['energy'][index]), occupied=occupied, virtual=virtual,
                diagnostics=diagnostics, replay=replay)


def solve_matrices(foo, fvv, factors, targets, controls, *, aux_offsets=None, matrix_kind='W'):
    replay = _request(foo, fvv, factors, targets, controls, aux_offsets, matrix_kind)
    return _result(_run(foo, fvv, factors, replay), 0, replay)


def _pullback(foo, fvv, factors, replay, energy, occupied, virtual):
    _check_replay(foo, fvv, factors, replay)
    nt = len(replay.targets)
    seeds = ([energy]*nt, [occupied]*nt, [virtual]*nt)
    raw = _run(foo, fvv, factors, replay, seeds)
    return {key: raw[key] for key in ('foo', 'fvv', 'B')}, raw['timings']


def pullback_matrices(*args):
    return _pullback(*args)[0]


def pullback_matrices_with_timings(*args):
    return _pullback(*args)


def solve_orbital_matrices(foo, fvv, factors, targets, controls, *, aux_offsets=None):
    replay = _request(foo, fvv, factors, targets, controls, aux_offsets)
    raw = _run(foo, fvv, factors, replay, per_target=True)
    return [_result(raw, index, replay, timings=index == 0) for index in range(len(targets))]


def pullback_orbital_matrices(foo, fvv, factors, replay, energies, occupied, virtuals):
    _check_replay(foo, fvv, factors, replay)
    raw = _run(foo, fvv, factors, replay, (energies, occupied, virtuals), per_target=True)
    return {key: raw[key] for key in ('foo', 'fvv', 'B')}


def begin_sequential_matrices(foo, fvv, factors, targets, controls, *, matrix_kind="W"):
    replay = _request(foo, fvv, factors, targets, controls, matrix_kind=matrix_kind)
    raw = _run(foo, fvv, factors, replay, sequential=True)
    result = _result(raw, 0, replay)
    result['workspace'] = (raw['workspace'], replay.controls['threads'])
    result['diagnostics']['sequential_virtual'] = True
    return result


def finish_sequential_virtual(workspace, rotation):
    cache, threads = workspace
    with threadpool_limits(threads, user_api='openmp'), threadpool_limits(1, user_api='blas'):
        return cpp.finish_deterministic_w(cache, np.require(rotation, dtype=np.float64, requirements=['C', 'A']))


def pullback_sequential_matrices(foo, fvv, factors, replay, rotation, rotation_pullback,
                                 energy, occupied, virtual):
    """One delayed replay, with the occupied-selection VJP between native stages."""
    _check_replay(foo, fvv, factors, replay)
    nt = len(replay.targets)
    def checked_pullback(bar):
        value = np.asarray(rotation_pullback(bar), dtype=np.float64)
        if value.shape != foo.shape or not np.isfinite(value).all():
            raise ValueError('occupied selection pullback must return a finite occupied-matrix cotangent')
        return np.require(value, requirements=['C', 'A'])
    raw = _run(foo, fvv, factors, replay, ([energy]*nt, [occupied]*nt, [virtual]*nt),
               occupied_rotation=rotation, rotation_pullback=checked_pullback)
    return {key: raw[key] for key in ('foo', 'fvv', 'B')}, raw['timings']


def solve_selected(foo, fvv, factors, targets, controls, *, aux_offsets=None, with_grad=True):
    from .driver import _driver_controls
    controls = _driver_controls(controls)
    from pyscf.lib import num_threads
    controls.setdefault("threads", num_threads())
    # These affect the old scalar scheduler, not the deterministic sum.
    controls = {k:v for k,v in controls.items() if k not in ('canonical_fock', 'virtual_block_size')}
    replay = _request(foo, fvv, factors, targets, controls, aux_offsets)
    if not factors.shape[0] or not fvv.shape[0]:
        return dict(energy=np.float64(0.), energy_standard_error=np.float64(0.),
                    diagnostics=dict(mode='deterministic', native_version=matrix_kernel_version),
                    cotangents={k: np.zeros_like(a) for k,a in zip(('foo','fvv','B'),(foo,fvv,factors))} if with_grad else {})
    result = dict(energy=np.float64(0.), energy_standard_error=np.float64(0.),
                  diagnostics=dict(mode='deterministic', native_version=matrix_kernel_version,
                                   native_timings=[]), cotangents={})
    if with_grad:
        result['cotangents'] = {k:np.zeros_like(a) for k,a in zip(('foo','fvv','B'),(foo,fvv,factors))}
    batch = replay.controls['target_batch_size'] or 1
    for first in range(0, len(replay.targets), batch):
        local = Replay(replay.targets[first:first+batch], replay.controls, replay.offsets, replay.fingerprint)
        nt = len(local.targets)
        seeds = ([1.]*nt, np.zeros((nt,*foo.shape)), np.zeros((nt,*fvv.shape))) if with_grad else None
        raw = _run(foo, fvv, factors, local, seeds, energy_only=True)
        result['energy'] += np.float64(raw['energy'][0])
        result['diagnostics']['native_timings'].append(raw['timings'])
        if with_grad:
            for key in result['cotangents']:
                result['cotangents'][key] += raw[key]
    return result
