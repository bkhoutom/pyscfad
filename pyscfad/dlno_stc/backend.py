"""In-memory domain MP2 and matrix solvers with explicit host cotangents."""
import importlib
import math
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
import numpy as np
from pyscfad.tools import resource_profile
from pyscfad.lib._threading import blas_threads
from .protocol import (ARRAY_NAMES, validate_full_inputs, validate_full_result,
    matrix_request_fingerprint, validate_matrix_result, validate_input_cotangents)


def _native_factor_layout(value):
    """Recognize the validated, contiguous-in-P host factor view."""
    return (isinstance(value, np.ndarray) and value.ndim == 3
            and value.flags.aligned and (not value.size or all(
                size == 1 or stride == expected
                for size, stride, expected in zip(value.shape, value.strides, (
                    value.itemsize, value.shape[0] * value.shape[2] * value.itemsize,
                    value.shape[0] * value.itemsize)))))


def host_inputs(inputs, *, native_factors=False):
    """Transfer each preparation output once, retaining its float64 dtype."""
    if not isinstance(inputs, dict) or set(inputs) != set(ARRAY_NAMES):
        raise ValueError("inputs must contain exactly " + ", ".join(ARRAY_NAMES))
    arrays = {key: (inputs[key] if key == 'B' and native_factors
                   and _native_factor_layout(inputs[key])
                   else np.ascontiguousarray(inputs[key])) for key in ARRAY_NAMES}
    validate_full_inputs(arrays)
    return arrays


def _stream_seed(global_seed, fragment_id, stream=0):
    for name, value in (("global_seed", global_seed), ("fragment_id", fragment_id)):
        if (not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_)) or value < 0):
            raise ValueError(f"{name} must be a nonnegative integer")
    # Preserve the serial sample stream; MPI rank never changes a fragment's seed.
    state = np.random.SeedSequence([int(global_seed), int(fragment_id), int(stream)])
    return int(state.generate_state(1, dtype=np.uint64)[0])


def _solve_local(inputs, metadata, controls, *, with_grad, aux_offsets):
    try:
        extension = importlib.import_module("pyscfad.dlno_stc._stc_mp2")
    except ImportError as exc:
        raise ImportError("STC-MP2 extension is unavailable; build pyscfad/dlno_stc/cpp "
                          "with CMake as described in pyscfad/dlno_stc/README.md") from exc
    serial_blas = not with_grad and controls.get('mode', 'deterministic') == 'deterministic'
    solve_selected = extension.solve_selected
    if (controls.get('mode', 'deterministic') == 'deterministic'
            and _native_factor_layout(inputs['B'])):
        solve_selected = extension._solve_selected_borrowed
    with blas_threads(1) if serial_blas else nullcontext():
        return solve_selected(*(inputs[key] for key in ARRAY_NAMES),
            tuple(metadata['target_indices']), controls, aux_offsets=aux_offsets,
            with_grad=with_grad)


def solve(inputs, metadata, controls, *, with_grad=True, aux_offsets=None):
    """Solve one fragment locally; the driver distributes complete groups."""
    if not isinstance(metadata, dict) or not isinstance(controls, dict):
        raise ValueError("metadata and controls must be dictionaries")
    arrays = host_inputs(inputs, native_factors=controls.get('mode', 'deterministic') == 'deterministic')
    if aux_offsets is None:
        group_size = controls.get('auxiliary_group_size', 32)
        if (not isinstance(group_size, (int, np.integer))
                or isinstance(group_size, (bool, np.bool_)) or group_size <= 0):
            raise ValueError('auxiliary_group_size must be a positive integer')
        naux = arrays['B'].shape[0]
        aux_offsets = np.asarray([*range(0, naux, int(group_size)), naux]
                                 if naux else [0, 0], dtype=np.int64)
    seed = _stream_seed(controls.get('global_seed', 0), metadata.get('fragment_id', 0))
    result = _solve_local(arrays, metadata, dict(controls, global_seed=seed),
                          with_grad=with_grad, aux_offsets=aux_offsets)
    validate_full_result(result, arrays, with_grad=with_grad)
    result['metadata'] = dict(result.get('metadata', {}), **metadata,
                              global_seed=int(controls.get('global_seed', 0)),
                              effective_seed=seed, cotangent_seed=1.0)
    return result


@dataclass(frozen=True)
class _MatrixReplay:
    """Owned, process-local plans; numerical tensors and AD tapes are not saved."""
    fingerprint: str
    aux_offsets: np.ndarray
    batches: tuple
    native_version: str
    energy_reverse_workspace_bytes: int = 0
    memory_warned: set = field(default_factory=set, compare=False, repr=False)


def _matrix_kernel_version(extension):
    version = getattr(extension, "matrix_kernel_version", None)
    if not isinstance(version, str) or not version:
        raise ImportError("rebuild the STC extension with native matrix v2 support")
    return version


def _matrix_targets(metadata, nocc):
    if not isinstance(metadata, dict):
        raise ValueError("matrix metadata must be a dictionary")
    targets = metadata.get("target_indices")
    if not isinstance(targets, (tuple, list)) or not targets:
        raise ValueError("target_indices must be a nonempty ordered list or tuple")
    if any(not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_))
           or i < 0 or i >= nocc for i in targets):
        raise ValueError("invalid selected occupied index")
    if len(set(targets)) != len(targets):
        raise ValueError("target_indices must be distinct")
    return tuple(int(i) for i in targets)


def _matrix_aux_offsets(naux, offsets, group_size):
    if offsets is None:
        offsets = [*range(0, naux, group_size), naux] if naux else [0, 0]
    offsets = np.asarray(offsets)
    if (offsets.ndim != 1 or offsets.dtype.kind not in "iu" or len(offsets) < 2
            or offsets[0] != 0 or offsets[-1] != naux
            or np.any(offsets[1:] < offsets[:-1])):
        raise ValueError("aux_offsets must be nondecreasing integers spanning [0,naux]")
    saved = np.array(offsets, dtype=np.int64, copy=True)
    saved.flags.writeable = False
    return saved


def _matrix_batch_controls(settings, batch_count, seed):
    """Independent batch sums share the requested total variance budget."""
    local = {key: value for key, value in settings.items()
             if key not in ("target_batch_size", "memory_limit_mb", "auxiliary_group_size")}
    local["global_seed"] = seed
    if local["mode"] == "stochastic":
        local["matrix_tolerance"] /= math.sqrt(batch_count)
        if "energy_tolerance" in local:
            local["energy_tolerance"] /= math.sqrt(batch_count)
    return local


def _check_matrix_memory(inputs, ntarget, settings, offsets, *,
                         matrix_kind="W", stage="forward", warned=None,
                         energy_reverse_workspace_bytes=0, output_count=1):
    """Check each staged call against current residency before tensor copies."""
    from ._memory import observe_memory
    from .controls import estimate_matrix_memory, check_solver_memory

    naux, nocc, nvir = inputs["B"].shape
    estimate = estimate_matrix_memory(naux, nocc, nvir, ntarget, settings,
                                      matrix_kind=matrix_kind, with_grad=True,
                                      auxiliary_group_count=len(offsets) - 1)
    if energy_reverse_workspace_bytes:
        estimate["energy_reverse_updates"] = energy_reverse_workspace_bytes
        estimate["total_bytes"] += energy_reverse_workspace_bytes
    if output_count > 1:
        # Per-target native outputs, Python results/SEs and reverse seed copies.
        extra = 4 * (output_count - 1) * 8 * (nocc*nocc + nvir*nvir)
        estimate['orbital_outputs'] = extra
        estimate['total_bytes'] += extra
    observation = observe_memory()
    resident_mb = observation.rss_mb
    peak_bytes = (None if resident_mb is None else
                  estimate["total_bytes"] + int(resident_mb*1e6))
    # Every call creates occupied/virtual output matrices. Nonempty native
    # calls unconditionally copy all three inputs into owned Armadillo arrays;
    # reverse also creates input-shaped bars, including on the empty fast path.
    # Count the largest single new allocation, never original input residency
    # or the conservative estimator's safety factors, as a proven lower bound.
    names = ARRAY_NAMES if (stage == "reverse" or (naux and nvir)) else ("foo", "fvv")
    unavoidable_bytes = max(
        math.prod(inputs[name].shape) * np.dtype(inputs[name].dtype).itemsize
        for name in names)
    forecast = estimate if peak_bytes is None else {"total_bytes": peak_bytes}
    check_solver_memory(forecast, settings.get("memory_limit_mb"),
                        stage=f"matrix {matrix_kind} {stage}",
                        estimate_kind="workspace" if peak_bytes is None else "peak",
                        unavoidable_bytes=unavoidable_bytes, warned=warned,
                        observation=observation)
    return estimate, resident_mb, peak_bytes


def solve_matrices(inputs, metadata, controls, *, matrix_kind="W", aux_offsets=None):
    """Return native energy and matrix blocks with a delayed arbitrary-seed reverse.

    Blocks are sums over selected targets in exactly the input frame. W's
    sampling allocation controls matrix error. Its native energy is the half
    sum of occupied and virtual traces, with their covariance included in SE.
    D returns the density blocks and allocates stochastic sampling for the
    selected MP2 energy separately.
    CCSD/(T) consumes this forward result for its MP2 baseline and LIS response.
    Standalone MP2 uses the scalar solve interface instead.
    """
    from .controls import normalize_matrix_controls

    settings = normalize_matrix_controls(controls, matrix_kind)
    validate_full_inputs(inputs)
    naux, nocc, nvir = inputs["B"].shape
    targets = _matrix_targets(metadata, nocc)
    offsets = _matrix_aux_offsets(naux, aux_offsets, settings["auxiliary_group_size"])
    memory_warned = set()
    estimate, resident_mb, peak_bytes = _check_matrix_memory(
        inputs, len(targets), settings, offsets, matrix_kind=matrix_kind,
        warned=memory_warned)
    # Check before making host/layout copies or entering the native allocator.
    arrays = host_inputs(inputs)
    saved_metadata = deepcopy(metadata)
    saved_metadata["target_indices"] = targets
    size = settings["target_batch_size"] or len(targets)
    target_batches = [targets[first:first+size] for first in range(0, len(targets), size)]
    if not naux or not nvir:
        target_batches = []
    seeds = [_stream_seed(settings["global_seed"], metadata.get("fragment_id", 0), batch)
             for batch in range(len(target_batches))]
    saved_metadata["effective_seeds"] = seeds
    extension = (importlib.import_module("pyscfad.dlno_stc._stc_mp2")
                 if target_batches else None)
    native_version = _matrix_kernel_version(extension) if extension else "empty_matrix_v2"
    saved_metadata["native_version"] = native_version
    saved_metadata["native_options"] = []
    result = {"energy": np.float64(0.), "occupied": np.zeros((nocc, nocc)),
              "virtual": np.zeros((nvir, nvir)), "matrix_kind": matrix_kind,
              "metadata": saved_metadata, "controls": deepcopy(settings)}
    plans, batch_diagnostics = [], []
    native_timings = {}
    variance = {name: 0. for name in ("energy", "occupied", "virtual")}
    matrix_variance = {name: np.zeros_like(result[name]) for name in ("occupied", "virtual")}
    production_draws = 0
    energy_reverse_workspace_bytes = 0
    native_kwargs = {"aux_offsets": offsets}
    if matrix_kind == "D":
        native_kwargs["matrix_kind"] = matrix_kind
    for targets_in_batch, seed in zip(target_batches, seeds):
        native_controls = _matrix_batch_controls(settings, len(target_batches), seed)
        local = extension.solve_matrices(*(arrays[key] for key in ARRAY_NAMES),
                                         targets_in_batch, native_controls, **native_kwargs)
        validate_matrix_result(local, arrays, matrix_kind=matrix_kind)
        for name in ("energy", "occupied", "virtual"):
            result[name] += local[name]
        diagnostics = dict(local["diagnostics"])
        for name, seconds in diagnostics.get("native_timings", {}).items():
            native_timings[name] = native_timings.get(name, 0.) + float(seconds)
        if diagnostics["native_version"] != native_version:
            raise ValueError("matrix result native version changed during forward")
        saved_metadata["native_options"].append(deepcopy(diagnostics["effective_options"]))
        for name in ("energy", "occupied", "virtual"):
            key = "energy_standard_error" if name == "energy" else name+"_frobenius_se"
            variance[name] += float(diagnostics[key])**2
        for name in ("occupied", "virtual"):
            matrix_variance[name] += np.square(diagnostics.pop(name+"_standard_error"))
        if settings["mode"] == "stochastic":
            diagnostics["matrix_tolerance"] = native_controls["matrix_tolerance"]
        production_draws += int(diagnostics.get("production_draws", 0))
        # Keep summaries and the small replay, never each batch's tensor outputs.
        batch_diagnostics.append(diagnostics)
        plan = local["replay"]
        plans.append(plan)
        if matrix_kind == "D" and settings["mode"] == "stochastic":
            # Batches replay in sequence. Use the owned native plan, whose
            # sample allocations cannot be changed by editing diagnostics.
            energy_reverse_workspace_bytes = max(energy_reverse_workspace_bytes,
                                                 plan.energy_reverse_workspace_bytes)
    diagnostics = {
        "mode": settings["mode"], "matrix_kind": matrix_kind,
        "batch_count": len(plans), "batch_seeds": seeds,
        "batches": batch_diagnostics, "production_draws": production_draws,
        "memory_estimate_bytes": estimate, "current_rss_mb": resident_mb,
        "estimated_process_peak_mb": None if peak_bytes is None else peak_bytes/1e6,
        "uncertainty_convention": "independent_batch_sum",
        "energy_standard_error": math.sqrt(variance["energy"]),
        "native_timings": native_timings,
    }
    for name in ("occupied", "virtual"):
        se = math.sqrt(variance[name])
        diagnostics[name+"_frobenius_se"] = se
        diagnostics[name+"_standard_error"] = np.sqrt(matrix_variance[name])
        diagnostics[name+"_target_met"] = (
            settings["mode"] == "deterministic" or se <= settings["matrix_tolerance"])
    result["diagnostics"] = diagnostics
    if native_timings:
        resource_profile.checkpoint('stc.matrix.native_forward_timings',
            frag=metadata.get('fragment_id', 0) + 1, timings=native_timings)
    # Fingerprint resolved source defaults as well as the requested controls.
    fingerprint = matrix_request_fingerprint(arrays, saved_metadata, settings,
                                             offsets, matrix_kind=matrix_kind)
    result["replay"] = _MatrixReplay(fingerprint, offsets, tuple(plans), native_version,
                                     energy_reverse_workspace_bytes, memory_warned)
    validate_matrix_result(result, arrays)
    return result


@dataclass(frozen=True)
class OrbitalMatrices:
    """Per-orbital W outputs and small replays sharing one domain's inputs."""
    forwards: tuple
    metadata: dict
    controls: dict
    aux_offsets: np.ndarray
    fingerprint: str
    batches: tuple


def solve_orbital_matrices(inputs, fragments, controls):
    """Construct every singleton W before CC, sharing deterministic MP2 work.

    Stochastic matrices retain independent target streams and their existing
    replay rules. Neither path constructs a summed atom W. Target batching
    bounds amplitude storage without changing the deterministic functional.
    """
    from .controls import normalize_matrix_controls
    settings = normalize_matrix_controls(controls)
    arrays = host_inputs(inputs)
    fragments = tuple(fragments)
    targets = tuple(t for _, selected in fragments for t in selected)
    if any(len(selected) != 1 for _, selected in fragments):
        raise ValueError('orbital matrices require singleton targets')
    metadata = dict(target_indices=targets, fragment_ids=tuple(i for i, _ in fragments))
    _matrix_targets(metadata, arrays['foo'].shape[0])
    offsets = _matrix_aux_offsets(arrays['B'].shape[0], None, settings['auxiliary_group_size'])
    fingerprint = matrix_request_fingerprint(arrays, metadata, settings, offsets)
    if settings['mode'] == 'stochastic' or not arrays['B'].size:
        forwards = tuple(solve_matrices(arrays, dict(fragment_id=i, target_indices=selected),
                                        settings, aux_offsets=offsets) for i, selected in fragments)
        return OrbitalMatrices(forwards, metadata, settings, offsets, fingerprint, ())
    extension = importlib.import_module('pyscfad.dlno_stc._stc_mp2')
    if not hasattr(extension, 'solve_orbital_matrices'):
        raise ImportError('rebuild the STC extension for atom_singleton orbital W support')
    size = settings['target_batch_size'] or len(targets)
    forwards, batches = [], []
    for first in range(0, len(targets), size):
        selected = targets[first:first+size]
        _check_matrix_memory(arrays, len(selected), settings, offsets, output_count=len(selected))
        native_controls = _matrix_batch_controls(settings, 1, settings['global_seed'])
        local = extension.solve_orbital_matrices(*(arrays[key] for key in ARRAY_NAMES),
            selected, native_controls, aux_offsets=offsets)
        if len(local) != len(selected):
            raise ValueError('native orbital matrix count mismatch')
        batches.append((first, len(local), local[0]['replay']))
        for slot, forward in enumerate(local, first):
            validate_matrix_result(forward, arrays)
            forward['metadata'] = dict(fragment_id=fragments[slot][0],
                                       target_indices=(targets[slot],))
            forward['controls'] = settings
            forward['matrix_kind'] = 'W'
            forwards.append(forward)
    return OrbitalMatrices(tuple(forwards), metadata, settings, offsets, fingerprint, tuple(batches))


def pullback_orbital_matrices(inputs, batch, matrix_bars):
    """Consume labeled W seeds after all CC reverses; return one input bar tree.

    Every target contributes its MP2 energy once. Arbitrary W seeds stay
    separate until their target-specific contractions have been reversed.
    """
    if not isinstance(batch, OrbitalMatrices):
        raise TypeError('orbital reverse requires its forward batch')
    if len(matrix_bars) != len(batch.forwards):
        raise ValueError('orbital matrix cotangent count mismatch')
    arrays = host_inputs(inputs)
    if matrix_request_fingerprint(arrays, batch.metadata, batch.controls,
                                  batch.aux_offsets) != batch.fingerprint:
        raise ValueError('orbital matrix replay inputs or controls changed')
    for forward, bars in zip(batch.forwards, matrix_bars):
        validate_matrix_result(forward, arrays)
        if set(bars) != {'occupied', 'virtual'}:
            raise ValueError('orbital seeds must contain occupied and virtual')
        for name, source in (('occupied', 'foo'), ('virtual', 'fvv')):
            value = np.asarray(bars[name])
            if (value.dtype != np.float64 or value.shape != arrays[source].shape
                    or not np.isfinite(value).all()):
                raise ValueError('orbital W cotangents must be finite float64 matrices')
    total = None
    if not batch.batches:
        packets = (pullback_matrices(arrays, forward, energy_bar=1.,
                    occupied_bar=bars['occupied'], virtual_bar=bars['virtual'])
                   for forward, bars in zip(batch.forwards, matrix_bars))
    else:
        extension = importlib.import_module('pyscfad.dlno_stc._stc_mp2')
        def native_packets():
            for first, count, replay in batch.batches:
                _check_matrix_memory(arrays, count, batch.controls, batch.aux_offsets,
                                     stage='reverse', output_count=count)
                seeds = matrix_bars[first:first+count]
                yield extension.pullback_orbital_matrices(
                    *(arrays[key] for key in ARRAY_NAMES), replay, [1.] * count,
                    [np.ascontiguousarray(b['occupied']) for b in seeds],
                    [np.ascontiguousarray(b['virtual']) for b in seeds])
        packets = native_packets()
    for bars in packets:
        validate_input_cotangents(bars, arrays)
        if total is None:
            total = {key: value if value.flags.writeable else value.copy()
                     for key, value in bars.items()}
        else:
            for key in ARRAY_NAMES:
                total[key] += bars[key]
    return total


def pullback_matrices(inputs, forward, *, energy_bar, occupied_bar, virtual_bar):
    """Replay production plans and return already-seeded foo/fvv/B bars."""
    validate_full_inputs(inputs)
    validate_matrix_result(forward, inputs)
    replay = forward["replay"]
    if not isinstance(replay, _MatrixReplay):
        raise ValueError("matrix replay must come from solve_matrices in this process")
    matrix_kind = forward.get("matrix_kind")
    fingerprint = matrix_request_fingerprint(
        inputs, forward["metadata"], forward["controls"],
        replay.aux_offsets, matrix_kind=matrix_kind)
    if fingerprint != replay.fingerprint:
        raise ValueError("matrix replay fingerprint changed: inputs, frame, kind or controls differ")
    extension = (importlib.import_module("pyscfad.dlno_stc._stc_mp2")
                 if replay.batches else None)
    if extension and _matrix_kernel_version(extension) != replay.native_version:
        raise ValueError("matrix replay native version differs from the loaded extension")
    seed = np.asarray(energy_bar)
    if seed.shape != () or not np.isfinite(seed):
        raise ValueError("energy_bar must be a finite scalar")
    # Downstream work may have allocated memory since the forward call.
    _check_matrix_memory(inputs, len(forward["metadata"]["target_indices"]),
                         forward["controls"], replay.aux_offsets,
                         matrix_kind=matrix_kind, stage="reverse",
                         warned=replay.memory_warned,
                         energy_reverse_workspace_bytes=(
                             replay.energy_reverse_workspace_bytes if seed != 0. else 0))
    seeds = {"occupied": np.asarray(occupied_bar), "virtual": np.asarray(virtual_bar)}
    for name, source in (("occupied", "foo"), ("virtual", "fvv")):
        value = seeds[name]
        if (value.shape != inputs[source].shape or value.dtype != np.dtype("float64")
                or not np.isfinite(value).all()):
            raise ValueError(f"{name}_bar must be a finite float64 matrix in the input frame")
        seeds[name] = np.ascontiguousarray(value)
    if not replay.batches:
        return {name: np.zeros_like(inputs[name]) for name in ARRAY_NAMES}
    arrays = host_inputs(inputs)
    total = None
    native_timings = {}
    timed_pullback = getattr(extension, 'pullback_matrices_with_timings', None)
    for plan in replay.batches:
        arguments = (*(arrays[key] for key in ARRAY_NAMES), plan,
                     float(seed), seeds["occupied"], seeds["virtual"])
        if timed_pullback is None:
            bars = extension.pullback_matrices(*arguments)
        else:
            bars, timings = timed_pullback(*arguments)
            for name, seconds in timings.items():
                native_timings[name] = native_timings.get(name, 0.) + float(seconds)
        validate_input_cotangents(bars, arrays)
        if total is None:
            total = bars
        else:
            for name in ARRAY_NAMES:
                total[name] += bars[name]
        # Only the accumulated result is needed by the next native batch.
        # Keeping the last batch here overlaps another full B cotangent with
        # the following call (hundreds of MB for large local domains).
        del bars
    forward['diagnostics']['native_reverse_timings'] = native_timings
    if native_timings:
        resource_profile.checkpoint('stc.matrix.native_reverse_timings',
            frag=forward['metadata'].get('fragment_id', 0) + 1, timings=native_timings)
    return total


def begin_sequential_matrices(inputs, metadata, controls):
    """Occupied W and a call-local workspace for occupied-conditioned virtual W.

    Deterministic energy-only experiment. All target amplitudes are retained
    until their common occupied space has been selected; no DF or Laplace
    work is repeated in the second stage. The energy remains full-ED MP2.
    """
    from .controls import normalize_matrix_controls
    from ._memory import assess_memory
    settings = normalize_matrix_controls(controls)
    if settings['mode'] != 'deterministic':
        raise NotImplementedError('sequential_virtual requires deterministic matrix controls')
    validate_full_inputs(inputs)
    naux, no, nv = inputs['B'].shape
    targets = _matrix_targets(metadata, no)
    # T and spin(G) for all physical targets plus one projected pair, the DF
    # layout and small W matrices. Unlike the ordinary streaming path this
    # intentionally retains all target slices until occupied selection.
    retained_bytes = (16 * (len(targets)+1) * no
                      + 8 * min(no, settings['threads'])) * nv * nv
    assess_memory('sequential_w', (retained_bytes + 16*naux*no*nv)/1e6,
                  configured_mb=settings.get('memory_limit_mb'),
                  unavoidable_bytes=retained_bytes)
    arrays = host_inputs(inputs)
    extension = importlib.import_module('pyscfad.dlno_stc._stc_mp2')
    native_controls = _matrix_batch_controls(settings, 1, settings['global_seed'])
    with blas_threads(1):
        result = extension.begin_sequential_matrices(
            *(arrays[k] for k in ARRAY_NAMES), targets, native_controls)
    result['matrix_kind'] = 'W'
    return result


def finish_sequential_virtual(forward, occupied_rotation):
    """Consume the cached amplitudes/Grams using an orthonormal ED rotation."""
    from time import perf_counter
    extension = importlib.import_module('pyscfad.dlno_stc._stc_mp2')
    started = perf_counter()
    with blas_threads(1):
        result = extension.finish_sequential_virtual(
            forward.pop('workspace'), np.ascontiguousarray(occupied_rotation, dtype=np.float64))
    forward['diagnostics'].update(retained_occupied=occupied_rotation.shape[1],
                                 sequential_virtual_seconds=perf_counter()-started)
    return result
