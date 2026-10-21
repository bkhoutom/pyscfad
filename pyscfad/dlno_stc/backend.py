"""Optional in-memory STC solver and complete independent MPI replicas."""

import importlib
import math
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .protocol import (
    ARRAY_NAMES, validate_full_inputs, validate_full_result,
    matrix_request_fingerprint, validate_matrix_result, validate_input_cotangents,
)


def _scope_boundary(scope):
    if scope in ("domain", "system"):
        return ARRAY_NAMES, validate_full_inputs, validate_full_result
    raise ValueError("scope must be domain or system")


def host_inputs(inputs, *, scope="domain"):
    """Transfer each preparation output once, retaining its float64 dtype."""
    names, validate, _ = _scope_boundary(scope)
    if not isinstance(inputs, dict) or set(inputs) != set(names):
        raise ValueError("inputs must contain exactly " + ", ".join(names))
    arrays = {key: np.ascontiguousarray(inputs[key]) for key in names}
    validate(arrays)
    return arrays


def _replica_seed(global_seed, fragment_id, rank):
    for name, value in (("global_seed", global_seed), ("fragment_id", fragment_id)):
        if (not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_)) or value < 0):
            raise ValueError(f"{name} must be a nonnegative integer")
    state = np.random.SeedSequence([int(global_seed), int(fragment_id), int(rank)])
    return int(state.generate_state(1, dtype=np.uint64)[0])


def _solve_local(inputs, metadata, controls, *, with_grad, aux_offsets, scope="domain"):
    try:
        extension = importlib.import_module("pyscfad.dlno_stc._stc_mp2")
    except ImportError as exc:
        raise ImportError(
            "STC-MP2 extension is unavailable; build pyscfad/dlno_stc/cpp "
            "with CMake as described in pyscfad/dlno_stc/README.md"
        ) from exc
    names, _, _ = _scope_boundary(scope)
    solver = extension.solve_selected if scope == "domain" else extension.solve_full
    arguments = (tuple(metadata["target_indices"]), controls) if scope == "domain" else (controls,)
    return solver(
        *(inputs[key] for key in names), *arguments,
        aux_offsets=aux_offsets, with_grad=with_grad,
    )


def solve(inputs, metadata, controls, *, comm=None, with_grad=True, aux_offsets=None,
          scope="domain"):
    """Average full energy/bar replicas; only communicator root returns data.

    Local import, conversion, numerical, and result-validation errors are
    collected before any numeric reductions, so every rank raises together.
    MPI transport failures remain the calling application's responsibility.
    A single-rank communicator needs no reductions and follows the serial path.
    """
    if comm is not None and comm.Get_size() == 1:
        comm = None
    rank = 0 if comm is None else comm.Get_rank()
    size = 1 if comm is None else comm.Get_size()
    try:
        if not isinstance(metadata, dict) or not isinstance(controls, dict):
            raise ValueError("metadata and controls must be dictionaries")
        names, _, validate = _scope_boundary(scope)
        arrays = host_inputs(inputs, scope=scope)
        if aux_offsets is None:
            group_size = controls.get("auxiliary_group_size", 32)
            if (not isinstance(group_size, (int, np.integer))
                    or isinstance(group_size, (bool, np.bool_)) or group_size <= 0):
                raise ValueError("auxiliary_group_size must be a positive integer")
            naux = arrays["B"].shape[0]
            aux_offsets = np.asarray([*range(0, naux, int(group_size)), naux],
                                     dtype=np.int64)
        requested_seed = controls.get("global_seed", 0)
        effective_seed = _replica_seed(requested_seed, metadata.get("fragment_id", 0), rank)
        settings = dict(controls, global_seed=effective_seed)
        scope_options = {} if scope == "domain" else {"scope": scope}
        local = _solve_local(arrays, metadata, settings,
                             with_grad=with_grad, aux_offsets=aux_offsets, **scope_options)
        validate(local, arrays, with_grad=with_grad)
        local_energy = np.asarray(local["energy"], dtype=np.float64).reshape(())
        local_variance = np.asarray(float(local["energy_standard_error"]) ** 2,
                                    dtype=np.float64).reshape(())
        if not np.isfinite(local_variance):
            raise ValueError("replica energy variance overflowed")
        bars = ({key: np.ascontiguousarray(local["cotangents"][key])
                 for key in names} if with_grad else {})
        diagnostics = dict(local.get("diagnostics", {}))
        error = None
    except Exception as exc:
        if comm is None:
            raise
        error = f"rank {rank}: {type(exc).__name__}: {exc}"

    if comm is not None:
        failures = [message for message in comm.allgather(error) if message is not None]
        if failures:
            raise RuntimeError("STC replica failed: " + "; ".join(failures))
        from mpi4py import MPI

        # Root's local arrays become the receive buffers, avoiding a second
        # complete bar_B allocation solely for replica reduction.
        for value in (local_energy, local_variance, *bars.values()):
            if rank == 0:
                comm.Reduce(MPI.IN_PLACE, value, op=MPI.SUM, root=0)
            else:
                comm.Reduce(value, None, op=MPI.SUM, root=0)
        replica_diagnostics = comm.gather(diagnostics, root=0)
        replica_seeds = comm.gather(effective_seed, root=0)
        if rank != 0:
            return None
    else:
        replica_diagnostics, replica_seeds = [diagnostics], [effective_seed]

    result = {
        "energy": np.float64(local_energy / size),
        "energy_standard_error": np.float64(math.sqrt(float(local_variance)) / size),
        "metadata": {
            "energy_kind": ("boys_target_laplace_mp2_v1" if scope == "domain"
                            else "full_laplace_mp2_v1"), "cotangent_seed": 1.0,
            "target_indices": metadata.get("target_indices"),
            "fragment_id": int(metadata.get("fragment_id", 0)),
            "replica_count": size, "global_seed": int(requested_seed),
            "derivative_convention": "fixed_sample",
        },
        "diagnostics": {"target_indices": metadata.get("target_indices"),
                        "energy_tolerance": controls.get("energy_tolerance"),
                        "sampling_mode": ("fixed_count" if "production_samples" in controls
                                          else controls.get("mode", "deterministic")),
                        "replica_seeds": replica_seeds,
                        "replicas": replica_diagnostics},
    }
    if "request_fingerprint" in metadata:
        result["metadata"] = dict(metadata, **{
            "cotangent_seed": 1.0, "backend": "stc_selected", "backend_version": "1",
            "actual_sample_count": sum(int(d.get("actual_sample_count", 0))
                                       for d in replica_diagnostics),
            "derivative_convention": "fixed_sample",
            "seed_replay": {"global_seed": int(requested_seed),
                            "replica_seeds": replica_seeds},
            "replica_count": size,
        })
    for name in ("frame_convention", "global_target_orbitals", "candidate_order",
                 "virtual_anchor_parent_columns", "grouping_fingerprint", "budget"):
        if name in metadata:
            result["metadata"][name] = metadata[name]
    if with_grad:
        for value in bars.values():
            value /= size
        result["cotangents"] = bars
    return result


@dataclass(frozen=True)
class _MatrixReplay:
    """Owned, process-local plans; numerical tensors and AD tapes are not saved."""
    fingerprint: str
    aux_offsets: np.ndarray
    batches: tuple
    native_version: str


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


def _check_matrix_memory(inputs, ntarget, settings, matrix_kind, offsets):
    """Check each staged call against current residency before tensor copies."""
    from pyscf import lib
    from .controls import estimate_matrix_memory, check_solver_memory

    naux, nocc, nvir = inputs["B"].shape
    estimate = estimate_matrix_memory(naux, nocc, nvir, ntarget, settings,
                                      matrix_kind=matrix_kind, with_grad=True,
                                      auxiliary_group_count=len(offsets)-1)
    resident_mb = float(lib.current_memory()[0])
    peak_bytes = estimate["total_bytes"] + int(resident_mb*1e6)
    check_solver_memory({"total_bytes": peak_bytes}, settings.get("memory_limit_mb"))
    return estimate, resident_mb, peak_bytes


def solve_matrices(inputs, metadata, controls, *, matrix_kind="D", aux_offsets=None):
    """Return native E and D/W blocks, with a delayed arbitrary-seed reverse.

    Blocks are sums over selected targets in exactly the input frame. W's
    sampling allocation controls matrix error. Its native energy is the half
    sum of occupied and virtual traces, with their covariance included in SE.
    This host call is deliberately separate from the energy-only MPI solver.
    """
    from .controls import normalize_matrix_controls

    settings = normalize_matrix_controls(controls, matrix_kind)
    validate_full_inputs(inputs)
    naux, nocc, nvir = inputs["B"].shape
    targets = _matrix_targets(metadata, nocc)
    offsets = _matrix_aux_offsets(naux, aux_offsets, settings["auxiliary_group_size"])
    estimate, resident_mb, peak_bytes = _check_matrix_memory(
        inputs, len(targets), settings, matrix_kind, offsets)
    # Check before making host/layout copies or entering the native allocator.
    arrays = host_inputs(inputs)
    saved_metadata = deepcopy(metadata)
    saved_metadata["target_indices"] = targets
    size = settings["target_batch_size"]
    target_batches = [targets[first:first+size] for first in range(0, len(targets), size)]
    if not naux or not nvir:
        target_batches = []
    seeds = [_replica_seed(settings["global_seed"], metadata.get("fragment_id", 0), batch)
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
    variance = {name: 0. for name in ("energy", "occupied", "virtual")}
    matrix_variance = {name: np.zeros_like(result[name]) for name in ("occupied", "virtual")}
    production_draws = 0
    for targets_in_batch, seed in zip(target_batches, seeds):
        native_controls = _matrix_batch_controls(settings, len(target_batches), seed)
        local = extension.solve_matrices(*(arrays[key] for key in ARRAY_NAMES),
                                         targets_in_batch, native_controls,
                                         matrix_kind=matrix_kind, aux_offsets=offsets)
        validate_matrix_result(local, arrays)
        for name in ("energy", "occupied", "virtual"):
            result[name] += local[name]
        diagnostics = dict(local["diagnostics"])
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
        plans.append(local["replay"])
    diagnostics = {
        "mode": settings["mode"], "matrix_kind": matrix_kind,
        "batch_count": len(plans), "batch_seeds": seeds,
        "batches": batch_diagnostics, "production_draws": production_draws,
        "memory_estimate_bytes": estimate, "current_rss_mb": resident_mb,
        "estimated_process_peak_mb": peak_bytes/1e6,
        "uncertainty_convention": "independent_batch_sum",
        "energy_standard_error": math.sqrt(variance["energy"]),
    }
    for name in ("occupied", "virtual"):
        se = math.sqrt(variance[name])
        diagnostics[name+"_frobenius_se"] = se
        diagnostics[name+"_standard_error"] = np.sqrt(matrix_variance[name])
        diagnostics[name+"_target_met"] = (
            settings["mode"] == "deterministic" or se <= settings["matrix_tolerance"])
    result["diagnostics"] = diagnostics
    # Fingerprint resolved source defaults as well as the requested controls.
    fingerprint = matrix_request_fingerprint(arrays, saved_metadata, settings,
                                             matrix_kind, offsets)
    result["replay"] = _MatrixReplay(fingerprint, offsets, tuple(plans), native_version)
    validate_matrix_result(result, arrays)
    return result


def pullback_matrices(inputs, forward, *, energy_bar, occupied_bar, virtual_bar):
    """Replay production plans and return already-seeded foo/fvv/B bars."""
    validate_full_inputs(inputs)
    validate_matrix_result(forward, inputs)
    replay = forward["replay"]
    if not isinstance(replay, _MatrixReplay):
        raise ValueError("matrix replay must come from solve_matrices in this process")
    fingerprint = matrix_request_fingerprint(
        inputs, forward["metadata"], forward["controls"],
        forward["matrix_kind"], replay.aux_offsets)
    if fingerprint != replay.fingerprint:
        raise ValueError("matrix replay fingerprint changed: inputs, frame or controls differ")
    extension = (importlib.import_module("pyscfad.dlno_stc._stc_mp2")
                 if replay.batches else None)
    if extension and _matrix_kernel_version(extension) != replay.native_version:
        raise ValueError("matrix replay native version differs from the loaded extension")
    # Downstream work may have allocated memory since the forward call.
    _check_matrix_memory(inputs, len(forward["metadata"]["target_indices"]),
                         forward["controls"], forward["matrix_kind"], replay.aux_offsets)
    seed = np.asarray(energy_bar)
    if seed.shape != () or not np.isfinite(seed):
        raise ValueError("energy_bar must be a finite scalar")
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
    for plan in replay.batches:
        bars = extension.pullback_matrices(*(arrays[key] for key in ARRAY_NAMES), plan,
                                           float(seed), seeds["occupied"], seeds["virtual"])
        validate_input_cotangents(bars, arrays)
        if total is None:
            total = bars
        else:
            for name in ARRAY_NAMES:
                total[name] += bars[name]
    return total
