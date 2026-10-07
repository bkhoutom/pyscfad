"""Optional in-memory STC solver and complete independent MPI replicas."""

import importlib
import math

import numpy as np

from .protocol import (
    ARRAY_NAMES, WEIGHTED_ARRAY_NAMES, validate_full_inputs, validate_full_result,
    validate_weighted_inputs, validate_weighted_result,
)


def _scope_boundary(scope):
    if scope == "domain":
        return WEIGHTED_ARRAY_NAMES, validate_weighted_inputs, validate_weighted_result
    if scope == "system":
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
    solver = extension.solve if scope == "domain" else extension.solve_full
    return solver(
        *(inputs[key] for key in names), controls,
        aux_offsets=aux_offsets, with_grad=with_grad,
    )


def solve(inputs, metadata, controls, *, comm=None, with_grad=True, aux_offsets=None,
          scope="domain"):
    """Average full energy/bar replicas; only communicator root returns data.

    Local import, conversion, numerical, and result-validation errors are
    collected before any numeric reductions, so every rank raises together.
    MPI transport failures remain the calling application's responsibility.
    """
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
            "energy_kind": ("weighted_laplace_mp2_v1" if scope == "domain"
                            else "full_laplace_mp2_v1"), "cotangent_seed": 1.0,
            "fragment_id": int(metadata.get("fragment_id", 0)),
            "replica_count": size, "global_seed": int(requested_seed),
            "derivative_convention": "fixed_sample",
        },
        "diagnostics": {"replica_seeds": replica_seeds,
                        "replicas": replica_diagnostics},
    }
    if with_grad:
        for value in bars.values():
            value /= size
        result["cotangents"] = bars
    return result
