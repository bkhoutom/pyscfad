"""Host-side collective call for one domain on a supplied communicator."""

import numpy as np


_KEYS = ("foo", "fvv", "B")


def _preflight(inputs, metadata, controls, backend):
    if inputs is None:
        raise ValueError("root input is missing")
    if not callable(backend):
        raise ValueError("backend must be callable")
    if not isinstance(metadata, dict) or not isinstance(controls, dict):
        raise ValueError("metadata and controls must be dictionaries")
    if set(inputs) != set(_KEYS):
        raise ValueError("inputs must contain foo, fvv, B")
    foo, fvv, B = (np.asarray(inputs[key]) for key in _KEYS)
    no, nv = foo.shape[0], fvv.shape[0]
    if foo.shape != (no, no) or fvv.shape != (nv, nv) or B.ndim != 3 or B.shape[1:] != (no, nv):
        raise ValueError("invalid input shapes")
    for key in _KEYS:
        array = np.asarray(inputs[key])
        if array.dtype != np.float64 or not np.all(np.isfinite(array)):
            raise ValueError(f"{key} must contain finite float64 values")
    for key in ("foo", "fvv"):
        array = np.asarray(inputs[key])
        if not np.allclose(array, array.T, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{key} must be symmetric")
    if "sample_blocks" in controls:
        n = controls["sample_blocks"]
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("sample_blocks must be a positive integer")


def run_backend(inputs, metadata, controls, backend, *, comm=None):
    """Broadcast one domain packet, call backend collectively, return on root.

    The backend owns reductions and normalization. Its collective failures
    require application-level MPI abort; preflight failures are propagated
    before any numeric broadcast. MPI is imported only for a supplied comm.
    """
    if comm is None:
        _preflight(inputs, metadata, controls, backend)
        return backend(inputs, metadata, controls, comm=None)

    from mpi4py import MPI
    if MPI.Query_thread() < MPI.THREAD_FUNNELED:
        raise RuntimeError("MPI_THREAD_FUNNELED support is required")
    rank = comm.Get_rank()
    backend_status = comm.allgather(callable(backend))
    if not all(backend_status):
        bad_ranks = [index for index, valid in enumerate(backend_status) if not valid]
        raise ValueError(f"collective preflight failed: backend is not callable on ranks {bad_ranks}")
    if rank == 0:
        try:
            _preflight(inputs, metadata, controls, backend)
            header = (None, {key: np.asarray(inputs[key]).shape for key in _KEYS},
                      metadata, controls)
        except Exception as exc:
            header = (f"{type(exc).__name__}: {exc}", None, None, None)
    else:
        header = None
    error, shapes, shared_metadata, shared_controls = comm.bcast(header, root=0)
    if error is not None:
        raise ValueError(f"collective preflight failed: {error}")

    shared_inputs = {}
    for key in _KEYS:
        if rank == 0:
            array = np.ascontiguousarray(inputs[key], dtype=np.float64)
        else:
            array = np.empty(shapes[key], dtype=np.float64)
        comm.Bcast(array, root=0)
        shared_inputs[key] = array
    result = backend(shared_inputs, shared_metadata, shared_controls, comm=comm)
    return result if rank == 0 else None
