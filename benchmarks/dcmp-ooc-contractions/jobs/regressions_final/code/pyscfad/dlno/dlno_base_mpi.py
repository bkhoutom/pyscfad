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

"""MPI communication, gauge checks, and cotangent reduction shared by DLNO."""

from __future__ import annotations

from functools import wraps
import hashlib
import traceback

import jax
import jax.numpy as np
from mpi4py import MPI
import numpy

from ._restart import df_source_fingerprint, scientific_digest
from .targets import semantic_tuple as _semantic_tuple, validate_static_target_options


def _abort_collective_on_error(function):
    """Prevent a rank-local Python error from stranding MPI peers."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        comm = kwargs.get("comm", MPI.COMM_WORLD)
        if comm is None:
            comm = MPI.COMM_WORLD
        root = int(kwargs.get("root", 0))
        try:
            return function(*args, **kwargs)
        except Exception:
            if comm.Get_size() > 1:  # pragma: no cover - MPI failure path
                if comm.Get_rank() == root:
                    traceback.print_exc()
                comm.Abort(1)
            raise

    return wrapped


def _progress_enabled(progress):
    """Validate and normalize the public progress-reporting switch."""
    if progress is None or progress is False:
        return False
    if progress is True or callable(progress):
        return True
    raise TypeError("progress must be a bool, callable, or None")


def _progress_reporter(progress, *, rank, root):
    """Return a rank-root-only, line-buffered progress reporter."""
    enabled = _progress_enabled(progress)
    if not enabled or rank != root:
        return None
    if callable(progress):
        return progress

    def report(message):
        print(message, flush=True)

    return report


def _zero_mf_cotangent(mf):
    """Construct a live MF-shaped zero tree for checkpoint deserialization."""

    dtype = np.asarray(mf.mo_coeff).dtype
    _, pullback = jax.vjp(
        lambda mf_: np.zeros((), dtype=dtype), mf
    )
    mf_bar, = pullback(np.ones((), dtype=dtype))
    return mf_bar


def _exception_text(stage):
    return f"{stage} failed on an MPI rank:\n{traceback.format_exc()}"


def _raise_if_root_failed(comm, error, *, root):
    error = comm.bcast(error, root=root)
    if error is not None:
        raise RuntimeError(error)


def _raise_if_any_rank_failed(comm, local_error):
    errors = comm.allgather(local_error)
    failures = [error for error in errors if error is not None]
    if failures:
        raise RuntimeError("\n".join(failures))


def _to_host_leaf(leaf):
    if leaf is None:
        return None
    if hasattr(leaf, "dtype") and leaf.dtype == jax.dtypes.float0:
        return leaf
    try:
        return numpy.array(numpy.asarray(leaf), copy=True, order="C")
    except (TypeError, ValueError):
        return leaf


def _to_device_leaf(leaf):
    if leaf is None:
        return None
    if isinstance(leaf, numpy.ndarray):
        return np.asarray(leaf)
    return leaf


_TREE_REDUCE_CHUNK_BYTES = 64 * 1024**2
_TREE_REDUCE_MAX_COUNT = int(numpy.iinfo(numpy.int32).max)


def _tree_reduce_leaf_metadata(leaf):
    """Describe one cotangent leaf without serializing its data."""
    if leaf is None:
        return ("none",)
    if hasattr(leaf, "dtype") and leaf.dtype == jax.dtypes.float0:
        return ("float0", tuple(int(value) for value in leaf.shape))
    try:
        if hasattr(leaf, "dtype") and hasattr(leaf, "shape"):
            dtype = numpy.dtype(leaf.dtype)
            shape = tuple(int(value) for value in leaf.shape)
        else:
            array = numpy.asarray(leaf)
            dtype = array.dtype
            shape = tuple(int(value) for value in array.shape)
    except (TypeError, ValueError):
        return ("unsupported", type(leaf).__name__)
    if dtype.hasobject or dtype.kind not in "iufc":
        return ("unsupported", dtype.str)
    return ("numeric", shape, dtype.str)


def _tree_reduce_plans(paths, metadata_by_rank):
    """Validate collective leaf metadata and return reduction plans."""
    plans = {}
    for path in paths:
        descriptors = [metadata[path] for metadata in metadata_by_rank]
        unsupported = [
            (rank, descriptor)
            for rank, descriptor in enumerate(descriptors)
            if descriptor[0] == "unsupported"
        ]
        if unsupported:
            rank, descriptor = unsupported[0]
            raise TypeError(
                f"MPI cotangent leaf {path} on rank {rank} has unsupported "
                f"type {descriptor[1]}"
            )

        numeric = [
            descriptor for descriptor in descriptors
            if descriptor[0] == "numeric"
        ]
        if numeric:
            reference = numeric[0]
            if any(descriptor != reference for descriptor in numeric[1:]):
                raise RuntimeError(
                    f"MPI cotangent leaf {path} has inconsistent numeric "
                    f"shape or dtype across ranks: {descriptors}"
                )
            shape = reference[1]
            if any(
                descriptor[0] == "float0" and descriptor[1] != shape
                for descriptor in descriptors
            ):
                raise RuntimeError(
                    f"MPI cotangent leaf {path} has inconsistent zero and "
                    f"numeric shapes across ranks: {descriptors}"
                )
            plans[path] = reference
            continue

        float0 = [
            descriptor for descriptor in descriptors
            if descriptor[0] == "float0"
        ]
        if float0:
            reference = float0[0]
            if any(descriptor != reference for descriptor in float0[1:]):
                raise RuntimeError(
                    f"MPI cotangent leaf {path} has inconsistent float0 "
                    f"shapes across ranks: {descriptors}"
                )
            plans[path] = reference
        else:
            plans[path] = ("none",)
    return plans


def _tree_sum_to_root(comm, tree, *, root=0):
    """Sum numeric leaves of a JAX pytree onto ``root``.

    Paths, rather than registered object identities, align leaves across the
    independently constructed rank-0 and worker ``mf`` objects. Numeric
    leaves use bounded buffer reductions so the root never gathers every
    rank's serialized tree at once.
    """
    if comm.Get_size() == 1:
        return tree
    rank = comm.Get_rank()
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(
        tree, is_leaf=lambda value: value is None
    )
    paths = [jax.tree_util.keystr(path) for path, _ in leaves_with_path]
    if len(paths) != len(set(paths)):
        raise RuntimeError(
            f"MPI cotangent tree on rank {rank} contains duplicate paths"
        )
    all_paths = comm.allgather(tuple(paths))
    root_path_set = set(all_paths[root])
    if any(set(other) != root_path_set for other in all_paths):
        mismatch = next(
            index for index, other in enumerate(all_paths)
            if set(other) != root_path_set
        )
        other_paths = set(all_paths[mismatch])
        raise RuntimeError(
            "MPI cotangent pytrees differ between root and rank "
            f"{mismatch}; root-only={sorted(root_path_set - other_paths)[:5]}, "
            f"rank-only={sorted(other_paths - root_path_set)[:5]}"
        )

    leaves_by_path = {
        path: leaf
        for path, (_, leaf) in zip(paths, leaves_with_path)
    }
    local_metadata = {
        path: _tree_reduce_leaf_metadata(leaf)
        for path, leaf in leaves_by_path.items()
    }
    metadata_by_rank = comm.allgather(local_metadata)
    ordered_paths = all_paths[root]
    plans = _tree_reduce_plans(ordered_paths, metadata_by_rank)

    summed = {} if rank == root else None
    for path in ordered_paths:
        plan = plans[path]
        if plan[0] == "none":
            if rank == root:
                summed[path] = None
            continue
        if plan[0] == "float0":
            if rank == root:
                summed[path] = numpy.zeros(
                    plan[1], dtype=jax.dtypes.float0
                )
            continue

        shape, dtype = plan[1], numpy.dtype(plan[2])
        local_leaf = leaves_by_path[path]
        local_kind = local_metadata[path][0]
        if local_kind == "numeric":
            host_leaf = numpy.array(
                numpy.asarray(local_leaf), copy=True, order="C"
            )
            flat_leaf = host_leaf.reshape(-1)
        else:
            host_leaf = None
            flat_leaf = None

        if rank == root:
            if host_leaf is None:
                result = numpy.zeros(shape, dtype=dtype)
            else:
                result = host_leaf
            flat_result = result.reshape(-1)
        else:
            result = None
            flat_result = None

        size = int(numpy.prod(shape, dtype=numpy.int64))
        chunk_count = min(
            _TREE_REDUCE_MAX_COUNT,
            max(1, int(_TREE_REDUCE_CHUNK_BYTES) // dtype.itemsize),
        )
        for start in range(0, size, chunk_count):
            stop = min(size, start + chunk_count)
            if rank == root:
                comm.Reduce(
                    MPI.IN_PLACE,
                    flat_result[start:stop],
                    op=MPI.SUM,
                    root=root,
                )
            else:
                if flat_leaf is None:
                    send = numpy.zeros(stop - start, dtype=dtype)
                else:
                    send = flat_leaf[start:stop]
                comm.Reduce(send, None, op=MPI.SUM, root=root)
        if rank == root:
            summed[path] = result

    if rank != root:
        return None
    return jax.tree_util.tree_unflatten(
        treedef, [summed[path] for path in paths]
    )


def _array_tree_digest(tree):
    """Return a reproducible digest of all numeric leaves in ``tree``."""
    digest = hashlib.sha256()
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(
        tree, is_leaf=lambda value: value is None
    )
    for path, leaf in leaves_with_path:
        if leaf is None or not hasattr(leaf, "dtype"):
            continue
        array = numpy.ascontiguousarray(numpy.asarray(leaf))
        digest.update(jax.tree_util.keystr(path).encode("utf8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_target_options(
    topology,
    *,
    lo_type,
    lo_kwargs,
    frag_lolist,
    frag_atmlist,
    frozen,
):
    """Validate a supplied MP2 topology against the request."""
    validate_static_target_options(
        topology,
        lo_type=lo_type,
        lo_kwargs=lo_kwargs,
        frag_lolist=frag_lolist,
        frag_atmlist=frag_atmlist,
        frozen=frozen,
        selection_name="topology",
        nested_attr=None,
    )


def _verify_shared_reference(
    comm, canonical, mf, *, verify_df_source=False
):
    """Require identical canonical, molecular, and DF metadata on all ranks."""

    mol = mf.mol
    auxmol = getattr(getattr(mf, "with_df", None), "auxmol", None)
    local = _array_tree_digest(canonical)
    digests = comm.allgather(local)
    if len(set(digests)) != 1:
        details = ", ".join(
            f"rank {rank}: {value[:12]}"
            for rank, value in enumerate(digests)
        )
        raise RuntimeError(
            "broadcast canonical orbital gauges differ across MPI ranks ("
            + details + ")"
        )
    system_signature = (
        int(mol.natm),
        int(mol.nao),
        int(mol.charge),
        int(mol.spin),
        bool(getattr(mol, "cart", False)),
        tuple(mol.atom_symbol(index) for index in range(mol.natm)),
        _semantic_tuple(numpy.asarray(mol.atom_charges()).tolist()),
        _semantic_tuple(numpy.asarray(mol.atom_coords()).tolist()),
        _semantic_tuple(getattr(mol, "_basis", None)),
        _semantic_tuple(getattr(mol, "_ecp", None)),
        _semantic_tuple(getattr(mol, "_pseudo", None)),
        None if auxmol is None else int(auxmol.nao),
        _semantic_tuple(
            getattr(getattr(mf, "with_df", None), "auxbasis", None)
        ),
        _semantic_tuple(getattr(auxmol, "_basis", None)),
    )
    system_signatures = comm.allgather(system_signature)
    if len(set(system_signatures)) != 1:
        raise RuntimeError(
            "molecular geometry or orbital/auxiliary bases differ across "
            "MPI ranks"
        )
    if verify_df_source:
        # Shape-compatible but scientifically different CDERI files are a
        # particularly dangerous MPI failure mode: fragment energies remain
        # finite while belonging to different Hamiltonians.  Hash logical
        # HDF5 contents (or the in-memory factors) once before distributed
        # correlation/response work and require exact agreement with root.
        df_digest = scientific_digest(df_source_fingerprint(mf))
        df_digests = comm.allgather(df_digest)
        if len(set(df_digests)) != 1:
            details = ", ".join(
                f"rank {rank}: {value[:12]}"
                for rank, value in enumerate(df_digests)
            )
            raise RuntimeError(
                "density-fitting integral contents differ across MPI ranks "
                f"({details})"
            )


def _verify_shared_gauge(comm, canonical, common, mf):
    """Require a shared reference and byte-identical common orbital frame."""

    _verify_shared_reference(
        comm, canonical, mf, verify_df_source=True
    )
    local = _array_tree_digest(common)
    digests = comm.allgather(local)
    if len(set(digests)) != 1:
        details = ", ".join(
            f"rank {rank}: {value[:12]}"
            for rank, value in enumerate(digests)
        )
        raise RuntimeError(
            "broadcast IAO-MP2 common orbital gauges differ across MPI "
            f"ranks ({details})"
        )


def _zero_term_cotangents(mf, common):
    """Construct exact zero cotangents with the local pytree structures."""
    _, pullback = jax.vjp(
        lambda mf_, common_: np.zeros((), dtype=common_.s1e.dtype),
        mf,
        common,
    )
    return pullback(np.ones((), dtype=common.s1e.dtype))
