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

"""Gauge-consistent MPI IAO-fragment MP2 energies and gradients.

The shared SCF orbitals, IAOs, PAOs, fragment weights, and fixed topology are
constructed exactly once on ``root``.  Their numerical values are broadcast
to the workers, which differentiate independent strong-ED energies and
unordered weak multipole pairs with respect to the *same* ``(mf, common)``
coordinate system.  The resulting cotangents are reduced before rank 0
replays each gauge-defining orbital build, applies the one shared common-
orbital pullback, and finally applies the one implicit SCF pullback.

This separation is important for local-correlation gradients.  Rebuilding
canonical/local orbitals independently on every MPI rank permits otherwise
equivalent eigenspaces to acquire different signs or internal rotations; the
resulting cotangents then do not share a well-defined gauge.  Rank 0 therefore
builds and distributes the exact strong-ED and weak-pair endpoint frames in
bounded batches.  Workers differentiate only the scalar correlated
calculation in those frames.  After each batch, rank 0 deterministically
replays one ED/screen build at a time, verifies that its primal digest is
unchanged, and closes its cotangent into the common representation.  This
bounds stored orbital frames linearly in the MPI rank count and the root
orbital-build tape to one frame instead of retaining every fragment tape
simultaneously.

The correlation result is the complete IAO-DLNO-MP2 correction: exact strong
ED contributions plus every retained unordered weak multipole pair.  It is
the same local-MP2 correction used by serial IAO-DLNO-CCSD(T).  "Complete"
here means every term of the IAO-DLNO-MP2 model: the strong EDs use full-spin
MP2, while the distant term is Nagy's OS-based multipole approximation to the
omitted total pair correlation rather than a literal OS+SS integral
evaluation.
"""

from __future__ import annotations

import gc
import hashlib
import time
import traceback

import jax
import jax.numpy as jnp
import numpy
from mpi4py import MPI

from pyscfad import config_update
from pyscfad.df.mpi_df_jk import MPIDFJKExecutor, ServiceExit
from pyscfad.ops import stop_trace

from .iao_mp2 import (
    IAOFragmentMP2 as _SerialIAOFragmentMP2,
    IAOFragmentTopology,
)
from .iao_mp2_grad import (
    IAOFragmentMP2StaticSelections,
    IAOMP2GradientTiming,
    IAOMP2TermResult,
    _add_cotangent,
    _correlation_term_specs,
    _make_decomposition,
    build_strong_ed_domain,
    build_weak_multipole_screen,
    rebuild_iao_mp2_common,
    strong_domain_energy,
    weak_screen_pair_energy,
)


__all__ = [
    "IAOFragmentMP2",
    "correlation_value_and_grad",
]


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


def _report_progress(reporter, message):
    if reporter is not None:
        reporter(f"[IAO-MP2] {message}")


def _exception_text(stage):
    return f"{stage} failed on an MPI rank:\n{traceback.format_exc()}"


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
        return jnp.asarray(leaf)
    return leaf


def _path_key(path):
    return jax.tree_util.keystr(path)


def _tree_sum_to_root(comm, tree, *, root=0):
    """Sum numeric leaves of a JAX pytree onto ``root``.

    Paths, rather than registered object identities, align leaves across the
    independently constructed rank-0 and worker ``mf`` objects.
    """
    if comm.Get_size() == 1:
        return tree
    rank = comm.Get_rank()
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(
        tree, is_leaf=lambda value: value is None
    )
    paths = [_path_key(path) for path, _ in leaves_with_path]
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
    local = {
        path: _to_host_leaf(leaf)
        for path, (_, leaf) in zip(paths, leaves_with_path)
    }
    gathered = comm.gather(local, root=root)
    if rank != root:
        return None

    summed = dict(gathered[0])
    for other in gathered[1:]:
        for path, value in other.items():
            summed[path] = _add_cotangent(summed[path], value)
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
        digest.update(_path_key(path).encode("utf8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _semantic_tuple(value):
    """Convert nested PySCF basis metadata to a comparable value tuple."""
    if isinstance(value, dict):
        return tuple(
            (key, _semantic_tuple(item))
            for key, item in sorted(value.items(), key=lambda pair: pair[0])
        )
    if isinstance(value, (list, tuple)):
        return tuple(_semantic_tuple(item) for item in value)
    if isinstance(value, numpy.ndarray):
        return _semantic_tuple(value.tolist())
    if isinstance(value, numpy.generic):
        return value.item()
    return value


def _verify_shared_gauge(comm, canonical, common, mf):
    """Require byte-identical orbital, geometry, and basis data on all ranks."""
    mol = mf.mol
    auxmol = getattr(getattr(mf, "with_df", None), "auxmol", None)
    local = _array_tree_digest((canonical, common))
    digests = comm.allgather(local)
    if len(set(digests)) != 1:
        details = ", ".join(
            f"rank {rank}: {value[:12]}"
            for rank, value in enumerate(digests)
        )
        raise RuntimeError(
            "broadcast IAO-MP2 orbital gauges differ across MPI ranks ("
            + details + ")"
        )
    system_signature = (
        int(mol.natm),
        int(mol.nao),
        tuple(mol.atom_symbol(index) for index in range(mol.natm)),
        _semantic_tuple(numpy.asarray(mol.atom_coords()).tolist()),
        _semantic_tuple(getattr(mol, "_basis", None)),
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


def _zero_term_cotangents(mf, common):
    """Construct exact zero cotangents with the local pytree structures."""
    _, pullback = jax.vjp(
        lambda mf_, common_: jnp.zeros((), dtype=common_.s1e.dtype),
        mf,
        common,
    )
    return pullback(jnp.ones((), dtype=common.s1e.dtype))


def correlation_value_and_grad(
    mf,
    static,
    *,
    comm=MPI.COMM_WORLD,
    root=0,
    return_details=False,
    progress=False,
):
    """Return the full MPI IAO-DLNO-MP2 correlation energy and ``mf`` bar.

    All ranks must pass an ``mf`` object carrying the canonical MO state
    broadcast by rank ``root``.  Only root supplies ``static``; passing it on
    other ranks is harmless, but the root object is authoritative and is
    broadcast together with the root-built common IAO/PAO representation.

    The returned energy is available on every rank.  The summed ``mf``
    cotangent is returned on ``root`` and is ``None`` elsewhere, so an MPI
    DLNO-CCSD(T) driver can add it to its CC cotangent before one final SCF
    response.

    Set ``progress=True`` to stream stage, batch, and per-term timing/energy
    lines from rank ``root``.  A callable may be supplied instead and is
    called with each formatted line on root only.  The setting must be the
    same on every rank.  Progress timing is collected even when
    ``return_details`` is false.
    """
    rank = comm.Get_rank()
    nproc = comm.Get_size()
    if root < 0 or root >= nproc:
        raise ValueError(f"root={root} is invalid for {nproc} MPI ranks")
    progress_enabled = _progress_enabled(progress)
    progress_flags = comm.allgather(progress_enabled)
    if len(set(progress_flags)) != 1:
        raise ValueError("progress must be enabled consistently on all ranks")
    reporter = _progress_reporter(progress, rank=rank, root=root)
    collect_timing = return_details or progress_enabled
    total_start = time.perf_counter() if collect_timing else None

    if rank == root:
        if not isinstance(static, IAOFragmentMP2StaticSelections):
            raise TypeError(
                "root must supply IAOFragmentMP2StaticSelections"
            )
        _report_progress(
            reporter,
            "common IAO/PAO orbital build: starting",
        )
        common_start = time.perf_counter() if collect_timing else None
        common, common_pullback = jax.vjp(
            lambda mf_: rebuild_iao_mp2_common(mf_, static), mf
        )
        if collect_timing:
            jax.block_until_ready(common)
            common_forward_seconds = time.perf_counter() - common_start
            _report_progress(
                reporter,
                "common IAO/PAO orbital build: done in "
                f"{common_forward_seconds:.1f} s",
            )
        payload = (
            static,
            jax.tree_util.tree_map(_to_host_leaf, common),
        )
    else:
        common_pullback = None
        common_forward_seconds = 0.0
        payload = None

    static, common_host = comm.bcast(
        payload, root=root
    )
    common = jax.tree_util.tree_map(_to_device_leaf, common_host)

    canonical = (
        numpy.asarray(mf.mo_coeff),
        numpy.asarray(mf.mo_energy),
        numpy.asarray(mf.mo_occ),
    )
    _verify_shared_gauge(comm, canonical, common, mf)
    _report_progress(reporter, "shared orbital gauge verified on all ranks")

    work = _correlation_term_specs(static)
    nstrong_terms = sum(spec[0] == "strong" for spec in work)
    nweak_terms = len(work) - nstrong_terms
    local_energy = jnp.zeros((), dtype=common.s1e.dtype)
    mf_bar, _ = _zero_term_cotangents(mf, common)
    if rank == root:
        _, common_bar_root = _zero_term_cotangents(mf, common)
        term_results_root = [] if collect_timing else None
        progress_energy_root = 0.0
    else:
        common_bar_root = None
        term_results_root = None

    # One bounded batch contains at most one correlation term per rank.  Root
    # supplies the exact orbital frame for each term, workers return only its
    # frame cotangent, and root closes those bars before constructing the next
    # batch.  Peak orbital-frame storage is therefore O(nproc), not O(nfrag).
    nbatch = (len(work) + nproc - 1) // nproc
    _report_progress(
        reporter,
        f"correlation pullback: {nstrong_terms} strong ED rows + "
        f"{nweak_terms} weak pairs in {nbatch} batches on {nproc} ranks",
    )
    for batch_index in range(nbatch):
        if rank == root:
            first_term = batch_index * nproc + 1
            last_term = min((batch_index + 1) * nproc, len(work))
            _report_progress(
                reporter,
                f"batch {batch_index + 1}/{nbatch}: building orbital "
                f"frames for terms {first_term}-{last_term}",
            )
            batch_payloads = []
            for slot in range(nproc):
                work_index = batch_index * nproc + slot
                if work_index >= len(work):
                    batch_payloads.append(None)
                    continue
                spec = work[work_index]
                kind, left, right = spec
                frame_start = (
                    time.perf_counter() if collect_timing else None
                )
                if kind == "strong":
                    frame, pullback = jax.vjp(
                        lambda common_, _left=left:
                            build_strong_ed_domain(
                                common_, static, _left
                            ),
                        common,
                    )
                else:
                    left_frame, left_pullback = jax.vjp(
                        lambda common_, _left=left:
                            build_weak_multipole_screen(
                                common_, static, _left
                            ),
                        common,
                    )
                    right_frame, right_pullback = jax.vjp(
                        lambda common_, _right=right:
                            build_weak_multipole_screen(
                                common_, static, _right
                            ),
                        common,
                    )
                    frame = (left_frame, right_frame)
                    pullback = (left_pullback, right_pullback)
                del pullback
                frame_host = jax.tree_util.tree_map(_to_host_leaf, frame)
                frame_digest = _array_tree_digest(frame)
                if collect_timing:
                    frame_build_seconds = (
                        time.perf_counter() - frame_start
                    )
                    batch_payloads.append((
                        spec,
                        frame_host,
                        frame_digest,
                        frame_build_seconds,
                    ))
                else:
                    batch_payloads.append((
                        spec, frame_host, frame_digest
                    ))
                if kind == "weak":
                    del (
                        left_frame,
                        right_frame,
                        left_pullback,
                        right_pullback,
                    )
                del frame
                gc.collect()
        else:
            batch_payloads = None

        local_payload = comm.scatter(batch_payloads, root=root)
        if local_payload is None:
            local_result = None
        else:
            if collect_timing:
                (
                    spec,
                    frame_host,
                    frame_digest,
                    frame_build_seconds,
                ) = local_payload
            else:
                spec, frame_host, frame_digest = local_payload
            frame = jax.tree_util.tree_map(_to_device_leaf, frame_host)
            if _array_tree_digest(frame) != frame_digest:
                raise RuntimeError(
                    f"MPI corrupted the root-gauge frame for term {spec}"
                )
            del frame_host, frame_digest
            kind, left, right = spec
            forward_start = (
                time.perf_counter() if collect_timing else None
            )
            if kind == "strong":
                def term(mf_, domain_, _left=left):
                    return strong_domain_energy(
                        mf_, domain_, static, _left
                    ).total

                term_energy, pullback = jax.vjp(term, mf, frame)
            else:
                def term(
                    mf_, left_screen_, right_screen_,
                    _left=left, _right=right,
                ):
                    return weak_screen_pair_energy(
                        mf_,
                        left_screen_,
                        right_screen_,
                        static,
                        _left,
                        _right,
                    )

                term_energy, pullback = jax.vjp(
                    term, mf, frame[0], frame[1]
                )

            if collect_timing:
                jax.block_until_ready(term_energy)
                forward_seconds = time.perf_counter() - forward_start
                reverse_start = time.perf_counter()
            if kind == "strong":
                term_mf_bar, frame_bar = pullback(
                    jnp.ones((), dtype=term_energy.dtype)
                )
            else:
                (
                    term_mf_bar,
                    left_frame_bar,
                    right_frame_bar,
                ) = pullback(jnp.ones((), dtype=term_energy.dtype))
                frame_bar = (left_frame_bar, right_frame_bar)
                del left_frame_bar, right_frame_bar

            local_energy = local_energy + term_energy
            mf_bar = jax.tree_util.tree_map(
                _add_cotangent, mf_bar, term_mf_bar
            )
            jax.block_until_ready((local_energy, mf_bar, frame_bar))
            frame_bar_host = jax.tree_util.tree_map(
                _to_host_leaf, frame_bar
            )
            if collect_timing:
                reverse_seconds = time.perf_counter() - reverse_start
                local_result = (
                    spec,
                    frame_bar_host,
                    float(jax.device_get(term_energy)),
                    forward_seconds,
                    reverse_seconds,
                    rank,
                )
            else:
                local_result = (spec, frame_bar_host)
            del (
                frame,
                frame_bar,
                pullback,
                term,
                term_energy,
                term_mf_bar,
            )
            gc.collect()

        gathered = comm.gather(local_result, root=root)
        if rank == root:
            for slot, (sent, result) in enumerate(zip(
                batch_payloads, gathered
            )):
                if sent is None:
                    if result is not None:
                        raise RuntimeError(
                            "an idle MPI rank returned a correlation bar"
                        )
                    continue
                if collect_timing:
                    (
                        spec,
                        _,
                        expected_digest,
                        frame_build_seconds,
                    ) = sent
                else:
                    spec, _, expected_digest = sent
                if result is None or result[0] != spec:
                    raise RuntimeError(
                        f"MPI returned the wrong correlation term for {spec}"
                    )
                frame_bar = jax.tree_util.tree_map(
                    _to_device_leaf, result[1]
                )
                kind, left, right = spec
                replay_start = (
                    time.perf_counter() if collect_timing else None
                )
                if kind == "strong":
                    rebuilt, pullback = jax.vjp(
                        lambda common_, _left=left:
                            build_strong_ed_domain(
                                common_, static, _left
                            ),
                        common,
                    )
                    if _array_tree_digest(rebuilt) != expected_digest:
                        raise RuntimeError(
                            "rank-0 strong-domain gauge changed while "
                            f"replaying fragment {int(left) + 1}"
                        )
                    term_common_bar, = pullback(frame_bar)
                else:
                    rebuilt_left, left_pullback = jax.vjp(
                        lambda common_, _left=left:
                            build_weak_multipole_screen(
                                common_, static, _left
                            ),
                        common,
                    )
                    rebuilt_right, right_pullback = jax.vjp(
                        lambda common_, _right=right:
                            build_weak_multipole_screen(
                                common_, static, _right
                            ),
                        common,
                    )
                    if (
                        _array_tree_digest(
                            (rebuilt_left, rebuilt_right)
                        ) != expected_digest
                    ):
                        raise RuntimeError(
                            "rank-0 weak-screen gauge changed while "
                            f"replaying pair ({int(left) + 1}, "
                            f"{int(right) + 1})"
                        )
                    left_common_bar, = left_pullback(frame_bar[0])
                    right_common_bar, = right_pullback(frame_bar[1])
                    term_common_bar = jax.tree_util.tree_map(
                        _add_cotangent,
                        left_common_bar,
                        right_common_bar,
                    )
                    del (
                        rebuilt_left,
                        rebuilt_right,
                        left_pullback,
                        right_pullback,
                        left_common_bar,
                        right_common_bar,
                    )
                common_bar_root = jax.tree_util.tree_map(
                    _add_cotangent,
                    common_bar_root,
                    term_common_bar,
                )
                if collect_timing:
                    jax.block_until_ready(common_bar_root)
                    frame_replay_seconds = (
                        time.perf_counter() - replay_start
                    )
                    term_result = IAOMP2TermResult(
                        kind=str(kind),
                        left_fragment=int(left),
                        right_fragment=(
                            None if int(right) == -1 else int(right)
                        ),
                        energy=float(result[2]),
                        forward_seconds=float(result[3]),
                        reverse_seconds=float(result[4]),
                        frame_build_seconds=float(frame_build_seconds),
                        frame_replay_seconds=float(frame_replay_seconds),
                        worker_rank=int(result[5]),
                    )
                    term_results_root.append(term_result)
                    progress_energy_root += term_result.energy
                    term_number = batch_index * nproc + slot + 1
                    if kind == "strong":
                        fragment = static.fragments[int(left)]
                        n_atoms = numpy.asarray(
                            fragment.extended_atoms
                        ).size
                        n_ao = numpy.asarray(
                            fragment.extended_ao_indices
                        ).size
                        n_occ = numpy.asarray(
                            fragment.strong_occ_metric_keep
                        ).size
                        n_vir = numpy.asarray(
                            fragment.strong_virtual.metric_keep
                        ).size
                        label = (
                            f"strong ED fragment {int(left) + 1} "
                            f"[atoms={n_atoms}, AO={n_ao}, "
                            f"occ={n_occ}, vir={n_vir}]"
                        )
                    else:
                        label = (
                            f"weak pair ({int(left) + 1},"
                            f"{int(right) + 1})"
                        )
                    _report_progress(
                        reporter,
                        f"term {term_number}/{len(work)} {label} "
                        f"[rank {term_result.worker_rank}]: "
                        f"E={term_result.energy:+.10f} Eh; "
                        f"forward/reverse="
                        f"{term_result.forward_seconds:.1f}/"
                        f"{term_result.reverse_seconds:.1f} s; "
                        f"frame build/replay="
                        f"{term_result.frame_build_seconds:.1f}/"
                        f"{term_result.frame_replay_seconds:.1f} s; "
                        f"elapsed={time.perf_counter() - total_start:.1f} s",
                    )
                del frame_bar, term_common_bar
                if kind == "strong":
                    del rebuilt, pullback
                gc.collect()
            if collect_timing:
                _report_progress(
                    reporter,
                    f"batch {batch_index + 1}/{nbatch}: complete; "
                    f"accumulated E_corr={progress_energy_root:+.10f} Eh; "
                    f"elapsed={time.perf_counter() - total_start:.1f} s",
                )
            del batch_payloads, gathered, sent, result
        else:
            del gathered
        del local_payload, local_result
        gc.collect()

    _report_progress(reporter, "reducing correlation energy and cotangents")
    corr_energy = comm.allreduce(float(local_energy), op=MPI.SUM)
    mf_bar_root = _tree_sum_to_root(comm, mf_bar, root=root)

    common_reverse_seconds = 0.0
    if rank == root:
        _report_progress(reporter, "common-orbital pullback: starting")
        common_reverse_start = (
            time.perf_counter() if collect_timing else None
        )
        common_mf_bar, = common_pullback(common_bar_root)
        mf_bar_root = jax.tree_util.tree_map(
            _add_cotangent, mf_bar_root, common_mf_bar
        )
        jax.block_until_ready(mf_bar_root)
        if collect_timing:
            common_reverse_seconds = (
                time.perf_counter() - common_reverse_start
            )
            _report_progress(
                reporter,
                "common-orbital pullback: done in "
                f"{common_reverse_seconds:.1f} s",
            )
    if collect_timing:
        total_seconds = comm.allreduce(
            time.perf_counter() - total_start, op=MPI.MAX
        )
        if rank == root:
            strong_terms = tuple(
                term for term in term_results_root
                if term.kind == "strong"
            )
            weak_terms = tuple(
                term for term in term_results_root
                if term.kind == "weak"
            )
            timing = IAOMP2GradientTiming(
                common_forward_seconds=float(common_forward_seconds),
                strong_forward_seconds=sum(
                    term.forward_seconds for term in strong_terms
                ),
                strong_reverse_seconds=sum(
                    term.reverse_seconds for term in strong_terms
                ),
                weak_forward_seconds=sum(
                    term.forward_seconds for term in weak_terms
                ),
                weak_reverse_seconds=sum(
                    term.reverse_seconds for term in weak_terms
                ),
                frame_build_seconds=sum(
                    term.frame_build_seconds for term in term_results_root
                ),
                frame_replay_seconds=sum(
                    term.frame_replay_seconds for term in term_results_root
                ),
                common_reverse_seconds=float(common_reverse_seconds),
                total_seconds=float(total_seconds),
            )
            _report_progress(
                reporter,
                f"correlation pullback: complete; E_corr="
                f"{corr_energy:+.10f} Eh; wall={total_seconds:.1f} s",
            )
        if return_details:
            if rank == root:
                details = _make_decomposition(
                    static, corr_energy, term_results_root, timing
                )
            else:
                details = None
            details = comm.bcast(details, root=root)
            return corr_energy, mf_bar_root, details
    if return_details:
        # ``collect_timing`` is necessarily true in this branch.
        raise AssertionError("unreachable return_details state")
    return corr_energy, mf_bar_root


class IAOFragmentMP2(_SerialIAOFragmentMP2):
    """MPI-parallel fixed-topology IAO-fragment MP2 driver."""

    @classmethod
    def value_and_grad(
        cls,
        mol,
        *,
        build_mf,
        frag_lolist=None,
        frag_atmlist=None,
        frozen=None,
        thresholds=None,
        pair_energy_model="multipole",
        force_full_domains=False,
        topology=None,
        include_hf=True,
        parallel_scf_jk=False,
        comm=MPI.COMM_WORLD,
        root=0,
        return_details=False,
        progress=False,
    ):
        """Return full IAO-DLNO-MP2 energy and gradient using MPI.

        ``build_mf`` follows the MPI DLNO-CCSD(T) convention.  On root it is
        called as ``build_mf(mol)`` and must run the converged DF-RHF SCF.  On
        workers it is called with ``mo_coeff_init``, ``mo_energy_init``,
        ``mo_occ_init``, and ``e_tot_init`` keyword arguments and must create
        a matching DF skeleton while adopting those arrays without another
        SCF or orbital diagonalization.  With the integral-direct local-Lov
        path, a worker needs the same auxiliary basis and DF pytree leaves but
        does not need to build or read a global CDERI file.

        Rank 0 alone owns the SCF VJP and fixed IAO topology.  The complete
        strong-plus-weak correlation cotangent is reduced before that one
        standard implicit SCF pullback.  The energy is returned on all ranks;
        the Mole-shaped gradient is returned on root and ``None`` elsewhere.

        With ``parallel_scf_jk=True`` and more than one rank, workers also
        serve disjoint auxiliary-function blocks while root evaluates the
        forward DF-SCF J/K builds and the density-response part of the SCF
        pullback.  In the final CDERI-to-coordinate pullback, AO-pair shell
        blocks of the three-centre integral derivative are also distributed;
        root retains the coupled whitening algebra and two-centre metric
        derivative.
        Before the forward service starts, workers call the existing
        ``build_mf`` reconstruction path with a dummy closed-shell canonical
        state.  Thus no additional builder API is required, but that path
        must attach the same rank-accessible CDERI source as root.  After the
        forward service, the dummy state is replaced by root's converged
        canonical orbitals before the ordinary correlation work begins.

        With ``return_details=True``, a third return value contains the
        strong/weak energy split, unordered pair counts, ED dimensions, and
        scalar timings collected by these same one-term-at-a-time pullbacks.
        No term is reevaluated and no orbital frame or AD tape is retained
        for reporting.

        ``progress=True`` prints rank-0-only, flushed progress lines for the
        SCF, topology, correlation terms, common-orbital pullback, and final
        SCF response.  A callable may be supplied to receive those formatted
        lines instead.  Pass the same setting on every MPI rank.
        """
        rank = comm.Get_rank()
        nproc = comm.Get_size()
        progress_enabled = _progress_enabled(progress)
        local_schedule = (
            int(root),
            bool(parallel_scf_jk),
            bool(include_hf),
            bool(return_details),
            progress_enabled,
        )
        schedules = comm.allgather(local_schedule)
        if len(set(schedules)) != 1:
            raise ValueError(
                "root, parallel_scf_jk, include_hf, return_details, and "
                "progress must be consistent on all MPI ranks"
            )
        if root < 0 or root >= nproc:
            raise ValueError(f"root={root} is invalid for {nproc} MPI ranks")
        parallel_scf_jk = bool(parallel_scf_jk) and nproc > 1
        reporter = _progress_reporter(progress, rank=rank, root=root)
        overall_start = time.perf_counter()
        if (
            getattr(mol, "exp", None) is not None
            or getattr(mol, "ctr_coeff", None) is not None
        ):
            raise NotImplementedError(
                "MPI IAOFragmentMP2 currently differentiates nuclear "
                "coordinates only; build mol with trace_exp=False and "
                "trace_ctr_coeff=False"
            )

        scf_executor = None
        if parallel_scf_jk:
            if int(mol.nelectron) % 2 or int(mol.spin) != 0:
                raise NotImplementedError(
                    "parallel_scf_jk currently supports spin-zero, "
                    "closed-shell RHF only"
                )
            scf_executor = MPIDFJKExecutor(comm=comm, root=root)
            worker_setup_error = None
            if rank != root:
                try:
                    nao = int(mol.nao)
                    dummy_occ = numpy.zeros(nao)
                    dummy_occ[:int(mol.nelectron) // 2] = 2.0
                    mf = build_mf(
                        mol,
                        mo_coeff_init=numpy.eye(nao),
                        mo_energy_init=numpy.zeros(nao),
                        mo_occ_init=dummy_occ,
                        e_tot_init=0.0,
                    )
                    if getattr(mf, "with_df", None) is None:
                        raise TypeError(
                            "worker build_mf did not return a density-fitted "
                            "SCF object"
                        )
                    # The reconstruction path is allowed to return a lazy DF
                    # skeleton.  Materialize its rank-local in-core CDERI, or
                    # validate/attach the prebuilt outcore source, before the
                    # root can issue its first distributed J/K request.
                    mf.with_df.build()
                except Exception as error:  # collective preflight below
                    worker_setup_error = (
                        f"rank {rank}: {type(error).__name__}: {error}"
                    )
            worker_setup_errors = comm.allgather(worker_setup_error)
            worker_setup_errors = tuple(
                error for error in worker_setup_errors if error is not None
            )
            if worker_setup_errors:
                scf_executor.close_local()
                raise RuntimeError(
                    "parallel_scf_jk worker setup failed; the existing "
                    "mo_*_init build_mf path must construct a no-SCF DF "
                    "skeleton backed by the same prebuilt CDERI source as "
                    "root:\n" + "\n".join(worker_setup_errors)
                )

        if rank == root:
            scf_label = "MPI DF-RHF" if parallel_scf_jk else "DF-RHF"
            try:
                _report_progress(
                    reporter, f"{scf_label} SCF and VJP setup: starting"
                )
                scf_start = time.perf_counter()
                with (
                    config_update("pyscfad_scf_implicit_diff", True),
                    config_update("pyscfad_scf_first_order_custom", False),
                ):
                    if parallel_scf_jk:
                        with scf_executor.root_session(final=False):
                            mf, scf_pullback = jax.vjp(build_mf, mol)
                            jax.block_until_ready(mf.e_tot)
                    else:
                        mf, scf_pullback = jax.vjp(build_mf, mol)
                        jax.block_until_ready(mf.e_tot)
            except Exception:
                if scf_executor is not None:
                    scf_executor.stop_workers()
                raise
            topology_error = None
            try:
                _report_progress(
                    reporter,
                    f"{scf_label} SCF and VJP setup: done in "
                    f"{time.perf_counter() - scf_start:.1f} s; "
                    f"E_HF={float(mf.e_tot):+.10f} Eh",
                )
                topology_start = time.perf_counter()
                _report_progress(
                    reporter, "fixed IAO fragment topology: starting"
                )
                if topology is None:
                    fixed_topology = stop_trace(
                        lambda mf_: cls.build_static_topology(
                            mf_,
                            frozen=frozen,
                            frag_lolist=frag_lolist,
                            frag_atmlist=frag_atmlist,
                            thresholds=thresholds,
                            pair_energy_model=pair_energy_model,
                            force_full_domains=force_full_domains,
                        )
                    )(mf)
                elif isinstance(topology, IAOFragmentTopology):
                    from .iao_mp2_grad import build_iao_mp2_static_selections

                    fixed_topology = stop_trace(
                        lambda mf_: build_iao_mp2_static_selections(
                            mf_, topology
                        )
                    )(mf)
                elif isinstance(topology, IAOFragmentMP2StaticSelections):
                    fixed_topology = topology
                else:
                    raise TypeError(
                        "topology must be IAOFragmentTopology or "
                        "IAOFragmentMP2StaticSelections"
                    )

                term_specs = _correlation_term_specs(fixed_topology)
                nstrong_terms = sum(
                    spec[0] == "strong" for spec in term_specs
                )
                nweak_terms = len(term_specs) - nstrong_terms
                strong_mask = numpy.asarray(
                    fixed_topology.strong_mask, dtype=bool
                )
                nstrong_pairs = int(numpy.count_nonzero(
                    numpy.triu(strong_mask, k=1)
                ))
                _report_progress(
                    reporter,
                    f"fixed IAO fragment topology: done in "
                    f"{time.perf_counter() - topology_start:.1f} s; "
                    f"fragments={len(fixed_topology.fragments)}, "
                    f"strong/weak pairs={nstrong_pairs}/{nweak_terms}, "
                    f"correlation terms={len(term_specs)}",
                )

                canonical = {
                    "mo_coeff": numpy.asarray(mf.mo_coeff),
                    "mo_energy": numpy.asarray(mf.mo_energy),
                    "mo_occ": numpy.asarray(mf.mo_occ),
                    "e_tot": float(mf.e_tot),
                }
            except Exception:  # pragma: no cover - multi-rank failure path
                topology_error = _exception_text(
                    "root IAO fragment topology setup"
                )
                fixed_topology = canonical = None
        else:
            if parallel_scf_jk:
                service_exit = scf_executor.serve(mf.with_df)
                if service_exit is not ServiceExit.PAUSED:
                    raise RuntimeError(
                        "MPI DF-J/K forward worker service stopped before "
                        "the SCF completed"
                    )
            else:
                mf = None
            scf_pullback = fixed_topology = None
            canonical = None
            topology_error = None

        topology_error = comm.bcast(topology_error, root=root)
        if topology_error is not None:
            if scf_executor is not None:
                scf_executor.close_local()
            raise RuntimeError(topology_error)
        canonical = comm.bcast(canonical, root=root)
        if rank != root:
            if parallel_scf_jk:
                mf.mo_coeff = canonical["mo_coeff"]
                mf.mo_energy = canonical["mo_energy"]
                mf.mo_occ = canonical["mo_occ"]
                mf.e_tot = canonical["e_tot"]
                mf.converged = True
            else:
                try:
                    mf = build_mf(
                        mol,
                        mo_coeff_init=canonical["mo_coeff"],
                        mo_energy_init=canonical["mo_energy"],
                        mo_occ_init=canonical["mo_occ"],
                        e_tot_init=canonical["e_tot"],
                    )
                except TypeError as error:
                    raise TypeError(
                        "non-root build_mf must accept mo_coeff_init, "
                        "mo_energy_init, mo_occ_init, and e_tot_init and "
                        "must not run SCF"
                    ) from error

        _report_progress(reporter, "MPI correlation energy/gradient: starting")
        corr_result = correlation_value_and_grad(
            mf,
            fixed_topology,
            comm=comm,
            root=root,
            return_details=return_details,
            progress=progress,
        )
        corr_energy, mf_bar_root = corr_result[:2]

        response_setup_error = None
        mol_bar = None
        if rank == root:
            try:
                if include_hf:
                    _report_progress(
                        reporter, "adding the Hartree-Fock energy seed"
                    )
                    e_hf, hf_pullback = jax.vjp(lambda mf_: mf_.e_tot, mf)
                    hf_bar, = hf_pullback(
                        jnp.ones((), dtype=jnp.asarray(e_hf).dtype)
                    )
                    mf_bar_root = jax.tree_util.tree_map(
                        _add_cotangent, mf_bar_root, hf_bar
                    )
                    energy = float(e_hf) + corr_energy
                else:
                    energy = corr_energy
                _report_progress(
                    reporter,
                    "implicit SCF response for the total orbital "
                    "cotangent: " + (
                        "starting with MPI-parallel DF J/K response"
                        if parallel_scf_jk
                        else "starting on rank 0 (this is the long serial tail)"
                    ),
                )
                response_start = time.perf_counter()
            except Exception:  # pragma: no cover - multi-rank failure path
                response_setup_error = _exception_text(
                    "root implicit SCF response setup"
                )
        else:
            energy = (
                canonical["e_tot"] + corr_energy
                if include_hf else corr_energy
            )

        response_setup_error = comm.bcast(
            response_setup_error, root=root
        )
        if response_setup_error is not None:
            if scf_executor is not None:
                scf_executor.close_local()
            raise RuntimeError(response_setup_error)

        response_error = None
        if rank == root:
            try:
                if parallel_scf_jk:
                    with scf_executor.root_session(final=True):
                        mol_bar, = scf_pullback(mf_bar_root)
                        jax.block_until_ready(mol_bar)
                else:
                    mol_bar, = scf_pullback(mf_bar_root)
                    jax.block_until_ready(mol_bar)
                gradient_norm = float(numpy.linalg.norm(
                    numpy.asarray(mol_bar.coords)
                ))
                _report_progress(
                    reporter,
                    "implicit SCF response: done in "
                    f"{time.perf_counter() - response_start:.1f} s; "
                    f"|gradient|={gradient_norm:.6e} Eh/bohr; "
                    f"total elapsed="
                    f"{time.perf_counter() - overall_start:.1f} s",
                )
            except Exception:  # pragma: no cover - multi-rank failure path
                response_error = _exception_text(
                    "root implicit SCF response"
                )
        elif parallel_scf_jk:
            try:
                service_exit = scf_executor.serve(mf.with_df)
                if service_exit is not ServiceExit.STOPPED:
                    raise RuntimeError(
                        "MPI DF-J/K reverse worker service paused before "
                        "the SCF pullback completed"
                    )
            except Exception:  # pragma: no cover - multi-rank failure path
                response_error = _exception_text(
                    f"implicit SCF response worker rank {rank}"
                )
        _raise_if_any_rank_failed(comm, response_error)
        if return_details:
            return energy, mol_bar, corr_result[2]
        return energy, mol_bar
