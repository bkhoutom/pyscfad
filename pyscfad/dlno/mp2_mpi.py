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

"""MPI DLNO-MP2 with one shared orbital gauge and progressive pullbacks."""

from __future__ import annotations

import gc
from dataclasses import replace
from pathlib import Path
import time

import jax
import jax.numpy as np
from mpi4py import MPI
import numpy

from pyscfad import config_update
from pyscfad.lib._compilation_cache import configure_compilation_cache
from pyscfad.df.mpi_df_jk import MPIDFJKExecutor, ServiceExit
from pyscfad.ops import stop_trace
from pyscfad.tools import resource_profile

from ._restart import RestartManager
from ._selection import DomainSelections
from .dlno_base import build_strong_ed_domain, rebuild_domain_data
from .dlno_base_mpi import (
    _abort_collective_on_error, _array_tree_digest, _exception_text,
    _progress_enabled, _progress_reporter, _raise_if_any_rank_failed,
    _raise_if_root_failed, _to_device_leaf, _to_host_leaf, _tree_sum_to_root,
    _validate_target_options, _verify_shared_gauge, _verify_shared_reference,
    _zero_mf_cotangent, _zero_term_cotangents,
)
from .domain import DLNOThresholds, DomainTopology
from .mp2 import (
    DLNOMP2 as _SerialDLNOMP2, MP2GradientTiming, MP2TermResult, _add_cotangent,
    _correlation_term_specs, _details_from_restart_metadata,
    _details_restart_metadata, _fix_restart_mo_phases, _make_decomposition,
    _serial_restart_scientific_payload, strong_domain_energy,
    build_weak_endpoint, _check_weak_endpoint_gauge,
    _align_weak_endpoint_signs,
    _weak_endpoint_pair_value_and_grad,
)
from .targets import resolve_target_options, semantic_tuple as _semantic_tuple


__all__ = [
    "DLNOMP2",
    "correlation_value_and_grad",
]


def _report_progress(reporter, message):
    if reporter is not None:
        reporter(f"[IAO-MP2] {message}")


def _mpi_restart_scientific_payload(
    mol,
    mf,
    *,
    frag_lolist,
    frag_atmlist,
    frozen,
    thresholds,
    pair_energy_model,
    force_full_domains,
    include_hf,
    lo_type="iao",
    lo_kwargs=None,
    nproc,
    root,
):
    """Return the scientific and collective identity of an MPI restart."""

    payload = _serial_restart_scientific_payload(
        mol,
        mf,
        frag_lolist=frag_lolist,
        frag_atmlist=frag_atmlist,
        frozen=frozen,
        thresholds=thresholds,
        pair_energy_model=pair_energy_model,
        force_full_domains=force_full_domains,
        include_hf=include_hf,
        lo_type=lo_type,
        lo_kwargs=lo_kwargs,
    )
    payload["driver"] = "mpi-iao-dlno-mp2-gradient"
    # Keep the collective service/scheduling layout identical on resume.
    # Correlation checkpoints themselves contain the fully closed canonical
    # cotangent and never advertise a partially completed reordered phase.
    payload["mpi"] = {"size": int(nproc), "root": int(root)}
    return payload


def _weak_endpoint_correlation_pullback(
    mf, common, static, weak_work, *, comm, root, collect_timing=False,
    reporter=None,
):
    """Distribute compact weak pairs and close each endpoint on its owner.

    Endpoint values are small enough to share on every rank.  The much
    larger AO moment VJP is constructed only on the endpoint's owning rank,
    used once with the reduced incident-pair cotangent, and released.
    """
    configure_compilation_cache()
    rank, nproc = comm.Get_rank(), comm.Get_size()
    indices = sorted({i for _, left, right in weak_work for i in (left, right)})
    owners = {index: slot % nproc for slot, index in enumerate(indices)}
    mf_bar, common_bar = _zero_term_cotangents(mf, common)
    local_energy = np.zeros((), dtype=common.s1e.dtype)
    if not indices:
        return local_energy, mf_bar, common_bar, ()

    _report_progress(reporter, f"weak endpoints: preparing {len(indices)} fragments")
    prepared = []
    profile_prepare = resource_profile.start()
    for index in indices:
        if owners[index] != rank:
            continue
        start = time.perf_counter() if collect_timing else None
        endpoint = build_weak_endpoint(mf, common, static, index)
        if collect_timing or profile_prepare is not None:
            jax.block_until_ready(endpoint)
        prepared.append((
            index, jax.tree_util.tree_map(_to_host_leaf, endpoint),
            _array_tree_digest(endpoint),
            time.perf_counter() - start if collect_timing else 0.0,
        ))
        del endpoint
    resource_profile.finish(
        "dlno.mp2_mpi.weak_endpoints.prepare", profile_prepare,
        n_endpoints=len(prepared),
    )
    endpoints, build_seconds = {}, {}
    for rank_prepared in comm.allgather(prepared):
        for index, endpoint_host, digest, seconds in rank_prepared:
            endpoint = jax.tree_util.tree_map(_to_device_leaf, endpoint_host)
            if _array_tree_digest(endpoint) != digest:
                raise RuntimeError(f"MPI corrupted weak endpoint {index}")
            endpoints[index] = endpoint
            build_seconds[index] = seconds
            del endpoint, endpoint_host
    del prepared, rank_prepared
    endpoint_bars = {
        index: jax.tree_util.tree_map(np.zeros_like, endpoint)
        for index, endpoint in endpoints.items()
    }

    _report_progress(reporter, f"weak pairs: contracting {len(weak_work)} compact pairs")
    local_results = []
    profile_pairs = resource_profile.start()
    for spec in weak_work[rank::nproc]:
        _, left, right = spec
        start = time.perf_counter() if collect_timing else None
        term_energy, (left_bar, right_bar) = _weak_endpoint_pair_value_and_grad(
            endpoints[left], endpoints[right], static.thresholds.multipole_order
        )
        if collect_timing:
            jax.block_until_ready((term_energy, left_bar, right_bar))
            forward_seconds = time.perf_counter() - start
            start = time.perf_counter()
        endpoint_bars[left] = jax.tree_util.tree_map(
            _add_cotangent, endpoint_bars[left], left_bar
        )
        endpoint_bars[right] = jax.tree_util.tree_map(
            _add_cotangent, endpoint_bars[right], right_bar
        )
        local_energy = local_energy + term_energy
        if collect_timing:
            jax.block_until_ready((local_energy, endpoint_bars))
            local_results.append(MP2TermResult(
                kind="weak", left_fragment=int(left), right_fragment=int(right),
                energy=float(jax.device_get(term_energy)),
                forward_seconds=float(forward_seconds),
                reverse_seconds=float(time.perf_counter() - start),
                worker_rank=int(rank),
            ))
        del left_bar, right_bar
    jax.block_until_ready((local_energy, endpoint_bars))
    resource_profile.finish(
        "dlno.mp2_mpi.weak_endpoints.pairs", profile_pairs,
        n_pairs=len(weak_work[rank::nproc]),
    )

    # Reduce each compact bar directly to the rank that built its endpoint.
    # Collective waits stay outside the local computation profile sections.
    owned_bars = {}
    for index in indices:
        reduced = _tree_sum_to_root(comm, endpoint_bars[index], root=owners[index])
        if owners[index] == rank:
            owned_bars[index] = reduced
    del endpoint_bars, reduced
    endpoints = {index: endpoints[index] for index in owned_bars}

    _report_progress(reporter, "weak endpoints: replaying distributed response")
    response_seconds = {}
    profile_response = resource_profile.start()
    owned_indices = tuple(owned_bars)
    for index in owned_indices:
        endpoint_bar = owned_bars.pop(index)
        start = time.perf_counter() if collect_timing else None
        reference = endpoints.pop(index)
        # Owner-local alignment is part of the differentiated map; pair bars
        # remain in their shared forward frame and need no manual correction.
        rebuilt, pullback = jax.vjp(
            lambda mf_, common_: _align_weak_endpoint_signs(
                build_weak_endpoint(mf_, common_, static, index), reference
            ), mf, common,
        )
        _check_weak_endpoint_gauge(
            rebuilt, reference, index, "MPI response replay"
        )
        endpoint_mf_bar, endpoint_common_bar = pullback(endpoint_bar)
        mf_bar = jax.tree_util.tree_map(_add_cotangent, mf_bar, endpoint_mf_bar)
        common_bar = jax.tree_util.tree_map(
            _add_cotangent, common_bar, endpoint_common_bar
        )
        jax.block_until_ready((mf_bar, common_bar))
        if collect_timing:
            response_seconds[index] = time.perf_counter() - start
        del reference, rebuilt, pullback
        del endpoint_mf_bar, endpoint_common_bar, endpoint_bar
    if owned_indices:
        # Keep only one live endpoint tape and collect after the entire
        # owned response, rather than scanning the heap for every endpoint.
        gc.collect()
    resource_profile.finish(
        "dlno.mp2_mpi.weak_endpoints.response", profile_response,
        n_endpoints=len(owned_indices),
    )
    results = None
    if collect_timing:
        gathered_results = comm.gather(local_results, root=root)
        gathered_response = comm.gather(response_seconds, root=root)
        if rank == root:
            by_spec = {
                ("weak", row.left_fragment, row.right_fragment): row
                for rows in gathered_results for row in rows
            }
            all_response = {
                index: seconds for rows in gathered_response
                for index, seconds in rows.items()
            }
            results = []
            for spec in weak_work:
                _, left, right = spec
                row = by_spec[spec]
                results.append(replace(
                    row,
                    frame_build_seconds=sum(
                        build_seconds.pop(index, 0.0) for index in (left, right)
                    ),
                    frame_replay_seconds=sum(
                        all_response.pop(index, 0.0) for index in (left, right)
                    ),
                ))
    return local_energy, mf_bar, common_bar, results


@_abort_collective_on_error
def correlation_value_and_grad(
    mf,
    static,
    *,
    comm=MPI.COMM_WORLD,
    root=0,
    return_details=False,
    progress=False,
    restart=None,
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
    configure_compilation_cache()
    progress_enabled = _progress_enabled(progress)
    progress_flags = comm.allgather(progress_enabled)
    if len(set(progress_flags)) != 1:
        raise ValueError("progress must be enabled consistently on all ranks")
    reporter = _progress_reporter(progress, rank=rank, root=root)
    restart_enabled = bool(
        restart is not None and getattr(restart, "enabled", False)
    )
    restart_state = (
        restart_enabled,
        None if not restart_enabled else str(restart.path),
        False if not restart_enabled else bool(restart.resume),
    )
    if len(set(comm.allgather(restart_state))) != 1:
        raise ValueError(
            "MPI correlation restart path/mode must be identical on all ranks"
        )
    collect_timing = return_details or progress_enabled or restart_enabled
    total_start = time.perf_counter() if collect_timing else None

    # A common-closed record is self-contained in the canonical MF gauge.  It
    # therefore supersedes all frame construction and worker term VJPs, while
    # still leaving the final SCF response to the outer driver.
    closed = None
    closed_payload = None
    closed_error = None
    if rank == root and restart_enabled and restart.resume:
        try:
            zero_mf_bar = _zero_mf_cotangent(mf)
            closed = restart.load_record(
                "mpi_correlation_closed",
                templates={"mf_bar": zero_mf_bar},
                missing_ok=True,
            )
            del zero_mf_bar
            if closed is not None:
                closed_payload = (
                    float(closed.scalars["energy"]),
                    closed.metadata,
                )
        except Exception:
            closed_error = _exception_text(
                "root MPI correlation restart load"
            )
    _raise_if_root_failed(comm, closed_error, root=root)
    closed_payload = comm.bcast(closed_payload, root=root)
    if closed_payload is not None:
        corr_energy, closed_metadata = closed_payload
        _report_progress(
            reporter,
            "restart: loaded completed common-closed correlation cotangent",
        )
        if return_details:
            details_error = None
            details = None
            if rank == root:
                try:
                    details = _details_from_restart_metadata(
                        static, corr_energy, closed_metadata
                    )
                except Exception:
                    details_error = _exception_text(
                        "root restarted MPI correlation diagnostics"
                    )
            _raise_if_root_failed(comm, details_error, root=root)
            details = comm.bcast(
                details, root=root
            )
            return corr_energy, (
                closed.trees["mf_bar"] if rank == root else None
            ), details
        return corr_energy, (
            closed.trees["mf_bar"] if rank == root else None
        )

    profile_total = resource_profile.start()
    if rank == root:
        if not isinstance(static, DomainSelections):
            raise TypeError(
                "root must supply DomainSelections"
            )
        _report_progress(
            reporter,
            "common IAO/PAO orbital build: starting",
        )
        common_start = time.perf_counter() if collect_timing else None
        profile_common_forward = resource_profile.start()
        common, common_pullback = jax.vjp(
            lambda mf_: rebuild_domain_data(mf_, static), mf
        )
        if collect_timing or profile_common_forward is not None:
            jax.block_until_ready(common)
        resource_profile.finish(
            "dlno.mp2_mpi.correlation.common_forward",
            profile_common_forward,
            n_fragments=len(static.fragments),
        )
        if collect_timing:
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

    all_work = _correlation_term_specs(static)
    work = tuple(spec for spec in all_work if spec[0] == "strong")
    weak_work = tuple(spec for spec in all_work if spec[0] == "weak")
    nstrong_terms, nweak_terms = len(work), len(weak_work)
    local_energy = np.zeros((), dtype=common.s1e.dtype)
    mf_bar, _ = _zero_term_cotangents(mf, common)
    if rank == root:
        _, common_bar_root = _zero_term_cotangents(mf, common)
        term_results_root = [] if collect_timing else None
        progress_energy_root = 0.0
    else:
        common_bar_root = None
        term_results_root = None

    # Strong EDs retain bounded root-gauge batches.  Weak endpoint values,
    # compact pair contractions, and endpoint response form a separate phase.
    nbatch = (len(work) + nproc - 1) // nproc
    _report_progress(
        reporter,
        f"correlation pullback: {nstrong_terms} strong ED rows in {nbatch} "
        f"batches + {nweak_terms} endpoint-factored weak pairs on {nproc} ranks",
    )
    for batch_index in range(nbatch):
        if rank == root:
            batch_payloads = []
            for slot in range(nproc):
                work_index = batch_index * nproc + slot
                if work_index >= len(work):
                    batch_payloads.append(None)
                    continue
                spec = work[work_index]
                _, left, _ = spec
                profile_frame_build = resource_profile.start()
                frame_start = time.perf_counter() if collect_timing else None
                frame, pullback = jax.vjp(
                    lambda common_: build_strong_ed_domain(common_, static, left),
                    common,
                )
                del pullback
                frame_host = jax.tree_util.tree_map(_to_host_leaf, frame)
                frame_digest = _array_tree_digest(frame)
                resource_profile.finish(
                    "dlno.mp2_mpi.correlation.frame_build", profile_frame_build,
                    batch_index=batch_index, term_index=work_index,
                    kind="strong", left_fragment=left, right_fragment=None,
                )
                batch_payloads.append((
                    spec, frame_host, frame_digest,
                    time.perf_counter() - frame_start if collect_timing else 0.0,
                ))
                del frame
                gc.collect()
        else:
            batch_payloads = None
        local_payload = comm.scatter(batch_payloads, root=root)
        local_result = None
        if local_payload is not None:
            spec, frame_host, frame_digest, _ = local_payload
            _, left, _ = spec
            frame = jax.tree_util.tree_map(_to_device_leaf, frame_host)
            if _array_tree_digest(frame) != frame_digest:
                raise RuntimeError(f"MPI corrupted root-gauge strong frame {spec}")
            work_index = batch_index * nproc + rank
            profile_forward = resource_profile.start()
            start = time.perf_counter() if collect_timing else None
            term_energy, pullback = jax.vjp(
                lambda mf_, domain_: strong_domain_energy(
                    mf_, domain_, static, left
                ).total, mf, frame,
            )
            if collect_timing or profile_forward is not None:
                jax.block_until_ready(term_energy)
            resource_profile.finish(
                "dlno.mp2_mpi.correlation.term_forward", profile_forward,
                batch_index=batch_index, term_index=work_index,
                kind="strong", left_fragment=left, right_fragment=None,
            )
            forward_seconds = time.perf_counter() - start if collect_timing else 0.0
            start = time.perf_counter() if collect_timing else None
            profile_reverse = resource_profile.start()
            term_mf_bar, frame_bar = pullback(np.ones((), dtype=term_energy.dtype))
            local_energy = local_energy + term_energy
            mf_bar = jax.tree_util.tree_map(_add_cotangent, mf_bar, term_mf_bar)
            jax.block_until_ready((local_energy, mf_bar, frame_bar))
            frame_bar_host = jax.tree_util.tree_map(_to_host_leaf, frame_bar)
            resource_profile.finish(
                "dlno.mp2_mpi.correlation.term_reverse", profile_reverse,
                batch_index=batch_index, term_index=work_index,
                kind="strong", left_fragment=left, right_fragment=None,
            )
            local_result = (
                spec, frame_bar_host, float(jax.device_get(term_energy)),
                forward_seconds,
                time.perf_counter() - start if collect_timing else 0.0,
                rank,
            )
            del frame, frame_bar, pullback, term_mf_bar, term_energy
            gc.collect()
        gathered = comm.gather(local_result, root=root)
        if rank == root:
            for slot, (sent, result) in enumerate(zip(batch_payloads, gathered)):
                if sent is None:
                    if result is not None:
                        raise RuntimeError("idle MPI rank returned a strong bar")
                    continue
                spec, _, expected_digest, frame_build_seconds = sent
                if result is None or result[0] != spec:
                    raise RuntimeError(f"MPI returned wrong strong term for {spec}")
                _, left, _ = spec
                frame_bar = jax.tree_util.tree_map(_to_device_leaf, result[1])
                profile_replay = resource_profile.start()
                start = time.perf_counter() if collect_timing else None
                rebuilt, pullback = jax.vjp(
                    lambda common_: build_strong_ed_domain(common_, static, left),
                    common,
                )
                if _array_tree_digest(rebuilt) != expected_digest:
                    raise RuntimeError(
                        f"root strong-domain gauge changed on replay: {left}"
                    )
                term_common_bar, = pullback(frame_bar)
                common_bar_root = jax.tree_util.tree_map(
                    _add_cotangent, common_bar_root, term_common_bar
                )
                if collect_timing or profile_replay is not None:
                    jax.block_until_ready(common_bar_root)
                resource_profile.finish(
                    "dlno.mp2_mpi.correlation.frame_replay", profile_replay,
                    batch_index=batch_index,
                    term_index=batch_index * nproc + slot,
                    kind="strong", left_fragment=left, right_fragment=None,
                )
                if collect_timing:
                    row = MP2TermResult(
                        kind="strong", left_fragment=int(left), right_fragment=None,
                        energy=float(result[2]), forward_seconds=float(result[3]),
                        reverse_seconds=float(result[4]),
                        frame_build_seconds=float(frame_build_seconds),
                        frame_replay_seconds=float(time.perf_counter() - start),
                        worker_rank=int(result[5]),
                    )
                    term_results_root.append(row)
                    progress_energy_root += row.energy
                    _report_progress(
                        reporter,
                        f"term {batch_index * nproc + slot + 1}/{len(all_work)} "
                        f"strong ED fragment {left + 1} [rank {row.worker_rank}]: "
                        f"E={row.energy:+.10f} Eh; forward/reverse="
                        f"{row.forward_seconds:.1f}/{row.reverse_seconds:.1f} s; "
                        f"frame build/replay={row.frame_build_seconds:.1f}/"
                        f"{row.frame_replay_seconds:.1f} s",
                    )
                del rebuilt, pullback, frame_bar, term_common_bar
                gc.collect()
            del batch_payloads
        del local_payload, local_result, gathered
        gc.collect()

    weak_energy, weak_mf_bar, weak_common_bar, weak_results = (
        _weak_endpoint_correlation_pullback(
            mf, common, static, weak_work, comm=comm, root=root,
            collect_timing=collect_timing, reporter=reporter,
        )
    )
    local_energy = local_energy + weak_energy
    mf_bar = jax.tree_util.tree_map(_add_cotangent, mf_bar, weak_mf_bar)
    weak_common_bar_root = _tree_sum_to_root(comm, weak_common_bar, root=root)
    if rank == root:
        common_bar_root = jax.tree_util.tree_map(
            _add_cotangent, common_bar_root, weak_common_bar_root
        )
        if collect_timing:
            term_results_root.extend(weak_results)
            spec_order = {spec: index for index, spec in enumerate(all_work)}
            term_results_root.sort(key=lambda row: spec_order[(
                row.kind, row.left_fragment,
                -1 if row.right_fragment is None else row.right_fragment,
            )])
    del weak_mf_bar, weak_common_bar, weak_common_bar_root, weak_results

    _report_progress(reporter, "reducing correlation energy and cotangents")
    corr_energy = comm.allreduce(float(local_energy), op=MPI.SUM)
    mf_bar_root = _tree_sum_to_root(comm, mf_bar, root=root)

    common_reverse_seconds = 0.0
    if rank == root:
        _report_progress(reporter, "common-orbital pullback: starting")
        profile_common_reverse = resource_profile.start()
        common_reverse_start = (
            time.perf_counter() if collect_timing else None
        )
        common_mf_bar, = common_pullback(common_bar_root)
        mf_bar_root = jax.tree_util.tree_map(
            _add_cotangent, mf_bar_root, common_mf_bar
        )
        jax.block_until_ready(mf_bar_root)
        resource_profile.finish(
            "dlno.mp2_mpi.correlation.common_reverse",
            profile_common_reverse,
            n_fragments=len(static.fragments),
        )
        if collect_timing:
            common_reverse_seconds = (
                time.perf_counter() - common_reverse_start
            )
            _report_progress(
                reporter,
                "common-orbital pullback: done in "
                f"{common_reverse_seconds:.1f} s",
            )
    details = None
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
            timing = MP2GradientTiming(
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
            if return_details or restart_enabled:
                details = _make_decomposition(
                    static, corr_energy, term_results_root, timing
                )

    checkpoint_error = None
    if rank == root and restart_enabled:
        try:
            if details is None:
                raise RuntimeError(
                    "restart-enabled MPI correlation did not collect "
                    "term diagnostics"
                )
            restart.save_record(
                "mpi_correlation_closed",
                scalars={"energy": float(corr_energy)},
                trees={"mf_bar": mf_bar_root},
                metadata=_details_restart_metadata(details),
            )
            _report_progress(
                reporter,
                "restart: saved completed common-closed correlation "
                "cotangent",
            )
        except Exception:
            checkpoint_error = _exception_text(
                "root MPI correlation checkpoint write"
            )
    _raise_if_root_failed(comm, checkpoint_error, root=root)

    resource_profile.finish(
        "dlno.mp2_mpi.correlation.total",
        profile_total,
        n_strong_terms=nstrong_terms,
        n_weak_terms=nweak_terms,
        n_batches=nbatch,
    )

    if return_details:
        if collect_timing:
            details = comm.bcast(details, root=root)
            return corr_energy, mf_bar_root, details
        raise AssertionError("unreachable return_details state")
    return corr_energy, mf_bar_root


class DLNOMP2(_SerialDLNOMP2):
    """MPI-parallel fixed-topology IAO-fragment MP2 driver."""

    @classmethod
    @_abort_collective_on_error
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
        lo_type="iao",
        lo_kwargs=None,
        topology=None,
        include_hf=True,
        parallel_scf_jk=False,
        comm=MPI.COMM_WORLD,
        root=0,
        return_details=False,
        progress=False,
        checkpoint_dir=None,
        resume=False,
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

        ``checkpoint_dir`` must name storage shared read/write by every rank.
        Set ``resume=True`` on all ranks to reuse it.  Partial MPI records are
        tied to the same ``comm`` size and ``root``; the manifest rejects a
        different layout.  One checkpoint directory may be used by only one
        running MPI job at a time.  A checkpoint-seeded ``build_mf`` must
        represent the same converged implicit DF-RHF response map.
        """
        rank = comm.Get_rank()
        nproc = comm.Get_size()
        lo_type, lo_kwargs = resolve_target_options(lo_type, lo_kwargs)
        progress_enabled = _progress_enabled(progress)
        local_schedule = (
            int(root),
            bool(parallel_scf_jk),
            bool(include_hf),
            bool(return_details),
            progress_enabled,
            lo_type,
            _semantic_tuple(lo_kwargs),
            None if checkpoint_dir is None else str(
                Path(checkpoint_dir).expanduser().resolve()
            ),
            bool(resume),
        )
        schedules = comm.allgather(local_schedule)
        if len(set(schedules)) != 1:
            raise ValueError(
                "root, parallel_scf_jk, include_hf, return_details, progress, "
                "lo_type, lo_kwargs, checkpoint_dir, and resume must be consistent "
                "on all MPI "
                "ranks"
            )
        if root < 0 or root >= nproc:
            raise ValueError(f"root={root} is invalid for {nproc} MPI ranks")
        if resume and checkpoint_dir is None:
            raise ValueError("resume=True requires checkpoint_dir")
        parallel_scf_jk = bool(parallel_scf_jk) and nproc > 1
        reporter = _progress_reporter(progress, rank=rank, root=root)
        overall_start = time.perf_counter()
        if (
            getattr(mol, "exp", None) is not None
            or getattr(mol, "ctr_coeff", None) is not None
        ):
            raise NotImplementedError(
                "MPI DLNOMP2 currently differentiates nuclear "
                "coordinates only; build mol with trace_exp=False and "
                "trace_ctr_coeff=False"
            )
        if thresholds is None:
            thresholds = DLNOThresholds()

        scf_builder = build_mf
        if checkpoint_dir is not None:
            def scf_builder(mol_):
                return _fix_restart_mo_phases(build_mf(mol_))

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
                profile_scf_forward = resource_profile.start()
                with (
                    config_update("pyscfad_scf_implicit_diff", True),
                    config_update("pyscfad_scf_first_order_custom", False),
                ):
                    if parallel_scf_jk:
                        with scf_executor.root_session(final=False):
                            mf, scf_pullback = jax.vjp(scf_builder, mol)
                            jax.block_until_ready(mf.e_tot)
                    else:
                        mf, scf_pullback = jax.vjp(scf_builder, mol)
                        jax.block_until_ready(mf.e_tot)
                resource_profile.finish(
                    "dlno.mp2_mpi.scf_forward",
                    profile_scf_forward,
                    parallel_scf_jk=parallel_scf_jk,
                )
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
                canonical = {
                    "mo_coeff": numpy.asarray(mf.mo_coeff),
                    "mo_energy": numpy.asarray(mf.mo_energy),
                    "mo_occ": numpy.asarray(mf.mo_occ),
                    "e_tot": float(mf.e_tot),
                }
                restart_payload = _mpi_restart_scientific_payload(
                    mol,
                    mf,
                    frag_lolist=frag_lolist,
                    frag_atmlist=frag_atmlist,
                    frozen=frozen,
                    thresholds=thresholds,
                    pair_energy_model=pair_energy_model,
                    force_full_domains=force_full_domains,
                    include_hf=include_hf,
                    lo_type=lo_type,
                    lo_kwargs=lo_kwargs,
                    nproc=nproc,
                    root=root,
                )
                restart = RestartManager(
                    checkpoint_dir,
                    resume=resume,
                    method="mpi-dlno-mp2",
                    scientific_payload=restart_payload,
                    initialize=True,
                )
                if topology is not None and not isinstance(
                    topology,
                    (DomainTopology, DomainSelections),
                ):
                    raise TypeError(
                        "topology must be DomainTopology or "
                        "DomainSelections"
                    )
                if topology is not None:
                    _validate_target_options(
                        topology,
                        lo_type=lo_type,
                        lo_kwargs=lo_kwargs,
                        frag_lolist=frag_lolist,
                        frag_atmlist=frag_atmlist,
                        frozen=frozen,
                    )
                topology_start = time.perf_counter()
                profile_topology = resource_profile.start()
                _report_progress(
                    reporter, "fixed IAO fragment topology: starting"
                )
                saved_topology = None
                if resume and restart.enabled:
                    saved_topology = restart.load_static(
                        expected_type=DomainSelections
                    )
                if saved_topology is not None:
                    if isinstance(
                        topology, DomainSelections
                    ):
                        restart.bind_static(topology)
                    elif isinstance(topology, DomainTopology):
                        from ._selection import build_domain_selections

                        supplied_static = stop_trace(
                            lambda mf_: build_domain_selections(
                                mf_, topology
                            )
                        )(mf)
                        restart.bind_static(supplied_static)
                    fixed_topology = saved_topology
                    _report_progress(
                        reporter,
                        "restart: loaded fixed IAO fragment topology",
                    )
                elif topology is None:
                    fixed_topology = stop_trace(
                        lambda mf_: cls.build_static_topology(
                            mf_,
                            frozen=frozen,
                            frag_lolist=frag_lolist,
                            frag_atmlist=frag_atmlist,
                            thresholds=thresholds,
                            pair_energy_model=pair_energy_model,
                            force_full_domains=force_full_domains,
                            lo_type=lo_type,
                            lo_kwargs=lo_kwargs,
                        )
                    )(mf)
                elif isinstance(topology, DomainTopology):
                    from ._selection import build_domain_selections

                    fixed_topology = stop_trace(
                        lambda mf_: build_domain_selections(
                            mf_, topology
                        )
                    )(mf)
                elif isinstance(topology, DomainSelections):
                    fixed_topology = topology
                else:
                    raise TypeError(
                        "topology must be DomainTopology or "
                        "DomainSelections"
                    )
                _validate_target_options(
                    fixed_topology,
                    lo_type=lo_type,
                    lo_kwargs=lo_kwargs,
                    frag_lolist=frag_lolist,
                    frag_atmlist=frag_atmlist,
                    frozen=frozen,
                )
                if restart.enabled and saved_topology is None:
                    restart.save_static(fixed_topology)
                _report_progress(
                    reporter,
                    f"target localization: mode={lo_type}; options="
                    f"{_semantic_tuple(lo_kwargs)!r}",
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
                resource_profile.finish(
                    "dlno.mp2_mpi.topology",
                    profile_topology,
                    lo_type=lo_type,
                    n_fragments=len(fixed_topology.fragments),
                    n_strong_pairs=nstrong_pairs,
                    n_weak_terms=nweak_terms,
                )
            except Exception:  # pragma: no cover - multi-rank failure path
                topology_error = _exception_text(
                    "root IAO fragment topology setup"
                )
                fixed_topology = canonical = restart = restart_payload = None
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
            restart_payload = None
            topology_error = None

        topology_error = comm.bcast(topology_error, root=root)
        if topology_error is not None:
            if scf_executor is not None:
                scf_executor.close_local()
            raise RuntimeError(topology_error)
        canonical = comm.bcast(canonical, root=root)
        restart_payload = comm.bcast(restart_payload, root=root)

        pre_scf = None
        pre_scf_error = None
        e_hf = hf_bar = None
        if rank == root:
            try:
                e_hf, hf_pullback = jax.vjp(lambda mf_: mf_.e_tot, mf)
                hf_bar, = hf_pullback(
                    np.ones((), dtype=np.asarray(e_hf).dtype)
                )
                if resume and restart.enabled:
                    pre_scf = restart.load_record(
                        "pre_scf",
                        templates={"mf_bar": hf_bar},
                        missing_ok=True,
                    )
            except Exception:
                pre_scf_error = _exception_text(
                    "root pre-SCF restart load"
                )
        pre_scf_error = comm.bcast(pre_scf_error, root=root)
        if pre_scf_error is not None:
            if scf_executor is not None:
                scf_executor.close_local()
            raise RuntimeError(pre_scf_error)
        pre_scf_payload = None
        if rank == root and pre_scf is not None:
            pre_scf_payload = (
                float(pre_scf.scalars["energy"]),
                float(pre_scf.scalars["corr_energy"]),
            )
        pre_scf_payload = comm.bcast(pre_scf_payload, root=root)

        if rank != root:
            if parallel_scf_jk:
                mf.mo_coeff = canonical["mo_coeff"]
                mf.mo_energy = canonical["mo_energy"]
                mf.mo_occ = canonical["mo_occ"]
                mf.e_tot = canonical["e_tot"]
                mf.converged = True
            elif pre_scf_payload is None:
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

        restart_error = None
        if rank != root and pre_scf_payload is None:
            try:
                restart = RestartManager(
                    checkpoint_dir,
                    resume=resume,
                    method="mpi-dlno-mp2",
                    scientific_payload=restart_payload,
                    initialize=False,
                )
            except Exception:
                restart_error = _exception_text(
                    f"MPI restart setup on rank {rank}"
                )
        _raise_if_any_rank_failed(comm, restart_error)
        if pre_scf_payload is None or parallel_scf_jk:
            _verify_shared_reference(
                comm,
                canonical,
                mf,
                verify_df_source=parallel_scf_jk,
            )

        if pre_scf_payload is not None:
            energy, corr_energy = pre_scf_payload
            mf_bar_root = (
                pre_scf.trees["mf_bar"] if rank == root else None
            )
            _report_progress(
                reporter,
                "restart: loaded pre-SCF total cotangent; all local MP2 "
                "work is skipped",
            )
            if return_details:
                closed = None
                details_error = None
                if rank == root:
                    try:
                        closed = restart.load_record(
                            "mpi_correlation_closed",
                            templates={"mf_bar": _zero_mf_cotangent(mf)},
                            missing_ok=False,
                        )
                        details = _details_from_restart_metadata(
                            fixed_topology,
                            corr_energy,
                            closed.metadata,
                        )
                    except Exception:
                        details_error = _exception_text(
                            "root restarted MP2 diagnostics load"
                        )
                        details = None
                _raise_if_root_failed(comm, details_error, root=root)
                details = comm.bcast(details, root=root)
                corr_result = (corr_energy, mf_bar_root, details)
            else:
                corr_result = (corr_energy, mf_bar_root)
        else:
            _report_progress(
                reporter, "MPI correlation energy/gradient: starting"
            )
            corr_result = correlation_value_and_grad(
                mf,
                fixed_topology,
                comm=comm,
                root=root,
                return_details=return_details,
                progress=progress,
                restart=restart,
            )
            corr_energy, mf_bar_root = corr_result[:2]
            checkpoint_error = None
            if rank == root:
                try:
                    if include_hf:
                        _report_progress(
                            reporter, "adding the Hartree-Fock energy seed"
                        )
                        mf_bar_root = jax.tree_util.tree_map(
                            _add_cotangent, mf_bar_root, hf_bar
                        )
                        energy = float(e_hf) + corr_energy
                    else:
                        energy = corr_energy
                    jax.block_until_ready(mf_bar_root)
                    if restart.enabled:
                        restart.save_record(
                            "pre_scf",
                            scalars={
                                "energy": float(energy),
                                "corr_energy": float(corr_energy),
                            },
                            trees={"mf_bar": mf_bar_root},
                            metadata={"include_hf": bool(include_hf)},
                        )
                        _report_progress(
                            reporter,
                            "restart: saved pre-SCF total cotangent",
                        )
                except Exception:
                    checkpoint_error = _exception_text(
                        "root pre-SCF checkpoint write"
                    )
            _raise_if_root_failed(comm, checkpoint_error, root=root)
            energy = comm.bcast(
                energy if rank == root else None, root=root
            )

        response_setup_error = None
        mol_bar = None
        if rank == root:
            try:
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
            energy = float(energy)

        response_setup_error = comm.bcast(
            response_setup_error, root=root
        )
        if response_setup_error is not None:
            if scf_executor is not None:
                scf_executor.close_local()
            raise RuntimeError(response_setup_error)

        response_error = None
        profile_scf_response = resource_profile.start()
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
        resource_profile.finish(
            "dlno.mp2_mpi.scf_response",
            profile_scf_response,
            role="root" if rank == root else "worker",
            parallel_scf_jk=parallel_scf_jk,
        )
        if return_details:
            return energy, mol_bar, corr_result[2]
        return energy, mol_bar
