"""MPI fragment parallelism for IAO-DLNO-CCSD(T) gradients.

Each MPI task owns complete fragment calculations.  In particular, a worker
constructs one fragment LIS, evaluates its CCSD(T) and matching MP2
subtraction, and closes that entire scalar calculation back into the shared
``(mf, common)`` coordinates before communicating any cotangent.  Raw LIS
coefficients and LIS-frame cotangents never cross MPI, so independently
constructed fragment gauges do not need to be aligned between ranks.

The molecule-wide IAO-DLNO-MP2 correction is evaluated by the existing
gauge-safe MPI implementation.  Root then combines the fragment, MP2, and HF
mean-field cotangents and applies exactly one implicit SCF pullback.
"""

from __future__ import annotations

import gc
import os
import time
import traceback
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy
from mpi4py import MPI

from pyscfad import config_update
from pyscfad.df.mpi_df_jk import MPIDFJKExecutor, ServiceExit
from pyscfad.ops import stop_trace

from ._output import (
    emit_lines,
    energy_summary_lines,
    fragment_energy_lines,
    lis_active_space_lines,
    lis_dimensions_from_static,
    local_correlation_settings_lines,
    mp2_prescreened_domain_lines,
    nuclear_force_lines,
)
from .ccsd import DLNOCCSD as _SerialDLNOCCSD
from .iao_ccsd import (
    _add_cotangent,
    _fragment_value_and_grad,
    _validate_solver_options,
    build_iao_dlno_ccsd_static_selections,
)
from .iao_lis import (
    IAO_LIS_INTERNAL_RANK_THRESHOLD,
    IAOFragmentLISStaticSelections,
)
from .iao_mp2 import IAOFragmentMP2Thresholds
from .iao_mp2_grad import rebuild_iao_mp2_common
from .iao_mp2_mpi import (
    _progress_enabled,
    _to_device_leaf,
    _to_host_leaf,
    _tree_sum_to_root,
    _verify_shared_gauge,
    _zero_term_cotangents,
    correlation_value_and_grad as _mp2_correlation_value_and_grad,
)


__all__ = [
    "DLNOCCSD",
    "IAODLNOCCSDMPIFragmentResult",
    "IAODLNOCCSDMPIResult",
]


@dataclass(frozen=True)
class IAODLNOCCSDMPIFragmentResult:
    """Scalar record for one complete fragment job."""

    fragment_index: int
    worker_rank: int
    e_mp2_lis: float
    e_ccsd: float
    e_ccsd_t: float
    lis_occupied: int
    lis_virtual: int
    wall_seconds: float


@dataclass(frozen=True)
class IAODLNOCCSDMPIResult:
    """MPI energy decomposition, fragment ownership, and scalar timings."""

    e_hf: float
    e_iao_mp2: float
    e_mp2_lis: float
    e_ccsd: float
    e_ccsd_t: float
    e_corr: float
    e_total: float
    fragments: tuple[IAODLNOCCSDMPIFragmentResult, ...]
    nproc: int
    total_seconds: float


def _progress_reporter(progress, *, rank, root):
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
        reporter(f"[IAO-CC] {message}")


def _report_summary(reporter, lines):
    if reporter is None:
        return
    emit_lines(lambda line: _report_progress(reporter, line), lines)


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


def _validate_cc_cderi(mf, rank):
    """Fail early when a rank cannot supply the DF integrals needed by CC."""
    with_df = getattr(mf, "with_df", None)
    if with_df is None or not hasattr(with_df, "_get_cderi_source"):
        raise RuntimeError(
            f"MPI rank {rank} has no density-fitting CDERI source for CCSD"
        )
    source = with_df._get_cderi_source()
    if source is None:
        raise RuntimeError(
            f"MPI rank {rank} has no built CDERI source.  A non-root "
            "build_mf must build or attach real DF three-index integrals; "
            "an MP2-only empty placeholder is insufficient for CCSD(T)."
        )
    if isinstance(source, (str, bytes, os.PathLike)):
        path = os.fsdecode(os.fspath(source))
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"MPI rank {rank} CDERI file does not exist: {path}"
            )


def _build_static_selections(
    mf,
    *,
    frag_lolist,
    frag_atmlist,
    frozen,
    thresholds,
    pair_energy_model,
    force_full_domains,
    thresh_occ,
    thresh_vir,
    internal_rank_threshold,
    static_selections,
):
    if static_selections is None:
        return stop_trace(
            lambda mf_: build_iao_dlno_ccsd_static_selections(
                mf_,
                frag_lolist=frag_lolist,
                frag_atmlist=frag_atmlist,
                frozen=frozen,
                thresholds=thresholds,
                pair_energy_model=pair_energy_model,
                force_full_domains=force_full_domains,
                thresh_occ=thresh_occ,
                thresh_vir=thresh_vir,
                internal_rank_threshold=internal_rank_threshold,
            )
        )(mf)
    if not isinstance(static_selections, IAOFragmentLISStaticSelections):
        raise TypeError(
            "static_selections must be IAOFragmentLISStaticSelections"
        )
    return static_selections


def _value_and_grad(
    mol,
    *,
    build_mf,
    frag_lolist=None,
    frag_atmlist=None,
    frozen=None,
    thresholds=None,
    pair_energy_model="multipole",
    force_full_domains=False,
    thresh_occ=1e-4,
    thresh_vir=1e-5,
    internal_rank_threshold=IAO_LIS_INTERNAL_RANK_THRESHOLD,
    ccsd_t=False,
    dcsd=False,
    verbose_imp=0,
    static_selections=None,
    parallel_scf_jk=False,
    comm=MPI.COMM_WORLD,
    root=0,
    return_details=False,
    progress=False,
):
    """Implementation body for :meth:`DLNOCCSD.value_and_grad`."""
    rank = comm.Get_rank()
    nproc = comm.Get_size()
    root = int(root)
    if root < 0 or root >= nproc:
        raise ValueError(f"root={root} is invalid for {nproc} MPI ranks")
    _validate_solver_options(ccsd_t=ccsd_t, dcsd=dcsd)
    progress_enabled = _progress_enabled(progress)
    schedules = comm.allgather((
        root,
        bool(return_details),
        progress_enabled,
        bool(parallel_scf_jk),
    ))
    if len(set(schedules)) != 1:
        raise ValueError(
            "root, return_details, progress, and parallel_scf_jk must be "
            "consistent on all ranks"
        )
    parallel_scf_jk = bool(parallel_scf_jk) and nproc > 1
    reporter = _progress_reporter(progress, rank=rank, root=root)
    started = time.perf_counter()

    if (
        getattr(mol, "exp", None) is not None
        or getattr(mol, "ctr_coeff", None) is not None
    ):
        raise NotImplementedError(
            "MPI IAO-DLNO-CCSD(T) currently differentiates nuclear "
            "coordinates only; build mol with trace_exp=False and "
            "trace_ctr_coeff=False"
        )
    if thresholds is None:
        thresholds = IAOFragmentMP2Thresholds()

    scf_executor = None
    mf = None
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
                mf.with_df.build()
                _validate_cc_cderi(mf, rank)
            except Exception:
                worker_setup_error = _exception_text(
                    f"MPI DF-J/K worker setup on rank {rank}"
                )
        worker_setup_errors = tuple(
            error for error in comm.allgather(worker_setup_error)
            if error is not None
        )
        if worker_setup_errors:
            scf_executor.close_local()
            raise RuntimeError("\n".join(worker_setup_errors))

    # Root alone owns the converged SCF tape.  Workers receive its exact
    # canonical orbitals and construct an otherwise equivalent DF skeleton.
    setup_error = None
    if rank == root:
        try:
            scf_label = "MPI DF-RHF" if parallel_scf_jk else "DF-RHF"
            _report_progress(
                reporter, f"{scf_label} SCF and VJP setup: starting"
            )
            scf_started = time.perf_counter()
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
            canonical = {
                "mo_coeff": numpy.asarray(mf.mo_coeff),
                "mo_energy": numpy.asarray(mf.mo_energy),
                "mo_occ": numpy.asarray(mf.mo_occ),
                "e_tot": float(mf.e_tot),
            }
            _report_progress(
                reporter,
                f"{scf_label} SCF and VJP setup: done in "
                f"{time.perf_counter() - scf_started:.1f} s; "
                f"E_HF={canonical['e_tot']:+.10f} Eh",
            )
        except Exception:  # pragma: no cover - multi-rank failure path
            if scf_executor is not None:
                scf_executor.stop_workers()
            setup_error = _exception_text("root SCF/VJP setup")
            mf = scf_pullback = canonical = None
    else:
        scf_pullback = canonical = None
        if parallel_scf_jk:
            try:
                service_exit = scf_executor.serve(mf.with_df)
                if service_exit is not ServiceExit.PAUSED:
                    raise RuntimeError(
                        "MPI DF-J/K forward worker service stopped before "
                        "the SCF completed"
                    )
            except Exception:
                setup_error = _exception_text(
                    f"MPI DF-J/K forward worker rank {rank}"
                )
    setup_errors = tuple(
        error for error in comm.allgather(setup_error) if error is not None
    )
    if setup_errors:
        if scf_executor is not None:
            scf_executor.close_local()
        raise RuntimeError("\n".join(setup_errors))
    canonical = comm.bcast(canonical, root=root)

    worker_error = None
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
            except TypeError:
                worker_error = (
                    f"non-root build_mf failed on MPI rank {rank}; it must "
                    "accept mo_coeff_init, mo_energy_init, mo_occ_init, and "
                    "e_tot_init, must not run SCF, and must attach a real "
                    "rank-accessible CDERI source:\n"
                    + traceback.format_exc()
                )
            except Exception:
                worker_error = _exception_text(
                    f"non-root build_mf on MPI rank {rank}"
                )
    _raise_if_any_rank_failed(comm, worker_error)

    # Root makes every discrete topology/rank decision and owns the saved
    # common-orbital VJP.  The exact continuous common value is then copied to
    # every rank, including root, before fragment differentiation.
    static_error = None
    if rank == root:
        try:
            _report_progress(
                reporter, "fixed fragment topology and LIS ranks: starting"
            )
            static_started = time.perf_counter()
            static_selections = _build_static_selections(
                mf,
                frag_lolist=frag_lolist,
                frag_atmlist=frag_atmlist,
                frozen=frozen,
                thresholds=thresholds,
                pair_energy_model=pair_energy_model,
                force_full_domains=force_full_domains,
                thresh_occ=thresh_occ,
                thresh_vir=thresh_vir,
                internal_rank_threshold=internal_rank_threshold,
                static_selections=static_selections,
            )
            mp2_static = static_selections.mp2_static
            common_original, common_pullback = jax.vjp(
                lambda mf_: rebuild_iao_mp2_common(mf_, mp2_static), mf
            )
            payload = (
                static_selections,
                jax.tree_util.tree_map(_to_host_leaf, common_original),
            )
            _report_progress(
                reporter,
                "fixed fragment topology and LIS ranks: done in "
                f"{time.perf_counter() - static_started:.1f} s; "
                f"fragments={len(static_selections.fragments)}",
            )
            if reporter is not None:
                _report_summary(
                    reporter,
                    local_correlation_settings_lines(
                        static_selections,
                        ccsd_t=ccsd_t,
                        dcsd=dcsd,
                        nproc=nproc,
                        pair_energy_model=pair_energy_model,
                        force_full_domains=force_full_domains,
                    ),
                )
                _report_summary(
                    reporter,
                    mp2_prescreened_domain_lines(static_selections),
                )
                fixed_lis_occ, fixed_lis_vir = lis_dimensions_from_static(
                    static_selections
                )
                _report_summary(
                    reporter,
                    lis_active_space_lines(
                        static_selections,
                        fixed_lis_occ,
                        fixed_lis_vir,
                        worker_ranks=tuple(
                            fragment_index % nproc
                            for fragment_index in range(
                                len(static_selections.fragments)
                            )
                        ),
                    ),
                )
        except Exception:  # pragma: no cover - multi-rank failure path
            static_error = _exception_text(
                "root fragment topology/common-orbital setup"
            )
            common_pullback = payload = None
    else:
        common_pullback = payload = None
    _raise_if_root_failed(comm, static_error, root=root)
    static_selections, common_host = comm.bcast(payload, root=root)
    common = jax.tree_util.tree_map(_to_device_leaf, common_host)
    if rank == root:
        del common_original, payload

    local_canonical = (
        numpy.asarray(mf.mo_coeff),
        numpy.asarray(mf.mo_energy),
        numpy.asarray(mf.mo_occ),
    )
    _verify_shared_gauge(comm, local_canonical, common, mf)
    _report_progress(reporter, "shared SCF/common orbital gauge verified")

    nfragment = len(static_selections.fragments)
    local_indices = tuple(range(rank, nfragment, nproc))
    cderi_error = None
    if local_indices:
        try:
            _validate_cc_cderi(mf, rank)
        except Exception:
            cderi_error = _exception_text(
                f"CDERI preflight on MPI rank {rank}"
            )
    _raise_if_any_rank_failed(comm, cderi_error)

    _report_progress(
        reporter,
        f"fragment pullbacks: {nfragment} fragments on {nproc} ranks; "
        "round-robin ownership",
    )
    local_mf_bar, local_common_bar = _zero_term_cotangents(mf, common)
    local_records = []
    nbatch = (nfragment + nproc - 1) // nproc
    for batch_index in range(nbatch):
        fragment_index = batch_index * nproc + rank
        fragment_error = None
        if fragment_index < nfragment:
            try:
                fragment_started = time.perf_counter()
                fragment = _fragment_value_and_grad(
                    mf,
                    common,
                    static_selections,
                    fragment_index,
                    verbose_imp=verbose_imp,
                    ccsd_t=ccsd_t,
                    dcsd=dcsd,
                )
                local_mf_bar = jax.tree_util.tree_map(
                    _add_cotangent, local_mf_bar, fragment.mf_bar
                )
                local_common_bar = jax.tree_util.tree_map(
                    _add_cotangent, local_common_bar, fragment.common_bar
                )
                jax.block_until_ready((local_mf_bar, local_common_bar))
                local_records.append(IAODLNOCCSDMPIFragmentResult(
                    fragment_index=int(fragment_index),
                    worker_rank=int(rank),
                    e_mp2_lis=float(jax.device_get(fragment.e_mp2_lis)),
                    e_ccsd=float(jax.device_get(fragment.e_ccsd)),
                    e_ccsd_t=float(jax.device_get(fragment.e_ccsd_t)),
                    lis_occupied=int(fragment.lis_occupied),
                    lis_virtual=int(fragment.lis_virtual),
                    wall_seconds=float(
                        time.perf_counter() - fragment_started
                    ),
                ))
                del fragment
                gc.collect()
            except Exception:  # pragma: no cover - MPI failure jobs
                fragment_error = _exception_text(
                    f"fragment {fragment_index} on MPI rank {rank}"
                )
        # Synchronize once per round-robin batch.  A failed fragment therefore
        # waits at most for the other ranks' current fragments, not for their
        # entire remaining queues, before the communicator is aborted.
        _raise_if_any_rank_failed(comm, fragment_error)

    gathered_records = comm.gather(tuple(local_records), root=root)
    coverage_error = None
    if rank == root:
        try:
            records = tuple(sorted(
                (
                    record
                    for rank_records in gathered_records
                    for record in rank_records
                ),
                key=lambda record: record.fragment_index,
            ))
            indices = tuple(record.fragment_index for record in records)
            expected = tuple(range(nfragment))
            if indices != expected:
                raise RuntimeError(
                    "MPI fragment coverage is not exactly once: "
                    f"expected {expected}, received {indices}"
                )
            components = {
                "e_mp2_lis": sum(record.e_mp2_lis for record in records),
                "e_ccsd": sum(record.e_ccsd for record in records),
                "e_ccsd_t": sum(record.e_ccsd_t for record in records),
            }
            coverage_payload = (records, components)
        except Exception:  # pragma: no cover - defensive MPI invariant
            coverage_error = _exception_text("root fragment coverage check")
            coverage_payload = None
    else:
        coverage_payload = None
    _raise_if_root_failed(comm, coverage_error, root=root)
    records, components = comm.bcast(coverage_payload, root=root)
    if rank == root and reporter is not None:
        _report_summary(
            reporter,
            fragment_energy_lines(
                records,
                correlated_method="DCSD" if dcsd else "CCSD",
                include_triples=ccsd_t,
            ),
        )

    _report_progress(reporter, "reducing fragment cotangents to root")
    cc_mf_bar_root = _tree_sum_to_root(
        comm, local_mf_bar, root=root
    )
    cc_common_bar_root = _tree_sum_to_root(
        comm, local_common_bar, root=root
    )
    if rank == root:
        common_mf_bar, = common_pullback(cc_common_bar_root)
        cc_mf_bar_root = jax.tree_util.tree_map(
            _add_cotangent, cc_mf_bar_root, common_mf_bar
        )
        jax.block_until_ready(cc_mf_bar_root)
        del common_mf_bar, cc_common_bar_root, common_pullback
    del local_mf_bar, local_common_bar, common, common_host
    gc.collect()

    # The complete strong+weak MP2 correction has its own bounded, gauge-safe
    # MPI pullback and returns an already common-closed mf cotangent on root.
    _report_progress(reporter, "molecule-wide MPI IAO-DLNO-MP2: starting")
    mp2_result = _mp2_correlation_value_and_grad(
        mf,
        static_selections.mp2_static,
        comm=comm,
        root=root,
        return_details=False,
        progress=progress,
    )
    e_iao_mp2, mp2_mf_bar_root = mp2_result
    e_iao_mp2 = float(e_iao_mp2)

    e_corr = (
        components["e_ccsd"]
        + components["e_ccsd_t"]
        - components["e_mp2_lis"]
        + e_iao_mp2
    )
    energy = float(canonical["e_tot"]) + e_corr

    response_setup_error = None
    mol_bar = None
    total_mf_bar = None
    if rank == root:
        try:
            e_hf, hf_pullback = jax.vjp(lambda mf_: mf_.e_tot, mf)
            hf_bar, = hf_pullback(
                jnp.ones((), dtype=jnp.asarray(e_hf).dtype)
            )
            total_mf_bar = jax.tree_util.tree_map(
                _add_cotangent, cc_mf_bar_root, mp2_mf_bar_root
            )
            total_mf_bar = jax.tree_util.tree_map(
                _add_cotangent, total_mf_bar, hf_bar
            )
            _report_progress(
                reporter,
                "single implicit SCF response: " + (
                    "starting with MPI-parallel DF J/K and coordinate VJP"
                    if parallel_scf_jk else "starting on root"
                ),
            )
            response_started = time.perf_counter()
        except Exception:  # pragma: no cover - multi-rank failure path
            response_setup_error = _exception_text(
                "root implicit SCF response setup"
            )
    response_setup_error = comm.bcast(response_setup_error, root=root)
    if response_setup_error is not None:
        if scf_executor is not None:
            scf_executor.close_local()
        raise RuntimeError(response_setup_error)

    response_error = None
    if rank == root:
        try:
            if parallel_scf_jk:
                with scf_executor.root_session(final=True):
                    mol_bar, = scf_pullback(total_mf_bar)
                    jax.block_until_ready(mol_bar)
            else:
                mol_bar, = scf_pullback(total_mf_bar)
                jax.block_until_ready(mol_bar)
            gradient_norm = float(numpy.linalg.norm(
                numpy.asarray(mol_bar.coords)
            ))
            _report_progress(
                reporter,
                "single implicit SCF response: done in "
                f"{time.perf_counter() - response_started:.1f} s; "
                f"|gradient|={gradient_norm:.6e} Eh/bohr",
            )
            del hf_pullback, hf_bar, total_mf_bar
        except Exception:  # pragma: no cover - multi-rank failure path
            response_error = _exception_text("root implicit SCF response")
    elif parallel_scf_jk:
        try:
            service_exit = scf_executor.serve(mf.with_df)
            if service_exit is not ServiceExit.STOPPED:
                raise RuntimeError(
                    "MPI DF-J/K reverse worker service paused before the "
                    "SCF pullback completed"
                )
        except Exception:
            response_error = _exception_text(
                f"implicit SCF response worker rank {rank}"
            )
    _raise_if_any_rank_failed(comm, response_error)

    total_seconds = comm.allreduce(
        time.perf_counter() - started, op=MPI.MAX
    )
    _report_progress(
        reporter,
        f"complete: E={energy:.12f} Eh; wall={total_seconds:.1f} s",
    )
    if rank == root and reporter is not None:
        _report_summary(
            reporter,
            energy_summary_lines(
                e_hf=canonical["e_tot"],
                e_iao_mp2=e_iao_mp2,
                e_mp2_lis=components["e_mp2_lis"],
                e_ccsd=components["e_ccsd"],
                e_ccsd_t=components["e_ccsd_t"],
                e_corr=e_corr,
                e_total=energy,
                correlated_method="DCSD" if dcsd else "CCSD",
                include_triples=ccsd_t,
            ),
        )
        _report_summary(
            reporter,
            nuclear_force_lines(mol, numpy.asarray(mol_bar.coords)),
        )
    if return_details:
        if rank == root:
            details = IAODLNOCCSDMPIResult(
                e_hf=float(canonical["e_tot"]),
                e_iao_mp2=e_iao_mp2,
                e_mp2_lis=float(components["e_mp2_lis"]),
                e_ccsd=float(components["e_ccsd"]),
                e_ccsd_t=float(components["e_ccsd_t"]),
                e_corr=float(e_corr),
                e_total=float(energy),
                fragments=records,
                nproc=int(nproc),
                total_seconds=float(total_seconds),
            )
        else:
            details = None
        details = comm.bcast(details, root=root)
        return energy, mol_bar, details
    return energy, mol_bar


class DLNOCCSD(_SerialDLNOCCSD):
    """IAO-DLNO-CCSD(T) with MPI distribution over complete fragments."""

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
        thresh_occ=1e-4,
        thresh_vir=1e-5,
        internal_rank_threshold=IAO_LIS_INTERNAL_RANK_THRESHOLD,
        ccsd_t=False,
        dcsd=False,
        verbose_imp=0,
        static_selections=None,
        parallel_scf_jk=False,
        comm=None,
        root=0,
        return_details=False,
        progress=False,
    ):
        """Return total energy on all ranks and the nuclear gradient on root.

        Root calls ``build_mf(mol)`` and owns the converged SCF VJP.  Every
        non-root rank calls ``build_mf`` with ``mo_coeff_init``,
        ``mo_energy_init``, ``mo_occ_init``, and ``e_tot_init``.  That worker
        path must not run SCF, but it must construct a matching DF object and
        build or attach a real rank-accessible CDERI source because fragment
        CCSD(T), unlike the integral-direct MP2 terms, reads those integrals
        in both its forward and reverse passes.

        Complete fragment VJPs are assigned round-robin.  No LIS coefficient
        or LIS-frame cotangent is communicated.  With ``return_details=True``
        a third, immutable decomposition object is broadcast to every rank.
        ``progress`` follows the same root-only bool/callable convention as
        :mod:`pyscfad.dlno.iao_mp2_mpi` and must be consistent on all ranks.
        With ``parallel_scf_jk=True``, workers additionally serve the forward
        DF-SCF J/K builds, implicit density response, and distributed
        three-centre coordinate VJP before fragment work and during the final
        SCF pullback.
        """
        if comm is None:
            comm = MPI.COMM_WORLD
        try:
            return _value_and_grad(
                mol,
                build_mf=build_mf,
                frag_lolist=frag_lolist,
                frag_atmlist=frag_atmlist,
                frozen=frozen,
                thresholds=thresholds,
                pair_energy_model=pair_energy_model,
                force_full_domains=force_full_domains,
                thresh_occ=thresh_occ,
                thresh_vir=thresh_vir,
                internal_rank_threshold=internal_rank_threshold,
                ccsd_t=ccsd_t,
                dcsd=dcsd,
                verbose_imp=verbose_imp,
                static_selections=static_selections,
                parallel_scf_jk=parallel_scf_jk,
                comm=comm,
                root=root,
                return_details=return_details,
                progress=progress,
            )
        except Exception:
            if comm.Get_size() > 1:  # pragma: no cover - MPI failure path
                if comm.Get_rank() == int(root):
                    traceback.print_exc()
                comm.Abort(1)
            raise

    def kernel(self, *args, comm=None, root=0, **kwargs):
        """Preserve serial energy-only behavior on one rank.

        Option A parallelizes the progressive energy-and-gradient driver.  A
        separately constructed multi-rank instance kernel has no root-owned
        SCF pullback and is intentionally outside that contract.
        """
        if comm is None:
            comm = MPI.COMM_WORLD
        if comm.Get_size() != 1:
            raise NotImplementedError(
                "multi-rank Option A is exposed by "
                "DLNOCCSD.value_and_grad; the instance energy-only kernel "
                "remains serial"
            )
        if int(root) != comm.Get_rank():
            raise ValueError("root must identify the sole COMM_SELF rank")
        return super().kernel(*args, **kwargs)
