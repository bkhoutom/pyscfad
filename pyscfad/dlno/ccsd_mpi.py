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

For fixed LIS selection, the root rank alone fixes the fragment/ED topology
and constructs ED orbital frames.  It streams at most one frame per MPI rank
in each batch; the ranks independently form the target-conditioned MP2
density and select the fixed LIS labels.  Only those small label records are
gathered, so neither ED frames nor MP2-density intermediates accumulate over
all fragments.
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
    _restart_scientific_payload,
    _validate_solver_options,
    build_iao_dlno_ccsd_domain_selections,
)
from .iao_lis import (
    IAO_LIS_INTERNAL_RANK_THRESHOLD,
    IAOLISFragmentStaticSelection,
    IAOFragmentLISStaticSelections,
    build_iao_lis_fragment_static_selection,
)
from .iao_mp2 import IAOFragmentMP2Thresholds, _fix_restart_mo_phases
from .iao_mp2_grad import build_strong_ed_domain, rebuild_iao_mp2_common
from .iao_mp2_mpi import (
    _progress_enabled,
    _to_device_leaf,
    _to_host_leaf,
    _tree_sum_to_root,
    _verify_shared_reference,
    _verify_shared_gauge,
    _zero_term_cotangents,
    correlation_value_and_grad as _mp2_correlation_value_and_grad,
)
from ._restart import RestartManager


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


def _fragment_record_metadata(record):
    return {
        "fragment_index": int(record.fragment_index),
        "worker_rank": int(record.worker_rank),
        "e_mp2_lis": float(record.e_mp2_lis),
        "e_ccsd": float(record.e_ccsd),
        "e_ccsd_t": float(record.e_ccsd_t),
        "lis_occupied": int(record.lis_occupied),
        "lis_virtual": int(record.lis_virtual),
        "wall_seconds": float(record.wall_seconds),
    }


def _fragment_record_from_metadata(row):
    return IAODLNOCCSDMPIFragmentResult(
        fragment_index=int(row["fragment_index"]),
        worker_rank=int(row["worker_rank"]),
        e_mp2_lis=float(row["e_mp2_lis"]),
        e_ccsd=float(row["e_ccsd"]),
        e_ccsd_t=float(row["e_ccsd_t"]),
        lis_occupied=int(row["lis_occupied"]),
        lis_virtual=int(row["lis_virtual"]),
        wall_seconds=float(row["wall_seconds"]),
    )


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
    thresh_occ,
    thresh_vir,
    internal_rank_threshold,
    ccsd_t,
    dcsd,
    nproc,
    root,
):
    """Return the scientific identity of an MPI CC gradient restart."""

    payload = _restart_scientific_payload(
        mol,
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
        ccsd_t=ccsd_t,
        dcsd=dcsd,
    )
    payload["driver_schema"] = "mpi-iao-dlno-gradient-v1"
    # Rank-local cumulative CC states use the current round-robin ownership.
    # A partial restart therefore deliberately requires the same layout.
    payload["mpi"] = {"size": int(nproc), "root": int(root)}
    return payload


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


def _build_mp2_static_selections(
    mf,
    *,
    frag_lolist,
    frag_atmlist,
    frozen,
    thresholds,
    pair_energy_model,
    force_full_domains,
    static_selections,
):
    if static_selections is None:
        return stop_trace(
            lambda mf_: build_iao_dlno_ccsd_domain_selections(
                mf_,
                frag_lolist=frag_lolist,
                frag_atmlist=frag_atmlist,
                frozen=frozen,
                thresholds=thresholds,
                pair_energy_model=pair_energy_model,
                force_full_domains=force_full_domains,
            )
        )(mf)
    if not isinstance(static_selections, IAOFragmentLISStaticSelections):
        raise TypeError(
            "static_selections must be IAOFragmentLISStaticSelections"
        )
    return static_selections.mp2_static


def _lis_fragment_cost(mp2_static, fragment_index):
    """Return a deterministic MP2-density work estimate for one ED."""

    fragment = mp2_static.fragments[int(fragment_index)]
    nocc = max(int(fragment.strong_occ_metric_keep.size), 1)
    nvir = max(int(fragment.strong_virtual.metric_keep.size), 1)
    return nocc * nocc * nvir * nvir


def _lis_fragment_owners(mp2_static, nproc, *, root):
    """Assign fragment MP2 densities by greedy cost-weighted scheduling."""

    nproc = int(nproc)
    root = int(root)
    if nproc <= 0:
        raise ValueError("nproc must be positive")
    ranks = tuple((root + offset) % nproc for offset in range(nproc))
    rank_order = {rank: offset for offset, rank in enumerate(ranks)}
    loads = [0] * nproc
    owners = [-1] * len(mp2_static.fragments)
    jobs = sorted(
        range(len(mp2_static.fragments)),
        key=lambda index: (-_lis_fragment_cost(mp2_static, index), index),
    )
    for fragment_index in jobs:
        owner = min(
            ranks,
            key=lambda rank: (loads[rank], rank_order[rank]),
        )
        owners[fragment_index] = owner
        loads[owner] += _lis_fragment_cost(mp2_static, fragment_index)
    return tuple(owners)


def _assemble_lis_static_selections(
    mp2_static,
    gathered_records,
    *,
    thresh_occ,
    thresh_vir,
    internal_rank_threshold,
):
    """Validate and order the small per-fragment MPI selection records."""

    records = [
        record
        for rank_records in gathered_records
        for record in rank_records
    ]
    records.sort(key=lambda record: int(record[0]))
    indices = tuple(int(record[0]) for record in records)
    expected = tuple(range(len(mp2_static.fragments)))
    if indices != expected:
        raise RuntimeError(
            "MPI LIS rank selection did not cover every fragment exactly "
            f"once; expected={expected}, received={indices}"
        )
    fragments = []
    for fragment_index, selection in records:
        fragment_index = int(fragment_index)
        if not isinstance(selection, IAOLISFragmentStaticSelection):
            raise TypeError(
                "MPI LIS worker returned a non-LIS selection for fragment "
                f"{fragment_index}"
            )
        if int(selection.fragment_index) != fragment_index:
            raise RuntimeError(
                "MPI LIS worker returned a mismatched fragment index: "
                f"record={fragment_index}, selection="
                f"{selection.fragment_index}"
            )
        fragments.append(selection)
    return IAOFragmentLISStaticSelections(
        mp2_static=mp2_static,
        thresh_occ=float(thresh_occ),
        thresh_vir=float(thresh_vir),
        internal_rank_threshold=float(internal_rank_threshold),
        fragments=tuple(fragments),
    )


def _finish_value_and_grad(
    mol,
    *,
    mf,
    scf_pullback,
    scf_executor,
    parallel_scf_jk,
    total_mf_bar,
    energy,
    canonical,
    e_iao_mp2,
    components,
    e_corr,
    records,
    ccsd_t,
    dcsd,
    comm,
    root,
    reporter,
    started,
    return_details,
):
    """Apply the one final SCF response and assemble collective outputs."""

    rank = comm.Get_rank()
    nproc = comm.Get_size()
    response_setup_error = None
    mol_bar = None
    if rank == root:
        try:
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
                e_iao_mp2=float(e_iao_mp2),
                e_mp2_lis=float(components["e_mp2_lis"]),
                e_ccsd=float(components["e_ccsd"]),
                e_ccsd_t=float(components["e_ccsd_t"]),
                e_corr=float(e_corr),
                e_total=float(energy),
                fragments=tuple(records),
                nproc=int(nproc),
                total_seconds=float(total_seconds),
            )
        else:
            details = None
        details = comm.bcast(details, root=root)
        return energy, mol_bar, details
    return energy, mol_bar


def _complete_after_cc(
    mol,
    *,
    mf,
    scf_pullback,
    scf_executor,
    parallel_scf_jk,
    cc_mf_bar_root,
    hf_bar,
    canonical,
    static_selections,
    components,
    records,
    restart,
    ccsd_t,
    dcsd,
    comm,
    root,
    reporter,
    progress,
    started,
    return_details,
):
    """Add/resume global MP2, save the pre-SCF bar, and finish response."""

    rank = comm.Get_rank()
    _report_progress(reporter, "molecule-wide MPI IAO-DLNO-MP2: starting")
    e_iao_mp2, mp2_mf_bar_root = _mp2_correlation_value_and_grad(
        mf,
        static_selections.mp2_static,
        comm=comm,
        root=root,
        return_details=False,
        progress=progress,
        restart=restart,
    )
    e_iao_mp2 = float(e_iao_mp2)
    e_corr = (
        components["e_ccsd"]
        + components["e_ccsd_t"]
        - components["e_mp2_lis"]
        + e_iao_mp2
    )
    energy = float(canonical["e_tot"]) + e_corr

    response_setup_error = None
    total_mf_bar = None
    if rank == root:
        try:
            total_mf_bar = jax.tree_util.tree_map(
                _add_cotangent, cc_mf_bar_root, mp2_mf_bar_root
            )
            total_mf_bar = jax.tree_util.tree_map(
                _add_cotangent, total_mf_bar, hf_bar
            )
            jax.block_until_ready(total_mf_bar)
            if restart.enabled:
                restart.save_record(
                    "pre_scf",
                    scalars={
                        "e_iao_mp2": float(e_iao_mp2),
                        "e_mp2_lis": float(components["e_mp2_lis"]),
                        "e_ccsd": float(components["e_ccsd"]),
                        "e_ccsd_t": float(components["e_ccsd_t"]),
                        "e_corr": float(e_corr),
                        "e_total": float(energy),
                    },
                    trees={"mf_bar": total_mf_bar},
                    metadata={
                        "fragment_records": [
                            _fragment_record_metadata(record)
                            for record in records
                        ]
                    },
                )
                _report_progress(
                    reporter,
                    "restart: saved total pre-SCF energy and mean-field "
                    "cotangent",
                )
        except Exception:  # pragma: no cover - multi-rank failure path
            response_setup_error = _exception_text(
                "root pre-SCF cotangent assembly/checkpoint"
            )
    _raise_if_root_failed(comm, response_setup_error, root=root)
    return _finish_value_and_grad(
        mol,
        mf=mf,
        scf_pullback=scf_pullback,
        scf_executor=scf_executor,
        parallel_scf_jk=parallel_scf_jk,
        total_mf_bar=total_mf_bar,
        energy=energy,
        canonical=canonical,
        e_iao_mp2=e_iao_mp2,
        components=components,
        e_corr=e_corr,
        records=records,
        ccsd_t=ccsd_t,
        dcsd=dcsd,
        comm=comm,
        root=root,
        reporter=reporter,
        started=started,
        return_details=return_details,
    )


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
    checkpoint_dir=None,
    resume=False,
):
    """Implementation body for :meth:`DLNOCCSD.value_and_grad`."""
    rank = comm.Get_rank()
    nproc = comm.Get_size()
    root = int(root)
    if root < 0 or root >= nproc:
        raise ValueError(f"root={root} is invalid for {nproc} MPI ranks")
    _validate_solver_options(ccsd_t=ccsd_t, dcsd=dcsd)
    if resume and checkpoint_dir is None:
        raise ValueError("resume=True requires checkpoint_dir")
    progress_enabled = _progress_enabled(progress)
    schedules = comm.allgather((
        root,
        bool(return_details),
        progress_enabled,
        bool(parallel_scf_jk),
        bool(ccsd_t),
        bool(dcsd),
        None if checkpoint_dir is None else str(
            os.path.abspath(os.path.expanduser(os.fspath(checkpoint_dir)))
        ),
        bool(resume),
    ))
    if len(set(schedules)) != 1:
        raise ValueError(
            "root, return_details, progress, parallel_scf_jk, "
            "ccsd_t, dcsd, checkpoint_dir, and resume must be consistent "
            "on all ranks"
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

    scf_builder = build_mf
    if checkpoint_dir is not None:
        def scf_builder(mol_):
            return _fix_restart_mo_phases(build_mf(mol_))

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
                        mf, scf_pullback = jax.vjp(scf_builder, mol)
                        jax.block_until_ready(mf.e_tot)
                else:
                    mf, scf_pullback = jax.vjp(scf_builder, mol)
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
            restart_payload = _mpi_restart_scientific_payload(
                mol,
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
                ccsd_t=ccsd_t,
                dcsd=dcsd,
                nproc=nproc,
                root=root,
            )
            restart = RestartManager(
                checkpoint_dir,
                resume=resume,
                method=(
                    "mpi-iao-dlno-dcsd-gradient" if dcsd else
                    "mpi-iao-dlno-ccsd-t-gradient" if ccsd_t else
                    "mpi-iao-dlno-ccsd-gradient"
                ),
                scientific_payload=restart_payload,
                initialize=True,
            )
        except Exception:  # pragma: no cover - multi-rank failure path
            if scf_executor is not None:
                scf_executor.stop_workers()
            setup_error = _exception_text("root SCF/VJP setup")
            mf = scf_pullback = canonical = None
            restart = restart_payload = None
    else:
        scf_pullback = canonical = None
        restart = restart_payload = None
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
    restart_payload = comm.bcast(restart_payload, root=root)

    # Probe the final pre-SCF boundary before non-root CC/DF skeleton setup.
    # Without MPI-parallel SCF response those skeletons are not used at all by
    # a completed restart; with it, workers already own the paused DF service
    # object created above and only need root's canonical state installed.
    pre_scf = None
    pre_scf_payload = None
    pre_scf_error = None
    e_hf = hf_pullback = hf_bar = None
    if rank == root:
        try:
            e_hf, hf_pullback = jax.vjp(lambda mf_: mf_.e_tot, mf)
            hf_bar, = hf_pullback(
                jnp.ones((), dtype=jnp.asarray(e_hf).dtype)
            )
            if resume and restart.enabled:
                if static_selections is not None:
                    if not isinstance(
                        static_selections,
                        IAOFragmentLISStaticSelections,
                    ):
                        raise TypeError(
                            "static_selections must be "
                            "IAOFragmentLISStaticSelections"
                        )
                    restart.bind_static(static_selections)
                pre_scf = restart.load_record(
                    "pre_scf",
                    templates={"mf_bar": hf_bar},
                    missing_ok=True,
                )
                if pre_scf is not None:
                    pre_scf_payload = (
                        dict(pre_scf.scalars),
                        tuple(
                            _fragment_record_from_metadata(row)
                            for row in pre_scf.metadata.get(
                                "fragment_records", ()
                            )
                        ),
                    )
        except Exception:
            pre_scf_error = _exception_text("root pre-SCF restart load")
    _raise_if_root_failed(comm, pre_scf_error, root=root)
    pre_scf_payload = comm.bcast(pre_scf_payload, root=root)

    worker_error = None
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
    if pre_scf_payload is None or parallel_scf_jk:
        _verify_shared_reference(
            comm,
            canonical,
            mf,
            verify_df_source=parallel_scf_jk,
        )

    # The fully accumulated pre-SCF cotangent is the last restart boundary.
    # Check it before rebuilding the common IAO/PAO frame or touching LIS
    # domains: a restart at this point should pay only for recreating the SCF
    # VJP (and the rank-local DF skeleton needed by an MPI SCF response).
    if pre_scf_payload is not None:
        scalars, records = pre_scf_payload
        components = {
            "e_mp2_lis": float(scalars["e_mp2_lis"]),
            "e_ccsd": float(scalars["e_ccsd"]),
            "e_ccsd_t": float(scalars["e_ccsd_t"]),
        }
        e_iao_mp2 = float(scalars["e_iao_mp2"])
        e_corr = float(scalars["e_corr"])
        energy = float(scalars["e_total"])
        total_mf_bar = (
            pre_scf.trees["mf_bar"] if rank == root else None
        )
        _report_progress(
            reporter,
            "restart: loaded pre-SCF total cotangent; common/LIS, fragment, "
            "and molecule-wide MP2 work is skipped",
        )
        return _finish_value_and_grad(
            mol,
            mf=mf,
            scf_pullback=scf_pullback,
            scf_executor=scf_executor,
            parallel_scf_jk=parallel_scf_jk,
            total_mf_bar=total_mf_bar,
            energy=energy,
            canonical=canonical,
            e_iao_mp2=e_iao_mp2,
            components=components,
            e_corr=e_corr,
            records=records,
            ccsd_t=ccsd_t,
            dcsd=dcsd,
            comm=comm,
            root=root,
            reporter=reporter,
            started=started,
            return_details=return_details,
        )

    # Root owns the discrete fragment/ED topology, the ED orbital frames, and
    # the saved common-orbital VJP.  Only the expensive target-conditioned
    # MP2 densities and the resulting fixed LIS rank selections are
    # distributed.  The exact continuous common value is copied to every rank
    # so those independent selections use one shared orbital gauge.
    static_error = None
    if rank == root:
        try:
            prebuilt_static = static_selections
            saved_static = None
            if resume and restart.enabled:
                saved_static = restart.load_static(
                    expected_type=IAOFragmentLISStaticSelections
                )
            if prebuilt_static is None and saved_static is not None:
                prebuilt_static = saved_static
                _report_progress(
                    reporter,
                    "restart: loaded fixed fragment topology and LIS "
                    "selections",
                )
            elif prebuilt_static is not None:
                restart.bind_static(prebuilt_static)
            _report_progress(
                reporter,
                "fixed fragment/ED domain topology on root rank "
                f"{root}: starting",
            )
            static_started = time.perf_counter()
            mp2_static = _build_mp2_static_selections(
                mf,
                frag_lolist=frag_lolist,
                frag_atmlist=frag_atmlist,
                frozen=frozen,
                thresholds=thresholds,
                pair_energy_model=pair_energy_model,
                force_full_domains=force_full_domains,
                static_selections=prebuilt_static,
            )
            common_original, common_pullback = jax.vjp(
                lambda mf_: rebuild_iao_mp2_common(mf_, mp2_static), mf
            )
            common_host = jax.tree_util.tree_map(
                _to_host_leaf, common_original
            )
            if prebuilt_static is None:
                lis_owners = _lis_fragment_owners(
                    mp2_static, nproc, root=root
                )
            else:
                lis_owners = None
            payload = (
                mp2_static,
                common_host,
                lis_owners,
                prebuilt_static,
                (
                    float(thresh_occ),
                    float(thresh_vir),
                    float(internal_rank_threshold),
                ),
            )
            _report_progress(
                reporter,
                "fixed fragment/ED domain topology on root rank "
                f"{root}: done in "
                f"{time.perf_counter() - static_started:.1f} s; "
                f"fragments={len(mp2_static.fragments)}",
            )
        except Exception:  # pragma: no cover - multi-rank failure path
            static_error = _exception_text(
                "root fragment topology/common-orbital setup"
            )
            common_pullback = payload = None
    else:
        common_pullback = payload = None
    _raise_if_root_failed(comm, static_error, root=root)
    (
        mp2_static,
        common_host,
        lis_owners,
        prebuilt_static,
        lis_cutoffs,
    ) = comm.bcast(payload, root=root)
    thresh_occ, thresh_vir, internal_rank_threshold = lis_cutoffs
    common = jax.tree_util.tree_map(_to_device_leaf, common_host)
    del common_host
    if rank == root:
        del payload

    local_canonical = (
        numpy.asarray(mf.mo_coeff),
        numpy.asarray(mf.mo_energy),
        numpy.asarray(mf.mo_occ),
    )
    _verify_shared_gauge(comm, local_canonical, common, mf)
    _report_progress(reporter, "shared SCF/common orbital gauge verified")

    nfragment = len(mp2_static.fragments)
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

    if prebuilt_static is None:
        lis_queues = tuple(
            tuple(sorted(
                (
                    fragment_index
                    for fragment_index, owner in enumerate(lis_owners)
                    if owner == worker
                ),
                key=lambda fragment_index: (
                    -_lis_fragment_cost(mp2_static, fragment_index),
                    fragment_index,
                ),
            ))
            for worker in range(nproc)
        )
        assignment_counts = tuple(map(len, lis_queues))
        nbatch = max(assignment_counts, default=0)
        _report_progress(
            reporter,
            "MPI LIS MP2 density/rank selection: starting; "
            f"fragments={len(mp2_static.fragments)}; "
            f"cost-weighted counts/rank={assignment_counts}; "
            f"bounded batches={nbatch}",
        )
        lis_started = time.perf_counter()
        local_lis_seconds = 0.0
        local_lis_records = []
        for batch_index in range(nbatch):
            domain_error = None
            if rank == root:
                domain_batch = [None] * nproc
                try:
                    for worker, queue in enumerate(lis_queues):
                        if batch_index >= len(queue):
                            continue
                        fragment_index = queue[batch_index]
                        domain = stop_trace(
                            lambda common_: build_strong_ed_domain(
                                common_, mp2_static, fragment_index
                            )
                        )(common_original)
                        domain_batch[worker] = (
                            fragment_index,
                            jax.tree_util.tree_map(_to_host_leaf, domain),
                        )
                        del domain
                except Exception:  # pragma: no cover - MPI failure path
                    domain_error = _exception_text(
                        f"root ED construction for LIS batch {batch_index}"
                    )
                    domain_batch = None
            else:
                domain_batch = None
            _raise_if_root_failed(comm, domain_error, root=root)
            local_domain = comm.scatter(domain_batch, root=root)
            if rank == root:
                del domain_batch

            lis_error = None
            if local_domain is not None:
                local_lis_started = time.perf_counter()
                domain_host = domain = selection = None
                try:
                    fragment_index, domain_host = local_domain
                    domain = jax.tree_util.tree_map(
                        _to_device_leaf, domain_host
                    )
                    selection = stop_trace(
                        lambda mf_, common_, domain_: (
                            build_iao_lis_fragment_static_selection(
                                mf_,
                                mp2_static,
                                fragment_index,
                                common=common_,
                                domain=domain_,
                                thresh_occ=thresh_occ,
                                thresh_vir=thresh_vir,
                                internal_rank_threshold=(
                                    internal_rank_threshold
                                ),
                            )
                        )
                    )(mf, common, domain)
                    local_lis_records.append((fragment_index, selection))
                except Exception:  # pragma: no cover - MPI failure path
                    lis_error = _exception_text(
                        f"LIS MP2 rank selection on MPI rank {rank}"
                    )
                local_lis_seconds += (
                    time.perf_counter() - local_lis_started
                )
                del domain_host, domain, selection
            del local_domain
            _raise_if_any_rank_failed(comm, lis_error)
            completed = tuple(
                queue[batch_index] + 1
                if batch_index < len(queue) else None
                for queue in lis_queues
            )
            _report_progress(
                reporter,
                f"MPI LIS batch {batch_index + 1}/{nbatch}: "
                f"completed fragment numbers/rank={completed}",
            )

        gathered_lis_records = comm.allgather(tuple(local_lis_records))
        lis_stats = comm.allgather((
            len(local_lis_records),
            local_lis_seconds,
        ))
        assembly_error = None
        try:
            static_selections = _assemble_lis_static_selections(
                mp2_static,
                gathered_lis_records,
                thresh_occ=thresh_occ,
                thresh_vir=thresh_vir,
                internal_rank_threshold=internal_rank_threshold,
            )
        except Exception:  # pragma: no cover - multi-rank failure path
            assembly_error = _exception_text(
                f"LIS selection assembly on MPI rank {rank}"
            )
            static_selections = None
        _raise_if_any_rank_failed(comm, assembly_error)
        if rank == root:
            rank_timing = ", ".join(
                f"r{worker}:{count}/{seconds:.1f}s"
                for worker, (count, seconds) in enumerate(lis_stats)
            )
            _report_progress(
                reporter,
                "MPI LIS MP2 density/rank selection: done in "
                f"{time.perf_counter() - lis_started:.1f} s; "
                f"rank fragments/work={rank_timing}",
            )
        del gathered_lis_records, lis_queues, local_lis_records
    else:
        static_selections = prebuilt_static
        _report_progress(
            reporter,
            "using supplied fixed LIS rank selections; MPI LIS setup skipped",
        )
    if rank == root:
        del common_original

    checkpoint_error = None
    if rank == root:
        try:
            restart.bind_static(static_selections)
            if restart.enabled and not restart.static_path.is_file():
                restart.save_static(static_selections)
                _report_progress(
                    reporter,
                    "restart: saved fixed fragment topology and LIS "
                    "selections",
                )
        except Exception:
            checkpoint_error = _exception_text(
                "root fixed-selection checkpoint write"
            )
    _raise_if_root_failed(comm, checkpoint_error, root=root)

    restart_error = None
    if rank != root:
        try:
            restart = RestartManager(
                checkpoint_dir,
                resume=resume,
                method=(
                    "mpi-iao-dlno-dcsd-gradient" if dcsd else
                    "mpi-iao-dlno-ccsd-t-gradient" if ccsd_t else
                    "mpi-iao-dlno-ccsd-gradient"
                ),
                scientific_payload=restart_payload,
                initialize=False,
            )
        except Exception:
            restart_error = _exception_text(
                f"MPI restart setup on rank {rank}"
            )
    _raise_if_any_rank_failed(comm, restart_error)

    cc_closed = None
    cc_closed_payload = None
    cc_closed_error = None
    if rank == root and resume and restart.enabled:
        try:
            cc_closed = restart.load_record(
                "mpi_cc_closed",
                templates={"mf_bar": hf_bar},
                missing_ok=True,
            )
            if cc_closed is not None:
                cc_closed_payload = (
                    dict(cc_closed.scalars),
                    tuple(
                        _fragment_record_from_metadata(row)
                        for row in cc_closed.metadata.get(
                            "fragment_records", ()
                        )
                    ),
                )
        except Exception:
            cc_closed_error = _exception_text(
                "root common-closed CC restart load"
            )
    _raise_if_root_failed(comm, cc_closed_error, root=root)
    cc_closed_payload = comm.bcast(cc_closed_payload, root=root)
    if cc_closed_payload is not None:
        components, records = cc_closed_payload
        components = {
            name: float(value) for name, value in components.items()
        }
        cc_mf_bar_root = (
            cc_closed.trees["mf_bar"] if rank == root else None
        )
        _report_progress(
            reporter,
            "restart: loaded completed common-closed CC fragment "
            "cotangent; rank fragment checkpoints are skipped",
        )
        del common
        if rank == root:
            del common_pullback
        gc.collect()
        return _complete_after_cc(
            mol,
            mf=mf,
            scf_pullback=scf_pullback,
            scf_executor=scf_executor,
            parallel_scf_jk=parallel_scf_jk,
            cc_mf_bar_root=cc_mf_bar_root,
            hf_bar=hf_bar,
            canonical=canonical,
            static_selections=static_selections,
            components=components,
            records=records,
            restart=restart,
            ccsd_t=ccsd_t,
            dcsd=dcsd,
            comm=comm,
            root=root,
            reporter=reporter,
            progress=progress,
            started=started,
            return_details=return_details,
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

    if len(static_selections.fragments) != nfragment:
        raise RuntimeError(
            "LIS selections and MP2 domain topology have different "
            "fragment counts"
        )

    _report_progress(
        reporter,
        f"fragment pullbacks: {nfragment} fragments on {nproc} ranks; "
        "round-robin ownership",
    )
    local_mf_bar, local_common_bar = _zero_term_cotangents(mf, common)
    local_records = []
    completed_local = set()
    rank_progress_error = None
    try:
        if resume and restart.enabled:
            rank_progress = restart.load_record(
                "mpi_cc_progress",
                key=f"rank-{rank:06d}",
                templates={
                    "mf_bar": local_mf_bar,
                    "common_bar": local_common_bar,
                },
                missing_ok=True,
            )
            if rank_progress is not None:
                completed_ids = tuple(
                    int(value) for value in
                    rank_progress.metadata.get("completed_ids", ())
                )
                expected_ids = tuple(local_indices[:len(completed_ids)])
                if completed_ids != expected_ids:
                    raise RuntimeError(
                        f"MPI rank {rank} CC restart does not contain its "
                        "round-robin fragment prefix"
                    )
                local_records = [
                    _fragment_record_from_metadata(row)
                    for row in rank_progress.metadata.get(
                        "fragment_records", ()
                    )
                ]
                if tuple(
                    record.fragment_index for record in local_records
                ) != completed_ids:
                    raise RuntimeError(
                        f"MPI rank {rank} CC restart diagnostics do not "
                        "match its completed fragment IDs"
                    )
                local_mf_bar = rank_progress.trees["mf_bar"]
                local_common_bar = rank_progress.trees["common_bar"]
                completed_local.update(completed_ids)
                _report_progress(
                    reporter,
                    f"restart: rank {rank} loaded {len(completed_ids)}/"
                    f"{len(local_indices)} completed fragment pullbacks",
                )
    except Exception:
        rank_progress_error = _exception_text(
            f"MPI CC progress restart load on rank {rank}"
        )
    _raise_if_any_rank_failed(comm, rank_progress_error)
    nbatch = (nfragment + nproc - 1) // nproc
    for batch_index in range(nbatch):
        fragment_index = batch_index * nproc + rank
        fragment_error = None
        if (
            fragment_index < nfragment
            and fragment_index not in completed_local
        ):
            try:
                fragment_started = time.perf_counter()
                forward_record = None
                if resume and restart.enabled and ccsd_t:
                    saved_forward = restart.load_record(
                        "fragment_forward",
                        key=fragment_index,
                        templates={},
                        missing_ok=True,
                    )
                    if saved_forward is not None:
                        forward_record = dict(saved_forward.scalars)
                        _report_progress(
                            reporter,
                            f"restart: fragment {fragment_index + 1} has a "
                            "saved (T) forward energy; using backward replay",
                        )

                def save_forward(values, _index=fragment_index):
                    restart.save_record(
                        "fragment_forward",
                        key=_index,
                        scalars=values,
                    )

                fragment = _fragment_value_and_grad(
                    mf,
                    common,
                    static_selections,
                    fragment_index,
                    verbose_imp=verbose_imp,
                    ccsd_t=ccsd_t,
                    dcsd=dcsd,
                    forward_record=forward_record,
                    save_forward=(
                        save_forward
                        if restart.enabled and ccsd_t
                        and forward_record is None else None
                    ),
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
                completed_local.add(int(fragment_index))
                if restart.enabled:
                    restart.save_record(
                        "mpi_cc_progress",
                        key=f"rank-{rank:06d}",
                        trees={
                            "mf_bar": local_mf_bar,
                            "common_bar": local_common_bar,
                        },
                        metadata={
                            "completed_ids": sorted(completed_local),
                            "fragment_records": [
                                _fragment_record_metadata(record)
                                for record in local_records
                            ],
                        },
                    )
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
    checkpoint_error = None
    if rank == root and restart.enabled:
        try:
            restart.save_record(
                "mpi_cc_closed",
                scalars=components,
                trees={"mf_bar": cc_mf_bar_root},
                metadata={
                    "fragment_records": [
                        _fragment_record_metadata(record)
                        for record in records
                    ]
                },
            )
            _report_progress(
                reporter,
                "restart: saved completed common-closed CC fragment "
                "cotangent",
            )
        except Exception:
            checkpoint_error = _exception_text(
                "root CC checkpoint write"
            )
    _raise_if_root_failed(comm, checkpoint_error, root=root)
    del local_mf_bar, local_common_bar, common
    gc.collect()

    return _complete_after_cc(
        mol,
        mf=mf,
        scf_pullback=scf_pullback,
        scf_executor=scf_executor,
        parallel_scf_jk=parallel_scf_jk,
        cc_mf_bar_root=cc_mf_bar_root,
        hf_bar=hf_bar,
        canonical=canonical,
        static_selections=static_selections,
        components=components,
        records=records,
        restart=restart,
        ccsd_t=ccsd_t,
        dcsd=dcsd,
        comm=comm,
        root=root,
        reporter=reporter,
        progress=progress,
        started=started,
        return_details=return_details,
    )


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
        checkpoint_dir=None,
        resume=False,
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

        ``checkpoint_dir`` must be shared read/write storage visible at the
        same path on every rank.  ``resume=True`` must be passed collectively,
        and partial rank records require the same communicator size and root.
        Do not run two calculations concurrently in one checkpoint directory.
        A checkpoint-seeded ``build_mf`` must retain the same converged
        implicit DF-RHF response map as the original run.
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
                checkpoint_dir=checkpoint_dir,
                resume=resume,
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
