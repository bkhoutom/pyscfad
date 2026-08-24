"""MPI IAO-DLNO-MP2 gradient convergence for the OMCB example.

This is the local-MP2 prelude to the IAO-DLNO-CCSD(T) threshold study.  Rank
zero first evaluates a canonical DF-MP2 total energy and nuclear gradient.
Each requested pair threshold is then evaluated by the gauge-safe MPI driver:
rank zero owns the SCF and common IAO orbital pullbacks, while strong ED and
weak multipole terms are distributed without independently rebuilding orbital
gauges on workers.

The defaults reproduce the molecular, orbital-basis, auxiliary-basis, and
all-electron OMCB reference setup: OMCB/cc-pVTZ with PySCF's automatic
JK-fit basis (cc-pVTZ-JKFIT for both C and H) and no frozen orbitals.  The
pair thresholds deliberately straddle actual pair-selection boundaries for
this compact cage; all thresholds at or below ``1e-3`` have the same
all-strong topology.

Run from the repository root, for example::

    mpirun -np 2 .venv/bin/python \
        examples/lno/20-mpi_omcb_iao_mp2_gradient_sweep.py

The CSV, compressed NPZ, and JSON files are checkpointed after the canonical
reference and after every completed local threshold.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time

# Rank-by-rank BLAS/OpenMP oversubscription is particularly harmful here.
# Explicit settings supplied by a scheduler or the user still take priority.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("PYSCFAD_LNO_LOCAL_DIRECT_INT3C_BLOCK_MB", "128")
os.environ.setdefault("PYSCFAD_DF_CDERI_BAR_AUX_BLOCK_MB", "128")

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI
import pyscf
from pyscf import df as pyscf_df
import psutil

from pyscfad import config, gto, scf
from pyscfad.dlno.iao_mp2 import (
    IAOFragmentMP2Thresholds,
    build_iao_fragment_topology,
)
from pyscfad.dlno.iao_mp2_mpi import IAOFragmentMP2
from pyscfad.mp import dfmp2


CSV_FIELDS = (
    "pair_threshold",
    "e_corr",
    "e_total",
    "energy_error_eh",
    "energy_error_millihartree",
    "gradient_rms_error",
    "gradient_max_abs_error",
    "gradient_l2_error",
    "gradient_relative_l2_error",
    "strong_pair_count",
    "weak_pair_count",
    "total_pair_count",
    "nfragment",
    "mean_ed_atoms",
    "max_ed_atoms",
    "mean_ed_aos",
    "max_ed_aos",
    "topology_statistics_seconds",
    "mpi_energy_gradient_seconds",
)


def _parse_frozen(value):
    normalized = str(value).strip().lower()
    if normalized in {"none", "null", "all-electron", "all_electron"}:
        return None
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("frozen must be non-negative")
    return parsed


def _parse_auxbasis(value):
    normalized = str(value).strip()
    if normalized.lower() in {"auto", "none", "null"}:
        return None
    return normalized


def _basis_label(value):
    normalized = str(value).strip().lower().replace("_", "-")
    return {
        "ccpvtz": "cc-pvtz",
        "ccpvdz": "cc-pvdz",
        "ccpvqz": "cc-pvqz",
    }.get(normalized, normalized)


def _parser():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry", default=str(here / "OMCB.xyz"),
        help="OMCB XYZ geometry",
    )
    parser.add_argument("--basis", default="ccpvtz")
    parser.add_argument(
        "--auxbasis", type=_parse_auxbasis, default=None,
        help="auxiliary basis; auto (default) uses the OMCB reference setup",
    )
    parser.add_argument("--frozen", type=_parse_frozen, default=None)
    parser.add_argument(
        "--pair-thresholds", type=float, nargs="+",
        default=[3e-2, 1e-2, 1e-3],
    )
    parser.add_argument("--max-memory", type=float, default=6000.0)
    parser.add_argument("--mp2-block-memory", type=float, default=128.0)
    parser.add_argument("--scf-conv-tol", type=float, default=1e-10)
    parser.add_argument("--scf-max-cycle", type=int, default=100)
    parser.add_argument("--min-free-disk-gib", type=float, default=45.0)
    parser.add_argument("--max-combined-rss-gib", type=float, default=36.0)
    parser.add_argument(
        "--output-prefix",
        default="notes/results/omcb_ccpvtz_iao_mp2_mpi_pair_sweep",
    )
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()
    if any(value < 0.0 for value in args.pair_thresholds):
        parser.error("pair thresholds must be non-negative")
    if args.scf_max_cycle <= 0:
        parser.error("--scf-max-cycle must be positive")
    if args.min_free_disk_gib <= 0.0 or args.max_combined_rss_gib <= 0.0:
        parser.error("resource guards must be positive")
    return args


def _density_fit(rhf, auxbasis):
    """Preserve PySCF's automatic JK-fit resolution when auxbasis is None."""
    return rhf.density_fit() if auxbasis is None else rhf.density_fit(
        auxbasis=auxbasis
    )


def _free_disk_gib(path):
    return shutil.disk_usage(Path(path).resolve()).free / 1024.0**3


def _require_free_disk(path, minimum_gib):
    available = _free_disk_gib(path)
    if available < float(minimum_gib):
        raise RuntimeError(
            f"only {available:.2f} GiB is free at {path}; "
            f"the configured minimum is {minimum_gib:.2f} GiB"
        )
    return available


def _combined_rss_gib(comm):
    local_bytes = psutil.Process().memory_info().rss
    total_bytes = comm.allreduce(local_bytes, op=MPI.SUM)
    return total_bytes / 1024.0**3


def _require_combined_rss(comm, maximum_gib):
    combined = _combined_rss_gib(comm)
    if combined > float(maximum_gib):
        raise MemoryError(
            f"combined MPI RSS is {combined:.2f} GiB; the configured limit "
            f"is {maximum_gib:.2f} GiB"
        )
    return combined


def _add_cotangents(left, right):
    def add_leaf(x, y):
        if x is None:
            return y
        if y is None:
            return x
        if hasattr(x, "dtype") and x.dtype == jax.dtypes.float0:
            return y
        if hasattr(y, "dtype") and y.dtype == jax.dtypes.float0:
            return x
        return x + y

    return jax.tree_util.tree_map(add_leaf, left, right)


def _canonical_value_and_grad(mol, build_mf, frozen):
    """Canonical total DF-MP2 energy/gradient with a single SCF response."""
    start = time.perf_counter()
    mf, scf_pullback = jax.vjp(build_mf, mol)
    jax.block_until_ready(mf.e_tot)
    scf_forward_seconds = time.perf_counter() - start

    def correlation(mf_):
        energy, _ = dfmp2.MP2(mf_, frozen=frozen).kernel(with_t2=False)
        return energy

    start = time.perf_counter()
    e_corr, corr_pullback = jax.vjp(correlation, mf)
    jax.block_until_ready(e_corr)
    corr_forward_seconds = time.perf_counter() - start
    start = time.perf_counter()
    corr_bar, = corr_pullback(jnp.ones((), dtype=jnp.asarray(e_corr).dtype))
    jax.block_until_ready(corr_bar)
    corr_reverse_seconds = time.perf_counter() - start

    e_hf, hf_pullback = jax.vjp(lambda mf_: mf_.e_tot, mf)
    hf_bar, = hf_pullback(jnp.ones((), dtype=jnp.asarray(e_hf).dtype))
    start = time.perf_counter()
    mol_bar, = scf_pullback(_add_cotangents(hf_bar, corr_bar))
    jax.block_until_ready(mol_bar)
    scf_pullback_seconds = time.perf_counter() - start
    return (
        mf,
        float(e_hf),
        float(e_corr),
        np.asarray(mol_bar.coords),
        {
            "scf_forward_seconds": scf_forward_seconds,
            "correlation_forward_seconds": corr_forward_seconds,
            "correlation_reverse_seconds": corr_reverse_seconds,
            "scf_pullback_seconds": scf_pullback_seconds,
        },
    )


def _topology_statistics(mf, *, frozen, thresholds):
    start = time.perf_counter()
    topology = build_iao_fragment_topology(
        mf,
        frozen=frozen,
        thresholds=thresholds,
        pair_energy_model="multipole",
    )
    elapsed = time.perf_counter() - start
    nfragment = len(topology.frag_lolist)
    upper = np.triu(np.asarray(topology.strong_mask, dtype=bool), k=1)
    total_pairs = nfragment * (nfragment - 1) // 2
    strong_pairs = int(np.count_nonzero(upper))
    atom_sizes = np.asarray([
        len(np.asarray(atoms)) for atoms in topology.extended_domain
    ])
    ao_sizes = np.asarray([
        sum(
            mf.mol.aoslice_by_atom()[atom, 3]
            - mf.mol.aoslice_by_atom()[atom, 2]
            for atom in np.asarray(atoms, dtype=int)
        )
        for atoms in topology.extended_domain
    ])
    return {
        "strong_pair_count": strong_pairs,
        "weak_pair_count": total_pairs - strong_pairs,
        "total_pair_count": total_pairs,
        "nfragment": nfragment,
        "mean_ed_atoms": float(atom_sizes.mean()),
        "max_ed_atoms": int(atom_sizes.max()),
        "mean_ed_aos": float(ao_sizes.mean()),
        "max_ed_aos": int(ao_sizes.max()),
        "topology_statistics_seconds": elapsed,
    }


def _error_statistics(gradient, reference):
    error = np.asarray(gradient) - np.asarray(reference)
    l2 = float(np.linalg.norm(error))
    reference_l2 = float(np.linalg.norm(reference))
    return {
        "gradient_rms_error": float(np.sqrt(np.mean(error * error))),
        "gradient_max_abs_error": float(np.max(np.abs(error))),
        "gradient_l2_error": l2,
        "gradient_relative_l2_error": (
            l2 / reference_l2 if reference_l2 else float("nan")
        ),
    }


def _write_outputs(prefix, metadata, rows, gradients, reference_gradient):
    prefix = Path(prefix).expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = prefix.with_suffix(".csv")
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in CSV_FIELDS})
    csv_tmp.replace(csv_path)

    json_path = prefix.with_suffix(".json")
    json_tmp = json_path.with_suffix(".json.tmp")
    payload = dict(metadata)
    payload["rows"] = rows
    with json_tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    json_tmp.replace(json_path)

    npz_path = prefix.with_suffix(".npz")
    npz_tmp = npz_path.with_suffix(".npz.tmp")
    gradient_shape = (0,) + tuple(np.asarray(reference_gradient).shape)
    local_gradients = (
        np.stack(gradients) if gradients else np.empty(gradient_shape)
    )
    with npz_tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            pair_thresholds=np.asarray([
                row["pair_threshold"] for row in rows
            ]),
            total_energies=np.asarray([row["e_total"] for row in rows]),
            correlation_energies=np.asarray([
                row["e_corr"] for row in rows
            ]),
            gradients=local_gradients,
            reference_gradient=np.asarray(reference_gradient),
            reference_total_energy=np.asarray(
                metadata["reference"]["e_total"]
            ),
            reference_correlation_energy=np.asarray(
                metadata["reference"]["e_corr"]
            ),
        )
    npz_tmp.replace(npz_path)
    return csv_path, npz_path, json_path


def main():
    args = _parser()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()

    config.update("pyscfad_moleintor_opt", True)
    config.update("pyscfad_scf_implicit_diff", True)
    config.update("pyscfad_scf_first_order_custom", False)

    geometry = str(Path(args.geometry).expanduser().resolve())
    geometry_sha256 = hashlib.sha256(Path(geometry).read_bytes()).hexdigest()
    mol = gto.Mole(
        atom=geometry,
        basis=args.basis,
        max_memory=args.max_memory,
        verbose=args.verbose if rank == 0 else 0,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    output_parent = Path(args.output_prefix).expanduser().resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    _require_free_disk(output_parent, args.min_free_disk_gib)
    resolved_auxbasis = pyscf_df.addons.make_auxbasis(
        mol, mp2fit=False
    ) if args.auxbasis is None else args.auxbasis

    # Each worker owns only a placeholder path.  Rank zero alone creates and
    # reads the global out-of-core CDERI; local ED transforms are integral
    # direct on every rank.
    with tempfile.TemporaryDirectory(
        prefix=f"pyscfad-omcb-mp2-rank{rank}-"
    ) as scratch:
        cderi_path = str(Path(scratch) / "cderi.h5")
        if rank == 0:
            start = time.perf_counter()
            builder = _density_fit(scf.RHF(mol), args.auxbasis).with_df
            builder.max_memory = mol.max_memory
            builder._cderi_to_save = cderi_path
            builder.build()
            cderi_seconds = time.perf_counter() - start
            del builder
        else:
            cderi_seconds = None
        comm.Barrier()

        def build_mf(
            mol_,
            *,
            mo_coeff_init=None,
            mo_energy_init=None,
            mo_occ_init=None,
            e_tot_init=None,
        ):
            mf = _density_fit(scf.RHF(mol_), args.auxbasis)
            mf.with_df.max_memory = mol_.max_memory
            mf.with_df.attach_outcore_cderi(cderi_path)
            mf.conv_tol = args.scf_conv_tol
            mf.conv_tol_grad = max(args.scf_conv_tol**0.5, 1e-10)
            mf.max_cycle = args.scf_max_cycle
            if mo_coeff_init is None:
                mf.kernel()
                if not mf.converged:
                    raise RuntimeError("DF-RHF did not converge")
            else:
                mf.mo_coeff = mo_coeff_init
                mf.mo_energy = mo_energy_init
                mf.mo_occ = mo_occ_init
                mf.e_tot = e_tot_init
                mf.converged = True
            return mf

        # The canonical reference is deliberately completed and checkpointed
        # before any local row is attempted.
        if rank == 0:
            print(
                f"OMCB {_basis_label(args.basis)}/"
                f"{args.auxbasis if args.auxbasis is not None else 'auto'} "
                f"(resolved {resolved_auxbasis}), frozen={args.frozen}: "
                f"{mol.natm} atoms, {mol.nao} AOs; MPI ranks={nproc}",
                flush=True,
            )
            reference_mf, e_hf, e_ref_corr, reference_gradient, ref_timing = (
                _canonical_value_and_grad(mol, build_mf, args.frozen)
            )
            nocc = int(np.count_nonzero(np.asarray(reference_mf.mo_occ)))
            naux = int(reference_mf.with_df.get_naoaux())
            metadata = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "python_version": platform.python_version(),
                "jax_version": jax.__version__,
                "pyscf_version": pyscf.__version__,
                "system": {
                    "name": "OMCB",
                    "formula": "C12H24",
                    "geometry": geometry,
                    "geometry_sha256": geometry_sha256,
                    "natom": mol.natm,
                    "nao": mol.nao,
                    "nocc": nocc,
                    "nactive_occ": nocc - (args.frozen or 0),
                    "nactive_vir": mol.nao - nocc,
                    "naux": naux,
                    "basis": _basis_label(args.basis),
                    "basis_requested": args.basis,
                    "auxbasis": (
                        "auto" if args.auxbasis is None else args.auxbasis
                    ),
                    "auxbasis_requested": args.auxbasis,
                    "auxbasis_resolved": resolved_auxbasis,
                    "frozen": args.frozen,
                },
                "settings": {
                    "pair_thresholds": args.pair_thresholds,
                    "pair_energy_model": "multipole",
                    "mp2_block_memory_mb": args.mp2_block_memory,
                    "scf_conv_tol": args.scf_conv_tol,
                    "scf_max_cycle": args.scf_max_cycle,
                    "topology_derivative": "fixed",
                    "mpi_ranks": nproc,
                    "min_free_disk_gib": args.min_free_disk_gib,
                    "max_combined_rss_gib": args.max_combined_rss_gib,
                    "local_direct_int3c_block_mb": os.environ.get(
                        "PYSCFAD_LNO_LOCAL_DIRECT_INT3C_BLOCK_MB"
                    ),
                    "cderi_bar_aux_block_mb": os.environ.get(
                        "PYSCFAD_DF_CDERI_BAR_AUX_BLOCK_MB"
                    ),
                    "omp_num_threads_per_rank": os.environ.get(
                        "OMP_NUM_THREADS"
                    ),
                    "openblas_num_threads_per_rank": os.environ.get(
                        "OPENBLAS_NUM_THREADS"
                    ),
                    "veclib_maximum_threads_per_rank": os.environ.get(
                        "VECLIB_MAXIMUM_THREADS"
                    ),
                },
                "reference": {
                    "method": "canonical DF-MP2",
                    "e_hf": e_hf,
                    "e_corr": e_ref_corr,
                    "e_total": e_hf + e_ref_corr,
                    "gradient_l2_norm": float(
                        np.linalg.norm(reference_gradient)
                    ),
                    "gradient_max_abs": float(
                        np.max(np.abs(reference_gradient))
                    ),
                    "timing": ref_timing,
                },
                "scf": {
                    "response_backend": "standard_fixed_point_implicit",
                    "first_order_custom": False,
                    "conv_tol": args.scf_conv_tol,
                    "max_cycle": args.scf_max_cycle,
                },
                "cderi_build_seconds": cderi_seconds,
                "energy_units": "Eh",
                "gradient_units": "Eh/bohr",
            }
            rows = []
            gradients = []
            paths = _write_outputs(
                args.output_prefix,
                metadata,
                rows,
                gradients,
                reference_gradient,
            )
            print(
                f"Canonical DF-MP2 total={e_hf + e_ref_corr:.12f}, "
                f"Ecorr={e_ref_corr:.12f}; reference checkpoint={paths[1]}",
                flush=True,
            )
            statistics_by_threshold = {}
            for pair_threshold in args.pair_thresholds:
                statistics_by_threshold[float(pair_threshold)] = (
                    _topology_statistics(
                        reference_mf,
                        frozen=args.frozen,
                        thresholds=IAOFragmentMP2Thresholds(
                            pair_energy=pair_threshold,
                            mp2_block_memory_mb=args.mp2_block_memory,
                        ),
                    )
                )
            del reference_mf
            reference_mf = None
            gc.collect()
        else:
            reference_mf = metadata = rows = gradients = None
            reference_gradient = None
            statistics_by_threshold = None
        comm.Barrier()

        for pair_threshold in args.pair_thresholds:
            thresholds = IAOFragmentMP2Thresholds(
                pair_energy=pair_threshold,
                mp2_block_memory_mb=args.mp2_block_memory,
            )
            if rank == 0:
                statistics = statistics_by_threshold[float(pair_threshold)]
                free_disk = _require_free_disk(
                    output_parent, args.min_free_disk_gib
                )
                print(
                    f"Starting tau_pair={pair_threshold:.1e}: "
                    f"strong={statistics['strong_pair_count']}/"
                    f"{statistics['total_pair_count']}; "
                    f"free_disk={free_disk:.1f} GiB",
                    flush=True,
                )
            else:
                statistics = None
            comm.Barrier()
            rss_before = _require_combined_rss(
                comm, args.max_combined_rss_gib
            )
            start = time.perf_counter()
            energy, mol_bar = IAOFragmentMP2.value_and_grad(
                mol,
                build_mf=build_mf,
                frozen=args.frozen,
                thresholds=thresholds,
                pair_energy_model="multipole",
                include_hf=True,
                comm=comm,
            )
            elapsed = time.perf_counter() - start
            rss_after = _require_combined_rss(
                comm, args.max_combined_rss_gib
            )

            if rank == 0:
                gradient = np.asarray(mol_bar.coords)
                energy_error = float(energy) - metadata["reference"]["e_total"]
                row = {
                    "pair_threshold": float(pair_threshold),
                    "e_corr": float(energy) - metadata["reference"]["e_hf"],
                    "e_total": float(energy),
                    "energy_error_eh": energy_error,
                    "energy_error_millihartree": 1000.0 * energy_error,
                    **_error_statistics(gradient, reference_gradient),
                    **statistics,
                    "mpi_energy_gradient_seconds": elapsed,
                    "combined_rss_gib_before": rss_before,
                    "combined_rss_gib_after": rss_after,
                    "free_disk_gib_after": _free_disk_gib(output_parent),
                }
                rows.append(row)
                gradients.append(gradient)
                paths = _write_outputs(
                    args.output_prefix,
                    metadata,
                    rows,
                    gradients,
                    reference_gradient,
                )
                print(
                    f"Finished tau_pair={pair_threshold:.1e}: "
                    f"dE={row['energy_error_millihartree']:+.6f} mEh, "
                    f"RMS(dG)={row['gradient_rms_error']:.6e}, "
                    f"max|dG|={row['gradient_max_abs_error']:.6e}, "
                    f"wall={elapsed:.1f} s, RSS(after)={rss_after:.2f} GiB; "
                    f"checkpoint={paths[0]}",
                    flush=True,
                )
                del mol_bar, gradient
                gc.collect()
            comm.Barrier()

        if rank == 0:
            print(f"Completed {len(rows)} thresholds: {paths[0]}", flush=True)
        comm.Barrier()


if __name__ == "__main__":
    main()
