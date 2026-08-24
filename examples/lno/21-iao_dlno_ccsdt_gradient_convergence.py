"""Two-stage IAO-DLNO-MP2 and IAO-DLNO-CCSD(T) convergence benchmark.

The defaults reproduce the original example-13 system exactly: the OMCB
geometry, cc-pVTZ orbital basis, PySCF's automatic cc-pVTZ-JKFIT auxiliary
basis, and no frozen orbitals.  The gauge-safe example-20 MPI MP2 checkpoint
is imported by default and its canonical-accuracy ``1e-3`` pair cutoff is
held fixed for the CCSD(T) LNO sweep.

The all-electron OMCB/cc-pVTZ canonical AD-CCSD(T) gradient has a live-tensor
floor well above workstation memory.  Therefore the default run performs a
dimension-based preflight before building CDERI or starting SCF and refuses
the missing canonical reference.  Continue only with a compatible validated
``--reference-npz``/``--resume`` checkpoint, or explicitly accept the OOM
risk with ``--allow-canonical-memory-oversubscription``.  No smaller-basis
result is substituted for the requested calculation.

The calculation has two deliberately separate stages:

1. Import (or, when explicitly requested, evaluate) complete strong-plus-weak
   IAO local MP2 rows against canonical DF-MP2.  The exact-system default uses
   the gauge-safe MPI checkpoint and fixes the pair cutoff at ``1e-3``.  Pass
   ``--fixed-pair-cutoff auto`` to select the loosest sampled cutoff whose
   absolute energy error is below ``--pair-energy-target-mha``.
2. Hold that pair cutoff fixed and vary the occupied LNO threshold.  The
   virtual threshold is ``--lno-vir-ratio`` times the occupied threshold.
   IAO-DLNO-CCSD(T) total energies and full nuclear gradients are compared
   with canonical DF-CCSD(T).

Every topology and LIS rank decision is constructed at the reference
geometry and passed back as a fixed selection to the differentiated energy.
CSV, JSON, and NPZ files are replaced atomically after the canonical
references and after every completed row, so a failed tighter point does not
erase earlier work.  ``--resume`` validates and reuses a compatible partial
checkpoint.

Example smoke run::

    python examples/lno/21-iao_dlno_ccsdt_gradient_convergence.py \
      --xyz examples/lno/water_dimer.xyz --basis sto-3g \
      --auxbasis weigend --frozen 0 --mpi-pair-results none \
      --pair-cutoffs 1e-4 --fixed-pair-cutoff 1e-4 \
      --lno-thresholds 1e-3 \
      --output-prefix /tmp/iao_dlno_ccsdt_smoke

To reuse the gauge-safe MPI pair sweep and avoid repeating it serially::

    python examples/lno/21-iao_dlno_ccsdt_gradient_convergence.py \
      --mpi-pair-results \
        notes/results/omcb_ccpvtz_iao_mp2_mpi_pair_sweep \
      --fixed-pair-cutoff 1e-3 \
      --reference-npz /path/to/validated_omcb_ccpvtz_reference.npz \
      --output-prefix notes/results/iao_dlno_ccsdt_omcb_ccpvtz

All energies are in Eh and gradients in Eh/bohr.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time

import jax
import numpy as np
import pyscf

from pyscfad import config, df, gto, scf
from pyscfad.cc import dfccsd
from pyscfad.dlno.ccsd import DLNOCCSD
from pyscfad.dlno.iao_lis import build_iao_lis_static_selections
from pyscfad.dlno.iao_mp2 import (
    IAOFragmentMP2,
    IAOFragmentMP2Thresholds,
    build_iao_fragment_topology,
)
from pyscfad.dlno.iao_mp2_grad import (
    build_iao_mp2_static_selections,
    rebuild_iao_mp2_common,
)
from pyscfad.mp import dfmp2
from pyscfad.ops import stop_trace


HERE = Path(__file__).resolve().parent
DEFAULT_XYZ = HERE / "OMCB.xyz"
DEFAULT_MPI_PAIR_RESULTS = (
    "notes/results/omcb_ccpvtz_iao_mp2_mpi_pair_sweep"
)
DEFAULT_OUTPUT_PREFIX = "notes/results/iao_dlno_ccsdt_omcb_ccpvtz"


CSV_FIELDS = (
    "stage", "label", "reference_method", "pair_cutoff",
    "lno_occ_threshold", "lno_vir_threshold", "energy", "energy_error",
    "energy_abs_error_mha", "gradient_max_abs_error",
    "gradient_rms_error", "gradient_l2_error",
    "gradient_relative_l2_error", "net_force_max_abs",
    "nfragment", "strong_pair_count", "weak_pair_count",
    "total_pair_count", "mean_ed_atoms", "max_ed_atoms",
    "mean_ed_aos", "max_ed_aos", "mean_ed_occ", "max_ed_occ",
    "mean_ed_vir", "max_ed_vir", "mean_lis_occ", "max_lis_occ",
    "mean_lis_vir", "max_lis_vir", "topology_seconds",
    "static_selection_seconds", "gradient_seconds", "row_total_seconds",
    "peak_rss_gib",
)


def _parse_frozen(value):
    normalized = str(value).strip().lower()
    if normalized in {"none", "null", "all-electron", "all_electron"}:
        return None
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("frozen must be non-negative")
    return parsed


def _frozen_count(value):
    return 0 if value is None else int(value)


def _parse_auxbasis(value):
    normalized = str(value).strip()
    if normalized.lower() in {"auto", "none", "null"}:
        return None
    return normalized


def _parse_optional_path(value):
    normalized = str(value).strip()
    if normalized.lower() in {"none", "null", "off"}:
        return None
    return normalized


def _parse_optional_float(value):
    normalized = str(value).strip().lower()
    if normalized in {"auto", "none", "null"}:
        return None
    return float(value)


def _basis_label(value):
    normalized = str(value).strip().lower().replace("_", "-")
    return {
        "ccpvtz": "cc-pvtz",
        "ccpvdz": "cc-pvdz",
        "ccpvqz": "cc-pvqz",
    }.get(normalized, normalized)


def _auxbasis_label(value):
    return "auto" if value is None else str(value).strip().lower()


def _file_sha256(path):
    """Return a content identity that is stable when a file is moved."""
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _geometry_sha256(path):
    return _file_sha256(path)


def _mpi_pair_result_identity(prefix):
    """Identify an imported checkpoint by contents, not checkout path."""
    npz_path, json_path = _result_paths(prefix)
    return {
        "npz_sha256": _file_sha256(npz_path),
        "json_sha256": _file_sha256(json_path),
    }


def _block(value):
    jax.block_until_ready(value)
    return value


def _peak_rss_gib():
    """Return process maximum RSS in GiB on macOS and Linux."""
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / 1024.0**3
    return value * 1024.0 / 1024.0**3


def _error_statistics(gradient, reference_gradient):
    gradient = np.asarray(gradient, dtype=float)
    reference_gradient = np.asarray(reference_gradient, dtype=float)
    error = gradient - reference_gradient
    error_l2 = float(np.linalg.norm(error))
    reference_l2 = float(np.linalg.norm(reference_gradient))
    return {
        "gradient_max_abs_error": float(np.max(np.abs(error))),
        "gradient_rms_error": float(np.sqrt(np.mean(error * error))),
        "gradient_l2_error": error_l2,
        "gradient_relative_l2_error": (
            error_l2 / reference_l2 if reference_l2 else float("nan")
        ),
        "net_force_max_abs": float(
            np.max(np.abs(np.sum(gradient, axis=0)))
        ),
    }


def _domain_statistics(mp2_static):
    fragments = mp2_static.fragments
    atom_sizes = np.asarray([
        len(fragment.extended_atoms) for fragment in fragments
    ])
    ao_sizes = np.asarray([
        len(fragment.extended_ao_indices) for fragment in fragments
    ])
    occ_sizes = np.asarray([
        len(fragment.strong_occ_metric_keep) for fragment in fragments
    ])
    vir_sizes = np.asarray([
        len(fragment.strong_virtual.metric_keep) for fragment in fragments
    ])
    upper = np.triu(np.asarray(mp2_static.strong_mask, dtype=bool), k=1)
    total_pairs = len(fragments) * (len(fragments) - 1) // 2
    strong_pairs = int(np.count_nonzero(upper))
    return {
        "nfragment": len(fragments),
        "strong_pair_count": strong_pairs,
        "weak_pair_count": total_pairs - strong_pairs,
        "total_pair_count": total_pairs,
        "mean_ed_atoms": float(atom_sizes.mean()),
        "max_ed_atoms": int(atom_sizes.max()),
        "mean_ed_aos": float(ao_sizes.mean()),
        "max_ed_aos": int(ao_sizes.max()),
        "mean_ed_occ": float(occ_sizes.mean()),
        "max_ed_occ": int(occ_sizes.max()),
        "mean_ed_vir": float(vir_sizes.mean()),
        "max_ed_vir": int(vir_sizes.max()),
    }


def _lis_statistics(static):
    nocc = len(static.mp2_static.active_occ_indices)
    nvir = len(static.mp2_static.active_vir_indices)
    occupied = []
    virtual = []
    for selection in static.fragments:
        occupied.append(
            nocc if selection.full_occupied_space else
            len(selection.internal_occ_keep)
            + len(selection.occupied_lno_keep)
        )
        virtual.append(
            nvir if selection.full_virtual_space else
            len(selection.internal_vir_keep)
            + len(selection.virtual_lno_keep)
        )
    return {
        "lis_occupied_ranks": [int(value) for value in occupied],
        "lis_virtual_ranks": [int(value) for value in virtual],
        "mean_lis_occ": float(np.mean(occupied)),
        "max_lis_occ": int(np.max(occupied)),
        "mean_lis_vir": float(np.mean(virtual)),
        "max_lis_vir": int(np.max(virtual)),
    }


def select_pair_cutoff(rows, target_mha):
    """Return the loosest sampled pair cutoff below the energy target."""
    target_eh = float(target_mha) * 1e-3
    candidates = [
        row for row in rows
        if row.get("stage") == "pair"
        and abs(float(row["energy_error"])) < target_eh
    ]
    if not candidates:
        raise RuntimeError(
            "no pair cutoff met the requested absolute energy target of "
            f"{target_mha:g} mEh; add a tighter --pair-cutoffs value"
        )
    return max(float(row["pair_cutoff"]) for row in candidates)


def _thresholds(args, pair_cutoff):
    return IAOFragmentMP2Thresholds(
        bp_occ=args.bp_occ,
        bp_primary=args.bp_primary,
        bp_ed=args.bp_ed,
        bp_pao=args.bp_pao,
        pao_norm=args.pao_norm,
        domain_pao=args.domain_pao,
        ed_pao=args.ed_pao,
        occupied_weight=args.occupied_weight,
        metric_rank=args.metric_rank,
        pair_energy=float(pair_cutoff),
        near_pair_distance=args.near_pair_distance,
        multipole_order=args.multipole_order,
        mp2_block_memory_mb=args.mp2_block_memory,
    )


def prepare_outcore_cderi(mol, *, auxbasis, path):
    with_df = (
        df.DF(mol, incore=False)
        if auxbasis is None
        else df.DF(mol, auxbasis=auxbasis, incore=False)
    )
    with_df.max_memory = mol.max_memory
    with_df._cderi_to_save = str(path)
    with_df.build()


def make_build_mf(*, auxbasis, conv_tol, cderi_path):
    def build_mf(mol):
        rhf = scf.RHF(mol)
        mf = (
            rhf.density_fit()
            if auxbasis is None
            else rhf.density_fit(auxbasis=auxbasis)
        )
        mf.with_df.max_memory = mol.max_memory
        mf.with_df.attach_outcore_cderi(str(cderi_path))
        mf.conv_tol = conv_tol
        mf.conv_tol_grad = max(conv_tol**0.5, 1e-10)
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("density-fitted RHF did not converge")
        return mf
    return build_mf


def _canonical_mp2_total(mol, *, build_mf, frozen):
    mf = build_mf(mol)
    e_corr, _ = dfmp2.MP2(mf, frozen=frozen).kernel(with_t2=False)
    return mf.e_tot + e_corr


def _canonical_ccsdt_total(mol, *, build_mf, frozen):
    mf = build_mf(mol)
    cc = dfccsd.RCCSD(mf, frozen=frozen)
    eris = cc.ao2mo()
    cc.kernel(eris=eris)
    if not cc.converged:
        raise RuntimeError("canonical DF-CCSD did not converge")
    return cc.e_tot + cc.ccsd_t(eris=eris)


def estimate_canonical_dfccsdt_gradient_memory(
    *, nocc, nvir, naux, nao, nmo,
):
    """Estimate the live tensor state of the canonical AD reference.

    The factor-native DF implementation no longer stores global ``ovvv`` or
    ``wvvov``.  Its response calculation nevertheless has nine simultaneously
    live ``(o,o,v,v)``-sized arrays: the ``ovov``/``oovv`` ERI blocks,
    converged and response amplitudes, update/output work arrays, and the two
    ``wVOov``/``wVooV`` update/response intermediates.  The estimate also
    includes the persistent three-index factors and smaller occupied blocks.
    It does not include JAX/XLA allocator overhead, DIIS history, cache tiles,
    or BLAS workspaces, so it is a preflight floor rather than a peak-RSS
    prediction.
    """
    nocc = int(nocc)
    nvir = int(nvir)
    naux = int(naux)
    nao = int(nao)
    nmo = int(nmo)
    if min(nocc, nvir, naux, nao, nmo) <= 0:
        raise ValueError("canonical memory dimensions must be positive")

    itemsize = np.dtype(np.float64).itemsize
    t2_bytes = nocc * nocc * nvir * nvir * itemsize
    components = {
        "nine_oovv_live_arrays": 9 * t2_bytes,
        "Loo": naux * nocc * nocc * itemsize,
        "Lov": naux * nocc * nvir * itemsize,
        "Lvv_packed": naux * nvir * (nvir + 1) // 2 * itemsize,
        "oooo": nocc**4 * itemsize,
        "ovoo": nocc**3 * nvir * itemsize,
        "fock_and_mo_coeff": (nmo * nmo + nao * nmo) * itemsize,
    }
    response_bytes = sum(components.values())

    # AO->MO builds Lpq before slicing the persistent factors and four-index
    # occupied/mixed blocks.  This peak is normally lower than response, but
    # include it for small-o/large-auxiliary edge cases.
    ao2mo_bytes = (
        naux * nmo * nmo * itemsize
        + components["Loo"] + components["Lov"]
        + components["Lvv_packed"] + components["oooo"]
        + components["ovoo"] + 2 * t2_bytes
        + components["fock_and_mo_coeff"]
    )
    estimated_bytes = max(response_bytes, ao2mo_bytes)
    gib = 1024.0**3
    return {
        "estimated_bytes": int(estimated_bytes),
        "estimated_gib": float(estimated_bytes / gib),
        "response_live_state_gib": float(response_bytes / gib),
        "ao2mo_live_state_gib": float(ao2mo_bytes / gib),
        "one_oovv_array_gib": float(t2_bytes / gib),
        "components_gib": {
            name: float(value / gib) for name, value in components.items()
        },
        "excluded_overheads": [
            "JAX/XLA allocator and compilation",
            "DIIS history",
            "triples cache tiles and cotangents",
            "BLAS workspaces",
        ],
    }


def _enforce_canonical_ccsdt_memory_preflight(
    estimate, *, max_memory_mb, allow_oversubscription,
):
    """Refuse a canonical reference whose tensor floor exceeds its budget."""
    budget_bytes = float(max_memory_mb) * 1e6
    if budget_bytes <= 0.0:
        raise ValueError("--max-memory must be positive")
    budget_gib = budget_bytes / 1024.0**3
    estimate["configured_budget_gib"] = budget_gib
    estimate["oversubscription_override"] = bool(allow_oversubscription)
    if estimate["estimated_bytes"] <= budget_bytes:
        estimate["accepted"] = True
        return estimate
    if allow_oversubscription:
        estimate["accepted"] = True
        return estimate

    estimate["accepted"] = False
    one = estimate["one_oovv_array_gib"]
    raise RuntimeError(
        "canonical DF-CCSD(T) gradient resource preflight refused this "
        f"calculation: the dimension-based live-tensor floor is "
        f"{estimate['estimated_gib']:.1f} GiB, above the configured "
        f"--max-memory budget of {budget_gib:.1f} GiB.  One (o,o,v,v) "
        f"array is {one:.2f} GiB and the response retains about nine such "
        "arrays in addition to Loo/Lov/packed-Lvv and occupied ERI blocks. "
        "This estimate already assumes the factor-native triples/lambda "
        "paths (no global ovvv or wvvov) and excludes runtime overhead. "
        "Supply a validated --reference-npz/--resume checkpoint, increase "
        "the memory allocation, or pass "
        "--allow-canonical-memory-oversubscription to accept the OOM risk."
    )


def _value_and_grad(function, mol):
    start = time.perf_counter()
    energy, mol_bar = jax.value_and_grad(function)(mol)
    _block((energy, mol_bar))
    return (
        float(np.asarray(energy)), np.asarray(mol_bar.coords),
        time.perf_counter() - start,
    )


def _signature(args, mol):
    frozen = _frozen_count(args.frozen)
    return {
        "geometry_sha256": _geometry_sha256(args.xyz),
        "natom": int(mol.natm),
        "nao": int(mol.nao),
        "nelectron": int(mol.nelectron),
        "basis": _basis_label(args.basis),
        "auxbasis": _auxbasis_label(args.auxbasis),
        "frozen": None if args.frozen is None else frozen,
        "pair_model": str(args.pair_model),
        "pair_cutoffs": [float(value) for value in args.pair_cutoffs],
        "lno_thresholds": [float(value) for value in args.lno_thresholds],
        "lno_vir_ratio": float(args.lno_vir_ratio),
        "fixed_pair_cutoff": (
            None if args.fixed_pair_cutoff is None
            else float(args.fixed_pair_cutoff)
        ),
        "pair_energy_target_mha": float(args.pair_energy_target_mha),
        "internal_rank_threshold": float(args.internal_rank_threshold),
        "scf_conv_tol": float(args.scf_conv_tol),
        "mpi_pair_results": (
            None if args.mpi_pair_results is None
            else _mpi_pair_result_identity(args.mpi_pair_results)
        ),
        "domain_thresholds": asdict(_thresholds(args, 0.0)),
    }


def _same_signature(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _atomic_outputs(prefix, metadata, pair_rows, lno_rows,
                    pair_gradients, lno_gradients, references):
    prefix = Path(prefix).expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = prefix.with_suffix(".csv")
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in (*pair_rows, *lno_rows):
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})
    csv_tmp.replace(csv_path)

    json_path = prefix.with_suffix(".json")
    json_tmp = json_path.with_suffix(".json.tmp")
    payload = dict(metadata)
    payload["references"] = {
        name: {
            key: value for key, value in reference.items()
            if key != "gradient"
        }
        for name, reference in references.items()
    }
    payload["pair_rows"] = pair_rows
    payload["lno_rows"] = lno_rows
    with json_tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=True)
        handle.write("\n")
    json_tmp.replace(json_path)

    natom = int(metadata["signature"]["natom"])
    empty = np.empty((0, natom, 3))
    npz_path = prefix.with_suffix(".npz")
    npz_tmp = npz_path.with_suffix(".npz.tmp")
    with npz_tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            pair_gradients=(
                np.stack(pair_gradients) if pair_gradients else empty
            ),
            lno_gradients=(
                np.stack(lno_gradients) if lno_gradients else empty
            ),
            pair_cutoffs=np.asarray([
                row["pair_cutoff"] for row in pair_rows
            ]),
            lno_occ_thresholds=np.asarray([
                row["lno_occ_threshold"] for row in lno_rows
            ]),
            lno_vir_thresholds=np.asarray([
                row["lno_vir_threshold"] for row in lno_rows
            ]),
            reference_mp2_total_energy=np.asarray(
                references["dfmp2"]["energy"]
            ),
            reference_mp2_gradient=np.asarray(
                references["dfmp2"]["gradient"]
            ),
            reference_ccsdt_total_energy=np.asarray(
                references["dfccsdt"]["energy"]
            ),
            reference_ccsdt_gradient=np.asarray(
                references["dfccsdt"]["gradient"]
            ),
            signature=np.asarray(json.dumps(
                metadata["signature"], sort_keys=True
            )),
        )
    npz_tmp.replace(npz_path)
    return csv_path, npz_path, json_path


def _load_checkpoint(prefix, signature):
    prefix = Path(prefix).expanduser().resolve()
    json_path = prefix.with_suffix(".json")
    npz_path = prefix.with_suffix(".npz")
    if not json_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError(
            "--resume requires both checkpoint JSON and NPZ files"
        )
    with json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not _same_signature(payload.get("signature"), signature):
        raise ValueError("checkpoint settings do not match this run")
    with np.load(npz_path, allow_pickle=False) as arrays:
        references = {
            "dfmp2": {
                **payload["references"]["dfmp2"],
                "gradient": np.array(
                    arrays["reference_mp2_gradient"], copy=True
                ),
            },
            "dfccsdt": {
                **payload["references"]["dfccsdt"],
                "gradient": np.array(
                    arrays["reference_ccsdt_gradient"], copy=True
                ),
            },
        }
        pair_gradients = [
            np.array(value, copy=True) for value in arrays["pair_gradients"]
        ]
        lno_gradients = [
            np.array(value, copy=True) for value in arrays["lno_gradients"]
        ]
    pair_rows = list(payload.get("pair_rows", ()))
    lno_rows = list(payload.get("lno_rows", ()))
    if len(pair_rows) != len(pair_gradients):
        raise ValueError("pair checkpoint rows and gradients are inconsistent")
    if len(lno_rows) != len(lno_gradients):
        raise ValueError("LNO checkpoint rows and gradients are inconsistent")
    return payload, references, pair_rows, lno_rows, pair_gradients, lno_gradients


def _load_reference_npz(path, signature):
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as arrays:
        source_signature = json.loads(str(arrays["signature"].item()))
        # Sweeps may change, but the molecular and domain definition must not.
        keys = (
            "geometry_sha256", "natom", "nao", "nelectron", "basis",
            "auxbasis", "frozen",
        )
        if any(source_signature.get(key) != signature.get(key) for key in keys):
            raise ValueError("reference NPZ describes a different system")
        return {
            "dfmp2": {
                "energy": float(arrays["reference_mp2_total_energy"]),
                "gradient": np.array(
                    arrays["reference_mp2_gradient"], copy=True
                ),
                "seconds": 0.0,
                "provenance": str(path),
            },
            "dfccsdt": {
                "energy": float(arrays["reference_ccsdt_total_energy"]),
                "gradient": np.array(
                    arrays["reference_ccsdt_gradient"], copy=True
                ),
                "seconds": 0.0,
                "provenance": str(path),
            },
        }


def _merge_validated_references(current, loaded):
    """Merge a validated CC reference with an imported MPI MP2 reference."""
    if current is None:
        return loaded
    merged = dict(current)
    if "dfmp2" in merged and "dfmp2" in loaded:
        left = merged["dfmp2"]
        right = loaded["dfmp2"]
        if not np.isclose(
            float(left["energy"]), float(right["energy"]),
            rtol=0.0, atol=1e-8,
        ) or not np.allclose(
            np.asarray(left["gradient"]),
            np.asarray(right["gradient"]),
            rtol=1e-8, atol=1e-8,
        ):
            raise ValueError(
                "validated reference NPZ and MPI pair checkpoint disagree "
                "on the canonical DF-MP2 reference"
            )
    for name, reference in loaded.items():
        if name != "dfmp2" or name not in merged:
            merged[name] = reference
    return merged


def _canonical_memory_preflight_from_dimensions(
    mol, *, frozen, naux, max_memory_mb, allow_oversubscription,
):
    """Run the canonical CCSD(T) tensor-floor check without SCF/CDERI."""
    nocc_total = int(mol.nelectron // 2)
    frozen_count = _frozen_count(frozen)
    nmo = int(mol.nao)
    estimate = estimate_canonical_dfccsdt_gradient_memory(
        nocc=nocc_total - frozen_count,
        nvir=nmo - nocc_total,
        naux=int(naux),
        nao=int(mol.nao),
        nmo=nmo - frozen_count,
    )
    return _enforce_canonical_ccsdt_memory_preflight(
        estimate,
        max_memory_mb=max_memory_mb,
        allow_oversubscription=allow_oversubscription,
    )


def _result_paths(prefix):
    """Return companion NPZ/JSON paths from either path or a bare prefix."""
    prefix = Path(prefix).expanduser().resolve()
    if prefix.suffix.lower() in {".npz", ".json", ".csv"}:
        prefix = prefix.with_suffix("")
    return prefix.with_suffix(".npz"), prefix.with_suffix(".json")


def _load_mpi_pair_results(prefix, args, signature):
    """Validate and convert example-20 MPI pair results.

    The MPI file supplies the canonical DF-MP2 reference, every pair row, and
    their full gradients.  It deliberately does not supply a CCSD(T)
    reference; this driver computes that once before starting the LNO sweep.
    """
    npz_path, json_path = _result_paths(prefix)
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(
            "--mpi-pair-results requires companion NPZ and JSON files"
        )
    with json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    system = payload.get("system", {})
    settings = payload.get("settings", {})
    expected = {
        "natom": signature["natom"],
        "nao": signature["nao"],
        "basis": signature["basis"],
        "auxbasis": signature["auxbasis"],
        "frozen": signature["frozen"],
    }
    for key, value in expected.items():
        observed = system.get(key)
        if key == "basis":
            matches = _basis_label(observed) == _basis_label(value)
        elif key == "auxbasis":
            matches = _auxbasis_label(observed) == _auxbasis_label(value)
        elif key == "frozen":
            matches = (
                observed is None and value is None
            ) or (
                observed is not None and value is not None
                and int(observed) == int(value)
            )
        else:
            matches = observed is not None and int(observed) == int(value)
        if not matches:
            raise ValueError(
                f"MPI pair result system field {key!r} is incompatible: "
                f"got {observed!r}, expected {value!r}"
            )
    source_geometry_sha256 = system.get("geometry_sha256")
    if source_geometry_sha256 is None:
        raise ValueError(
            "MPI pair results lack system.geometry_sha256 provenance"
        )
    if source_geometry_sha256 != signature["geometry_sha256"]:
        raise ValueError("MPI pair results use different geometry contents")
    if settings.get("pair_energy_model") != args.pair_model:
        raise ValueError("MPI pair results use a different pair-energy model")

    # Example 20 uses the production dataclass defaults for every domain
    # threshold except pair_energy and block memory.  Reject a silent mixture
    # if this CC run has customized any of those topology controls.
    source_defaults = asdict(IAOFragmentMP2Thresholds())
    requested = asdict(_thresholds(args, 0.0))
    ignored = {"pair_energy", "mp2_block_memory_mb"}
    for key in source_defaults.keys() - ignored:
        if requested[key] != source_defaults[key]:
            raise ValueError(
                "MPI pair results assume the default domain thresholds, but "
                f"this run changes {key!r}"
            )

    source_rows = payload.get("rows", [])
    with np.load(npz_path, allow_pickle=False) as arrays:
        pair_cutoffs = np.asarray(arrays["pair_thresholds"], dtype=float)
        energies = np.asarray(arrays["total_energies"], dtype=float)
        gradients = np.asarray(arrays["gradients"], dtype=float)
        reference_energy = float(arrays["reference_total_energy"])
        reference_gradient = np.asarray(
            arrays["reference_gradient"], dtype=float
        )
    nrow = len(source_rows)
    if not (
        pair_cutoffs.shape == (nrow,)
        and energies.shape == (nrow,)
        and gradients.shape == (nrow, signature["natom"], 3)
        and reference_gradient.shape == (signature["natom"], 3)
    ):
        raise ValueError("MPI pair JSON and NPZ row shapes are inconsistent")

    converted_rows = []
    converted_gradients = []
    for index, (source, cutoff, energy, gradient) in enumerate(zip(
        source_rows, pair_cutoffs, energies, gradients, strict=True
    )):
        if not np.isclose(
            cutoff, float(source["pair_threshold"]), rtol=0.0, atol=1e-15
        ):
            raise ValueError(f"MPI pair row {index} has inconsistent cutoff")
        if not np.isclose(
            energy, float(source["e_total"]), rtol=0.0, atol=1e-10
        ):
            raise ValueError(f"MPI pair row {index} has inconsistent energy")
        statistics = {
            "nfragment": int(source["nfragment"]),
            "strong_pair_count": int(source["strong_pair_count"]),
            "weak_pair_count": int(source["weak_pair_count"]),
            "total_pair_count": int(source["total_pair_count"]),
            "mean_ed_atoms": float(source["mean_ed_atoms"]),
            "max_ed_atoms": int(source["max_ed_atoms"]),
            "mean_ed_aos": float(source["mean_ed_aos"]),
            "max_ed_aos": int(source["max_ed_aos"]),
            "mean_ed_occ": None,
            "max_ed_occ": None,
            "mean_ed_vir": None,
            "max_ed_vir": None,
        }
        converted_rows.append(_row(
            "pair", f"{cutoff:.1e}", "canonical DF-MP2", cutoff,
            energy, gradient,
            {"energy": reference_energy, "gradient": reference_gradient},
            statistics,
            {
                "mean_lis_occ": None, "max_lis_occ": None,
                "mean_lis_vir": None, "max_lis_vir": None,
                "topology_seconds": float(
                    source["topology_statistics_seconds"]
                ),
                "static_selection_seconds": 0.0,
                "gradient_seconds": float(
                    source["mpi_energy_gradient_seconds"]
                ),
                "row_total_seconds": float(
                    source["topology_statistics_seconds"]
                    + source["mpi_energy_gradient_seconds"]
                ),
            },
        ))
        converted_rows[-1]["source_mpi_ranks"] = settings.get("mpi_ranks")
        converted_rows[-1]["source_row_index"] = index
        converted_rows[-1]["peak_rss_gib"] = None
        converted_gradients.append(np.array(gradient, copy=True))

    timing = payload.get("reference", {}).get("timing", {})
    reference_seconds = sum(
        float(value) for key, value in timing.items()
        if key.endswith("_seconds")
    )
    reference = {
        "energy": reference_energy,
        "gradient": np.array(reference_gradient, copy=True),
        "seconds": reference_seconds,
        "provenance": str(npz_path),
    }
    return reference, converted_rows, converted_gradients, payload


def _row(stage, label, reference_method, pair_cutoff, energy, gradient,
         reference, statistics, timings, *, lno_occ=None, lno_vir=None):
    energy_error = float(energy - reference["energy"])
    return {
        "stage": stage,
        "label": label,
        "reference_method": reference_method,
        "pair_cutoff": float(pair_cutoff),
        "lno_occ_threshold": lno_occ,
        "lno_vir_threshold": lno_vir,
        "energy": float(energy),
        "energy_error": energy_error,
        "energy_abs_error_mha": abs(energy_error) * 1e3,
        **_error_statistics(gradient, reference["gradient"]),
        **statistics,
        **timings,
        "peak_rss_gib": _peak_rss_gib(),
    }


def _print_row(row):
    lno = "" if row["stage"] == "pair" else (
        f" LNO={row['lno_occ_threshold']:.1e}/"
        f"{row['lno_vir_threshold']:.1e}"
    )
    print(
        f"{row['stage']:>4s} {row['label']:>9s}{lno}: "
        f"dE={row['energy_error']:+.3e} Eh "
        f"max|dG|={row['gradient_max_abs_error']:.3e} Eh/bohr "
        f"strong={row['strong_pair_count']}/{row['total_pair_count']} "
        f"wall={row['row_total_seconds']:.1f} s",
        flush=True,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz", default=str(DEFAULT_XYZ))
    parser.add_argument("--basis", default="cc-pvtz")
    parser.add_argument(
        "--auxbasis", type=_parse_auxbasis, default=None,
        help="auxiliary basis; auto (default) uses the OMCB reference setup",
    )
    parser.add_argument(
        "--frozen", type=_parse_frozen, default=None,
        help="frozen occupied orbitals; none (default) is all-electron",
    )
    parser.add_argument(
        "--pair-cutoffs", type=float, nargs="+",
        default=[3e-2, 1e-2, 1e-3],
    )
    parser.add_argument(
        "--fixed-pair-cutoff", type=_parse_optional_float, default=1e-3,
        help=(
            "pair cutoff for the CCSD(T) LNO sweep (default: 1e-3); "
            "pass 'auto' to select from the pair rows"
        ),
    )
    parser.add_argument("--pair-energy-target-mha", type=float, default=1.0)
    parser.add_argument(
        "--lno-thresholds", type=float, nargs="+",
        default=[1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5],
        help="occupied LNO thresholds",
    )
    parser.add_argument("--lno-vir-ratio", type=float, default=0.1)
    parser.add_argument(
        "--pair-model", choices=("multipole", "exact", "all"),
        default="multipole",
    )
    parser.add_argument("--bp-occ", type=float, default=0.985)
    parser.add_argument("--bp-primary", type=float, default=0.999)
    parser.add_argument("--bp-ed", type=float, default=0.9998)
    parser.add_argument("--bp-pao", type=float, default=0.98)
    parser.add_argument("--pao-norm", type=float, default=1e-4)
    parser.add_argument("--domain-pao", type=float, default=1e-4)
    parser.add_argument("--ed-pao", type=float, default=0.995)
    parser.add_argument("--occupied-weight", type=float, default=1e-4)
    parser.add_argument("--metric-rank", type=float, default=1e-10)
    parser.add_argument("--near-pair-distance", type=float, default=3.5)
    parser.add_argument("--multipole-order", type=int, default=4)
    parser.add_argument("--mp2-block-memory", type=float, default=128.0)
    parser.add_argument("--internal-rank-threshold", type=float, default=1e-6)
    parser.add_argument("--scf-conv-tol", type=float, default=1e-10)
    parser.add_argument("--max-memory", type=float, default=12000.0)
    parser.add_argument(
        "--allow-canonical-memory-oversubscription", action="store_true",
        help=(
            "run a missing canonical DF-CCSD(T) gradient even when its "
            "dimension-based tensor floor exceeds --max-memory"
        ),
    )
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument(
        "--reference-npz", default=None,
        help=(
            "validated example-21 NPZ containing compatible canonical "
            "DF-MP2 and DF-CCSD(T) energy/gradient references"
        ),
    )
    parser.add_argument(
        "--mpi-pair-results", type=_parse_optional_path,
        default=DEFAULT_MPI_PAIR_RESULTS, metavar="PREFIX|NPZ|JSON|none",
        help=(
            "reuse canonical DF-MP2 and pair rows from the gauge-safe "
            "example-20 MPI checkpoint; pass 'none' for a serial smoke run"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    nonnegative = {
        "--pair-energy-target-mha": args.pair_energy_target_mha,
        "--lno-vir-ratio": args.lno_vir_ratio,
        "--internal-rank-threshold": args.internal_rank_threshold,
        **{f"--pair-cutoffs[{index}]": value
           for index, value in enumerate(args.pair_cutoffs)},
        **{f"--lno-thresholds[{index}]": value
           for index, value in enumerate(args.lno_thresholds)},
    }
    for name, value in nonnegative.items():
        if value < 0.0:
            parser.error(f"{name} must be non-negative")
    if args.fixed_pair_cutoff is not None and args.fixed_pair_cutoff < 0.0:
        parser.error("--fixed-pair-cutoff must be non-negative")
    if args.reference_npz is not None and args.resume:
        parser.error("--reference-npz and --resume are mutually exclusive")
    if args.max_memory <= 0.0:
        parser.error("--max-memory must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    for key, value in {
        "pyscfad_moleintor_opt": True,
        "pyscfad_scf_implicit_diff": True,
        "pyscfad_scf_first_order_custom": False,
        "pyscfad_ccsd_implicit_diff": True,
        "pyscfad_dfccsd_custom_response": True,
    }.items():
        config.update(key, value)

    xyz = Path(args.xyz).expanduser().resolve()
    if not xyz.is_file():
        raise FileNotFoundError(f"XYZ file does not exist: {xyz}")
    mol = gto.Mole(
        atom=str(xyz), basis=args.basis, verbose=args.verbose,
        max_memory=args.max_memory,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    nocc_total = mol.nelectron // 2
    frozen_count = _frozen_count(args.frozen)
    if frozen_count >= nocc_total:
        raise ValueError("--frozen leaves no active occupied orbitals")

    signature = _signature(args, mol)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "geometry_path": str(xyz),
        "signature": signature,
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "pyscf_version": pyscf.__version__,
        "host": platform.node(),
        "gradient_units": "Eh/bohr",
        "energy_units": "Eh",
        "topology_derivative": "fixed at reference geometry",
        "runtime": {
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "veclib_maximum_threads": os.environ.get(
                "VECLIB_MAXIMUM_THREADS"
            ),
        },
    }

    references = None
    pair_rows = []
    lno_rows = []
    pair_gradients = []
    lno_gradients = []
    source_payload = None
    if args.resume:
        (old_metadata, references, pair_rows, lno_rows,
         pair_gradients, lno_gradients) = _load_checkpoint(
            args.output_prefix, signature
        )
        metadata["created_utc"] = old_metadata.get(
            "created_utc", metadata["created_utc"]
        )
        if "mpi_pair_results" in old_metadata:
            metadata["mpi_pair_results"] = old_metadata["mpi_pair_results"]
        print(
            f"Resuming {len(pair_rows)} pair and {len(lno_rows)} LNO rows",
            flush=True,
        )
    elif args.mpi_pair_results is not None:
        mp2_reference, pair_rows, pair_gradients, source_payload = (
            _load_mpi_pair_results(args.mpi_pair_results, args, signature)
        )
        references = {"dfmp2": mp2_reference}
        metadata["mpi_pair_results"] = {
            "npz": str(_result_paths(args.mpi_pair_results)[0]),
            "json": str(_result_paths(args.mpi_pair_results)[1]),
            "mpi_ranks": source_payload.get("settings", {}).get("mpi_ranks"),
            "imported_rows": len(pair_rows),
        }
        print(
            f"Imported {len(pair_rows)} gauge-safe MPI pair rows from "
            f"{metadata['mpi_pair_results']['npz']}",
            flush=True,
        )

    if args.reference_npz is not None:
        loaded_references = _load_reference_npz(
            args.reference_npz, signature
        )
        references = _merge_validated_references(
            references, loaded_references
        )
        metadata["validated_reference_npz"] = str(
            Path(args.reference_npz).expanduser().resolve()
        )

    # The exact default MPI artifact records naux, so reject the impossible
    # all-electron canonical AD reference before allocating CDERI or running
    # SCF.  Explicit small-system serial smoke runs are checked later, after
    # their actual auxiliary dimension is known from the eager mean field.
    if (
        (references is None or "dfccsdt" not in references)
        and source_payload is not None
        and source_payload.get("system", {}).get("naux") is not None
    ):
        metadata["canonical_ccsdt_memory_preflight"] = (
            _canonical_memory_preflight_from_dimensions(
                mol,
                frozen=args.frozen,
                naux=int(source_payload["system"]["naux"]),
                max_memory_mb=args.max_memory,
                allow_oversubscription=(
                    args.allow_canonical_memory_oversubscription
                ),
            )
        )

    scratch = tempfile.TemporaryDirectory(prefix="iao_dlno_ccsdt_cderi_")
    try:
        cderi_path = Path(scratch.name) / "cderi.h5"
        start = time.perf_counter()
        prepare_outcore_cderi(
            mol, auxbasis=args.auxbasis, path=cderi_path
        )
        metadata["outcore_cderi_build_seconds"] = (
            time.perf_counter() - start
        )
        build_mf = make_build_mf(
            auxbasis=args.auxbasis,
            conv_tol=args.scf_conv_tol,
            cderi_path=cderi_path,
        )

        eager_mf = build_mf(mol)
        nmo = int(eager_mf.mo_coeff.shape[1])
        metadata["system"] = {
            "natom": int(mol.natm),
            "nao": int(mol.nao),
            "nmo": nmo,
            "nocc_total": nocc_total,
            "nocc_active": nocc_total - frozen_count,
            "nvir_active": nmo - nocc_total,
            "naux": int(eager_mf.with_df.get_naoaux()),
        }
        print(
            f"Benchmark system: {mol.natm} atoms, {mol.nao} AOs, "
            f"active {nocc_total - frozen_count}o/{nmo - nocc_total}v",
            flush=True,
        )

        if references is None:
            references = {}
        if "dfmp2" not in references:
            e_mp2, g_mp2, seconds = _value_and_grad(
                lambda mol_: _canonical_mp2_total(
                    mol_, build_mf=build_mf, frozen=args.frozen
                ),
                mol,
            )
            references["dfmp2"] = {
                "energy": e_mp2,
                "gradient": g_mp2,
                "seconds": seconds,
                "provenance": "computed_in_run",
            }
            print(
                f"Canonical DF-MP2 E={e_mp2:.12f}, {seconds:.1f} s",
                flush=True,
            )
            gc.collect()
        if "dfccsdt" not in references:
            estimate = estimate_canonical_dfccsdt_gradient_memory(
                nocc=nocc_total - frozen_count,
                nvir=nmo - nocc_total,
                naux=int(eager_mf.with_df.get_naoaux()),
                nao=int(mol.nao),
                nmo=nmo - frozen_count,
            )
            metadata["canonical_ccsdt_memory_preflight"] = estimate
            _enforce_canonical_ccsdt_memory_preflight(
                estimate,
                max_memory_mb=args.max_memory,
                allow_oversubscription=(
                    args.allow_canonical_memory_oversubscription
                ),
            )
            print(
                "Canonical DF-CCSD(T) tensor preflight: "
                f"{estimate['estimated_gib']:.2f} GiB floor for "
                f"{estimate['configured_budget_gib']:.2f} GiB budget",
                flush=True,
            )
            e_cc, g_cc, seconds = _value_and_grad(
                lambda mol_: _canonical_ccsdt_total(
                    mol_, build_mf=build_mf, frozen=args.frozen
                ),
                mol,
            )
            references["dfccsdt"] = {
                "energy": e_cc,
                "gradient": g_cc,
                "seconds": seconds,
                "provenance": "computed_in_run",
            }
            print(
                f"Canonical DF-CCSD(T) E={e_cc:.12f}, {seconds:.1f} s",
                flush=True,
            )
            gc.collect()
        _atomic_outputs(
            args.output_prefix, metadata, pair_rows, lno_rows,
            pair_gradients, lno_gradients, references,
        )

        completed_pair = {
            float(row["pair_cutoff"]) for row in pair_rows
        }
        pair_jobs = (
            () if args.mpi_pair_results is not None else args.pair_cutoffs
        )
        for cutoff in pair_jobs:
            cutoff = float(cutoff)
            if cutoff in completed_pair:
                continue
            row_start = time.perf_counter()
            thresholds = _thresholds(args, cutoff)
            selection_timing = {}

            def build_static(mf_):
                start = time.perf_counter()
                topology = build_iao_fragment_topology(
                    mf_, frozen=args.frozen, thresholds=thresholds,
                    pair_energy_model=args.pair_model,
                )
                selection_timing["topology"] = time.perf_counter() - start
                start = time.perf_counter()
                static_ = build_iao_mp2_static_selections(mf_, topology)
                selection_timing["static"] = time.perf_counter() - start
                return static_

            static = stop_trace(build_static)(eager_mf)
            statistics = _domain_statistics(static)
            start = time.perf_counter()
            energy, mol_bar = IAOFragmentMP2.value_and_grad(
                mol,
                build_mf=build_mf,
                frozen=args.frozen,
                thresholds=thresholds,
                pair_energy_model=args.pair_model,
                topology=static,
                include_hf=True,
            )
            _block((energy, mol_bar))
            gradient_seconds = time.perf_counter() - start
            gradient = np.asarray(mol_bar.coords)
            row = _row(
                "pair", f"{cutoff:.1e}", "canonical DF-MP2", cutoff,
                float(energy), gradient, references["dfmp2"], statistics,
                {
                    "mean_lis_occ": None, "max_lis_occ": None,
                    "mean_lis_vir": None, "max_lis_vir": None,
                    "topology_seconds": selection_timing["topology"],
                    "static_selection_seconds": selection_timing["static"],
                    "gradient_seconds": gradient_seconds,
                    "row_total_seconds": time.perf_counter() - row_start,
                },
            )
            pair_rows.append(row)
            pair_gradients.append(gradient)
            _atomic_outputs(
                args.output_prefix, metadata, pair_rows, lno_rows,
                pair_gradients, lno_gradients, references,
            )
            _print_row(row)
            del static, mol_bar
            gc.collect()

        fixed_pair_cutoff = (
            float(args.fixed_pair_cutoff)
            if args.fixed_pair_cutoff is not None
            else select_pair_cutoff(pair_rows, args.pair_energy_target_mha)
        )
        if args.mpi_pair_results is not None and not any(
            np.isclose(
                fixed_pair_cutoff, float(row["pair_cutoff"]),
                rtol=0.0, atol=1e-15,
            )
            for row in pair_rows
        ):
            raise ValueError(
                "the fixed pair cutoff is absent from the imported MPI sweep"
            )
        metadata["selected_pair_cutoff"] = fixed_pair_cutoff
        metadata["pair_selection"] = (
            "explicit --fixed-pair-cutoff"
            if args.fixed_pair_cutoff is not None
            else "loosest sampled cutoff satisfying the energy target"
        )
        print(
            f"Fixed pair cutoff for CCSD(T): {fixed_pair_cutoff:.1e}",
            flush=True,
        )

        completed_lno = {
            (float(row["lno_occ_threshold"]),
             float(row["lno_vir_threshold"]))
            for row in lno_rows
        }
        thresholds = _thresholds(args, fixed_pair_cutoff)
        paths = _atomic_outputs(
            args.output_prefix, metadata, pair_rows, lno_rows,
            pair_gradients, lno_gradients, references,
        )
        for occ_threshold in args.lno_thresholds:
            occ_threshold = float(occ_threshold)
            vir_threshold = occ_threshold * float(args.lno_vir_ratio)
            if (occ_threshold, vir_threshold) in completed_lno:
                continue
            row_start = time.perf_counter()
            selection_timing = {}

            def build_cc_static(mf_):
                start = time.perf_counter()
                topology = build_iao_fragment_topology(
                    mf_, frozen=args.frozen, thresholds=thresholds,
                    pair_energy_model=args.pair_model,
                )
                selection_timing["topology"] = time.perf_counter() - start
                start = time.perf_counter()
                mp2_static = build_iao_mp2_static_selections(mf_, topology)
                common = rebuild_iao_mp2_common(mf_, mp2_static)
                static_ = build_iao_lis_static_selections(
                    mf_, mp2_static, common=common,
                    thresh_occ=occ_threshold,
                    thresh_vir=vir_threshold,
                    internal_rank_threshold=args.internal_rank_threshold,
                )
                selection_timing["static"] = time.perf_counter() - start
                return static_

            static = stop_trace(build_cc_static)(eager_mf)
            statistics = {
                **_domain_statistics(static.mp2_static),
                **_lis_statistics(static),
            }
            start = time.perf_counter()
            energy, mol_bar = DLNOCCSD.value_and_grad(
                mol,
                build_mf=build_mf,
                frozen=args.frozen,
                thresholds=thresholds,
                pair_energy_model=args.pair_model,
                thresh_occ=occ_threshold,
                thresh_vir=vir_threshold,
                internal_rank_threshold=args.internal_rank_threshold,
                ccsd_t=True,
                static_selections=static,
            )
            _block((energy, mol_bar))
            gradient_seconds = time.perf_counter() - start
            gradient = np.asarray(mol_bar.coords)
            row = _row(
                "lno", f"{occ_threshold:.1e}",
                "canonical DF-CCSD(T)", fixed_pair_cutoff,
                float(energy), gradient, references["dfccsdt"], statistics,
                {
                    "topology_seconds": selection_timing["topology"],
                    "static_selection_seconds": selection_timing["static"],
                    "gradient_seconds": gradient_seconds,
                    "row_total_seconds": time.perf_counter() - row_start,
                },
                lno_occ=occ_threshold, lno_vir=vir_threshold,
            )
            lno_rows.append(row)
            lno_gradients.append(gradient)
            paths = _atomic_outputs(
                args.output_prefix, metadata, pair_rows, lno_rows,
                pair_gradients, lno_gradients, references,
            )
            _print_row(row)
            del static, mol_bar
            gc.collect()

        print("\nCompleted rows:", flush=True)
        print(
            "| stage | cutoff | LNO occ/vir | dE (mEh) | "
            "max |dG| (Eh/bohr) | strong pairs |",
            flush=True,
        )
        print("|---|---:|---:|---:|---:|---:|", flush=True)
        for row in (*pair_rows, *lno_rows):
            lno = "--" if row["stage"] == "pair" else (
                f"{row['lno_occ_threshold']:.1e}/"
                f"{row['lno_vir_threshold']:.1e}"
            )
            print(
                f"| {row['stage']} | {row['pair_cutoff']:.1e} | {lno} | "
                f"{row['energy_error'] * 1e3:+.4f} | "
                f"{row['gradient_max_abs_error']:.3e} | "
                f"{row['strong_pair_count']}/{row['total_pair_count']} |",
                flush=True,
            )
        print("Wrote " + ", ".join(str(path) for path in paths), flush=True)
        return pair_rows, lno_rows
    finally:
        scratch.cleanup()


if __name__ == "__main__":
    main()
