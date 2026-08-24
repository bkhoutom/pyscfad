"""Fixed-topology IAO-fragment MP2 energy/gradient convergence benchmark.

The production-sized default is the linear alkane C25H52 in cc-pVDZ with
cc-pVDZ-RI and one frozen carbon 1s orbital per carbon.  A substantially
smaller smoke test can be run with, for example::

    python examples/lno/17-iao_fragment_mp2_gradient_convergence.py \
        --ncarbon 1 --basis sto-3g --auxbasis weigend \
        --pair-cutoffs 1e-4 --output-prefix /tmp/iao_mp2_grad_smoke

One differentiable DF-RHF calculation and its saved response map are reused
for the canonical DF-MP2 reference and every local threshold.  Each local
topology is nevertheless built independently at the reference geometry and
then held fixed during its reverse pass.  Thus atom lists, strong/weak pair
classes, PAO column choices, and retained ranks are non-differentiable, while
all continuous orbital, integral, strong-pair, and weak-multipole quantities
remain differentiable.

After one run has produced a checkpoint, ``--reference-npz PATH`` reuses its
canonical energies and full gradient.  The current DF-RHF calculation is
still built so every local row can use its saved response map, but canonical
DF-MP2 differentiation and the canonical SCF pullback are then skipped.

The driver writes ``.csv`` scalar summaries, ``.npz`` full gradients, and a
``.json`` provenance/metadata record after every completed cutoff.  This
checkpointing preserves completed points if a later, tighter point fails.
All gradients and gradient errors are reported in Eh/bohr.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np
import pyscf

from pyscfad import config, df, gto, scf
from pyscfad.dlno.alkane import make_n_alkane
from pyscfad.dlno.iao_mp2 import (
    IAOFragmentMP2Thresholds,
    build_iao_fragment_topology,
)
from pyscfad.dlno.iao_mp2_grad import (
    build_iao_mp2_static_selections,
    correlation_value_and_grad,
)
from pyscfad.mp import dfmp2
from pyscfad.ops import stop_trace


def build_molecule(atoms, *, basis, verbose, max_memory):
    mol = gto.Mole()
    mol.atom = atoms
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.verbose = verbose
    mol.max_memory = max_memory
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def prepare_outcore_cderi(mol, *, auxbasis, path):
    """Build one reusable CDERI file outside all JAX traces."""
    with_df = df.DF(mol, auxbasis=auxbasis, incore=False)
    with_df.max_memory = mol.max_memory
    with_df._cderi_to_save = str(path)
    with_df.build()


def make_build_mf(*, auxbasis, conv_tol, cderi_path):
    """Return the common differentiable DF-RHF builder used by every row."""
    def build_mf(mol):
        mf = scf.RHF(mol).density_fit(auxbasis=auxbasis)
        mf.with_df.max_memory = mol.max_memory
        mf.with_df.attach_outcore_cderi(str(cderi_path))
        mf.conv_tol = conv_tol
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("density-fitted RHF did not converge")
        return mf
    return build_mf


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


def _block(value):
    jax.block_until_ready(value)
    return value


def _canonical_correlation_value_and_grad(mf, frozen):
    """Return canonical correlation energy, SCF cotangent, and timings."""
    def energy(mf_):
        e_corr, _ = dfmp2.MP2(mf_, frozen=frozen).kernel(with_t2=False)
        return e_corr

    start = time.perf_counter()
    e_corr, pullback = jax.vjp(energy, mf)
    _block(e_corr)
    forward_seconds = time.perf_counter() - start
    start = time.perf_counter()
    mf_bar, = pullback(jnp.ones((), dtype=jnp.asarray(e_corr).dtype))
    _block(mf_bar)
    reverse_seconds = time.perf_counter() - start
    return e_corr, mf_bar, forward_seconds, reverse_seconds


_REFERENCE_SYSTEM_FIELDS = (
    "formula", "ncarbon", "natom", "nao", "nmo", "nocc",
    "nactive_occ", "nactive_vir", "naux", "basis", "auxbasis",
    "frozen_core",
)


def _npz_scalar(arrays, key):
    value = np.asarray(arrays[key])
    if value.size != 1:
        raise ValueError(
            f"reference NPZ field {key!r} must contain exactly one value"
        )
    value = value.reshape(()).item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


def _same_system_value(key, observed, expected):
    if key in {"formula", "basis", "auxbasis"}:
        return str(observed).strip().lower() == str(expected).strip().lower()
    try:
        return int(observed) == int(expected)
    except (TypeError, ValueError):
        return observed == expected


def _load_reference_npz(
    path, *, e_hf, expected_gradient_shape, expected_system,
):
    """Load and validate a canonical reference produced by this driver.

    A companion JSON (or equivalent embedded provenance) must identify the
    validated standard fixed-point implicit SCF response.  Checkpoints made
    with the experimental finite-replay response are intentionally rejected,
    even when their energies and system dimensions otherwise match.
    """
    start = time.perf_counter()
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"reference NPZ does not exist: {path}")

    required = {
        "reference_gradient",
        "reference_correlation_energy",
        "reference_total_energy",
    }
    with np.load(path, allow_pickle=False) as arrays:
        missing = sorted(required.difference(arrays.files))
        if missing:
            raise ValueError(
                "reference NPZ is missing required field(s): "
                + ", ".join(missing)
            )
        gradient = np.array(arrays["reference_gradient"], dtype=float, copy=True)
        e_corr = float(_npz_scalar(arrays, "reference_correlation_energy"))
        e_total = float(_npz_scalar(arrays, "reference_total_energy"))
        embedded_system = {
            key: _npz_scalar(arrays, f"system_{key}")
            for key in _REFERENCE_SYSTEM_FIELDS
            if f"system_{key}" in arrays.files
        }
        embedded_scf = {
            key: _npz_scalar(arrays, f"scf_{key}")
            for key in ("response_backend", "first_order_custom")
            if f"scf_{key}" in arrays.files
        }

    expected_gradient_shape = tuple(expected_gradient_shape)
    if gradient.shape != expected_gradient_shape:
        raise ValueError(
            "reference gradient shape is incompatible with this molecule: "
            f"got {gradient.shape}, expected {expected_gradient_shape}"
        )
    if not np.all(np.isfinite(gradient)):
        raise ValueError("reference gradient contains non-finite values")
    if not np.isfinite(e_corr) or not np.isfinite(e_total):
        raise ValueError("reference energies must be finite")

    current_hf = float(np.asarray(e_hf))
    source_hf = e_total - e_corr
    if not np.isclose(source_hf, current_hf, rtol=1e-10, atol=1e-8):
        raise ValueError(
            "reference NPZ is incompatible with the current HF calculation: "
            f"E_total-E_corr={source_hf:.12f}, E_HF={current_hf:.12f}"
        )

    companion_path = path.with_suffix(".json")
    companion_system = {}
    companion_scf = {}
    if companion_path.is_file():
        try:
            with companion_path.open(encoding="utf-8") as handle:
                companion = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"could not read reference companion JSON {companion_path}"
            ) from error
        if isinstance(companion.get("system"), dict):
            companion_system = companion["system"]
        if isinstance(companion.get("scf"), dict):
            companion_scf = companion["scf"]

    scf_sources = [
        ("NPZ", embedded_scf),
        ("companion JSON", companion_scf),
    ]
    valid_scf_sources = []
    for source_name, source_scf in scf_sources:
        if not source_scf:
            continue
        backend = source_scf.get("response_backend")
        custom = source_scf.get("first_order_custom")
        if (
            backend != "standard_fixed_point_implicit"
            or custom is not False
        ):
            raise ValueError(
                f"reference {source_name} used an unvalidated SCF response "
                f"backend: response_backend={backend!r}, "
                f"first_order_custom={custom!r}"
            )
        valid_scf_sources.append(source_name)
    if not valid_scf_sources:
        raise ValueError(
            "reference checkpoint lacks validated SCF response provenance; "
            "regenerate it with the standard fixed-point implicit backend"
        )

    checked_fields = []
    for source_name, source_system in (
        ("NPZ", embedded_system), ("companion JSON", companion_system),
    ):
        for key in _REFERENCE_SYSTEM_FIELDS:
            if key not in source_system or key not in expected_system:
                continue
            if not _same_system_value(
                key, source_system[key], expected_system[key]
            ):
                raise ValueError(
                    f"reference {source_name} system field {key!r} is "
                    f"incompatible: got {source_system[key]!r}, expected "
                    f"{expected_system[key]!r}"
                )
            checked_fields.append(f"{source_name}:{key}")

    load_seconds = time.perf_counter() - start
    reference = {
        "e_corr": e_corr,
        "e_total": e_total,
        "provenance": "reused_npz",
        "reused": True,
        "source_npz": str(path),
        "source_companion_json": (
            str(companion_path) if companion_path.is_file() else None
        ),
        "compatibility_checks": [
            "gradient_shape",
            "finite_energy_and_gradient",
            "hf_energy_from_total_minus_correlation",
            *(f"{name}:validated_scf_response" for name in valid_scf_sources),
            *checked_fields,
        ],
        "load_seconds": load_seconds,
        "correlation_forward_seconds": 0.0,
        "correlation_reverse_seconds": 0.0,
        "scf_pullback_seconds": 0.0,
        "gradient_l2_norm": float(np.linalg.norm(gradient)),
        "gradient_max_abs": float(np.max(np.abs(gradient))),
    }
    return gradient, reference


def _domain_statistics(static):
    fragments = static.fragments
    atom_sizes = np.asarray([len(fragment.extended_atoms) for fragment in fragments])
    ao_sizes = np.asarray([
        len(fragment.extended_ao_indices) for fragment in fragments
    ])
    occ_sizes = np.asarray([
        len(fragment.strong_occ_metric_keep) for fragment in fragments
    ])
    vir_sizes = np.asarray([
        len(fragment.strong_virtual.metric_keep) for fragment in fragments
    ])
    upper = np.triu(np.asarray(static.strong_mask, dtype=bool), k=1)
    total_pairs = len(fragments) * (len(fragments) - 1) // 2
    strong_pairs = int(np.count_nonzero(upper))
    return {
        "nfragment": len(fragments),
        "total_pair_count": total_pairs,
        "strong_pair_count": strong_pairs,
        "weak_pair_count": total_pairs - strong_pairs,
        "mean_ed_atoms": float(atom_sizes.mean()),
        "max_ed_atoms": int(atom_sizes.max()),
        "mean_ed_aos": float(ao_sizes.mean()),
        "max_ed_aos": int(ao_sizes.max()),
        "mean_ed_occ": float(occ_sizes.mean()),
        "max_ed_occ": int(occ_sizes.max()),
        "mean_ed_vir": float(vir_sizes.mean()),
        "max_ed_vir": int(vir_sizes.max()),
    }


def _error_statistics(gradient, reference_gradient):
    error = np.asarray(gradient) - np.asarray(reference_gradient)
    reference_norm = float(np.linalg.norm(reference_gradient))
    error_norm = float(np.linalg.norm(error))
    return {
        "gradient_max_abs_error": float(np.max(np.abs(error))),
        "gradient_rms_error": float(np.sqrt(np.mean(error * error))),
        "gradient_l2_error": error_norm,
        "gradient_relative_l2_error": (
            error_norm / reference_norm if reference_norm else float("nan")
        ),
    }


CSV_FIELDS = (
    "label", "pair_cutoff", "e_corr", "e_total", "energy_error",
    "gradient_max_abs_error", "gradient_rms_error", "gradient_l2_error",
    "gradient_relative_l2_error", "strong_pair_count", "weak_pair_count",
    "total_pair_count", "nfragment", "mean_ed_atoms", "max_ed_atoms",
    "mean_ed_aos", "max_ed_aos", "mean_ed_occ", "max_ed_occ",
    "mean_ed_vir", "max_ed_vir", "topology_seconds",
    "static_selection_seconds", "correlation_gradient_seconds",
    "scf_pullback_seconds", "row_total_seconds",
)


def write_outputs(prefix, metadata, rows, gradients, reference_gradient):
    """Atomically checkpoint all three machine-readable output formats."""
    prefix = Path(prefix).expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = prefix.with_suffix(".csv")
    csv_tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})
    csv_tmp.replace(csv_path)

    json_path = prefix.with_suffix(".json")
    json_tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    payload = dict(metadata)
    payload["rows"] = rows
    with json_tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=True)
        handle.write("\n")
    json_tmp.replace(json_path)

    npz_path = prefix.with_suffix(".npz")
    npz_tmp = npz_path.with_suffix(npz_path.suffix + ".tmp")
    shape = (0,) + tuple(np.asarray(reference_gradient).shape)
    local_gradients = (
        np.stack(gradients, axis=0) if gradients else np.empty(shape)
    )
    with npz_tmp.open("wb") as handle:
        arrays = {
            "pair_cutoffs": np.asarray([
                row["pair_cutoff"] for row in rows
            ]),
            "correlation_energies": np.asarray([
                row["e_corr"] for row in rows
            ]),
            "total_energies": np.asarray([
                row["e_total"] for row in rows
            ]),
            "gradients": local_gradients,
            "reference_gradient": np.asarray(reference_gradient),
            "reference_correlation_energy": np.asarray(
                metadata["reference"]["e_corr"]
            ),
            "reference_total_energy": np.asarray(
                metadata["reference"]["e_total"]
            ),
        }
        system = metadata.get("system", {})
        arrays.update({
            f"system_{key}": np.asarray(system[key])
            for key in _REFERENCE_SYSTEM_FIELDS if key in system
        })
        scf = metadata.get("scf", {})
        arrays.update({
            f"scf_{key}": np.asarray(scf[key])
            for key in ("response_backend", "first_order_custom")
            if key in scf
        })
        np.savez_compressed(handle, **arrays)
    npz_tmp.replace(npz_path)
    return csv_path, npz_path, json_path


def _print_row(row):
    print(
        f"{row['label']:>10s}  E_corr={row['e_corr']:+.12f}  "
        f"dE={row['energy_error']:+.3e}  "
        f"max|dG|={row['gradient_max_abs_error']:.3e}  "
        f"strong={row['strong_pair_count']}/{row['total_pair_count']}  "
        f"ED AO(mean/max)={row['mean_ed_aos']:.1f}/{row['max_ed_aos']}  "
        f"topo/grad/SCF={row['topology_seconds'] + row['static_selection_seconds']:.1f}/"
        f"{row['correlation_gradient_seconds']:.1f}/"
        f"{row['scf_pullback_seconds']:.1f} s"
    )


def _parse_frozen_core(value):
    if value.lower() == "auto":
        return None
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("frozen core must be non-negative")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncarbon", type=int, default=25)
    parser.add_argument("--basis", default="cc-pvdz")
    parser.add_argument("--auxbasis", default="cc-pvdz-ri")
    parser.add_argument(
        "--frozen-core", type=_parse_frozen_core, default=None, metavar="N|auto",
        help="frozen occupied orbitals; auto (default) freezes one C 1s per C",
    )
    parser.add_argument(
        "--pair-cutoffs", "--pair-thresholds", dest="pair_cutoffs",
        type=float, nargs="+", default=[1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 0.0],
    )
    parser.add_argument(
        "--pair-model", choices=("multipole", "exact"), default="multipole"
    )
    parser.add_argument("--bp-occ", type=float, default=0.985)
    parser.add_argument("--bp-primary", type=float, default=0.999)
    parser.add_argument("--bp-ed", type=float, default=0.9998)
    parser.add_argument("--bp-pao", type=float, default=0.98)
    parser.add_argument("--domain-pao", type=float, default=1e-4)
    parser.add_argument("--ed-pao", type=float, default=0.995)
    parser.add_argument("--occupied-weight", type=float, default=1e-4)
    parser.add_argument("--near-pair-distance", type=float, default=3.5)
    parser.add_argument("--mp2-block-memory", type=float, default=256.0)
    parser.add_argument("--full-domain-check", action="store_true")
    parser.add_argument("--scf-conv-tol", type=float, default=1e-10)
    parser.add_argument("--max-memory", type=float, default=12000.0)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument(
        "--reference-npz", default=None, metavar="PATH",
        help=(
            "reuse canonical energies and the full nuclear gradient from a "
            "compatible checkpoint NPZ, skipping canonical DF-MP2 and its "
            "SCF response"
        ),
    )
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args(argv)
    if args.ncarbon < 1:
        parser.error("--ncarbon must be at least one")
    if any(value < 0.0 for value in args.pair_cutoffs):
        parser.error("pair cutoffs must be non-negative")
    if args.frozen_core is None:
        args.frozen_core = args.ncarbon
    if args.output_prefix is None:
        basis_tag = args.basis.lower().replace("-", "")
        args.output_prefix = (
            f"notes/results/iao_mp2_gradient_c{args.ncarbon}_{basis_tag}_fixed"
        )
    return args


def main(argv=None):
    args = parse_args(argv)
    config.update("pyscfad_moleintor_opt", True)
    config.update("pyscfad_scf_implicit_diff", True)
    # Local-domain energies produce a general, nonstationary SCF cotangent.
    # The experimental custom first-order backend differentiates a finite SCF
    # replay and can amplify its residual for such cotangents.  Use the
    # fixed-point implicit response that is validated against displaced-SCF
    # finite differences.
    config.update("pyscfad_scf_first_order_custom", False)

    atoms = make_n_alkane(args.ncarbon)
    mol = build_molecule(
        atoms, basis=args.basis, verbose=args.verbose,
        max_memory=args.max_memory,
    )
    print(
        f"C{args.ncarbon}H{2 * args.ncarbon + 2}, {args.basis}/"
        f"{args.auxbasis}, frozen={args.frozen_core}; "
        f"{mol.natm} atoms, {mol.nao} AOs"
    )
    scratch = tempfile.TemporaryDirectory(prefix="iao_mp2_grad_cderi_")
    cderi_path = Path(scratch.name) / "cderi.h5"
    start = time.perf_counter()
    prepare_outcore_cderi(
        mol, auxbasis=args.auxbasis, path=cderi_path
    )
    cderi_build_seconds = time.perf_counter() - start
    build_mf = make_build_mf(
        auxbasis=args.auxbasis,
        conv_tol=args.scf_conv_tol,
        cderi_path=cderi_path,
    )

    start = time.perf_counter()
    mf, scf_pullback = jax.vjp(build_mf, mol)
    _block(mf.e_tot)
    scf_forward_seconds = time.perf_counter() - start
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ)))
    if args.frozen_core >= nocc:
        raise ValueError(
            f"frozen={args.frozen_core} leaves no active occupied orbitals"
        )
    nmo = int(np.asarray(mf.mo_coeff).shape[1])
    naux = int(mf.with_df.get_naoaux())
    print(
        f"DF-RHF E={float(np.asarray(mf.e_tot)):.12f}; "
        f"nmo={nmo}, nocc={nocc}, naux={naux}; "
        f"CDERI/forward-VJP={cderi_build_seconds:.1f}/"
        f"{scf_forward_seconds:.1f} s"
    )

    e_hf, hf_pullback = jax.vjp(lambda mf_: mf_.e_tot, mf)
    hf_bar, = hf_pullback(jnp.ones((), dtype=jnp.asarray(e_hf).dtype))

    system = {
        "formula": f"C{args.ncarbon}H{2 * args.ncarbon + 2}",
        "ncarbon": args.ncarbon,
        "natom": mol.natm,
        "nao": mol.nao,
        "nmo": nmo,
        "nocc": nocc,
        "nactive_occ": nocc - args.frozen_core,
        "nactive_vir": nmo - nocc,
        "naux": naux,
        "basis": args.basis,
        "auxbasis": args.auxbasis,
        "frozen_core": args.frozen_core,
    }
    if args.reference_npz is None:
        e_ref_corr, ref_corr_bar, ref_forward_seconds, ref_reverse_seconds = (
            _canonical_correlation_value_and_grad(mf, args.frozen_core)
        )
        start = time.perf_counter()
        reference_mol_bar, = scf_pullback(
            _add_cotangents(hf_bar, ref_corr_bar)
        )
        _block(reference_mol_bar)
        ref_scf_pullback_seconds = time.perf_counter() - start
        reference_gradient = np.asarray(reference_mol_bar.coords)
        reference = {
            "e_corr": float(np.asarray(e_ref_corr)),
            "e_total": float(np.asarray(e_hf + e_ref_corr)),
            "provenance": "computed_in_run",
            "reused": False,
            "source_npz": None,
            "source_companion_json": None,
            "compatibility_checks": [],
            "load_seconds": 0.0,
            "correlation_forward_seconds": ref_forward_seconds,
            "correlation_reverse_seconds": ref_reverse_seconds,
            "scf_pullback_seconds": ref_scf_pullback_seconds,
            "gradient_l2_norm": float(np.linalg.norm(reference_gradient)),
            "gradient_max_abs": float(np.max(np.abs(reference_gradient))),
        }
        print(
            f"Canonical DF-MP2 E_corr={reference['e_corr']:.12f}; "
            f"gradient/SCF={ref_forward_seconds + ref_reverse_seconds:.1f}/"
            f"{ref_scf_pullback_seconds:.1f} s"
        )
    else:
        reference_gradient, reference = _load_reference_npz(
            args.reference_npz,
            e_hf=e_hf,
            expected_gradient_shape=(mol.natm, 3),
            expected_system=system,
        )
        print(
            f"Reused canonical DF-MP2 E_corr={reference['e_corr']:.12f} "
            f"from {reference['source_npz']}; load="
            f"{reference['load_seconds']:.1f} s; canonical and reference "
            "SCF pullbacks skipped"
        )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "pyscf_version": pyscf.__version__,
        "system": system,
        "settings": {
            key: value for key, value in vars(args).items()
            if key != "output_prefix"
        },
        "scf": {
            "e_hf": float(np.asarray(e_hf)),
            "outcore_cderi_build_seconds": cderi_build_seconds,
            "forward_vjp_setup_seconds": scf_forward_seconds,
            "conv_tol": args.scf_conv_tol,
            "response_backend": "standard_fixed_point_implicit",
            "first_order_custom": False,
        },
        "runtime": {
            "host": platform.node(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "veclib_maximum_threads": os.environ.get(
                "VECLIB_MAXIMUM_THREADS"
            ),
            "direct_int3c_block_mb": os.environ.get(
                "PYSCFAD_LNO_LOCAL_DIRECT_INT3C_BLOCK_MB"
            ),
            "cderi_bar_aux_block_mb": os.environ.get(
                "PYSCFAD_DF_CDERI_BAR_AUX_BLOCK_MB"
            ),
        },
        "reference": reference,
        "gradient_units": "Eh/bohr",
        "energy_units": "Eh",
        "topology_derivative": "frozen",
    }
    rows = []
    gradients = []

    jobs = [(f"{cutoff:.1e}", cutoff, False) for cutoff in args.pair_cutoffs]
    if args.full_domain_check:
        jobs.append(("full", 0.0, True))

    for label, pair_cutoff, force_full in jobs:
        if force_full:
            thresholds = IAOFragmentMP2Thresholds(
                pair_energy=0.0,
                pao_norm=1e-10,
                domain_pao=0.0,
                ed_pao=0.0,
                occupied_weight=1e-12,
                mp2_block_memory_mb=args.mp2_block_memory,
            )
            pair_model = "all"
        else:
            thresholds = IAOFragmentMP2Thresholds(
                bp_occ=args.bp_occ,
                bp_primary=args.bp_primary,
                bp_ed=args.bp_ed,
                bp_pao=args.bp_pao,
                domain_pao=args.domain_pao,
                ed_pao=args.ed_pao,
                occupied_weight=args.occupied_weight,
                pair_energy=pair_cutoff,
                near_pair_distance=args.near_pair_distance,
                mp2_block_memory_mb=args.mp2_block_memory,
            )
            pair_model = args.pair_model

        row_start = time.perf_counter()
        # Both thresholding stages run on the detached reconstruction made by
        # stop_trace.  Besides defining the derivative boundary, this keeps
        # lazy DF/domain caches from mutating ``mf``, whose exact pytree shape
        # belongs to the saved SCF pullback reused by every table row.
        selection_timing = {}

        def build_fixed_topology(mf_):
            start = time.perf_counter()
            topology = build_iao_fragment_topology(
                mf_,
                frozen=args.frozen_core,
                thresholds=thresholds,
                pair_energy_model=pair_model,
                force_full_domains=force_full,
            )
            selection_timing["topology"] = time.perf_counter() - start
            start = time.perf_counter()
            static_ = build_iao_mp2_static_selections(mf_, topology)
            selection_timing["static"] = time.perf_counter() - start
            return static_

        static = stop_trace(build_fixed_topology)(mf)
        topology_seconds = selection_timing["topology"]
        static_selection_seconds = selection_timing["static"]
        statistics = _domain_statistics(static)

        start = time.perf_counter()
        e_corr, corr_bar = correlation_value_and_grad(mf, static)
        _block((e_corr, corr_bar))
        correlation_gradient_seconds = time.perf_counter() - start
        start = time.perf_counter()
        local_mol_bar, = scf_pullback(_add_cotangents(hf_bar, corr_bar))
        _block(local_mol_bar)
        scf_pullback_seconds = time.perf_counter() - start

        gradient = np.asarray(local_mol_bar.coords)
        e_corr_float = float(np.asarray(e_corr))
        row = {
            "label": label,
            "pair_cutoff": float(pair_cutoff),
            "e_corr": e_corr_float,
            "e_total": float(np.asarray(e_hf)) + e_corr_float,
            "energy_error": e_corr_float - reference["e_corr"],
            **_error_statistics(gradient, reference_gradient),
            **statistics,
            "topology_seconds": topology_seconds,
            "static_selection_seconds": static_selection_seconds,
            "correlation_gradient_seconds": correlation_gradient_seconds,
            "scf_pullback_seconds": scf_pullback_seconds,
            "row_total_seconds": time.perf_counter() - row_start,
        }
        rows.append(row)
        gradients.append(gradient)
        paths = write_outputs(
            args.output_prefix, metadata, rows, gradients, reference_gradient
        )
        _print_row(row)

    print("\nCompleted convergence table (energies in Eh; gradients in Eh/bohr):")
    print(
        "| pair cutoff | strong pairs | mean/max ED AOs | dE | "
        "max abs dG | RMS dG | topology s | gradient s | SCF pullback s |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['label']} | {row['strong_pair_count']}/"
            f"{row['total_pair_count']} | {row['mean_ed_aos']:.1f}/"
            f"{row['max_ed_aos']} | {row['energy_error']:+.3e} | "
            f"{row['gradient_max_abs_error']:.3e} | "
            f"{row['gradient_rms_error']:.3e} | "
            f"{row['topology_seconds'] + row['static_selection_seconds']:.1f} | "
            f"{row['correlation_gradient_seconds']:.1f} | "
            f"{row['scf_pullback_seconds']:.1f} |"
        )
    print("\nWrote " + ", ".join(str(path) for path in paths))
    scratch.cleanup()
    return rows


if __name__ == "__main__":
    main()
