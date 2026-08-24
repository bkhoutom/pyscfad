"""Validate and summarize the exact example-13 OMCB/cc-pVTZ MP2 sweep.

The input is the checkpoint prefix written by
``examples/lno/20-mpi_omcb_iao_mp2_gradient_sweep.py``.  All three companion
files (CSV, JSON, and NPZ) are required and cross-checked before a table or
figure is written.  In particular, this script refuses the older STO-3G
pilot: the accepted calculation is OMCB/cc-pVTZ, PySCF's automatically chosen
cc-pVTZ-JKFIT basis, no frozen orbitals, and a multi-rank fixed-gauge MPI run.

From the repository root::

    .venv/bin/python notes/results/summarize_omcb_ccpvtz_mp2.py

The default outputs are a LaTeX table fragment in ``notes/results`` and a
two-panel convergence plot in ``notes/figures``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


DEFAULT_PREFIX = Path(
    "notes/results/omcb_ccpvtz_iao_mp2_mpi_pair_sweep"
)
DEFAULT_FIGURE = Path(
    "notes/figures/omcb_ccpvtz_iao_mp2_mpi_pair_convergence.png"
)
DEFAULT_TEX = DEFAULT_PREFIX.with_name(DEFAULT_PREFIX.name + "_table.tex")
DEFAULT_INTERPRETATION = DEFAULT_PREFIX.with_name(
    DEFAULT_PREFIX.name + "_interpretation.json"
)

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

_ROW_FLOAT_FIELDS = CSV_FIELDS[:9] + CSV_FIELDS[13:]
_ROW_INT_FIELDS = CSV_FIELDS[9:13]
_EXPECTED_SYSTEM = {
    "name": "OMCB",
    "formula": "C12H24",
    "natom": 36,
    "nao": 696,
    "nocc": 48,
    "nactive_occ": 48,
    "nactive_vir": 648,
    "naux": 1668,
}
_EXPECTED_GEOMETRY_SHA256 = (
    "950b357e267d4872d8ad5da0f8f4e5af5e6165eb6f28e2ec695fcea52783a837"
)
_EXPECTED_PAIR_THRESHOLDS = (3e-2, 1e-2, 1e-3)
_STRESS_THRESHOLD = 3e-2
_STRESS_PREFLIGHT = {
    "strong_pair_count": 57,
    "weak_pair_count": 9,
    "energy_error_millihartree": -60.151708345,
}
def _prefix_from_path(path):
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() in {".csv", ".json", ".npz"}:
        path = path.with_suffix("")
    return path


def _basis_key(value):
    return "".join(character for character in str(value).lower()
                   if character.isalnum())


def _resolved_auxbasis_names(system):
    value = system.get("auxbasis_resolved")
    if value is None:
        value = system.get("resolved_auxbasis")
    if value is None:
        # A resolved string in ``auxbasis`` is accepted for backward
        # compatibility, but ``None``/``auto`` is not sufficient provenance.
        value = system.get("auxbasis")
    if isinstance(value, dict):
        values = value.values()
    elif isinstance(value, str):
        values = (value,)
    else:
        return set()
    return {_basis_key(item) for item in values}


def _finite_float(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _integer(value, label):
    result = _finite_float(value, label)
    if not result.is_integer():
        raise ValueError(f"{label} must be an integer")
    return int(result)


def _close(left, right, *, atol=1e-12, rtol=1e-12):
    return bool(np.isclose(float(left), float(right), atol=atol, rtol=rtol))


def _tex_scientific(value, digits=3):
    value = float(value)
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    coefficient = value / (10.0 ** exponent)
    text = f"{coefficient:.{digits}f}".rstrip("0").rstrip(".")
    if text == "1":
        return f"10^{{{exponent}}}"
    if text == "-1":
        return f"-10^{{{exponent}}}"
    return f"{text}\\times10^{{{exponent}}}"


def _validate_metadata(metadata):
    system = metadata.get("system", {})
    for field, expected in _EXPECTED_SYSTEM.items():
        actual = system.get(field)
        if isinstance(expected, int):
            actual = _integer(actual, f"system.{field}")
        if actual != expected:
            raise ValueError(
                f"system.{field} is {actual!r}; expected {expected!r} for "
                "the exact example-13 OMCB calculation"
            )
    if _basis_key(system.get("basis")) != "ccpvtz":
        raise ValueError("system.basis must be cc-pVTZ")
    if system.get("geometry_sha256") != _EXPECTED_GEOMETRY_SHA256:
        raise ValueError(
            "system.geometry_sha256 does not identify examples/lno/OMCB.xyz"
        )
    if system.get("frozen", "missing") is not None:
        raise ValueError("system.frozen must be null (frozen=None)")

    requested = system.get("auxbasis_requested", system.get("auxbasis"))
    if requested not in (None, "auto"):
        raise ValueError("the auxiliary basis must be requested automatically")
    resolved = _resolved_auxbasis_names(system)
    if resolved != {"ccpvtzjkfit"}:
        raise ValueError(
            "system metadata must resolve the automatic auxiliary basis to "
            "cc-pVTZ-JKFIT for every element"
        )

    settings = metadata.get("settings", {})
    if settings.get("pair_energy_model") != "multipole":
        raise ValueError("settings.pair_energy_model must be 'multipole'")
    if settings.get("topology_derivative") != "fixed":
        raise ValueError("settings.topology_derivative must be 'fixed'")
    if _integer(settings.get("mpi_ranks"), "settings.mpi_ranks") < 2:
        raise ValueError("settings.mpi_ranks must be at least two")

    scf = metadata.get("scf", {})
    if (
        scf.get("response_backend") != "standard_fixed_point_implicit"
        or scf.get("first_order_custom") is not False
    ):
        raise ValueError(
            "checkpoint lacks the validated standard fixed-point implicit "
            "SCF-gradient provenance"
        )

    if metadata.get("energy_units") != "Eh":
        raise ValueError("energy_units must be 'Eh'")
    if metadata.get("gradient_units") != "Eh/bohr":
        raise ValueError("gradient_units must be 'Eh/bohr'")

    reference = metadata.get("reference", {})
    if reference.get("method") != "canonical DF-MP2":
        raise ValueError("reference.method must be 'canonical DF-MP2'")
    for field in ("e_hf", "e_corr", "e_total"):
        _finite_float(reference.get(field), f"reference.{field}")
    if not _close(
        reference["e_hf"] + reference["e_corr"], reference["e_total"]
    ):
        raise ValueError("reference total is inconsistent with HF + MP2")
    return system, settings, reference


def _validate_rows(rows, reference):
    parsed = []
    for index, row in enumerate(rows):
        label = f"row {index + 1}"
        missing = [field for field in CSV_FIELDS if field not in row]
        if missing:
            raise ValueError(f"{label} is missing {', '.join(missing)}")
        item = {
            field: _finite_float(row[field], f"{label}.{field}")
            for field in _ROW_FLOAT_FIELDS
        }
        item.update({
            field: _integer(row[field], f"{label}.{field}")
            for field in _ROW_INT_FIELDS
        })
        if item["pair_threshold"] <= 0.0:
            raise ValueError(f"{label}.pair_threshold must be positive")
        if item["nfragment"] != 12 or item["total_pair_count"] != 66:
            raise ValueError(f"{label} must contain 12 fragments and 66 pairs")
        if (
            item["strong_pair_count"] + item["weak_pair_count"]
            != item["total_pair_count"]
        ):
            raise ValueError(f"{label} strong/weak pair counts are inconsistent")
        if not 0 <= item["strong_pair_count"] <= item["total_pair_count"]:
            raise ValueError(f"{label}.strong_pair_count is out of range")
        if item["gradient_rms_error"] < 0.0:
            raise ValueError(f"{label}.gradient_rms_error must be non-negative")
        if item["gradient_max_abs_error"] < item["gradient_rms_error"]:
            raise ValueError(f"{label} has RMS gradient error above its maximum")
        expected_error = item["e_total"] - float(reference["e_total"])
        if not _close(item["energy_error_eh"], expected_error):
            raise ValueError(f"{label}.energy_error_eh is inconsistent")
        if not _close(
            item["energy_error_millihartree"],
            1000.0 * item["energy_error_eh"],
            atol=1e-8,
        ):
            raise ValueError(
                f"{label}.energy_error_millihartree is inconsistent"
            )
        parsed.append(item)

    parsed.sort(key=lambda row: row["pair_threshold"], reverse=True)
    cutoffs = [row["pair_threshold"] for row in parsed]
    if len(cutoffs) != len(set(cutoffs)):
        raise ValueError("rows contain duplicate pair thresholds")
    strong = [row["strong_pair_count"] for row in parsed]
    if any(right < left for left, right in zip(strong, strong[1:])):
        raise ValueError("strong-pair count decreases as the cutoff is tightened")
    return parsed


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(f"{path} has an unexpected CSV schema")
        return list(reader)


def _crosscheck_csv(json_rows, csv_rows):
    if len(json_rows) != len(csv_rows):
        raise ValueError("CSV and JSON contain different row counts")
    csv_by_cutoff = {float(row["pair_threshold"]): row for row in csv_rows}
    for row in json_rows:
        cutoff = row["pair_threshold"]
        if cutoff not in csv_by_cutoff:
            raise ValueError("CSV and JSON pair-threshold grids differ")
        csv_row = csv_by_cutoff[cutoff]
        for field in CSV_FIELDS:
            if not _close(row[field], csv_row[field], atol=1e-9):
                raise ValueError(f"CSV and JSON disagree in {field}")


def _crosscheck_npz(path, rows, system, reference):
    required = {
        "pair_thresholds", "total_energies", "correlation_energies",
        "gradients", "reference_gradient", "reference_total_energy",
        "reference_correlation_energy",
    }
    with np.load(path) as arrays:
        missing = required - set(arrays.files)
        if missing:
            raise ValueError(f"{path} is missing {', '.join(sorted(missing))}")
        nrow = len(rows)
        if arrays["pair_thresholds"].shape != (nrow,):
            raise ValueError("NPZ pair_thresholds has the wrong shape")
        if arrays["total_energies"].shape != (nrow,):
            raise ValueError("NPZ total_energies has the wrong shape")
        if arrays["correlation_energies"].shape != (nrow,):
            raise ValueError("NPZ correlation_energies has the wrong shape")
        expected_gradient_shape = (nrow, system["natom"], 3)
        if arrays["gradients"].shape != expected_gradient_shape:
            raise ValueError("NPZ gradients has the wrong shape")
        if arrays["reference_gradient"].shape != (system["natom"], 3):
            raise ValueError("NPZ reference_gradient has the wrong shape")

        # The writer preserves run order, while rows above are sorted for the
        # presentation.  Compare after sorting the NPZ grid the same way.
        order = np.argsort(-np.asarray(arrays["pair_thresholds"]))
        comparisons = (
            (
                "pair thresholds",
                np.asarray(arrays["pair_thresholds"])[order],
                [row["pair_threshold"] for row in rows],
                0.0,
            ),
            (
                "total energies",
                np.asarray(arrays["total_energies"])[order],
                [row["e_total"] for row in rows],
                1e-12,
            ),
            (
                "correlation energies",
                np.asarray(arrays["correlation_energies"])[order],
                [row["e_corr"] for row in rows],
                1e-12,
            ),
        )
        for label, actual, expected, tolerance in comparisons:
            if not np.allclose(
                actual, expected, rtol=tolerance, atol=tolerance
            ):
                raise ValueError(f"NPZ and JSON {label} disagree")
        if not _close(arrays["reference_total_energy"], reference["e_total"]):
            raise ValueError("NPZ and JSON reference total energies disagree")
        if not _close(
            arrays["reference_correlation_energy"], reference["e_corr"]
        ):
            raise ValueError("NPZ and JSON reference MP2 energies disagree")
        gradient = np.asarray(arrays["reference_gradient"])
        if "gradient_l2_norm" in reference and not _close(
            np.linalg.norm(gradient), reference["gradient_l2_norm"], atol=1e-9
        ):
            raise ValueError("NPZ and JSON reference-gradient norms disagree")
        if "gradient_max_abs" in reference and not _close(
            np.max(np.abs(gradient)), reference["gradient_max_abs"], atol=1e-9
        ):
            raise ValueError("NPZ and JSON reference-gradient maxima disagree")
        gradient_ordered = np.asarray(arrays["gradients"])[order]
        reference_l2 = float(np.linalg.norm(gradient))
        row_diagnostics = []
        for row, local_gradient in zip(rows, gradient_ordered, strict=True):
            error = local_gradient - gradient
            metrics = {
                "gradient_rms_error": float(np.sqrt(np.mean(error * error))),
                "gradient_max_abs_error": float(np.max(np.abs(error))),
                "gradient_l2_error": float(np.linalg.norm(error)),
            }
            metrics["gradient_relative_l2_error"] = (
                metrics["gradient_l2_error"] / reference_l2
                if reference_l2 else float("nan")
            )
            for field, value in metrics.items():
                if not _close(row[field], value, atol=1e-12, rtol=1e-10):
                    raise ValueError(
                        f"NPZ gradient and JSON {field} disagree at "
                        f"pair threshold {row['pair_threshold']:.1e}"
                    )
            net_force = np.sum(local_gradient, axis=0)
            row_diagnostics.append({
                "pair_threshold": row["pair_threshold"],
                "net_force_l2_norm": float(np.linalg.norm(net_force)),
                "net_force_max_abs": float(np.max(np.abs(net_force))),
            })
        reference_net_force = np.sum(gradient, axis=0)
        return {
            "reference_net_force_l2_norm": float(
                np.linalg.norm(reference_net_force)
            ),
            "reference_net_force_max_abs": float(
                np.max(np.abs(reference_net_force))
            ),
            "rows": row_diagnostics,
        }


def load_checkpoint(path=DEFAULT_PREFIX):
    """Load and cross-validate an exact OMCB/cc-pVTZ checkpoint triplet."""
    prefix = _prefix_from_path(path)
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    npz_path = prefix.with_suffix(".npz")
    for companion in (csv_path, json_path, npz_path):
        if not companion.exists():
            raise FileNotFoundError(f"missing checkpoint companion {companion}")

    with json_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    system, settings, reference = _validate_metadata(metadata)
    rows = _validate_rows(metadata.get("rows", []), reference)
    csv_rows = _read_csv(csv_path)
    _crosscheck_csv(rows, csv_rows)
    diagnostics = _crosscheck_npz(npz_path, rows, system, reference)

    requested = [float(value) for value in settings.get("pair_thresholds", [])]
    if len(requested) != len(_EXPECTED_PAIR_THRESHOLDS) or not np.allclose(
        requested, _EXPECTED_PAIR_THRESHOLDS, rtol=0.0, atol=0.0
    ):
        raise ValueError(
            "settings.pair_thresholds must be [3e-2, 1e-2, 1e-3]; this "
            "grid has two loose selection-changing rows and only one "
            "all-strong endpoint"
        )
    completed = {row["pair_threshold"] for row in rows}
    if not completed.issubset(set(requested)):
        raise ValueError("completed rows are not a subset of requested thresholds")
    metadata = dict(metadata)
    metadata["_validation_diagnostics"] = diagnostics
    return metadata, rows


def write_tex_table(path, metadata, rows):
    """Write a compact, input-ready LaTeX reference/comparison table."""
    if not rows:
        raise ValueError("at least one completed local-MP2 row is required")
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    reference = metadata["reference"]
    stress_row = next(
        (row for row in rows if row["pair_threshold"] == _STRESS_THRESHOLD),
        None,
    )
    lines = [
        "% Generated by summarize_omcb_ccpvtz_mp2.py; do not edit.",
        "The all-electron canonical DF--MP2 reference has correlation energy",
        rf"${float(reference['e_corr']):.12f}\,E_h$ and total energy",
        rf"${float(reference['e_total']):.12f}\,E_h$.",
    ]
    if stress_row is not None:
        lines.extend([
            "The $3\\times10^{-2}\\,E_h$ row is a cutoff-sensitive stress",
            "point.  An independent energy preflight selected 57/66 strong",
            "pairs and gave $\\Delta E=-60.151708$~m$E_h$, whereas the",
            "production gradient run selected",
            f"{stress_row['strong_pair_count']}/66 and gave "
            f"$\\Delta E={stress_row['energy_error_millihartree']:+.6f}$~m$E_h$.",
            "It is excluded from parameter selection and is not assumed to",
            "lie on a smooth convergence curve.",
        ])
    lines.extend([
        "\\begin{table}[htbp]",
        " \\centering",
        " \\caption{MPI IAO--local-MP2 convergence for the original",
        " example-13 OMCB/cc-pVTZ calculation.  The automatic auxiliary",
        " basis is cc-pVTZ-JKFIT and no orbitals are frozen.  Energy errors",
        " are in m$E_h$, gradient errors in $E_h/a_0$, and wall times in",
        " minutes.  The stress row is reported but excluded from parameter",
        " selection.}",
        " \\label{tab:omcb-ccpvtz-mp2-convergence}",
        " \\begin{tabular}{@{}rlcrrrr@{}}",
        "  \\toprule",
        "  $\\tau_{\\mathrm{pair}}$ & role & strong/weak & "
        "$\\Delta E$ (m$E_h$) & $\\Delta g_{\\mathrm{RMS}}$ & "
        "$\\Delta g_{\\max}$ & wall (min) \\\\",
        "  \\midrule",
    ])
    for row in rows:
        role = (
            "stress"
            if row["pair_threshold"] == _STRESS_THRESHOLD
            else "convergence"
        )
        energy_error = row["energy_error_millihartree"]
        energy_text = (
            _tex_scientific(energy_error)
            if abs(energy_error) < 1e-3
            else f"{energy_error:+.6f}"
        )
        lines.append(
            f"  ${_tex_scientific(row['pair_threshold'], digits=1)}$ & "
            f"{role} & "
            f"{row['strong_pair_count']}/{row['weak_pair_count']} & "
            f"${energy_text}$ & "
            f"${_tex_scientific(row['gradient_rms_error'])}$ & "
            f"${_tex_scientific(row['gradient_max_abs_error'])}$ & "
            f"{row['mpi_energy_gradient_seconds'] / 60.0:.2f} \\\\"
        )
    lines.extend([
        "  \\bottomrule",
        " \\end{tabular}",
        "\\end{table}",
    ])
    convergence_rows = [
        row for row in rows if row["pair_threshold"] != _STRESS_THRESHOLD
    ]
    if convergence_rows:
        loose = max(convergence_rows, key=lambda row: row["pair_threshold"])
        tight = min(convergence_rows, key=lambda row: row["pair_threshold"])
        lines.extend([
            f"At $\\tau_{{\\mathrm{{pair}}}}="
            f"{_tex_scientific(loose['pair_threshold'], digits=1)}\\,E_h$, "
            f"the energy error is {abs(loose['energy_error_millihartree']):.6f}~m$E_h$",
            "and the RMS gradient error is $"
            f"{_tex_scientific(loose['gradient_rms_error'])}\\,E_h/a_0$.",
            "The tightest sampled row, $"
            f"{_tex_scientific(tight['pair_threshold'], digits=1)}\\,E_h$, is the only",
            "non-stress row below 1~m$E_h$; it selects all 66 pairs as strong",
            f"and agrees with canonical DF--MP2 to "
            f"${_tex_scientific(abs(tight['energy_error_eh']))}\\,E_h$ in energy and",
            f"${_tex_scientific(tight['gradient_rms_error'])}\\,E_h/a_0$ "
            "in the RMS gradient.",
            "Thus the sub-m$E_h$ endpoint is numerically exact here, but it",
            "does not retain pair locality for this compact cage.",
        ])

    reference_seconds = float(metadata.get("cderi_build_seconds", 0.0)) + sum(
        float(value) for value in reference.get("timing", {}).values()
    )
    topology_seconds = sum(row["topology_statistics_seconds"] for row in rows)
    local_seconds = sum(row["mpi_energy_gradient_seconds"] for row in rows)
    recorded_seconds = reference_seconds + topology_seconds + local_seconds
    diagnostics = metadata.get("_validation_diagnostics", {})
    force_norms = [diagnostics.get("reference_net_force_l2_norm", 0.0)]
    force_norms.extend(
        row["net_force_l2_norm"] for row in diagnostics.get("rows", [])
    )
    lines.extend([
        f"The recorded reference stage took {reference_seconds / 60.0:.2f}~min;",
        "the three MPI local energy/gradient rows took "
        + ", ".join(
            f"{row['mpi_energy_gradient_seconds'] / 60.0:.2f}"
            for row in rows
        )
        + "~min, respectively.",
        f"Including topology construction, the recorded components total "
        f"{recorded_seconds / 60.0:.2f}~min.",
        "The largest reference/local net-force norm is $"
        f"{_tex_scientific(max(force_norms))}\\,E_h/a_0$.",
        "",
    ])
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    return path


def write_interpretation(path, metadata, rows):
    """Write analysis metadata without altering the raw production files."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    stress_row = next(
        (row for row in rows if row["pair_threshold"] == _STRESS_THRESHOLD),
        None,
    )
    payload = {
        "source_created_utc": metadata.get("created_utc"),
        "source_geometry_sha256": metadata["system"]["geometry_sha256"],
        "cutoff_sensitive_stress_thresholds": [_STRESS_THRESHOLD],
        "parameter_selection_excluded_thresholds": [_STRESS_THRESHOLD],
        "preflight_stress_observation": {
            "pair_threshold": _STRESS_THRESHOLD,
            **_STRESS_PREFLIGHT,
        },
        "production_stress_observation": stress_row,
        "interpretation": (
            "The 3e-2 row is a near-threshold branch-sensitivity stress "
            "point. It is plotted without a connecting line and is not used "
            "for parameter selection. No monotonicity of energy or gradient "
            "errors is assumed across that row."
        ),
        "convergence_thresholds": [
            row["pair_threshold"] for row in rows
            if row["pair_threshold"] != _STRESS_THRESHOLD
        ],
        "timing_summary": {
            "reference_seconds": (
                float(metadata.get("cderi_build_seconds", 0.0))
                + sum(
                    float(value)
                    for value in metadata["reference"].get("timing", {}).values()
                )
            ),
            "topology_statistics_seconds": sum(
                row["topology_statistics_seconds"] for row in rows
            ),
            "local_energy_gradient_seconds": sum(
                row["mpi_energy_gradient_seconds"] for row in rows
            ),
        },
        "gradient_translation_checks": metadata.get(
            "_validation_diagnostics", {}
        ),
    }
    timing = payload["timing_summary"]
    timing["recorded_total_seconds"] = sum(timing.values())
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def write_figure(path, rows):
    """Plot energy and gradient errors against the strong-pair threshold."""
    if not rows:
        raise ValueError("at least one completed local-MP2 row is required")
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pyscfad-mpl-cache")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    cutoff = np.asarray([row["pair_threshold"] for row in rows])
    energy = np.abs(np.asarray([
        row["energy_error_millihartree"] for row in rows
    ]))
    rms = np.asarray([row["gradient_rms_error"] for row in rows])
    maximum = np.asarray([row["gradient_max_abs_error"] for row in rows])

    with plt.rc_context({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "savefig.dpi": 320,
    }):
        figure, axes = plt.subplots(
            1, 2, figsize=(7.2, 3.0), constrained_layout=True
        )
        stress = np.isclose(
            cutoff, _STRESS_THRESHOLD, rtol=0.0, atol=0.0
        )
        convergence = ~stress
        if np.any(convergence):
            axes[0].plot(
                cutoff[convergence], energy[convergence], "o-",
                color="#1769aa", label="convergence rows",
            )
        if np.any(stress):
            axes[0].plot(
                cutoff[stress], energy[stress], linestyle="none", marker="D",
                markerfacecolor="white", markeredgecolor="#1769aa",
                markeredgewidth=1.4, label=r"$3\times10^{-2}$ stress point",
            )
        # A full-strong-pair row can agree with canonical MP2 to roundoff and
        # have exactly zero energy error; symlog retains that datum honestly.
        positive = energy[energy > 0.0]
        linear_width = max(
            (float(positive.min()) * 0.1 if positive.size else 1e-12), 1e-12
        )
        axes[0].set_yscale("symlog", linthresh=linear_width)
        axes[0].set_ylabel(r"$|\Delta E|$ (m$E_h$)")
        axes[0].set_title("Total-energy error")
        axes[0].legend(frameon=False)

        if np.any(convergence):
            axes[1].plot(
                cutoff[convergence], rms[convergence], "o-",
                color="#2e7d32", label=r"$\Delta g_{\mathrm{RMS}}$",
            )
            axes[1].plot(
                cutoff[convergence], maximum[convergence], "s--",
                color="#c62828", label=r"$\Delta g_{\max}$",
            )
        if np.any(stress):
            axes[1].plot(
                cutoff[stress], rms[stress], linestyle="none", marker="D",
                markerfacecolor="white", markeredgecolor="#2e7d32",
                markeredgewidth=1.4,
                label=r"$\Delta g_{\mathrm{RMS}}$ (stress)",
            )
            axes[1].plot(
                cutoff[stress], maximum[stress], linestyle="none", marker="D",
                markerfacecolor="white", markeredgecolor="#c62828",
                markeredgewidth=1.4,
                label=r"$\Delta g_{\max}$ (stress)",
            )
        gradient_positive = np.concatenate((rms[rms > 0.0], maximum[maximum > 0.0]))
        gradient_linear_width = max(
            (
                float(gradient_positive.min()) * 0.1
                if gradient_positive.size else 1e-15
            ),
            1e-15,
        )
        axes[1].set_yscale("symlog", linthresh=gradient_linear_width)
        axes[1].set_ylabel(r"gradient error ($E_h/a_0$)")
        axes[1].set_title("Nuclear-gradient error")
        axes[1].legend(frameon=False)

        for axis in axes:
            axis.set_xscale("log")
            axis.invert_xaxis()
            axis.set_xlabel(r"pair threshold $\tau_{\mathrm{pair}}$ ($E_h$)")
            axis.grid(True, which="major", color="0.86", linewidth=0.7)
            axis.grid(True, which="minor", color="0.93", linewidth=0.45)
            axis.tick_params(direction="in", top=True, right=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        figure.savefig(temporary, format=path.suffix.lstrip("."),
                       bbox_inches="tight")
        plt.close(figure)
        temporary.replace(path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", default=str(DEFAULT_PREFIX))
    parser.add_argument("--figure", default=str(DEFAULT_FIGURE))
    parser.add_argument("--tex-output", default=str(DEFAULT_TEX))
    parser.add_argument(
        "--interpretation-json", default=str(DEFAULT_INTERPRETATION)
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    try:
        metadata, rows = load_checkpoint(args.checkpoint)
        if not args.validate_only:
            figure = write_figure(args.figure, rows)
            table = write_tex_table(args.tex_output, metadata, rows)
            interpretation = write_interpretation(
                args.interpretation_json, metadata, rows
            )
    except (FileNotFoundError, OSError, ValueError, AssertionError) as error:
        parser.error(str(error))

    system = metadata["system"]
    print(
        f"validated OMCB/{system['basis']} with {system['naux']} auxiliary "
        f"functions and {len(rows)} completed threshold(s)"
    )
    if not args.validate_only:
        print(f"figure: {figure}")
        print(f"table: {table}")
        print(f"interpretation: {interpretation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
