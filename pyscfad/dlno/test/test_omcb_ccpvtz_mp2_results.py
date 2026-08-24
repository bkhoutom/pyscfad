import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "notes/results/summarize_omcb_ccpvtz_mp2.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "summarize_omcb_ccpvtz_mp2", _SCRIPT
)
summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary)


def _write_checkpoint(prefix, *, basis="cc-pvtz", frozen=None):
    reference_gradient = np.arange(108.0).reshape(36, 3) * 1e-6
    local_gradient = reference_gradient + 1e-5
    local_gradient[0, 0] += 2e-5
    error = local_gradient - reference_gradient
    error_l2 = float(np.linalg.norm(error))
    reference = {
        "method": "canonical DF-MP2",
        "e_hf": -462.0,
        "e_corr": -2.0,
        "e_total": -464.0,
        "gradient_l2_norm": float(np.linalg.norm(reference_gradient)),
        "gradient_max_abs": float(np.max(np.abs(reference_gradient))),
    }
    row = {
        "pair_threshold": 3e-2,
        "e_corr": -1.9998,
        "e_total": -463.9998,
        "energy_error_eh": 2e-4,
        "energy_error_millihartree": 0.2,
        "gradient_rms_error": float(np.sqrt(np.mean(error * error))),
        "gradient_max_abs_error": float(np.max(np.abs(error))),
        "gradient_l2_error": error_l2,
        "gradient_relative_l2_error": (
            error_l2 / float(np.linalg.norm(reference_gradient))
        ),
        "strong_pair_count": 60,
        "weak_pair_count": 6,
        "total_pair_count": 66,
        "nfragment": 12,
        "mean_ed_atoms": 30.0,
        "max_ed_atoms": 36,
        "mean_ed_aos": 600.0,
        "max_ed_aos": 696,
        "topology_statistics_seconds": 2.0,
        "mpi_energy_gradient_seconds": 120.0,
    }
    metadata = {
        "system": {
            "name": "OMCB",
            "formula": "C12H24",
            "natom": 36,
            "nao": 696,
            "nocc": 48,
            "nactive_occ": 48,
            "nactive_vir": 648,
            "naux": 1668,
            "basis": basis,
            "geometry_sha256": summary._EXPECTED_GEOMETRY_SHA256,
            "auxbasis": None,
            "auxbasis_requested": None,
            "auxbasis_resolved": {
                "C": "cc-pvtz-jkfit",
                "H": "cc-pvtz-jkfit",
            },
            "frozen": frozen,
        },
        "settings": {
            "pair_thresholds": list(summary._EXPECTED_PAIR_THRESHOLDS),
            "pair_energy_model": "multipole",
            "topology_derivative": "fixed",
            "mpi_ranks": 2,
        },
        "scf": {
            "response_backend": "standard_fixed_point_implicit",
            "first_order_custom": False,
        },
        "reference": reference,
        "energy_units": "Eh",
        "gradient_units": "Eh/bohr",
        "rows": [row],
    }
    prefix.with_suffix(".json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    with prefix.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary.CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        pair_thresholds=np.asarray([row["pair_threshold"]]),
        total_energies=np.asarray([row["e_total"]]),
        correlation_energies=np.asarray([row["e_corr"]]),
        gradients=local_gradient[None],
        reference_gradient=reference_gradient,
        reference_total_energy=np.asarray(reference["e_total"]),
        reference_correlation_energy=np.asarray(reference["e_corr"]),
    )
    return metadata


def test_exact_checkpoint_generates_table_and_plot(tmp_path):
    prefix = tmp_path / "exact"
    _write_checkpoint(prefix)
    loaded, rows = summary.load_checkpoint(prefix)
    assert loaded["system"]["auxbasis_resolved"]["C"] == "cc-pvtz-jkfit"
    assert rows[0]["pair_threshold"] == 3e-2

    figure = summary.write_figure(tmp_path / "result.png", rows)
    table = summary.write_tex_table(tmp_path / "result.tex", loaded, rows)
    interpretation = summary.write_interpretation(
        tmp_path / "interpretation.json", loaded, rows
    )
    assert figure.stat().st_size > 1000
    text = table.read_text(encoding="utf-8")
    assert "canonical DF--MP2" in text
    assert "3\\times10^{-2}" in text
    assert "+0.200000" in text
    assert "stress" in text
    analysis = json.loads(interpretation.read_text(encoding="utf-8"))
    assert analysis["parameter_selection_excluded_thresholds"] == [3e-2]
    assert analysis["convergence_thresholds"] == []
    assert analysis["preflight_stress_observation"]["strong_pair_count"] == 57
    assert analysis["timing_summary"]["recorded_total_seconds"] == 122.0
    assert analysis["gradient_translation_checks"]["rows"][0][
        "pair_threshold"
    ] == 3e-2


@pytest.mark.parametrize(
    "basis,frozen,message",
    [
        ("sto-3g", None, "system.basis"),
        ("cc-pvtz", 12, "system.frozen"),
    ],
)
def test_rejects_surrogate_or_frozen_checkpoint(
    tmp_path, basis, frozen, message
):
    prefix = tmp_path / "wrong"
    _write_checkpoint(prefix, basis=basis, frozen=frozen)
    with pytest.raises(ValueError, match=message):
        summary.load_checkpoint(prefix)


def test_rejects_cross_file_energy_mismatch(tmp_path):
    prefix = tmp_path / "mismatch"
    _write_checkpoint(prefix)
    with np.load(prefix.with_suffix(".npz")) as arrays:
        payload = {name: arrays[name] for name in arrays.files}
    payload["total_energies"] = np.asarray([-999.0])
    np.savez_compressed(prefix.with_suffix(".npz"), **payload)
    with pytest.raises(ValueError, match="total energies disagree"):
        summary.load_checkpoint(prefix)


def test_rejects_redundant_all_strong_cutoff_grid(tmp_path):
    prefix = tmp_path / "redundant_grid"
    _write_checkpoint(prefix)
    json_path = prefix.with_suffix(".json")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    metadata["settings"]["pair_thresholds"] = [1e-3, 3e-4, 1e-4]
    json_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="only one all-strong endpoint"):
        summary.load_checkpoint(prefix)
