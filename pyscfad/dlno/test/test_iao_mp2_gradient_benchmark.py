import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples/lno/17-iao_fragment_mp2_gradient_convergence.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "iao_fragment_mp2_gradient_convergence", _SCRIPT
)
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def test_production_defaults_and_small_alkane_formula():
    args = benchmark.parse_args([])
    assert args.ncarbon == 25
    assert args.basis == "cc-pvdz"
    assert args.auxbasis == "cc-pvdz-ri"
    assert args.frozen_core == 25
    assert args.reference_npz is None
    assert args.output_prefix.endswith("_fixed")
    assert not hasattr(args, "local_lov_mode")
    assert not hasattr(args, "local_df_cache_size")

    reused = benchmark.parse_args(["--reference-npz", "reference.npz"])
    assert reused.reference_npz == "reference.npz"

    atoms = benchmark.make_n_alkane(4)
    symbols = [symbol for symbol, _ in atoms]
    assert symbols.count("C") == 4
    assert symbols.count("H") == 10


def test_domain_statistics_use_final_fixed_ranks():
    fragments = []
    for natom, nao, nocc, nvir in ((3, 8, 2, 5), (5, 13, 4, 8)):
        fragments.append(SimpleNamespace(
            extended_atoms=np.arange(natom),
            extended_ao_indices=np.arange(nao),
            strong_occ_metric_keep=np.arange(nocc),
            strong_virtual=SimpleNamespace(metric_keep=np.arange(nvir)),
        ))
    static = SimpleNamespace(
        fragments=tuple(fragments),
        strong_mask=np.asarray([[True, False], [False, True]]),
    )
    values = benchmark._domain_statistics(static)
    assert values["strong_pair_count"] == 0
    assert values["weak_pair_count"] == 1
    assert values["mean_ed_aos"] == 10.5
    assert values["max_ed_occ"] == 4
    assert values["max_ed_vir"] == 8


def test_checkpoint_outputs_round_trip(tmp_path):
    reference_gradient = np.arange(12.0).reshape(4, 3)
    metadata = {
        "reference": {"e_corr": -0.2, "e_total": -10.2},
        "system": {"formula": "test"},
        "scf": {
            "response_backend": "standard_fixed_point_implicit",
            "first_order_custom": False,
        },
    }
    row = {field: 0.0 for field in benchmark.CSV_FIELDS}
    row.update({
        "label": "1.0e-04",
        "pair_cutoff": 1e-4,
        "e_corr": -0.19,
        "e_total": -10.19,
    })
    prefix = tmp_path / "convergence"
    paths = benchmark.write_outputs(
        prefix, metadata, [row], [reference_gradient + 1.0],
        reference_gradient,
    )
    assert all(path.exists() for path in paths)
    with prefix.with_suffix(".json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["rows"][0]["pair_cutoff"] == 1e-4
    arrays = np.load(prefix.with_suffix(".npz"))
    assert arrays["gradients"].shape == (1, 4, 3)
    assert arrays["system_formula"].item() == "test"
    assert arrays["scf_response_backend"].item() == (
        "standard_fixed_point_implicit"
    )
    assert not arrays["scf_first_order_custom"].item()
    np.testing.assert_array_equal(
        arrays["reference_gradient"], reference_gradient
    )


def test_load_reference_npz_validates_and_records_reuse(tmp_path):
    path = tmp_path / "reference.npz"
    gradient = np.arange(12.0).reshape(4, 3) * 1e-3
    np.savez_compressed(
        path,
        reference_gradient=gradient,
        reference_correlation_energy=np.asarray(-0.2),
        reference_total_energy=np.asarray(-10.2),
        system_formula=np.asarray("C1H4"),
        system_natom=np.asarray(4),
        system_nao=np.asarray(9),
        system_basis=np.asarray("sto-3g"),
        system_frozen_core=np.asarray(1),
    )
    path.with_suffix(".json").write_text(json.dumps({
        "scf": {
            "response_backend": "standard_fixed_point_implicit",
            "first_order_custom": False,
        }
    }))
    loaded_gradient, reference = benchmark._load_reference_npz(
        path,
        e_hf=-10.0,
        expected_gradient_shape=(4, 3),
        expected_system={
            "formula": "C1H4",
            "natom": 4,
            "nao": 9,
            "basis": "STO-3G",
            "frozen_core": 1,
        },
    )
    np.testing.assert_array_equal(loaded_gradient, gradient)
    assert reference["e_corr"] == -0.2
    assert reference["e_total"] == -10.2
    assert reference["provenance"] == "reused_npz"
    assert reference["reused"] is True
    assert reference["correlation_forward_seconds"] == 0.0
    assert reference["correlation_reverse_seconds"] == 0.0
    assert reference["scf_pullback_seconds"] == 0.0
    assert "NPZ:basis" in reference["compatibility_checks"]
    assert (
        "companion JSON:validated_scf_response"
        in reference["compatibility_checks"]
    )


@pytest.mark.parametrize(
    "gradient,e_total,system_nao,message",
    [
        (np.zeros((3, 3)), -10.2, 9, "gradient shape"),
        (np.zeros((4, 3)), -11.2, 9, "current HF calculation"),
        (np.zeros((4, 3)), -10.2, 10, "system field 'nao'"),
    ],
)
def test_load_reference_npz_rejects_incompatible_checkpoint(
    tmp_path, gradient, e_total, system_nao, message,
):
    path = tmp_path / "bad_reference.npz"
    np.savez_compressed(
        path,
        reference_gradient=gradient,
        reference_correlation_energy=np.asarray(-0.2),
        reference_total_energy=np.asarray(e_total),
        system_nao=np.asarray(system_nao),
    )
    path.with_suffix(".json").write_text(json.dumps({
        "scf": {
            "response_backend": "standard_fixed_point_implicit",
            "first_order_custom": False,
        }
    }))
    with pytest.raises(ValueError, match=message):
        benchmark._load_reference_npz(
            path,
            e_hf=-10.0,
            expected_gradient_shape=(4, 3),
            expected_system={"nao": 9},
        )


@pytest.mark.parametrize(
    "scf_metadata,message",
    [
        ({}, "lacks validated SCF response provenance"),
        ({
            "response_backend": "standard_fixed_point_implicit",
        }, "unvalidated SCF response backend"),
        ({
            "response_backend": "finite_iteration_replay",
            "first_order_custom": True,
        }, "unvalidated SCF response backend"),
    ],
)
def test_load_reference_npz_rejects_unvalidated_scf_response(
    tmp_path, scf_metadata, message,
):
    path = tmp_path / "unsafe_reference.npz"
    np.savez_compressed(
        path,
        reference_gradient=np.zeros((4, 3)),
        reference_correlation_energy=np.asarray(-0.2),
        reference_total_energy=np.asarray(-10.2),
        system_nao=np.asarray(9),
    )
    if scf_metadata:
        path.with_suffix(".json").write_text(
            json.dumps({"scf": scf_metadata})
        )
    with pytest.raises(ValueError, match=message):
        benchmark._load_reference_npz(
            path,
            e_hf=-10.0,
            expected_gradient_shape=(4, 3),
            expected_system={"nao": 9},
        )
