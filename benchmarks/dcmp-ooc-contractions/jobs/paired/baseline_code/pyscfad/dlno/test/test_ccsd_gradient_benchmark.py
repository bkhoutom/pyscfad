import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples/lno/21-iao_dlno_ccsdt_gradient_convergence.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "iao_dlno_ccsdt_gradient_convergence", _SCRIPT
)
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def _normalized_basis(value):
    return str(value).strip().lower().replace("_", "-").replace("-", "")


def test_exact_omcb_defaults_and_threshold_grid():
    args = benchmark.parse_args([])
    assert Path(args.xyz).name.lower() == "omcb.xyz"
    assert _normalized_basis(args.basis) == "ccpvtz"
    assert args.auxbasis is None
    assert args.frozen is None
    assert args.pair_cutoffs == [3e-2, 1e-2, 1e-3]
    assert args.fixed_pair_cutoff == 1e-3
    assert args.lno_thresholds == [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5]
    assert args.lno_vir_ratio == 0.1
    assert args.pair_energy_target_mha == 1.0
    assert Path(args.mpi_pair_results).as_posix() == (
        "notes/results/omcb_ccpvtz_iao_mp2_mpi_pair_sweep"
    )
    assert Path(args.output_prefix).as_posix() == (
        "notes/results/iao_dlno_ccsdt_omcb_ccpvtz"
    )
    assert "sto" not in args.output_prefix.lower()
    assert args.mp2_block_memory == 128.0
    assert args.scf_conv_tol == 1e-10
    assert args.reference_npz is None
    assert not args.resume
    assert not args.allow_canonical_memory_oversubscription


def test_exact_defaults_are_described_as_ccpvtz_auto_jkfit_all_electron():
    documentation = benchmark.__doc__.lower()
    assert "omcb/cc-pvtz" in documentation
    assert "automatic" in documentation
    assert "cc-pvtz-jkfit" in documentation
    assert "all-electron" in documentation
    assert "omcb_ccpvtz_iao_mp2_mpi_pair_sweep" in documentation
    assert "iao_dlno_ccsdt_omcb_ccpvtz" in documentation


def test_auto_auxbasis_and_all_electron_cli_spellings_are_preserved():
    for auxbasis in ("auto", "none", "null"):
        args = benchmark.parse_args(["--auxbasis", auxbasis])
        assert args.auxbasis is None
    for frozen in ("none", "null", "all-electron", "all_electron"):
        args = benchmark.parse_args(["--frozen", frozen])
        assert args.frozen is None


def test_exact_default_signature_matches_example20_checkpoint_metadata():
    args = benchmark.parse_args([])
    mol = SimpleNamespace(natm=36, nao=696, nelectron=96)
    signature = benchmark._signature(args, mol)
    assert signature["geometry_sha256"] == (
        "950b357e267d4872d8ad5da0f8f4e5af5e6165eb6f28e2ec695fcea52783a837"
    )
    assert signature["basis"] == "cc-pvtz"
    assert signature["auxbasis"] == "auto"
    assert signature["frozen"] is None
    assert signature["fixed_pair_cutoff"] == 1e-3


def test_canonical_ccsdt_memory_preflight_rejects_exact_defaults():
    args = benchmark.parse_args([])
    frozen12 = benchmark.estimate_canonical_dfccsdt_gradient_memory(
        nocc=36, nvir=648, naux=1668, nao=696, nmo=684,
    )
    all_electron = benchmark.estimate_canonical_dfccsdt_gradient_memory(
        nocc=48, nvir=648, naux=1668, nao=696, nmo=696,
    )
    assert frozen12["one_oovv_array_gib"] == pytest.approx(4.054573)
    assert frozen12["estimated_gib"] > 39.0
    assert all_electron["estimated_gib"] > 68.0
    assert "nine_oovv_live_arrays" in frozen12["components_gib"]

    # The exact defaults have no validated CCSD(T) reference checkpoint, so
    # main must take this branch and refuse the canonical AD reference before
    # allocating its all-electron oovv state.
    assert args.reference_npz is None
    assert not args.resume
    with pytest.raises(RuntimeError, match="resource preflight refused") as err:
        benchmark._enforce_canonical_ccsdt_memory_preflight(
            all_electron.copy(), max_memory_mb=args.max_memory,
            allow_oversubscription=False,
        )
    assert "no global ovvv or wvvov" in str(err.value)

    accepted = benchmark._enforce_canonical_ccsdt_memory_preflight(
        all_electron.copy(), max_memory_mb=args.max_memory,
        allow_oversubscription=True,
    )
    assert accepted["accepted"]
    assert accepted["oversubscription_override"]


def test_exact_default_main_refuses_before_cderi_or_scf(monkeypatch, tmp_path):
    reference_gradient = np.zeros((36, 3))
    source_payload = {
        "system": {"naux": 1668},
        "settings": {"mpi_ranks": 2},
    }

    def fake_mpi_results(*args, **kwargs):
        reference = {
            "energy": -470.0,
            "gradient": reference_gradient,
            "seconds": 0.0,
            "provenance": "validated-test-fixture",
        }
        return reference, [], [], source_payload

    def forbidden_expensive_setup(*args, **kwargs):
        pytest.fail("exact-default preflight did not precede CDERI/SCF setup")

    monkeypatch.setattr(
        benchmark, "_load_mpi_pair_results", fake_mpi_results
    )
    monkeypatch.setattr(
        benchmark, "prepare_outcore_cderi", forbidden_expensive_setup
    )
    monkeypatch.setattr(benchmark, "make_build_mf", forbidden_expensive_setup)

    with pytest.raises(RuntimeError, match="resource preflight refused"):
        benchmark.main([
            "--output-prefix", str(tmp_path / "must_not_be_written"),
        ])


def test_canonical_ccsdt_memory_preflight_accepts_small_reference():
    estimate = benchmark.estimate_canonical_dfccsdt_gradient_memory(
        nocc=5, nvir=6, naux=50, nao=11, nmo=11,
    )
    accepted = benchmark._enforce_canonical_ccsdt_memory_preflight(
        estimate, max_memory_mb=1000.0,
        allow_oversubscription=False,
    )
    assert accepted["accepted"]
    assert accepted["estimated_gib"] < 0.01


def test_pair_cutoff_selection_is_loose_but_strictly_sub_target():
    rows = [
        {"stage": "pair", "pair_cutoff": 1e-3, "energy_error": -1.1e-3},
        {"stage": "pair", "pair_cutoff": 3e-4, "energy_error": -4.2e-4},
        {"stage": "pair", "pair_cutoff": 1e-4, "energy_error": -1.0e-4},
        {"stage": "lno", "pair_cutoff": 1.0, "energy_error": 0.0},
    ]
    assert benchmark.select_pair_cutoff(rows, 1.0) == 3e-4
    with pytest.raises(RuntimeError, match="no pair cutoff"):
        benchmark.select_pair_cutoff(rows[:1], 1.0)


def test_lis_statistics_record_every_fragment_rank():
    selections = (
        SimpleNamespace(
            full_occupied_space=False,
            full_virtual_space=False,
            internal_occ_keep=np.arange(2),
            occupied_lno_keep=np.arange(3),
            internal_vir_keep=np.arange(1),
            virtual_lno_keep=np.arange(7),
        ),
        SimpleNamespace(
            full_occupied_space=True,
            full_virtual_space=True,
            internal_occ_keep=np.arange(1),
            occupied_lno_keep=np.arange(0),
            internal_vir_keep=np.arange(2),
            virtual_lno_keep=np.arange(0),
        ),
    )
    static = SimpleNamespace(
        mp2_static=SimpleNamespace(
            active_occ_indices=np.arange(12),
            active_vir_indices=np.arange(20),
        ),
        fragments=selections,
    )
    values = benchmark._lis_statistics(static)
    assert values["lis_occupied_ranks"] == [5, 12]
    assert values["lis_virtual_ranks"] == [8, 20]
    assert values["mean_lis_occ"] == 8.5
    assert values["max_lis_vir"] == 20


def test_atomic_checkpoint_round_trip_and_resume(tmp_path):
    signature = {
        "geometry_sha256": "test-geometry", "natom": 2, "nao": 4,
        "nelectron": 4, "basis": "sto-3g", "auxbasis": "weigend",
        "frozen": 0,
    }
    metadata = {"signature": signature, "created_utc": "test"}
    reference_gradient = np.arange(6.0).reshape(2, 3) * 1e-3
    references = {
        "dfmp2": {
            "energy": -2.1, "gradient": reference_gradient,
            "seconds": 1.0, "provenance": "test",
        },
        "dfccsdt": {
            "energy": -2.2, "gradient": reference_gradient * 2.0,
            "seconds": 2.0, "provenance": "test",
        },
    }
    base = {field: None for field in benchmark.CSV_FIELDS}
    pair_row = {
        **base, "stage": "pair", "label": "1.0e-4",
        "pair_cutoff": 1e-4, "energy": -2.09,
    }
    lno_row = {
        **base, "stage": "lno", "label": "1.0e-3",
        "pair_cutoff": 1e-4, "lno_occ_threshold": 1e-3,
        "lno_vir_threshold": 1e-4, "energy": -2.19,
    }
    prefix = tmp_path / "benchmark"
    paths = benchmark._atomic_outputs(
        prefix, metadata, [pair_row], [lno_row],
        [reference_gradient + 1.0], [reference_gradient + 2.0], references,
    )
    assert all(path.exists() for path in paths)
    with prefix.with_suffix(".json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["pair_rows"][0]["pair_cutoff"] == 1e-4
    assert "gradient" not in payload["references"]["dfmp2"]

    loaded = benchmark._load_checkpoint(prefix, signature)
    _, loaded_references, pair_rows, lno_rows, pair_gradients, lno_gradients = loaded
    assert len(pair_rows) == len(pair_gradients) == 1
    assert len(lno_rows) == len(lno_gradients) == 1
    np.testing.assert_array_equal(
        loaded_references["dfccsdt"]["gradient"], reference_gradient * 2.0
    )

    with pytest.raises(ValueError, match="settings do not match"):
        benchmark._load_checkpoint(prefix, {**signature, "basis": "6-31g"})


def test_small_sto3g_smoke_configuration_is_an_explicit_nondefault_override(
    tmp_path,
):
    xyz = tmp_path / "h2.xyz"
    xyz.write_text("2\nH2\nH 0 0 0\nH 0 0 0.7\n", encoding="utf-8")
    output = tmp_path / "sto3g_smoke"
    args = benchmark.parse_args([
        "--xyz", str(xyz),
        "--basis", "sto-3g",
        "--auxbasis", "weigend",
        "--frozen", "0",
        "--pair-cutoffs", "1e-4",
        "--fixed-pair-cutoff", "1e-4",
        "--lno-thresholds", "1e-3",
        "--mpi-pair-results", "none",
        "--output-prefix", str(output),
    ])
    assert args.basis == "sto-3g"
    assert args.auxbasis == "weigend"
    assert args.frozen == 0
    assert args.pair_cutoffs == [1e-4]
    assert args.fixed_pair_cutoff == 1e-4
    assert args.lno_thresholds == [1e-3]
    assert args.mpi_pair_results is None
    assert Path(args.output_prefix) == output


def _write_mpi_pair_checkpoint(tmp_path, *, basis="sto-3g"):
    xyz = tmp_path / "molecule.xyz"
    xyz.write_text("2\nH2\nH 0 0 0\nH 0 0 0.7\n")
    prefix = tmp_path / "mpi_pair"
    reference_energy = -1.1
    reference_gradient = np.arange(6.0).reshape(2, 3) * 1e-4
    gradient = reference_gradient + 2e-5
    source_row = {
        "pair_threshold": 3e-4,
        "e_corr": -0.1,
        "e_total": reference_energy - 4.2e-4,
        "energy_error_eh": -4.2e-4,
        "energy_error_millihartree": -0.42,
        "gradient_rms_error": 2e-5,
        "gradient_max_abs_error": 2e-5,
        "gradient_l2_error": float(np.linalg.norm(gradient-reference_gradient)),
        "gradient_relative_l2_error": 0.1,
        "strong_pair_count": 5,
        "weak_pair_count": 1,
        "total_pair_count": 6,
        "nfragment": 4,
        "mean_ed_atoms": 1.5,
        "max_ed_atoms": 2,
        "mean_ed_aos": 2.5,
        "max_ed_aos": 3,
        "topology_statistics_seconds": 0.2,
        "mpi_energy_gradient_seconds": 1.7,
    }
    payload = {
        "system": {
            "geometry": str(xyz), "natom": 2, "nao": 4,
            "geometry_sha256": benchmark._geometry_sha256(xyz),
            "basis": basis, "auxbasis": "weigend", "frozen": 0,
        },
        "settings": {
            "pair_energy_model": "multipole", "mpi_ranks": 2,
        },
        "reference": {
            "timing": {"scf_forward_seconds": 0.3,
                       "scf_pullback_seconds": 0.7},
        },
        "rows": [source_row],
    }
    prefix.with_suffix(".json").write_text(json.dumps(payload))
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        pair_thresholds=np.asarray([3e-4]),
        total_energies=np.asarray([source_row["e_total"]]),
        gradients=np.asarray([gradient]),
        reference_total_energy=np.asarray(reference_energy),
        reference_gradient=reference_gradient,
    )
    return xyz, prefix, reference_gradient, gradient


def test_mpi_pair_checkpoint_import_is_validated_and_converted(tmp_path):
    xyz, prefix, reference_gradient, gradient = _write_mpi_pair_checkpoint(
        tmp_path
    )
    args = benchmark.parse_args([
        "--xyz", str(xyz),
        "--basis", "sto-3g", "--auxbasis", "weigend",
        "--frozen", "0",
        "--mpi-pair-results", str(prefix),
        "--fixed-pair-cutoff", "3e-4",
    ])
    signature = {
        "geometry_sha256": benchmark._geometry_sha256(xyz),
        "natom": 2, "nao": 4,
        "nelectron": 2, "basis": "sto-3g", "auxbasis": "weigend",
        "frozen": 0,
    }
    reference, rows, gradients, payload = benchmark._load_mpi_pair_results(
        prefix.with_suffix(".npz"), args, signature
    )
    assert reference["energy"] == -1.1
    np.testing.assert_array_equal(reference["gradient"], reference_gradient)
    assert len(rows) == len(gradients) == 1
    assert rows[0]["stage"] == "pair"
    assert rows[0]["pair_cutoff"] == 3e-4
    assert rows[0]["energy_abs_error_mha"] == pytest.approx(0.42)
    assert rows[0]["source_mpi_ranks"] == 2
    assert rows[0]["mean_ed_occ"] is None
    np.testing.assert_array_equal(gradients[0], gradient)
    assert payload["settings"]["mpi_ranks"] == 2


def test_mpi_pair_checkpoint_rejects_system_or_domain_mismatch(tmp_path):
    xyz, prefix, _, _ = _write_mpi_pair_checkpoint(
        tmp_path, basis="6-31g"
    )
    args = benchmark.parse_args([
        "--xyz", str(xyz),
        "--basis", "sto-3g", "--auxbasis", "weigend",
        "--frozen", "0",
        "--mpi-pair-results", str(prefix),
    ])
    signature = {
        "geometry_sha256": benchmark._geometry_sha256(xyz),
        "natom": 2, "nao": 4,
        "nelectron": 2, "basis": "sto-3g", "auxbasis": "weigend",
        "frozen": 0,
    }
    with pytest.raises(ValueError, match="field 'basis'"):
        benchmark._load_mpi_pair_results(prefix, args, signature)

    # Once the system metadata matches, a changed ED threshold is also
    # rejected because it would not describe the imported pair topology.
    with prefix.with_suffix(".json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["system"]["basis"] = "sto-3g"
    prefix.with_suffix(".json").write_text(json.dumps(payload))
    args.domain_pao = 1e-3
    with pytest.raises(ValueError, match="changes 'domain_pao'"):
        benchmark._load_mpi_pair_results(prefix, args, signature)


def test_bundled_mpi_pair_checkpoint_accepts_copied_omcb_geometry(tmp_path):
    source_xyz = _SCRIPT.with_name("OMCB.xyz")
    copied_xyz = tmp_path / "copied-and-renamed-omcb.xyz"
    copied_xyz.write_bytes(source_xyz.read_bytes())

    source_args = benchmark.parse_args([])
    copied_args = benchmark.parse_args(["--xyz", str(copied_xyz)])
    mol = SimpleNamespace(natm=36, nao=696, nelectron=96)
    source_signature = benchmark._signature(source_args, mol)
    copied_signature = benchmark._signature(copied_args, mol)
    assert copied_signature == source_signature
    assert "xyz" not in copied_signature
    assert set(copied_signature["mpi_pair_results"]) == {
        "npz_sha256", "json_sha256",
    }

    reference, rows, gradients, payload = benchmark._load_mpi_pair_results(
        source_args.mpi_pair_results, copied_args, copied_signature
    )
    assert reference["energy"] == pytest.approx(-470.8347515172865)
    assert len(rows) == len(gradients) == 3
    assert payload["system"]["geometry_sha256"] == (
        copied_signature["geometry_sha256"]
    )

    reference_path = tmp_path / "portable-reference.npz"
    reference_gradient = np.zeros((36, 3))
    np.savez_compressed(
        reference_path,
        signature=np.asarray(json.dumps(source_signature, sort_keys=True)),
        reference_mp2_total_energy=np.asarray(-470.0),
        reference_mp2_gradient=reference_gradient,
        reference_ccsdt_total_energy=np.asarray(-471.0),
        reference_ccsdt_gradient=reference_gradient,
    )
    loaded = benchmark._load_reference_npz(
        reference_path, copied_signature
    )
    assert loaded["dfccsdt"]["energy"] == -471.0
