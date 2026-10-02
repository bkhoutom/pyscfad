"""Host workflow reporting and file exchange behavior."""

import numpy as np
import pytest

from pyscfad.dlno_stc.exchange import read_input, read_result, write_input
from pyscfad.dlno_stc.mp2 import format_domain_dimensions, solve_request


def test_dimension_table_reports_each_domain():
    table = format_domain_dimensions([
        {"fragment_id": 0, "ed_ao": 7, "nocc": 2, "nvir": 3},
        {"fragment_id": 1, "ed_ao": 11, "nocc": 4, "nvir": 5},
    ])
    lines = table.splitlines()
    assert "Fragment" in lines[0]
    assert "ED AO" in lines[0]
    assert "Occ" in lines[0]
    assert "Vir" in lines[0]
    assert len(lines) == 4
    assert lines[2].split() == ["0", "7", "2", "3"]
    assert lines[3].split() == ["1", "11", "4", "5"]


def test_shared_run_payload_preserves_checkpoint_fingerprints():
    """Moving host helpers must not invalidate existing full or finite runs."""
    from types import SimpleNamespace

    from pyscfad.dlno._restart import scientific_digest
    from pyscfad.dlno_stc._workflow import run_payload
    from pyscfad.dlno_stc.mp2 import _run_payload

    mol = SimpleNamespace(
        natm=1, atom_coords=lambda: np.array([[0.0, 0.0, 0.5]]),
        atom_symbol=lambda index: "H", charge=0, spin=0,
        _basis={"H": [[0, [1.0, 1.0]]]}, _ecp={}, cart=False,
    )
    mf = SimpleNamespace(
        mol=mol, with_df=SimpleNamespace(auxbasis="weigend", _cderi=None),
        mo_occ=np.array([2.0, 0.0]),
    )
    full = run_payload(mf, 0, driver="dlno-stc-mp2")
    finite = run_payload(mf, 0, driver="dlno-stc-finite-three-tensor")
    assert scientific_digest(full) == (
        "c215f336e2e1c821716a83a1c4aad80dcfa57a30f58c4cc6914dc80236487e95"
    )
    assert scientific_digest(finite) == (
        "ee2ce7ae55e405c34af9f268c6b088a139797eed8696890891035add68c5f3c2"
    )
    assert scientific_digest(_run_payload(mf, 0)) == scientific_digest(full)


def test_disk_backend_matches_direct_call_and_refuses_overwrite(tmp_path):
    inputs = {
        "foo": np.array([[1.0, 0.2], [0.2, 2.0]]),
        "fvv": np.array([[3.0]]),
        "B": np.arange(4.0).reshape(2, 2, 1),
    }
    metadata = {
        "schema_version": 1, "method": "mp2",
        "energy_kind": "quadratic_test_v1", "fragment_id": 0,
        "target_index": 0, "basis_frame": "orthonormal_local",
        "B_axes": "Pia", "units": {"energy": "Eh", "length": "bohr"},
        "frozen": [0], "orbital_order": {"occupied": [0, 1], "virtual": [0]},
        "auxiliary_order": "test-aux", "checkpoint_id": "test-checkpoint",
        "domain_options": {"full_support": True}, "code_revision": "test",
        "dirty_worktree": False,
    }
    controls = {}
    request_path = tmp_path / "input.h5"
    result_path = tmp_path / "result.h5"
    write_input(request_path, inputs, metadata, controls)

    def backend(current_inputs, current_metadata, current_controls, *, comm=None):
        assert current_controls == controls
        energy = 0.5 * sum(np.sum(value**2) for value in current_inputs.values())
        return {
            "energy": float(energy), "cotangents": current_inputs,
            "metadata": {
                **{key: current_metadata[key] for key in (
                    "schema_version", "method", "energy_kind", "fragment_id",
                    "target_index", "basis_frame", "B_axes", "request_fingerprint",
                )},
                "cotangent_seed": 1.0, "backend": "test",
                "backend_version": "1", "actual_sample_count": 0,
                "derivative_convention": "fixed_sample",
                "seed_replay": {"mode": "deterministic"},
            },
            "diagnostics": {},
        }

    saved, saved_metadata, saved_controls = read_input(request_path)
    expected = backend(saved, saved_metadata, saved_controls)
    actual = solve_request(request_path, result_path, backend)
    assert actual["energy"] == expected["energy"]
    first = read_result(result_path, saved, saved_metadata, controls=saved_controls)
    second = read_result(result_path, saved, saved_metadata, controls=saved_controls)
    assert first["energy"] == second["energy"] == expected["energy"]
    with pytest.raises(FileExistsError):
        solve_request(request_path, result_path, backend)


def test_full_boys_disk_import_matches_in_memory_pullback(tmp_path):
    from pyscfad import gto, scf
    from pyscfad.dlno import _selection, domain as dlno_domain
    from pyscfad.dlno.mp2 import _fix_restart_mo_phases
    from pyscfad.dlno._restart import RestartMismatchError
    from pyscfad.dlno_stc.mp2 import (
        export_run, imported_value_and_grad, load_disk_packet,
        load_static_from_run,
    )
    from pyscfad.dlno_stc.test.reference import reference_backend

    mol = gto.Mole(
        atom="O 0 0 0; H 0.05 0.76 0.59; H -0.03 -0.70 0.63",
        basis="sto-3g", verbose=0, max_memory=3000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)

    def build_mf(current_mol):
        mf = scf.RHF(current_mol).density_fit()
        mf.conv_tol = 1e-11
        mf.conv_tol_grad = 1e-9
        mf.kernel()
        return _fix_restart_mo_phases(mf)

    mf = build_mf(mol)
    topology = dlno_domain.build_domain_topology(
        mf, frozen=1, lo_type="boys", force_full_domains=True,
        thresholds=dlno_domain.DLNOThresholds(
            domain_pao=0.0, ed_pao=0.0, pao_norm=1e-10),
    )
    static = _selection.build_domain_selections(mf, topology)
    output = []
    rows = export_run(mf, static, tmp_path, controls={}, reporter=output.append)
    assert len(rows) == len(static.fragments)
    assert "ED AO" in "\n".join(output)
    loaded_static = load_static_from_run(mf, tmp_path, frozen=1)
    assert len(loaded_static.fragments) == len(static.fragments)
    with pytest.raises(RestartMismatchError, match="checkpoint|manifest|mismatch"):
        load_static_from_run(mf, tmp_path, frozen=2)
    moved_coords = np.asarray(mol.atom_coords()).copy()
    moved_coords[1, 2] += 0.01
    moved_mol = mol.set_geom_(moved_coords, unit="Bohr", inplace=False)
    moved_mf = build_mf(moved_mol)
    with pytest.raises(RestartMismatchError, match="checkpoint|manifest|mismatch"):
        load_static_from_run(moved_mf, tmp_path, frozen=1)

    memory_packets = {}
    for fragment_id in range(len(static.fragments)):
        folder = tmp_path / f"fragment_{fragment_id:04d}"
        inputs, metadata, controls = read_input(folder / "input.h5")
        memory_packets[fragment_id] = (
            inputs, metadata, controls,
            reference_backend(inputs, metadata, controls),
        )
        solve_request(folder / "input.h5", folder / "result.h5", reference_backend)

    stale_packets = dict(memory_packets)
    saved_inputs, request_metadata, request_controls, saved_result = stale_packets[0]
    stale_packets[0] = (
        saved_inputs, request_metadata, request_controls,
        {**saved_result, "metadata": {
            **saved_result["metadata"], "request_fingerprint": "0" * 64,
        }},
    )
    with pytest.raises(ValueError, match="fingerprint"):
        imported_value_and_grad(
            mol, build_mf, loaded_static,
            lambda fragment_id: stale_packets[fragment_id],
        )

    from dataclasses import replace
    changed_selection = replace(loaded_static, frozen=2)
    with pytest.raises(ValueError, match="checkpoint|frozen|selection"):
        imported_value_and_grad(
            mol, build_mf, changed_selection,
            lambda fragment_id: memory_packets[fragment_id],
        )

    memory_energy, memory_bar = imported_value_and_grad(
        mol, build_mf, loaded_static,
        lambda fragment_id: memory_packets[fragment_id],
    )
    disk_energy, disk_bar = imported_value_and_grad(
        mol, build_mf, loaded_static,
        lambda fragment_id: load_disk_packet(tmp_path, fragment_id),
    )
    np.testing.assert_allclose(disk_energy, memory_energy, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        disk_bar.coords, memory_bar.coords, rtol=1e-7, atol=1e-9,
    )

    from pyscfad.dlno.mp2 import DLNOMP2
    deterministic_energy, deterministic_bar = DLNOMP2.value_and_grad(
        mol, build_mf=build_mf, frozen=1, topology=loaded_static,
        lo_type="boys",
    )
    np.testing.assert_allclose(
        disk_energy, deterministic_energy, rtol=1e-8, atol=1e-9,
    )
    np.testing.assert_allclose(
        disk_bar.coords, deterministic_bar.coords, rtol=2e-5, atol=2e-7,
    )
    partial_energy, partial_bar = imported_value_and_grad(
        mol, build_mf, loaded_static,
        lambda fragment_id: memory_packets[fragment_id],
        fragment_ids=(0,), include_hf=False,
    )
    np.testing.assert_allclose(
        partial_energy, memory_packets[0][3]["energy"], rtol=1e-10, atol=1e-10,
    )
    assert np.isfinite(np.asarray(partial_bar.coords)).all()
    with pytest.raises(ValueError, match="duplicate"):
        imported_value_and_grad(
            mol, build_mf, loaded_static,
            lambda fragment_id: memory_packets[fragment_id],
            fragment_ids=(0, 0), include_hf=False,
        )


def test_example_runs_export_solve_import_in_fresh_processes(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[3]
    script = root / "examples" / "dlno_stc" / "workflow.py"
    workdir = tmp_path / "fresh_run"
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def command(action):
        return subprocess.run(
            [sys.executable, str(script), action, "--workdir", str(workdir)],
            cwd=root, env=env, capture_output=True, text=True, timeout=120,
        )

    exported = command("export")
    assert exported.returncode == 0, exported.stderr
    assert "ED AO" in exported.stdout
    solved = command("reference-solve")
    assert solved.returncode == 0, solved.stderr
    first = command("import")
    assert first.returncode == 0, first.stderr
    assert "E_total" in first.stdout
    assert "dE/dR" in first.stdout
    second = command("import")
    assert second.returncode == 0, second.stderr
    assert first.stdout == second.stdout


def test_example_single_fragment_is_labeled_partial(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[3]
    script = root / "examples" / "dlno_stc" / "workflow.py"
    workdir = tmp_path / "partial_run"
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def command(action, *extra):
        return subprocess.run(
            [sys.executable, str(script), action, "--workdir", str(workdir),
             *extra],
            cwd=root, env=env, capture_output=True, text=True, timeout=120,
        )

    exported = command("export")
    assert exported.returncode == 0, exported.stderr
    solved = command("reference-solve", "--fragment", "0")
    assert solved.returncode == 0, solved.stderr
    partial = command("import", "--fragment", "0")
    assert partial.returncode == 0, partial.stderr
    assert "E_fragment" in partial.stdout
    assert "E_total" not in partial.stdout
    incomplete = command("import")
    assert incomplete.returncode != 0
    assert "incomplete run" in incomplete.stderr
