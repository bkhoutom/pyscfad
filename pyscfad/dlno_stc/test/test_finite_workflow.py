"""Finite tensor packets and molecular cotangent replay on truncated EDs."""

from dataclasses import replace

import numpy as np
import pytest

from pyscfad import gto, scf
from pyscfad.dlno._selection import build_domain_selections
from pyscfad.dlno.dlno_base import rebuild_domain_data
from pyscfad.dlno.domain import DLNOThresholds, build_domain_topology
from pyscfad.dlno_stc.exchange import read_input, write_result
from pyscfad.dlno_stc.finite import (
    _validate_packet_identity, export_finite_run, imported_finite_value_and_grad,
    load_finite_disk_packet, load_finite_static_from_run,
)
from pyscfad.dlno_stc.prepare import prepare_finite_inputs


def _mf(mol):
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    return mf


def _quadratic(inputs):
    return 0.5 * sum(np.sum(np.asarray(value) ** 2) for value in inputs.values())


def _quadratic_result(inputs, metadata):
    identity = (
        "schema_version", "method", "energy_kind", "fragment_id",
        "target_index", "basis_frame", "B_axes", "request_fingerprint",
    )
    return {
        "energy": np.float64(_quadratic(inputs)),
        "cotangents": {name: np.asarray(value) for name, value in inputs.items()},
        "metadata": {
            **{key: metadata[key] for key in identity},
            "cotangent_seed": 1.0,
            "backend": "quadratic_finite_test",
            "backend_version": "1",
            "actual_sample_count": 0,
            "derivative_convention": "fixed_sample",
            "seed_replay": {"mode": "deterministic"},
        },
        "diagnostics": {},
    }


def test_truncated_boys_packet_molecular_pullback_matches_displacement(tmp_path):
    mol = gto.Mole(
        atom="O 0 0 0; H 0.1 -0.75 0.57; H 0 0.8 0.61; "
             "O 0 0 8; H 0.13 -0.74 8.57; H 0.02 0.81 8.61",
        basis="sto-3g", verbose=0, max_memory=3000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = _mf(mol)
    topology = build_domain_topology(
        mf, frozen=2, lo_type="boys",
        lo_kwargs={"conv_tol": 1e-12, "conv_tol_grad": 1e-9},
        force_full_domains=False,
        thresholds=DLNOThresholds(pair_energy=1e-4),
    )
    static = build_domain_selections(mf, topology)
    with pytest.raises(ValueError, match="one Boys orbital"):
        export_finite_run(mf, replace(static, lo_type="iao"),
                          tmp_path / "non_boys", controls={})
    grouped_fragment = replace(
        static.fragments[0], iao_indices=np.array([0, 1], dtype=np.int32),
    )
    grouped_static = replace(
        static, fragments=(grouped_fragment, *static.fragments[1:]),
    )
    with pytest.raises(ValueError, match="one Boys orbital"):
        export_finite_run(mf, grouped_static,
                          tmp_path / "grouped", controls={})
    finite_id = next(
        i for i, fragment in enumerate(static.fragments)
        if len(fragment.extended_atoms) < mol.natm
    )
    output = []
    rows = export_finite_run(mf, static, tmp_path, controls={}, reporter=output.append)
    assert len(rows) == len(static.fragments)
    assert "ED AO" in "\n".join(output)
    assert rows[finite_id]["ed_ao"] < mol.nao
    loaded_static = load_finite_static_from_run(mf, tmp_path, frozen=2)
    folder = tmp_path / f"fragment_{finite_id:04d}"
    inputs, metadata, controls = read_input(folder / "input.h5")
    assert controls == {}
    assert metadata["target_index"] is None
    assert metadata["global_target_id"] == int(static.fragments[finite_id].iao_indices[0])
    assert metadata["energy_kind"] == "finite_three_tensor_v1"
    assert metadata["basis_frame"] == "orthonormal_local"
    assert "before Fock diagonalization" in metadata["orbital_order"]["occupied"]
    assert "fixed QR-selected PAO parent columns" in metadata["orbital_order"]["virtual"]
    assert "metric-Cholesky" in metadata["orbital_order"]["virtual"]
    assert "Fock diagonalization" in metadata["orbital_order"]["virtual"]
    assert np.linalg.norm(inputs["foo"] - np.diag(np.diag(inputs["foo"]))) > 1e-7
    assert np.linalg.norm(inputs["fvv"] - np.diag(np.diag(inputs["fvv"]))) > 1e-7
    assert metadata["domain_options"]["ed_ao"] == rows[finite_id]["ed_ao"]
    anchors = metadata["virtual_anchor_parent_columns"]
    parent = static.fragments[finite_id].strong_virtual.parent_columns
    assert len(anchors) == rows[finite_id]["nvir"]
    assert len(set(anchors)) == len(anchors)
    assert set(anchors) <= set(parent)
    replay = prepare_finite_inputs(
        mf, rebuild_domain_data(mf, loaded_static), loaded_static, finite_id,
        virtual_anchor_columns=tuple(anchors),
    )
    for name in inputs:
        np.testing.assert_allclose(replay[name], inputs[name], atol=1e-9)
    duplicate = dict(metadata, virtual_anchor_parent_columns=[
        anchors[0], anchors[0], *anchors[2:],
    ])
    with pytest.raises(ValueError, match="virtual anchor"):
        _validate_packet_identity(loaded_static, finite_id, inputs, duplicate)
    outside = dict(metadata, virtual_anchor_parent_columns=[
        int(max(parent)) + 1, *anchors[1:],
    ])
    with pytest.raises(ValueError, match="virtual anchor"):
        _validate_packet_identity(loaded_static, finite_id, inputs, outside)
    write_result(folder / "result.h5", _quadratic_result(inputs, metadata))

    energy, bar = imported_finite_value_and_grad(
        mol, _mf, loaded_static,
        lambda fragment_id: load_finite_disk_packet(tmp_path, fragment_id),
        fragment_ids=(finite_id,),
    )
    np.testing.assert_allclose(energy, _quadratic(inputs), atol=1e-10, rtol=0)
    direction = np.zeros((mol.natm, 3))
    direction[1, 2] = 1.0
    coords = np.asarray(mol.atom_coords())
    h = 1e-4

    def displaced(sign):
        moved = mol.set_geom_(coords + sign * h * direction,
                              unit="Bohr", inplace=False)
        current_mf = _mf(moved)
        common = rebuild_domain_data(current_mf, loaded_static)
        return _quadratic(prepare_finite_inputs(
            current_mf, common, loaded_static, finite_id,
            virtual_anchor_columns=tuple(anchors),
        ))

    finite_difference = (displaced(1) - displaced(-1)) / (2 * h)
    np.testing.assert_allclose(
        np.sum(np.asarray(bar.coords) * direction), finite_difference,
        atol=2e-5, rtol=3e-4,
    )
