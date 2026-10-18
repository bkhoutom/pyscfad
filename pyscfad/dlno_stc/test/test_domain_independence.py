"""The standalone Boys path must run while native DLNO imports are blocked."""

import json
import os
from pathlib import Path
import subprocess
import sys


def isolated_smoke(output_dir):
    import importlib.abc

    attempted = []

    class BlockNative(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "pyscfad.dlno" or fullname.startswith("pyscfad.dlno."):
                attempted.append(fullname)
                raise ImportError(f"native DLNO is blocked: {fullname}")
            return None

    sys.meta_path.insert(0, BlockNative())
    import jax
    import numpy as np
    from pyscfad.dlno_stc.test import _boys_acceptance as accept
    from pyscfad.dlno_stc import _restart, _stc_mp2, build_domain_groups
    from pyscfad.dlno_stc.adjoint import (
        apply_full_pullback, apply_selected_pullback, pullback_inputs,
    )
    from pyscfad.dlno_stc.domain import build_system_local_frame, rebuild_domain_data
    from pyscfad.dlno_stc.prepare import prepare_boys_group_inputs, prepare_system_inputs
    from pyscfad.dlno_stc._pairs import screen_pairs, weak_pair_energy, weak_pair_indices

    mf = accept.build_mf(accept.molecule())
    static = accept.selection(mf)
    groups = build_domain_groups(static)
    group = groups[0]
    common, common_pullback = jax.vjp(lambda x: rebuild_domain_data(x, static), mf)
    packet, pullback = jax.vjp(
        lambda x, c: prepare_boys_group_inputs(x, c, static, group), mf, common,
    )
    result = accept.selected_result(packet, static, group.representative_index,
                                    target_indices=group.target_indices)
    mf_bar, common_bar = apply_selected_pullback(pullback, packet, result)
    replay = pullback_inputs(
        lambda x, c: prepare_boys_group_inputs(x, c, static, group), (mf, common),
        packet, dict(result, metadata={"cotangent_seed": 1.0}),
    )
    for actual, expected in zip(jax.tree_util.tree_leaves(replay),
                                jax.tree_util.tree_leaves((mf_bar, common_bar))):
        np.testing.assert_array_equal(actual, expected)
    common_mf_bar, = common_pullback(common_bar)
    assert np.isfinite(np.asarray(common_mf_bar.mo_coeff)).all()
    exact_scores, _, _, _ = screen_pairs(mf, common, static, model="exact")
    assert exact_scores.shape == static.strong_mask.shape
    assert np.isfinite(exact_scores).all()

    pairs = tuple(weak_pair_indices(static))
    assert pairs
    i, j = pairs[0]
    weak, weak_pullback = jax.vjp(
        lambda x, c: weak_pair_energy(x, c, static, i, j), mf, common,
    )
    weak_mf_bar, weak_common_bar = weak_pullback(jax.numpy.ones_like(weak))
    weak_common_mf_bar, = common_pullback(weak_common_bar)
    assert np.isfinite(np.asarray(weak_mf_bar.mol.coords)).all()
    assert np.isfinite(np.asarray(weak_common_mf_bar.mo_coeff)).all()

    frame = build_system_local_frame(mf, static.active_occ_indices,
                                    static.active_vir_indices)
    system, system_pullback = jax.vjp(
        lambda x: prepare_system_inputs(
            x, static.active_occ_indices, static.active_vir_indices,
            boys_reference=frame.boys_reference,
            virtual_anchor_columns=frame.virtual_anchor_columns,
        ), mf,
    )
    assert set(packet) == set(system) == {"foo", "fvv", "B"}
    system_result = _stc_mp2.solve_full(
        *(np.ascontiguousarray(system[name]) for name in ("foo", "fvv", "B")),
        accept.controls(), with_grad=True,
    )
    system_bar, = apply_full_pullback(system_pullback, system, system_result)
    assert np.isfinite(np.asarray(system_bar.mo_coeff)).all()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manager = _restart.RestartManager(
        output_dir / "checkpoint", method="boys-selected",
        scientific_payload={"frozen": 2, "equation": "boys_target_laplace_mp2_v1",
                            "grouping": _restart.grouping_payload(static, groups)},
    )
    saved = {"static": static, "groups": groups}
    digest = manager.save_static(saved)
    restored = manager.load_static(expected_type=dict)
    assert restored["groups"] == groups
    _restart.validate_grouping_payload(
        _restart.grouping_payload(restored["static"], restored["groups"]), static, groups,
    )
    assert _restart.scientific_digest(restored) == digest
    trees = {"mf_bar": mf_bar, "common_bar": common_bar, "packet": packet}
    manager.save_record("response", trees=trees, scalars={"energy": result["energy"]})
    record = manager.load_record("response", templates=trees)
    for actual, expected in zip(jax.tree_util.tree_leaves(record.trees),
                                jax.tree_util.tree_leaves(trees)):
        np.testing.assert_array_equal(actual, expected)
    manager.close()
    # Exercise complete supported production entry points while the blocker is active.
    from pyscfad import dlno_stc
    from pyscfad.dlno_stc import backend, mp2
    controls = dict(accept.controls(), energy_tolerance=.0003)
    direct, direct_bar = dlno_stc.value_and_grad(
        mf.mol, accept.build_mf, static, controls=controls)
    np.testing.assert_allclose(dlno_stc.kernel(mf, static, controls=controls), direct,
                               atol=1e-10, rtol=0)
    workdir = output_dir / "production-exchange"
    rows = mp2.export_run(mf, static, workdir, controls=controls)
    assert len(rows) == len(groups)
    restored_static = mp2.load_static_from_run(mf, workdir, frozen=2, controls=controls)
    for group_id in range(len(groups)):
        folder = workdir / f"fragment_{group_id:04d}"
        mp2.solve_request(folder / "input.h5", folder / "result.h5", backend.solve)
    imported, imported_bar = mp2.imported_value_and_grad(
        mf.mol, accept.build_mf, restored_static,
        lambda i: mp2.load_disk_packet(workdir, i), include_hf=False)
    np.testing.assert_allclose(imported, direct, atol=1e-10, rtol=0)
    np.testing.assert_allclose(imported_bar.coords, direct_bar.coords, atol=1e-8, rtol=0)
    system_energy, system_mol_bar = dlno_stc.value_and_grad(
        mf.mol, accept.build_mf, scope="system", frozen=2,
        controls=controls, lo_kwargs=accept.LO_OPTIONS)
    np.testing.assert_allclose(dlno_stc.kernel(mf, scope="system", frozen=2,
        controls=controls, lo_kwargs=accept.LO_OPTIONS), system_energy, atol=1e-10, rtol=0)
    assert np.isfinite(np.asarray(system_mol_bar.coords)).all()
    assert not attempted
    assert not any(name == "pyscfad.dlno" or name.startswith("pyscfad.dlno.")
                   for name in sys.modules)
    summary = {
        "status": "passed", "native_import_attempts": attempted,
        "production_domain_energy_Eh": float(direct),
        "production_disk_energy_Eh": float(imported),
        "production_system_energy_Eh": float(system_energy),
        "strong_domain_job_count": len(static.fragments),
        "solver_group_count": len(groups),
        "group_members": [list(g.fragment_indices) for g in groups],
        "group_targets": [list(g.target_indices) for g in groups],
        "ordered_global_targets": [int(x[0]) for x in static.frag_lolist],
        "ordered_local_target_slots": [int(x.target_index) for x in static.fragments],
        "weak_unordered_pair_count": len(pairs),
        "finite_shapes": {k: list(v.shape) for k, v in packet.items()},
        "system_shapes": {k: list(v.shape) for k, v in system.items()},
        "selected_energy_Eh": float(result["energy"]),
        "system_energy_Eh": float(system_result["energy"]),
        "weak_pair_energy_Eh": float(weak), "static_digest": digest,
        "checks": ["production direct/domain gradient", "production export/solve/restart/replay",
                   "production system energy/gradient", "selection", "exact pair classification", "common rebuild/VJP", "grouped finite preparation/VJP",
                   "grouped selected solver", "grouped strict disk replay", "weak energy/VJP",
                   "system preparation/VJP", "typed groups/static and cotangent checkpoint"],
    }
    (output_dir / "independence_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


def test_standalone_construction_runs_without_native_dlno(tmp_path):
    env = dict(os.environ, OMP_NUM_THREADS="1", OMP_THREAD_LIMIT="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", JAX_PLATFORMS="cpu",
               XLA_FLAGS="--xla_cpu_multi_thread_eigen=false")
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
        env=env, capture_output=True, text=True, timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads((tmp_path / "independence_results.json").read_text())
    assert summary["status"] == "passed"
    assert summary["native_import_attempts"] == []
    assert summary["strong_domain_job_count"] == 8
    assert summary["solver_group_count"] == 2
    assert summary["group_members"] == [[0, 1, 4, 5], [2, 3, 6, 7]]
    assert summary["weak_unordered_pair_count"] > 0
    from pyscfad.dlno_stc.test._boys_acceptance import record_result
    record_result("independence_results", summary)


if __name__ == "__main__":
    isolated_smoke(sys.argv[1])
