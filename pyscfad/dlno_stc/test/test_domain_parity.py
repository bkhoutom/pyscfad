"""Explicit native support parity and independent labelled-frame acceptance."""

from dataclasses import fields

import jax
import numpy as np
import pytest

from pyscfad.dlno_stc.test import _boys_acceptance as accept


@pytest.fixture(scope="module")
def finite_case():
    from pyscfad.dlno_stc.domain import rebuild_domain_data

    mf = accept.build_mf(accept.molecule())
    static = accept.selection(mf)
    return mf, static, rebuild_domain_data(mf, static)


@pytest.fixture(scope="module")
def native_case(finite_case):
    native = pytest.importorskip("pyscfad.dlno.domain")
    from pyscfad.dlno._selection import build_domain_selections
    from pyscfad.dlno.dlno_base import rebuild_domain_data

    mf, static, _ = finite_case
    options = {f.name: getattr(static.thresholds, f.name)
               for f in fields(static.thresholds)
               if f.name in {x.name for x in fields(native.DLNOThresholds)}}
    topology = native.build_domain_topology(
        mf, frozen=2, lo_type="boys", lo_kwargs=accept.LO_OPTIONS,
        frag_lolist=static.frag_lolist, thresholds=native.DLNOThresholds(**options),
    )
    reference = build_domain_selections(mf, topology)
    return topology, reference, rebuild_domain_data(mf, reference)


def test_boys_localization_and_common_arrays_match_native(finite_case, native_case):
    _, static, common = finite_case
    topology, reference, old = native_case
    np.testing.assert_allclose(static.target_reference_coeff, topology.iao_coeff,
                               atol=2e-7, rtol=0)
    # Replaying identical fixed references tests the reconstruction rather than
    # optimizer noise in two independently selected localization solutions.
    from dataclasses import replace
    from pyscfad.dlno_stc.domain import rebuild_domain_data
    aligned = replace(static, target_reference_coeff=reference.target_reference_coeff,
                      target_reference_coords=reference.target_reference_coords)
    replay = rebuild_domain_data(finite_case[0], aligned)
    residuals = {}
    for name in ("s1e", "fock", "occupied_coeff", "virtual_coeff",
                 "occupied_energy", "virtual_energy", "iao_coeff", "pao_coeff"):
        np.testing.assert_allclose(getattr(replay, name), getattr(old, name),
                                   atol=1e-9, rtol=0, err_msg=name)
        residuals[name] = float(np.max(np.abs(np.asarray(getattr(replay, name))
                                            - np.asarray(getattr(old, name)))))
    assert np.isfinite(np.asarray(common.iao_coeff)).all()
    accept.record_result("common_parity", {
        "reference_localization_max_abs": float(np.max(np.abs(
            static.target_reference_coeff - topology.iao_coeff))),
        "rebuild_max_abs": residuals,
    })


def test_fixed_supports_ranks_masks_and_target_mapping_match_native(
        finite_case, native_case):
    mf, static, _ = finite_case
    topology, old, _ = native_case
    for name in ("active_occ_indices", "active_vir_indices",
                 "pao_projected_out_indices", "pao_parent_ao_indices",
                 "ao2pao_map", "strong_mask"):
        np.testing.assert_array_equal(getattr(static, name), getattr(old, name))
    np.testing.assert_array_equal(static.forced_strong_mask, topology.forced_strong_mask)
    for name in ("pair_energy", "weak_pair_energy"):
        np.testing.assert_allclose(getattr(static, name), getattr(topology, name),
                                   atol=1e-12, rtol=0, err_msg=name)
    assert tuple(int(x[0]) for x in static.frag_lolist) == accept.TARGET_ORDER
    assert len(static.fragments) == 8
    assert np.count_nonzero(~static.strong_mask) > 0
    for i, (new, ref) in enumerate(zip(static.fragments, old.fragments)):
        for name in ("iao_indices", "strong_fragments", "extended_atoms",
                     "extended_ao_indices", "pao_center_atoms", "primary_atoms",
                     "primary_ao_indices", "primary_bp_atoms", "weak_occ_norm_keep",
                     "weak_occ_span_metric_keep"):
            np.testing.assert_array_equal(getattr(new, name), getattr(ref, name),
                                          err_msg=f"fragment {i}: {name}")
        assert len(new.extended_ao_indices) < mf.mol.nao
        assert new.target_index == list(new.strong_fragments).index(i)
        assert len(new.strong_occ_metric_keep) == len(new.strong_fragments)
        for name in ("strong_virtual", "weak_virtual"):
            space, old_space = getattr(new, name), getattr(ref, name)
            if space is None:
                assert old_space is None
                continue
            for field in fields(space):
                np.testing.assert_array_equal(getattr(space, field.name),
                                              getattr(old_space, field.name),
                                              err_msg=f"{i}: {name}.{field.name}")
    # Every hierarchy has its own role, even when this fixture's lists agree.
    for name in ("compact_bp_domain", "primary_bp_domain", "tight_bp_domain",
                 "pao_bp_domain"):
        for got, want in zip(getattr(static, name), getattr(topology, name)):
            np.testing.assert_array_equal(got, want)
    accept.record_result("selection_parity", {
        "status": "passed", "strong_domain_job_count": len(static.fragments),
        "ordered_global_targets": [int(x[0]) for x in static.frag_lolist],
        "ordered_local_target_slots": [int(x.target_index) for x in static.fragments],
        "weak_unordered_pair_count": int(np.count_nonzero(
            np.triu(~static.strong_mask, k=1))),
        "extended_ao_counts": [len(x.extended_ao_indices) for x in static.fragments],
        "occupied_ranks": [len(x.strong_occ_metric_keep) for x in static.fragments],
        "virtual_ranks": [len(x.strong_virtual.metric_keep) for x in static.fragments],
    })


def test_finite_occupied_frame_follows_ordered_projected_cholesky(finite_case):
    from pyscfad.dlno_stc.domain import build_boys_ed_frame
    from pyscfad.dlno_stc.prepare import prepare_inputs

    mf, static, common = finite_case
    residuals = []
    for i, fragment in enumerate(static.fragments):
        frame = build_boys_ed_frame(common, static, i)
        ao = fragment.extended_ao_indices
        s = np.asarray(common.s1e)
        s22 = s[np.ix_(ao, ao)]
        candidates = np.concatenate([
            np.asarray(common.fragment_occupied_data[int(p)].iao_coeff)
            for p in fragment.strong_fragments
        ], axis=1)
        projected = np.linalg.solve(s22, s[ao] @ candidates)
        gram = projected.T @ s22 @ projected
        lower = np.linalg.cholesky((gram + gram.T) / 2)
        expected = np.linalg.solve(lower, projected.T).T
        pivots = np.argmax(np.abs(expected), axis=0)
        expected *= np.where(expected[pivots, np.arange(expected.shape[1])] < 0,
                             -1.0, 1.0)[None, :]
        np.testing.assert_allclose(frame.occupied_coeff, expected, atol=1e-9, rtol=0)
        residuals.append(float(np.max(np.abs(np.asarray(frame.occupied_coeff) - expected))))
        assert frame.target_index == list(fragment.strong_fragments).index(i)
        packet = prepare_inputs(mf, common, static, i)
        assert set(packet) == {"foo", "fvv", "B"}
        assert packet["B"].shape[1:] == (len(fragment.strong_fragments),
                                         len(fragment.strong_virtual.metric_keep))
        result = accept.selected_result(packet, static, i)
        assert set(result["cotangents"]) == set(packet)
        assert np.isfinite(float(result["energy"]))
    accept.record_result("ordered_frame_parity", {"occupied_max_abs": residuals})


@pytest.mark.parametrize("step", [1e-5, 5e-6])
def test_selected_three_bars_follow_symmetric_fock_direction(finite_case, step):
    from pyscfad.dlno_stc.adjoint import apply_selected_pullback
    from pyscfad.dlno_stc.prepare import prepare_inputs

    mf, static, common = finite_case
    direction = np.random.default_rng(221).normal(size=common.fock.shape)
    direction = jax.numpy.asarray((direction + direction.T) / 2)

    def prepare(scale):
        return prepare_inputs(
            mf, common._replace(fock=common.fock + scale * direction), static, 0,
        )

    packet, pullback = jax.vjp(prepare, jax.numpy.asarray(0.0))
    result = accept.selected_result(packet, static, 0)
    (actual,) = apply_selected_pullback(pullback, packet, result)
    plus = accept.selected_result(prepare(step), static, 0, with_grad=False)["energy"]
    minus = accept.selected_result(prepare(-step), static, 0, with_grad=False)["energy"]
    np.testing.assert_allclose(actual, (plus - minus) / (2 * step),
                               rtol=1e-6, atol=1e-8)
    difference = float((plus - minus) / (2 * step))
    accept.record_result("fock_direction_" + str(step), {
        "step": step, "adjoint": float(actual), "central_difference": difference,
        "absolute_error": abs(float(actual) - difference),
    })


def test_fixed_choices_keep_complete_coordinate_response(finite_case):
    mf, static, _ = finite_case
    energy, bar = accept.correlation_value_and_grad(mf.mol, static)
    np.testing.assert_allclose(energy, accept.correlation_energy(mf, static),
                               atol=1e-10, rtol=0)
    coords = np.asarray(mf.mol.atom_coords())
    direction = np.zeros_like(coords)
    direction[1, 2] = 1.0
    actual = float(np.sum(np.asarray(bar.coords) * direction))
    assert abs(actual) > 1e-6
    differences = []
    for step in (1e-5, 5e-6):
        values = []
        for sign in (1, -1):
            moved = mf.mol.set_geom_(coords + sign * step * direction,
                                     unit="Bohr", inplace=False)
            values.append(accept.correlation_energy(accept.build_mf(moved), static))
        np.testing.assert_allclose(actual, (values[0] - values[1]) / (2 * step),
                                   rtol=1e-6, atol=1e-8)
        difference = (values[0] - values[1]) / (2 * step)
        differences.append({"step_bohr": step, "central_difference_Eh_per_bohr": difference,
                            "absolute_error_Eh_per_bohr": abs(actual - difference)})
    accept.record_result("coordinate_response", {
        "correlation_energy_Eh": energy, "adjoint_Eh_per_bohr": actual,
        "gradient_Eh_per_bohr": np.asarray(bar.coords).tolist(),
        "force_Eh_per_bohr": (-np.asarray(bar.coords)).tolist(),
        "fixed_choice_central_differences": differences,
    })
