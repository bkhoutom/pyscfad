"""Independent labelled-frame and selected-response acceptance."""

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
