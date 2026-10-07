"""The in-memory weighted boundary retains raw projections and all VJP paths."""

import jax
import numpy as np
import pytest

from pyscfad import numpy as jnp
from pyscfad.dlno_stc import adjoint, prepare, protocol
from pyscfad.dlno_stc.test.test_finite_prepare import finite_water_dimer


def _inputs(nvir=2):
    return {
        "foo": np.array([[-0.8, 0.1], [0.1, -0.5]]),
        "fvv": np.diag(np.arange(nvir, dtype=np.float64) + 0.2),
        "B": np.arange(3 * 2 * nvir, dtype=np.float64).reshape(3, 2, nvir) / 10,
        "target_projection": np.array([[0.6, -0.3]]),
        "partner_weight": np.array([[0.7, 0.15], [0.15, 0.4]]),
    }


def _result(inputs):
    return {
        "energy": -0.125,
        "energy_standard_error": 0.01,
        "cotangents": {name: np.ones_like(value) for name, value in inputs.items()},
        "diagnostics": {},
    }


def test_weighted_inputs_accept_raw_weights_and_empty_virtuals():
    for nvir in (0, 2):
        inputs = _inputs(nvir)
        protocol.validate_weighted_inputs(inputs)
        protocol.validate_weighted_result(_result(inputs), inputs)


@pytest.mark.parametrize("name,value,match", [
    ("target_projection", np.ones((2, 2)), "shape"),
    ("partner_weight", np.eye(3), "shape"),
    ("partner_weight", np.array([[1., .2], [.1, 1.]]), "symmetric"),
    ("foo", np.eye(2, dtype=np.float32), "float64"),
    ("B", np.full((3, 2, 2), np.nan), "finite"),
    ("target_projection", np.ones((1, 2), dtype=np.complex128), "float64"),
])
def test_weighted_inputs_reject_invalid_physical_arrays(name, value, match):
    inputs = _inputs()
    inputs[name] = value
    with pytest.raises(ValueError, match=match):
        protocol.validate_weighted_inputs(inputs)


def test_weighted_contract_rejects_missing_keys_and_empty_occupied_space():
    inputs = _inputs()
    inputs.pop("partner_weight")
    with pytest.raises(ValueError, match="exactly"):
        protocol.validate_weighted_inputs(inputs)
    empty = {"foo": np.zeros((0, 0)), "fvv": np.eye(2),
             "B": np.zeros((3, 0, 2)), "target_projection": np.zeros((1, 0)),
             "partner_weight": np.zeros((0, 0))}
    with pytest.raises(ValueError, match="occupied"):
        protocol.validate_weighted_inputs(empty)


def test_energy_only_result_permits_omitted_bars_but_gradient_requires_all_five():
    inputs = _inputs()
    result = _result(inputs)
    result.pop("cotangents")
    protocol.validate_weighted_result(result, inputs, with_grad=False)
    with pytest.raises(ValueError, match="cotangent"):
        protocol.validate_weighted_result(result, inputs)
    result = _result(inputs)
    result["cotangents"].pop("target_projection")
    with pytest.raises(ValueError, match="exactly"):
        protocol.validate_weighted_result(result, inputs)


@pytest.mark.parametrize("field,value,match", [
    ("energy", np.inf, "energy"),
    ("energy_standard_error", -0.1, "standard_error"),
    ("energy_standard_error", np.nan, "standard_error"),
    ("energy_standard_error", np.array([0.1]), "scalar"),
    ("diagnostics", [], "dictionary"),
])
def test_weighted_result_rejects_invalid_estimates(field, value, match):
    inputs = _inputs()
    result = _result(inputs)
    result[field] = value
    with pytest.raises(ValueError, match=match):
        protocol.validate_weighted_result(result, inputs)


def test_weighted_result_rejects_wrong_shape_and_nonsymmetric_bars():
    inputs = _inputs()
    result = _result(inputs)
    result["cotangents"]["B"] = np.zeros((3, 2, 1))
    with pytest.raises(ValueError, match="shape"):
        protocol.validate_weighted_result(result, inputs)
    result = _result(inputs)
    result["cotangents"]["partner_weight"][0, 1] = 0
    with pytest.raises(ValueError, match="symmetric"):
        protocol.validate_weighted_result(result, inputs)


def test_preparation_keeps_unnormalized_target_and_partner_projections(finite_water_dimer):
    mf, common, static = finite_water_dimer
    # Shrink the original target to catch normalization or an identity weight.
    records = list(common.fragment_occupied_data)
    records[0] = records[0]._replace(iao_coeff=0.8 * records[0].iao_coeff)
    current = common._replace(fragment_occupied_data=tuple(records))
    anchors = prepare.select_virtual_anchor_columns(current, static, 0)
    inputs = prepare.prepare_weighted_inputs(
        mf, current, static, 0, virtual_anchor_columns=anchors,
    )
    protocol.validate_weighted_inputs(inputs)
    domain = prepare.build_local_strong_ed_domain(
        current, static, 0, virtual_anchor_columns=anchors,
    )
    fragment = static.fragments[0]
    co = np.asarray(domain.occupied_coeff)
    ao = np.asarray(fragment.extended_ao_indices)
    overlap_to_local = np.asarray(current.s1e)[:, ao] @ co
    target = np.asarray(records[0].iao_coeff).T @ overlap_to_local
    partners = np.concatenate([
        np.asarray(records[int(index)].iao_coeff) for index in fragment.strong_fragments
    ], axis=1).T @ overlap_to_local
    np.testing.assert_allclose(inputs["target_projection"], target, atol=1e-10)
    np.testing.assert_allclose(inputs["partner_weight"], partners.T @ partners, atol=1e-10)
    assert np.linalg.norm(target) < 0.81
    assert not np.allclose(inputs["partner_weight"], np.eye(co.shape[1]))
    for name in ("foo", "fvv", "B"):
        np.testing.assert_allclose(inputs[name], prepare.prepare_finite_inputs(
            mf, current, static, 0, virtual_anchor_columns=anchors,
        )[name], atol=1e-10)
    traced, _ = jax.vjp(lambda data: prepare.prepare_weighted_inputs(
        mf, data, static, 0, virtual_anchor_columns=anchors,
    ), current)
    for name in inputs:
        np.testing.assert_allclose(traced[name], inputs[name], atol=1e-10)
    with pytest.raises(ValueError, match="anchor"):
        jax.vjp(lambda data: prepare.prepare_weighted_inputs(mf, data, static, 0), current)


def test_immediate_pullback_scales_all_five_bars_and_preserves_primal_tree():
    primal = {"x": jnp.array([0.4, -0.2]), "y": (jnp.array(0.3, dtype=jnp.float32),)}

    def make_inputs(data):
        x = data["x"]
        y = data["y"][0].astype(jnp.float64)
        return {
            "foo": jnp.diag(x), "fvv": jnp.eye(2) * y,
            "B": (x[None, :, None] + y) * jnp.ones((3, 2, 2)),
            "target_projection": (x * y)[None, :],
            "partner_weight": jnp.outer(x, x),
        }

    inputs, pullback = jax.vjp(make_inputs, primal)
    bars = {name: np.full(value.shape, index + 1., dtype=np.float64)
            for index, (name, value) in enumerate(inputs.items())}
    result = _result(inputs)
    result["cotangents"] = bars
    seed = -0.7
    (actual,) = adjoint.apply_weighted_pullback(pullback, inputs, result, energy_bar=seed)
    expected = jax.grad(lambda data: seed * sum(
        jnp.sum(value * bars[name]) for name, value in make_inputs(data).items()
    ))(primal)
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(primal)
    for actual_leaf, expected_leaf, primal_leaf in zip(
        jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected),
        jax.tree_util.tree_leaves(primal),
    ):
        assert actual_leaf.dtype == primal_leaf.dtype
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-7, atol=1e-7)
    for invalid in (np.nan, np.array([1.])):
        with pytest.raises(ValueError, match="scalar|finite"):
            adjoint.apply_weighted_pullback(pullback, inputs, result, energy_bar=invalid)
