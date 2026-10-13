"""Endpoint phase alignment keeps the primal and its VJP in one frame."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.lib import _multipole


def _endpoint(x, order):
    mu = jnp.arange(1., 13.).reshape(3, 4) * x
    theta = jnp.arange(1., 37.).reshape(3, 3, 4) * x**2 if order > 2 else None
    omega = jnp.arange(1., 109.).reshape(3, 3, 3, 4) * jnp.sin(x) if order > 3 else None
    return jnp.array([x, x**2, -x]), mu, jnp.arange(1., 5.) + x, theta, omega


def _rephase(endpoint, signs):
    center, mu, gap, theta, omega = endpoint
    return (center, mu*signs, gap,
            None if theta is None else theta*signs,
            None if omega is None else omega*signs)


@pytest.mark.parametrize('order', [2, 3, 4])
def test_alignment_inside_vjp_preserves_input_cotangent(order):
    x = jnp.asarray(.7)
    reference, expected_pullback = jax.vjp(lambda value: _endpoint(value, order), x)
    signs = jnp.array([1., -1., -1., 1.])

    def replay(value):
        raw = _rephase(_endpoint(value, order), signs)
        return _multipole.align_endpoint_signs(raw, reference)

    aligned, actual_pullback = jax.vjp(replay, x)
    for actual, expected in zip(jax.tree_util.tree_leaves(aligned),
                                jax.tree_util.tree_leaves(reference)):
        np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)
    bar = jax.tree_util.tree_map(lambda leaf: jnp.ones_like(leaf)*.13, reference)
    np.testing.assert_allclose(actual_pullback(bar), expected_pullback(bar),
                               atol=1e-13, rtol=1e-13)


def test_zero_moment_columns_remain_finite_and_keep_invariant_fields():
    reference = (jnp.array([1., 2., 3.]), jnp.zeros((3, 2)),
                 jnp.array([.5, .8]), jnp.zeros((3, 3, 2)), None)
    aligned = _multipole.align_endpoint_signs(reference, reference)
    for actual, expected in zip(jax.tree_util.tree_leaves(aligned),
                                jax.tree_util.tree_leaves(reference)):
        np.testing.assert_array_equal(actual, expected)
        assert np.isfinite(actual).all()


@pytest.mark.parametrize('change', ['shape', 'structure'])
def test_malformed_endpoint_is_left_for_callers_validation(change):
    reference = _endpoint(jnp.asarray(.7), 4)
    malformed = list(reference)
    malformed[1] = malformed[1][:, :2] if change == 'shape' else None
    malformed = tuple(malformed)
    result = _multipole.align_endpoint_signs(malformed, reference)
    assert jax.tree_util.tree_structure(result) == jax.tree_util.tree_structure(malformed)
    for actual, expected in zip(jax.tree_util.tree_leaves(result),
                                jax.tree_util.tree_leaves(malformed)):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('slot', [1, 3, 4])
def test_replay_check_scales_cancelled_components_by_their_own_moment_column(slot):
    reference = [np.array(leaf) for leaf in _endpoint(jnp.asarray(.7), 4)]
    reference[slot][..., 0] = 0.
    reference[slot].reshape(-1, 4)[0, 0] = 10.
    rebuilt = [leaf.copy() for leaf in reference]
    rebuilt[slot].reshape(-1, 4)[1, 0] = 2.5e-9
    assert _multipole.endpoint_matches_reference(tuple(rebuilt), tuple(reference))


@pytest.mark.parametrize('slot', [1, 3, 4])
def test_replay_check_preserves_component_sized_absolute_roundoff(slot):
    reference = [np.array(leaf) for leaf in _endpoint(jnp.asarray(.7), 4)]
    reference[slot][:] = 0.
    rebuilt = [leaf.copy() for leaf in reference]
    rebuilt[slot][:] = .99e-9
    assert _multipole.endpoint_matches_reference(tuple(rebuilt), tuple(reference))


@pytest.mark.parametrize('large_field', ['other_column', 'higher_moment'])
def test_replay_check_does_not_borrow_scale_from_other_columns_or_orders(large_field):
    reference = [np.array(leaf) for leaf in _endpoint(jnp.asarray(.7), 4)]
    reference[1][:] = 0.
    if large_field == 'other_column':
        reference[1][:, 1] = 1e6
    else:
        reference[4][:] = 1e6
    rebuilt = [leaf.copy() for leaf in reference]
    rebuilt[1][0, 0] = 1e-6
    assert not _multipole.endpoint_matches_reference(tuple(rebuilt), tuple(reference))


@pytest.mark.parametrize('invalid', [np.nan, np.inf])
def test_replay_check_rejects_nonfinite_moments(invalid):
    reference = [np.array(leaf) for leaf in _endpoint(jnp.asarray(.7), 4)]
    reference[4].flat[0] = invalid
    assert not _multipole.endpoint_matches_reference(tuple(reference), tuple(reference))


def test_alignment_uses_higher_moments_when_dipoles_vanish():
    reference = list(_endpoint(jnp.asarray(.7), 4))
    reference[1] = jnp.zeros_like(reference[1])
    reference[3] = jnp.zeros_like(reference[3])
    reference = tuple(reference)
    aligned = _multipole.align_endpoint_signs(
        _rephase(reference, jnp.array([-1., 1., -1., 1.])), reference)
    for actual, expected in zip(jax.tree_util.tree_leaves(aligned),
                                jax.tree_util.tree_leaves(reference)):
        np.testing.assert_array_equal(actual, expected)
