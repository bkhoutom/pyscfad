"""AO-moment replay checks compare each virtual column independently."""
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.lib import _multipole


def _endpoint(x, order):
    mu = jnp.arange(1., 13.).reshape(3, 4) * x
    theta = jnp.arange(1., 37.).reshape(3, 3, 4) * x**2 if order > 2 else None
    omega = jnp.arange(1., 109.).reshape(3, 3, 3, 4) * jnp.sin(x) if order > 3 else None
    return jnp.array([x, x**2, -x]), mu, jnp.arange(1., 5.) + x, theta, omega


@pytest.mark.parametrize('change', ['shape', 'structure'])
def test_replay_check_rejects_changed_shapes_and_structure(change):
    reference = _endpoint(jnp.asarray(.7), 4)
    malformed = list(reference)
    malformed[1] = malformed[1][:, :2] if change == 'shape' else None
    malformed = tuple(malformed)
    assert not _multipole.endpoint_matches_reference(malformed, reference)


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


def test_zero_moment_columns_match_without_normalization():
    reference = (jnp.array([1., 2., 3.]), jnp.zeros((3, 2)),
                 jnp.array([.5, .8]), jnp.zeros((3, 3, 2)), None)
    assert _multipole.endpoint_matches_reference(reference, reference)
