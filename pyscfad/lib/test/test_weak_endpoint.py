"""Keep the forward orbital linearization while replaying only AO moments."""
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('gap', [0.2, 1e-7])
def test_saved_screen_pullback_matches_direct_derivative_without_rebuilding(gap):
    from pyscfad.lib import _weak_endpoint

    mf = jnp.asarray(.7)
    common = jnp.asarray(.31)
    calls = Counter()
    differentiated_moments = []

    def screen(value):
        matrix = jnp.array([[1. + value, .03 * gap * value],
                            [.03 * gap * value, 1. + value + gap]])
        energy, orbitals = jnp.linalg.eigh(matrix)
        return energy, orbitals, jnp.array([.4, .9]) + value**2

    def moments(m, saved):
        energy, orbitals, weights = saved
        return (jnp.sin(m) * orbitals, energy + m**2, weights * (m + 1.))

    def counted_screen(value):
        calls['screen'] += 1
        assert calls['screen'] == 1, 'the forward orbital frame was rebuilt'
        return screen(value)

    def counted_moments(m, saved):
        calls['moments'] += 1
        differentiated_moments.append(isinstance(m, jax.core.Tracer))
        return moments(m, saved)

    reference, direct_pullback = jax.vjp(lambda m, c: moments(m, screen(c)), mf, common)
    bars = jax.tree_util.tree_map(lambda value: jnp.arange(value.size).reshape(value.shape) + .2,
                                 reference)
    endpoint, linearization = _weak_endpoint.prepare_endpoint(
        counted_screen, counted_moments, mf, common)
    rebuilt, pullback = _weak_endpoint.replay_endpoint(counted_moments, mf, linearization)
    for actual in (endpoint, rebuilt):
        for got, want in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(reference)):
            np.testing.assert_array_equal(got, want)
    for got, want in zip(pullback(bars), direct_pullback(bars)):
        np.testing.assert_allclose(got, want, rtol=2e-12, atol=2e-12)
    assert calls == Counter(screen=1, moments=2)
    assert differentiated_moments == [False, True]


def test_orbital_pullback_retains_response_of_discarded_space():
    from pyscfad.lib import _weak_endpoint

    common = jnp.asarray(.15)

    def screen(value):
        matrix = jnp.array([[1., value, .2], [value, 2., .1], [.2, .1, 4.]])
        _, vectors = jnp.linalg.eigh(matrix)
        return vectors[:, :2]

    def moments(m, saved):
        return m * saved

    endpoint, linearization = _weak_endpoint.prepare_endpoint(screen, moments, jnp.asarray(.8), common)
    _, pullback = _weak_endpoint.replay_endpoint(moments, jnp.asarray(.8), linearization)
    _, common_bar = pullback(jnp.ones_like(endpoint))
    direct = jax.grad(lambda c: jnp.sum(moments(.8, screen(c))))(common)
    assert abs(float(direct)) > 1e-3
    np.testing.assert_allclose(common_bar, direct, rtol=2e-12, atol=2e-12)
