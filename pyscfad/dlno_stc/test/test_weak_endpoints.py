"""Weak response shares endpoint work while retaining the full chain rule."""
from collections import Counter
from types import SimpleNamespace
from typing import NamedTuple
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import _pairs


class Inputs(NamedTuple):
    coordinates: object
    scale: object


class Common(NamedTuple):
    s1e: object


def _fixture(order):
    mf = Inputs(jnp.array([[0., .1, .2], [.2, .1, -.1], [-.1, .3, .2]]), jnp.array(.7))
    common = Common(jnp.array([.2, .4, .6]))
    strong = np.ones((4, 4), dtype=bool)
    for left, right in ((0, 1), (0, 2), (1, 2)):
        strong[left, right] = strong[right, left] = False
    static = SimpleNamespace(fragments=(None,) * 4, strong_mask=strong,
                             thresholds=SimpleNamespace(multipole_order=order))

    def endpoint(m, c, s, index):
        center = m.coordinates[index] + jnp.array([5. * index, .2 * index, 0.])
        mu = jnp.arange(1., 7.).reshape(3, 2) * (.03 + .01 * c.s1e[index])
        gap = jnp.array([.6, .9]) + .01 * m.scale + .02 * c.s1e[index]
        theta = jnp.arange(18.).reshape(3, 3, 2) * (.001 + .0001 * m.scale) if order > 2 else None
        omega = jnp.arange(54.).reshape(3, 3, 3, 2) * (.0001 + .00001 * c.s1e[index]) if order > 3 else None
        return center, mu, gap, theta, omega

    return mf, common, static, endpoint


@pytest.mark.parametrize('order', [2, 3, 4])
def test_staged_response_matches_direct_chain_rule_with_shared_endpoints(monkeypatch, order):
    build = getattr(_pairs, 'build_weak_endpoint', None)
    assert callable(build), 'weak response needs a reusable endpoint boundary'
    mf, common, static, endpoint = _fixture(order)
    indices = _pairs.weak_pair_indices(static)

    def direct(m, c):
        return sum(_pairs._multipole_pair_energy(endpoint(m, c, static, i),
                   endpoint(m, c, static, j), order) for i, j in indices)

    expected, expected_bars = jax.value_and_grad(direct, argnums=(0, 1))(mf, common)
    calls = Counter()

    def counted(m, c, s, index):
        calls[index] += 1
        return endpoint(m, c, s, index)

    monkeypatch.setattr(_pairs, 'build_weak_endpoint', counted)
    actual, actual_bars = _pairs.weak_pair_energy_and_grad(mf, common, static)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-14)
    for got, want in zip(jax.tree_util.tree_leaves(actual_bars), jax.tree_util.tree_leaves(expected_bars)):
        np.testing.assert_allclose(got, want, rtol=2e-11, atol=1e-13)
    assert calls == Counter({0: 2, 1: 2, 2: 2})  # value pass, then one bounded VJP replay


@pytest.mark.parametrize('order', [2, 3, 4])
def test_forward_weak_sum_builds_each_incident_endpoint_once(monkeypatch, order):
    total = getattr(_pairs, 'weak_pair_energy_sum', None)
    assert callable(total), 'weak forward energy needs endpoint reuse'
    mf, common, static, endpoint = _fixture(order)
    calls = Counter()

    def counted(m, c, s, index):
        calls[index] += 1
        return endpoint(m, c, s, index)

    monkeypatch.setattr(_pairs, 'build_weak_endpoint', counted)
    actual = total(mf, common, static)
    expected = sum(_pairs._multipole_pair_energy(endpoint(mf, common, static, i),
                   endpoint(mf, common, static, j), order) for i, j in _pairs.weak_pair_indices(static))
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-14)
    assert calls == Counter({0: 1, 1: 1, 2: 1})


def test_empty_weak_graph_has_zero_response(monkeypatch):
    mf, common, static, _ = _fixture(4)
    static.strong_mask[:] = True

    def forbidden(*args):
        raise AssertionError('an all-strong graph built a weak endpoint')

    monkeypatch.setattr(_pairs, 'build_weak_endpoint', forbidden, raising=False)
    monkeypatch.setattr(_pairs.gc, 'collect', forbidden)
    value, bars = _pairs.weak_pair_energy_and_grad(mf, common, static)
    assert float(value) == 0.
    for leaf in jax.tree_util.tree_leaves(bars):
        np.testing.assert_array_equal(leaf, np.zeros_like(leaf))


@pytest.mark.parametrize('order', [2, 3, 4])
def test_endpoint_replay_aligns_virtual_signs_inside_pullback(monkeypatch, order):
    mf, common, static, endpoint = _fixture(order)
    pairs = _pairs.weak_pair_indices(static)
    calls = Counter()

    def direct(m, c):
        return sum(_pairs._multipole_pair_energy(endpoint(m, c, static, i),
                   endpoint(m, c, static, j), order) for i, j in pairs)

    expected, expected_bars = jax.value_and_grad(direct, argnums=(0, 1))(mf, common)

    def changed_signs(m, c, s, index):
        calls[index] += 1
        value = endpoint(m, c, s, index)
        if calls[index] > 1:
            center, mu, gap, theta, omega = value
            signs = jnp.array([-1., 1.]) if index % 2 == 0 else jnp.array([1., -1.])
            return (center, mu * signs, gap,
                    None if theta is None else theta * signs,
                    None if omega is None else omega * signs)
        return value

    monkeypatch.setattr(_pairs, 'build_weak_endpoint', changed_signs)
    actual, actual_bars = _pairs.weak_pair_energy_and_grad(mf, common, static)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-14)
    for got, want in zip(jax.tree_util.tree_leaves(actual_bars),
                         jax.tree_util.tree_leaves(expected_bars)):
        np.testing.assert_allclose(got, want, rtol=2e-11, atol=1e-13)
    assert calls == Counter({0: 2, 1: 2, 2: 2})


@pytest.mark.parametrize('change', ['gap', 'moment', 'rotation'])
def test_endpoint_replay_rejects_changes_beyond_virtual_signs(monkeypatch, change):
    mf, common, static, endpoint = _fixture(4)
    calls = Counter()

    def changed_endpoint(m, c, s, index):
        calls[index] += 1
        value = endpoint(m, c, s, index)
        if index == 0 and calls[index] > 1:
            center, mu, gap, theta, omega = value
            if change == 'gap':
                gap = gap + .1
            elif change == 'moment':
                mu = mu.at[0, 0].add(.1)
            else:
                angle = .2
                rotation = jnp.array([[jnp.cos(angle), -jnp.sin(angle)],
                                      [jnp.sin(angle), jnp.cos(angle)]])
                mu, theta, omega = (moment @ rotation for moment in (mu, theta, omega))
            return center, mu, gap, theta, omega
        return value

    monkeypatch.setattr(_pairs, 'build_weak_endpoint', changed_endpoint)
    with pytest.raises(RuntimeError, match='endpoint gauge changed on replay'):
        _pairs.weak_pair_energy_and_grad(mf, common, static)


def test_endpoint_replay_accepts_c20_near_zero_moment_roundoff():
    mf, common, static, endpoint = _fixture(4)
    original = list(endpoint(mf, common, static, 0))
    replay = list(original)
    # Measured eager/VJP components from the C20 endpoint-0 reproduction.
    # Its virtual-frame overlap agreed with identity to 2.3e-10.
    for slot, saved, rebuilt in (
        (1, -4.52621069e-9, -4.58022677e-9),
        (3, 2.44461657e-9, 2.47384181e-9),
        (4, -3.82001262e-8, -3.86564252e-8),
    ):
        original[slot] = np.array(original[slot], copy=True)
        replay[slot] = original[slot].copy()
        original[slot].flat[0] = saved
        replay[slot].flat[0] = rebuilt
    _pairs._check_endpoint_replay(tuple(replay), tuple(original), 0)


def test_endpoint_replay_accepts_cancelled_octupole_component_roundoff():
    mf, common, static, endpoint = _fixture(4)
    original = list(endpoint(mf, common, static, 0))
    original[4] = np.zeros_like(original[4])
    # C20 endpoint 58 has a moment-column norm about 2.91, but a near-zero
    # Cartesian component can change by 2.47e-9 between eager and VJP builds.
    original[4][0, 0, 0, 0] = 2.91
    replay = list(original)
    replay[4] = original[4].copy()
    replay[4][0, 0, 1, 0] = 2.47e-9
    _pairs._check_endpoint_replay(tuple(replay), tuple(original), 58)


@pytest.mark.parametrize('with_grad', [False, True])
def test_domain_collects_once_after_complete_weak_response(monkeypatch, with_grad):
    from pyscfad.dlno_stc import driver
    from pyscfad.dlno_stc.adjoint import zero_term_cotangents
    mf, common, static, endpoint = _fixture(4)
    pairs = _pairs.weak_pair_indices(static)
    calls, collections = Counter(), []
    compact_values = []

    def direct(m, c):
        return sum(_pairs._multipole_pair_energy(endpoint(m, c, static, i),
                   endpoint(m, c, static, j), 4) for i, j in pairs)

    expected, expected_bars = jax.value_and_grad(direct, argnums=(0, 1))(mf, common)

    def counted(m, c, s, index):
        calls[index] += 1
        result = endpoint(m, c, s, index)
        if calls[index] == 1:
            compact_values.append(weakref.ref(result[0]))
        return result

    def collect():
        collections.append(calls.copy())
        assert all(value() is None for value in compact_values)

    def initialize():
        mf_bar, common_bar = zero_term_cotangents(mf, common)
        return dict(energy=jnp.asarray(0.), variance=0., weak_pairs=pairs,
                    mf=mf, common=common, mf_bar=mf_bar, common_bar=common_bar)

    monkeypatch.setattr(_pairs, 'build_weak_endpoint', counted)
    monkeypatch.setattr(driver, 'build_domain_groups', lambda *a, **k: ())
    monkeypatch.setattr(driver, '_finish', lambda state, **k:
                        (state['energy'], (state['mf_bar'], state['common_bar'])))
    monkeypatch.setattr(_pairs.gc, 'collect', collect)
    actual, actual_bars = driver._run_domain(
        initialize, static, {}, None, with_grad=with_grad)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-14)
    if with_grad:
        for got, want in zip(jax.tree_util.tree_leaves(actual_bars),
                             jax.tree_util.tree_leaves(expected_bars)):
            np.testing.assert_allclose(got, want, rtol=2e-11, atol=1e-13)
    assert collections == [Counter({0: 2 if with_grad else 1,
                                    1: 2 if with_grad else 1,
                                    2: 2 if with_grad else 1})]
