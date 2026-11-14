"""Selection reuses primary spaces without changing screening or domain labels."""
from dataclasses import replace
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import _pairs, _selection, domain
from pyscfad.dlno_stc.test.test_boys_prepare import boys_water_dimer


@pytest.mark.parametrize('grouping,order', [('singleton', 2), ('singleton', 4), ('atom', 4)])
def test_primary_reuse_matches_replay_and_releases_ao_operators(
    boys_water_dimer, monkeypatch, grouping, order,
):
    mf, _, reference = boys_water_dimer
    original_screen = _pairs.screen_pairs
    results = []
    def replay(*args, **kwargs):
        kwargs.pop('_reference_endpoints', None)
        result = original_screen(*args, **kwargs)
        results.append(result)
        return result

    kwargs = dict(frozen=reference.frozen,
                  target_coeff=reference.target_reference_coeff,
                  thresholds=replace(reference.thresholds, multipole_order=order),
                  boys_grouping=grouping)
    monkeypatch.setattr(_pairs, 'screen_pairs', replay)
    expected = _selection.build_domain_selections(mf, **kwargs)

    builds, live = [], []
    original_moments = _pairs._origin_zero_moments
    def moments(mol, max_order):
        assert all(ref() is None for ref in live), 'previous AO support retained'
        value = original_moments(mol, max_order)
        builds.append(1)
        live[:] = [weakref.ref(x) for x in value if x is not None]
        return value

    def reused(*args, **kwargs):
        assert kwargs['_reference_endpoints'] is not None
        assert all(ref() is None for ref in live), 'raw operators retained into pair loop'
        result = original_screen(*args, **kwargs)
        results.append(result)
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError('forward selection rebuilt its primary virtuals')

    monkeypatch.setattr(_pairs, 'screen_pairs', reused)
    monkeypatch.setattr(_pairs, '_origin_zero_moments', moments)
    monkeypatch.setattr(domain, '_rebuild_selected_pao_preprojection', forbidden)
    actual = _selection.build_domain_selections(mf, **kwargs)
    supports = {tuple(f.primary_atoms) for f in actual.fragments if f.weak_virtual is not None}
    assert len(builds) == len(supports)
    for got, want in zip(results[-1][:2], results[0][:2]):
        np.testing.assert_allclose(got, want, atol=2e-12, rtol=2e-10)
    for got, want in zip(results[-1][2:], results[0][2:]):
        np.testing.assert_array_equal(got, want)
    for got, want in zip(actual.fragments, expected.fragments):
        for name in ('strong_fragments', 'extended_atoms', 'extended_ao_indices',
                     'virtual_anchor_columns'):
            np.testing.assert_array_equal(getattr(got, name), getattr(want, name))
        for space in ('weak_virtual', 'strong_virtual'):
            if getattr(got, space) is None:
                assert getattr(want, space) is None
                continue
            for name in getattr(got, space).__dataclass_fields__:
                np.testing.assert_array_equal(getattr(getattr(got, space), name),
                                              getattr(getattr(want, space), name))


def test_reference_endpoints_cannot_replace_live_response():
    with pytest.raises(ValueError, match='cannot replace response'):
        _pairs.screen_pairs(None, None, None, prepared_endpoints={}, _reference_endpoints={})


def test_reused_overlap_solve_preserves_pao_and_overlap_response(monkeypatch):
    from pyscfad.dlno_stc.test.test_pao_response_reuse import _fixture
    _, common, static = _fixture(monkeypatch)
    ao = static.fragments[0].primary_ao_indices
    selected = replace(static.fragments[0].weak_virtual, support_ao_indices=ao)
    # Reordering the support preserves the mathematical projection but forces
    # the separate representation solve, providing an independent comparison.
    separate = replace(selected, support_ao_indices=ao[::-1].copy())
    weight = jnp.asarray(np.random.default_rng(103).normal(size=(len(ao), len(ao))))
    def objective(current, labels):
        coeff = domain._rebuild_selected_pao_preprojection(
            current, selection=labels, representation_ao_indices=ao)
        return jnp.sum(weight * (coeff @ coeff.T))
    value, bars = jax.value_and_grad(objective)(common, selected)
    expected, expected_bars = jax.value_and_grad(objective)(common, separate)
    np.testing.assert_allclose(value, expected, atol=2e-12, rtol=2e-11)
    for got, want in zip(jax.tree_util.tree_leaves(bars), jax.tree_util.tree_leaves(expected_bars)):
        np.testing.assert_allclose(got, want, atol=2e-11, rtol=2e-10)
    rng = np.random.default_rng(104)
    ds = rng.normal(size=common.s1e.shape)
    ds = jnp.asarray((ds + ds.T) * .01)
    dp = jnp.asarray(rng.normal(size=common.pao_coeff.shape) * .01)
    step = 1e-5
    plus = common._replace(s1e=common.s1e + step*ds, pao_coeff=common.pao_coeff + step*dp)
    minus = common._replace(s1e=common.s1e - step*ds, pao_coeff=common.pao_coeff - step*dp)
    finite = (objective(plus, selected) - objective(minus, selected)) / (2*step)
    analytic = jnp.sum(bars.s1e * ds) + jnp.sum(bars.pao_coeff * dp)
    np.testing.assert_allclose(analytic, finite, atol=2e-9, rtol=2e-7)
