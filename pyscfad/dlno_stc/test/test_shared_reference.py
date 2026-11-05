"""Fresh selection shares numerical reference work without freezing response."""
from collections import Counter

import jax
import numpy as np
import pytest

from pyscfad.dlno_stc import build_domain_selections, domain, driver
from pyscfad.dlno_stc import _selection
from pyscfad.lo import boys
from .test_boys_prepare import boys_water_dimer


def test_fresh_selection_localizes_and_builds_pao_products_once(
        boys_water_dimer, monkeypatch):
    mf, _, _ = boys_water_dimer
    counts = Counter()

    def count(module, name):
        original = getattr(module, name)
        def wrapped(*args, **kwargs):
            counts[name] += 1
            return original(*args, **kwargs)
        monkeypatch.setattr(module, name, wrapped)

    count(boys, '_boys')
    count(domain, '_normalized_pao')
    count(domain, '_pao_products')
    count(_selection, '_reference_pao_products')
    static, _, _, receipt = driver._select_domain_reference(
        mf, lambda reference: build_domain_selections(reference, frozen=2))
    assert counts['_boys'] == 1, counts
    # The fused reference builder replaces both of these separate PAO paths.
    assert counts['_normalized_pao'] == 0, counts
    assert counts['_reference_pao_products'] == 0, counts
    assert counts['_pao_products'] == 0, counts
    common, pullback, pending = receipt.take(static)
    assert common is not None and callable(pullback)
    assert pending is not None and pending[0]
    assert receipt.products_preparation is None
    assert receipt.take(static) is None
    assert not hasattr(mf, '_stc_selection_response')
    assert not hasattr(static, 'common_pullback')


def test_shared_reference_retains_full_common_to_mf_response(boys_water_dimer):
    mf, _, _ = boys_water_dimer
    static, data, data_pb, receipt = driver._select_domain_reference(
        mf, lambda reference: build_domain_selections(reference, frozen=2,
            pair_energy_model='all', lo_kwargs={'conv_tol': 1e-12,
                'conv_tol_grad': 1e-10, 'max_cycle': 300}))
    common, pullback, pending = receipt.take(static)
    replay, replay_pb = driver._build_common_with_pullback(mf, static, data, data_pb)
    for actual, expected in zip(jax.tree_util.tree_leaves(common),
                                jax.tree_util.tree_leaves(replay)):
        np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-10)
    rng = np.random.default_rng(704)
    bar = jax.tree_util.tree_map(lambda x: rng.normal(size=x.shape), common)
    actual, = pullback(bar)
    expected, = replay_pb(bar)
    for a, b in zip(jax.tree_util.tree_leaves(actual),
                    jax.tree_util.tree_leaves(expected)):
        if a.dtype != jax.dtypes.float0:
            np.testing.assert_allclose(a, b, atol=2e-8, rtol=2e-9)
    assert pending is None


def test_shared_screening_matches_fresh_weak_response_with_permuted_targets(
        boys_water_dimer):
    from pyscfad.dlno_stc._pairs import weak_pair_energy_and_grad
    mf, _, reference = boys_water_dimer
    groups = tuple(reversed(reference.frag_lolist))
    static, _, _, receipt = driver._select_domain_reference(
        mf, lambda current: build_domain_selections(current, frozen=2,
                                                     frag_lolist=groups))
    assert receipt.products_preparation is None
    common, _, (pairs, actual) = receipt.take(static)
    expected = weak_pair_energy_and_grad(mf, common, static, pairs=pairs)
    for a, b in zip(jax.tree_util.tree_leaves(actual),
                    jax.tree_util.tree_leaves(expected)):
        if a.dtype != jax.dtypes.float0:
            np.testing.assert_allclose(a, b, atol=2e-9, rtol=2e-8)
    np.testing.assert_array_equal(static.frag_lolist, groups)


@pytest.mark.parametrize('fallback', ['explicit', 'changed', 'disabled'])
def test_uncertified_references_preserve_ordinary_selection(
        boys_water_dimer, monkeypatch, fallback):
    mf, _, reference = boys_water_dimer
    builds = []
    original = domain.build_scf_data
    def build_data(current):
        builds.append(current)
        return original(current)
    monkeypatch.setattr(domain, 'build_scf_data', build_data)
    def forbidden(*args, **kwargs):
        raise AssertionError('uncertified reference attempted live sharing')
    monkeypatch.setattr(domain, '_build_selection_reference', forbidden)

    def select(current):
        options = {}
        if fallback == 'explicit':
            options['target_coeff'] = reference.target_reference_coeff
        elif fallback == 'changed':
            current.mo_energy = current.mo_energy + .001
        return build_domain_selections(current, frozen=2,
                                       pair_energy_model='all', **options)

    selected, _, _, receipt = driver._select_domain_reference(
        mf, select, reuse_weak=fallback != 'disabled')
    assert selected is not None
    assert len(builds) == (1 if fallback == 'changed' else 0)
    if receipt is not None:
        assert receipt.common is None and receipt.products_preparation is None
        assert receipt.take(selected) is None
