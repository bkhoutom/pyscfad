"""Primary selection retains its own live response instead of rebuilding."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import _selection, _pairs, domain, driver
from pyscfad.dlno_stc._primary_response import prepare_primary
from pyscfad.dlno_stc.test.test_boys_prepare import boys_water_dimer


def _compare(actual, expected, atol=2e-9):
    for got, want in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        if got.dtype != jax.dtypes.float0:
            np.testing.assert_allclose(got, want, atol=atol, rtol=2e-8)


@pytest.mark.parametrize('grouping,method', [('singleton', 'boys'), ('atom', 'boys'),
                                           ('singleton', 'lambda_boys')])
def test_fresh_selection_never_rebuilds_primary_spaces(boys_water_dimer, monkeypatch, grouping, method):
    mf, _, _ = boys_water_dimer
    original = _selection._reference_pao_overlap_selection
    def strong_only(*args, **kwargs):
        assert kwargs['completeness_threshold'] != .98, 'duplicate detached primary selection'
        return original(*args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError('primary selection was replayed to build its derivative')
    monkeypatch.setattr(_selection, '_reference_pao_overlap_selection', strong_only)
    monkeypatch.setattr(domain, '_rebuild_selected_pao_preprojection', forbidden)
    static, _, _, receipt = driver._select_domain_reference(
        mf, lambda current: _selection.build_domain_selections(
            current, frozen=2, boys_grouping=grouping, lo_kwargs={'method': method}))
    common, _, pending = receipt.take(static)
    assert common is not None and pending is not None
    assert np.isfinite(pending[1][0])


@pytest.mark.parametrize('order', [2, 4])
def test_shared_primary_selection_matches_independent_replay(boys_water_dimer, order):
    mf, common, static = boys_water_dimer
    static = replace(static, thresholds=replace(static.thresholds, multipole_order=order))
    products = _pairs._cached_weak_screen_products(common, _pairs._weak_screen_products(common))
    preparations, cache = {}, {}
    fragments = []
    for i, fragment in enumerate(static.fragments):
        ao, norm, span, selected, preparation = prepare_primary(
            common, products, mf.mol, static.ao2pao_map, static.thresholds, i,
            static.primary_bp_domain[i], static.primary_domain[i], cache)
        np.testing.assert_array_equal(norm, fragment.weak_occ_norm_keep)
        np.testing.assert_array_equal(span, fragment.weak_occ_span_metric_keep)
        for name in selected.__dataclass_fields__:
            np.testing.assert_array_equal(getattr(selected, name), getattr(fragment.weak_virtual, name))
        fragments.append(replace(fragment, weak_virtual=selected))
        preparations[i] = preparation
    static = replace(static, fragments=tuple(fragments))
    prepared, response = {}, {}
    actual_scores = _pairs.screen_pairs(
        mf, common, static, prepared_endpoints=prepared, _primary_preparations=preparations,
        _products_preparation=products, _pair_response=response)
    assert not preparations
    reference_scores = _pairs.screen_pairs(mf, common, static)
    for got, want in zip(actual_scores[:2], reference_scores[:2]):
        np.testing.assert_allclose(got, want, atol=2e-12, rtol=2e-9)
    np.testing.assert_array_equal(actual_scores[-1], reference_scores[-1])
    static = replace(static, strong_mask=actual_scores[-1])
    actual = _pairs.weak_pair_energy_and_grad(
        mf, common, static, prepared_endpoints=prepared,
        _prepared_pairs=response, _grouped_ao_response=True)
    expected = _pairs.weak_pair_energy_and_grad(mf, common, static)
    _compare(actual, expected)
    assert not prepared and not response
    # Independent finite difference in overlap, Fock and PAO coefficients.
    rng = np.random.default_rng(51)
    tangent = {}
    for name in ('s1e', 'fock', 'pao_coeff'):
        value = rng.normal(size=getattr(common, name).shape) * .001
        if name != 'pao_coeff':
            value = (value + value.T) * .5
        tangent[name] = jnp.asarray(value)
    h = 1e-4
    plus = common._replace(**{name:getattr(common, name)+h*x for name,x in tangent.items()})
    minus = common._replace(**{name:getattr(common, name)-h*x for name,x in tangent.items()})
    finite = (_pairs.weak_pair_energy_sum(mf, plus, static)
              - _pairs.weak_pair_energy_sum(mf, minus, static)) / (2*h)
    analytic = sum(jnp.sum(getattr(actual[1][1], name)*x) for name,x in tangent.items())
    np.testing.assert_allclose(analytic, finite, atol=2e-10, rtol=2e-6)


@pytest.mark.parametrize('empty', [False, True])
def test_rank_selection_keeps_live_subspace_response(monkeypatch, empty):
    from pyscfad.dlno_stc._primary_response import _select_preprojection
    from pyscfad.dlno_stc.test.test_pao_response_reuse import _fixture
    _, common, _ = _fixture(monkeypatch)
    pao = common.pao_coeff.at[:, 0].multiply(1e-5)
    if empty:
        pao = pao * 1e-8
    common = common._replace(pao_coeff=pao)
    columns, support, ao = np.arange(4), np.arange(4), np.arange(7)
    thresholds = _selection.BoysDomainThresholds(bp_pao=.1, domain_pao=.1)
    def select(current):
        return _select_preprojection(
            current, domain._pao_products(current.pao_coeff, current.s1e),
            columns=columns, support=support, ao=ao, thresholds=thresholds)
    coeff, labels = select(common)
    assert len(labels[0]) == (0 if empty else 3)
    assert coeff.shape[1] == 0 if empty else coeff.shape[1] > 0
    selection = _selection.FixedPAOSubspaceSelection(
        columns, support, *labels, np.arange(coeff.shape[1]))
    weight = jnp.asarray(np.random.default_rng(57).normal(size=(7, 7)))
    def scalar(current, fresh):
        c = (select(current)[0] if fresh else
             domain._rebuild_selected_pao_preprojection(
                 current, selection=selection, representation_ao_indices=ao))
        return jnp.sum(weight * (c @ c.T))
    actual = jax.value_and_grad(scalar)(common, True)
    expected = jax.value_and_grad(scalar)(common, False)
    _compare(actual, expected)
    tangent = jnp.asarray(np.random.default_rng(58).normal(size=pao.shape) * .001)
    step = 1e-5
    plus = common._replace(pao_coeff=pao + step*tangent)
    minus = common._replace(pao_coeff=pao - step*tangent)
    finite = (scalar(plus, False) - scalar(minus, False))/(2*step)
    analytic = jnp.sum(actual[1].pao_coeff*tangent)
    np.testing.assert_allclose(analytic, finite, atol=2e-9, rtol=2e-6)
