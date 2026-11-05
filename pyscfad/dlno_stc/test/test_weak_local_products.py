"""Weak PAO groups use local blocks of one differentiable product cache."""
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import _pairs, domain
from pyscfad.dlno_stc.test.test_pao_response_reuse import _fixture, _assert_bars


def _legacy_preprojection(common, selection, representation_ao_indices, **unused):
    """Frozen pre-local-block algebra, including its global orbital frame."""
    overlap = common.s1e
    parent = common.pao_coeff[:, selection.parent_columns]
    gram = parent.T.conj() @ overlap @ parent
    gram = .5 * (gram + gram.T.conj())
    _, vectors = domain.scipy.linalg.eigh(gram, deg_thresh=1e-6)
    orth = domain._cholesky_metric_orthonormalize(
        parent @ vectors[:, selection.canonical_keep], overlap)
    support = selection.support_ao_indices
    projected = overlap[support] @ orth
    metric = projected.T.conj() @ jnp.linalg.solve(overlap[np.ix_(support, support)], projected)
    _, rotation = domain.scipy.linalg.eigh(.5 * (metric + metric.T.conj()), deg_thresh=1e-6)
    candidate = (orth @ rotation[:, selection.overlap_keep])[:, selection.completeness_keep]
    return domain._project_to_ao_support(candidate, overlap, representation_ao_indices)


def _reference_energy_and_grad(monkeypatch, mf, common, static, *, pairs=None):
    def legacy(common, *args, selection, representation_ao_indices, **kwargs):
        return _legacy_preprojection(common, selection, representation_ao_indices)

    # The endpoint route without preparations retains the original occupied
    # projection and uses the frozen preprojection above for this reference.
    with monkeypatch.context() as context:
        context.setattr(domain, '_rebuild_selected_pao_preprojection', legacy)
        return _pairs.weak_pair_energy_and_grad(mf, common, static, pairs=pairs)


def _dots(jaxpr):
    jaxpr = getattr(jaxpr, 'jaxpr', jaxpr)
    for equation in jaxpr.eqns:
        if equation.primitive.name == 'dot_general':
            yield tuple(tuple(value.aval.shape) for value in equation.invars)
        for value in equation.params.values():
            values = value if isinstance(value, (tuple, list)) else (value,)
            for child in values:
                if hasattr(child, 'jaxpr') or hasattr(child, 'eqns'):
                    yield from _dots(child)


def test_preprojection_has_no_full_ao_products_with_supplied_products(monkeypatch):
    mf, common, static = _fixture(monkeypatch)
    build = getattr(domain, '_pao_products', None)
    assert callable(build), 'weak preparation needs one shared S@P and PAO Gram builder'
    products = build(common.pao_coeff, common.s1e)
    fragment = static.fragments[0]
    operation = lambda c, p: domain._rebuild_selected_pao_preprojection(
        c, p, selection=fragment.weak_virtual,
        representation_ao_indices=fragment.primary_ao_indices)
    traced = jax.make_jaxpr(operation)(common, products)
    products_trace = jax.make_jaxpr(build)(common.pao_coeff, common.s1e)
    assert Counter(_dots(products_trace)) == Counter({((8, 8), (8, 4)): 1,
                                                    ((4, 8), (8, 4)): 1})
    assert list(_dots(traced))
    assert all(8 not in shape for operands in _dots(traced) for shape in operands)
    expected, expected_pullback = jax.vjp(
        lambda c: _legacy_preprojection(c, fragment.weak_virtual, fragment.primary_ao_indices), common)
    actual, actual_pullback = jax.vjp(
        lambda c: operation(c, build(c.pao_coeff, c.s1e)), common)
    seed = jnp.sin(jnp.arange(actual.size).reshape(actual.shape) + .3)
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)
    _assert_bars(actual_pullback(seed), expected_pullback(seed))


@pytest.mark.parametrize('grouped_ao', [False, True])
@pytest.mark.parametrize('changed', [None, ('canonical_keep', [1, 2, 3]),
                                    ('support_ao_indices', [1, 2, 3, 4])])
def test_one_global_product_response_preserves_every_common_gradient(monkeypatch, grouped_ao, changed):
    mf, common, static = _fixture(monkeypatch, changed)
    build = getattr(_pairs, '_weak_screen_products', None)
    assert callable(build), 'prepared endpoints must share full-AO products and their VJP'
    expected, expected_bars = _reference_energy_and_grad(monkeypatch, mf, common, static)
    calls = Counter()
    tapes = []
    vjp = jax.vjp

    def counted(fun, *args, **kwargs):
        value, pullback = vjp(fun, *args, **kwargs)
        if fun is build:
            calls['forward'] += 1

            def tracked(bar):
                calls['reverse'] += 1
                return pullback(bar)

            tapes.append(weakref.ref(tracked))
            return value, tracked
        return value, pullback

    monkeypatch.setattr(jax, 'vjp', counted)
    prepared = {}
    _pairs.screen_pairs(mf, common, static, prepared_endpoints=prepared)
    assert calls == Counter(forward=1)
    value, bars = _pairs.weak_pair_energy_and_grad(
        mf, common, static, prepared_endpoints=prepared, _grouped_ao_response=grouped_ao)
    assert calls == Counter(forward=1, reverse=1)
    assert all(tape() is None for tape in tapes)
    np.testing.assert_allclose(value, expected, rtol=5e-11, atol=1e-12)
    _assert_bars(bars, expected_bars)
    for name in ('s1e', 'pao_coeff', 'fock'):
        assert np.linalg.norm(getattr(bars[1], name)) > 1e-9
    for target in bars[1].fragment_occupied_data:
        assert np.linalg.norm(target.iao_coeff) > 1e-9
    np.testing.assert_array_equal(bars[1].iao_coeff, 0.)


@pytest.mark.parametrize('pairs', [(), ((0, 1),), ((0, 1), (0, 2))])
def test_products_ignore_unused_groups_and_release_after_last_active_target(monkeypatch, pairs):
    mf, common, static = _fixture(monkeypatch, ('canonical_keep', [1, 2, 3]))
    build = getattr(_pairs, '_weak_screen_products', None)
    assert callable(build), 'weak products need invocation-owned response'
    calls = Counter()
    tapes = []
    vjp = jax.vjp

    def counted(fun, *args, **kwargs):
        value, pullback = vjp(fun, *args, **kwargs)
        if fun is build:
            calls['forward'] += 1

            def tracked(bar):
                calls['reverse'] += 1
                return pullback(bar)

            tapes.append(weakref.ref(tracked))
            return value, tracked
        return value, pullback

    expected, expected_bars = _reference_energy_and_grad(monkeypatch, mf, common, static, pairs=pairs)
    monkeypatch.setattr(jax, 'vjp', counted)
    prepared = {}
    _pairs.screen_pairs(mf, common, static, prepared_endpoints=prepared)
    value, bars = _pairs.weak_pair_energy_and_grad(
        mf, common, static, pairs=pairs, prepared_endpoints=prepared, _grouped_ao_response=True)
    assert calls == Counter(forward=1, reverse=int(bool(pairs)))
    assert all(tape() is None for tape in tapes)
    np.testing.assert_allclose(value, expected, rtol=5e-11, atol=1e-12)
    _assert_bars(bars, expected_bars)
    assert prepared == {}


@pytest.mark.parametrize('missing', [(), (1,), (0, 1, 2)])
def test_shared_products_allow_missing_preparations_and_different_metric_ranks(monkeypatch, missing):
    mf, common, static = _fixture(monkeypatch)
    static.thresholds.metric_rank = .745
    fragment = static.fragments[0]
    fragment.weak_virtual = replace(fragment.weak_virtual, metric_keep=np.array([1]))
    expected, expected_bars = _reference_energy_and_grad(monkeypatch, mf, common, static)
    prepared = {}
    _pairs.screen_pairs(mf, common, static, prepared_endpoints=prepared)
    for index in missing:
        del prepared[index]
    value, bars = _pairs.weak_pair_energy_and_grad(
        mf, common, static, prepared_endpoints=prepared, _grouped_ao_response=True)
    np.testing.assert_allclose(value, expected, rtol=5e-11, atol=1e-12)
    _assert_bars(bars, expected_bars)


def test_all_strong_screening_never_builds_global_products(monkeypatch):
    mf, common, static = _fixture(monkeypatch)
    build = getattr(_pairs, '_weak_screen_products', None)
    assert callable(build), 'shared product boundary is missing'

    def forbidden(*args, **kwargs):
        raise AssertionError('all-strong screening built unused global products')

    monkeypatch.setattr(_pairs, '_weak_screen_products', forbidden)
    prepared = {0: object()}
    _pairs.screen_pairs(mf, common, static, model='all', prepared_endpoints=prepared)
    assert prepared == {}


def test_target_reverse_uses_only_local_dot_operands(monkeypatch):
    mf, common, static = _fixture(monkeypatch)
    traced_products = jax.make_jaxpr(_pairs._weak_screen_products)(common)
    assert Counter(_dots(traced_products)) == Counter({
        ((8, 8), (8, 4)): 1, ((4, 8), (8, 4)): 1, ((8, 8), (8, 3)): 1})
    prepared = {}
    _pairs.screen_pairs(mf, common, static, prepared_endpoints=prepared)
    for _, (screen, response) in prepared.values():
        seed = jax.tree_util.tree_map(jnp.ones_like, screen)
        traced = jax.make_jaxpr(response.target_pullback)(seed)
        assert list(_dots(traced))
        assert all(8 not in shape for operands in _dots(traced) for shape in operands)


@pytest.mark.parametrize('grouped_ao', [False, True])
def test_product_tape_released_before_remaining_legacy_endpoint(monkeypatch, grouped_ao):
    mf, common, static = _fixture(monkeypatch)
    static.fragments[2].weak_virtual = SimpleNamespace(**vars(static.fragments[2].weak_virtual))
    expected, expected_bars = _reference_energy_and_grad(monkeypatch, mf, common, static)
    prepared = {}
    _pairs.screen_pairs(mf, common, static, prepared_endpoints=prepared)
    product_tape = weakref.ref(prepared[0][1][1].products_pullback.pullback)
    moments = _pairs._weak_endpoint_from_screen

    def checked(m, screen, static, fragment_index, **kwargs):
        if fragment_index == 2:
            assert product_tape() is None, 'finished global product tape remained live'
        return moments(m, screen, static, fragment_index, **kwargs)

    monkeypatch.setattr(_pairs, '_weak_endpoint_from_screen', checked)
    value, bars = _pairs.weak_pair_energy_and_grad(
        mf, common, static, prepared_endpoints=prepared, _grouped_ao_response=grouped_ao)
    assert product_tape() is None
    np.testing.assert_allclose(value, expected, rtol=5e-11, atol=1e-12)
    _assert_bars(bars, expected_bars)


def test_unprepared_endpoint_builds_products_only_for_selected_parents(monkeypatch):
    mf, common, static = _fixture(monkeypatch)
    common = common._replace(pao_coeff=jnp.concatenate(
        (common.pao_coeff, common.pao_coeff[:, :2] * .3), axis=1))
    selection = static.fragments[0].weak_virtual
    widths = []
    products = domain._pao_products

    def track(parent, overlap):
        widths.append(parent.shape[1])
        return products(parent, overlap)

    monkeypatch.setattr(domain, '_pao_products', track)
    actual = domain._rebuild_selected_pao_preprojection(
        common, selection=selection, representation_ao_indices=np.arange(7))
    expected = _legacy_preprojection(common, selection, np.arange(7))
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)
    assert widths == [4], 'unprepared endpoint formed a Gram over unrelated PAO parents'


def test_full_ao_identity_preserves_direct_parent_and_shared_product_gradients(monkeypatch):
    mf, common, static = _fixture(monkeypatch)
    selection = static.fragments[0].weak_virtual
    ao = np.arange(8)
    expected, expected_pullback = jax.vjp(
        lambda c: _legacy_preprojection(c, selection, ao), common)
    actual, actual_pullback = jax.vjp(
        lambda c: domain._rebuild_selected_pao_preprojection(
            c, domain._pao_products(c.pao_coeff, c.s1e),
            selection=selection, representation_ao_indices=ao), common)
    seed = jnp.sin(jnp.arange(actual.size).reshape(actual.shape) + .3)
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)
    _assert_bars(actual_pullback(seed), expected_pullback(seed))
    assert np.linalg.norm(actual_pullback(seed)[0].pao_coeff) > 1e-9


def test_full_ao_endpoint_response_matches_expanded_oracle(monkeypatch):
    mf, common, static = _fixture(monkeypatch)
    for fragment in static.fragments:
        fragment.primary_ao_indices = np.arange(8)
    raw = np.random.default_rng(21).normal(size=(3, 8, 8))
    raw = jnp.asarray((raw + raw.transpose(0, 2, 1)) / 2)

    def moments(mol, order):
        shift = .02 * jnp.sum(mol.coordinates**2)
        return jnp.eye(8), raw * (1 + shift), None, None

    monkeypatch.setattr(_pairs, '_origin_zero_moments', moments)
    expected, expected_bars = _reference_energy_and_grad(monkeypatch, mf, common, static)
    prepared = {}
    _pairs.screen_pairs(mf, common, static, prepared_endpoints=prepared)
    value, bars = _pairs.weak_pair_energy_and_grad(
        mf, common, static, prepared_endpoints=prepared, _grouped_ao_response=True)
    np.testing.assert_allclose(value, expected, rtol=5e-11, atol=1e-12)
    _assert_bars(bars, expected_bars)


@pytest.mark.parametrize('entry', ['screening', 'energy_sum', 'explicit_sum'])
@pytest.mark.parametrize('changed', [None, ('canonical_keep', [1, 2, 3])])
def test_forward_only_multi_endpoint_calls_share_global_products_without_tapes(monkeypatch, entry, changed):
    mf, common, static = _fixture(monkeypatch, changed)
    expected, _ = _reference_energy_and_grad(monkeypatch, mf, common, static)
    products = _pairs._weak_screen_products
    preprojection = domain._rebuild_selected_pao_preprojection
    calls = Counter()

    def counted(current):
        calls['products'] += 1
        return products(current)

    def projected(*args, **kwargs):
        calls['preprojection'] += 1
        return preprojection(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('forward-only weak preparation constructed a reverse tape')

    monkeypatch.setattr(_pairs, '_weak_screen_products', counted)
    monkeypatch.setattr(domain, '_rebuild_selected_pao_preprojection', projected)
    monkeypatch.setattr(jax, 'vjp', forbidden)
    if entry == 'screening':
        _, weak, _, _ = _pairs.screen_pairs(mf, common, static)
        actual = sum(weak[i, j] for i, j in _pairs.weak_pair_indices(static))
    elif entry == 'explicit_sum':
        actual = _pairs.weak_pair_energy_sum(
            mf, common, static, pairs=_pairs.weak_pair_indices(static))
    else:
        actual = _pairs.weak_pair_energy_sum(mf, common, static)
    np.testing.assert_allclose(actual, expected, rtol=5e-11, atol=1e-12)
    assert calls == Counter(products=1, preprojection=1 + int(changed is not None))


def test_forward_product_sharing_preserves_direct_autodiff_chain_rule(monkeypatch):
    mf, common, static = _fixture(monkeypatch, ('canonical_keep', [1, 2, 3]))
    expected, expected_bars = _reference_energy_and_grad(monkeypatch, mf, common, static)
    value, bars = jax.value_and_grad(
        lambda m, c: _pairs.weak_pair_energy_sum(m, c, static), argnums=(0, 1))(mf, common)
    np.testing.assert_allclose(value, expected, rtol=5e-11, atol=1e-12)
    _assert_bars(bars, expected_bars)


@pytest.mark.parametrize('entry', ['pair', 'partial_sum'])
def test_partial_queries_use_only_selected_parents_with_full_gradient_parity(monkeypatch, entry):
    mf, common, static = _fixture(monkeypatch)
    common = common._replace(pao_coeff=jnp.concatenate(
        (common.pao_coeff, common.pao_coeff[:, :2] * .3), axis=1))
    expected, expected_bars = _reference_energy_and_grad(
        monkeypatch, mf, common, static, pairs=((0, 1),))
    widths = []
    products = domain._pao_products

    def selected_products(parent, overlap):
        widths.append(parent.shape[1])
        return products(parent, overlap)

    def forbidden(*args, **kwargs):
        raise AssertionError('partial query computed all-system weak products')

    def evaluate(m, c):
        if entry == 'pair':
            return _pairs.weak_pair_energy(m, c, static, 0, 1)
        return _pairs.weak_pair_energy_sum(m, c, static, pairs=((0, 1),))

    monkeypatch.setattr(_pairs, '_weak_screen_products', forbidden)
    monkeypatch.setattr(domain, '_pao_products', selected_products)
    actual = evaluate(mf, common)
    assert widths == [4, 4]
    np.testing.assert_allclose(actual, expected, rtol=5e-11, atol=1e-12)
    widths.clear()
    value, bars = jax.value_and_grad(evaluate, argnums=(0, 1))(mf, common)
    assert widths == [4, 4]
    np.testing.assert_allclose(value, expected, rtol=5e-11, atol=1e-12)
    _assert_bars(bars, expected_bars)
