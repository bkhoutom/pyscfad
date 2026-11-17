"""Explicit host weak response versus independent JAX reverse rules."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.lib import _multipole as reference, _multipole_numpy as host
from pyscfad.lib._cartesian_moments import cartesian_components
from pyscfad.dlno_stc._pairs import WeakScreen
from pyscfad.dlno_stc.test import _weak_reference as oracle
from pyscfad.dlno_stc import _weak_numpy as spaces, domain, _pairs
from pyscfad.dlno_stc.test.test_pao_response_reuse import _fixture
from pyscfad.dlno_stc.test.test_boys_prepare import boys_water_dimer


def compare(actual, expected, atol=2e-10):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, atol=atol, rtol=2e-10)


@pytest.mark.parametrize('order', [2, 3, 4])
@pytest.mark.parametrize('sizes', [(2, 5), (0, 3)])
def test_pair_host_reverse(order, sizes):
    rng = np.random.default_rng(51)
    endpoints = []
    for i, n in enumerate(sizes):
        endpoints.append((rng.normal(size=3) + i*3, rng.normal(size=(3, n)),
            rng.uniform(.1, 2., n), rng.normal(size=(3, 3, n)) if order > 2 else None,
            rng.normal(size=(3, 3, 3, n)) if order > 3 else None))
    expected = reference.pair_energy_and_grad(*endpoints, order)
    actual = host.pair(*endpoints, order, with_grad=True)
    compare(actual, expected)
    compare(host.pair(*endpoints, order), actual[0])


@pytest.mark.parametrize('order', [2, 3, 4])
@pytest.mark.parametrize('packed', [False, True])
def test_endpoint_host_reverse(order, packed):
    rng = np.random.default_rng(42)
    n, v = 7, 4
    screen = WeakScreen(np.array([-1.]), rng.normal(size=(n, 1)),
                         np.arange(v)*.1 + .3, rng.normal(size=(n, v)))
    operators = []
    for rank in range(4):
        if rank >= order:
            operators.append(None)
        else:
            count = len(cartesian_components(rank)[0]) if packed else 3**rank
            shape = (count, n, n) if packed else (3,)*rank+(n, n)
            operators.append(rng.normal(size=shape))
    operators[0] = operators[0].reshape(n, n)
    operators = tuple(operators)
    def evaluate(screen, operators):
        center, mu, theta, omega = reference.transition_moments(
            screen.occupied_coeff, screen.virtual_coeff, *operators, order)
        return (center[0], mu[0], screen.virtual_energy-screen.occupied_energy[0],
                None if theta is None else theta[0], None if omega is None else omega[0])
    value, pullback = jax.vjp(evaluate, screen, operators)
    actual, reverse = host.endpoint(screen, operators, order, with_grad=True)
    seed = jax.tree.map(lambda x: rng.normal(size=x.shape), value)
    compare(actual, value)
    compare(host.endpoint(screen, operators, order), value)
    compare(reverse(seed), pullback(seed), atol=2e-9)


@pytest.mark.parametrize('keep', [None, (0, 1, 2), (1, 2), ()])
def test_host_metric_space_pullback(keep):
    rng = np.random.default_rng(181)
    c = rng.normal(size=(7, 3))
    root = rng.normal(size=(7, 7))
    s = root.T @ root + np.eye(7)
    if keep is None:
        actual, reverse = spaces.normalize(c, s)
        fun = domain._cholesky_metric_orthonormalize
    else:
        actual, reverse, _ = spaces.metric_space(c, s, 1e-9, keep)
        fun = lambda c, s: domain._fixed_metric_orthonormalize(c, s, np.array(keep, int))
    expected, pb = jax.vjp(fun, c, s)
    seed = rng.normal(size=actual.shape)
    compare(actual, expected)
    compare(reverse(seed), pb(seed))


def test_host_degenerate_eigh_response():
    matrix = np.diag([-.4, .5, .5, 1.1])
    actual, reverse = spaces.eigh(matrix, deg_thresh=1e-8)
    expected, pb = jax.vjp(lambda a: domain.scipy.linalg.eigh(a, deg_thresh=1e-8), matrix)
    rng = np.random.default_rng(12)
    seed = tuple(rng.normal(size=x.shape) for x in expected)
    compare(actual, expected)
    compare(reverse(*seed), pb(seed)[0])


@pytest.mark.parametrize('empty', [False, True])
def test_host_primary_rank_selection_reverse(monkeypatch, empty):
    from pyscfad.dlno_stc.test._weak_reference import _select_preprojection
    from pyscfad.dlno_stc._selection import BoysDomainThresholds
    _, common, _ = _fixture(monkeypatch)
    pao = np.array(common.pao_coeff)
    pao[:, 0] *= 1e-5
    if empty:
        pao *= 1e-8
    common = common._replace(pao_coeff=pao)
    opts = dict(columns=np.arange(4), support=np.arange(4), ao=np.arange(7),
                thresholds=BoysDomainThresholds(bp_pao=.1, domain_pao=.1))
    p = domain._pao_products(common.pao_coeff, common.s1e)
    actual, reverse, labels = spaces.preprojection(common, p, **opts)
    expected, pb, expected_labels = jax.vjp(
        lambda c, p: _select_preprojection(c, p, **opts), common, p, has_aux=True)
    compare(labels, expected_labels)
    compare(actual, expected)
    seed = np.random.default_rng(57).normal(size=actual.shape)
    compare(reverse(seed), pb(seed))


def test_host_fixed_empty_virtual_space_reverse(monkeypatch):
    from dataclasses import replace
    _, common, static = _fixture(monkeypatch)
    fragment = static.fragments[0]
    fragment.weak_virtual = replace(fragment.weak_virtual,
        completeness_keep=np.empty(0, dtype=int), metric_keep=np.empty(0, dtype=int))
    actual, reverse = spaces.fixed_screen(common, static, 0)
    expected, pb = jax.vjp(lambda c: oracle.build_weak_multipole_screen(c, static, 0), common)
    compare(actual, expected)
    seed = jax.tree.map(lambda x: np.ones_like(x), actual)
    compare(reverse(seed), pb(seed))


def test_host_products_reverse(monkeypatch):
    _, common, _ = _fixture(monkeypatch)
    actual, reverse = spaces.products(common)
    expected, pb = jax.vjp(oracle._weak_screen_products, common)
    rng = np.random.default_rng(81)
    seed = jax.tree.map(lambda x: rng.normal(size=x.shape), expected)
    compare(actual, expected)
    compare(reverse.pullback(seed), pb(seed))


@pytest.mark.parametrize('ao', [np.arange(7), np.arange(4), np.arange(8)])
def test_host_preprojection_reverse(monkeypatch, ao):
    _, common, static = _fixture(monkeypatch)
    selected = static.fragments[0].weak_virtual
    p = domain._pao_products(common.pao_coeff, common.s1e)
    actual, reverse, _ = spaces.preprojection(common, p,
        columns=selected.parent_columns, support=selected.support_ao_indices,
        ao=ao, selection=selected)
    expected, pb = jax.vjp(lambda c, p: domain._rebuild_selected_pao_preprojection(
        c, p, selection=selected, representation_ao_indices=ao), common, p)
    seed = np.random.default_rng(30).normal(size=actual.shape)
    compare(actual, expected)
    compare(reverse(seed), pb(seed))


def test_host_target_reverse(monkeypatch):
    from pyscfad.dlno_stc.test._weak_reference import _select_target
    _, common, static = _fixture(monkeypatch)
    fragment = static.fragments[0]
    candidate = domain._rebuild_selected_pao_preprojection(common,
        selection=fragment.weak_virtual, representation_ao_indices=fragment.primary_ao_indices)
    rhs = common.s1e @ common.fragment_occupied_data[0].iao_coeff
    opts = dict(ao=fragment.primary_ao_indices, thresholds=static.thresholds, index=0)
    actual, reverse, labels = spaces.target(common, candidate, rhs, **opts)
    expected, pb, expected_labels = jax.vjp(lambda *args: _select_target(*args, **opts),
                                           common, candidate, rhs, has_aux=True)
    compare(labels, expected_labels)
    compare(actual, expected)
    rng = np.random.default_rng(77)
    seed = jax.tree.map(lambda x: rng.normal(size=x.shape), expected)
    compare(reverse(seed), pb(seed))


@pytest.mark.parametrize('grouped', [False, True])
def test_host_shared_products_and_projection_reversed_once(monkeypatch, grouped):
    from collections import Counter
    mf, common, static = _fixture(monkeypatch, numpy_backend=True)
    expected = jax.value_and_grad(lambda mf, common: oracle.energy(mf, common, static), argnums=(0,1))(mf, common)
    calls = Counter()
    products, preprojection, target = spaces.products, spaces.fixed_preprojection, spaces.fixed_target
    def counted_products(*args, **kwargs):
        value, response = products(*args, **kwargs)
        calls['products'] += 1
        def reverse(bar):
            calls['products_reverse'] += 1
            return response.pullback(bar)
        return value, response._replace(pullback=reverse)
    def counted_preprojection(*args, **kwargs):
        value, pb = preprojection(*args, **kwargs)
        calls['projection'] += 1
        def reverse(bar):
            calls['projection_reverse'] += 1
            return pb(bar)
        return value, reverse
    def counted_target(*args, **kwargs):
        value, pb = target(*args, **kwargs)
        calls['target'] += 1
        def reverse(bar):
            calls['target_reverse'] += 1
            return pb(bar)
        return value, reverse
    monkeypatch.setattr(spaces, 'products', counted_products)
    monkeypatch.setattr(spaces, 'fixed_preprojection', counted_preprojection)
    monkeypatch.setattr(spaces, 'fixed_target', counted_target)
    preparations = {}
    _pairs.screen_pairs(mf, common, static, prepared_endpoints=preparations)
    actual = _pairs.weak_pair_energy_and_grad(mf, common, static,
        prepared_endpoints=preparations, _grouped_ao_response=grouped)
    compare(actual, expected)
    assert calls == Counter(products=1, products_reverse=1, projection=1,
                            projection_reverse=1, target=3, target_reverse=3)
    assert preparations == {}


@pytest.mark.parametrize('order', [2, 3, 4])
def test_molecular_host_weak_forward_reverse_never_calls_jax_vjp(boys_water_dimer, monkeypatch, order):
    from dataclasses import replace
    mf, common, static = boys_water_dimer
    static = replace(static, thresholds=replace(static.thresholds, multipole_order=order))
    expected = _pairs.weak_pair_energy_and_grad(mf, common, static,
                                               _grouped_ao_response=False)
    def forbidden(*args, **kwargs):
        raise AssertionError('host weak calculation invoked JAX tracing/compiled multipoles')
    monkeypatch.setattr(jax, 'vjp', forbidden)
    actual = _pairs.weak_pair_energy_and_grad(mf, common, static,
                                             _grouped_ao_response=True)
    compare(actual, expected, atol=2e-9)
    compare(_pairs.weak_pair_energy_sum(mf, common, static), actual[0])


def test_weak_host_rejects_traced_and_complex_inputs(monkeypatch):
    _, common, _ = _fixture(monkeypatch)
    with pytest.raises(TypeError, match='explicit NumPy/SciPy pullback'):
        jax.vjp(_pairs._weak_screen_products, common)
    with pytest.raises(TypeError, match='real float64'):
        _pairs._weak_screen_products(common._replace(s1e=np.asarray(common.s1e, dtype=complex)))
