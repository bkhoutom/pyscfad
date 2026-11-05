"""A shared initial reference retains ambient PAO and cached-product response."""
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import domain, _pairs
from pyscfad.dlno_stc.test.test_pao_response_reuse import _fixture, _assert_bars


class ReferenceMF(NamedTuple):
    mo_coeff: object
    mo_energy: object
    mol: object = None


def _reference_fixture(monkeypatch, *, zero_parents=False):
    rng = np.random.default_rng(714)
    if zero_parents:
        overlap, coeff = np.eye(6), np.eye(6)
    else:
        root = np.eye(6) + .1 * rng.normal(size=(6, 6))
        overlap = root.T @ root
        orth, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        coeff = np.linalg.solve(np.linalg.cholesky(overlap).T, orth)
    mf = ReferenceMF(jnp.asarray(coeff), jnp.arange(6.) - 2)
    data = jnp.asarray(overlap), jnp.diag(jnp.arange(6.) - 2)
    options = dict(active_occ=np.array([0, 1]), active_vir=np.arange(2, 6),
                   projected_out=np.array([0, 1]), groups=(np.array([1]), np.array([0])),
                   reference_coeff=coeff[:, :2], reference_coords=np.zeros((0, 3)),
                   lo_kwargs={}, pao_norm=1e-4)
    calls = []

    def targets(current, occupied, static, **kwargs):
        calls.append(kwargs)
        return occupied

    monkeypatch.setattr(domain, '_rebuild_boys_targets', targets)
    return mf, data, options, calls


def _legacy_common(mf, data, options, parents):
    overlap, fock = data
    occupied = mf.mo_coeff[:, options['active_occ']]
    virtual = mf.mo_coeff[:, options['active_vir']]
    pao = domain._normalized_pao(mf.mo_coeff[:, options['projected_out']], overlap, parents)
    return domain.DomainData(overlap, fock, occupied, virtual,
                             mf.mo_energy[options['active_occ']], mf.mo_energy[options['active_vir']],
                             occupied, pao, tuple(domain.TargetOccupiedData(occupied[:, group])
                                                  for group in options['groups']))


@pytest.mark.parametrize('zero_parents', [False, True])
def test_initial_common_matches_full_ambient_response_and_discards_zero_norms(monkeypatch, zero_parents):
    mf, data, options, calls = _reference_fixture(monkeypatch, zero_parents=zero_parents)
    build = getattr(domain, '_build_selection_reference', None)
    assert callable(build), 'selection needs one differentiable initial PAO/reference preparation'
    common, pullback, (parents, products) = jax.vjp(
        lambda m, d: build(m, d, **options), mf, data, has_aux=True)
    expected_parents = np.arange(2 if zero_parents else 0, 6)
    np.testing.assert_array_equal(parents, expected_parents)
    expected, expected_pullback = jax.vjp(
        lambda m, d: _legacy_common(m, d, options, expected_parents), mf, data)
    _assert_bars(common, expected)
    _assert_bars(products, _pairs._weak_screen_products(common))
    seed = jax.tree_util.tree_map(
        lambda value: jnp.sin(jnp.arange(value.size).reshape(value.shape) + .1), common)
    actual_bars, expected_bars = pullback(seed), expected_pullback(seed)
    _assert_bars(actual_bars, expected_bars)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree_util.tree_leaves(actual_bars))
    assert calls == [{'_stationary': True}]
    # Auxiliary primals contribute through the ambient product pullback once,
    # then through this common-frame VJP; has_aux must not freeze or double it.
    product_seed = jax.tree_util.tree_map(jnp.ones_like, products)
    _, product_response = _pairs._cached_weak_screen_products(common, products)
    product_common_bar, = product_response.pullback(product_seed)
    combined = jax.tree_util.tree_map(lambda left, right: left + right, seed, product_common_bar)

    def joint(m, d):
        current = _legacy_common(m, d, options, expected_parents)
        return current, _pairs._weak_screen_products(current)

    _, joint_pullback = jax.vjp(joint, mf, data)
    _assert_bars(pullback(combined), joint_pullback((seed, product_seed)))


@pytest.mark.parametrize('kind', ['real', 'asymmetric', 'complex'])
def test_cached_product_pullback_matches_independent_common_leaves(monkeypatch, kind):
    mf, common, static = _fixture(monkeypatch)
    rng = np.random.default_rng(33)
    if kind == 'asymmetric':
        common = common._replace(s1e=common.s1e + .02 * rng.normal(size=common.s1e.shape))
    if kind == 'complex':
        common = jax.tree_util.tree_map(
            lambda value: value.astype(jnp.complex128) + .03j * rng.normal(size=value.shape), common)
    products, expected_pullback = jax.vjp(_pairs._weak_screen_products, common)
    cached = getattr(_pairs, '_cached_weak_screen_products', None)
    assert callable(cached), 'cached products need their exact ambient common-data response'

    def forbidden(*args, **kwargs):
        raise AssertionError('cached preparation recomputed global product primals')

    monkeypatch.setattr(_pairs, '_weak_screen_products', forbidden)
    monkeypatch.setattr(domain, '_pao_products', forbidden)
    values, response = cached(common, products)
    assert values is products
    seed = jax.tree_util.tree_map(
        lambda value: jnp.asarray(rng.normal(size=value.shape), dtype=value.dtype), products)
    actual, = response.pullback(seed)
    expected, = expected_pullback(seed)
    _assert_bars(actual, expected)
    assert np.linalg.norm(actual.s1e) > 0
    assert np.linalg.norm(actual.pao_coeff) > 0
    assert all(np.linalg.norm(target.iao_coeff) > 0 for target in actual.fragment_occupied_data)
    np.testing.assert_array_equal(actual.iao_coeff, 0.)
    np.testing.assert_array_equal(actual.occupied_coeff, 0.)


@pytest.mark.parametrize('prepared', [False, True])
def test_screening_reuses_supplied_products_without_primal_rebuild(monkeypatch, prepared):
    mf, common, static = _fixture(monkeypatch)
    expected, expected_bars = _pairs.weak_pair_energy_and_grad(mf, common, static)
    products = _pairs._weak_screen_products(common)
    cached = getattr(_pairs, '_cached_weak_screen_products', None)
    assert callable(cached), 'screening must accept a shared initial product preparation'
    preparation = cached(common, products)

    def forbidden(*args, **kwargs):
        raise AssertionError('screening rebuilt already prepared global products')

    monkeypatch.setattr(_pairs, '_weak_screen_products', forbidden)
    monkeypatch.setattr(domain, '_pao_products', forbidden)
    endpoints = {}
    kwargs = (dict(prepared_endpoints=endpoints, _products_preparation=preparation)
              if prepared else dict(_products=products))
    _, weak, _, _ = _pairs.screen_pairs(mf, common, static, **kwargs)
    np.testing.assert_allclose(sum(weak[i, j] for i, j in _pairs.weak_pair_indices(static)),
                               expected, rtol=5e-11, atol=1e-12)
    if prepared:
        value, bars = _pairs.weak_pair_energy_and_grad(
            mf, common, static, prepared_endpoints=endpoints, _grouped_ao_response=True)
        np.testing.assert_allclose(value, expected, rtol=5e-11, atol=1e-12)
        _assert_bars(bars, expected_bars)


def test_cached_product_reverse_batches_independent_occupied_targets(monkeypatch):
    from collections import Counter
    from pyscfad.dlno_stc.test.test_weak_local_products import _dots
    mf, common, static = _fixture(monkeypatch)
    products = _pairs._weak_screen_products(common)
    _, response = _pairs._cached_weak_screen_products(common, products)
    seed = jax.tree_util.tree_map(jnp.ones_like, products)
    operations = Counter(_dots(jax.make_jaxpr(response.pullback)(seed)))
    assert operations[((8, 3), (8, 3))] == 1
    assert operations[((8, 1), (8, 1))] == 0, 'target response formed a full overlap bar per endpoint'


@pytest.mark.parametrize('failure', [None, 'stationarity', 'response'])
def test_stationary_reference_route_retains_alignment_and_narrow_fallback(monkeypatch, failure):
    from pyscfad import gto
    from pyscfad.lo import boys
    mol = gto.Mole(atom='O 0 0 0; H .1 -.75 .57; H 0 .8 .61', basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    overlap = np.asarray(mol.intor('int1e_ovlp'))
    rng = np.random.default_rng(217)
    coeff = np.linalg.inv(np.linalg.cholesky(overlap).T) @ np.linalg.qr(rng.normal(size=(7, 3)))[0]
    reference = coeff @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
    options = dict(conv_tol=1e-10, conv_tol_grad=1e-8, max_cycle=123,
                   init_guess='cholesky', gmres_options=dict(restart=17, maxiter=31), symmetry=True)
    static = SimpleNamespace(target_reference_coeff=reference,
                             target_reference_coords=np.asarray(mol.atom_coords()), lo_kwargs=options)
    calls = []

    def refine(mol, orbitals, settings):
        calls.append(('refine', settings))
        return orbitals

    def stationary(mol, orbitals, **settings):
        calls.append(('stationary', settings))
        if failure == 'stationarity':
            raise boys.BoysStationarityError('not stationary')
        if failure == 'response':
            raise RuntimeError('incompatible localization response')
        return orbitals

    monkeypatch.setattr(domain, '_build_boys_targets', refine)
    monkeypatch.setattr(boys, '_boys_at_stationary_frame', stationary)
    legacy = lambda m, c: domain._rebuild_boys_targets(SimpleNamespace(mol=m), c, static)
    cached = lambda m, c: domain._rebuild_boys_targets(SimpleNamespace(mol=m), c, static, _stationary=True)
    expected, expected_pullback = jax.vjp(legacy, mol, jnp.asarray(coeff))
    calls.clear()
    if failure == 'response':
        with pytest.raises(RuntimeError, match='incompatible localization response'):
            cached(mol, jnp.asarray(coeff))
        assert [name for name, _ in calls] == ['stationary']
        return
    actual, actual_pullback = jax.vjp(cached, mol, jnp.asarray(coeff))
    np.testing.assert_allclose(actual, expected, atol=2e-13)
    seed = jnp.asarray(rng.normal(size=actual.shape))
    _assert_bars(actual_pullback(seed), expected_pullback(seed))
    assert [name for name, _ in calls] == (['stationary', 'refine'] if failure else ['stationary'])
    assert calls[0][1] == {key: options[key] for key in
                          ('conv_tol', 'conv_tol_grad', 'symmetry', 'gmres_options')}
    if failure:
        assert calls[1][1]['max_saddle_escapes'] == 0
        np.testing.assert_array_equal(calls[1][1]['init_guess'], np.eye(3))
