"""Padded workspaces preserve physical target spaces and ambient response."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import _primary_response as primary
from pyscfad.dlno_stc._selection import BoysDomainThresholds


@pytest.mark.parametrize('rows,columns,deficient,complex_values', [
    (7, 1, False, False), (9, 4, False, True), (35, 17, False, False),
    (35, 17, True, True), (67, 35, True, False),
])
def test_bucketed_target_matches_unpadded_space_and_response(rows, columns, deficient, complex_values):
    rng = np.random.default_rng(717)

    def random(shape):
        value = rng.normal(size=shape)
        return value + .2j * rng.normal(size=shape) if complex_values else value

    size = rows + 3
    ao = np.r_[0, rng.permutation(np.arange(1, size))[:rows-1]].astype(np.int32)
    root = np.eye(size) + .01 * random((size, size))
    overlap = root.T.conj() @ root
    fock = np.diag(np.r_[-3., np.arange(1., size)])
    if complex_values:
        fock = fock.astype(complex)
    candidate = random((rows, columns))
    candidate[0] = 0
    if deficient:
        candidate[:, -1] = .7 * candidate[:, 0]
    rhs = overlap[:, :1]
    arguments = tuple(map(jnp.asarray, (overlap, fock, candidate, rhs)))
    thresholds = BoysDomainThresholds()
    weights = jnp.asarray(random((rows, rows)))
    occupied_weight = jnp.asarray(random((rows, 1)))

    def evaluate(function, s, f, c, r):
        return function(SimpleNamespace(s1e=s, fock=f), c, r,
                        ao=ao, thresholds=thresholds, index=4)

    expected, expected_labels = evaluate(primary._select_target_reference, *arguments)
    actual, actual_labels = evaluate(primary._select_target, *arguments)
    for got, want in zip(actual_labels, expected_labels):
        np.testing.assert_array_equal(got, want)
    assert actual.virtual_coeff.shape == (rows, columns - int(deficient))
    np.testing.assert_allclose(actual.occupied_coeff, expected.occupied_coeff, atol=2e-12)
    np.testing.assert_allclose(actual.virtual_energy, expected.virtual_energy, atol=2e-10)

    def functional(function, *args):
        screen, _ = evaluate(function, *args)
        cv = screen.virtual_coeff
        # Gauge-invariant observables exercise the eigenspace and its energy
        # response without assuming LAPACK eigenvector signs in padded blocks.
        density = cv @ cv.T.conj()
        weighted = (cv * screen.virtual_energy[None, :]) @ cv.T.conj()
        return jnp.real(jnp.sum(weights * (density + .01 * weighted))
                        + jnp.sum(occupied_weight * screen.occupied_coeff)
                        + jnp.sum(screen.occupied_energy**2))

    reference = jax.value_and_grad(lambda *a: functional(primary._select_target_reference, *a),
                                    argnums=(0, 1, 2, 3))(*arguments)
    bucketed = jax.value_and_grad(lambda *a: functional(primary._select_target, *a),
                                   argnums=(0, 1, 2, 3))(*arguments)
    for got, want in zip(jax.tree_util.tree_leaves(bucketed), jax.tree_util.tree_leaves(reference)):
        np.testing.assert_allclose(got, want, atol=2e-8, rtol=2e-7)


@pytest.mark.parametrize('zero_occupied,zero_virtual', [(True, False), (False, True)])
def test_bucket_padding_does_not_create_physical_rank(zero_occupied, zero_virtual):
    common = SimpleNamespace(s1e=jnp.eye(7), fock=jnp.diag(jnp.arange(7.)-2))
    candidate = jnp.eye(7)[:, 1:4]
    rhs = jnp.eye(7)[:, :1]
    if zero_occupied:
        rhs = rhs * 0
    if zero_virtual:
        candidate = candidate * 0
    kwargs = dict(ao=np.arange(7, dtype=np.int32), thresholds=BoysDomainThresholds(), index=0)
    actual, labels = primary._select_target(common, candidate, rhs, **kwargs)
    expected, reference_labels = primary._select_target_reference(common, candidate, rhs, **kwargs)
    assert actual is expected is None
    for got, want in zip(labels, reference_labels):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize('different_support,deficient,complex_values', [
    (False, False, False), (True, False, True), (True, True, False),
])
def test_bucketed_projection_matches_independent_product_response(
        different_support, deficient, complex_values):
    rng = np.random.default_rng(301)

    def random(shape):
        value = rng.normal(size=shape)
        return value + .1j * rng.normal(size=shape) if complex_values else value

    n = 43
    root = np.eye(n) + .005 * random((n, n))
    overlap = root.T.conj() @ root
    pao = random((n, 35)) / np.sqrt(n)
    if deficient:
        pao[:, 0] *= 1e-5
    # Reverse these products as independent ambient inputs, as required by
    # the shared PAO product tape. Do not assume their physical symmetry here.
    product = overlap @ pao
    gram = pao.T.conj() @ product
    columns = rng.permutation(35).astype(np.int32)
    support = np.arange(37, dtype=np.int32)
    ao = np.arange(41, dtype=np.int32) if different_support else support[::-1].copy()
    if not different_support:
        support = ao.copy()
    thresholds = BoysDomainThresholds(domain_pao=.01, bp_pao=.05)
    kwargs = dict(columns=columns, support=support, ao=ao, thresholds=thresholds)
    args = tuple(map(jnp.asarray, (overlap, gram, product)))
    weight = jnp.asarray(random((len(ao), len(ao))))

    def evaluate(function, s, g, p):
        return function(SimpleNamespace(s1e=s), (g, p), **kwargs)

    expected, labels = evaluate(primary._select_preprojection_reference, *args)
    actual, selected = evaluate(primary._select_preprojection, *args)
    for got, want in zip(selected, labels):
        np.testing.assert_array_equal(got, want)
    assert actual.shape == expected.shape
    assert actual.shape[1] > 0
    np.testing.assert_allclose(actual @ actual.T.conj(), expected @ expected.T.conj(), atol=2e-10)

    def functional(function, *values):
        coeff, _ = evaluate(function, *values)
        return jnp.real(jnp.sum(weight * (coeff @ coeff.T.conj())))

    expected = jax.value_and_grad(lambda *v: functional(primary._select_preprojection_reference, *v),
                                  argnums=(0, 1, 2))(*args)
    actual = jax.value_and_grad(lambda *v: functional(primary._select_preprojection, *v),
                                argnums=(0, 1, 2))(*args)
    for got, want in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_allclose(got, want, atol=3e-8, rtol=2e-7)


def test_degenerate_projection_keeps_original_completeness_selection():
    rng = np.random.default_rng(442)
    q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    theta = rng.uniform(0, 2*np.pi)
    r = np.array([[0, 0, 1], [np.cos(theta), np.sin(theta), 0],
                  [-np.sin(theta), np.cos(theta), 0]])
    values = np.array([.97, .97, .8])
    virtual = np.concatenate((q @ np.diag(np.sqrt(values)), r @ np.diag(np.sqrt(1-values))))
    projector = virtual @ virtual.T
    pao = projector[:, :3] / np.sqrt(np.diag(projector)[:3])
    common = SimpleNamespace(s1e=jnp.eye(6))
    products = jnp.asarray(pao.T @ pao), jnp.asarray(pao)
    kwargs = dict(columns=np.arange(3), support=np.arange(3), ao=np.arange(5),
                  thresholds=BoysDomainThresholds())
    expected, labels = primary._select_preprojection_reference(common, products, **kwargs)
    actual, selected = primary._select_preprojection(common, products, **kwargs)
    for got, want in zip(selected, labels):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_allclose(actual @ actual.T, expected @ expected.T, atol=2e-11)


@pytest.mark.parametrize('columns', [1, 2])
def test_real_target_with_complex_fock_preserves_reference_dtypes(columns):
    overlap, rhs = jnp.eye(7), jnp.eye(7)[:, :1]
    candidate = jnp.eye(7)[:, 1:1+columns]
    fock = jnp.diag(jnp.arange(7.)-2).astype(complex).at[1, 2].set(.1j).at[2, 1].set(-.1j)
    kwargs = dict(ao=np.arange(7), thresholds=BoysDomainThresholds(), index=0)
    common = SimpleNamespace(s1e=overlap, fock=fock)
    expected, _ = primary._select_target_reference(common, candidate, rhs, **kwargs)
    actual, _ = primary._select_target(common, candidate, rhs, **kwargs)
    for got, want in zip(actual, expected):
        assert got.dtype == want.dtype
        np.testing.assert_allclose(got, want, atol=2e-12)


def test_degenerate_fock_keeps_original_weak_denominator_response():
    rng = np.random.default_rng(818)
    candidate = np.zeros((7, 3))
    candidate[1:4] = rng.normal(size=(3, 3))
    dipole = jnp.asarray(rng.normal(size=7))
    fock = jnp.diag(jnp.array([-2., 1, 1, 1, 3, 4, 5]))
    kwargs = dict(ao=np.arange(7), thresholds=BoysDomainThresholds(), index=0)

    def energy(function, f):
        screen, _ = function(SimpleNamespace(s1e=jnp.eye(7), fock=f), jnp.asarray(candidate),
                             jnp.eye(7)[:, :1], **kwargs)
        return jnp.sum((dipole @ screen.virtual_coeff)**2 /
                       (screen.virtual_energy - screen.occupied_energy[0] + 2))

    expected = jax.value_and_grad(lambda f: energy(primary._select_target_reference, f))(fock)
    actual = jax.value_and_grad(lambda f: energy(primary._select_target, f))(fock)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, atol=2e-11)


def test_degenerate_fock_restores_original_preprojection_too(monkeypatch):
    from pyscfad.dlno_stc import domain, _pairs
    rng = np.random.default_rng(118)
    pao = rng.normal(size=(7, 3))
    pao[0] = 0
    occupied = jnp.eye(7)[:, :1]
    common = domain.DomainData(jnp.eye(7), jnp.diag(jnp.array([-2., 1, 1, 1, 1, 1, 1])),
        occupied, jnp.eye(7)[:, 1:], jnp.array([-2.]), jnp.ones(6), occupied,
        jnp.asarray(pao), (domain.TargetOccupiedData(occupied),))
    preparation = _pairs._cached_weak_screen_products(common, _pairs._weak_screen_products(common))
    products, _ = preparation
    columns, ao, thresholds = np.arange(3), np.arange(7), BoysDomainThresholds()
    monkeypatch.setattr(primary, '_pao_columns_on_atoms', lambda *a: columns)
    monkeypatch.setattr(primary.tools, 'ao_index_by_atom', lambda *a: ao)
    candidate, _ = primary._select_preprojection_reference(common, products.pao,
        columns=columns, support=ao, ao=ao, thresholds=thresholds)
    reference, reference_pb, _ = jax.vjp(lambda c, v, r: primary._select_target_reference(
        c, v, r, ao=ao, thresholds=thresholds, index=0), common, candidate,
        products.occupied[0], has_aux=True)
    *_, (actual, response) = primary.prepare_primary(
        common, preparation, None, None, thresholds, 0, (0,), (0,), {})
    dipole = jnp.asarray(rng.normal(size=7))

    def energy(screen):
        return jnp.sum((dipole @ screen.virtual_coeff)**2 /
                       (screen.virtual_energy - screen.occupied_energy[0] + 2))

    expected_fock_bar = reference_pb(jax.grad(energy)(reference))[0].fock
    actual_fock_bar = response.target_pullback(jax.grad(energy)(actual))[0].fock
    np.testing.assert_allclose(energy(actual), energy(reference), atol=2e-11)
    np.testing.assert_allclose(actual_fock_bar, expected_fock_bar, atol=2e-11)
