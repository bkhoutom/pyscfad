"""Saved QR anchors remove internal virtual-gauge response, without freezing it."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad import dlno_stc
from pyscfad.dlno_stc import domain
from pyscfad.dlno_stc.test.test_boys_construction import reference


@pytest.mark.parametrize('width,threshold,rank,ao_grams', [
    (3, .1, 3, 1), (3, 1.5, 2, 1), (9, 1.5, 7, 2)])
def test_metric_space_reuses_gram_without_increasing_contraction_size(
        monkeypatch, width, threshold, rank, ao_grams):
    from pyscfad.dlno_stc import _primary_response
    coeff = jnp.asarray(np.eye(8, width) * np.sqrt(np.arange(1, width + 1)))
    overlap = jnp.eye(8)
    frame, keep = _primary_response._metric_space(coeff, overlap, threshold)
    assert frame.shape[1] == rank
    # Freeze only the eager integer rank decision to inspect the real traced
    # contractions. A second C^H S C duplicates an AO Gram in the first two
    # cases; rotating the Gram in the last case would do larger work instead.
    monkeypatch.setattr(_primary_response, '_keep', lambda values, cutoff: keep)
    traced = jax.make_jaxpr(lambda c, s: _primary_response._metric_space(c, s, threshold)[0])(
        coeff, overlap)
    products_with_overlap = [eqn for eqn in traced.jaxpr.eqns
                             if eqn.primitive.name == 'dot_general'
                             and traced.jaxpr.invars[1] in eqn.invars]
    assert len(products_with_overlap) == ao_grams
    np.testing.assert_allclose(frame.T @ overlap @ frame, np.eye(rank), atol=2e-13)


@pytest.mark.parametrize('threshold', [.1, 1.5])
def test_metric_space_normalization_keeps_full_metric_response(threshold):
    from pyscfad.dlno_stc._primary_response import _metric_space
    rng = np.random.default_rng(139)
    coeff = jnp.asarray(np.eye(6, 3) * np.sqrt([1., 2., 2.]))
    dc = jnp.asarray(rng.normal(size=coeff.shape) * .03)
    ds = rng.normal(size=(6, 6)) * .02
    ds = jnp.asarray(ds + ds.T)
    seed = jnp.asarray(rng.normal(size=(6, 6)))
    def objective(x):
        frame, _ = _metric_space(coeff + x * dc, jnp.eye(6) + x * ds, threshold)
        return jnp.sum((frame @ frame.T) * seed)
    actual = jax.grad(objective)(0.)
    finite = (objective(1e-5) - objective(-1e-5)) / 2e-5
    assert abs(float(actual)) > 1e-3
    np.testing.assert_allclose(actual, finite, atol=2e-8, rtol=2e-7)


@pytest.mark.parametrize('indices', [np.arange(5), np.array([0, 2, 4])])
def test_ao_support_projection_value_and_response(indices):
    rng = np.random.default_rng(91)
    raw = jnp.asarray(rng.normal(size=(5, 5)))
    overlap = raw.T @ raw + jnp.eye(5)
    coeff = jnp.asarray(rng.normal(size=(5, 3)))
    dc = jnp.asarray(rng.normal(size=coeff.shape))
    ds = jnp.asarray(rng.normal(size=overlap.shape)); ds = (ds + ds.T) / 2
    expected = lambda c, s: jnp.linalg.solve(s[np.ix_(indices, indices)], s[indices] @ c)
    actual = lambda c, s: domain._project_to_ao_support(c, s, indices)
    for got, want in zip(jax.jvp(actual, (coeff, overlap), (dc, ds)),
                         jax.jvp(expected, (coeff, overlap), (dc, ds))):
        np.testing.assert_allclose(got, want, rtol=2e-12, atol=2e-12)


def test_saved_anchor_replay_is_invariant_to_internal_rotations_and_keeps_response():
    rng = np.random.default_rng(58)
    parent = rng.normal(size=(5, 4))
    basis = np.linalg.qr(rng.normal(size=(5, 2)))[0]
    direction = rng.normal(size=(5, 2))
    seed = jnp.asarray(rng.normal(size=(5, 2)))
    common = SimpleNamespace(s1e=jnp.eye(5), pao_coeff=jnp.asarray(parent))
    fragment = SimpleNamespace(strong_virtual=SimpleNamespace(parent_columns=np.arange(4)))

    def frame(x, theta):
        raw = domain._cholesky_metric_orthonormalize(
            jnp.asarray(basis) + x * direction, jnp.eye(5))
        rotation = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                              [jnp.sin(theta), jnp.cos(theta)]])
        value, columns = domain._stable_virtual_frame(
            common, fragment, np.arange(5), jnp.eye(5), raw @ rotation, (0, 1))
        assert columns == (0, 1)
        return value

    np.testing.assert_allclose(frame(0., 0.), frame(0., .731), atol=2e-13)
    objective = lambda x, theta: jnp.sum(frame(x, theta) * seed)
    np.testing.assert_allclose(jax.grad(objective, 1)(0., .731), 0., atol=2e-12)
    actual = jax.grad(objective, 0)(0., .731)
    finite = (objective(1e-5, .731) - objective(-1e-5, .731)) / 2e-5
    assert abs(float(actual)) > 1e-3
    np.testing.assert_allclose(actual, finite, atol=2e-8, rtol=2e-7)


def test_real_boys_replay_uses_saved_anchors_without_qr(reference, monkeypatch):
    static = dlno_stc.build_domain_selections(
        reference, frozen=1, target_coeff=reference.mo_coeff[:, 1:5],
        force_full_domains=True,
    )
    common = domain.rebuild_domain_data(reference, static)
    static = dlno_stc.materialize_domain_selections(reference, static, common=common)

    def forbidden_qr(*args, **kwargs):
        raise AssertionError("preparation must reuse saved QR pivots")

    monkeypatch.setattr(domain.scipy_linalg, "qr", forbidden_qr)
    def forbidden_lu(*args, **kwargs):
        raise AssertionError("full-AO frame replay needs no general LU solve")
    monkeypatch.setattr(jnp.linalg, "solve", forbidden_lu)
    assert len(dlno_stc.build_domain_groups(static)) == 1
    for index, fragment in enumerate(static.fragments):
        frame = domain.build_boys_ed_frame(common, static, index)
        assert frame.virtual_anchor_columns == fragment.virtual_anchor_columns
        overlap = np.asarray(common.s1e)[np.ix_(frame.extended_ao_indices,
                                               frame.extended_ao_indices)]
        virtual = np.asarray(frame.virtual_coeff)
        np.testing.assert_allclose(virtual.T @ overlap @ virtual, np.eye(2), atol=1e-9)
        raw, _ = domain._local_ed_pao_space(
            common, fragment.strong_virtual.parent_columns, frame.extended_ao_indices,
            frame.occupied_coeff, static.thresholds, selection=fragment.strong_virtual)
        np.testing.assert_allclose(virtual @ virtual.T, raw @ raw.T, atol=1e-9)
    def coefficients(current):
        rebuilt = domain.build_boys_ed_frame(current, static, 0)
        return rebuilt.occupied_coeff, rebuilt.virtual_coeff
    values, reverse = jax.vjp(coefficients, common)
    bars = reverse(jax.tree_util.tree_map(jnp.ones_like, values))
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(bars))


def test_system_saved_anchors_replay_without_qr(reference, monkeypatch):
    occupied, virtual = np.arange(1, 5), np.array([6, 5])
    frame = domain.build_system_local_frame(reference, occupied, virtual)
    def forbidden_qr(*args, **kwargs):
        raise AssertionError("system preparation must reuse saved QR pivots")
    monkeypatch.setattr(domain.scipy_linalg, "qr", forbidden_qr)
    _, pullback = jax.vjp(lambda mf: domain.build_system_local_frame(
        mf, occupied, virtual, boys_reference=frame.boys_reference,
        virtual_anchor_columns=frame.virtual_anchor_columns).virtual_coeff, reference)
    bar, = pullback(jnp.asarray(np.random.default_rng(4).normal(size=frame.virtual_coeff.shape)))
    assert np.isfinite(np.asarray(bar.mo_coeff)).all()
    np.testing.assert_allclose(frame.virtual_coeff @ frame.virtual_coeff.T,
                               reference.mo_coeff[:, virtual] @ reference.mo_coeff[:, virtual].T,
                               atol=1e-9)
