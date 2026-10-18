"""Saved QR anchors remove internal virtual-gauge response, without freezing it."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad import dlno_stc
from pyscfad.dlno_stc import domain
from pyscfad.dlno_stc.test.test_boys_construction import reference


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

    def forbidden_qr(*args, **kwargs):
        raise AssertionError("preparation must reuse saved QR pivots")

    monkeypatch.setattr(domain.scipy_linalg, "qr", forbidden_qr)
    assert len(dlno_stc.build_domain_groups(static)) == 1
    for index, fragment in enumerate(static.fragments):
        frame = domain.build_boys_ed_frame(common, static, index)
        assert frame.virtual_anchor_columns == fragment.virtual_anchor_columns
        overlap = np.asarray(common.s1e)[np.ix_(frame.extended_ao_indices,
                                               frame.extended_ao_indices)]
        virtual = np.asarray(frame.virtual_coeff)
        np.testing.assert_allclose(virtual.T @ overlap @ virtual, np.eye(2), atol=1e-9)
        raw = domain._rebuild_local_virtual_subspace(
            common, fragment.strong_virtual, frame.extended_ao_indices,
            frame.occupied_coeff, boys_metric_threshold=static.thresholds.metric_rank)
        np.testing.assert_allclose(virtual @ virtual.T, raw @ raw.T, atol=1e-9)


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
