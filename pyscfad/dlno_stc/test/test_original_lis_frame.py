"""Original occupied LIS ordering, moving targets and retained-space response."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import lis
from pyscfad.dlno_stc.smooth import (
    AnchoredFrame, orthonormalize, select_occupied_frame,
)


def test_original_lno_complement_is_reused_without_another_qr(monkeypatch):
    import scipy.linalg
    q = np.linalg.qr(np.random.default_rng(63).normal(size=(8, 6)))[0]

    def forbidden(*args, **kwargs):
        pytest.fail('the original truncated LIS already supplies its complement')

    monkeypatch.setattr(scipy.linalg, 'qr', forbidden)
    frame = select_occupied_frame(q, q[:, :2], np.eye(8))
    np.testing.assert_array_equal(frame.anchor, q)
    assert frame.n_internal == 2


def test_target_first_retained_frame_keeps_density_and_target_response():
    rng = np.random.default_rng(964)
    q = np.linalg.qr(rng.normal(size=(6, 6)))[0]
    target = jnp.asarray(q[:, :1])
    w = jnp.asarray(q @ np.diag([0., .01, .2, .2, .8, 1.2]) @ q.T)
    anchor = q[:, [0, 2, 3, 4]]
    split = AnchoredFrame(anchor, n_internal=1)
    mixed = AnchoredFrame(anchor @ np.linalg.qr(rng.normal(size=(4, 4)))[0])
    identity = jnp.eye(6)
    columns = lis._range_columns(lis._external_map(identity, target))
    dw = rng.normal(size=(6, 6)); dw = jnp.asarray((dw + dw.T)/2)
    dt = jnp.asarray(rng.normal(size=(6, 1)))

    def build(x, frame):
        internal = orthonormalize(target + x*dt, identity)
        mapping, density = lis._compact_mapped_density(
            lis._external_map(identity, internal), w+x*dw, columns)
        c = lis._mapped_density_space(density, mapping, internal, (1, 2, 3),
                                      identity, identity, frame)
        return c, internal

    c, dc = jax.jvp(lambda x: build(x, split)[0], (0.,), (1.,))
    h = 1e-5
    fd = (build(h, split)[0]-build(-h, split)[0])/(2*h)
    np.testing.assert_allclose(dc, fd, atol=3e-8, rtol=3e-7)
    np.testing.assert_allclose(c[:, :1], target, atol=2e-15)
    np.testing.assert_allclose(c.T @ c, np.eye(4), atol=2e-14)
    np.testing.assert_allclose(c, anchor, atol=2e-14)
    assert np.linalg.norm(dc[:, :1]) > .1
    assert np.linalg.norm(dc[:, 1:]) > .1
    for x in (0., .002):
        a, t = build(x, split); b, _ = build(x, mixed)
        np.testing.assert_allclose(a[:, :1], t, atol=2e-14)
        np.testing.assert_allclose(a @ a.T, b @ b.T, atol=2e-13)


@pytest.mark.parametrize('full', [False, True])
def test_domain_builder_keeps_targets_first_with_moving_ao_metric(monkeypatch, full):
    rng = np.random.default_rng(448)
    q = np.linalg.qr(rng.normal(size=(7, 7)))[0]
    nocc = 4 if full else 3
    anchor = q[:, :4] if full else q[:, [0, 2, 3]]
    frame = SimpleNamespace(occupied_coeff=jnp.asarray(q[:, :4]),
                            virtual_coeff=jnp.asarray(q[:, 4:]),
                            extended_ao_indices=np.arange(7))
    selection = SimpleNamespace(occupied_lno_keep=np.array([1, 2]),
        full_occupied_space=full, virtual_lno_keep=np.array([0, 1]),
        full_virtual_space=False, occupied_map_columns=np.array([1, 2, 3]))
    static = SimpleNamespace(fragments=(selection,), mp2_static=None,
        smooth_frames=((AnchoredFrame(anchor, n_internal=1),
                        AnchoredFrame(q[:, 4:6])),))
    monkeypatch.setattr(lis, '_domain_targets',
        lambda common, domains, index, current: jnp.eye(4)[:, :1])
    ds = rng.normal(size=(7, 7)); ds = jnp.asarray((ds+ds.T)*.03)

    def build(x):
        metric = jnp.eye(7) + x*ds
        # Move the parent with the metric. The internal target follows this
        # parent, rather than a fixed AO anchor that could mix S and E again.
        parent = orthonormalize(jnp.asarray(q), metric)
        current = SimpleNamespace(occupied_coeff=parent[:, :4],
            virtual_coeff=parent[:, 4:], extended_ao_indices=np.arange(7))
        common = SimpleNamespace(s1e=metric, fock=jnp.eye(7))
        value = lis._build_domain_lis(None, common, static, 0, current,
            jnp.diag(jnp.array([0., .01, .3, .7])),
            jnp.diag(jnp.array([.1, .4, .8])))
        return value.active_occupied_coeff, value.fragment_iao_coeff

    (co, target), (dco, dtarget) = jax.jvp(build, (0.,), (1.,))
    np.testing.assert_allclose(co[:, :1], target, atol=2e-14)
    np.testing.assert_allclose(dco[:, :1], dtarget, atol=2e-13)
    np.testing.assert_allclose(co.T @ co, np.eye(nocc), atol=2e-14)
    np.testing.assert_allclose(dco.T @ co + co.T @ dco + co.T @ ds @ co,
                               0., atol=2e-13)
    h = 1e-5
    np.testing.assert_allclose(dco, (build(h)[0]-build(-h)[0])/(2*h),
                               atol=2e-9, rtol=2e-6)
