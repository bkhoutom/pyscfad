"""The SCF density response has no denominators within equal-occupation blocks."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import eigh as scipy_eigh
from pyscfad.scipy.linalg import eigh
from pyscfad.scf.hf import _density_from_fock, _density_from_fock_rhf


@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('gap', [0., 2e-12, 2e-9, 2e-7, .1])
def test_density_projector_joint_fock_overlap_response(gap, compact):
    rng = np.random.default_rng(191)
    s = rng.normal(size=(6, 6))
    s = s @ s.T + np.eye(6)
    c = np.linalg.inv(np.linalg.cholesky(s).T) @ np.linalg.qr(rng.normal(size=(6, 6)))[0]
    energy = np.array([-1., -1.+gap, -.4, .3, .3+gap, .8])
    occupation = jnp.array([2., 2., 2., 0., 0., 0.])
    f = s @ c @ np.diag(energy) @ c.T @ s
    df = rng.normal(size=(6, 6)); df = .1 * (df + df.T)
    ds = rng.normal(size=(6, 6)); ds = .1 * (ds + ds.T)
    seed = rng.normal(size=(6, 6))

    def density(t):
        ft, st = jnp.asarray(f) + t*df, jnp.asarray(s) + t*ds
        e, v = eigh(ft, st)
        return (_density_from_fock_rhf(ft, st, e, v, 3) if compact else
                _density_from_fock(ft, st, e, v, occupation))

    value, tangent = jax.jvp(density, (jnp.asarray(0.),), (jnp.asarray(1.),))
    reverse = jax.grad(lambda t: jnp.sum(density(t) * seed))(jnp.asarray(0.))
    np.testing.assert_allclose(reverse, np.sum(np.asarray(tangent)*seed), atol=1e-13)
    for h in (1e-4, 3e-5):
        def independent(t):
            _, v = scipy_eigh(f+t*df, s+t*ds)
            return 2 * v[:, :3] @ v[:, :3].T
        fd = (independent(h)-independent(-h))/(2*h)
        np.testing.assert_allclose(tangent, fd, atol=2e-9, rtol=2e-7)
    # Electron count remains constant even as the AO overlap changes.
    np.testing.assert_allclose(np.trace(tangent @ s + value @ ds), 0., atol=2e-13)


def test_equal_occupation_rotations_are_exactly_absent():
    energy = jnp.array([-1., -1.+2e-12, .3, .3+2e-12])
    f, s, occupation = jnp.diag(energy), jnp.eye(4), jnp.array([2., 2., 0., 0.])
    perturbation = jnp.array([[0., 1., 0., 0.], [1., 0., 0., 0.],
                              [0., 0., 0., 1.], [0., 0., 1., 0.]])
    _, derivative = jax.jvp(lambda df: _density_from_fock(df, s, energy, s, occupation),
                           (f,), (perturbation,))
    np.testing.assert_array_equal(derivative, np.zeros((4, 4)))


def test_occupation_response_is_retained():
    energy, c = jnp.array([-1., .2]), jnp.eye(2)
    _, tangent = jax.jvp(lambda n: _density_from_fock(jnp.diag(energy), c, energy, c, n),
                        (jnp.array([1.9, .1]),), (jnp.array([-.2, .2]),))
    np.testing.assert_array_equal(tangent, np.diag([-.2, .2]))


def test_density_response_matches_hermitian_eigh_convention():
    e, c, n = jnp.array([-1., .2]), jnp.eye(2), jnp.array([2., 0.])
    f = jnp.diag(e)
    anti = jnp.array([[0., 1.], [-1., 0.]])
    _, tangent = jax.jvp(lambda a, b: _density_from_fock(a, b, e, c, n),
                        (f, c), (anti, anti))
    np.testing.assert_array_equal(tangent, np.zeros((2, 2)))
    _, reverse = jax.vjp(lambda a, b: _density_from_fock(a, b, e, c, n), f, c)
    fbar, sbar = reverse(jnp.array([[.2, 1.], [0., .3]]))
    np.testing.assert_array_equal(fbar, fbar.T)
    np.testing.assert_array_equal(sbar, sbar.T)


@pytest.mark.parametrize('nocc', [0, 2, 4])
def test_compact_projector_matches_full_jvp_and_vjp(nocc):
    rng = np.random.default_rng(2026)
    c = rng.normal(size=(4, 4))
    e = jnp.array([-1., -.5, .3, .7])
    f, s, df, ds, seed = map(jnp.asarray, rng.normal(size=(5, 4, 4)))
    occupation = jnp.asarray([2.]*nocc+[0.]*(4-nocc))
    full = lambda a, b: _density_from_fock(a, b, e, c, occupation)
    compact = lambda a, b: _density_from_fock_rhf(a, b, e, c, nocc)
    np.testing.assert_allclose(jax.jvp(compact, (f, s), (df, ds)),
                               jax.jvp(full, (f, s), (df, ds)), atol=2e-13)
    np.testing.assert_allclose(jax.vjp(compact, f, s)[1](seed),
                               jax.vjp(full, f, s)[1](seed), atol=2e-13)
