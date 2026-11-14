"""CPU LAPACK backend, batching, and derivative regression tests."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from pyscfad.scipy.linalg import eigh


def _problem(dtype, n=5):
    rng = np.random.default_rng(82)
    x = rng.normal(size=(n, n))
    y = rng.normal(size=(n, n))
    if np.issubdtype(dtype, np.complexfloating):
        x = x + 1j * rng.normal(size=(n, n))
        y = y + 1j * rng.normal(size=(n, n))
    a = ((x + x.conj().T) / 2).astype(dtype)
    b = (y @ y.conj().T + np.eye(n)).astype(dtype)
    return a, b


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize('itype', [1, 2, 3])
@pytest.mark.parametrize('lower', [True, False])
def test_generalized_cpu_backend(dtype, itype, lower):
    a, b = _problem(dtype)
    tol = 3e-5 if np.dtype(dtype).itemsize <= (8 if np.iscomplexobj(a) else 4) else 2e-12
    a, b = jax.device_put((a, b), jax.local_devices(backend='cpu')[0])
    solve = jax.jit(partial(eigh, type=itype, lower=lower))
    w, v = map(np.asarray, solve(a, b))
    expected = scipy.linalg.eigh(a, b, type=itype, lower=lower, driver='gvd')[0]
    np.testing.assert_allclose(w, expected, atol=tol, rtol=tol)
    lhs = a @ v if itype == 1 else (a @ b @ v if itype == 2 else b @ a @ v)
    rhs = (b @ v if itype == 1 else v) * w
    np.testing.assert_allclose(lhs, rhs, atol=tol * np.linalg.norm(lhs), rtol=tol)
    metric = np.linalg.inv(b) if itype == 3 else b
    np.testing.assert_allclose(v.conj().T @ metric @ v, np.eye(len(w)), atol=tol, rtol=tol)


def test_batched_and_standard_eigh():
    a, b = _problem(np.float64)
    aa = np.stack([a, 2 * a])
    w, _ = jax.jit(jax.vmap(eigh, in_axes=(0, None)))(aa, b)
    expected = scipy.linalg.eigh(a, b)[0]
    np.testing.assert_allclose(w, [expected, 2 * expected], atol=1e-12)
    w, _ = jax.jit(eigh)(aa, np.stack([b, b]))
    np.testing.assert_allclose(w, [expected, 2 * expected], atol=1e-12)
    np.testing.assert_allclose(eigh(a, eigvals_only=True), scipy.linalg.eigvalsh(a), atol=1e-12)


@pytest.mark.parametrize('complex_problem', [False, True])
def test_generalized_derivatives(complex_problem):
    a, b = _problem(np.complex128 if complex_problem else np.float64)
    da, db = _problem(a.dtype)
    db *= 0.1

    def objective(a, b):
        w, v = eigh(a, b)
        # Phase-independent, but sensitive to eigenvectors and the B metric.
        return jnp.sum(jnp.sin(w)) + jnp.sum(jnp.abs(v[0]) ** 2 * jnp.arange(5))

    value, tangent = jax.jit(lambda a, b: jax.jvp(objective, (a, b), (jnp.array(da), jnp.array(db))))(a, b)
    ga, gb = jax.jit(jax.grad(objective, argnums=(0, 1)))(a, b)
    reverse = np.real(np.sum(ga * da) + np.sum(gb * db))
    eps = 1e-5
    fd = (objective(a + eps * da, b + eps * db) - objective(a - eps * da, b - eps * db)) / (2 * eps)
    assert np.isfinite(value)
    np.testing.assert_allclose(tangent, fd, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(reverse, tangent, atol=2e-12, rtol=2e-12)


def test_degeneracy_and_non_positive_metric():
    a = jnp.diag(jnp.array([1., 1., 3.]))
    _, tangent = jax.jvp(eigh, (a,), (jnp.ones_like(a),))
    assert all(np.isfinite(t).all() for t in tangent)
    w, v = jax.jit(eigh)(a, -jnp.eye(3))
    assert np.isnan(w).all() and np.isnan(v).all()
