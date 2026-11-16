"""Singleton normalization preserves the cutoff and live overlap response."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import _primary_response, _selection


@pytest.fixture(params=['reference', 'response'])
def scalar_metric(request, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('a scalar metric needs no eigensolve')
    if request.param == 'reference':
        monkeypatch.setattr(_selection, '_eigh_keep_numpy', forbidden)
        return _selection._reference_metric_space
    monkeypatch.setattr(_primary_response, '_eigh', forbidden)
    return _primary_response._metric_space


@pytest.mark.parametrize('amplitude,threshold,rank', [
    (2., 3., 1), (2., 4., 0), (2., 5., 0), (0., 0., 0)])
def test_scalar_metric_keeps_strict_cutoff_without_eigensolve(
        scalar_metric, amplitude, threshold, rank):
    frame, keep = scalar_metric(jnp.array([[amplitude], [0.]]), jnp.eye(2), threshold)
    np.testing.assert_array_equal(keep, np.arange(rank, dtype=np.int32))
    assert keep.dtype == np.int32 and frame.shape == (2, rank)
    if rank:
        np.testing.assert_allclose(frame, [[1.], [0.]], atol=1e-14)


def test_empty_metric_needs_no_eigensolve(scalar_metric):
    frame, keep = scalar_metric(jnp.zeros((2, 0)), jnp.eye(2), 1e-10)
    assert frame.shape == (2, 0) and keep.shape == (0,)


@pytest.mark.parametrize('value', [np.nan, np.inf])
def test_scalar_metric_rejects_nonfinite_norm(scalar_metric, value):
    with pytest.raises(RuntimeError, match='nonfinite'):
        scalar_metric(jnp.ones((1, 1)), jnp.array([[value]]), 1e-10)


def test_scalar_normalization_retains_coefficient_and_overlap_derivatives(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('a scalar metric needs no eigensolve')
    monkeypatch.setattr(_primary_response, '_eigh', forbidden)
    coeff = jnp.array([[.7], [-.2], [.5]])
    dc = jnp.array([[.1], [.2], [-.3]])
    overlap = jnp.diag(jnp.array([1.2, 1.5, .9]))
    ds = jnp.array([[.2, .1, 0.], [.1, -.1, .05], [0., .05, .3]])
    hamiltonian = jnp.diag(jnp.array([-.5, .1, .9]))
    def objective(x):
        frame, _ = _primary_response._metric_space(coeff+x*dc, overlap+x*ds, 1e-10)
        return jnp.sum(frame * (hamiltonian @ frame))
    def rayleigh(x):
        c, s = coeff+x*dc, overlap+x*ds
        return jnp.sum(c * (hamiltonian @ c)) / jnp.sum(c * (s @ c))
    value, reverse = jax.value_and_grad(objective)(0.)
    _, forward = jax.jvp(objective, (0.,), (1.,))
    step = 1e-5
    finite = (rayleigh(step)-rayleigh(-step))/(2*step)
    assert abs(float(finite)) > .01
    np.testing.assert_allclose(value, rayleigh(0.), atol=2e-14)
    np.testing.assert_allclose([reverse, forward], finite, atol=2e-10, rtol=2e-8)
