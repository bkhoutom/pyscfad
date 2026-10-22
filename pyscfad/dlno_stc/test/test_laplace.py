"""Positive fitted quadrature, validation, and unchanged default behavior."""
from types import SimpleNamespace

import numpy as np
import pytest

from pyscfad.dlno_stc import laplace
from pyscfad.dlno_stc.controls import default_laplace_grid


OCC = np.array([-1., -.5])
VIR = np.array([.1, .2, .8])


def test_default_is_exact_independent_and_ignores_fit_parameters():
    roots, weights, diagnostics = laplace.make_laplace_grid(
        points=-1, lambda_gradient=float('nan'))
    expected = default_laplace_grid()
    for actual, old in zip((roots, weights), expected):
        np.testing.assert_array_equal(actual, old)
        assert actual.dtype == np.float64
    assert diagnostics['fitting_seconds'] == 0
    assert diagnostics['denominator_bounds'] is None
    assert diagnostics['max_relative_energy_error'] is None
    roots[:] = weights[:] = 0
    fresh = laplace.make_laplace_grid()
    np.testing.assert_array_equal(fresh[0], expected[0])
    np.testing.assert_array_equal(fresh[1], expected[1])


def test_histogram_is_self_convolution():
    delta, probability = laplace.denominator_histogram(
        np.array([0., -1.]), np.array([1., 2.]), 3, 2., 6.)
    np.testing.assert_allclose(delta, [2, 3, 4, 5, 6])
    np.testing.assert_allclose(probability, np.array([1, 4, 6, 4, 1])/16)


def test_linear_deposition_preserves_mass_mean_and_bounds():
    delta, probability = laplace.denominator_histogram(
        np.array([0.]), np.array([1., 1.25, 2.]), 3, 2., 4.)
    np.testing.assert_allclose(delta, [2, 2.5, 3, 3.5, 4])
    np.testing.assert_allclose(probability, [2.25/9, 1.5/9, 3.25/9, 1/9, 1/9])
    assert np.sum(probability) == pytest.approx(1)
    assert np.dot(delta, probability) == pytest.approx(17/6)


@pytest.mark.parametrize('lam', [0., .1])
def test_joint_jacobian_matches_finite_difference(lam):
    z = np.log([.1, .7, 1.1, .8])
    delta = np.array([.5, 1., 3., 10.])
    probability = np.array([.1, .2, .3, .4])
    _, actual = laplace.residual_and_jacobian(z, delta, probability, lam)
    numerical = np.empty_like(actual)
    for column in range(z.size):
        plus, minus = z.copy(), z.copy()
        plus[column] += 1e-6
        minus[column] -= 1e-6
        numerical[:, column] = (laplace.residual_and_jacobian(plus, delta, probability, lam)[0]
                                - laplace.residual_and_jacobian(minus, delta, probability, lam)[0])/2e-6
    np.testing.assert_allclose(actual, numerical, rtol=1e-8, atol=1e-10)


def test_six_point_fit_meets_energy_and_derivative_kernel_tolerances():
    roots, weights, diagnostics = laplace.make_laplace_grid(OCC, VIR, kind='fitted')
    assert roots.shape == weights.shape == (6,)
    assert roots.dtype == weights.dtype == np.float64
    assert np.all(np.isfinite(roots)) and np.all(roots > 0)
    assert np.all(np.isfinite(weights)) and np.all(weights > 0)
    delta = np.linspace(1.2, 3.6, 10001)
    kernel = np.exp(-delta[:, None]*roots)
    energy_error = np.max(np.abs(delta*(kernel@weights)-1))
    derivative_error = np.max(np.abs(delta**2*(kernel@(roots*weights))-1))
    assert energy_error < 1e-6
    assert derivative_error < 1e-5
    np.testing.assert_allclose(diagnostics['denominator_bounds'], [1.2, 3.6])
    assert diagnostics['max_relative_energy_error'] == pytest.approx(energy_error)
    assert diagnostics['max_relative_derivative_error'] == pytest.approx(derivative_error)
    assert diagnostics['fitting_seconds'] >= 0
    assert diagnostics['weighted_energy_rms'] >= 0
    assert diagnostics['weighted_derivative_rms'] >= 0


@pytest.mark.parametrize('occupied,virtual', [
    (None, VIR), (OCC, None), ([], VIR), (OCC, []),
    ([[0.]], VIR), (OCC, [[1.]]), ([np.nan], VIR), (OCC, [np.inf]),
    ([1.], [1.]), ([2.], [1.]), ([1j], VIR),
])
def test_invalid_spectra_rejected(occupied, virtual):
    with pytest.raises(ValueError):
        laplace.make_laplace_grid(occupied, virtual, kind='fitted')


@pytest.mark.parametrize('points', [0, -1, 1.5, True, '6'])
def test_invalid_point_counts_rejected(points):
    with pytest.raises(ValueError, match='points'):
        laplace.make_laplace_grid(OCC, VIR, kind='fitted', points=points)


@pytest.mark.parametrize('lam', [-1., np.nan, np.inf, 'bad'])
def test_invalid_gradient_weight_rejected(lam):
    with pytest.raises(ValueError, match='lambda_gradient'):
        laplace.make_laplace_grid(OCC, VIR, kind='fitted', lambda_gradient=lam)


def test_unknown_kind_rejected():
    with pytest.raises(ValueError, match='kind'):
        laplace.make_laplace_grid(kind='other')


@pytest.mark.parametrize('points', [1, 6])
def test_single_distinct_denominator_supported(points):
    roots, weights, diagnostic = laplace.make_laplace_grid(
        [-.5, -.5], [.1], kind='fitted', points=points)
    assert roots.size == weights.size == points
    assert np.all(roots > 0) and np.all(weights > 0)
    assert diagnostic['max_relative_energy_error'] < 1e-6
    assert diagnostic['max_relative_derivative_error'] < 1e-5


@pytest.mark.parametrize('failure', ['unsuccessful', 'nonfinite', 'underflow', 'exception'])
def test_all_failed_starts_raise(monkeypatch, failure):
    def failed(fun, z, **kwargs):
        if failure == 'exception':
            raise ValueError('optimizer failure')
        x = np.full_like(z, np.nan if failure == 'nonfinite' else
                         -1000 if failure == 'underflow' else 0)
        return SimpleNamespace(success=failure != 'unsuccessful', x=x, fun=np.zeros(4))
    monkeypatch.setattr(laplace, 'least_squares', failed)
    with pytest.raises(RuntimeError, match='All.*failed'):
        laplace.make_laplace_grid(OCC, VIR, kind='fitted')


def test_failed_start_does_not_hide_later_success(monkeypatch):
    optimize = laplace.least_squares
    calls = 0
    def first_fails(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError('first start failed')
        return optimize(*args, **kwargs)
    monkeypatch.setattr(laplace, 'least_squares', first_fails)
    _, _, diagnostics = laplace.make_laplace_grid(OCC, VIR, kind='fitted')
    assert diagnostics['max_relative_energy_error'] < 1e-6
