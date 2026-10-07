"""Tiny independent references for the optional weighted native backend."""
import importlib

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import pytest

KEYS = ('foo', 'fvv', 'B', 'target_projection', 'partner_weight')


def native():
    # Optional when absent; an installed extension with broken dependencies fails.
    import os
    if importlib.util.find_spec('pyscfad.dlno_stc._stc_mp2') is None:
        if os.environ.get('PYSCFAD_REQUIRE_STC_EXTENSION'):
            pytest.fail('weighted C++ extension has not been built')
        pytest.skip('optional weighted C++ extension is not built')
    return importlib.import_module('pyscfad.dlno_stc._stc_mp2')


def fixture(repeated=False):
    rng = np.random.default_rng(41)
    foo = np.array([[-.8, .07], [.07, -.5]])
    fvv = np.array([[.3, -.02, .04], [-.02, .6, .01], [.04, .01, .9]])
    if repeated:
        foo = -.6 * np.eye(2)
        fvv = .4 * np.eye(3) + np.diag([0., 1e-13, 0.])
    return dict(zip(KEYS, [foo, fvv, rng.normal(size=(4, 2, 3)) / 5,
                          np.array([[.6, -.27]]),
                          np.array([[.8, .13], [.13, .4]])]))


def controls():
    # Finite interval Gaussian grid makes quadrature independently controllable.
    nodes, weights = np.polynomial.legendre.leggauss(48)
    return dict(mode='deterministic', laplace_roots=(nodes + 1) * 15,
                laplace_weights=weights * 15, virtual_block_size=2)


def reference(x, grid):
    foo, fvv, B, M, W = (jnp.asarray(x[k]) for k in KEYS)
    # Four-index tensor only in this tiny independent oracle.
    g = jnp.einsum('Pia,Pkb->ikab', B, B)
    t = jnp.zeros_like(g)
    for beta, weight in zip(grid['laplace_roots'], grid['laplace_weights']):
        O = jsp_linalg.expm(beta * foo)
        V = jsp_linalg.expm(-beta * fvv)
        A = jnp.einsum('ij,Pjb,ba->Pia', O, B, V)
        t = t - weight * jnp.einsum('Pia,Pkb->ikab', A, A)
    return jnp.einsum('ij,kl,ikab,jlab->', M.T @ M, W, t,
                      2 * g - g.transpose(0, 1, 3, 2))


def solve(x, grid=None, **kwargs):
    return native().solve(*(x[k] for k in KEYS), controls() if grid is None else grid,
                          **kwargs)


@pytest.mark.parametrize('repeated', [False, True])
def test_weighted_energy_five_bars_layout_and_degenerate_spectra(repeated):
    x, grid = fixture(repeated), controls()
    original = {k: a.copy() for k, a in x.items()}
    result = solve(x, grid, aux_offsets=np.array([0, 1, 4]), with_grad=True)
    energy, bars = jax.value_and_grad(reference)(x, grid)
    np.testing.assert_allclose(result['energy'], energy, rtol=2e-11, atol=2e-13)
    assert result['energy_standard_error'] == 0
    for k in KEYS:
        expected = np.asarray(bars[k])
        if k in ('foo', 'fvv', 'partner_weight'):
            expected = (expected + expected.T) / 2
        np.testing.assert_allclose(result['cotangents'][k], expected,
                                   rtol=2e-10, atol=2e-12)
        np.testing.assert_array_equal(x[k], original[k])
        assert result['cotangents'][k].flags.owndata
    # An off-diagonal perturbation tests the full symmetric matrix convention.
    for k in ('foo', 'fvv', 'partner_weight'):
        direction = np.zeros_like(x[k]); direction[0, 1] = direction[1, 0] = 1
        plus, minus = dict(x), dict(x)
        plus[k] = x[k] + 1e-5 * direction
        minus[k] = x[k] - 1e-5 * direction
        finite = (solve(plus)['energy'] - solve(minus)['energy']) / 2e-5
        np.testing.assert_allclose(np.sum(result['cotangents'][k] * direction),
                                   finite, rtol=2e-7, atol=2e-10)
    assert 'cotangents' not in solve(x)


def test_quadrature_separately_matches_exact_denominator():
    x = fixture()
    eo, Co = np.linalg.eigh(x['foo']); ev, Cv = np.linalg.eigh(x['fvv'])
    B = np.einsum('ij,Pia,ab->Pjb', Co, x['B'], Cv)
    M = x['target_projection'] @ Co
    W = Co.T @ x['partner_weight'] @ Co
    g = np.einsum('Pia,Pkb->ikab', B, B)
    d = eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]
    exact = np.einsum('ij,kl,ikab,jlab->', M.T @ M, W, g / d,
                      2*g - g.transpose(0, 1, 3, 2))
    np.testing.assert_allclose(solve(x)['energy'], exact, rtol=2e-11, atol=2e-13)


def test_empty_virtual_domain_and_invalid_buffers():
    x = fixture()
    x['fvv'] = np.empty((0, 0)); x['B'] = np.empty((4, 2, 0))
    result = solve(x, with_grad=True)
    assert result['energy'] == result['energy_standard_error'] == 0
    for k in KEYS:
        np.testing.assert_array_equal(result['cotangents'][k], np.zeros_like(x[k]))
    x = fixture(); x['B'] = np.asfortranarray(x['B'])
    with pytest.raises((ValueError, TypeError), match='contiguous'):
        solve(x)
    x = fixture(); x['fvv'] -= 10 * np.eye(3)
    with pytest.raises(ValueError, match='gap'):
        solve(x)
    x = fixture(); grid = controls(); grid['laplace_roots'][0] = -1
    with pytest.raises(ValueError, match='roots'):
        solve(x, grid)


def stochastic_grid():
    return dict(mode='stochastic', laplace_roots=[0., .3, 1.1],
                laplace_weights=[.07, .4, .6], virtual_block_size=2,
                virtual_keep_fraction=.34, production_samples=12000,
                min_production_samples=2)


def test_sampled_mixed_products_joint_reverse_and_total_variance():
    x, grid = fixture(), stochastic_grid()
    exact = solve(x, dict(grid, mode='deterministic'), with_grad=True)
    rng = np.random.default_rng(29)
    direction = {k: rng.normal(size=v.shape) for k, v in x.items()}
    for k in ('foo', 'fvv', 'partner_weight'):
        direction[k] = (direction[k]+direction[k].T)/2
    expected = sum(np.sum(exact['cotangents'][k]*direction[k]) for k in KEYS)
    derivatives, energies, errors = [], [], []
    for seed in range(6):
        result = solve(x, dict(grid, global_seed=seed),
                       aux_offsets=np.array([0, 1, 4]), with_grad=True)
        energies.append(result['energy']); errors.append(result['energy_standard_error'])
        derivatives.append(sum(np.sum(result['cotangents'][k]*direction[k]) for k in KEYS))
        streams = result['diagnostics']['residuals']
        assert {row['term'] for row in streams} == {'direct', 'exchange'}
        assert {row['residual'] for row in streams} == {1, 2}
        assert all(row['production_samples'] == 12000 for row in streams)
        np.testing.assert_allclose(result['energy_standard_error']**2,
                                   sum(row['variance_of_mean'] for row in streams), rtol=2e-14)
    assert abs(np.mean(energies)-exact['energy']) < 5*np.linalg.norm(errors)/6
    assert abs(np.mean(derivatives)-expected) < 5*np.std(derivatives, ddof=1)/np.sqrt(6)
    # Frozen realized choices give exactly the same energy with/without bars.
    energy_only = solve(x, dict(grid, global_seed=0), aux_offsets=np.array([0, 1, 4]))
    assert energy_only['energy'] == energies[0]
    assert 'cotangents' not in energy_only
    all_kept = solve(x, dict(grid, virtual_keep_fraction=1), with_grad=True)
    np.testing.assert_allclose(all_kept['energy'], exact['energy'], atol=2e-14)
    assert all_kept['energy_standard_error'] == 0


def test_zero_weight_samples_still_reverse_and_adaptive_minimum():
    x, grid = fixture(), stochastic_grid()
    x['partner_weight'] = np.zeros((2, 2))
    expected = solve(x, dict(grid, mode='deterministic'), with_grad=True)
    derivatives = []
    for seed in range(6):
        result = solve(x, dict(grid, global_seed=seed), aux_offsets=np.array([0, 1, 4]), with_grad=True)
        assert result['energy'] == result['energy_standard_error'] == 0
        derivatives.append(np.sum(result['cotangents']['partner_weight']))
    target = np.sum(expected['cotangents']['partner_weight'])
    assert abs(target) > 1e-4
    assert abs(np.mean(derivatives)-target) < 5*np.std(derivatives, ddof=1)/np.sqrt(6)
    adaptive = dict(grid, energy_tolerance=.01, pilot_samples=16,
                    min_production_samples=8, max_production_samples=100)
    del adaptive['production_samples']
    result = solve(x, adaptive, with_grad=True)
    assert result['energy'] == 0
    assert all(row['production_samples'] >= 8 and row['pilot_samples'] == 16
               for row in result['diagnostics']['residuals'])
    assert np.linalg.norm(result['cotangents']['partner_weight']) > 0


@pytest.mark.parametrize('occupied,virtual,beta', [
    (-2., 2., 1e308),
    (1e308, 1.1e308, 1.),
])
def test_exponential_underflow_and_large_common_shift_return_finite_zero_bars(
    occupied, virtual, beta,
):
    # One occupied/virtual scalar: E=-exp(2*beta*(foo-fvv)). Both cases
    # underflow to zero, and every physical derivative underflows to zero.
    # The first catches inf-inf in the VJP; the second catches midpoint overflow.
    x = dict(zip(KEYS, [np.array([[occupied]]), np.array([[virtual]]),
                        np.ones((1, 1, 1)), np.ones((1, 1)), np.ones((1, 1))]))
    grid = dict(mode='deterministic', laplace_roots=[beta], laplace_weights=[1.])
    result = solve(x, grid, with_grad=True)
    assert result['energy'] == result['energy_standard_error'] == 0.
    for key in KEYS:
        assert np.isfinite(result['cotangents'][key]).all()
        np.testing.assert_array_equal(result['cotangents'][key], np.zeros_like(x[key]))
