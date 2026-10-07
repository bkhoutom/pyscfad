"""Independent tiny full-system formula and sampled-product reverse checks."""
import itertools

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import pytest

from pyscfad.dlno_stc.test.test_backend import fixture, native

KEYS = ('foo', 'fvv', 'B')


def grid(**kwargs):
    return dict(mode='deterministic', laplace_roots=[0., .3, 1.1],
                laplace_weights=[.07, .4, .6], virtual_block_size=2, **kwargs)


def full_fixture(repeated=False):
    return {k: v for k, v in fixture(repeated).items() if k in KEYS}


def reference(x, controls):
    foo, fvv, B = (jnp.asarray(x[k]) for k in KEYS)
    value = 0.
    for beta, weight in zip(controls['laplace_roots'], controls['laplace_weights']):
        T = jnp.einsum('ij,Pjb,ba->Pia', jsp_linalg.expm(.5*beta*foo), B,
                       jsp_linalg.expm(-.5*beta*fvv))
        G = jnp.einsum('Pia,Pkb->ikab', T, T)
        value += weight*jnp.sum(G*(G.transpose(0, 1, 3, 2)-2*G))
    return value


def solve(x, controls=None, **kwargs):
    return native().solve_full(*(x[k] for k in KEYS), grid() if controls is None else controls,
                               **kwargs)


@pytest.mark.parametrize('repeated', [False, True])
def test_full_formula_three_bars_and_symmetric_directions(repeated):
    x, controls = full_fixture(repeated), grid()
    saved = {k: v.copy() for k, v in x.items()}
    out = solve(x, controls, with_grad=True)
    expected, bars = jax.value_and_grad(reference)(x, controls)
    np.testing.assert_allclose(out['energy'], expected, rtol=2e-11, atol=2e-13)
    assert set(out['cotangents']) == set(KEYS)
    assert out['energy_standard_error'] == 0
    for key in KEYS:
        bar = np.asarray(bars[key])
        if key != 'B':
            bar = (bar+bar.T)/2
        np.testing.assert_allclose(out['cotangents'][key], bar, rtol=2e-10, atol=2e-12)
        np.testing.assert_array_equal(x[key], saved[key])
    for key in ('foo', 'fvv'):
        direction = np.zeros_like(x[key]); direction[0, 1] = direction[1, 0] = 1
        plus, minus = dict(x), dict(x)
        plus[key] = x[key]+1e-5*direction
        minus[key] = x[key]-1e-5*direction
        finite = (solve(plus)['energy']-solve(minus)['energy'])/2e-5
        np.testing.assert_allclose(np.sum(out['cotangents'][key]*direction), finite,
                                   rtol=2e-7, atol=2e-10)
    assert 'cotangents' not in solve(x)


def test_full_equals_sum_of_weighted_identity_rows():
    x, controls = full_fixture(), grid()
    full = solve(x, controls, with_grad=True)
    rows = [native().solve(x['foo'], x['fvv'], x['B'], np.eye(2)[i:i+1].copy(),
                           np.eye(2), controls, with_grad=True) for i in range(2)]
    np.testing.assert_allclose(full['energy'], sum(r['energy'] for r in rows), atol=2e-13)
    for key in KEYS:
        np.testing.assert_allclose(full['cotangents'][key],
                                   sum(r['cotangents'][key] for r in rows), atol=2e-12)


def test_full_sampled_partition_reproducibility_and_all_kept():
    x = full_fixture()
    controls = dict(grid(), mode='stochastic', virtual_keep_fraction=.34,
                    production_samples=24000, min_production_samples=2, global_seed=11)
    exact = solve(x, dict(controls, mode='deterministic'), with_grad=True)
    out = solve(x, controls, aux_offsets=np.array([0, 1, 4]), with_grad=True)
    again = solve(x, controls, aux_offsets=np.array([0, 1, 4]), with_grad=True)
    assert out['energy'] == again['energy']
    assert abs(out['energy']-exact['energy']) < 6*out['energy_standard_error']
    assert {(r['term'], r['residual']) for r in out['diagnostics']['residuals']} == {
        ('direct', 1), ('direct', 2), ('exchange', 1), ('exchange', 2)}
    for key in KEYS:
        assert np.isfinite(out['cotangents'][key]).all()
        np.testing.assert_array_equal(out['cotangents'][key], again['cotangents'][key])
    kept = solve(x, dict(controls, virtual_keep_fraction=1), with_grad=True)
    assert kept['energy_standard_error'] == 0
    for key in KEYS:
        np.testing.assert_allclose(kept['cotangents'][key], exact['cotangents'][key], atol=2e-13)
    np.testing.assert_allclose(kept['energy'], exact['energy'], atol=2e-13)


def test_uniform_draw_expectation_zero_products_and_coincident_updates():
    # Enumerate the tiny uniform distribution independently, including every
    # zero-product derivative. Dropping energy-zero draws loses B[0,0,1]'s bar.
    B = np.array([[[1., 0.], [1., 1.]], [[.5, 1.], [0., -.5]]])
    x = dict(foo=-np.eye(2), fvv=np.eye(2), B=B)
    N = 60000
    controls = dict(mode='stochastic', laplace_roots=[0.], laplace_weights=[1.],
                    virtual_keep_fraction=0, uniform_mixture=1.,
                    production_samples=N, min_production_samples=2, global_seed=7)
    means, variances, dropped = [], [], []
    choices = list(itertools.product(range(2), repeat=6))
    for coefficient in (-2., 1.):
        samples, skipped = [], []
        for i, k, a, b, g, h in choices:
            left = (g, i, a), (g, k, b)
            right = ((h, i, a), (h, k, b)) if coefficient == -2 else ((h, i, b), (h, k, a))
            s, t = B[left[0]]*B[left[1]], B[right[0]]*B[right[1]]
            bar = np.zeros_like(B)
            scale = coefficient*len(choices)
            bar[left[0]] += scale*t*B[left[1]]
            bar[left[1]] += scale*t*B[left[0]]
            bar[right[0]] += scale*s*B[right[1]]
            bar[right[1]] += scale*s*B[right[0]]
            samples.append(bar)
            skipped.append(bar if s*t else np.zeros_like(B))
        means.append(np.mean(samples, axis=0)); variances.append(np.var(samples, axis=0)/N)
        dropped.append(np.mean(skipped, axis=0))
    expected, se = sum(means), np.sqrt(sum(variances))
    assert abs(expected[0, 0, 1]-sum(dropped)[0, 0, 1]) > 1
    out = solve(x, controls, aux_offsets=np.array([0, 1, 2]), with_grad=True)
    assert np.all(np.abs(out['cotangents']['B']-expected) < 7*se+1e-12)
    # All four roles coincide, so every draw has derivative -4 exactly.
    scalar = dict(foo=np.array([[-1.]]), fvv=np.array([[1.]]), B=np.ones((1, 1, 1)))
    result = solve(scalar, dict(controls, production_samples=2), with_grad=True)
    assert result['energy'] == -1
    np.testing.assert_allclose(result['cotangents']['B'], -4.)


@pytest.mark.parametrize('empty', ['virtual', 'auxiliary'])
def test_full_empty_shapes_and_cutoff_validation(empty):
    x = full_fixture()
    if empty == 'virtual':
        x['fvv'] = np.empty((0, 0)); x['B'] = np.empty((4, 2, 0))
    else:
        x['B'] = np.empty((0, 2, 3))
    out = solve(x, with_grad=True)
    assert out['energy'] == out['energy_standard_error'] == 0
    for k in KEYS:
        np.testing.assert_array_equal(out['cotangents'][k], np.zeros_like(x[k]))
    for bad in (0., -1., np.nan, np.inf):
        with pytest.raises(ValueError, match='system_workload_cutoff'):
            solve(full_fixture(), grid(system_workload_cutoff=bad))
    solve(full_fixture(), grid(system_workload_cutoff=2.))


def test_absolute_weight_scaled_cutoff_and_adaptive_zero_support():
    scalar = dict(foo=np.array([[-1.]]), fvv=np.array([[1.]]), B=np.ones((1, 1, 1)))
    controls = dict(mode='stochastic', laplace_roots=[0.], laplace_weights=[1.],
                    production_samples=2, min_production_samples=2)
    kept = solve(scalar, dict(controls, system_workload_cutoff=.5), with_grad=True)
    outside = solve(scalar, dict(controls, system_workload_cutoff=1.), with_grad=True)
    scaled = solve(scalar, dict(controls, system_workload_cutoff=.5,
                                laplace_weights=[1/16]), with_grad=True)
    overridden = solve(scalar, dict(controls, system_workload_cutoff=2.,
                                    virtual_keep_fraction=1), with_grad=True)
    assert kept['diagnostics']['actual_sample_count'] == 0
    assert overridden['diagnostics']['actual_sample_count'] == 0
    assert outside['diagnostics']['actual_sample_count'] == 4
    assert scaled['diagnostics']['actual_sample_count'] == 4
    assert kept['energy'] == outside['energy'] == overridden['energy'] == -1
    assert scaled['energy'] == -1/16
    # All proposal scores zero still retain support and independent pilot streams.
    scalar['B'][:] = 0
    adaptive = dict(controls, energy_tolerance=.01, pilot_samples=8,
                    min_production_samples=8, max_production_samples=10)
    del adaptive['production_samples']
    zero = solve(scalar, adaptive, with_grad=True)
    assert zero['energy'] == zero['energy_standard_error'] == 0
    rows = zero['diagnostics']['residuals']
    assert len(rows) == 2
    assert all(row['pilot_samples'] == row['production_samples'] == 8 for row in rows)
    assert len({row[key] for row in rows for key in ('pilot_seed', 'production_seed')}) == 4
    for key in KEYS:
        np.testing.assert_array_equal(zero['cotangents'][key], np.zeros_like(scalar[key]))
