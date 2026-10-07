"""Native adaptive policy regressions without a large molecular calculation."""
import numpy as np
import pytest

from pyscfad.dlno_stc.test.test_backend import native


def scalar_inputs(zero=False):
    return np.array([[-1.]]), np.array([[1.]]), np.full((1, 1, 1), 0. if zero else 1.)


def adaptive(**kwargs):
    return dict(mode='stochastic', laplace_roots=[0.], laplace_weights=[1.],
                virtual_keep_fraction=0., energy_tolerance=.3, **kwargs)


def test_full_adaptive_uses_collaborator_pilot_and_production_minimum():
    out = native().solve_full(*scalar_inputs(), adaptive(), with_grad=True)
    rows = out['diagnostics']['residuals']
    assert len(rows) == 2
    assert all(row['pilot_samples'] == 100000 for row in rows)
    assert all(row['production_samples'] == 10000 for row in rows)
    assert out['energy'] == -1.
    assert out['energy_standard_error'] == 0.
    np.testing.assert_allclose(out['cotangents']['B'], -4., atol=2e-12)


@pytest.mark.parametrize('shift', [0., 1e5])
def test_full_point_budgets_follow_trace_model(shift):
    controls = adaptive()
    controls.update(laplace_roots=[0., np.log(2.)/1.4], laplace_weights=[1., 1.])
    foo, fvv, B = scalar_inputs(zero=True)
    out = native().solve_full(foo+shift, fvv+shift, B, controls)
    # trace product=exp(-beta); model scores are 1 and 1/2.
    # The variance budget .3^2 therefore splits as .06 and .03.
    targets = out['diagnostics'].get('point_variance_targets')
    assert targets is not None, 'weighted point budgets were not reported'
    np.testing.assert_allclose(targets, [.06, .03], rtol=2e-14)


def test_adaptive_production_can_exceed_one_million_draws():
    foo, fvv = np.array([[-1.]]), np.eye(2)
    B = np.array([[[1., 2.]]])
    controls = adaptive(pilot_samples=4096, global_seed=7, uniform_mixture=1.)
    controls['energy_tolerance'] = .04
    out = native().solve_full(foo, fvv, B, controls)
    rows = out['diagnostics']['residuals']
    assert max(row['production_samples'] for row in rows) > 1000000
    assert out['diagnostics']['actual_sample_count'] == sum(row['production_samples'] for row in rows)
    np.testing.assert_allclose(out['energy_standard_error']**2,
                               sum(row['variance_of_mean'] for row in rows), rtol=2e-14)
    # Uniform samples have population variances 2124 (direct), 531 (exchange).
    # Independent energy oracle: -(1^2+2^2)^2=-25.
    assert abs(out['energy']+25.) < 6*out['energy_standard_error']
    assert out['energy_standard_error'] <= 1.05*controls['energy_tolerance']


def test_difficult_full_pilots_refine_with_independent_streams():
    foo, fvv = np.array([[-1.]]), np.eye(2)
    B = np.array([[[1., 2.]]])
    controls = adaptive(pilot_samples=64, global_seed=7, uniform_mixture=1.,
                        min_production_samples=2, max_production_samples=64)
    controls['energy_tolerance'] = .01
    out = native().solve_full(foo, fvv, B, controls)
    rows = out['diagnostics']['residuals']
    assert all(row['pilot_samples'] == 1000000 for row in rows)
    assert all(row['production_samples'] == 64 for row in rows)
    seeds = [row[key] for row in rows
             for key in ('pilot_seed', 'pilot_refinement_seed', 'production_seed')]
    assert len(set(seeds)) == len(seeds)


@pytest.mark.parametrize('scope', ['full', 'domain'])
def test_unrepresentable_adaptive_budget_raises_instead_of_truncating(scope):
    rng = np.random.default_rng(71)
    foo, fvv = -np.eye(2), np.eye(2)
    B = rng.normal(size=(2, 2, 2))
    controls = adaptive(pilot_samples=64, global_seed=7, uniform_mixture=1.)
    controls['energy_tolerance'] = 1e-300
    with pytest.raises(OverflowError, match='sampling budget'):
        if scope == 'full':
            native().solve_full(foo, fvv, B, controls, np.array([0, 1, 2]))
        else:
            native().solve(foo, fvv, B, np.ones((1, 2)), np.eye(2), controls,
                           np.array([0, 1, 2]))
