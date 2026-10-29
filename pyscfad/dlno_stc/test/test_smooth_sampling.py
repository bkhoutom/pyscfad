"""Energy budgets, independent pilots and ten-point triples quadrature."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import smooth_triples as triples
from pyscfad.dlno_stc.smooth import BlockFrame


def test_ten_point_grid_covers_c6h14_triples_denominators():
    nodes, weights, diagnostics = triples.make_grid(np.array([-.9, -.3]), np.array([.33, 3.263333333]))
    assert len(nodes) == len(weights) == diagnostics['points'] == 10
    assert np.all(nodes > 0) and np.all(weights > 0)
    assert diagnostics['max_relative_energy_error'] < 1e-7
    assert diagnostics['max_relative_derivative_error'] < 1e-6


@pytest.mark.parametrize('exponent', [1.4, 2.1, 5.0])
def test_trace_model_matches_mp2_and_is_invariant_to_energy_zero(exponent):
    # One occupied/virtual pair; the trace-product scores are 1 and 1/2.
    nodes, weights = np.array([0., np.log(2.)/1.4]), np.ones(2)
    for shift in (0., 1e5):
        fractions = triples.point_variance_fractions(np.array([[-1.+shift]]),
            np.array([[1.+shift]]), nodes, weights, exponent=exponent)
        expected = np.array([1., 2**(-exponent/1.4)])
        np.testing.assert_allclose(fractions, expected/expected.sum(), rtol=2e-14)


@pytest.mark.parametrize('exponent', [1.4, 2.1, 5.0])
def test_pilots_set_counts_for_each_variance_target_and_preserve_gradient(monkeypatch, exponent):
    nodes, weights = np.array([.1, .4]), np.array([.3, .7])
    frames = (BlockFrame(np.eye(1), ((0,),)),) * 2
    calls = []
    sigmas = np.array([2., 5.])
    def native(values, samples, seed, target=0., with_grad=True, *, pilot=None):
        q = len(calls)
        calls.append((samples, seed, with_grad, pilot))
        assert pilot is not None
        samples = max(samples, int(np.ceil((sigmas[q]/pilot['allocation_target'])**2)))
        result = dict(energy=float(np.sum(np.asarray(values[0])**2)),
                      standard_error=sigmas[q]/np.sqrt(samples), samples=samples,
                      pilot_samples=samples, independent_pilot_samples=pilot['samples'],
                      independent_pilot_standard_error=sigmas[q]/np.sqrt(pilot['samples']),
                      independent_pilot_seconds=0.)
        if with_grad:
            result['cotangents'] = dict(zip(triples.NAMES, [2*np.asarray(values[0]),
                *(np.zeros_like(v) for v in values[1:])]))
        return result
    monkeypatch.setattr(triples, '_native', native)
    target = .01
    solver = triples.make_solver(dict(mode='stochastic', nsamples=7,
        target_standard_error=target, global_seed=71, trace_exponent=exponent, adaptive=False),
        frames, (nodes, weights, {}), with_grad=True)
    eris = SimpleNamespace(Lov=jnp.ones((1, 1, 1)), Loo=jnp.ones((1, 1, 1)),
        Lvv=jnp.ones((1, 1)), fock=jnp.diag(jnp.array([-1., 1.])))
    t1 = jnp.array([[.2]])
    def evaluate(t1):
        return solver(None, eris, jnp.eye(1), t1=t1, t2=jnp.ones((1, 1, 1, 1)))
    (energy, diagnostics), grad = jax.value_and_grad(evaluate, has_aux=True)(t1)
    np.testing.assert_allclose(energy, .2**2 * np.sum(weights*np.exp(-2*nodes)), rtol=1e-14)
    np.testing.assert_allclose(grad, .4 * np.sum(weights*np.exp(-2*nodes)), rtol=1e-14)
    assert [call[0] for call in calls[:2]] == [10000, 10000]
    assert all(call[2] for call in calls)
    assert len(calls) == 2  # One native preparation per point, including its pilot.
    assert len({s for call in calls for s in (call[1], call[3]['seed'])}) == 4
    fractions = np.asarray(diagnostics['point_variance_fractions'])
    expected_counts = np.maximum(10000, np.ceil((weights*sigmas)**2/(target**2*fractions)))
    np.testing.assert_array_equal(diagnostics['point_production_samples'], expected_counts)
    np.testing.assert_allclose(np.sum(diagnostics['point_variance_targets']), target**2, rtol=1e-14)
    np.testing.assert_allclose(diagnostics['standard_error']**2,
        np.sum(diagnostics['point_weighted_variances']), rtol=1e-14)
    assert diagnostics['pilot_samples'] == 20000
    assert diagnostics['trace_exponent'] == exponent
    assert diagnostics['standard_error'] <= target
    assert diagnostics['point_preparation_reused']


def test_allocation_maintains_production_for_zero_variance_pilot():
    counts = triples.production_allocation(np.array([0., 2.]), np.array([.5, .5]), .1)
    assert counts[0] == 10000
    assert np.all(counts >= 10000)


@pytest.mark.parametrize('exponent', [0., -1., float('nan'), float('inf'), True])
def test_invalid_trace_exponent_rejected(exponent):
    from pyscfad.dlno_stc.controls import normalize_triples_controls
    with pytest.raises(ValueError, match='trace_exponent'):
        normalize_triples_controls(dict(mode='stochastic', nsamples=2,
            trace_exponent=exponent), ccsd_t=True)


def test_default_exponent_is_explicit_in_checkpoint_controls():
    from pyscfad.dlno_stc.ccsd import _allocate_triples_controls
    static = SimpleNamespace(lis_static=SimpleNamespace(smooth_frames=(object(),)))
    controls = dict(mode='stochastic', nsamples=2)
    resolved = _allocate_triples_controls(controls, 1, static=static)
    assert resolved['trace_exponent'] == 2.1
    assert resolved['adaptive'] is True
    assert resolved['dressing'] == 'asymmetric'
    assert _allocate_triples_controls(dict(controls, dressing='equal'), 1,
        static=static)['dressing'] == 'equal'
    assert _allocate_triples_controls(dict(controls, adaptive=False), 1,
        static=static)['adaptive'] is False
    assert 'trace_exponent' not in controls
    assert _allocate_triples_controls(dict(controls, trace_exponent=1.4),
        1, static=static)['trace_exponent'] == 1.4
    with pytest.raises(ValueError, match='requires the smooth'):
        _allocate_triples_controls(dict(controls, trace_exponent=2.1), 1)
    with pytest.raises(ValueError, match='requires the smooth'):
        _allocate_triples_controls(dict(controls, dressing='asymmetric'), 1)


@pytest.mark.parametrize('mode', ['deterministic', 'stochastic'])
def test_dressing_control_is_validated(mode):
    from pyscfad.dlno_stc.controls import normalize_triples_controls
    controls = dict(mode=mode, **({'nsamples': 2} if mode == 'stochastic' else {}))
    for dressing in ('equal', 'asymmetric'):
        assert normalize_triples_controls(dict(controls, dressing=dressing),
                                          ccsd_t=True)['dressing'] == dressing
    with pytest.raises(ValueError, match='dressing'):
        normalize_triples_controls(dict(controls, dressing='unknown'), ccsd_t=True)


def test_adaptive_production_uses_point_target_and_keeps_extended_gradient(monkeypatch):
    frames = (BlockFrame(np.eye(1), ((0,),)),) * 2
    calls = []
    def native(values, samples, seed, target=0., with_grad=True, *, pilot=None):
        calls.append((samples, seed, target, pilot))
        sigma = 2.
        actual = max(samples, int(np.ceil((sigma/target)**2)))
        result = dict(energy=float(np.sum(np.asarray(values[0])**2)),
            standard_error=sigma/np.sqrt(actual), samples=actual,
            pilot_standard_error=sigma/np.sqrt(samples), pilot_samples=samples,
            independent_pilot_samples=pilot['samples'],
            independent_pilot_standard_error=.1/np.sqrt(pilot['samples']),
            independent_pilot_seconds=0.)
        if with_grad:
            result['cotangents'] = dict(zip(triples.NAMES, [2*np.asarray(values[0]),
                *(np.zeros_like(v) for v in values[1:])]))
        return result
    monkeypatch.setattr(triples, '_native', native)
    solver = triples.make_solver(dict(mode='stochastic', nsamples=2,
        target_standard_error=.001, trace_exponent=5.),
        frames, (np.array([.2]), np.array([1.]), {}), with_grad=True)
    eris = SimpleNamespace(Lov=jnp.ones((1,1,1)), Loo=jnp.ones((1,1,1)),
        Lvv=jnp.ones((1,1)), fock=jnp.diag(jnp.array([-1., 1.])))
    def evaluate(t1):
        return solver(None, eris, jnp.eye(1), t1=t1, t2=jnp.ones((1,1,1,1)))
    (energy, d), grad = jax.value_and_grad(evaluate, has_aux=True)(jnp.array([[.2]]))
    assert len(calls) == 1 and calls[0][0] == 10000
    assert 0 < calls[0][2] <= .001 and calls[0][1] != calls[0][3]['seed']
    assert d['initial_production_samples'] == 10000
    assert d['samples'] > d['initial_production_samples']
    assert d['adaptive_samples'] == d['samples']-d['initial_production_samples']
    assert d['adaptive'] and d['target_met'] and d['standard_error'] <= .001
    np.testing.assert_allclose(energy, .04*np.exp(-.4), rtol=1e-14)
    np.testing.assert_allclose(grad, .4*np.exp(-.4), rtol=1e-14)


@pytest.mark.parametrize('value', [1, 0, 'yes', None, np.nan])
def test_invalid_adaptive_control_rejected(value):
    from pyscfad.dlno_stc.controls import normalize_triples_controls
    with pytest.raises(ValueError, match='adaptive'):
        normalize_triples_controls(dict(mode='stochastic', nsamples=2, adaptive=value), ccsd_t=True)


@pytest.mark.parametrize('error', [1., np.nan])
def test_adaptive_result_must_meet_target(monkeypatch, error):
    frames = (BlockFrame(np.eye(1), ((0,),)),) * 2
    monkeypatch.setattr(triples, '_native', lambda values, samples, seed, target=0., **kw:
        dict(energy=0., standard_error=error if target else .1, samples=samples))
    solver = triples.make_solver(dict(mode='stochastic', nsamples=2, target_standard_error=.001),
        frames, (np.array([.2]), np.array([1.]), {}), with_grad=False)
    eris = SimpleNamespace(Lov=jnp.ones((1,1,1)), Loo=jnp.ones((1,1,1)),
        Lvv=jnp.ones((1,1)), fock=jnp.diag(jnp.array([-1., 1.])))
    with pytest.raises(RuntimeError, match='point SE target'):
        solver(None, eris, jnp.eye(1), t1=jnp.ones((1,1)), t2=jnp.ones((1,1,1,1)))
