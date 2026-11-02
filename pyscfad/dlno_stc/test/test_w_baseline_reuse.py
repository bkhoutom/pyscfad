"""Deterministic W forwards also supply exactly their owned baseline energy."""
import jax
import numpy as np
import pytest

from pyscfad.dlno_stc.test.test_full_kernel import reference
from pyscfad.dlno_stc.test.test_shared_ccsd_pullback import group_case


def assert_matches_separate_baseline(case, state, options):
    """Compare against a full selected native solve and correction-only VJP."""
    correction, _ = case.run(**dict(options, include_baseline=False))
    if not options.get('include_baseline', True):
        assert 'baseline_energy' not in state
        np.testing.assert_allclose(state['energy'], correction['energy'], atol=2e-13)
        assert state['variance'] == 0.
        for name in ('mf_bar', 'common_bar'):
            np.testing.assert_allclose(state[name], correction[name], atol=2e-13)
        return
    inputs, pullback = jax.vjp(case.prepare, .2, -.1)
    controls = options.get('baseline_controls', case.controls)
    baseline = case.solve(inputs, {'fragment_id': 0,
                                  'target_indices': options.get('group_targets', (0, 1))},
                          controls, with_grad=options.get('with_grad', True))
    np.testing.assert_allclose(state['baseline_energy'], baseline['energy'], atol=2e-13)
    np.testing.assert_allclose(state['energy'], baseline['energy'] + correction['energy'],
                               atol=2e-13)
    np.testing.assert_allclose(state['variance'], float(baseline['energy_standard_error'])**2,
                               rtol=1e-13, atol=1e-16)
    if options.get('with_grad', True):
        baseline_bars = pullback(baseline['cotangents'])
        for name, bar in zip(('mf_bar', 'common_bar'), baseline_bars):
            np.testing.assert_allclose(state[name], correction[name] + bar,
                                       rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize('with_grad', [False, True])
def test_matching_w_forwards_replace_full_baseline_without_changing_response(group_case, with_grad):
    case = group_case
    # Equal grid values must match across independent containers and harmless
    # batching/seed differences; comparing control dictionaries is too strict.
    options = dict(matrix_kind='W',
                   baseline_controls=dict(case.controls, global_seed=11, target_batch_size=2,
                                          energy_tolerance=3e-4),
                   matrix_controls=dict(case.controls, global_seed=29,
                                        laplace_roots=list(case.controls['laplace_roots']),
                                        laplace_weights=list(case.controls['laplace_weights'])))
    state, rows = case.run(**options, with_grad=with_grad)
    assert case.baseline_targets == []
    assert case.replay_energy_bars == ([1., 1.] if with_grad else [])
    assert len(case.preparations[0]) == int(with_grad)

    # Full-system Laplace contraction is an independent baseline oracle; the
    # correction-only path supplies the unchanged CC and direct host response.
    baseline_energy, baseline_bars = jax.value_and_grad(
        lambda mf, common: reference(case.prepare(mf, common), case.controls),
        argnums=(0, 1))(.2, -.1)
    correction, correction_rows = case.run(**options, include_baseline=False,
                                            with_grad=with_grad)
    np.testing.assert_allclose(state['baseline_energy'], baseline_energy, atol=2e-13)
    np.testing.assert_allclose(state['energy'], baseline_energy + correction['energy'], atol=2e-13)
    assert state['variance'] == 0.
    assert [row.fragment_index for row in rows] == [0, 1]
    for row, correction_row in zip(rows, correction_rows):
        assert row.matrix_energy_diagnostic == correction_row.matrix_energy_diagnostic
    if not with_grad:
        assert float(state['mf_bar']) == .17 and float(state['common_bar']) == -.23
        return
    for key, baseline_bar in zip(('mf_bar', 'common_bar'), baseline_bars):
        np.testing.assert_allclose(state[key], correction[key] + baseline_bar,
                                   rtol=2e-11, atol=2e-12)
    step = 1e-5
    for name, dm, dc, initial in [('mf_bar', step, 0., .17),
                                ('common_bar', 0., step, -.23)]:
        plus, _ = case.run(**options, with_grad=False, mf=.2+dm, common=-.1+dc)
        minus, _ = case.run(**options, with_grad=False, mf=.2-dm, common=-.1-dc)
        np.testing.assert_allclose(state[name] - initial,
                                   (plus['energy'] - minus['energy']) / (2*step),
                                   rtol=2e-7, atol=2e-9)


@pytest.mark.parametrize('options,baseline_targets,energy_seeds', [
    pytest.param({'indices': (1,)}, [(0,)], [1.], id='partial-selection'),
    pytest.param({'indices': (0,), 'with_grad': False}, [(1,)], [], id='partial-forward'),
    pytest.param({'indices': ()}, [(0, 1)], [], id='empty-selection'),
    pytest.param({'include_baseline': False}, [], [0., 0.], id='baseline-disabled'),
    pytest.param({'include_baseline': False, 'indices': ()}, [], [], id='empty-no-baseline'),
    pytest.param({'indices': None, 'fragments': ((0, (1, 0)),)}, [], [1.],
                 id='whole-multitarget-fragment'),
    pytest.param({'group_targets': (0,), 'indices': (0,), 'fragments': ((0, (0, 1)),)},
                 [(0,)], [0.], id='fragment-extends-beyond-baseline'),
    pytest.param({'fragments': ((0, (0,)), (1, (0, 1)))}, [(1,)], [1., 0.],
                 id='overlapping-fragment-keeps-residual'),
    pytest.param({'fragments': ((0, (0, 1)), (1, (1,)))}, [], [1., 0.],
                 id='covered-target-never-counted-twice'),
])
def test_only_whole_executed_w_targets_replace_baseline(group_case, options,
                                                      baseline_targets, energy_seeds):
    case = group_case
    options = dict(options, matrix_kind='W')
    state, _ = case.run(**options)
    assert case.baseline_targets == baseline_targets
    assert case.replay_energy_bars == energy_seeds
    contributes = options.get('include_baseline', True) or bool(options.get('indices', (0, 1)))
    assert len(case.preparations[0]) == int(options.get('with_grad', True) and contributes)
    for bars, snapshot in case.packets:
        for name in bars:
            np.testing.assert_array_equal(bars[name], snapshot[name])
    assert_matches_separate_baseline(case, state, options)


@pytest.mark.parametrize('fallback', ['D', 'roots', 'weights', 'baseline-stochastic',
                                    'matrix-stochastic', 'native-sampling-option'])
def test_ineligible_controls_keep_independent_baseline(group_case, fallback):
    case = group_case
    baseline = dict(case.controls)
    matrix = dict(case.controls)
    if fallback in ('roots', 'weights'):
        name = 'laplace_' + fallback
        matrix[name] = np.array(matrix[name], copy=True)
        matrix[name][0] += .1
    elif fallback == 'baseline-stochastic':
        baseline.pop('threads')
        baseline.update(mode='stochastic', production_samples=4, min_production_samples=2,
                        pilot_samples=2, system_workload_cutoff=2.)
    elif fallback == 'matrix-stochastic':
        matrix.update(mode='stochastic', matrix_tolerance=1., pair_pilot_samples=4,
                      min_samples=2, domain_tau=0., virtual_domain_tau=0.)
    elif fallback == 'native-sampling-option':
        baseline['pilot_samples'] = 2
    options = dict(matrix_kind='D' if fallback == 'D' else 'W',
                   baseline_controls=baseline, matrix_controls=matrix)
    state, _ = case.run(**options)
    assert case.baseline_targets == [(0, 1)]
    assert case.replay_energy_bars == [0., 0.]
    assert len(case.preparations[0]) == 1
    assert_matches_separate_baseline(case, state, options)


@pytest.mark.parametrize('invalid', [
    {'unknown_baseline_control': 1}, {'threads': 0}, {'target_batch_size': 0},
    {'pilot_samples': 1}, {'virtual_keep_fraction': 0.},
    {'target_batch_size': 2**63}, {'virtual_block_size': 2**63},
], ids=['unknown', 'threads', 'target-batch', 'pilot-samples', 'keep-fraction',
        'target-batch-overflow', 'virtual-block-overflow'])
def test_w_reuse_preserves_native_baseline_control_validation(group_case, invalid):
    case = group_case
    with pytest.raises((ValueError, RuntimeError)):
        case.run(matrix_kind='W', baseline_controls=dict(case.controls, **invalid),
                 matrix_controls=dict(case.controls))
