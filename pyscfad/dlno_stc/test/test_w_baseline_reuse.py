"""W traces supply the complete domain MP2 energy without a separate solve."""
import jax
import numpy as np
import pytest

from pyscfad.dlno_stc.test.test_full_kernel import reference
from pyscfad.dlno_stc.test.test_shared_ccsd_pullback import group_case


def assert_matches_native_reference(case, state, options):
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


@pytest.mark.parametrize('options,energy_seeds', [
    pytest.param({'indices': (1,)}, [1., 1.], id='partial-selection'),
    pytest.param({'indices': (0,), 'with_grad': False}, [], id='partial-forward'),
    pytest.param({'indices': ()}, [1., 1.], id='empty-selection'),
    pytest.param({'include_baseline': False}, [0., 0.], id='baseline-disabled'),
    pytest.param({'include_baseline': False, 'indices': ()}, [], id='empty-no-baseline'),
    pytest.param({'indices': None, 'fragments': ((0, (1, 0)),)}, [1.],
                 id='whole-multitarget-fragment'),
])
def test_all_owned_w_targets_supply_mp2_with_partial_cc_selection(group_case, options,
                                                                 energy_seeds):
    case = group_case
    options = dict(options, matrix_kind='W')
    state, _ = case.run(**options)
    assert case.baseline_targets == []
    assert case.replay_energy_bars == energy_seeds
    contributes = options.get('include_baseline', True) or bool(options.get('indices', (0, 1)))
    assert len(case.preparations[0]) == int(options.get('with_grad', True) and contributes)
    for bars, snapshot in case.packets:
        for name in bars:
            np.testing.assert_array_equal(bars[name], snapshot[name])
    assert_matches_native_reference(case, state, options)


@pytest.mark.parametrize('indices', [(0, 1), (1,), ()])
@pytest.mark.parametrize('with_grad', [False, True])
def test_sampled_domain_mp2_is_read_from_w_traces(group_case, monkeypatch, indices, with_grad):
    from pyscfad.dlno_stc import ccsd
    case = group_case
    forwards = []
    solve_matrices = ccsd.backend.solve_matrices
    def observed(*args, **kwargs):
        result = solve_matrices(*args, **kwargs)
        forwards.append(result)
        return result
    def unexpected_baseline(*args, **kwargs):
        pytest.fail('W must not launch a separate MP2 baseline solver')
    monkeypatch.setattr(ccsd.backend, 'solve_matrices', observed)
    monkeypatch.setattr(ccsd.backend, 'solve', unexpected_baseline)
    matrix = dict(case.controls, mode='stochastic', global_seed=71,
                  matrix_tolerance=.03, pair_pilot_samples=32, min_samples=32,
                  domain_tau=10., virtual_domain_tau=10.)
    state, rows = case.run(matrix_kind='W', matrix_controls=matrix,
                           indices=indices, with_grad=with_grad)
    assert len(forwards) == 2
    assert all(f['diagnostics']['production_draws'] > 0 for f in forwards)
    trace_energy = sum(.5 * (np.trace(f['occupied']) + np.trace(f['virtual']))
                       for f in forwards)
    np.testing.assert_allclose(state['baseline_energy'], trace_energy, atol=2e-13)
    correction = sum(row.e_ccsd + row.e_triples - row.e_mp2_lis for row in rows)
    np.testing.assert_allclose(state['energy'], trace_energy + correction, atol=2e-13)
    np.testing.assert_allclose(state['variance'],
        sum(f['diagnostics']['energy_standard_error']**2 for f in forwards), atol=1e-16)
    assert case.replay_energy_bars == ([1., 1.] if with_grad else [])


def test_d_keeps_its_independent_mp2_solver(group_case):
    case = group_case
    options = dict(matrix_kind='D')
    state, _ = case.run(**options)
    assert case.baseline_targets == [(0, 1)]
    assert case.replay_energy_bars == [0., 0.]
    assert_matches_native_reference(case, state, options)
