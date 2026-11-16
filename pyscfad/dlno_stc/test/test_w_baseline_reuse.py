"""W traces supply the complete domain MP2 energy without a separate solve."""
import jax
import numpy as np
import pytest

from pyscfad.dlno_stc.test.test_full_kernel import reference
from pyscfad.dlno_stc.test.test_shared_ccsd_pullback import group_case


@pytest.mark.parametrize('with_grad', [False, True])
def test_complete_w_baseline_matches_independent_laplace_energy(group_case, with_grad):
    case = group_case
    state, rows = case.run(with_grad=with_grad)
    expected = reference(case.prepare(.2, -.1), case.controls)
    np.testing.assert_allclose(state['baseline_energy'], expected, atol=2e-13)
    correction = sum(row.e_ccsd+row.e_triples-row.e_mp2_lis for row in rows)
    np.testing.assert_allclose(state['energy'], expected+correction, atol=2e-13)
    assert case.baseline_calls == []
    assert case.replay_energy_bars == ([1., 1.] if with_grad else [])


@pytest.mark.parametrize('with_grad', [False, True])
def test_sampled_domain_mp2_is_read_from_w_traces(group_case, monkeypatch, with_grad):
    from pyscfad.dlno_stc import ccsd
    case = group_case
    forwards = []
    solve_matrices = ccsd.backend.solve_matrices
    def observed(*args, **kwargs):
        result = solve_matrices(*args, **kwargs)
        forwards.append(result)
        return result
    monkeypatch.setattr(ccsd.backend, 'solve_matrices', observed)
    matrix = dict(case.controls, mode='stochastic', global_seed=71,
                  matrix_tolerance=.03, pair_pilot_samples=32, min_samples=32,
                  domain_tau=10., virtual_domain_tau=10.)
    state, rows = case.run( matrix_controls=matrix,
                           with_grad=with_grad)
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
