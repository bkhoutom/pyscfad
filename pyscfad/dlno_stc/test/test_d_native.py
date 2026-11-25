"""Native selected D contractions and replay of the explicit MP2 energy."""
import numpy as np
from scipy.linalg import expm

from pyscfad.dlno_stc.test.test_backend import native
from pyscfad.dlno_stc.test.test_full_kernel import full_fixture


def test_d_contracts_two_amplitudes_and_computes_mp2_energy():
    extension = native()
    foo, fvv, factors = full_fixture().values()
    targets = (1,)
    controls = dict(laplace_roots=[.2, .5], laplace_weights=[.4, .6])
    amplitudes = np.zeros((1, 2, 3, 3))
    for root, weight in zip(controls['laplace_roots'], controls['laplace_weights']):
        dressed = np.einsum('ij,pjb,ba->pia', expm(root*foo), factors, expm(-root*fvv))
        amplitudes -= weight * np.einsum('psa,pjb->sjab', dressed[:, targets], dressed)
    bare = np.einsum('psa,pjb->sjab', factors[:, targets], factors)
    spin_amplitudes = 2*amplitudes - amplitudes.swapaxes(-1, -2)
    energy = np.sum(bare * spin_amplitudes)
    occupied = np.einsum('sjab,skab->jk', amplitudes, spin_amplitudes)
    virtual = .5 * (np.einsum('sjac,sjbc->ab', amplitudes, spin_amplitudes)
                    + np.einsum('sjca,sjcb->ab', amplitudes, spin_amplitudes))

    def solve(inputs):
        return extension.solve_matrices(*inputs, targets, controls, matrix_kind='D')

    inputs = (foo, fvv, factors)
    forward = solve(inputs)
    for key, expected in (('energy', energy), ('occupied', occupied),
                          ('virtual', virtual)):
        np.testing.assert_allclose(forward[key], expected, atol=2e-13, rtol=2e-12)
    default = extension.solve_matrices(*inputs, targets, controls)
    assert default['diagnostics']['matrix_kind'] == 'W'
    assert forward['diagnostics']['matrix_kind'] == 'D'
    np.testing.assert_allclose(forward['energy'], default['energy'], atol=2e-13)

    rng = np.random.default_rng(38)
    occupied_seed, virtual_seed = rng.normal(size=(2, 2)), rng.normal(size=(3, 3))
    energy_seed = .7
    bars = extension.pullback_matrices(*inputs, forward['replay'], energy_seed,
                                       occupied_seed, virtual_seed)
    directions = [rng.normal(size=value.shape) for value in inputs]
    for i in (0, 1):
        directions[i] = .5 * (directions[i] + directions[i].T)

    def seeded_value(result):
        return (energy_seed*result['energy']
                + np.sum(occupied_seed*result['occupied'])
                + np.sum(virtual_seed*result['virtual']))

    step = 1e-5
    for i, key in enumerate(('foo', 'fvv', 'B')):
        plus, minus = list(inputs), list(inputs)
        plus[i] = inputs[i] + step*directions[i]
        minus[i] = inputs[i] - step*directions[i]
        finite = (seeded_value(solve(plus)) - seeded_value(solve(minus))) / (2*step)
        np.testing.assert_allclose(np.sum(bars[key]*directions[i]), finite,
                                   atol=3e-10, rtol=2e-7)


def test_sampled_d_replay_combines_energy_and_density_seeds():
    extension = native()
    inputs = tuple(full_fixture().values())
    targets = (0, 1)
    controls = dict(mode='stochastic', laplace_roots=[.2, .5], laplace_weights=[.4, .6],
                    energy_tolerance=.03, system_workload_cutoff=10., matrix_tolerance=.03,
                    pair_pilot_samples=32, min_samples=32, global_seed=74,
                    domain_tau=10., virtual_domain_tau=10.)
    offsets = np.array([0, 2, inputs[-1].shape[0]])
    forward = extension.solve_matrices(*inputs, targets, controls, matrix_kind='D',
                                       aux_offsets=offsets)
    diagnostics = forward['diagnostics']
    assert diagnostics['production_draws'] > 0
    assert 0 < diagnostics['energy_pilot_draws'] <= diagnostics['pilot_draws']
    assert 0 < diagnostics['energy_peak_production_draws'] < diagnostics['energy_production_draws']
    assert diagnostics['energy_production_draws'] <= diagnostics['production_draws']
    assert diagnostics['energy_standard_error'] > 0
    assert diagnostics['effective_options']['energy_tolerance'] == controls['energy_tolerance']
    assert diagnostics['effective_options']['system_workload_cutoff'] == controls['system_workload_cutoff']
    assert forward['energy'] < 0.

    def pullback(energy, occupied, virtual):
        return extension.pullback_matrices(*inputs, forward['replay'],
                                           energy, occupied, virtual)

    rng = np.random.default_rng(71)
    occupied, virtual = rng.normal(size=(2, 2)), rng.normal(size=(3, 3))
    energy = pullback(.7, np.zeros_like(occupied), np.zeros_like(virtual))
    density = pullback(0., occupied, virtual)
    joint = pullback(.7, occupied, virtual)
    assert np.linalg.norm(energy['B']) > 0
    assert np.linalg.norm(density['B']) > 0
    for name in joint:
        np.testing.assert_allclose(joint[name], energy[name] + density[name],
                                   atol=2e-12, rtol=2e-10)
