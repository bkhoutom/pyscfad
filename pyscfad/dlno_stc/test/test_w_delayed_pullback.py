"""Shared fragment propagation and delayed energy/W adjoints."""
import numpy as np
import pytest

from pyscfad.dlno_stc import backend
from pyscfad.dlno_stc.controls import estimate_matrix_memory
from pyscfad.dlno_stc.test.test_backend import native
from pyscfad.dlno_stc.test.test_full_kernel import full_fixture, grid as energy_grid


def grid():
    return {key: value for key, value in energy_grid().items() if key != 'virtual_block_size'}


@pytest.mark.parametrize('matrix_kind', ['D', 'W'])
def test_default_matrix_call_shares_all_fragment_targets(matrix_kind):
    native()
    x = full_fixture()
    metadata = dict(fragment_id=7, target_indices=(0, 1))
    shared = backend.solve_matrices(x, metadata, grid(), matrix_kind=matrix_kind)
    split = backend.solve_matrices(x, metadata, dict(grid(), target_batch_size=1),
                                   matrix_kind=matrix_kind)
    assert shared['diagnostics']['batch_count'] == 1
    assert split['diagnostics']['batch_count'] == 2
    assert tuple(shared['replay'].batches) and 'cotangents' not in shared
    for key in ('energy', 'occupied', 'virtual'):
        np.testing.assert_allclose(shared[key], split[key], rtol=2e-12, atol=2e-13)
    seeds = dict(energy_bar=.7, occupied_bar=np.eye(2), virtual_bar=np.eye(3) * .4)
    a = backend.pullback_matrices(x, shared, **seeds)
    b = backend.pullback_matrices(x, split, **seeds)
    for key in x:
        np.testing.assert_allclose(a[key], b[key], rtol=2e-11, atol=2e-12)
    all_memory = estimate_matrix_memory(4, 2, 3, 2, grid(), matrix_kind=matrix_kind)
    one_memory = estimate_matrix_memory(4, 2, 3, 2, dict(grid(), target_batch_size=1),
                                         matrix_kind=matrix_kind)
    assert all_memory['amplitude_buffers'] > one_memory['amplitude_buffers']


def test_stochastic_w_delayed_energy_seed_uses_the_saved_w_sample():
    native()
    x = full_fixture()
    controls = dict(grid(), mode='stochastic', threads=2, global_seed=71,
                    matrix_tolerance=.03, pair_pilot_samples=32, min_samples=32,
                    domain_tau=10., virtual_domain_tau=10.)
    forward = backend.solve_matrices(x, dict(target_indices=(0, 1)), controls,
                                     matrix_kind='W')
    assert forward['diagnostics']['batch_count'] == 1
    assert forward['diagnostics']['production_draws'] > 0
    assert 'cotangents' not in forward
    np.testing.assert_allclose(forward['energy'],
        .5 * (np.trace(forward['occupied']) + np.trace(forward['virtual'])), atol=2e-13)
    rng = np.random.default_rng(719)
    wo, wv = rng.normal(size=(2, 2)), rng.normal(size=(3, 3))
    zero = dict(occupied_bar=np.zeros((2, 2)), virtual_bar=np.zeros((3, 3)))
    energy = backend.pullback_matrices(x, forward, energy_bar=1., **zero)
    downstream = backend.pullback_matrices(x, forward, energy_bar=0.,
                                           occupied_bar=wo, virtual_bar=wv)
    combined = backend.pullback_matrices(x, forward, energy_bar=1.,
                                         occupied_bar=wo, virtual_bar=wv)
    trace = backend.pullback_matrices(x, forward, energy_bar=0.,
                                      occupied_bar=.5*np.eye(2), virtual_bar=.5*np.eye(3))
    for key in x:
        np.testing.assert_allclose(combined[key], energy[key] + downstream[key],
                                   rtol=2e-10, atol=2e-12)
        np.testing.assert_allclose(energy[key], trace[key], rtol=2e-11, atol=2e-12)


def test_w_energy_requires_matching_baseline_and_matrix_quadratures():
    from pyscfad.dlno_stc import ccsd
    baseline = dict(laplace_roots=[.1], laplace_weights=[.5])
    matrix = dict(laplace_roots=[.2], laplace_weights=[.5])
    with pytest.raises(ValueError, match='Laplace grids must match'):
        ccsd._normalize_controls(baseline, matrix, 'W')
