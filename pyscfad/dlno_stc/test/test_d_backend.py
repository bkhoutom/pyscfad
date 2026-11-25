"""Density matrix forward values, precision controls, and delayed replay."""

import math
import warnings

import numpy as np
import pytest

from pyscfad.dlno_stc import backend
from pyscfad.dlno_stc.controls import (
    estimate_matrix_memory,
    normalize_matrix_controls,
)
from pyscfad.dlno_stc.test.test_backend import native
from pyscfad.dlno_stc.test.test_full_kernel import full_fixture
from pyscfad.dlno_stc.test.test_matrix_memory import observation
from pyscfad.dlno_stc.test.test_w_delayed_pullback import grid


def test_deterministic_density_batches_preserve_values_and_pullback():
    native()
    inputs = full_fixture()
    metadata = {'target_indices': (1, 0)}
    shared = backend.solve_matrices(inputs, metadata, grid(), matrix_kind='D')
    split = backend.solve_matrices(
        inputs, metadata, dict(grid(), target_batch_size=1), matrix_kind='D')
    assert shared['matrix_kind'] == shared['diagnostics']['matrix_kind'] == 'D'
    assert shared['diagnostics']['batch_count'] == 1
    assert split['diagnostics']['batch_count'] == 2
    for name in ('energy', 'occupied', 'virtual'):
        np.testing.assert_allclose(split[name], shared[name], rtol=2e-12, atol=2e-13)

    rng = np.random.default_rng(12)
    seeds = dict(energy_bar=.7, occupied_bar=rng.normal(size=(2, 2)),
                 virtual_bar=rng.normal(size=(3, 3)))
    expected = backend.pullback_matrices(inputs, shared, **seeds)
    bars = backend.pullback_matrices(inputs, split, **seeds)
    for name in inputs:
        np.testing.assert_allclose(bars[name], expected[name], rtol=2e-11, atol=2e-12)


def test_stochastic_density_replays_separate_energy_and_matrix_samples():
    native()
    inputs = full_fixture()
    controls = dict(grid(), mode='stochastic', threads=2, global_seed=71,
                    target_batch_size=1, matrix_tolerance=.03, energy_tolerance=.02,
                    pair_pilot_samples=32, min_samples=32, resident_points=2,
                    domain_tau=10., virtual_domain_tau=10., system_workload_cutoff=10.)
    forward = backend.solve_matrices(
        inputs, {'target_indices': (0, 1)}, controls, matrix_kind='D')
    diagnostics = forward['diagnostics']
    assert diagnostics['production_draws'] > 0
    assert diagnostics['energy_standard_error'] > 0
    assert len(diagnostics['batches']) == 2
    assert diagnostics['energy_standard_error'] == pytest.approx(math.hypot(*(
        batch['energy_standard_error'] for batch in diagnostics['batches'])))
    for batch in diagnostics['batches']:
        effective = batch['effective_options']
        assert effective['energy_tolerance'] == pytest.approx(.02 / math.sqrt(2))
        assert effective['system_workload_cutoff'] == 10.
        assert effective['occupied']['resident_points'] == 2
        assert batch['matrix_tolerance'] == pytest.approx(.03 / math.sqrt(2))

    rng = np.random.default_rng(19)
    matrices = dict(occupied_bar=rng.normal(size=(2, 2)), virtual_bar=rng.normal(size=(3, 3)))
    zeros = {name: np.zeros_like(seed) for name, seed in matrices.items()}
    energy = backend.pullback_matrices(inputs, forward, energy_bar=1., **zeros)
    density = backend.pullback_matrices(inputs, forward, energy_bar=0., **matrices)
    combined = backend.pullback_matrices(inputs, forward, energy_bar=1., **matrices)
    for name in inputs:
        np.testing.assert_allclose(combined[name], energy[name] + density[name],
                                   rtol=2e-10, atol=2e-12)


def test_density_replay_rejects_a_different_matrix_kind():
    inputs = full_fixture()
    inputs['B'] = inputs['B'][:0]
    forward = backend.solve_matrices(inputs, {'target_indices': (0,)}, {}, matrix_kind='D')
    seeds = dict(energy_bar=1., occupied_bar=np.eye(2), virtual_bar=np.eye(3))
    bars = backend.pullback_matrices(inputs, forward, **seeds)
    for name, value in inputs.items():
        np.testing.assert_array_equal(bars[name], np.zeros_like(value))
    changed = dict(forward, matrix_kind='W')
    with pytest.raises(ValueError, match='matrix result kind differs'):
        backend.pullback_matrices(inputs, changed, **seeds)
    changed['diagnostics'] = dict(forward['diagnostics'], matrix_kind='W')
    with pytest.raises(ValueError, match='matrix replay fingerprint changed'):
        backend.pullback_matrices(inputs, changed, **seeds)


def test_sampled_d_energy_reverse_forecasts_saved_updates(monkeypatch):
    from pyscfad.dlno_stc import _memory

    native()
    inputs = full_fixture()
    controls = dict(grid(), mode='stochastic', energy_tolerance=.03, matrix_tolerance=.03,
                    pair_pilot_samples=32, min_samples=32, domain_tau=10.,
                    virtual_domain_tau=10., system_workload_cutoff=10.)
    matrix_only = estimate_matrix_memory(4, 2, 3, 1, controls, matrix_kind='D')['total_bytes']
    budget = matrix_only / 1e6 + 1.
    monkeypatch.setattr(_memory, 'observe_memory', lambda: observation(
        rss_mb=0., enforced_limit_mb=budget, allowance_mb=None))
    forward = backend.solve_matrices(
        inputs, {'target_indices': (0,)}, dict(controls, memory_limit_mb=budget), matrix_kind='D')
    # Caller-editable diagnostics must not influence the memory guard.
    forward['diagnostics']['batches'][0]['energy_peak_production_draws'] = 0
    seeds = dict(occupied_bar=np.eye(2), virtual_bar=np.eye(3))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        backend.pullback_matrices(inputs, forward, energy_bar=0., **seeds)
    assert not caught
    with pytest.warns(RuntimeWarning, match='matrix D reverse'):
        backend.pullback_matrices(inputs, forward, energy_bar=1., **seeds)


def test_density_sampling_controls_and_resident_point_memory():
    controls = dict(grid(), mode='stochastic')
    density = normalize_matrix_controls(controls, 'D')
    assert density['energy_tolerance'] == 3e-4
    assert density['system_workload_cutoff'] == .0065
    assert density['resident_points'] == 3
    for key in ('energy_tolerance', 'system_workload_cutoff', 'resident_points'):
        with pytest.raises(ValueError, match=key):
            normalize_matrix_controls(dict(controls, **{key: density[key]}))
    with pytest.raises(ValueError, match='resident_points'):
        normalize_matrix_controls(dict(controls, resident_points=4), 'D')
    with pytest.raises(ValueError, match='matrix_kind'):
        normalize_matrix_controls(controls, 'unsupported')

    default = estimate_matrix_memory(4, 2, 3, 2, controls, matrix_kind='D')
    two_points = estimate_matrix_memory(4, 2, 3, 2, dict(controls, resident_points=2),
                                        matrix_kind='D')
    assert default['compact_caches'] > two_points['compact_caches']
    assert default['total_bytes'] > two_points['total_bytes']
