"""Shared fragment propagation and delayed energy/W adjoints."""
import numpy as np
import pytest

from pyscfad.dlno_stc import backend
from pyscfad.dlno_stc.controls import estimate_matrix_memory
from pyscfad.dlno_stc.test.test_backend import native
from pyscfad.dlno_stc.test.test_full_kernel import full_fixture, grid as energy_grid


def grid():
    return {key: value for key, value in energy_grid().items() if key != 'virtual_block_size'}


def test_w_consumes_only_native_df_copy_and_replay_remains_reusable():
    native()
    x = full_fixture()
    original = x['B'].copy()
    x['B'].flags.writeable = False
    seeds = dict(energy_bar=.7, occupied_bar=np.eye(2), virtual_bar=np.eye(3)*.4)
    forward = backend.solve_matrices(x, dict(target_indices=(0, 1)),
                                     dict(grid(), threads=2), matrix_kind='W')
    first = backend.pullback_matrices(x, forward, **seeds)
    second = backend.pullback_matrices(x, forward, **seeds)
    np.testing.assert_array_equal(x['B'], original)
    for name in first:
        np.testing.assert_allclose(first[name], second[name], rtol=2e-12, atol=2e-13)


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


def test_four_target_w_batches_preserve_values_and_all_input_cotangents():
    native()
    rng = np.random.default_rng(20261103)
    foo, fvv = rng.normal(size=(4, 4)), rng.normal(size=(5, 5))
    x = dict(foo=(foo + foo.T) * .02 - np.eye(4),
             fvv=(fvv + fvv.T) * .02 + np.eye(5),
             B=rng.normal(size=(6, 4, 5)) * .1)
    seeds = dict(energy_bar=.7, occupied_bar=rng.normal(size=(4, 4)),
                 virtual_bar=rng.normal(size=(5, 5)))
    outputs = []
    for batch_size in (1, 2, 4):
        forward = backend.solve_matrices(
            x, dict(target_indices=(0, 1, 2, 3)),
            dict(grid(), target_batch_size=batch_size, threads=2), matrix_kind='W')
        bars = backend.pullback_matrices(x, forward, **seeds)
        assert forward['diagnostics']['batch_count'] == 4 // batch_size
        outputs.append((forward, bars))
    reference, reference_bars = outputs[0]
    for forward, bars in outputs[1:]:
        for name in ('energy', 'occupied', 'virtual'):
            np.testing.assert_allclose(forward[name], reference[name], rtol=2e-12, atol=2e-13)
        for name in x:
            np.testing.assert_allclose(bars[name], reference_bars[name], rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize('matrix_kind', ['D', 'W'])
def test_native_matrix_timing_receipts_preserve_cotangents(matrix_kind, monkeypatch):
    extension = native()
    if not hasattr(extension, 'pullback_matrices_with_timings'):
        pytest.skip('installed STC extension predates native timing receipts')
    x = full_fixture()
    metadata = dict(fragment_id=7, target_indices=(0, 1))
    seeds = dict(energy_bar=.7, occupied_bar=np.eye(2), virtual_bar=np.eye(3)*.4)
    events, reverse_batches = [], []
    original = extension.pullback_matrices_with_timings

    def timed(*args):
        bars, timings = original(*args)
        reverse_batches.append(timings)
        return bars, timings

    monkeypatch.setattr(extension, 'pullback_matrices_with_timings', timed)
    with backend.resource_profile.capture_events(events.append):
        forward = backend.solve_matrices(x, metadata, dict(grid(), target_batch_size=1),
                                         matrix_kind=matrix_kind)
        bars = backend.pullback_matrices(x, forward, **seeds)
    assert set(bars) == set(x)
    forward_timings = forward['diagnostics']['native_timings']
    reverse_timings = forward['diagnostics']['native_reverse_timings']
    forward_batches = [row['native_timings'] for row in forward['diagnostics']['batches']]
    for aggregate, batches in ((forward_timings, forward_batches),
                               (reverse_timings, reverse_batches)):
        assert len(batches) == 2
        assert aggregate == {name: sum(row[name] for row in batches) for name in aggregate}
        assert all(np.isfinite(seconds) and seconds >= 0 for seconds in aggregate.values())
        exclusive = sum(aggregate[name] for name in (
            'propagation_seconds', 'amplitude_seconds', 'contraction_seconds',
            'seed_seconds', 'reverse_seconds'))
        assert exclusive <= aggregate['native_total_seconds'] + 1e-8
    assert forward_timings['reverse_seconds'] == 0
    assert reverse_timings['reverse_seconds'] > 0
    receipts = [event for event in events if event['phase'].startswith('stc.matrix.native_')]
    assert [event['phase'] for event in receipts] == [
        'stc.matrix.native_forward_timings', 'stc.matrix.native_reverse_timings']
    assert all(event['details']['frag'] == 8 for event in receipts)
    assert receipts[0]['details']['timings'] == forward_timings
    assert receipts[1]['details']['timings'] == reverse_timings

    # The original native API and older-extension Python fallback still return
    # only the three numerical cotangents, with identical numerical results.
    arguments = (x['foo'], x['fvv'], x['B'], forward['replay'].batches[0],
                 seeds['energy_bar'], seeds['occupied_bar'], seeds['virtual_bar'])
    plain = extension.pullback_matrices(*arguments)
    measured, _ = original(*arguments)
    assert set(plain) == set(x)
    for name in x:
        np.testing.assert_array_equal(plain[name], measured[name])
    monkeypatch.delattr(extension, 'pullback_matrices_with_timings')
    fallback = backend.pullback_matrices(x, forward, **seeds)
    assert forward['diagnostics']['native_reverse_timings'] == {}
    for name in x:
        np.testing.assert_array_equal(fallback[name], bars[name])


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
