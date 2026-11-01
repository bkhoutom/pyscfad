"""Prepared pilot/production agrees with independent legacy native calls."""
import numpy as np
import pytest

from pyscfad.dlno_stc.test.test_triples import triples_fixture, triples_native


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('with_grad', [False, True])
@pytest.mark.parametrize('adaptive', [False, True])
def test_reused_pilot_matches_separate_calls(weighted, with_grad, adaptive):
    native = triples_native()
    inputs = triples_fixture()
    inputs.pop('U')
    inputs['fov'] = np.array([[.03, -.02, .01], [-.04, .05, -.06]])
    if weighted:
        inputs['M'] = np.array([[.36, -.30], [-.768, .64]])
    name = 'triples_' + ('weighted' if weighted else 'whole') + '_energy'
    function = getattr(native, name + ('_vjp' if with_grad else ''))
    pilot_count, pilot_seed, production_seed = 4096, 29, 71
    trial = function(**inputs, nsamples=pilot_count, seed=pilot_seed)
    target = trial['standard_error'] * .5
    number = max(10000, int(np.ceil((trial['standard_error'] * np.sqrt(pilot_count) / target)**2)))
    production_target = target * (1-1e-12) if adaptive else 0.
    reference = function(**inputs, nsamples=number, seed=production_seed,
                         target_standard_error=production_target)
    actual = function(**inputs, nsamples=10000, seed=production_seed,
                      target_standard_error=production_target,
                      pilot=dict(samples=pilot_count, seed=pilot_seed,
                                 allocation_target=target))
    assert actual['independent_pilot_samples'] == pilot_count
    assert actual['independent_pilot_standard_error'] == trial['standard_error']
    assert actual['independent_pilot_seconds'] > 0.
    for key in ('energy', 'standard_error', 'samples', 'pilot_samples', 'pilot_standard_error'):
        assert actual[key] == reference[key], key
    assert actual['pilot_samples'] == number
    if adaptive:
        assert actual['standard_error'] <= production_target
    if with_grad:
        for name, value in actual['cotangents'].items():
            np.testing.assert_allclose(value, reference['cotangents'][name],
                                       rtol=1e-10, atol=1e-12, err_msg=name)


@pytest.mark.parametrize('pilot', [
    {}, {'samples': 1, 'seed': 29, 'allocation_target': .001},
    {'samples': True, 'seed': 29, 'allocation_target': .001},
    {'samples': 10, 'seed': 71, 'allocation_target': .001},
    {'samples': 10, 'seed': 29, 'allocation_target': 0.},
    {'samples': 10, 'seed': 29, 'allocation_target': np.nan},
])
def test_invalid_independent_pilot_rejected(pilot):
    inputs = triples_fixture()
    inputs.pop('U')
    with pytest.raises((TypeError, ValueError)):
        triples_native().triples_whole_energy(**inputs, nsamples=10000, seed=71,
                                             pilot=pilot)
