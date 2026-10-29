"""Direct S contraction: exact algebra, all cotangents, and sampled moments."""
import numpy as np
import pytest

from pyscfad.dlno_stc.test.test_triples import triples_fixture, triples_native


def inputs_and_weighted_reference(grouped=True, zero_left=False):
    inputs = triples_fixture(grouped=grouped)
    u = inputs.pop('U')
    rng = np.random.default_rng(409)
    inputs['fov'] = rng.normal(scale=.03, size=inputs['t1'].shape)
    left, right = rng.normal(size=u.shape), rng.normal(size=u.shape)
    if zero_left:
        left[:] = 0.
    inputs['eocc'][:] = -1/3
    inputs['evir'][:] = 0.
    expected = triples_native().triples_weighted_energy_vjp(
        **inputs, M=left @ right.T, nsamples=0)
    inputs.pop('eocc')
    inputs.pop('evir')
    expected['cotangents']['L'] = expected['cotangents']['M'] @ right
    expected['cotangents']['R'] = expected['cotangents']['M'].T @ left
    return dict(inputs, L=left, R=right), expected


@pytest.mark.parametrize('grouped', [False, True])
@pytest.mark.parametrize('zero_left', [False, True])
def test_exact_asymmetric_energy_and_every_cotangent(grouped, zero_left):
    # Arbitrary L/R, hence a nonsymmetric M, catches projector orientation.
    # Zero L tests derivative support when the energy itself is zero.
    inputs, expected = inputs_and_weighted_reference(grouped, zero_left)
    result = triples_native().triples_asymmetric_energy_vjp(**inputs, nsamples=0)
    np.testing.assert_allclose(result['energy'], expected['energy'], atol=2e-15)
    assert result['samples'] == result['standard_error'] == 0
    for name in inputs:
        np.testing.assert_allclose(result['cotangents'][name], expected['cotangents'][name],
                                   rtol=2e-12, atol=2e-14, err_msg=name)


@pytest.mark.parametrize('zero_left', [False, True])
def test_sampled_asymmetric_energy_and_derivatives(zero_left):
    inputs, expected = inputs_and_weighted_reference(zero_left=zero_left)
    draws = [triples_native().triples_asymmetric_energy_vjp(
        **inputs, nsamples=32768, seed=seed) for seed in (17, 29, 37, 53)]
    energy_se = np.sqrt(sum(d['standard_error']**2 for d in draws))/len(draws)
    assert abs(np.mean([d['energy'] for d in draws])-expected['energy']) <= 8*energy_se+1e-12
    for name in inputs:
        bars = np.asarray([d['cotangents'][name] for d in draws])
        # Independent streams test derivatives without differentiating draws.
        se = bars.std(axis=0, ddof=1)/np.sqrt(len(draws))
        assert np.all(abs(bars.mean(axis=0)-expected['cotangents'][name]) <= 10*se+1e-10), name


def test_asymmetric_fov_sample_derivative_and_pilot_reuse():
    inputs, _ = inputs_and_weighted_reference()
    function = triples_native().triples_asymmetric_energy_vjp
    pilot_count = 4096
    pilot = function(**inputs, nsamples=pilot_count, seed=29)
    target = pilot['standard_error']*.5
    count = max(10000, int(np.ceil((pilot['standard_error']*np.sqrt(pilot_count)/target)**2)))
    expected = function(**inputs, nsamples=count, seed=71, target_standard_error=target)
    actual = function(**inputs, nsamples=10000, seed=71, target_standard_error=target,
        pilot=dict(samples=pilot_count, seed=29, allocation_target=target))
    assert actual['independent_pilot_standard_error'] == pilot['standard_error']
    for name in ('energy', 'standard_error', 'samples', 'pilot_samples'):
        assert actual[name] == expected[name]
    assert actual['standard_error'] <= target
    for name in inputs:
        np.testing.assert_allclose(actual['cotangents'][name], expected['cotangents'][name],
                                   rtol=2e-12, atol=2e-14, err_msg=name)
    # Fov is absent from the proposal, so fixed samples give a valid FD check.
    direction = np.random.default_rng(52).normal(size=inputs['fov'].shape)
    step = 1e-4
    options = dict(nsamples=4096, seed=71)
    base = function(**inputs, **options)
    plus = function(**dict(inputs, fov=inputs['fov']+step*direction), **options)
    minus = function(**dict(inputs, fov=inputs['fov']-step*direction), **options)
    np.testing.assert_allclose(np.sum(base['cotangents']['fov']*direction),
        (plus['energy']-minus['energy'])/(2*step), rtol=2e-9, atol=1e-11)


@pytest.mark.parametrize('value', [None, np.ones((3, 2), dtype=np.float32),
                                 np.ones((3, 1)), np.full((3, 2), np.nan)])
def test_asymmetric_invalid_right_projection_rejected(value):
    inputs, _ = inputs_and_weighted_reference()
    with pytest.raises((TypeError, ValueError), match='R'):
        triples_native().triples_asymmetric_energy_vjp(**dict(inputs, R=value), nsamples=2)
