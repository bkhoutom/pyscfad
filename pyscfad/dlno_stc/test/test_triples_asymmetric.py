"""Direct S contraction: exact algebra, all cotangents, and sampled moments."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc.test.test_triples import triples_fixture, triples_native, triples_matrix_reference


def inputs_and_reference(grouped=True, zero_left=False):
    inputs = triples_fixture(grouped=grouped)
    u = inputs.pop('U')
    rng = np.random.default_rng(409)
    inputs['fov'] = rng.normal(scale=.03, size=inputs['t1'].shape)
    left, right = rng.normal(size=u.shape), rng.normal(size=u.shape)
    if zero_left:
        left[:] = 0.
    inputs['eocc'][:] = -1/3
    inputs['evir'][:] = 0.
    inputs.pop('eocc')
    inputs.pop('evir')
    inputs = dict(inputs, L=left, R=right)
    def reference(x):
        no, nv = x['t1'].shape
        # The slow LNO integral layout transposes the same-space DF pair;
        # match that convention explicitly, including nonsymmetric directions.
        return triples_matrix_reference(x['t1'], x['t2'], x['Bvo'],
            x['Boo'].swapaxes(1, 2), x['Bvv'].swapaxes(1, 2),
            -jnp.ones(no)/3, jnp.zeros(nv), x['L'] @ x['R'].T, x['fov'])
    energy, bars = jax.value_and_grad(reference)({k:jnp.asarray(v) for k,v in inputs.items()})
    return inputs, dict(energy=float(energy), cotangents=bars)



@pytest.mark.parametrize('grouped', [False, True])
@pytest.mark.parametrize('zero_left', [False, True])
def test_exact_asymmetric_energy_and_every_cotangent(grouped, zero_left):
    # Arbitrary L/R, hence a nonsymmetric M, catches projector orientation.
    # Zero L tests derivative support when the energy itself is zero.
    inputs, expected = inputs_and_reference(grouped, zero_left)
    result = triples_native().triples_asymmetric_energy_vjp(**inputs, nsamples=0)
    np.testing.assert_allclose(result['energy'], expected['energy'], atol=2e-15)
    assert result['samples'] == result['standard_error'] == 0
    for name in inputs:
        np.testing.assert_allclose(result['cotangents'][name], expected['cotangents'][name],
                                   rtol=2e-12, atol=2e-14, err_msg=name)


@pytest.mark.parametrize('zero_left', [False, True])
def test_sampled_asymmetric_energy_and_derivatives(zero_left):
    inputs, expected = inputs_and_reference(zero_left=zero_left)
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
    inputs, _ = inputs_and_reference()
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
    inputs, _ = inputs_and_reference()
    with pytest.raises((TypeError, ValueError), match='R'):
        triples_native().triples_asymmetric_energy_vjp(**dict(inputs, R=value), nsamples=2)


def test_only_asymmetric_native_entry_points_remain():
    native = triples_native()
    assert {name for name in dir(native) if name.startswith('triples_')} == {
        'triples_asymmetric_energy', 'triples_asymmetric_energy_vjp'}
    assert 'asymmetric-only' in native.TRIPLES_VERSION


def binding_inputs():
    values = triples_fixture(grouped=True)
    u = values.pop('U')
    values.pop('eocc'); values.pop('evir')
    return dict(values, L=u, R=u)


@pytest.mark.parametrize('name', ['t1', 't2', 'Bvo', 'Boo', 'Bvv', 'fov', 'L', 'R'])
@pytest.mark.parametrize('invalid', ['dtype', 'nan', 'rank'])
def test_native_input_contract(name, invalid):
    values = binding_inputs()
    if invalid == 'dtype':
        values[name] = values[name].astype(np.float32)
    elif invalid == 'nan':
        values[name] = np.full_like(values[name], np.nan)
    else:
        values[name] = values[name].ravel()
    with pytest.raises((ValueError, TypeError), match=name):
        triples_native().triples_asymmetric_energy_vjp(**values, nsamples=2)


@pytest.mark.parametrize('value', [1, -1, True, 2.5])
def test_invalid_native_sample_counts(value):
    with pytest.raises((ValueError, TypeError, OverflowError)):
        triples_native().triples_asymmetric_energy(**binding_inputs(), nsamples=value)


@pytest.mark.parametrize('value', [-1., np.nan, np.inf, True, '1e-4'])
def test_invalid_native_tolerance(value):
    with pytest.raises((ValueError, TypeError), match='target_standard_error'):
        triples_native().triples_asymmetric_energy(**binding_inputs(), nsamples=2,
                                                 target_standard_error=value)


def test_native_strided_inputs_and_shared_sample_derivatives():
    values = binding_inputs()
    function = triples_native().triples_asymmetric_energy_vjp
    expected = function(**values, nsamples=257, seed=19)
    strided = {}
    for name, value in values.items():
        storage = np.zeros(tuple(2*n for n in value.shape))
        view = storage[tuple(slice(None, None, -2) for _ in value.shape)]
        view[...] = value
        strided[name] = view
    actual = function(**strided, nsamples=257, seed=19)
    assert actual['energy'] == expected['energy']
    for name in values:
        np.testing.assert_array_equal(actual['cotangents'][name], expected['cotangents'][name])
    energy_only = triples_native().triples_asymmetric_energy(**values, nsamples=257, seed=19)
    # Gradient support can alter proposal weights; each call reports its own SE.
    assert np.isfinite(energy_only['energy']) and energy_only['samples'] == 257
