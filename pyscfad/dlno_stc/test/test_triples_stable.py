"""Inverse-free triples against a six-index reference and broad spectra."""
import itertools
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscfad.dlno_stc import _stc_mp2 as native
from pyscfad.dlno_stc.test._triples_dressing_reference import symmetric_exp


def transform(tensor, matrices):
    for axis, matrix in enumerate(matrices):
        tensor = jnp.moveaxis(jnp.tensordot(matrix, tensor, axes=(1, axis)), 0, axis)
    return tensor


def reference(inputs, node):
    """Form the raw numerator first; dress its external legs only."""
    t1, t2, gv, go, gg, fov, foo, fvv, u = inputs
    w = jnp.einsum('iabf,kjcf->ijkabc', gv, t2) - jnp.einsum('iajm,mkbc->ijkabc', go, t2)
    v = jnp.einsum('iajb,kc->ijkabc', gg, t1) + jnp.einsum('ijab,kc->ijkabc', t2, fov)
    permutations = list(itertools.permutations(range(3)))
    w = sum(w.transpose((*p, *(i+3 for i in p))) for p in permutations)
    v = sum(v.transpose((*p, *(i+3 for i in p))) for p in permutations)
    z = w + .5*v
    z = (4*z + z.transpose(1,2,0,3,4,5) + z.transpose(2,0,1,3,4,5)
         - 2*z.transpose(2,1,0,3,4,5) - 2*z.transpose(0,2,1,3,4,5)
         - 2*z.transpose(1,0,2,3,4,5))
    o, v = symmetric_exp(node*foo/2), symmetric_exp(-node*fvv/2)
    left = transform(w, (u.T, o, o, v, v, v))
    right = transform(z, (u.T@symmetric_exp(node*foo), o, o, v, v, v))
    return -jnp.sum(left*right)/3


def fixture(selected=2, wide=False, degenerate=False):
    rng = np.random.default_rng(382)
    no, nv = 3, 4
    arrays = tuple(rng.normal(scale=.1, size=shape) for shape in
                   ((no,nv), (no,no,nv,nv), (no,nv,nv,nv),
                    (no,nv,no,no), (no,nv,no,nv), (no,nv)))
    eo = np.array([-11., -.6, -.35]) if wide else np.array([-1.2, -.8, -.4])
    ev = np.array([.23, 1.2, 5., 85.]) if wide else np.array([.3, .7, 1.1, 1.6])
    if degenerate:
        eo[:], ev[:] = -.7, .8
    qo = np.linalg.qr(rng.normal(size=(no,no)))[0]
    qv = np.linalg.qr(rng.normal(size=(nv,nv)))[0]
    return arrays + ((qo*eo)@qo.T, (qv*ev)@qv.T,
                     np.eye(no) if selected==no else rng.normal(size=(no,selected)))


def evaluate(inputs, node, cache=0, samples=0):
    engine = native.StableTriplesDressing(tuple(np.asarray(x) for x in inputs))
    values = engine.prepare(node, cache_limit_bytes=cache)
    result = native.triples_prepared_energy_vjp(values, nsamples=samples, seed=71)
    bars = engine.pullback(node, result['cotangents'], cache_limit_bytes=cache)
    return result, bars


@pytest.mark.parametrize('selected', [2, 3])
@pytest.mark.parametrize('node', [0., .37, 3.17])
@pytest.mark.parametrize('spectrum', ['ordinary', 'degenerate', 'wide'])
def test_energy_and_all_responses_against_six_index_reference(selected, node, spectrum):
    inputs = fixture(selected, wide=spectrum=='wide', degenerate=spectrum=='degenerate')
    expected, expected_bars = jax.value_and_grad(reference)(tuple(map(jnp.asarray, inputs)), node)
    result, bars = evaluate(inputs, node, cache=2**24)
    np.testing.assert_allclose(result['energy'], expected, rtol=3e-10, atol=3e-13)
    for i, (got, want) in enumerate(zip(bars, expected_bars)):
        assert np.isfinite(got).all()
        np.testing.assert_allclose(got, want, rtol=2e-9, atol=2e-12, err_msg=f'input {i}')


def test_broad_spectrum_off_diagonal_fock_finite_differences_and_cache():
    inputs = fixture(wide=True)
    result, bars = evaluate(inputs, 3.17)
    cached, cached_bars = evaluate(inputs, 3.17, cache=2**24)
    np.testing.assert_allclose(result['energy'], cached['energy'], atol=1e-15)
    for a, b in zip(bars, cached_bars):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)
    for index in (6, 7):
        direction = np.zeros_like(inputs[index]); direction[0,1] = direction[1,0] = 1.
        h = 1e-5
        plus, minus = list(inputs), list(inputs)
        plus[index] = inputs[index] + h*direction
        minus[index] = inputs[index] - h*direction
        derivative = (evaluate(plus, 3.17)[0]['energy'] - evaluate(minus, 3.17)[0]['energy'])/(2*h)
        np.testing.assert_allclose(np.sum(bars[index]*direction), derivative, rtol=2e-6, atol=1e-11)


def test_sampling_uses_same_stream_for_energy_and_response():
    engine = native.StableTriplesDressing(fixture(wide=True))
    values = engine.prepare(3.17)
    energy = native.triples_prepared_energy(values, nsamples=1025, seed=41)
    response = native.triples_prepared_energy_vjp(values, nsamples=1025, seed=41)
    for key in ('energy', 'standard_error', 'samples'):
        assert energy[key] == response[key]
    assert all(np.isfinite(bar).all() for bar in engine.pullback(3.17, response['cotangents']))


def test_common_energy_shift_and_rebuilt_tape():
    inputs = fixture(wide=True)
    expected, bars = evaluate(inputs, 3.17)
    shifted = list(inputs)
    shifted[6] = inputs[6] + 1000*np.eye(3)
    shifted[7] = inputs[7] + 1000*np.eye(4)
    actual, shifted_bars = evaluate(shifted, 3.17)
    np.testing.assert_allclose(actual['energy'], expected['energy'], rtol=3e-11, atol=1e-14)
    for a, b in zip(bars, shifted_bars):
        np.testing.assert_allclose(a, b, rtol=3e-9, atol=1e-13)
    engine = native.StableTriplesDressing(inputs)
    prepared = engine.prepare(3.17)
    out = native.triples_prepared_energy_vjp(prepared, nsamples=0)
    rebuilt = engine.pullback(3.17, out['cotangents'], cache_limit_bytes=2**24)
    assert engine.cache_rebuilds == 1 and engine.cache_bytes == 0
    for a, b in zip(bars, rebuilt):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


def test_stochastic_energy_and_fock_response_against_exact():
    inputs = fixture(wide=True)
    engine = native.StableTriplesDressing(inputs)
    values = engine.prepare(3.17)
    exact = native.triples_prepared_energy_vjp(values, nsamples=0)
    exact_bars = engine.pullback(3.17, exact['cotangents'])
    draws, errors = [], []
    for seed in range(12):
        sample = native.triples_prepared_energy_vjp(values, nsamples=4096, seed=seed)
        bars = engine.pullback(3.17, sample['cotangents'])
        draws.append([sample['energy'], bars[6][0,1], bars[7][0,1], bars[8][0,0]])
        errors.append(sample['standard_error'])
    draws = np.asarray(draws)
    expected = np.array([exact['energy'], exact_bars[6][0,1], exact_bars[7][0,1], exact_bars[8][0,0]])
    se = draws.std(axis=0, ddof=1)/np.sqrt(len(draws))
    assert np.all(np.abs(draws.mean(axis=0)-expected) < 6*se + 1e-13)
    assert abs(draws[:,0].mean()-expected[0]) < 6*np.linalg.norm(errors)/len(errors)
