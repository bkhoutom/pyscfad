"""Native panel dressing and its full VJP against the original JAX transform."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscfad.dlno_stc import _stc_mp2 as native
from pyscfad.dlno_stc.test._triples_dressing_reference import symmetric_exp, dress


@pytest.mark.parametrize('fragment', ['whole', 'selected'])
@pytest.mark.parametrize('degenerate', [False, True])
@pytest.mark.parametrize('node', [0., .37, 3.])
def test_native_dressing_all_cotangents(fragment, degenerate, node):
    rng = np.random.default_rng(293)
    no, nv, npaux = 3, 4, 5
    foo = rng.normal(size=(no,no))*.03; foo = (foo+foo.T)/2 - np.eye(no)
    fvv = rng.normal(size=(nv,nv))*.03; fvv = (fvv+fvv.T)/2 + np.eye(nv)
    if degenerate:
        foo = -np.eye(no); fvv = np.eye(nv)
    arrays = [rng.normal(size=s) for s in [(no,nv), (no,no,nv,nv),
        (npaux,no,nv), (npaux,no,no), (npaux,nv,nv), (no,nv)]] + [foo, fvv]
    arrays.append(np.eye(no) if fragment == 'whole' else rng.normal(size=(no, 2)))
    def transform(*inputs):
        result = dress(*inputs[:8], node)
        left = symmetric_exp(inputs[6] * (-node/2))
        right = symmetric_exp(inputs[6] * (node/2))
        result = (*result, left @ inputs[8], right @ inputs[8])
        return result
    expected, pullback = jax.vjp(transform, *map(jnp.asarray, arrays))
    engine = native.TriplesDressing(tuple(arrays))
    actual = engine.prepare(node)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=2e-12, atol=2e-12)
    cotangents = tuple(rng.normal(size=x.shape) for x in expected)
    bars = engine.pullback(node, cotangents)
    reference = pullback(tuple(map(jnp.asarray, cotangents)))
    for got, want in zip(bars, reference):
        np.testing.assert_allclose(got, want, rtol=2e-11, atol=5e-11)


def test_panel_boundaries_and_full_fock_blas_pullback():
    rng = np.random.default_rng(127)
    no, nv, naux = 3, 18, 1024
    foo = -np.eye(no)
    q = np.linalg.qr(rng.normal(size=(nv, nv)))[0]
    fvv = (q * np.linspace(.4, 1.2, nv)) @ q.T
    inputs = tuple(rng.normal(size=s) for s in [(no,nv), (no,no,nv,nv),
        (naux,no,nv), (naux,no,no), (naux,nv,nv), (no,nv)]) + (foo, fvv, np.eye(no))
    def transform(*x):
        return (*dress(*x[:8], .7), symmetric_exp(-.35*x[6]) @ x[8],
                symmetric_exp(.35*x[6]) @ x[8])
    expected, pullback = jax.vjp(transform, *map(jnp.asarray, inputs))
    engine = native.TriplesDressing(inputs)
    for got, want in zip(engine.prepare(.7), expected):
        np.testing.assert_allclose(got, want, rtol=3e-11, atol=3e-12)
    cotangents = tuple(rng.normal(size=x.shape) for x in expected)
    for got, want in zip(engine.pullback(.7, cotangents), pullback(tuple(map(jnp.asarray, cotangents)))):
        np.testing.assert_allclose(got, want, rtol=3e-10, atol=3e-10)
