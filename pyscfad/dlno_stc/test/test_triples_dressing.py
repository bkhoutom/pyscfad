"""Native panel dressing and its full VJP against the original JAX transform."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscfad.dlno_stc import _stc_mp2 as native
from pyscfad.dlno_stc.smooth import block_exp
from pyscfad.dlno_stc.smooth_triples import dress


@pytest.mark.parametrize('fragment', ['whole', 'equal', 'asymmetric'])
@pytest.mark.parametrize('degenerate', [False, True])
@pytest.mark.parametrize('node', [0., .37, 3.])
def test_native_dressing_all_cotangents(fragment, degenerate, node):
    rng = np.random.default_rng(293)
    no, nv, npaux = 3, 4, 5
    # Noncontiguous clusters catch assumptions about contiguous eigenvalues.
    occupied, virtual = ((2, 0), (1,)), ((3, 1, 0), (2,))
    foo = rng.normal(size=(no,no))*.03; foo = (foo+foo.T)/2 - np.eye(no)
    fvv = rng.normal(size=(nv,nv))*.03; fvv = (fvv+fvv.T)/2 + np.eye(nv)
    if degenerate:
        foo = -np.eye(no); fvv = np.eye(nv)
    arrays = [rng.normal(size=s) for s in [(no,nv), (no,no,nv,nv),
        (npaux,no,nv), (npaux,no,no), (npaux,nv,nv), (no,nv)]] + [foo, fvv]
    if fragment != 'whole':
        arrays.append(rng.normal(size=(no, 2 if fragment == 'asymmetric' else no)))
    def transform(*inputs):
        result = dress(*inputs[:8], node, occupied_clusters=occupied, virtual_clusters=virtual)
        if fragment != 'whole':
            left = block_exp(inputs[6], -node/2, occupied)
            right = block_exp(inputs[6], node/2, occupied)
            result = (*result, left @ inputs[8], right @ inputs[8]) if fragment == 'asymmetric' else (
                *result, left @ inputs[8] @ right)
        return result
    expected, pullback = jax.vjp(transform, *map(jnp.asarray, arrays))
    engine = native.TriplesDressing(tuple(arrays), occupied, virtual, fragment == 'asymmetric')
    actual = engine.prepare(node)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=2e-12, atol=2e-12)
    cotangents = tuple(rng.normal(size=x.shape) for x in expected)
    bars = engine.pullback(node, cotangents)
    reference = pullback(tuple(map(jnp.asarray, cotangents)))
    for got, want in zip(bars, reference):
        np.testing.assert_allclose(got, want, rtol=2e-11, atol=5e-11)


def test_panel_boundaries_and_large_cluster_blas_pullback():
    rng = np.random.default_rng(127)
    no, nv, naux = 3, 18, 1024
    occupied, virtual = ((0, 2), (1,)), (tuple(range(nv)),)
    foo = -np.eye(no)
    q = np.linalg.qr(rng.normal(size=(nv, nv)))[0]
    fvv = (q * np.linspace(.4, 1.2, nv)) @ q.T
    inputs = tuple(rng.normal(size=s) for s in [(no,nv), (no,no,nv,nv),
        (naux,no,nv), (naux,no,no), (naux,nv,nv), (no,nv)]) + (foo, fvv)
    def transform(*x):
        return dress(*x, .7, occupied_clusters=occupied, virtual_clusters=virtual)
    expected, pullback = jax.vjp(transform, *map(jnp.asarray, inputs))
    engine = native.TriplesDressing(inputs, occupied, virtual)
    for got, want in zip(engine.prepare(.7), expected):
        np.testing.assert_allclose(got, want, rtol=3e-11, atol=3e-12)
    cotangents = tuple(rng.normal(size=x.shape) for x in expected)
    for got, want in zip(engine.pullback(.7, cotangents), pullback(tuple(map(jnp.asarray, cotangents)))):
        np.testing.assert_allclose(got, want, rtol=3e-10, atol=3e-10)
