"""Stored triples integral blocks must preserve response and avoid DF rebuilds."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscfad import lib
from pyscfad.dlno_stc.smooth_triples import _integral_blocks


class CachedOnly(SimpleNamespace):
    @property
    def Lov(self):
        raise AssertionError('Fully cached integrals must not read DF factors')


def test_default_path_reuses_cached_blocks_with_their_response():
    no, nv = 2, 3
    rng = np.random.default_rng(31)
    ovvv = jnp.asarray(rng.normal(size=(no, nv, nv*(nv+1)//2)))
    ovoo = jnp.asarray(rng.normal(size=(no, nv, no, no)))
    ovov = jnp.asarray(rng.normal(size=(no, nv, no, nv)))
    eris = CachedOnly(ovvv=ovvv, ovoo=ovoo, ovov=ovov)
    blocks = _integral_blocks(eris, no, nv)
    assert blocks[1] is ovoo and blocks[2] is ovov
    np.testing.assert_array_equal(blocks[0][..., 2, 0], ovvv[..., 3])
    def objective(x):
        cached = CachedOnly(ovvv=ovvv, ovoo=x, ovov=ovov)
        return jnp.sum(_integral_blocks(cached, no, nv)[1]**2)
    np.testing.assert_allclose(jax.grad(objective)(ovoo), 2*ovoo)


@pytest.mark.parametrize('missing', ['ovvv', 'ovoo', 'ovov'])
def test_default_constructs_only_missing_blocks(missing):
    no, nv, na = 2, 3, 4
    rng = np.random.default_rng(32)
    lov = jnp.asarray(rng.normal(size=(na, no, nv)))
    loo = jnp.asarray(rng.normal(size=(na, no, no)))
    dense = rng.normal(size=(na, nv, nv));dense = (dense+dense.transpose(0, 2, 1))/2
    lvv = jnp.asarray(dense)
    raw = dict(ovvv=jnp.einsum('Pia,Pbc->iabc', lov, lvv),
               ovoo=jnp.einsum('Pia,Pjk->iajk', lov, loo),
               ovov=jnp.einsum('Pia,Pjb->iajb', lov, lov))
    stored = dict(raw)
    stored['ovvv'] = lib.pack_tril(raw['ovvv'].reshape(no*nv, nv, nv)).reshape(no, nv, -1)
    stored[missing] = None
    class Partial(SimpleNamespace):
        @property
        def Loo(self):
            assert missing == 'ovoo', 'Rebuilt an already stored ovoo block'
            return loo
        @property
        def Lvv(self):
            assert missing == 'ovvv', 'Rebuilt an already stored ovvv block'
            return lib.pack_tril(lvv)
    eris = Partial(**stored, Lov=lov)
    blocks = _integral_blocks(eris, no, nv)
    for name, block in zip(('ovvv', 'ovoo', 'ovov'), blocks):
        np.testing.assert_allclose(block, raw[name], rtol=2e-14, atol=2e-14)


def test_default_does_not_probe_a_lazy_integral_property():
    class Lazy(SimpleNamespace):
        @property
        def ovvv(self):
            raise AssertionError('Cache discovery triggered a lazy integral builder')
    no, nv = 2, 3
    lov = jnp.ones((4, no, nv));lvv = jnp.ones((4, nv*(nv+1)//2))
    eris = Lazy(Lov=lov, Lvv=lvv, ovoo=jnp.ones((no, nv, no, no)),
                ovov=jnp.ones((no, nv, no, nv)))
    blocks = _integral_blocks(eris, no, nv)
    np.testing.assert_array_equal(blocks[0], np.full((no, nv, nv, nv), 4.))
