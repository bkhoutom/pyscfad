"""Consumed domain response arrays must be gone before SCF response starts."""

import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy

from pyscfad.dlno_stc import driver


def test_finish_releases_common_tape_before_scf_response(monkeypatch):
    matrix = jnp.arange(64., dtype=jnp.float64).reshape(8, 8) / 64
    expected = 1. + 2. * numpy.cos(numpy.asarray(matrix))
    common, pullback = jax.vjp(lambda x: jnp.sin(x).sum(), matrix)
    residual = max(jax.tree.leaves(pullback), key=lambda x: x.size)
    refs = [weakref.ref(x) for x in (common, pullback, residual)]
    state = dict(mf=SimpleNamespace(verbose=0), common=common,
                 common_pullback=pullback, common_bar=jnp.array(2.),
                 mf_bar=jnp.ones_like(matrix), energy=jnp.array(-1.), variance=0.)
    del matrix, common, pullback, residual

    def finish(state, energy, mf_bar, *, include_hf):
        assert all(ref() is None for ref in refs)
        assert not {'common', 'common_bar', 'common_pullback', 'mf_bar'} & state.keys()
        numpy.testing.assert_allclose(mf_bar, expected, atol=1e-14)
        assert include_hf
        return energy, mf_bar

    monkeypatch.setattr(driver, '_finish_response', finish)
    energy, gradient = driver._finish(state, with_grad=True, include_hf=True)
    assert energy == -1.
    numpy.testing.assert_allclose(gradient, expected, atol=1e-14)


def test_finish_releases_scf_tape_before_final_df_geometry(monkeypatch):
    matrix = jnp.arange(16., dtype=jnp.float64).reshape(4, 4) / 16
    expected = 2. * numpy.cos(numpy.asarray(matrix))
    _, pullback = jax.vjp(lambda x: jnp.sin(x).sum(), matrix)
    residual = max(jax.tree.leaves(pullback), key=lambda x: x.size)
    refs = [weakref.ref(x) for x in (pullback, residual)]
    state = dict(mf=SimpleNamespace(verbose=0), mf_bar=jnp.array(2.),
                 scf_pullback=pullback, energy=jnp.array(-1.), variance=0.)
    del matrix, pullback, residual
    finished = []

    @contextmanager
    def final_geometry(state):
        yield
        assert 'scf_pullback' not in state
        assert all(ref() is None for ref in refs)
        finished.append(True)

    monkeypatch.setattr(driver, '_global_df_response', final_geometry)
    energy, gradient = driver._finish(state, with_grad=True, include_hf=False)
    assert energy == -1.
    numpy.testing.assert_allclose(gradient, expected, atol=1e-14)
    assert finished == [True]
