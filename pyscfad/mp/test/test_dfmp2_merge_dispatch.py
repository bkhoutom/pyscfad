"""Public MP2 routing must retain fused eager AD and its memory fallback."""
from types import SimpleNamespace

import jax
import numpy
import pytest

from pyscfad import config_update
from pyscfad import numpy as np
from pyscfad.mp import dfmp2


def _problem():
    lov = np.asarray(numpy.random.default_rng(904).normal(size=(4, 6)))
    energies = np.asarray([-1.3, -.7, .2, .6, 1.])
    mp = SimpleNamespace(nocc=2, nmo=5, frozen=0, max_memory=8000,
                         loop_ao2mo=lambda *args: lov)
    def energy(e):
        eris = SimpleNamespace(mo_energy=e, mo_coeff=np.eye(5))
        return dfmp2.kernel(mp, eris=eris, with_t2=False)[0]
    reference = lambda e: dfmp2._contract_scan_impl(lov, e, 2, 3, False)[0]
    return mp, energies, energy, reference


@pytest.mark.parametrize('transform', ['grad', 'jvp', 'linearize'])
def test_public_eager_ad_uses_one_fused_pass(transform, monkeypatch):
    _, energies, energy, reference = _problem()
    direction = np.arange(5.) * .1
    def evaluate(fn):
        if transform == 'grad':
            return jax.value_and_grad(fn)(energies)
        if transform == 'jvp':
            return jax.jvp(fn, (energies,), (direction,))
        value, tangent = jax.linearize(fn, energies)
        return value, tangent(direction), tangent(-direction)
    expected = evaluate(reference)
    calls = []
    original = dfmp2._contract_d_vjp
    def counted(*args, **kwargs):
        calls.append(kwargs.get('with_energy', False))
        return original(*args, **kwargs)
    monkeypatch.setattr(dfmp2, '_contract_d_vjp', counted)
    with config_update('pyscfad_moleintor_opt', True):
        actual = evaluate(energy)
    assert calls == [True]
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        numpy.testing.assert_allclose(got, want, atol=1e-10, rtol=1e-10)


def test_ordinary_energy_gradient_respects_native_memory_guard(monkeypatch):
    mp, energies, energy, reference = _problem()
    mp.max_memory = 100.001
    monkeypatch.setattr(dfmp2, 'current_memory', lambda: (100., 0.))
    monkeypatch.setattr(dfmp2.pyscf_lib, 'num_threads', lambda: 4)
    def forbidden(*args, **kwargs):
        raise AssertionError('energy-only native route bypassed memory budget')
    monkeypatch.setattr(dfmp2, '_contract_native', forbidden)
    monkeypatch.setattr(dfmp2, '_contract_d_lov', forbidden)
    with config_update('pyscfad_moleintor_opt', True):
        actual = jax.value_and_grad(energy)(energies)
    expected = jax.value_and_grad(reference)(energies)
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        numpy.testing.assert_allclose(got, want, atol=1e-10, rtol=1e-10)
