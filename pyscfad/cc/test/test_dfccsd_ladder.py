"""Numerical and cache-lifecycle regressions for the DF virtual ladder."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy
import pytest

from pyscfad import config_update, gto, scf
from pyscfad.cc import dfccsd


def _symmetric_factor(packed, nvir):
    # Independent of the production triangular helpers, including their VJP.
    i, j = numpy.tril_indices(nvir)
    lookup = numpy.empty((nvir, nvir), dtype=int)
    lookup[i, j] = lookup[j, i] = numpy.arange(len(i))
    return packed[:, lookup]


@pytest.mark.parametrize('nvir', [1, 2, 5])
@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('optimized', [False, True])
def test_ladder_values_and_factor_amplitude_derivatives(nvir, cached, optimized):
    """Catch index permutations or missing factor terms in either ladder AD rule."""
    rng = numpy.random.default_rng(16 + nvir)
    packed = jnp.asarray(rng.normal(size=(4, nvir * (nvir + 1) // 2)))
    tau = jnp.asarray(rng.normal(size=(3, nvir, nvir)))
    cot = jnp.asarray(rng.normal(size=tau.shape))

    def reference(p, t):
        L = _symmetric_factor(p, nvir)
        return jnp.einsum('pcd,Pac,Pbd->pab', t, L, L)

    def candidate(p, t):
        V = None
        if cached:
            V = dfccsd._build_eris_vvvv(jax.lax.stop_gradient(
                _symmetric_factor(p, nvir)))
        return dfccsd._contract_vvvv_t2(None, None, p, t, vvvv=V)

    with config_update('pyscfad_moleintor_opt', optimized):
        value_ref, pull_ref = jax.vjp(reference, packed, tau)
        value, pull = jax.vjp(jax.jit(candidate), packed, tau)
        numpy.testing.assert_allclose(value, value_ref, rtol=1e-12, atol=2e-12)
        bars_ref = pull_ref(cot)
        for actual, expected in zip(pull(cot), bars_ref):
            numpy.testing.assert_allclose(actual, expected, rtol=1e-12, atol=3e-12)
        # Independently perturbed arguments exercise symbolic-zero AD paths.
        for argnum, expected in enumerate(bars_ref):
            actual = jax.grad(lambda p, t: jnp.vdot(candidate(p, t), cot),
                              argnums=argnum)(packed, tau)
            numpy.testing.assert_allclose(actual, expected, rtol=1e-12, atol=3e-12)
        if not optimized:
            dots = tuple(jnp.asarray(rng.normal(size=x.shape)) for x in (packed, tau))
            _, actual = jax.jvp(jax.jit(candidate), (packed, tau), dots)
            _, expected = jax.jvp(reference, (packed, tau), dots)
            numpy.testing.assert_allclose(actual, expected, rtol=1e-12, atol=3e-12)


@pytest.mark.parametrize('direct,remaining,cached', [
    (False, 1000, True), (True, 1000, False),
    (False, 0, False), (False, -1, False),
    (False, .0009, False), (False, .001, True),
])
def test_cache_policy_preserves_ladder_values(monkeypatch, direct, remaining, cached):
    """Explicit direct mode and exhausted memory must select the valid fallback."""
    mycc = SimpleNamespace(vvvv_direct=direct, max_memory=1000, verbose=0)
    monkeypatch.setattr(dfccsd, 'current_memory', lambda: (1000 - remaining, 0))
    eris = dfccsd._ChemistsERIs()
    eris.Lvv = jnp.arange(12.).reshape(2, 6) / 10
    eris._vvvv_pair = jnp.zeros((2, 6, 6))  # stale cache must be discarded
    tau = jnp.arange(18.).reshape(2, 3, 3) / 10
    result = dfccsd._maybe_cache_vvvv(mycc, eris)
    assert result is eris
    assert (eris._vvvv_pair is not None) == cached
    L = _symmetric_factor(eris.Lvv, 3)
    expected = jnp.einsum('pcd,Pac,Pbd->pab', tau, L, L)
    numpy.testing.assert_allclose(eris._contract_vvvv_t2(mycc, tau), expected,
                                  rtol=1e-12, atol=1e-12)


def test_cache_builder_stops_factor_gradient(monkeypatch):
    """The cache must not create a second factor-gradient path."""
    mycc = SimpleNamespace(vvvv_direct=False, max_memory=1000, verbose=0)
    monkeypatch.setattr(dfccsd, 'current_memory', lambda: (0, 0))

    def cache(packed):
        eris = dfccsd._ChemistsERIs()
        eris.Lvv = packed
        return dfccsd._maybe_cache_vvvv(mycc, eris)._vvvv_pair

    packed = jnp.arange(12.).reshape(2, 6)
    value, tangent = jax.jvp(cache, (packed,), (jnp.ones_like(packed),))
    assert numpy.linalg.norm(value) > 0
    numpy.testing.assert_array_equal(tangent, numpy.zeros(value.shape))


@pytest.mark.parametrize('optimized', [False, True])
def test_runtime_cache_is_nondifferentiable(optimized):
    """Runtime cache inputs cannot add a second physical factor derivative."""
    packed = jnp.arange(6.).reshape(2, 3)
    tau = jnp.arange(8.).reshape(2, 2, 2)
    L = _symmetric_factor(packed, 2)
    cache = dfccsd._build_eris_vvvv(L)

    def value(V):
        return dfccsd._contract_vvvv_t2(None, None, packed, tau, vvvv=V)

    with config_update('pyscfad_moleintor_opt', optimized):
        _, pull = jax.vjp(jax.jit(value), cache)
        bar, = pull(jnp.ones_like(tau))
        numpy.testing.assert_array_equal(bar, numpy.zeros(cache.shape))
        if not optimized:
            _, tangent = jax.jvp(jax.jit(value), (cache,), (jnp.ones_like(cache),))
            numpy.testing.assert_array_equal(tangent, numpy.zeros(tau.shape))


@pytest.mark.parametrize('optimized', [False, True])
def test_jitted_ladder_uses_each_runtime_cache(monkeypatch, optimized):
    """Same-shaped ERIs must reuse compiled code without reusing old integrals."""
    monkeypatch.setattr(dfccsd, 'current_memory', lambda: (0, 0))
    mycc = SimpleNamespace(vvvv_direct=False, max_memory=1000, verbose=0)
    tau = jnp.arange(8.).reshape(2, 2, 2)
    packed = jnp.arange(6.).reshape(2, 3)

    def compiled(eris, t):
        return eris._contract_vvvv_t2(None, t)

    with config_update('pyscfad_moleintor_opt', optimized):
        fn = jax.jit(compiled)
        for direct in (False, True):
            mycc.vvvv_direct = direct
            for scale in (1, 2, 3):
                eris = dfccsd._ChemistsERIs()
                eris.Lvv = packed * scale
                dfccsd._maybe_cache_vvvv(mycc, eris)
                L = _symmetric_factor(eris.Lvv, 2)
                expected = jnp.einsum('pcd,Pac,Pbd->pab', tau, L, L)
                numpy.testing.assert_allclose(fn(eris, tau), expected, atol=1e-11)
                if not direct:
                    assert eris._vvvv_pair.nbytes == 16 * 3**2
        # One program per cache mode; numerical cache values are runtime data.
        assert fn._cache_size() == 2


def test_canonical_builder_caches_integrals_and_keeps_factors(get_opt_mol):
    """Canonical ao2mo retains packed ovvv independently of ladder mode."""
    mf = scf.RHF(get_opt_mol).density_fit()
    mf.kernel()
    mycc = dfccsd.RCCSD(mf)
    eris = mycc.ao2mo()
    assert eris._vvvv_pair is not None
    assert eris.Loo is not None and eris.Lov is not None and eris.Lvv is not None
    assert eris.ovvv is not None
    mycc.vvvv_direct = True
    direct = mycc.ao2mo()
    assert direct._vvvv_pair is None
    numpy.testing.assert_allclose(eris.ovvv, direct.ovvv, atol=1e-12)
    nvir = mycc.nmo - mycc.nocc
    tau = jnp.arange(2 * nvir**2, dtype=float).reshape(2, nvir, nvir)
    numpy.testing.assert_allclose(eris._contract_vvvv_t2(mycc, tau),
                                  direct._contract_vvvv_t2(mycc, tau), atol=1e-10)


@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize('custom_response', [False, True])
def test_cached_and_direct_molecular_response_gradients(optimized, custom_response,
                                                       monkeypatch):
    """Runtime caches preserve coordinate and amplitude response in both AD paths."""
    mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    # Inspect the real response result, including the zero-cache representation.
    original_gradient = dfccsd.lagrangian_grad
    response_calls = []

    def inspect_gradient(*args, **kwargs):
        assert kwargs.get('intermediates') is not None
        bar = original_gradient(*args, **kwargs)
        assert bar._vvvv_pair is None
        assert bar.ovvv is None
        response_calls.append(True)
        return bar

    monkeypatch.setattr(dfccsd, 'lagrangian_grad', inspect_gradient)

    def objective(m, direct):
        mf = scf.RHF(m).density_fit()
        mf.conv_tol = 1e-11
        mf.kernel()
        mycc = dfccsd.RCCSD(mf)
        mycc.vvvv_direct = direct
        mycc.conv_tol = 1e-10
        mycc.conv_tol_normt = 1e-8
        eris = mycc.ao2mo()
        if direct:
            # Compare with the original factor-only update and response.
            eris.ovvv = None
        mycc.kernel(eris=eris)
        # Nonzero amplitude cotangents exercise the full custom response solve.
        return mycc.e_tot + .001 * jnp.sum(mycc.t1) + .002 * jnp.sum(mycc.t2)

    with config_update('pyscfad_moleintor_opt', optimized), \
         config_update('pyscfad_dfccsd_custom_response', custom_response), \
         config_update('pyscfad_scf_implicit_diff', True), \
         config_update('pyscfad_ccsd_implicit_diff', True):
        cached, cached_grad = jax.value_and_grad(objective)(mol, False)
        direct, direct_grad = jax.value_and_grad(objective)(mol, True)
    numpy.testing.assert_allclose(cached, direct, atol=1e-11, rtol=0)
    numpy.testing.assert_allclose(cached_grad.coords, direct_grad.coords,
                                  atol=2e-9, rtol=0)
    assert bool(response_calls) == custom_response
