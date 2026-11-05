"""Packed whitening and resident LIS adjoints retain the complete response."""
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
import pytest

from pyscfad.lno import _df_symmetric as sym, _df_shared
from pyscfad.lno._df_half import HalfTransformedDFCache
from pyscfad.lno.test import test_shared_local_df_reverse as shared


@pytest.mark.parametrize('order', ['C', 'F'])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('naux', [0, 1, 33, 65])
def test_native_pair_conversion_matches_bounded_fallback(monkeypatch, order, weighted, naux):
    rng = np.random.default_rng(52)
    a = rng.normal(size=(naux, 17*17))
    original = a.copy()
    actual = sym.pack_symmetric(a, 17, order=order, weighted=weighted)
    expanded = sym.unpack_symmetric(actual, 17)
    monkeypatch.setattr(sym, '_pack', None)
    monkeypatch.setattr(sym, '_unpack', None)
    expected = sym.pack_symmetric(a, 17, order=order, weighted=weighted)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(expanded, sym.unpack_symmetric(expected, 17))
    np.testing.assert_array_equal(a, original)
    assert actual.flags.f_contiguous if order == 'F' else actual.flags.c_contiguous


@pytest.mark.parametrize('width', [1, 3, 6])
def test_packed_adjoint_against_independent_autodiff(width):
    rng = np.random.default_rng(982)
    nq, nao = 8, 7
    low = np.tril(rng.normal(size=(nq, nq))*.1) + 2*np.eye(nq)
    raw = rng.normal(size=(nq, nao, nao))
    coeff = rng.normal(size=(nao, width))
    seed = rng.normal(size=(nq, width, width))  # Deliberately nonsymmetric.
    def factors(l, a, c):
        a = .5*(a + a.swapaxes(1, 2))
        t = jnp.einsum('puv,ui,vj->pij', a, c, c)
        return jsl.solve_triangular(l, t.reshape(nq, -1), lower=True)
    fitted, pb = jax.vjp(factors, jnp.asarray(low), jnp.asarray(raw), jnp.asarray(coeff))
    expected = pb(jnp.asarray(seed.reshape(nq, -1)))
    original = [a.copy() for a in (low, np.asarray(fitted), seed)]
    packed, lbar = sym.factor_pullback(low, np.asarray(fitted),
        seed.reshape(nq, -1), symmetric=True)
    z = sym.unpack_symmetric(packed, width)
    abar = np.einsum('ui,pij,vj->puv', coeff, z, coeff)
    cbar = 2*np.einsum('puv,vj,pij->ui', .5*(raw+raw.swapaxes(1, 2)), coeff, z)
    for actual, want in zip((np.tril(lbar), abar, cbar), expected):
        np.testing.assert_allclose(actual, want, rtol=2e-12, atol=2e-11)
    for actual, want in zip((low, np.asarray(fitted), seed), original):
        np.testing.assert_array_equal(actual, want)
    assert packed.shape == (nq, width*(width+1)//2)


def test_empty_pairs_and_slice_eligibility():
    z, lbar = sym.factor_pullback(np.eye(2), np.empty((2, 0)),
                                np.empty((2, 0)), symmetric=True)
    assert z.shape == (2, 0)
    assert sym.unpack_symmetric(z, 0).shape == (2, 0, 0)
    np.testing.assert_array_equal(lbar, np.zeros((2, 2)))
    assert sym.same_orbital_pair((2, 5, 2, 5))
    assert not sym.same_orbital_pair((0, 3, 3, 6))
    assert not sym.same_orbital_pair((0, 3, 0, 4))


@pytest.mark.parametrize('resident', [False, True])
@pytest.mark.parametrize('backend', ['legacy', 'tiled'])
def test_shared_packed_lis_matches_full_unsymmetrized_reverse(monkeypatch, resident, backend):
    mol, coeff = shared._problem(7)
    objective = lambda m, c: -.3 * shared._energy(m, c)
    with monkeypatch.context() as old:
        old.setattr(sym, 'same_orbital_pair', lambda slices: False)
        value_ref, reference = jax.value_and_grad(objective, argnums=(0, 1))(mol, coeff)
    monkeypatch.setenv('PYSCFAD_DF_MO_DENSITY_BACKEND', backend)
    expanded = []
    unpack = sym.unpack_symmetric
    def once(pairs, width):
        expanded.append((pairs.shape[0], width))
        return unpack(pairs, width)
    monkeypatch.setattr(sym, 'unpack_symmetric', once)
    with _df_shared.reverse_scope(shared._mf(mol), shared.ATOMS) as scope:
        if resident:
            scope.half = HalfTransformedDFCache(scope)
        value, bars = jax.value_and_grad(objective, argnums=(0, 1))(mol, coeff)
        full = [term.z for term in scope.terms if term.symmetric]
        assert len(full) == 1 and full[0].shape == (scope.auxmol.nao, 3, 3)
        np.testing.assert_array_equal(full[0], full[0].swapaxes(1, 2))
        assert scope.receipt['symmetric_consumers'] == 1
        assert scope.receipt['symmetric_expanded_bytes'] == scope.auxmol.nao * 3 * 8
        assert expanded == [(scope.auxmol.nao, 3)]
        geometry = scope.finish()
        assert expanded == [(scope.auxmol.nao, 3)]
        assert scope.receipt['adjoint_resident_bytes'] == 0
        assert scope.receipt['geometry_passes'] == scope.receipt['metric_passes'] == 1
    np.testing.assert_allclose(value, value_ref, atol=1e-12)
    np.testing.assert_allclose(bars[0].coords+geometry, reference[0].coords,
                               atol=2e-9, rtol=2e-8)
    np.testing.assert_allclose(bars[1], reference[1], atol=2e-10, rtol=2e-9)
