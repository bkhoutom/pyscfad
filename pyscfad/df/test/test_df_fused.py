"""Fused packed DF construction against the independent two-step AD route."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.linalg import solve_triangular
from pyscf.df import incore as pyscf_incore
from pyscfad import config_update, gto
from pyscfad.df import addons, incore, _int3c_cross_opt


@pytest.fixture
def molecules():
    mol = gto.Mole(atom='O 0 0 0; H 0 -.75 .58; H 0 .75 .58',
                   basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol, addons.make_auxmol(mol, 'weigend')


def _reference(mol, auxmol, low):
    # Retain the fork's public (pair, auxiliary) layout at this boundary.
    ints = _int3c_cross_opt.int3c_cross(mol, auxmol)
    return solve_triangular(low, ints.T, lower=True)


@pytest.mark.parametrize('multiblock', [False, True])
def test_fused_values_and_independent_molecule_auxiliary_metric_bars(molecules, multiblock):
    mol, auxmol = molecules
    assert hasattr(_int3c_cross_opt, 'int3c2e_cholesky')
    low = jnp.linalg.cholesky(auxmol.intor('int2c2e', hermi=1))
    ranges = ((0, 1, 0), (1, mol.nbas, 0)) if multiblock else ((0, mol.nbas, 0),)
    fused = lambda m, a, l: _int3c_cross_opt.int3c2e_cholesky(m, a, l, 'int3c2e', ranges)
    expected, reference_bar = jax.vjp(_reference, mol, auxmol, low)
    actual, fused_bar = jax.vjp(fused, mol, auxmol, low)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
    seed = np.random.default_rng(180).normal(size=actual.shape)
    first = fused_bar(seed)
    expected_bars = reference_bar(seed)
    for got, want in zip(jax.tree_util.tree_leaves(first),
                         jax.tree_util.tree_leaves(expected_bars)):
        np.testing.assert_allclose(got, want, atol=1e-8, rtol=1e-8)
    # A mutable triangular-solve buffer must not mutate the primal or residual.
    repeated = fused_bar(2 * seed)
    for got, want in zip(jax.tree_util.tree_leaves(repeated),
                         jax.tree_util.tree_leaves(first)):
        np.testing.assert_allclose(got, 2 * want, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(fused(mol, auxmol, low), expected, atol=1e-10, rtol=0)


def test_public_cholesky_uses_one_resident_numpy_factor(molecules):
    mol, auxmol = molecules
    expected = pyscf_incore.cholesky_eri(mol.to_pyscf(), auxmol=auxmol.to_pyscf())
    with config_update('pyscfad_moleintor_opt', True):
        actual = incore.cholesky_eri(mol, auxmol=auxmol)
    assert isinstance(actual, np.ndarray)
    assert actual.flags.c_contiguous
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)


def test_public_coordinate_gradient_retains_metric_response(molecules):
    mol, _ = molecules
    def objective(m, optimized):
        with config_update('pyscfad_moleintor_opt', optimized):
            b = incore.cholesky_eri(m, auxbasis='weigend')
            # The generic route stores all AO pairs; compare a physical norm.
            if not optimized:
                b = b.reshape((b.shape[0], m.nao, m.nao))
                i, j = np.tril_indices(m.nao)
                b = b[:, i, j]
            return jnp.sum(b * b)
    expected = jax.grad(lambda m: objective(m, False))(mol)
    actual = jax.grad(lambda m: objective(m, True))(mol)
    np.testing.assert_allclose(actual.coords, expected.coords, atol=1e-8, rtol=1e-8)


def test_fused_dispatch_preserves_generic_jvp(molecules):
    mol, _ = molecules
    direction = jax.tree_util.tree_map(jnp.ones_like, mol)
    direction.coords = jnp.asarray([[.1, -.2, .3], [-.2, .4, .1], [.3, .2, -.1]])
    def objective(m, optimized):
        with config_update('pyscfad_moleintor_opt', optimized):
            return jnp.sum(incore.cholesky_eri(m, auxbasis='weigend') ** 2)
    # A nonuniform displacement exercises the public JVP guard with a real bar.
    expected = jax.jvp(lambda m: objective(m, False), (mol,), (direction,))
    actual = jax.jvp(lambda m: objective(m, True), (mol,), (direction,))
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, atol=1e-8, rtol=1e-8)


def test_eigenvalue_fallback_does_not_enter_fused_producer(molecules, monkeypatch):
    mol, auxmol = molecules
    def unavailable(*args, **kwargs):
        raise np.linalg.LinAlgError('exercise the existing metric fallback')
    def forbidden(*args, **kwargs):
        raise AssertionError('eigenvalue metric entered the Cholesky-only producer')
    monkeypatch.setattr(incore.scipy.linalg, 'cholesky', unavailable)
    monkeypatch.setattr(_int3c_cross_opt, 'int3c2e_cholesky', forbidden)
    with config_update('pyscfad_moleintor_opt', True):
        actual = incore.cholesky_eri(mol, auxmol=auxmol)
    metric = np.asarray(auxmol.intor('int2c2e', hermi=1))
    ints = np.asarray(_int3c_cross_opt.int3c_cross(mol, auxmol))
    expected = ints @ np.linalg.solve(metric, ints.T)
    np.testing.assert_allclose(np.asarray(actual).T @ np.asarray(actual), expected,
                               atol=1e-9, rtol=1e-9)
