"""Generic low-rank DF exchange cotangents retain both density and factor bars."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad import df, gto
from pyscfad.df import _df_jk_opt


def _case(batch, seed=741):
    mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='cc-pvtz', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    obj = df.DF(mol)
    rng = np.random.default_rng(seed)
    n = mol.nao
    obj._cderi = rng.normal(size=(7, n * (n + 1) // 2)) / n
    obj.blockdim = 3
    occupied = rng.normal(size=batch + (n, 3)) / np.sqrt(n)
    dm = occupied @ occupied.swapaxes(-1, -2)
    return obj, dm, rng


def _reference(packed, dm):
    n = dm.shape[-1]
    i, j = np.tril_indices(n)
    factors = jnp.zeros((packed.shape[0], n, n))
    factors = factors.at[:, i, j].set(packed).at[:, j, i].set(packed)
    return (jnp.einsum('Lij,...ij,Lpq->...pq', factors, dm, factors),
            jnp.einsum('Lpi,...ij,Ljq->...pq', factors, dm, factors))


@pytest.mark.parametrize('batch', [(), (2,)])
@pytest.mark.parametrize('seed_kind', ['positive', 'indefinite', 'zero'])
def test_generic_lowrank_density_and_factor_bars_match_dense_tensors(
        monkeypatch, batch, seed_kind):
    """A signed low-rank seed must avoid dense exchange without losing either bar."""
    obj, dm, rng = _case(batch)
    n = obj.mol.nao
    vectors = rng.normal(size=batch + (n, 6)) / np.sqrt(n)
    signs = np.array([1., 1., 1., -1., -1., -1.]) if seed_kind == 'indefinite' else np.ones(6)
    if seed_kind == 'zero':
        signs[:] = 0
    kbar = (vectors * signs) @ vectors.swapaxes(-1, -2)
    jbar = rng.normal(size=dm.shape) / n
    _, reference = jax.vjp(_reference, jnp.asarray(obj._cderi), jnp.asarray(dm))
    expected_eri, expected_dm = reference((jbar, kbar))

    def dense_exchange(*args):
        raise AssertionError('low-rank exchange cotangent used the dense native transpose')
    monkeypatch.setattr(_df_jk_opt, '_DF_VK_DM_VJP', dense_exchange)
    actual_eri, actual_dm = _df_jk_opt.get_jk_bwd(
        1, True, True, 1e-13, (obj, dm), (jbar, kbar))
    np.testing.assert_allclose(actual_eri._cderi, expected_eri, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(actual_dm, expected_dm, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize('kind', ['full_rank', 'asymmetric', 'indefinite_density'])
def test_generic_exchange_unsupported_cases_preserve_dense_result(monkeypatch, kind):
    """Uncaptured rank or unsupported symmetry/density must retain the dense kernel."""
    obj, dm, rng = _case(())
    n = obj.mol.nao
    kbar = rng.normal(size=(n, n)) / n
    if kind != 'asymmetric':
        kbar += kbar.T
    if kind == 'indefinite_density':
        dm -= np.eye(n)
    jbar = rng.normal(size=(n, n)) / n
    _, reference = jax.vjp(_reference, jnp.asarray(obj._cderi), jnp.asarray(dm))
    expected_eri, expected_dm = reference((jbar, kbar))

    def lowrank_exchange(*args):
        raise AssertionError('unsupported cotangent used the low-rank transpose')
    monkeypatch.setattr(_df_jk_opt, '_df_vk_dm_vjp_lowrank', lowrank_exchange)
    actual_eri, actual_dm = _df_jk_opt.get_jk_bwd(
        1, True, True, 1e-13, (obj, dm), (jbar, kbar))
    np.testing.assert_allclose(actual_eri._cderi, expected_eri, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(actual_dm, expected_dm, atol=2e-11, rtol=2e-11)


def test_dfmp2_water_full_gradient_preserves_dense_exchange_and_finite_difference(monkeypatch):
    """The nonstationary MP2 objective exercises the full molecular/DF chain."""
    from pyscf import scf as ordinary_scf
    from pyscf.mp import dfmp2 as ordinary_mp2
    from pyscfad import config_update, scf
    from pyscfad.mp import dfmp2

    mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='cc-pvtz', verbose=0, max_memory=8000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    auxbasis = 'cc-pvtz-jkfit'

    def configure(mf):
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-9, 100
        mf.chkfile = None
        mf.kernel()
        assert mf.converged
        return mf

    def objective(current):
        mf = configure(scf.RHF(current).density_fit(auxbasis=auxbasis))
        mp = dfmp2.MP2(mf, frozen=1)
        mp.kernel(with_t2=False)
        return mp.e_tot

    original = _df_jk_opt._density_exchange_factors
    accepted = []
    def counted(*args):
        factors = original(*args)
        if factors is not None:
            accepted.append(1)
        return factors

    with (config_update('pyscfad_moleintor_opt', True),
          config_update('pyscfad_scf_implicit_diff', True),
          config_update('pyscfad_scf_df_precond', True)):
        with monkeypatch.context() as context:
            context.setattr(_df_jk_opt, '_density_exchange_factors', lambda *args: None)
            value_dense, gradient_dense = jax.value_and_grad(objective)(mol)
        monkeypatch.setattr(_df_jk_opt, '_density_exchange_factors', counted)
        value_fast, gradient_fast = jax.value_and_grad(objective)(mol)
    assert accepted
    np.testing.assert_allclose(value_fast, value_dense, atol=1e-11, rtol=0)
    np.testing.assert_allclose(gradient_fast.coords, gradient_dense.coords, atol=2e-8, rtol=0)

    ordinary = mol.to_pyscf()
    coordinates = ordinary.atom_coords()
    step, values = 1e-4, []
    for direction in (1, -1):
        displaced = coordinates.copy()
        displaced[1, 2] += direction * step
        current = ordinary.set_geom_(displaced, unit='Bohr', inplace=False)
        mf = configure(ordinary_scf.RHF(current).density_fit(auxbasis=auxbasis))
        mp = ordinary_mp2.MP2(mf, frozen=1)
        mp.kernel(with_t2=False)
        values.append(mp.e_tot)
    finite_difference = (values[0] - values[1]) / (2 * step)
    np.testing.assert_allclose(gradient_fast.coords[1, 2], finite_difference,
                               atol=2e-6, rtol=0)
