"""Bounded root-local occupied panels preserve density response and ownership."""
import importlib
import importlib.util
import os

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import lib

from pyscfad import config_update, df, gto, scf
from pyscfad._src import implicit_diff
from pyscfad.df import _df_jk_opt as jk


def module():
    name = 'pyscfad.lib._scf_response_cache'
    assert importlib.util.find_spec(name) is not None, 'root response cache missing'
    return importlib.import_module(name)


@pytest.fixture(autouse=True)
def deterministic_available_memory(monkeypatch):
    # These tests assert cache reservations and exact block counts. Unrelated
    # JAX allocations in preceding tests must not exhaust the fixture's 1000
    # MB advisory budget and select the valid streaming/four-row fallback.
    # Adaptive memory-policy behavior is covered by test_reverse_precond_cache.
    monkeypatch.setattr(lib, 'current_memory', lambda: (0., 0.))


@pytest.fixture
def problem(tmp_path, monkeypatch):
    monkeypatch.setenv('PYSCFAD_SCF_OCCUPIED_CACHE', '1')
    mol = gto.Mole(atom='H 0 0 0; H 0 0 .74', basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    packed = np.random.default_rng(81).normal(size=(9, 3))
    path = tmp_path / 'j3c.h5'
    with h5py.File(path, 'w') as handle:
        handle['j3c'] = packed
    obj = df.DF(mol, auxbasis='weigend', incore=False)
    obj.attach_outcore_cderi(str(path))
    obj.max_memory = 1000
    obj.blockdim = 4
    coeff = np.asfortranarray([[.7], [.3]])
    return obj, coeff, packed


def test_first_pass_population_survives_changed_block_boundaries(problem):
    cache = module()
    obj, coeff, _ = problem
    values = np.arange(18.).reshape(9, 1, 2)
    assert cache.get_panel(obj, coeff) is None
    with cache.root_scope() as scope:
        panel = cache.get_panel(obj, coeff)
        assert panel is not None
        assert panel.lookup_block(0, 4) is None
        panel.store_block(0, 4, values[:4].reshape(4, 2))
        assert panel.lookup_block(2, 6) is None
        panel.store_block(4, 9, values[4:].reshape(5, 2))
        np.testing.assert_array_equal(panel.lookup_block(2, 6), values[2:6].reshape(4, 2))
        assert cache.get_panel(obj, coeff.copy()) is panel
        assert scope.stats['occupied_panel_bytes'] == values.nbytes
    assert panel.closed
    assert cache.get_panel(obj, coeff) is None


def test_optional_budget_includes_preconditioner_coefficients_and_validity(problem, monkeypatch):
    cache = module()
    obj, coeff, _ = problem
    required = 9 * 2 * 8 + coeff.nbytes + 9
    monkeypatch.setenv('PYSCFAD_SCF_RESPONSE_FACTOR_CACHE_MB', str((required+100)/2**20))
    with cache.root_scope() as scope:
        assert cache.reserve_preconditioner(100)
        assert cache.get_panel(obj, coeff) is not None
        assert scope.stats['peak_optional_bytes'] == required + 100
        assert not cache.reserve_preconditioner(1)
        with cache.root_scope() as child:
            assert cache.get_panel(obj, coeff) is None
            assert not cache.reserve_preconditioner(1)
        assert child.stats['occupied_panel_bytes'] == 0
    with cache.root_scope():
        assert cache.reserve_preconditioner(101)
        assert cache.get_panel(obj, coeff) is None
    monkeypatch.setenv('PYSCFAD_SCF_RESPONSE_FACTOR_CACHE_MB', '0')
    with cache.root_scope():
        assert not cache.reserve_preconditioner(1)
        assert cache.get_panel(obj, coeff) is None


@pytest.mark.parametrize('change', ['coefficient', 'file', 'source_token'])
def test_changed_source_or_coefficients_invalidates_panel(problem, change):
    cache = module()
    obj, coeff, _ = problem
    with cache.root_scope():
        panel = cache.get_panel(obj, coeff)
        panel.store_block(0, 9, np.ones((9, 2)))
        if change == 'coefficient':
            coeff = coeff.copy()
            coeff[0, 0] += .01
        elif change == 'file':
            stat = os.stat(obj._cderi_to_save)
            os.utime(obj._cderi_to_save, ns=(stat.st_atime_ns, stat.st_mtime_ns+1))
        else:
            obj = df.DF(obj.mol, auxbasis='weigend', incore=False).attach_outcore_cderi(obj._cderi_to_save)
        assert cache.get_panel(obj, coeff) is None
        assert panel.closed


def test_disabled_and_failed_root_scopes_leave_no_cache(problem, monkeypatch):
    cache = module()
    obj, coeff, _ = problem
    with pytest.raises(RuntimeError, match='injected'):
        with cache.root_scope():
            panel = cache.get_panel(obj, coeff)
            assert cache.reserve_preconditioner(100)
            raise RuntimeError('injected')
    assert panel.closed
    monkeypatch.setenv('PYSCFAD_SCF_OCCUPIED_CACHE', '0')
    with cache.root_scope():
        assert cache.get_panel(obj, coeff) is None
        assert cache.reserve_preconditioner(2**40)  # Ordinary policy is unchanged.


def test_native_cache_avoids_occupied_transform_and_preserves_density_bar(problem, monkeypatch):
    cache = module()
    obj, coeff, packed = problem
    b = lib.unpack_tril(packed)
    response = np.asfortranarray([[.2], [-.1]])
    monkeypatch.setattr(jk, '_implicit_lowrank_exchange_factors', lambda *a: (coeff, response))
    calls = []
    original = jk.libao2mo.AO2MOnr_e2_drv
    def record(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(jk.libao2mo, 'AO2MOnr_e2_drv', record)
    with cache.root_scope() as scope, implicit_diff._implicit_diff_solve_matvec():
        for block, scale in ((4, 1.), (5, -.4), (3, 0.)):
            obj.blockdim = block
            response = np.asfortranarray(np.array([[.2], [-.1]])*scale)
            vk = response @ coeff.T + coeff @ response.T
            vj = -2 * vk
            _, actual = jk.get_jk_bwd(1, True, True, 1e-13,
                (obj, np.eye(2), (True, False, False, False)), (vj, vk))
            expected = (np.einsum('Lij,ij,Lpq->pq', b, vj, b)
                        + np.einsum('Lpi,ij,Ljq->pq', b, vk, b))
            np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
        assert len(calls) == 11  # first pass: 2*3; subsequent: 2+3 responses only
        assert scope.stats['occupied_transform_calls'] == 3
        assert scope.stats['cache_hit_blocks'] == 5


def test_root_cache_ends_before_external_vjp(problem):
    cache = module()
    obj, coeff, _ = problem
    panels = []
    def optimality(root, parameter):
        if implicit_diff.is_implicit_diff_external_vjp():
            assert cache.get_panel(obj, coeff) is None
        return parameter - root
    def solve(matvec, rhs, **kwargs):
        panels.append(cache.get_panel(obj, coeff))
        assert panels[-1] is not None
        return rhs / matvec(jnp.ones_like(rhs)), None
    actual, = implicit_diff.root_vjp(optimality, jnp.array(2.), (jnp.array(2.),),
                                     jnp.array(1.), solve=solve)
    assert float(actual) == 1.
    assert panels[0].closed


def test_preconditioner_reserves_existing_optional_cache(problem, monkeypatch):
    cache = module()
    obj, _, _ = problem
    monkeypatch.setenv('PYSCFAD_SCF_RESPONSE_FACTOR_CACHE_MB', str(72/2**20))
    with cache.root_scope() as scope:
        coul = jk.gen_coulomb_precond(obj, np.eye(2)[:, :1], np.eye(2)[:, 1:], np.ones((1, 1)))
        assert scope.stats['preconditioner_cache_bytes'] == 72
        assert cache.get_panel(obj, np.eye(2)[:, :1]) is None
        assert np.isfinite(coul(np.ones((1, 1)))).all()
        del coul


def test_recorder_retains_only_finished_receipts_and_tracing_has_no_panel(problem):
    cache = module()
    obj, coeff, _ = problem
    def solve(matvec, rhs, **kwargs):
        assert cache.get_panel(obj, coeff) is None
        return rhs / matvec(jnp.ones_like(rhs)), None
    def differentiate(seed):
        return implicit_diff.root_vjp(lambda root, p: p-root, jnp.array(2.),
                                     (jnp.array(2.),), seed, solve=solve)[0]
    with cache.record_response_caches() as receipts:
        assert float(jax.jit(differentiate)(jnp.array(1.))) == 1.
    assert len(receipts) == 1
    assert receipts[0]['finished'] and not receipts[0]['enabled']
    assert receipts[0]['occupied_panel_bytes'] == 0


def test_actual_ooc_scf_pullback_cache_matches_uncached(tmp_path, monkeypatch):
    cache = module()
    mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='sto-3g', verbose=0, max_memory=1000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    obj = df.DF(mol, auxbasis='weigend', incore=False)
    obj._cderi_to_save = str(tmp_path / 'scf.h5')
    obj.blockdim = 8
    def outputs(inputs):
        m, d = inputs
        mf = scf.RHF(m).density_fit(with_df=d)
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-9
        mf.kernel()
        return mf.e_tot, mf.mo_energy, mf.mo_coeff
    with (config_update('pyscfad_moleintor_opt', True),
          config_update('pyscfad_scf_implicit_diff', True),
          config_update('pyscfad_scf_df_precond', True)):
        obj.build()
        value, pullback = jax.vjp(outputs, (mol, obj))
        seeds = tuple(np.random.default_rng(i).normal(size=x.shape)*.01
                      for i, x in enumerate(value))
        monkeypatch.setenv('PYSCFAD_SCF_OCCUPIED_CACHE', '0')
        with cache.record_response_caches() as off:
            reference, = pullback(seeds)
        monkeypatch.setenv('PYSCFAD_SCF_OCCUPIED_CACHE', '1')
        with cache.record_response_caches() as on:
            actual, = pullback(seeds)
    for x, y in zip(jax.tree_util.tree_leaves(actual),
                    jax.tree_util.tree_leaves(reference), strict=True):
        np.testing.assert_allclose(x, y, atol=3e-10, rtol=3e-10)
    assert len(off) == len(on) == 1
    assert not off[0]['enabled'] and not off[0]['occupied_panel_bytes']
    assert on[0]['enabled'] and on[0]['cache_hit_blocks'] > 0
    assert off[0]['occupied_transform_calls'] > on[0]['occupied_transform_calls'] > 0
    assert off[0]['response_transform_calls'] == on[0]['response_transform_calls']
    assert on[0]['preconditioner_cache_bytes'] > 0
    assert on[0]['occupied_panel_bytes'] == obj.get_naoaux()*mol.nao*5*8
    assert on[0]['peak_optional_bytes'] <= on[0]['budget_bytes']
