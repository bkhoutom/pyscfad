"""The complete serial reverse merges global DF seeds at their producer."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad import config_update, df, gto, scf
from pyscfad.df import _df_jk_factorized, _ooc_cotangent
from pyscfad.dlno_stc import driver
from pyscfad.lib import _scf_response_cache
from pyscfad.tools.linear_solver import eager_gmres
from pyscfad.dlno_stc.test.test_fragment_df_inputs import molecular_case


@pytest.mark.parametrize('aux_scale', [1., 1.03])
def test_fock_and_scf_share_geometry_preserving_auxiliary_map(tmp_path, monkeypatch, aux_scale):
    from functools import partial
    from pyscfad.tools import linear_solver
    # Compare geometry assembly below the default 1e-6 root tolerance, and
    # leave room for JAX executables retained by preceding molecular tests.
    monkeypatch.setattr(scf.hf, 'gen_gmres', partial(
        linear_solver.gen_gmres_with_default_kwargs, tol=1e-11))
    mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='sto-3g', verbose=0, max_memory=10000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    path = str(tmp_path / 'scf.h5')
    seeds = np.random.default_rng(81).normal(size=(mol.nao, mol.nao))

    def source(m):
        obj = df.DF(m, auxbasis='weigend', incore=False)
        obj.auxmol = df.addons.make_auxmol(m, obj.auxbasis)
        obj.auxmol.coords = obj.auxmol.coords * aux_scale
        return obj

    def build(m):
        obj = source(m)
        obj.attach_outcore_cderi(path)
        mf = scf.RHF(m).density_fit(with_df=obj)
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
        with eager_gmres():
            mf.kernel()
        return mf

    def objective(mf):
        return (mf.e_tot + .01 * jnp.sum(mf.get_fock() * seeds)
                + .02 * jnp.sum(mf.mo_energy) + .003 * jnp.sum(mf.mo_coeff))

    calls, receipts, responses = [], [], []
    contract = _df_jk_factorized.contract_geometry
    finish = _ooc_cotangent.GeometryScope.finish
    def counted(contributions):
        calls.append(len(contributions))
        return contract(contributions)
    def finished(scope, bar):
        result = finish(scope, bar)
        receipts.append(dict(scope.stats))
        return result
    monkeypatch.setattr(_df_jk_factorized, 'contract_geometry', counted)
    monkeypatch.setattr(_ooc_cotangent.GeometryScope, 'finish', finished)
    monkeypatch.setenv('PYSCFAD_FACTOR_PANEL_MEMORY_MB', '0')
    with (config_update('pyscfad_moleintor_opt', True),
          config_update('pyscfad_scf_df_precond', True)):
        obj = source(mol)
        obj._cderi_to_save = path
        obj.build()
        values, work = [], []
        for enabled in (False, True):
            calls.clear()
            mf, scf_pull = driver._initialize_scf(mol, build, combine_df=enabled)
            value, pull = jax.vjp(objective, mf)
            with (_scf_response_cache.record_response_caches() as response,
                  driver._global_df_response({'mf': mf})):
                mf_bar, = pull(jnp.ones_like(value))
                mol_bar, = scf_pull(mf_bar)
                jax.block_until_ready(mol_bar)
            values.append((value, mol_bar.coords))
            work.append(len(calls))
            responses.append(response)
    np.testing.assert_allclose(values[1][0], values[0][0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(values[1][1], values[0][1], atol=3e-9, rtol=0)
    assert work[0] > 1 and work[1] == 1, work
    assert len(receipts) == 1
    assert receipts[0]['consumers'] > 1
    assert receipts[0]['metric_builds'] == receipts[0]['cholesky_builds'] == 1
    assert receipts[0]['disk_contributions'] > 0
    assert any(r['enabled'] and r['cache_hit_blocks'] > 0 for r in responses[1])
    # The same immutable SCF tape supports a new seed and a fresh accumulator.
    with (config_update('pyscfad_moleintor_opt', True),
          config_update('pyscfad_scf_df_precond', True),
          driver._global_df_response({'mf': mf})):
        mf_bar, = pull(jnp.asarray(-.4))
        repeated, = scf_pull(mf_bar)
    np.testing.assert_allclose(repeated.coords, -.4 * values[0][1], atol=3e-9, rtol=0)
    assert receipts[-1]['geometry_passes'] == 1
    assert _ooc_cotangent._scope.get() is None
    assert _scf_response_cache._current.get() is None


def test_response_budgets_follow_total_headroom_and_explicit_caps(monkeypatch):
    from types import SimpleNamespace
    from pyscf import lib
    monkeypatch.setattr(lib, 'current_memory', lambda: (40000., 0.))
    obj = SimpleNamespace(precond_total_memory=None,
                          precond_workspace_memory=None, max_memory=4000.,
                          mol=SimpleNamespace(max_memory=100000.))
    monkeypatch.delenv('PYSCFAD_SCF_OCCUPIED_CACHE', raising=False)
    monkeypatch.delenv('PYSCFAD_SCF_RESPONSE_FACTOR_CACHE_MB', raising=False)
    monkeypatch.delenv('PYSCFAD_FACTOR_PANEL_MEMORY_MB', raising=False)
    monkeypatch.delenv('PYSCFAD_PRECOND_CACHE_MEMORY_MB', raising=False)
    with _scf_response_cache.memory_scope(obj):
        assert _scf_response_cache._enabled()
        assert _scf_response_cache._budget_bytes() == 30000 * 10**6
        assert _df_jk_factorized._memory_mb('PYSCFAD_FACTOR_PANEL_MEMORY_MB', 256) == 30000e6 / 2**20
        from pyscfad.df import _df_jk_opt
        plan = _df_jk_opt._precond_memory_plan(obj, 1188, 2840, 81, 1107)
        assert plan['total_memory_mb'] == 100000 and plan['use_cache']
        monkeypatch.setenv('PYSCFAD_FACTOR_PANEL_MEMORY_MB', '8')
        monkeypatch.setenv('PYSCFAD_SCF_RESPONSE_FACTOR_CACHE_MB', '16')
        assert _df_jk_factorized._memory_mb('PYSCFAD_FACTOR_PANEL_MEMORY_MB', 256) == 8
        assert _scf_response_cache._budget_bytes() == 16 * 2**20
        monkeypatch.setattr(lib, 'current_memory', lambda: (100001., 0.))
        assert _scf_response_cache._budget_bytes() == 0
        assert _df_jk_factorized._memory_mb('PYSCFAD_FACTOR_PANEL_MEMORY_MB', 256) == 0
    assert _scf_response_cache.automatic_budget_mb() is None


def test_public_w_ccsd_stochastic_triples_combines_global_response(
        molecular_case, monkeypatch):
    from pyscfad.dlno_stc import ccsd
    case = molecular_case
    path = case.mf.with_df._cderi_to_save
    def build(m):
        mf = scf.RHF(m).density_fit()
        mf.with_df.attach_outcore_cderi(path)
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
        with eager_gmres():
            mf.kernel()
        return mf
    static = ccsd.build_static_selections(case.mf, case.domains, matrix_kind='W',
        thresh_occ=1e-4, thresh_vir=3e-5, baseline_controls=case.controls,
        matrix_controls=case.controls, single_pass=True, lis_space='domain')
    receipts = []
    finish = _ooc_cotangent.GeometryScope.finish
    def finished(scope, bar):
        result = finish(scope, bar)
        receipts.append(dict(scope.stats))
        return result
    monkeypatch.setattr(_ooc_cotangent.GeometryScope, 'finish', finished)
    results = []
    for enabled in ('0', '1'):
        monkeypatch.setenv('PYSCFAD_STC_COMBINE_SCF_DF', enabled)
        results.append(ccsd.value_and_grad(case.mf.mol, build, static,
            baseline_controls=case.controls, matrix_controls=case.controls,
            fragment_indices=case.indices[:1], include_baseline=False,
            ccsd_t=True, return_components=True,
            triples_controls=dict(mode='stochastic', integral_transform='rotate',
                nsamples=16, pilot_samples=16, energy_tolerance=1e-3, global_seed=71)))
        jax.block_until_ready(results[-1][:2])
    np.testing.assert_allclose(results[1][0], results[0][0], atol=1e-11, rtol=0)
    np.testing.assert_allclose(results[1][1].coords, results[0][1].coords, atol=2e-8, rtol=0)
    assert len(receipts) == receipts[0]['geometry_passes'] == 1
    assert receipts[0]['consumers'] > 1
