"""Saved RHF roots retain orbital/force response without SCF iterations."""
from functools import partial
from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad import config_update, gto, scf
from pyscfad.dlno_stc import driver
from pyscfad.dlno_stc import _ccsd_restart_reference as checkpoint
from pyscfad.scf._restart import restore_scf
from pyscfad.tools import linear_solver


@pytest.mark.parametrize('backend', ['df', 'df-opt', 'df-ooc', 'sgx'])
def test_restored_root_retains_orbital_and_shared_fock_response(tmp_path, monkeypatch, backend):
    mol = gto.Mole(atom='O 0 0 0; H .1 -.75 .57; H 0 .8 .61', basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    monkeypatch.setattr(scf.hf, 'gen_gmres', partial(
        linear_solver.gen_gmres_with_default_kwargs, tol=1e-10, maxiter=100))

    def new_mf(m):
        if backend == 'sgx':
            from pyscfad.sgx import sgx_fit
            obj = sgx_fit(scf.RHF(m), grid_level=0, fit_ovlp=True,
                          integral_backend='generated', screening='per_pair')
        else:
            obj = scf.RHF(m).density_fit()
            if backend == 'df-ooc':
                obj.with_df.incore = False
                obj.with_df.attach_outcore_cderi(str(tmp_path / 'cderi.h5'))
        obj.conv_tol, obj.conv_tol_grad = 1e-12, 1e-9
        return obj

    def fresh(m):
        obj = new_mf(m)
        obj.kernel()
        return obj

    # Include orbital and Fock seeds: energy-only RHF gradients can hide a
    # missing orbital response because RHF energy is stationary.
    def objective(mf, data):
        overlap, fock = data
        seeds = jnp.asarray(np.random.default_rng(13).normal(size=fock.shape))
        return (mf.e_tot + .002*jnp.sum(mf.mo_coeff*seeds)
                + .003*jnp.sum(mf.mo_energy**2) + .004*jnp.sum(fock*seeds)
                + .001*jnp.sum(overlap*seeds))

    def evaluate(build):
        mf, pull = driver._initialize_scf(mol, build, combine_df=True, share_scf_data=True)
        value, seed = jax.value_and_grad(objective, argnums=(0, 1))(mf, pull.scf_data)
        with driver._global_df_response({'mf': mf}):
            grad, = pull(*seed)
            jax.block_until_ready(grad)
        return mf, value, grad.coords, np.asarray(pull.scf_data[1])

    with (config_update('pyscfad_scf_implicit_diff', True),
          config_update('pyscfad_moleintor_opt', backend in ('df-opt', 'df-ooc')),
          linear_solver.eager_gmres() if backend == 'df-ooc' else nullcontext()):
        if backend == 'df-ooc':
            from pyscfad.df import DF
            factors = DF(mol)
            factors._cderi_to_save = str(tmp_path / 'cderi.h5')
            factors.build()
        original, expected, expected_grad, expected_fock = evaluate(fresh)
        checkpoint.save_early_scf_reference(tmp_path, original, fock=expected_fock)
        saved = checkpoint.load_scf_reference(tmp_path, mol)
        original_bytes = (tmp_path / '.scf-reference.h5').read_bytes()

        def forbidden(*args, **kwargs):
            raise AssertionError('restart ran an SCF solve or forward Fock build')

        monkeypatch.setattr(scf.hf, 'kernel', forbidden)
        monkeypatch.setattr(scf.hf, '_scf', forbidden)
        monkeypatch.setattr(scf.hf.SCF, 'get_init_guess', forbidden)
        monkeypatch.setattr(scf.hf.SCF, 'get_fock', forbidden)
        monkeypatch.setattr(scf.hf.RHF, 'get_veff', forbidden)
        if backend == 'sgx':
            from pyscfad.sgx.sgx import _Backend
            monkeypatch.setattr(_Backend, 'primal_jk', forbidden)
        else:
            from pyscfad.df import _df_jk_opt, df_jk
            monkeypatch.setattr(_df_jk_opt, 'get_jk', forbidden)
            monkeypatch.setattr(df_jk, 'get_jk', forbidden)
            monkeypatch.setattr(df_jk, 'get_jk_gen', forbidden)
            monkeypatch.setattr(_df_jk_opt, '_get_jk_gen_jax', forbidden)
            if backend == 'df-ooc':
                monkeypatch.setattr(DF, 'build', forbidden)

        def restored(m):
            return checkpoint.restore_scf_reference(new_mf(m), saved)

        actual, value, grad, fock = evaluate(restored)
        for name, array in saved['canonical'].items():
            np.testing.assert_array_equal(getattr(actual, name), array)
        np.testing.assert_allclose(value, expected, atol=2e-11, rtol=0.)
        np.testing.assert_allclose(grad, expected_grad, atol=2e-7, rtol=0.)
        np.testing.assert_array_equal(fock, expected_fock)
        assert (tmp_path / '.scf-reference.h5').read_bytes() == original_bytes


def test_saved_response_rejects_nonimplicit_mode():
    mol = gto.Mole(atom='H 0 0 0; H 0 0 .7', basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    with config_update('pyscfad_scf_implicit_diff', False):
        with pytest.raises(NotImplementedError, match='implicit'):
            restore_scf(scf.RHF(mol), {})


@pytest.mark.parametrize('backend', ['sgx', 'df-ooc'])
def test_lambda_and_force_assembly_resume_without_repeating_completed_work(
        tmp_path, monkeypatch, backend):
    from pyscfad.df import DF
    from pyscfad.dlno_stc import _restart as storage
    from pyscfad.tools import _response_checkpoint as response
    mol = gto.Mole(atom='O 0 0 0; H .1 -.75 .57; H 0 .8 .61', basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    monkeypatch.setattr(scf.hf, 'gen_gmres', partial(
        linear_solver.gen_gmres_with_default_kwargs, tol=1e-10, maxiter=100, restart=2))

    def new_mf(current):
        if backend == 'sgx':
            from pyscfad.sgx import sgx_fit
            obj = sgx_fit(scf.RHF(current), grid_level=0, fit_ovlp=True,
                          integral_backend='generated', screening='per_pair')
        else:
            obj = scf.RHF(current).density_fit()
            obj.with_df.attach_outcore_cderi(str(tmp_path / 'factors.h5'))
        obj.conv_tol, obj.conv_tol_grad = 1e-12, 1e-9
        return obj

    def fresh(current):
        obj = new_mf(current)
        obj.kernel()
        return obj

    def objective(mf, data):
        seed = jnp.asarray(np.random.default_rng(18).normal(size=data[1].shape))
        return mf.e_tot + .002*jnp.sum(mf.mo_coeff*seed) + .004*jnp.sum(data[1]*seed)

    def evaluate(build):
        mf, pull = driver._initialize_scf(mol, build, combine_df=True, share_scf_data=True)
        value, bars = jax.value_and_grad(objective, argnums=(0, 1))(mf, pull.scf_data)
        with driver._global_df_response({'mf': mf}):
            gradient, = pull(*bars)
            jax.block_until_ready(gradient)
        return mf, value, gradient.coords, np.asarray(pull.scf_data[1])

    class Interrupted(RuntimeError):
        pass

    with (config_update('pyscfad_scf_implicit_diff', True),
          config_update('pyscfad_moleintor_opt', True), linear_solver.eager_gmres()):
        if backend == 'df-ooc':
            factors = DF(mol)
            factors._cderi_to_save = str(tmp_path / 'factors.h5')
            factors.build()
        mf, expected, expected_grad, fock = evaluate(fresh)
        checkpoint.save_early_scf_reference(tmp_path / 'reference', mf, fock=fock)
        saved = checkpoint.load_scf_reference(tmp_path / 'reference', mol)
        restore = lambda current: checkpoint.restore_scf_reference(new_mf(current), saved)

        def forbidden(*args, **kwargs):
            pytest.fail('restart repeated completed SCF/Fock/response/contraction work')
        monkeypatch.setattr(scf.hf, 'kernel', forbidden)
        monkeypatch.setattr(scf.hf.SCF, 'get_fock', forbidden)
        monkeypatch.setattr(scf.hf.RHF, 'get_veff', forbidden)
        force_stage = 'scf-force-sgx-j' if backend == 'sgx' else 'scf-force-df-geometry'
        stages = ['scf-lambda-iterate', 'scf-lambda-converged', force_stage]
        for index, stage in enumerate(stages):
            manager = storage.RestartManager(tmp_path / 'response', resume=index > 0,
                method='test-response', scientific_payload={'backend': backend})
            def interrupt(current, key, path):
                if current == stage:
                    raise Interrupted(stage)
            monkeypatch.setattr(storage, '_CHECKPOINT_EVENT_HOOK', interrupt)
            try:
                with response.scope(manager), pytest.raises(Interrupted, match=stage):
                    evaluate(restore)
            finally:
                manager.close()
            if stage == 'scf-lambda-converged':
                monkeypatch.setattr(response.LambdaCheckpoint, 'solve', forbidden)

        monkeypatch.setattr(storage, '_CHECKPOINT_EVENT_HOOK', None)
        if backend == 'sgx':
            from pyscfad.sgx import _dfj
            monkeypatch.setattr(_dfj, 'geometry_many', forbidden)
        else:
            from pyscfad.df import _df_jk_factorized
            monkeypatch.setattr(_df_jk_factorized, 'contract_geometry', forbidden)
        manager = storage.RestartManager(tmp_path / 'response', resume=True,
            method='test-response', scientific_payload={'backend': backend})
        try:
            with response.scope(manager):
                _, actual, gradient, _ = evaluate(restore)
        finally:
            manager.close()
        np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=0.)
        np.testing.assert_allclose(gradient, expected_grad, atol=2e-7, rtol=0.)
