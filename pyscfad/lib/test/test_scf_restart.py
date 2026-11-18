"""Saved RHF roots retain orbital/force response without SCF iterations."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad import config_update, gto, scf
from pyscfad.dlno_stc import driver
from pyscfad.dlno_stc import _ccsd_restart_reference as checkpoint
from pyscfad.scf._restart import restore_scf
from pyscfad.tools import linear_solver


@pytest.mark.parametrize('backend', ['df', 'sgx'])
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
        return mf, value, grad.coords

    with config_update('pyscfad_scf_implicit_diff', True):
        original, expected, expected_grad = evaluate(fresh)
        checkpoint.save_early_scf_reference(tmp_path, original)
        saved = checkpoint.load_scf_reference(tmp_path, mol)
        original_bytes = (tmp_path / '.scf-reference.h5').read_bytes()

        def forbidden(*args, **kwargs):
            raise AssertionError('restart ran an SCF solve')

        monkeypatch.setattr(scf.hf, 'kernel', forbidden)
        monkeypatch.setattr(scf.hf, '_scf', forbidden)
        monkeypatch.setattr(scf.hf.SCF, 'get_init_guess', forbidden)

        def restored(m):
            return checkpoint.restore_scf_reference(new_mf(m), saved)

        actual, value, grad = evaluate(restored)
        for name, array in saved['canonical'].items():
            np.testing.assert_array_equal(getattr(actual, name), array)
        np.testing.assert_allclose(value, expected, atol=2e-11, rtol=0.)
        np.testing.assert_allclose(grad, expected_grad, atol=2e-7, rtol=0.)
        assert (tmp_path / '.scf-reference.h5').read_bytes() == original_bytes


def test_saved_response_rejects_nonimplicit_mode():
    mol = gto.Mole(atom='H 0 0 0; H 0 0 .7', basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    with config_update('pyscfad_scf_implicit_diff', False):
        with pytest.raises(NotImplementedError, match='implicit'):
            restore_scf(scf.RHF(mol), {})
