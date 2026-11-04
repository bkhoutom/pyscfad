"""Explicit J/K workspace keeps OOC response panels useful above the RSS budget."""
import h5py
import jax
import numpy as np
import pytest

from pyscfad import config_update, gto, scf
from pyscfad.df import _jk_memory
from pyscfad.df.df import DF


def test_ooc_scf_workspace_preserves_nonstationary_gradient(tmp_path, monkeypatch):
    mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='sto-3g', verbose=0, max_memory=1000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    path = str(tmp_path / 'factors.h5')
    with config_update('pyscfad_moleintor_opt', True):
        reference = scf.RHF(mol).density_fit(auxbasis='weigend')
        reference.conv_tol = 1e-12
        reference.kernel()
        with h5py.File(path, 'w') as handle:
            handle['j3c'] = np.asarray(reference.with_df._cderi)
    dm0 = np.asarray(reference.make_rdm1())
    def unexpected_build(*args, **kwargs):
        raise AssertionError('attached DF factors must never be rebuilt')
    monkeypatch.setattr(DF, 'build', unexpected_build)
    weights = jax.numpy.asarray(np.random.default_rng(917).normal(size=(mol.nao, 3)) * .02)
    monkeypatch.setattr(_jk_memory.lib, 'current_memory', lambda: (20., 0.))
    original_loop = DF.loop
    rows = []

    def observed_loop(obj, blksize=None, to_numpy=True):
        if blksize is not None:
            rows.append(blksize)
        yield from original_loop(obj, blksize, to_numpy)
    monkeypatch.setattr(DF, 'loop', observed_loop)

    def objective(m, workspace):
        mf = scf.RHF(m).density_fit(auxbasis='weigend')
        mf.with_df.incore = False
        mf.with_df.max_memory = .1
        mf.with_df.jk_workspace_memory = workspace
        mf.with_df.blockdim = 12
        mf.with_df.attach_outcore_cderi(path)
        view = _jk_memory.forward_df(mf.with_df)
        assert view._get_cderi_source() == path
        assert view._cderi is mf.with_df._cderi
        assert view._cderi.shape == (0, 0)
        assert view.auxmol is mf.with_df.auxmol
        assert mf.with_df.max_memory == .1
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-8
        mf.kernel(dm0=dm0)
        assert mf.with_df.max_memory == .1
        projections = weights.T @ mf.mo_coeff[:, :mol.nelectron // 2]
        return mf.e_tot + .5 * jax.numpy.sum(projections ** 2)

    with (config_update('pyscfad_moleintor_opt', True),
          config_update('pyscfad_scf_implicit_diff', True),
          config_update('pyscfad_scf_first_order_custom', False)):
        off_value, off_grad = jax.value_and_grad(lambda m: objective(m, None))(mol)
        assert rows[0] == 4
        minimum_panels = rows.count(4)
        rows.clear()
        on_value, on_grad = jax.value_and_grad(lambda m: objective(m, .1))(mol)
        assert rows[0] == 12 and rows.count(4) < minimum_panels
    np.testing.assert_allclose(on_value, off_value, atol=1e-11, rtol=0)
    np.testing.assert_allclose(on_grad.coords, off_grad.coords, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize('workspace', [0, -1, float('nan'), float('inf')])
def test_jk_workspace_rejects_invalid_allowance(workspace):
    class Object:
        jk_workspace_memory = workspace
    with pytest.raises(ValueError, match='jk_workspace_memory'):
        _jk_memory.workspace_memory(Object())
