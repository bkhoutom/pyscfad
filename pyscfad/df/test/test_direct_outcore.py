"""Final-layout DF factors preserve the existing gauge and SCF response."""
import h5py
import jax
import numpy as np
import pytest
from pyscf import gto as pgto
from pyscf.df import outcore

from pyscfad import config_update, gto, scf
from pyscfad.df import direct_outcore
from pyscfad.df.df import DF


def _water():
    return pgto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                  basis='cc-pvdz', verbose=0)


@pytest.mark.parametrize('aosym,cart', [('s2ij', False), ('s1', False), ('s2ij', True)])
def test_factors_match_standard_with_bounded_panels(tmp_path, monkeypatch, aosym, cart):
    mol = _water()
    mol.cart = cart
    standard = str(tmp_path / 'standard.h5')
    direct = tmp_path / 'direct.h5'
    # Use a modest allowance that requires several panels in the direct path.
    memory_mb = 1.
    outcore.cholesky_eri(mol, standard, auxbasis='weigend', aosym=aosym,
                         max_memory=memory_mb, verbose=0)
    writes = []
    original_write = direct_outcore._write_panel

    def write(dataset, panel, col0, tile_bytes):
        # There is exactly one newly constructed tensor, already in final shape.
        partials = list(tmp_path.glob('.direct.h5.direct-*.h5'))
        assert len(partials) == 1 and not direct.exists()
        writes.append((panel.shape, tile_bytes))
        return original_write(dataset, panel, col0, tile_bytes)

    def forbidden(*args, **kwargs):
        raise AssertionError('the direct builder must never create an upstream swap tensor')
    monkeypatch.setattr(direct_outcore, '_write_panel', write)
    monkeypatch.setattr(outcore, 'cholesky_eri_b', forbidden)
    direct_outcore.cholesky_eri(mol, direct, auxbasis='weigend', aosym=aosym,
                                max_memory=memory_mb, verbose=0)
    with h5py.File(standard) as first, h5py.File(direct) as second:
        np.testing.assert_allclose(second['j3c'][:], first['j3c'][:], atol=2e-12, rtol=2e-12)
        assert second['j3c'].chunks is None
        assert second['j3c'].dtype == np.float64
        assert second.attrs['pyscfad_direct_df_complete']
        assert second.attrs['planned_array_peak_bytes'] <= .8 * memory_mb * 1e6
        assert second.attrs['write_tile_bytes'] <= .08 * memory_mb * 1e6
    assert len(writes) > 1
    assert not list(tmp_path.glob('.direct.h5.direct-*'))


def test_insufficient_workspace_fails_before_integrals(tmp_path, monkeypatch):
    mol = _water()
    auxmol = direct_outcore.make_auxmol(mol, 'weigend')

    def forbidden(*args, **kwargs):
        raise AssertionError('memory validation must precede integral allocation')
    monkeypatch.setattr(auxmol, 'intor', forbidden)
    with pytest.raises(MemoryError, match='workspace.*at least'):
        direct_outcore.cholesky_eri(mol, tmp_path / 'tiny.h5', auxmol=auxmol,
                                    max_memory=.001, verbose=0)
    assert not list(tmp_path.iterdir())


def test_hdf5_staging_never_exceeds_tile_allowance():
    # Both auxiliary and AO-pair strides require conversion, and even one row
    # is larger than the allowance. Check what is handed to the HDF5 writer.
    panel = np.arange(31 * 509, dtype=float).reshape(509, 31).T
    observed = []

    class Dataset:
        values = np.zeros((31, 520))

        def __setitem__(self, key, value):
            assert value.flags.c_contiguous and value.nbytes <= 512
            observed.append(value.nbytes)
            self.values[key] = value

    dataset = Dataset()
    direct_outcore._write_panel(dataset, panel, 7, 512)
    np.testing.assert_array_equal(dataset.values[:, 7:516], panel)
    assert len(observed) > 31


def test_failed_write_removes_partial_and_preserves_old_file(tmp_path, monkeypatch):
    path = tmp_path / 'factors.h5'
    path.write_bytes(b'previous complete file')
    real_write = direct_outcore._write_panel

    def fail_after_write(*args, **kwargs):
        real_write(*args, **kwargs)
        raise OSError('injected disk full')
    monkeypatch.setattr(direct_outcore, '_write_panel', fail_after_write)
    with pytest.raises(OSError, match='injected disk full'):
        direct_outcore.cholesky_eri(_water(), path, auxbasis='weigend', max_memory=1., verbose=0)
    assert path.read_bytes() == b'previous complete file'
    assert list(tmp_path.iterdir()) == [path]


def test_eager_df_option_uses_current_dynamic_geometry(tmp_path):
    mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='sto-3g', verbose=0, max_memory=2000)
    mol.build()
    mol.coords = mol.coords.at[1, 2].add(.031)
    concrete = mol.to_pyscf().copy()
    concrete.set_geom_(np.asarray(mol.coords), unit='Bohr')
    standard = str(tmp_path / 'standard.h5')
    outcore.cholesky_eri(concrete, standard, auxbasis='weigend', verbose=0)
    obj = DF(mol, auxbasis='weigend', incore=False, outcore_builder='direct')
    obj._cderi_to_save = str(tmp_path / 'direct.h5')
    obj.build()
    with h5py.File(standard) as handle:
        np.testing.assert_allclose(np.vstack(list(obj.loop(blksize=7))), handle['j3c'][:],
                                   atol=2e-12, rtol=2e-12)
    assert obj._cderi.shape == (0, 0)
    # Attaching factors must continue to skip rebuilding them.
    obj.build()


def test_nonstationary_scf_response_matches_standard(tmp_path):
    mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='sto-3g', verbose=0, max_memory=2000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    standard = str(tmp_path / 'standard.h5')
    direct = str(tmp_path / 'direct.h5')
    outcore.cholesky_eri(mol.to_pyscf(), standard, auxbasis='weigend', verbose=0)
    direct_outcore.cholesky_eri(mol, direct, auxbasis='weigend', max_memory=1., verbose=0)
    weights = jax.numpy.asarray(np.random.default_rng(917).normal(size=(mol.nao, 3)) * .02)

    def objective(current, path):
        mf = scf.RHF(current).density_fit(auxbasis='weigend')
        mf.with_df.incore = False
        mf.with_df.jk_workspace_memory = 1.
        mf.with_df.attach_outcore_cderi(path)
        mf.conv_tol, mf.conv_tol_grad, mf.chkfile = 1e-12, 1e-8, None
        mf.kernel()
        assert mf.converged
        # This orbital-dependent contribution exercises the implicit response,
        # beyond the stationary HF energy's ordinary geometry derivative.
        projections = weights.T @ mf.mo_coeff[:, :mol.nelectron // 2]
        return mf.e_tot + .5 * jax.numpy.sum(projections ** 2)

    with (config_update('pyscfad_moleintor_opt', True),
          config_update('pyscfad_scf_implicit_diff', True),
          config_update('pyscfad_scf_first_order_custom', False)):
        first, first_grad = jax.value_and_grad(lambda m: objective(m, standard))(mol)
        second, second_grad = jax.value_and_grad(lambda m: objective(m, direct))(mol)
    np.testing.assert_allclose(second, first, atol=1e-11, rtol=0)
    np.testing.assert_allclose(second_grad.coords, first_grad.coords, atol=2e-9, rtol=2e-9)
