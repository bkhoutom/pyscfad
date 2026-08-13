"""Correctness tests for the factorized local DF coordinate response."""

import h5py
import jax
from jax.tree_util import tree_flatten
import numpy
import pytest

from pyscf import df as pyscf_df
from pyscfad import gto
from pyscfad import numpy as np
from pyscfad.ao2mo import _ao2mo
from pyscfad.df import _cderi_vjp, addons
from pyscfad.lno import lno_base


def _coords(tree):
    return numpy.asarray(tree_flatten(tree)[0][0])


def test_zero_padded_local_coeff_reproduces_forward_transform():
    rng = numpy.random.default_rng(16)
    global_nao = 8
    ao_idx = numpy.asarray([0, 2, 5, 7], dtype=numpy.int64)
    local_nao = ao_idx.size
    naux = 3
    nmo = 7
    orbs_slice = (0, 2, 2, 7)

    mo_local = rng.normal(size=(local_nao, nmo))
    pair_idx = lno_base._global_pair_indices_for_local_ao(ao_idx, global_nao)
    mo_full, recovered = lno_base._zero_pad_local_mo_coeff(
        mo_local, pair_idx, global_nao
    )
    numpy.testing.assert_array_equal(recovered, ao_idx)
    numpy.testing.assert_array_equal(mo_full[ao_idx], mo_local)
    outside = numpy.setdiff1d(numpy.arange(global_nao), ao_idx)
    numpy.testing.assert_array_equal(mo_full[outside], 0.0)

    global_pairs = global_nao * (global_nao + 1) // 2
    cderi = rng.normal(size=(naux, global_pairs))
    local = _ao2mo.nr_e2(
        np.asarray(cderi[:, pair_idx]), np.asarray(mo_local), orbs_slice,
        aosym='s2',
    )
    full = _ao2mo.nr_e2(
        np.asarray(cderi), np.asarray(mo_full), orbs_slice, aosym='s2'
    )
    numpy.testing.assert_allclose(full, local, atol=5e-14, rtol=5e-14)

    corrupted = pair_idx.copy()
    corrupted[1] += 1
    with pytest.raises(ValueError, match='complete packed triangle'):
        lno_base._recover_local_ao_indices(
            corrupted, local_nao, global_nao
        )


def test_factorized_local_custom_vjp_matches_pairwise(tmp_path, monkeypatch):
    rng = numpy.random.default_rng(17)
    mol = gto.Mole(
        atom='O 0 0 0; H 0 0 1; H 0 1 0',
        basis='sto-3g',
        verbose=0,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    auxmol = addons.make_auxmol(mol, 'weigend')
    cderi = pyscf_df.incore.cholesky_eri(
        mol.to_pyscf(), auxmol=auxmol.to_pyscf(), aosym='s2ij'
    )
    cderi_file = tmp_path / 'local-cderi.h5'
    with h5py.File(cderi_file, 'w') as h5f:
        h5f.create_dataset('j3c', data=cderi)

    ao_start, ao_stop = mol.aoslice_by_atom()[0, 2:4]
    ao_idx = numpy.arange(ao_start, ao_stop, dtype=numpy.int64)
    pair_idx = lno_base._global_pair_indices_for_local_ao(ao_idx, mol.nao)
    mo_local = np.asarray(rng.normal(size=(ao_idx.size, 5)))
    orbs_slice = (0, 2, 2, 5)
    ybar = np.asarray(rng.normal(size=(auxmol.nao, 6)))

    direct_calls = 0
    packed_calls = 0
    direct_impl = _cderi_vjp.cholesky_eri_vjp_from_mo_coeff_ybar
    packed_impl = _cderi_vjp.cholesky_eri_vjp_from_cderi_block_fn

    def direct_spy(*args, **kwargs):
        nonlocal direct_calls
        direct_calls += 1
        return direct_impl(*args, **kwargs)

    def packed_spy(*args, **kwargs):
        nonlocal packed_calls
        packed_calls += 1
        return packed_impl(*args, **kwargs)

    monkeypatch.setattr(
        _cderi_vjp, 'cholesky_eri_vjp_from_mo_coeff_ybar', direct_spy
    )
    monkeypatch.setattr(
        _cderi_vjp, 'cholesky_eri_vjp_from_cderi_block_fn', packed_spy
    )

    def local_transform(mol_, auxmol_, mo_coeff_):
        return lno_base._outcore_local_nr_e2_from_global_cderi(
            mol_, auxmol_, mo_coeff_, str(cderi_file), 1024, orbs_slice,
            's2', tuple(pair_idx.tolist()),
        )

    monkeypatch.setenv('PYSCFAD_LNO_DF_REVERSE_MODE', 'pairwise')
    forward_ref, pullback_ref = jax.vjp(
        local_transform, mol, auxmol, mo_local
    )
    bars_ref = pullback_ref(ybar)

    monkeypatch.setenv('PYSCFAD_LNO_DF_REVERSE_MODE', 'factorized')
    forward_test, pullback_test = jax.vjp(
        local_transform, mol, auxmol, mo_local
    )
    bars_test = pullback_test(ybar)

    numpy.testing.assert_allclose(
        numpy.asarray(forward_test), numpy.asarray(forward_ref),
        atol=0.0, rtol=0.0,
    )
    numpy.testing.assert_allclose(
        _coords(bars_test[0]), _coords(bars_ref[0]),
        atol=1e-8, rtol=1e-10,
    )
    numpy.testing.assert_allclose(
        _coords(bars_test[1]), _coords(bars_ref[1]),
        atol=1e-8, rtol=1e-10,
    )
    numpy.testing.assert_allclose(
        numpy.asarray(bars_test[2]), numpy.asarray(bars_ref[2]),
        atol=1e-10, rtol=1e-10,
    )
    assert packed_calls == 1
    assert direct_calls == 1


def test_local_df_reverse_mode_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv('PYSCFAD_LNO_DF_REVERSE_MODE', 'invalid')
    with pytest.raises(ValueError, match='must be auto, factorized, or pairwise'):
        lno_base._local_df_reverse_mode()
