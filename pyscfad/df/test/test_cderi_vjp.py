import h5py
import jax
import numpy

from pyscfad import config, gto
from pyscfad import numpy as np
from pyscfad.ao2mo import _ao2mo
from pyscfad.df import _cderi_vjp, addons, incore


def test_nr_e2_cderi_bar_packed_blocks_match_full_vjp():
    rng = numpy.random.default_rng(12)
    naux = 4
    nao = 5
    nmo = 4
    npair = nao * (nao + 1) // 2
    orbs_slice = (0, 3, 1, 4)

    cderi = rng.normal(size=(naux, npair))
    mo_coeff = rng.normal(size=(nao, nmo))
    ybar = rng.normal(size=(naux, (orbs_slice[1] - orbs_slice[0]) *
                            (orbs_slice[3] - orbs_slice[2])))

    def fn(cderi_):
        return _ao2mo.nr_e2(cderi_, np.asarray(mo_coeff), orbs_slice, aosym='s2')

    _, pullback = jax.vjp(fn, np.asarray(cderi))
    cderi_bar_ref = numpy.asarray(pullback(np.asarray(ybar))[0])

    blocks = []
    for p0 in range(0, npair, 4):
        p1 = min(p0 + 4, npair)
        blocks.append(
            _cderi_vjp.nr_e2_cderi_bar_packed_block(
                np.asarray(mo_coeff),
                np.asarray(ybar),
                orbs_slice,
                numpy.arange(p0, p1, dtype=numpy.int64),
            )
        )
    cderi_bar = numpy.concatenate(blocks, axis=1)

    assert numpy.allclose(cderi_bar, cderi_bar_ref, atol=1e-10, rtol=1e-10)


def test_nr_e2_mo_coeff_vjp_from_cderi_source_is_blocked(tmp_path):
    rng = numpy.random.default_rng(13)
    naux = 5
    nao = 4
    nmo = 4
    npair = nao * (nao + 1) // 2
    orbs_slice = (0, 2, 1, 4)

    cderi = rng.normal(size=(naux, npair))
    mo_coeff = rng.normal(size=(nao, nmo))
    ybar = rng.normal(size=(naux, (orbs_slice[1] - orbs_slice[0]) *
                            (orbs_slice[3] - orbs_slice[2])))

    cderi_file = tmp_path / 'cderi.h5'
    with h5py.File(cderi_file, 'w') as h5f:
        h5f.create_dataset('j3c', data=cderi)

    def fn(mo_coeff_):
        return _ao2mo.nr_e2(np.asarray(cderi), mo_coeff_, orbs_slice, aosym='s2')

    _, pullback = jax.vjp(fn, np.asarray(mo_coeff))
    mo_coeff_bar_ref = numpy.asarray(pullback(np.asarray(ybar))[0])

    mo_coeff_bar = _cderi_vjp.nr_e2_mo_coeff_vjp_from_cderi_source(
        str(cderi_file),
        np.asarray(mo_coeff),
        np.asarray(ybar),
        orbs_slice,
        aosym='s2',
    )

    assert numpy.allclose(mo_coeff_bar, mo_coeff_bar_ref, atol=1e-10, rtol=1e-10)


def test_nr_e2_mo_coeff_vjp_from_local_cderi_source(tmp_path):
    rng = numpy.random.default_rng(14)
    naux = 5
    nao = 5
    local_nao = 3
    nmo = 3
    npair = nao * (nao + 1) // 2
    orbs_slice = (0, 2, 1, 3)

    cderi = rng.normal(size=(naux, npair))
    mo_coeff = rng.normal(size=(local_nao, nmo))
    ybar = rng.normal(size=(naux, (orbs_slice[1] - orbs_slice[0]) *
                            (orbs_slice[3] - orbs_slice[2])))

    ao_idx = numpy.asarray([0, 2, 4])
    rows, cols = numpy.tril_indices(local_nao)
    pair_idx = ao_idx[rows] * (ao_idx[rows] + 1) // 2 + ao_idx[cols]
    local_cderi = cderi[:, pair_idx]

    cderi_file = tmp_path / 'cderi.h5'
    with h5py.File(cderi_file, 'w') as h5f:
        h5f.create_dataset('j3c', data=cderi)

    def fn(mo_coeff_):
        return _ao2mo.nr_e2(
            np.asarray(local_cderi), mo_coeff_, orbs_slice, aosym='s2'
        )

    _, pullback = jax.vjp(fn, np.asarray(mo_coeff))
    mo_coeff_bar_ref = numpy.asarray(pullback(np.asarray(ybar))[0])

    mo_coeff_bar = _cderi_vjp.nr_e2_mo_coeff_vjp_from_cderi_source(
        str(cderi_file),
        np.asarray(mo_coeff),
        np.asarray(ybar),
        orbs_slice,
        aosym='s2',
        pair_idx=tuple(pair_idx.tolist()),
    )

    assert numpy.allclose(mo_coeff_bar, mo_coeff_bar_ref, atol=1e-10, rtol=1e-10)


def test_cholesky_eri_mo_deriv_vjp_matches_cderi_bar_path(tmp_path):
    rng = numpy.random.default_rng(15)
    mol = gto.Mole(
        atom='O 0 0 0; H 0 0 1; H 0 1 0',
        basis='sto-3g',
        verbose=0,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    auxmol = addons.make_auxmol(mol, 'weigend')
    old_moleintor_opt = config.moleintor_opt
    config.update('pyscfad_moleintor_opt', True)
    try:
        cderi = numpy.asarray(
            incore.cholesky_eri(
                mol,
                auxmol=auxmol,
                int3c=mol._add_suffix('int3c2e'),
                int2c=mol._add_suffix('int2c2e'),
                aosym='s2ij',
                verbose=0,
            )
        )
    finally:
        config.update('pyscfad_moleintor_opt', old_moleintor_opt)
    cderi_file = tmp_path / 'cderi.h5'
    with h5py.File(cderi_file, 'w') as h5f:
        h5f.create_dataset('j3c', data=cderi)

    nao = mol.nao
    nmo = 4
    mo_coeff = rng.normal(size=(nao, nmo))
    orbs_slice = (0, nmo, 0, nmo)
    ybar = rng.normal(size=(auxmol.nao, nmo * nmo))
    def cderi_bar_block(p0, p1):
        return _cderi_vjp.nr_e2_cderi_bar_packed_block(
            np.asarray(mo_coeff),
            np.asarray(ybar),
            orbs_slice,
            numpy.arange(p0, p1, dtype=numpy.int64),
        )

    mol_ref, aux_ref = _cderi_vjp.cholesky_eri_vjp_from_cderi_block_fn(
        mol,
        auxmol,
        str(cderi_file),
        cderi_bar_block,
        1024,
        int3c=mol._add_suffix('int3c2e'),
        int2c=mol._add_suffix('int2c2e'),
        aosym='s2ij',
    )
    mol_test, aux_test = _cderi_vjp.cholesky_eri_vjp_from_mo_coeff_ybar(
        mol,
        auxmol,
        str(cderi_file),
        np.asarray(mo_coeff),
        np.asarray(ybar),
        orbs_slice,
        1024,
        int3c=mol._add_suffix('int3c2e'),
        int2c=mol._add_suffix('int2c2e'),
        aosym='s2ij',
    )

    assert numpy.allclose(
        numpy.asarray(mol_test.coords),
        numpy.asarray(mol_ref.coords),
        atol=1e-8,
        rtol=1e-8,
    )
    assert numpy.allclose(
        numpy.asarray(aux_test.coords),
        numpy.asarray(aux_ref.coords),
        atol=1e-8,
        rtol=1e-8,
    )
