import h5py
import jax
import numpy

from pyscfad import numpy as np
from pyscfad.ao2mo import _ao2mo
from pyscfad.df import _cderi_vjp


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


def test_nr_e2_mo_coeff_vjp_from_cderi_source_is_blocked(tmp_path, monkeypatch):
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

    monkeypatch.setenv('PYSCFAD_DF_NR_E2_VJP_BLOCK_MB', '0.001')
    mo_coeff_bar = _cderi_vjp.nr_e2_mo_coeff_vjp_from_cderi_source(
        str(cderi_file),
        np.asarray(mo_coeff),
        np.asarray(ybar),
        orbs_slice,
        aosym='s2',
    )

    assert numpy.allclose(mo_coeff_bar, mo_coeff_bar_ref, atol=1e-10, rtol=1e-10)


def test_nr_e2_mo_coeff_vjp_from_local_cderi_source(tmp_path, monkeypatch):
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

    monkeypatch.setenv('PYSCFAD_DF_NR_E2_VJP_BLOCK_MB', '0.001')
    mo_coeff_bar = _cderi_vjp.nr_e2_mo_coeff_vjp_from_cderi_source(
        str(cderi_file),
        np.asarray(mo_coeff),
        np.asarray(ybar),
        orbs_slice,
        aosym='s2',
        pair_idx=tuple(pair_idx.tolist()),
    )

    assert numpy.allclose(mo_coeff_bar, mo_coeff_bar_ref, atol=1e-10, rtol=1e-10)
