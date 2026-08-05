import h5py
import jax
import numpy
import pytest

from pyscfad import config, gto
from pyscfad import numpy as np
from pyscfad.ao2mo import _ao2mo
from pyscfad.df import _cderi_vjp, addons, incore


@pytest.mark.parametrize('project_mode', ('legacy', 'swapped', 'auto'))
def test_nr_e2_cderi_bar_packed_blocks_match_full_vjp(
        monkeypatch, project_mode):
    monkeypatch.setenv('PYSCFAD_DF_AO_PROJECTION_ORDER', project_mode)
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


def _assert_projection_close(actual, expected, tolerance=5e-12):
    difference = numpy.asarray(actual) - numpy.asarray(expected)
    scale = max(
        1.0,
        float(numpy.abs(actual).max(initial=0.0)),
        float(numpy.abs(expected).max(initial=0.0)),
    )
    norm_scale = max(float(numpy.linalg.norm(expected)), 1.0)
    assert float(numpy.abs(difference).max(initial=0.0)) / scale <= tolerance
    assert float(numpy.linalg.norm(difference)) / norm_scale <= tolerance


@pytest.mark.parametrize('backend', ('native', 'numpy'))
@pytest.mark.parametrize('project_mode', ('legacy', 'swapped', 'auto'))
@pytest.mark.parametrize(('kc', 'lc'), ((2, 7), (7, 2), (4, 4)))
def test_nr_e2_cderi_bar_project_modes_noncontiguous(
        monkeypatch, backend, project_mode, kc, lc):
    if backend == 'native':
        if not _cderi_vjp._NR_E2_CDERI_BAR_NATIVE:
            pytest.skip('native AO projection kernel is unavailable')
    else:
        monkeypatch.setattr(_cderi_vjp, '_NR_E2_CDERI_BAR_NATIVE', False)

    rng = numpy.random.default_rng(121)
    naux = 5
    npos = 11
    ybar = rng.normal(size=(naux, kc, 2 * lc))[..., ::2]
    mok_rows = rng.normal(size=(npos, 2 * kc))[:, ::2]
    mol_cols = rng.normal(size=(npos, 2 * lc))[:, ::2]
    assert not ybar.flags.c_contiguous
    assert not mok_rows.flags.c_contiguous
    assert not mol_cols.flags.c_contiguous

    reference = numpy.einsum(
        'Lij,pi,pj->Lp', ybar, mok_rows, mol_cols, optimize=True
    )
    result = _cderi_vjp._nr_e2_cderi_bar_project(
        ybar, mok_rows, mol_cols, order=project_mode
    )
    _assert_projection_close(result, reference)


@pytest.mark.parametrize('project_mode', ('legacy', 'swapped', 'auto'))
def test_nr_e2_cderi_bar_packed_block_unsorted_duplicate_positions(
        monkeypatch, project_mode):
    monkeypatch.setenv('PYSCFAD_DF_AO_PROJECTION_ORDER', project_mode)
    rng = numpy.random.default_rng(122)
    naux, nao, nmo = 6, 5, 8
    orbs_slice = (0, 2, 2, 8)
    mo_coeff = rng.normal(size=(nao, nmo))
    ybar = rng.normal(size=(naux, 12))
    positions = numpy.asarray([8, 0, 4, 8, 14, 2, 1], dtype=numpy.int64)

    rows, cols = numpy.tril_indices(nao)
    y3 = ybar.reshape(naux, 2, 6)
    reference = numpy.einsum(
        'Lij,pi,pj->Lp',
        y3,
        mo_coeff[rows[positions], :2],
        mo_coeff[cols[positions], 2:8],
        optimize=True,
    )
    offdiag = rows[positions] != cols[positions]
    reference[:, offdiag] += numpy.einsum(
        'Lij,pi,pj->Lp',
        y3,
        mo_coeff[cols[positions[offdiag]], :2],
        mo_coeff[rows[positions[offdiag]], 2:8],
        optimize=True,
    )
    result = _cderi_vjp.nr_e2_cderi_bar_packed_block(
        np.asarray(mo_coeff), np.asarray(ybar), orbs_slice, positions
    )
    _assert_projection_close(result, reference)

    empty = _cderi_vjp.nr_e2_cderi_bar_packed_block(
        np.asarray(mo_coeff), np.asarray(ybar), orbs_slice,
        numpy.empty(0, dtype=numpy.int64),
    )
    assert empty.shape == (naux, 0)


def test_ao_projection_order_resolution_default_and_invalid(monkeypatch):
    monkeypatch.delenv('PYSCFAD_DF_AO_PROJECTION_ORDER', raising=False)
    assert _cderi_vjp._ao_projection_order_mode() == 'auto'

    ybar = numpy.zeros((3, 2, 7))
    mok_rows = numpy.zeros((5, 2))
    mol_cols = numpy.zeros((5, 7))

    selected = _cderi_vjp._select_ao_projection_order(
        ybar, mok_rows, mol_cols, order='auto'
    )
    assert selected[4] == 'swapped'
    assert selected[0].shape == (3, 7, 2)
    selected = _cderi_vjp._select_ao_projection_order(
        ybar.transpose(0, 2, 1), mol_cols, mok_rows, order='auto'
    )
    assert selected[4] == 'legacy'

    monkeypatch.setenv('PYSCFAD_DF_AO_PROJECTION_ORDER', 'invalid')
    with pytest.raises(ValueError, match='must be one of'):
        _cderi_vjp._ao_projection_order_mode()


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


@pytest.mark.parametrize('project_mode', ('legacy', 'swapped', 'auto'))
def test_cholesky_eri_mo_deriv_vjp_matches_cderi_bar_path(
        tmp_path, monkeypatch, project_mode):
    monkeypatch.setenv('PYSCFAD_DF_AO_PROJECTION_ORDER', project_mode)
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
