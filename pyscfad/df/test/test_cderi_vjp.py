import os

import h5py
import jax
import numpy
import pytest

from pyscfad import config, gto
from pyscfad import numpy as np
from pyscfad.ao2mo import _ao2mo
from pyscfad.df import _cderi_vjp, addons, incore


@pytest.mark.parametrize(('kc', 'lc'), ((2, 7), (7, 2), (4, 4)))
@pytest.mark.parametrize('complex_data', (False, True))
def test_int3c_ip1_mo_density_contraction_matches_einstein_reference(
        kc, lc, complex_data):
    rng = numpy.random.default_rng(1201)
    naux = 5
    nao = 6
    ints = rng.normal(size=(3, nao, nao, naux))
    mo_k = rng.normal(size=(nao, kc))
    mo_l = rng.normal(size=(nao, lc))
    z_blk = rng.normal(size=(naux, kc, lc))
    if complex_data:
        ints = ints + 1j * rng.normal(size=ints.shape)
        mo_k = mo_k + 1j * rng.normal(size=mo_k.shape)
        mo_l = mo_l + 1j * rng.normal(size=mo_l.shape)
        z_blk = z_blk + 1j * rng.normal(size=z_blk.shape)

    reference = numpy.einsum(
        'pkl,uk,vl,xuvp->ux',
        z_blk,
        mo_k,
        mo_l,
        ints,
        optimize=True,
    )
    reference += numpy.einsum(
        'pkl,vk,ul,xuvp->ux',
        z_blk,
        mo_k,
        mo_l,
        ints,
        optimize=True,
    )

    result = _cderi_vjp._int3c_ip1_mo_density_contraction(
        ints, mo_k, mo_l, z_blk
    )
    numpy.testing.assert_allclose(result, reference, atol=2e-11, rtol=2e-11)


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
@pytest.mark.parametrize(('kc', 'lc'), ((2, 7), (7, 2), (4, 4)))
def test_nr_e2_cderi_bar_project_noncontiguous(
        monkeypatch, backend, kc, lc):
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
        ybar, mok_rows, mol_cols
    )
    _assert_projection_close(result, reference)


def test_nr_e2_cderi_bar_packed_block_unsorted_duplicate_positions():
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


@pytest.mark.parametrize('backend', ('native', 'numpy'))
@pytest.mark.parametrize(('kc', 'lc'), ((2, 7), (7, 2), (4, 4)))
def test_nr_e2_cderi_bar_packed_aux_block_matches_dense_reference(
        monkeypatch, backend, kc, lc):
    if backend == 'native':
        if not _cderi_vjp._NR_E2_CDERI_BAR_AUX_NATIVE:
            pytest.skip('native auxiliary-slab AO projection kernel is unavailable')
    else:
        monkeypatch.setattr(_cderi_vjp, '_NR_E2_CDERI_BAR_AUX_NATIVE', False)

    rng = numpy.random.default_rng(123)
    naux = 5
    nao = 6
    nmo = kc + lc
    ybar = rng.normal(size=(naux, kc, 2 * lc))[..., ::2]
    mo_coeff = rng.normal(size=(nao, 2 * nmo))[:, ::2]
    assert not ybar.flags.c_contiguous
    assert not mo_coeff.flags.c_contiguous
    orbs_slice = (0, kc, kc, nmo)

    dense = numpy.einsum(
        'Pij,ui,vj->Puv',
        ybar,
        mo_coeff[:, :kc],
        mo_coeff[:, kc:],
        optimize=True,
    )
    rows, cols = numpy.tril_indices(nao)
    reference = dense[:, rows, cols].copy()
    offdiag = rows != cols
    reference[:, offdiag] += dense[:, cols[offdiag], rows[offdiag]]

    result = _cderi_vjp.nr_e2_cderi_bar_packed_aux_block(
        np.asarray(mo_coeff), np.asarray(ybar), orbs_slice
    )
    _assert_projection_close(result, reference)


def test_nr_e2_cderi_bar_packed_disk_aux_write_pair_read(tmp_path, monkeypatch):
    monkeypatch.setattr(_cderi_vjp, '_NR_E2_CDERI_BAR_AUX_NATIVE', False)
    rng = numpy.random.default_rng(124)
    naux, nao, nocc, nvir = 5, 4, 2, 5
    nmo = nocc + nvir
    mo_coeff = rng.normal(size=(nao, nmo))
    ybar = rng.normal(size=(naux, nocc * nvir))
    orbs_slice = (0, nocc, nocc, nmo)
    reference = _cderi_vjp.nr_e2_cderi_bar_packed_aux_block(
        np.asarray(mo_coeff), np.asarray(ybar), orbs_slice
    )

    disk_path = None
    with _cderi_vjp.nr_e2_cderi_bar_packed_disk(
            np.asarray(mo_coeff), np.asarray(ybar), orbs_slice,
            directory=str(tmp_path), aux_blksize=2) as dataset:
        disk_path = dataset.file.filename
        assert dataset.shape == (naux, nao * (nao + 1) // 2)
        assert dataset.chunks[0] == 2
        assert numpy.allclose(numpy.asarray(dataset[:, :]), reference)
        # Exercise the intended second-phase access pattern explicitly.
        pair_blocks = []
        for q0 in range(0, dataset.shape[1], 3):
            q1 = min(q0 + 3, dataset.shape[1])
            pair_blocks.append(numpy.asarray(dataset[:, q0:q1]))
        assert numpy.allclose(numpy.concatenate(pair_blocks, axis=1), reference)
    assert disk_path is not None
    assert not os.path.exists(disk_path)


def test_nr_e2_cderi_bar_aux_blksize_accounts_for_all_slab_workspace(
        monkeypatch):
    monkeypatch.setenv('PYSCFAD_DF_CDERI_BAR_AUX_BLOCK_MB', '1')
    monkeypatch.setattr(_cderi_vjp.pyscf_lib, 'num_threads', lambda: 8)

    naux, nao, kc, lc = 100, 20, 80, 100
    mo_coeff = numpy.empty((nao, kc + lc), dtype=numpy.float64)
    ybar = numpy.empty((naux, kc * lc), dtype=numpy.float64)
    block = _cderi_vjp._nr_e2_cderi_bar_aux_blksize(
        mo_coeff, ybar, (0, kc, kc, kc + lc)
    )

    npair = nao * (nao + 1) // 2
    bytes_per_aux = (npair + nao * kc + kc * lc) * 8
    matrix_bytes = nao * nao * 8
    target_bytes = 1024**2 - nao * (kc + lc) * 8
    assert block * bytes_per_aux + min(block, 8) * matrix_bytes <= target_bytes
    assert (
        (block + 1) * bytes_per_aux
        + min(block + 1, 8) * matrix_bytes
        > target_bytes
    )


def test_nr_e2_cderi_bar_packed_disk_cleans_up_after_build_failure(
        tmp_path, monkeypatch):
    token = object()
    finished = []
    monkeypatch.setattr(_cderi_vjp.resource_profile, 'start', lambda: token)
    monkeypatch.setattr(
        _cderi_vjp.resource_profile,
        'finish',
        lambda phase, before, **details: finished.append((phase, before)),
    )

    def fail_projection(*args, **kwargs):
        raise RuntimeError('projection failed')

    monkeypatch.setattr(
        _cderi_vjp, '_nr_e2_cderi_bar_packed_aux_block_prepared',
        fail_projection,
    )
    mo_coeff = numpy.ones((3, 4))
    ybar = numpy.ones((2, 4))
    with pytest.raises(RuntimeError, match='projection failed'):
        with _cderi_vjp.nr_e2_cderi_bar_packed_disk(
                mo_coeff, ybar, (0, 2, 2, 4), directory=str(tmp_path)):
            pass

    assert finished == [('df_vjp.cderi_bar_disk_build_incomplete', token)]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('failing_call', ('start', 'finish'))
def test_nr_e2_cderi_bar_packed_disk_cleans_up_after_profiler_failure(
        tmp_path, monkeypatch, failing_call):
    token = object()

    def profile_start():
        if failing_call == 'start':
            raise RuntimeError('profile start failed')
        return token

    def profile_finish(*args, **kwargs):
        raise RuntimeError('profile finish failed')

    monkeypatch.setattr(_cderi_vjp.resource_profile, 'start', profile_start)
    monkeypatch.setattr(_cderi_vjp.resource_profile, 'finish', profile_finish)
    mo_coeff = numpy.ones((3, 4))
    ybar = numpy.ones((2, 4))

    with pytest.raises(RuntimeError, match=f'profile {failing_call} failed'):
        with _cderi_vjp.nr_e2_cderi_bar_packed_disk(
                mo_coeff, ybar, (0, 2, 2, 4), directory=str(tmp_path)):
            pass

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ('kc', 'lc', 'expected_order'),
    ((2, 7, 'transposed'), (7, 2, 'as_given'), (4, 4, 'as_given')),
)
def test_ao_projection_order_uses_smaller_trailing_dimension(
        kc, lc, expected_order):
    ybar = numpy.zeros((3, kc, lc))
    mok_rows = numpy.zeros((5, kc))
    mol_cols = numpy.zeros((5, lc))

    selected = _cderi_vjp._select_ao_projection_order(
        ybar, mok_rows, mol_cols
    )
    assert selected[3] == expected_order
    assert selected[0].shape == (3, max(kc, lc), min(kc, lc))


def test_ao_projection_order_validates_shapes():
    with pytest.raises(ValueError, match='rank 3'):
        _cderi_vjp._select_ao_projection_order(
            numpy.zeros((2, 3)),
            numpy.zeros((5, 3)),
            numpy.zeros((5, 3)),
        )
    with pytest.raises(ValueError, match='mok_rows has shape'):
        _cderi_vjp._select_ao_projection_order(
            numpy.zeros((2, 3, 4)),
            numpy.zeros((5, 2)),
            numpy.zeros((5, 4)),
        )
    with pytest.raises(ValueError, match='second dimension'):
        _cderi_vjp._select_ao_projection_order(
            numpy.zeros((2, 3, 4)),
            numpy.zeros((5, 3)),
            numpy.zeros((5, 2)),
        )


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


def test_local_cderi_bar_disk_scatter_matches_pair_projection(tmp_path):
    from pyscfad.lno import lno_base

    rng = numpy.random.default_rng(141)
    naux, global_nao, local_nao = 5, 6, 3
    nocc, nvir = 2, 4
    nmo = nocc + nvir
    mo_coeff = rng.normal(size=(local_nao, nmo))
    ybar = rng.normal(size=(naux, nocc * nvir))
    orbs_slice = (0, nocc, nocc, nmo)
    ao_idx = numpy.asarray([0, 2, 5], dtype=numpy.int64)
    rows, cols = numpy.tril_indices(local_nao)
    pair_idx = (
        ao_idx[rows] * (ao_idx[rows] + 1) // 2 + ao_idx[cols]
    )

    with _cderi_vjp.nr_e2_cderi_bar_packed_disk(
            np.asarray(mo_coeff), np.asarray(ybar), orbs_slice,
            directory=str(tmp_path), aux_blksize=2) as dataset:
        global_pair_count = global_nao * (global_nao + 1) // 2
        for p0 in range(0, global_pair_count, 4):
            p1 = min(p0 + 4, global_pair_count)
            reference = lno_base._nr_e2_local_cderi_bar_block(
                np.asarray(mo_coeff), np.asarray(ybar), orbs_slice,
                pair_idx, p0, p1,
            )
            result = lno_base._nr_e2_local_cderi_bar_disk_block(
                dataset, pair_idx, p0, p1
            )
            assert numpy.allclose(result, reference, atol=1e-10, rtol=1e-10)


def test_cholesky_eri_mo_deriv_vjp_matches_cderi_bar_path(
        tmp_path):
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


def test_local_disk_cderi_bar_cholesky_vjp_matches_pairwise_path(tmp_path):
    from pyscfad.lno import lno_base

    rng = numpy.random.default_rng(151)
    mol = gto.Mole(
        atom='H 0 0 0; H 0 0 1; H 0 1 0; H 1 0 0',
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
    cderi_file = tmp_path / 'local-cderi.h5'
    with h5py.File(cderi_file, 'w') as h5f:
        h5f.create_dataset('j3c', data=cderi)

    ao_idx = numpy.asarray([0, 2, 3], dtype=numpy.int64)
    local_nao = ao_idx.size
    rows, cols = numpy.tril_indices(local_nao)
    pair_idx = (
        ao_idx[rows] * (ao_idx[rows] + 1) // 2 + ao_idx[cols]
    )
    nocc, nvir = 1, 2
    mo_coeff = rng.normal(size=(local_nao, nocc + nvir))
    orbs_slice = (0, nocc, nocc, nocc + nvir)
    ybar = rng.normal(size=(auxmol.nao, nocc * nvir))

    def pairwise_block(p0, p1):
        return lno_base._nr_e2_local_cderi_bar_block(
            np.asarray(mo_coeff), np.asarray(ybar), orbs_slice,
            pair_idx, p0, p1,
        )

    mol_ref, aux_ref = _cderi_vjp.cholesky_eri_vjp_from_cderi_block_fn(
        mol, auxmol, str(cderi_file), pairwise_block, 1024,
        int3c=mol._add_suffix('int3c2e'),
        int2c=mol._add_suffix('int2c2e'),
        aosym='s2ij',
    )

    with _cderi_vjp.nr_e2_cderi_bar_packed_disk(
            np.asarray(mo_coeff), np.asarray(ybar), orbs_slice,
            directory=str(tmp_path), aux_blksize=3) as dataset:
        mol_test, aux_test = _cderi_vjp.cholesky_eri_vjp_from_cderi_block_fn(
            mol,
            auxmol,
            str(cderi_file),
            lambda p0, p1: lno_base._nr_e2_local_cderi_bar_disk_block(
                dataset, pair_idx, p0, p1
            ),
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
