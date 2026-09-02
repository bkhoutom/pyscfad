from pathlib import Path
from types import SimpleNamespace

import h5py
import jax
import numpy
import pytest
import scipy.linalg
from pyscf import lib as pyscf_lib
from pyscf.mp import dfmp2

from pyscfad import df, gto
from pyscfad import numpy as np
from pyscfad.lno import lno_base


def _water_mol():
    mol = gto.Mole(
        atom="O 0 0 0; H 0 0 1; H 0 1 0",
        basis="sto-3g",
        verbose=0,
        max_memory=200,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def _df_holder(mol):
    with_df = df.DF(mol, auxbasis="weigend", incore=True)
    with_df.max_memory = mol.max_memory
    return SimpleNamespace(mol=mol, with_df=with_df)


def _local_coeff(mol, atmlst):
    ao_idx = lno_base.dlno_util.ao_index_by_atom(mol, atmlst)
    rng = numpy.random.default_rng(14)
    return np.asarray(rng.normal(size=(ao_idx.size, 5)))


def _lov_norm(mol, coeff, atmlst, integral_direct):
    lov = lno_base.get_local_Lov(
        _df_holder(mol),
        coeff,
        2,
        atmlst,
        integral_direct=integral_direct,
    )
    return np.einsum("Lia,Lia->", lov, lov)


def _direct_local_problem(nocc=2):
    mol = _water_mol()
    atmlst = numpy.asarray([0, 1], dtype=numpy.int32)
    coeff = _local_coeff(mol, atmlst)
    fake_mol = lno_base.make_local_mol(mol, atmlst)
    auxmol = lno_base.df_addons.make_auxmol(
        fake_mol, _df_holder(mol).with_df.auxbasis
    )
    orbs_slice = (0, nocc, nocc, coeff.shape[1])
    lov = lno_base._local_direct_nr_e2_impl(
        fake_mol, auxmol, coeff, mol.max_memory, orbs_slice
    )
    return mol, fake_mol, auxmol, coeff, orbs_slice, lov


def _assert_tree_allclose(actual, expected, **kwargs):
    actual_leaves = jax.tree_util.tree_leaves(actual)
    expected_leaves = jax.tree_util.tree_leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
        numpy.testing.assert_allclose(
            numpy.asarray(actual_leaf), numpy.asarray(expected_leaf), **kwargs
        )


def _assert_disjoint_aux_coverage(ranges, naux):
    assert ranges
    assert ranges[0][0] == 0
    assert ranges[-1][1] == naux
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    assert all(p0 < p1 for p0, p1 in ranges)


def test_integral_direct_local_lov_matches_cderi_without_building_local_df(
    monkeypatch,
):
    mol = _water_mol()
    atmlst = numpy.asarray([0, 1], dtype=numpy.int32)
    coeff = _local_coeff(mol, atmlst)
    reference = lno_base.get_local_Lov(
        _df_holder(mol), coeff, 2, atmlst
    )

    def forbidden_local_df(*args, **kwargs):
        raise AssertionError("integral-direct Lov must not build an AO-pair CDERI")

    monkeypatch.setattr(lno_base, "get_local_df", forbidden_local_df)
    direct = lno_base.get_local_Lov(
        _df_holder(mol), coeff, 2, atmlst, integral_direct=True
    )

    numpy.testing.assert_allclose(
        numpy.asarray(direct), numpy.asarray(reference), atol=1e-11, rtol=1e-11
    )


def test_integral_direct_local_lov_preserves_coeff_and_coordinate_vjps():
    mol = _water_mol()
    atmlst = numpy.asarray([0, 1], dtype=numpy.int32)
    coeff = _local_coeff(mol, atmlst)

    value_ref, coeff_bar_ref = jax.value_and_grad(
        lambda coeff_: _lov_norm(mol, coeff_, atmlst, False)
    )(coeff)
    value_direct, coeff_bar_direct = jax.value_and_grad(
        lambda coeff_: _lov_norm(mol, coeff_, atmlst, True)
    )(coeff)
    numpy.testing.assert_allclose(value_direct, value_ref, atol=1e-11, rtol=1e-11)
    numpy.testing.assert_allclose(
        numpy.asarray(coeff_bar_direct),
        numpy.asarray(coeff_bar_ref),
        atol=1e-10,
        rtol=1e-10,
    )

    _, mol_bar_ref = jax.value_and_grad(
        lambda mol_: _lov_norm(mol_, coeff, atmlst, False)
    )(mol)
    _, mol_bar_direct = jax.value_and_grad(
        lambda mol_: _lov_norm(mol_, coeff, atmlst, True)
    )(mol)
    numpy.testing.assert_allclose(
        numpy.asarray(mol_bar_direct.coords),
        numpy.asarray(mol_bar_ref.coords),
        atol=1e-9,
        rtol=1e-9,
    )


@pytest.mark.parametrize(
    ("configured_mb", "expected_mb"),
    ((None, 256.0), (7.0, 7.0)),
)
def test_integral_direct_coordinate_vjp_uses_forward_block_budget(
    monkeypatch, configured_mb, expected_mb
):
    mol = _water_mol()
    atmlst = numpy.asarray([0, 1], dtype=numpy.int32)
    coeff = _local_coeff(mol, atmlst)
    if configured_mb is None:
        monkeypatch.delenv(
            "PYSCFAD_LNO_LOCAL_DIRECT_INT3C_BLOCK_MB", raising=False
        )
    else:
        monkeypatch.setenv(
            "PYSCFAD_LNO_LOCAL_DIRECT_INT3C_BLOCK_MB",
            str(configured_mb),
        )

    real_vjp = lno_base._cderi_vjp._int3c_mo_deriv_coords_vjp
    seen = []

    def checked_vjp(*args, **kwargs):
        seen.append(kwargs.get("block_memory_mb"))
        return real_vjp(*args, **kwargs)

    monkeypatch.setattr(
        lno_base._cderi_vjp,
        "_int3c_mo_deriv_coords_vjp",
        checked_vjp,
    )
    jax.value_and_grad(
        lambda mol_: _lov_norm(mol_, coeff, atmlst, True)
    )(mol)
    assert seen == [expected_mb]


def test_integral_direct_local_lov_general_cotangent_matches_coordinate_fd():
    """Exercise the nonsymmetric Lov cotangent generated by fragment MP2."""
    mol = _water_mol()
    atmlst = numpy.asarray([0, 1], dtype=numpy.int32)
    coeff = _local_coeff(mol, atmlst)
    reference = lno_base.get_local_Lov(
        _df_holder(mol), coeff, 2, atmlst, integral_direct=True
    )
    rng = numpy.random.default_rng(91)
    cotangent = np.asarray(rng.normal(size=reference.shape))

    def objective(mol_):
        lov = lno_base.get_local_Lov(
            _df_holder(mol_), coeff, 2, atmlst, integral_direct=True
        )
        return np.einsum("Lia,Lia->", cotangent, lov)

    _, mol_bar = jax.value_and_grad(objective)(mol)
    analytic = float(numpy.asarray(mol_bar.coords)[1, 2])
    coords = numpy.asarray(mol.atom_coords())
    symbols = [mol.atom_symbol(atom) for atom in range(mol.natm)]

    def displaced_value(delta):
        displaced_coords = coords.copy()
        displaced_coords[1, 2] += delta
        displaced = gto.Mole(
            atom=list(zip(symbols, displaced_coords)),
            unit="Bohr",
            basis="sto-3g",
            verbose=0,
            max_memory=200,
        )
        displaced.build(trace_exp=False, trace_ctr_coeff=False)
        return float(objective(displaced))

    step = 2e-4
    finite_difference = (
        displaced_value(-2 * step)
        - 8 * displaced_value(-step)
        + 8 * displaced_value(step)
        - displaced_value(2 * step)
    ) / (12 * step)
    numpy.testing.assert_allclose(
        analytic, finite_difference, atol=2e-8, rtol=2e-8
    )


def test_build_local_lov_h5_matches_integral_direct_pair_major(
    tmp_path, monkeypatch
):
    mol = _water_mol()
    atmlst = numpy.asarray([0, 1], dtype=numpy.int32)
    coeff = _local_coeff(mol, atmlst)
    nocc = 2
    reference = lno_base.get_local_Lov(
        _df_holder(mol), coeff, nocc, atmlst, integral_direct=True
    )
    path = tmp_path / "local-lov.h5"
    monkeypatch.setattr(lno_base.resource_profile, "enabled", lambda: True)

    info = lno_base.build_local_Lov_h5(
        _df_holder(mol), coeff, nocc, atmlst, path
    )

    reference_array = numpy.asarray(reference)
    naux = reference_array.shape[0]
    nvir = reference_array.shape[2]
    assert (info.path, info.naux, info.nocc, info.nvir, info.dtype) == (
        str(path), naux, nocc, nvir, str(reference_array.dtype)
    )
    assert info.hdf5_bytes_written == reference_array.nbytes
    assert info.hdf5_write_seconds >= 0.0
    assert info.lov_disk_mib == reference_array.nbytes / 1024.0**2
    with h5py.File(path, "r") as h5file:
        assert set(h5file) == {"lov"}
        assert h5file["lov"].shape == (nocc * nvir, naux)
        assert h5file["lov"].dtype == numpy.dtype(reference_array.dtype)
        assert h5file["lov"].chunks is None
        assert h5file["lov"].compression is None
        numpy.testing.assert_allclose(
            h5file["lov"][:],
            reference_array.reshape(naux, -1).T,
            rtol=2e-11,
            atol=2e-12,
        )


def test_build_local_lov_h5_skips_diagnostics_when_profile_disabled(
    tmp_path, monkeypatch
):
    mol = _water_mol()
    atmlst = numpy.asarray([0, 1], dtype=numpy.int32)
    coeff = _local_coeff(mol, atmlst)
    monkeypatch.setattr(lno_base.resource_profile, "enabled", lambda: False)
    class FailTime:
        @staticmethod
        def perf_counter():
            raise AssertionError("disabled profiling consulted the clock")

    monkeypatch.setattr(lno_base, "time", FailTime)

    info = lno_base.build_local_Lov_h5(
        _df_holder(mol), coeff, 2, atmlst, tmp_path / "disabled.h5"
    )

    assert info.hdf5_bytes_written == 0
    assert info.hdf5_write_seconds == 0.0
    assert info.lov_disk_mib == 0.0


def test_build_local_lov_h5_supports_zero_occupied_orbitals(tmp_path):
    mol = _water_mol()
    atmlst = numpy.asarray([0, 1], dtype=numpy.int32)
    coeff = _local_coeff(mol, atmlst)
    reference = lno_base.get_local_Lov(
        _df_holder(mol), coeff, 0, atmlst, integral_direct=True
    )
    path = tmp_path / "zero-occ.h5"

    info = lno_base.build_local_Lov_h5(
        _df_holder(mol), coeff, 0, atmlst, path
    )

    assert info.naux == reference.shape[0]
    assert info.nocc == 0
    assert info.nvir == coeff.shape[1]
    assert info.dtype == str(reference.dtype)
    with h5py.File(path, "r") as h5file:
        lov = h5file["lov"]
        assert lov.shape == (0, reference.shape[0])
        assert lov.dtype == numpy.dtype(reference.dtype)
        assert lov.chunks is None
        assert lov.compression is None


def test_build_local_lov_h5_supports_zero_virtual_orbitals(tmp_path):
    mol = _water_mol()
    atmlst = numpy.asarray([0, 1], dtype=numpy.int32)
    coeff = _local_coeff(mol, atmlst)
    nocc = coeff.shape[1]
    reference = lno_base.get_local_Lov(
        _df_holder(mol), coeff, nocc, atmlst, integral_direct=True
    )
    path = tmp_path / "zero-vir.h5"

    info = lno_base.build_local_Lov_h5(
        _df_holder(mol), coeff, nocc, atmlst, path
    )

    assert info.naux == reference.shape[0]
    assert info.nocc == nocc
    assert info.nvir == 0
    assert info.dtype == str(reference.dtype)
    with h5py.File(path, "r") as h5file:
        lov = h5file["lov"]
        assert lov.shape == (0, reference.shape[0])
        assert lov.dtype == numpy.dtype(reference.dtype)
        assert lov.chunks is None
        assert lov.compression is None


def test_build_local_lov_h5_rejects_insufficient_scratch_space(
    monkeypatch, tmp_path
):
    mol = _water_mol()
    atmlst = numpy.asarray([0, 1], dtype=numpy.int32)
    coeff = _local_coeff(mol, atmlst)
    nocc = 2
    nvir = coeff.shape[1] - nocc
    fake_mol = lno_base.make_local_mol(mol, atmlst)
    auxmol = lno_base.df_addons.make_auxmol(
        fake_mol, _df_holder(mol).with_df.auxbasis
    )
    required = int(3.25 * auxmol.nao * nocc * nvir * 8) + 1024**3
    path = tmp_path / "too-large.h5"
    monkeypatch.setattr(
        lno_base.shutil,
        "disk_usage",
        lambda unused: SimpleNamespace(free=required - 1),
    )

    with pytest.raises(OSError) as excinfo:
        lno_base.build_local_Lov_h5(
            _df_holder(mol), coeff, nocc, atmlst, path
        )

    message = str(excinfo.value)
    assert str(path) in message
    assert f"free bytes={required - 1}" in message
    assert f"required bytes={required}" in message
    assert not path.exists()


def test_build_local_lov_h5_uses_configured_tmpdir_and_removes_raw_store(
    monkeypatch, tmp_path
):
    mol = _water_mol()
    atmlst = numpy.asarray([0, 1], dtype=numpy.int32)
    coeff = _local_coeff(mol, atmlst)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    raw_paths = []
    real_h5tmp = pyscf_lib.H5TmpFile

    def tracked_h5tmp(*args, **kwargs):
        raw_file = real_h5tmp(*args, **kwargs)
        raw_paths.append(Path(raw_file.filename))
        return raw_file

    monkeypatch.setattr(pyscf_lib.param, "TMPDIR", str(scratch))
    monkeypatch.setattr(pyscf_lib, "H5TmpFile", tracked_h5tmp)

    lno_base.build_local_Lov_h5(
        _df_holder(mol), coeff, 2, atmlst, tmp_path / "local-lov.h5"
    )

    assert raw_paths
    assert all(raw_path.parent == scratch for raw_path in raw_paths)
    assert all(not raw_path.exists() for raw_path in raw_paths)


def test_build_local_lov_h5_removes_partial_target_after_failure(
    monkeypatch, tmp_path
):
    mol = _water_mol()
    atmlst = numpy.asarray([0, 1], dtype=numpy.int32)
    coeff = _local_coeff(mol, atmlst)
    path = tmp_path / "failed.h5"

    def failing_producer(with_df, occ_coeff, vir_coeff, max_memory,
                         h5obj=None, log=None):
        del with_df, max_memory, log
        h5obj.create_dataset(
            "ovL",
            (occ_coeff.shape[1] * vir_coeff.shape[1], 4),
            dtype=numpy.float64,
            chunks=(1, 4),
        )
        raise RuntimeError("producer failed")

    monkeypatch.setattr(dfmp2, "_init_mp_df_eris_direct", failing_producer)

    with pytest.raises(RuntimeError, match="producer failed"):
        lno_base.build_local_Lov_h5(
            _df_holder(mol), coeff, 2, atmlst, path
        )

    assert not path.exists()
    assert not list(tmp_path.glob(".*failed.h5.*"))


def test_build_local_lov_h5_never_converts_complete_lov_dataset(
    monkeypatch, tmp_path
):
    mol = _water_mol()
    atmlst = numpy.asarray([0, 1], dtype=numpy.int32)
    coeff = _local_coeff(mol, atmlst)
    path = tmp_path / "local-lov.h5"
    original_array = h5py.Dataset.__array__
    original_getitem = h5py.Dataset.__getitem__
    original_setitem = h5py.Dataset.__setitem__
    lov_reads = []
    lov_write_sizes = []

    def forbidden_array(dataset, *args, **kwargs):
        if dataset.name == "/lov":
            raise AssertionError("the complete /lov dataset was converted")
        return original_array(dataset, *args, **kwargs)

    def tracked_getitem(dataset, key):
        if dataset.name == "/lov":
            lov_reads.append(key)
        return original_getitem(dataset, key)

    def tracked_setitem(dataset, key, value):
        if dataset.name == "/lov":
            row_key = key[0] if isinstance(key, tuple) else key
            selected_rows = numpy.arange(dataset.shape[0])[row_key]
            lov_write_sizes.append(selected_rows.size * dataset.shape[1])
        return original_setitem(dataset, key, value)

    monkeypatch.setattr(h5py.Dataset, "__array__", forbidden_array)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", tracked_getitem)
    monkeypatch.setattr(h5py.Dataset, "__setitem__", tracked_setitem)

    info = lno_base.build_local_Lov_h5(
        _df_holder(mol), coeff, 2, atmlst, path
    )

    assert lov_reads == []
    assert lov_write_sizes
    assert max(lov_write_sizes) < 2 * (coeff.shape[1] - 2) * info.naux


def test_local_lov_h5_pair_tile_size_accounts_for_all_reverse_tiles():
    naux = 37
    npair = 101
    target_mb = 0.01
    target_bytes = target_mb * 1024.0**2
    expected = int(target_bytes // (3 * naux * numpy.dtype('float64').itemsize))

    assert lno_base._local_lov_h5_pair_tile_size(
        naux, npair, numpy.float64, target_mb=target_mb
    ) == max(1, min(npair, expected))
    assert lno_base._local_lov_h5_pair_tile_size(
        naux, 0, numpy.float64, target_mb=target_mb
    ) == 1

    for args in (
        (-1, npair, numpy.float64, target_mb),
        (naux, -1, numpy.float64, target_mb),
        (naux, npair, numpy.float64, 0.0),
        (naux, npair, numpy.dtype('O'), target_mb),
    ):
        with pytest.raises(ValueError):
            lno_base._local_lov_h5_pair_tile_size(
                args[0], args[1], args[2], target_mb=args[3]
            )


def test_local_lov_h5_z_chunks_are_two_dimensional_and_near_eight_mib():
    chunks = lno_base._local_lov_h5_z_chunks(
        naux=10_000,
        npair=100_000,
        dtype=numpy.float64,
        pair_tile_size=1_118,
    )
    chunk_bytes = numpy.prod(chunks) * numpy.dtype(numpy.float64).itemsize

    assert len(chunks) == 2
    assert chunks[0] < 10_000
    assert chunks[1] < 100_000
    assert 4 * 1024**2 <= chunk_bytes <= 16 * 1024**2


def test_local_lov_h5_z_aux_reads_are_memory_bounded_and_strictly_partial():
    small_rows = lno_base._local_lov_h5_z_aux_read_rows(
        naux=71, npair=6, dtype=numpy.float64
    )
    large_rows = lno_base._local_lov_h5_z_aux_read_rows(
        naux=1_000, npair=100_000, dtype=numpy.float64
    )

    assert small_rows == 70
    assert 1 <= large_rows < 1_000
    assert large_rows * 100_000 * 8 <= 64 * 1024**2


def test_local_lov_h5_reverse_dtype_includes_every_tile_participant():
    assert lno_base._local_lov_h5_reverse_dtype(
        numpy.float64, numpy.complex128, numpy.float32
    ) == numpy.dtype(numpy.complex128)
    assert lno_base._local_lov_h5_reverse_dtype(
        numpy.float64, numpy.float32, numpy.float32
    ) == numpy.dtype(numpy.float64)


def test_local_direct_mo_coeff_vjp_accepts_one_read_per_raw_block(monkeypatch):
    _, fake_mol, auxmol, coeff, orbs_slice, lov = _direct_local_problem()
    z = numpy.random.default_rng(183).normal(size=numpy.asarray(lov).shape)
    real_blocks = lno_base._local_direct_raw_int3c_blocks
    block_ranges = []

    def tracked_blocks(*args, **kwargs):
        for p0, p1, raw_ints in real_blocks(*args, **kwargs):
            block_ranges.append((p0, p1))
            yield p0, p1, raw_ints

    monkeypatch.setattr(
        lno_base, '_local_direct_raw_int3c_blocks', tracked_blocks
    )
    reference = lno_base._local_direct_mo_coeff_vjp(
        fake_mol, auxmol, coeff, z, orbs_slice
    )
    block_ranges.clear()
    calls = []

    def read_z_aux_block(p0, p1):
        calls.append((p0, p1))
        return z[p0:p1, :]

    result = lno_base._local_direct_mo_coeff_vjp(
        fake_mol, auxmol, coeff, read_z_aux_block, orbs_slice
    )

    assert calls == block_ranges
    numpy.testing.assert_allclose(
        numpy.asarray(result), numpy.asarray(reference),
        atol=2e-11, rtol=2e-11,
    )


def test_local_direct_mo_coeff_vjp_rejects_transposed_reader_block():
    _, fake_mol, auxmol, coeff, orbs_slice, lov = _direct_local_problem()
    z = numpy.random.default_rng(182).normal(size=numpy.asarray(lov).shape)

    def transposed_z_aux_block(p0, p1):
        return z[p0:p1, :].T

    with pytest.raises(ValueError, match='z auxiliary block.*shape'):
        lno_base._local_direct_mo_coeff_vjp(
            fake_mol, auxmol, coeff, transposed_z_aux_block, orbs_slice
        )


def test_local_direct_mo_coeff_vjp_oversized_shell_fallback_is_disjoint(
    monkeypatch,
):
    _, fake_mol, auxmol, coeff, orbs_slice, lov = _direct_local_problem()
    z = numpy.random.default_rng(181).normal(size=numpy.asarray(lov).shape)
    reference = lno_base._local_direct_mo_coeff_vjp(
        fake_mol, auxmol, coeff, z, orbs_slice
    )
    real_blocks = lno_base._local_direct_raw_int3c_blocks
    real_int3c_cross = lno_base._int3c_cross_opt.int3c_cross
    logical_ranges = []
    integral_calls = []

    def tracked_blocks(*args, **kwargs):
        for p0, p1, raw_ints in real_blocks(*args, **kwargs):
            logical_ranges.append((p0, p1))
            yield p0, p1, raw_ints

    def tracked_int3c_cross(*args, **kwargs):
        integral_calls.append(kwargs['shls_slice'])
        return real_int3c_cross(*args, **kwargs)

    monkeypatch.setattr(
        lno_base, '_local_direct_raw_int3c_blocks', tracked_blocks
    )
    monkeypatch.setattr(
        lno_base._int3c_cross_opt, 'int3c_cross', tracked_int3c_cross
    )
    read_ranges = []

    def read_z_aux_block(p0, p1):
        read_ranges.append((p0, p1))
        return z[p0:p1, :]

    result = lno_base._local_direct_mo_coeff_vjp(
        fake_mol,
        auxmol,
        coeff,
        read_z_aux_block,
        orbs_slice,
        z_aux_block_max_rows=1,
    )

    assert read_ranges == logical_ranges
    _assert_disjoint_aux_coverage(logical_ranges, auxmol.nao)
    assert all(p1 - p0 == 1 for p0, p1 in logical_ranges)
    assert len(integral_calls) < len(logical_ranges)
    numpy.testing.assert_allclose(
        numpy.asarray(result), numpy.asarray(reference),
        atol=2e-10, rtol=2e-10,
    )


def test_local_direct_nr_e2_h5_bwd_matches_full_general_cotangent(
    tmp_path, monkeypatch
):
    mol, fake_mol, auxmol, coeff, orbs_slice, lov = _direct_local_problem()
    lov = numpy.asarray(lov)
    rng = numpy.random.default_rng(184)
    lov_bar = rng.normal(size=lov.shape)
    reference = lno_base._local_direct_nr_e2_bwd(
        mol.max_memory,
        orbs_slice,
        (fake_mol, auxmol, coeff, np.asarray(lov)),
        np.asarray(lov_bar),
    )
    path = tmp_path / 'reverse.h5'
    with h5py.File(path, 'w') as h5file:
        h5file.create_dataset('lov', data=lov.T)
        h5file.create_dataset('lov_bar', data=lov_bar.T)
        h5file.attrs['pyscfad_fragment_index'] = 19
    profile_token = object()
    profile_events = []
    monkeypatch.setattr(
        lno_base.resource_profile, 'start', lambda: profile_token
    )
    monkeypatch.setattr(
        lno_base.resource_profile,
        'finish',
        lambda phase, before, **details: profile_events.append(
            (phase, before, details)
        ),
    )

    result = lno_base._local_direct_nr_e2_h5_bwd(
        fake_mol, auxmol, coeff, orbs_slice, path
    )

    _assert_tree_allclose(result[0], reference[0], atol=2e-9, rtol=2e-9)
    _assert_tree_allclose(result[1], reference[1], atol=2e-9, rtol=2e-9)
    numpy.testing.assert_allclose(
        numpy.asarray(result[2]), numpy.asarray(reference[2]),
        atol=2e-10, rtol=2e-10,
    )
    j2c = numpy.asarray(
        auxmol.intor(auxmol._add_suffix('int2c2e'), hermi=1)
    )
    low = scipy.linalg.cholesky(j2c, lower=True, check_finite=False)
    z_reference = scipy.linalg.solve_triangular(
        low.T, lov_bar, lower=False, check_finite=False
    )
    with h5py.File(path, 'r') as h5file:
        assert h5file['z'].shape == lov.shape
        assert h5file['z'].chunks is not None
        assert len(h5file['z'].chunks) == 2
        numpy.testing.assert_allclose(
            h5file['z'][:], z_reference, atol=2e-11, rtol=2e-11
        )
    assert len(profile_events) == 1
    phase, before, details = profile_events[0]
    assert phase == 'lno.local_direct_nr_e2_h5_bwd'
    assert before is profile_token
    assert details['status'] == 'ok'
    assert details['fragment_index'] == 19
    assert details['lov_h5_path_basename'] == path.name
    assert details['lov_disk_mib'] > 0.0
    assert details['lov_bar_disk_mib'] > 0.0
    assert details['z_disk_mib'] > 0.0
    assert details['hdf5_bytes_read'] > 0
    assert details['hdf5_bytes_written'] > 0
    assert details['hdf5_read_seconds'] >= 0.0
    assert details['hdf5_write_seconds'] >= 0.0


def test_local_direct_nr_e2_h5_bwd_uses_bounded_dataset_slices(
    monkeypatch, tmp_path
):
    _, fake_mol, auxmol, coeff, orbs_slice, lov = _direct_local_problem()
    lov = numpy.asarray(lov)
    lov_bar = numpy.random.default_rng(185).normal(size=lov.shape)
    path = tmp_path / 'bounded-reverse.h5'
    with h5py.File(path, 'w') as h5file:
        h5file.create_dataset('lov', data=lov.T)
        h5file.create_dataset('lov_bar', data=lov_bar.T)

    monkeypatch.setattr(
        lno_base, '_local_lov_h5_pair_tile_size', lambda *args, **kwargs: 2
    )
    original_array = h5py.Dataset.__array__
    original_getitem = h5py.Dataset.__getitem__
    original_int3c_cross = lno_base._int3c_cross_opt.int3c_cross
    reads = {'/lov': [], '/lov_bar': [], '/z': []}
    integral_ranges = []

    def forbidden_array(dataset, *args, **kwargs):
        if dataset.name in reads:
            raise AssertionError(f'complete conversion of {dataset.name}')
        return original_array(dataset, *args, **kwargs)

    def tracked_getitem(dataset, key):
        if dataset.name in reads:
            reads[dataset.name].append(key)
        return original_getitem(dataset, key)

    def tracked_int3c_cross(*args, **kwargs):
        shls_slice = kwargs['shls_slice']
        shl0 = shls_slice[4] - fake_mol.nbas
        shl1 = shls_slice[5] - fake_mol.nbas
        integral_ranges.append(
            (
                kwargs['intor'],
                int(auxmol.ao_loc[shl0]),
                int(auxmol.ao_loc[shl1]),
            )
        )
        return original_int3c_cross(*args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, '__array__', forbidden_array)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', tracked_getitem)
    monkeypatch.setattr(
        lno_base._int3c_cross_opt, 'int3c_cross', tracked_int3c_cross
    )

    lno_base._local_direct_nr_e2_h5_bwd(
        fake_mol, auxmol, coeff, orbs_slice, path
    )

    npair, naux = lov.T.shape
    for name in ('/lov', '/lov_bar'):
        assert len(reads[name]) == (npair + 1) // 2
        assert all(isinstance(key, tuple) and len(key) == 2 for key in reads[name])
        selected = [
            numpy.arange(npair)[key[0]].size * naux for key in reads[name]
        ]
        assert max(selected) <= 2 * naux
        assert max(selected) < npair * naux
    raw_ranges = [
        (p0, p1) for intor, p0, p1 in integral_ranges if 'ip1' not in intor
    ]
    ip1_ranges = [
        (p0, p1) for intor, p0, p1 in integral_ranges if 'ip1' in intor
    ]
    z_ranges = [(key[0].start, key[0].stop) for key in reads['/z']]
    _assert_disjoint_aux_coverage(raw_ranges, naux)
    _assert_disjoint_aux_coverage(ip1_ranges, naux)
    assert z_ranges == raw_ranges + ip1_ranges
    assert all(isinstance(key, tuple) and len(key) == 2 for key in reads['/z'])
    assert all(key[1] == slice(None) for key in reads['/z'])
    assert all(
        numpy.arange(naux)[key[0]].size < naux for key in reads['/z']
    )


@pytest.mark.parametrize(
    ('lov_dtype', 'lov_bar_dtype', 'coeff_dtype'),
    (
        (numpy.complex128, numpy.float64, numpy.float64),
        (numpy.int64, numpy.float64, numpy.float64),
        (numpy.float64, numpy.float32, numpy.float64),
        (numpy.float64, numpy.float64, numpy.float32),
    ),
)
def test_local_direct_nr_e2_h5_bwd_rejects_non_float64_contract(
    tmp_path, lov_dtype, lov_bar_dtype, coeff_dtype
):
    _, fake_mol, auxmol, coeff, orbs_slice, lov = _direct_local_problem()
    lov = numpy.asarray(lov)
    lov_bar = numpy.random.default_rng(188).normal(size=lov.shape)
    path = tmp_path / 'unsupported-dtype.h5'
    with h5py.File(path, 'w') as h5file:
        h5file.create_dataset('lov', data=lov.T.astype(lov_dtype))
        h5file.create_dataset(
            'lov_bar', data=lov_bar.T.astype(lov_bar_dtype)
        )

    with pytest.raises(ValueError, match='real float64'):
        lno_base._local_direct_nr_e2_h5_bwd(
            fake_mol,
            auxmol,
            np.asarray(numpy.asarray(coeff).astype(coeff_dtype)),
            orbs_slice,
            path,
        )


def test_local_direct_nr_e2_h5_bwd_recreates_z_and_cleans_failed_z(
    monkeypatch, tmp_path
):
    _, fake_mol, auxmol, coeff, orbs_slice, lov = _direct_local_problem()
    lov = numpy.asarray(lov)
    path = tmp_path / 'repeat-reverse.h5'
    first_bar = numpy.random.default_rng(186).normal(size=lov.shape)
    second_bar = numpy.random.default_rng(187).normal(size=lov.shape)
    with h5py.File(path, 'w') as h5file:
        h5file.create_dataset('lov', data=lov.T)
        h5file.create_dataset('lov_bar', data=first_bar.T)
        h5file.attrs['pyscfad_fragment_index'] = 29

    lno_base._local_direct_nr_e2_h5_bwd(
        fake_mol, auxmol, coeff, orbs_slice, path
    )
    with h5py.File(path, 'r+') as h5file:
        first_z = h5file['z'][:]
        h5file['lov_bar'][:] = second_bar.T
    lno_base._local_direct_nr_e2_h5_bwd(
        fake_mol, auxmol, coeff, orbs_slice, path
    )
    with h5py.File(path, 'r') as h5file:
        assert not numpy.allclose(h5file['z'][:], first_z)

    def fail_after_z(*unused_args, **unused_kwargs):
        raise RuntimeError('downstream failure')

    monkeypatch.setattr(
        lno_base, '_local_direct_mo_coeff_vjp', fail_after_z
    )
    token = object()
    events = []
    monkeypatch.setattr(lno_base.resource_profile, 'start', lambda: token)
    monkeypatch.setattr(
        lno_base.resource_profile,
        'finish',
        lambda phase, before, **details: events.append(
            (phase, before, details)
        ),
    )
    with pytest.raises(RuntimeError, match='downstream failure'):
        lno_base._local_direct_nr_e2_h5_bwd(
            fake_mol, auxmol, coeff, orbs_slice, path
        )
    with h5py.File(path, 'r') as h5file:
        assert set(h5file) == {'lov', 'lov_bar'}
    assert len(events) == 1
    phase, before, details = events[0]
    assert phase == 'lno.local_direct_nr_e2_h5_bwd'
    assert before is token
    assert details['status'] == 'failed'
    assert details['fragment_index'] == 29
    assert details['hdf5_bytes_read'] > 0


def test_local_direct_nr_e2_h5_bwd_supports_empty_pair_dimension(tmp_path):
    _, fake_mol, auxmol, coeff, _, _ = _direct_local_problem()
    orbs_slice = (0, 0, 0, coeff.shape[1])
    path = tmp_path / 'empty-reverse.h5'
    with h5py.File(path, 'w') as h5file:
        h5file.create_dataset('lov', shape=(0, auxmol.nao), dtype=numpy.float64)
        h5file.create_dataset(
            'lov_bar', shape=(0, auxmol.nao), dtype=numpy.float64
        )

    mol_bar, auxmol_bar, coeff_bar = lno_base._local_direct_nr_e2_h5_bwd(
        fake_mol, auxmol, coeff, orbs_slice, path
    )

    assert numpy.all(numpy.asarray(mol_bar.coords) == 0)
    assert numpy.all(numpy.asarray(auxmol_bar.coords) == 0)
    assert numpy.all(numpy.asarray(coeff_bar) == 0)
    with h5py.File(path, 'r') as h5file:
        assert h5file['z'].shape == (auxmol.nao, 0)
