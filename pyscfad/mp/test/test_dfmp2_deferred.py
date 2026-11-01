"""Canonical MP2 can defer source geometry while returning orbital response."""

from contextlib import ExitStack

import h5py
import jax
import numpy
import pytest
from pyscf import lib

from pyscfad import config_update, df, gto
from pyscfad import numpy as np
from pyscfad.ao2mo import _ao2mo
from pyscfad.df import _cderi_vjp
from pyscfad.mp import dfmp2


@pytest.fixture
def case(tmp_path, monkeypatch):
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    monkeypatch.setattr(lib.param, 'TMPDIR', str(scratch))
    monkeypatch.setenv('PYSCFAD_FACTOR_PANEL_MEMORY_MB', '16')
    monkeypatch.setenv('PYSCFAD_SOLVE_MEMORY_MB', '.001')
    mol = gto.Mole(atom='O .1 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    obj = df.DF(mol, auxbasis='weigend', incore=False)
    obj._cderi_to_save = str(tmp_path / 'cderi.h5')
    with config_update('pyscfad_moleintor_opt', True):
        obj.build()
    rng = numpy.random.default_rng(112092)
    coeff = np.asarray(rng.normal(size=(mol.nao, 6)) * .2)
    return obj, coeff, rng, scratch


def _dense_transform(mol, auxmol, coeff, orbs):
    from pyscfad.df import incore
    fitted = incore.cholesky_eri(mol, auxmol=auxmol, aosym='s2ij')
    return _ao2mo.nr_e2(fitted, coeff, orbs, aosym='s2')


class _ColumnReader:
    """Guard a real array/dataset against full materialization or row reads."""
    def __init__(self, value):
        self.value = value
        self.shape, self.dtype = value.shape, value.dtype
        self.widths = []

    def __array__(self, *args, **kwargs):
        raise AssertionError('raw MO seed was fully materialized')

    def __getitem__(self, key):
        rows, columns = key
        assert rows == slice(None)
        assert isinstance(columns, slice)
        self.widths.append(columns.stop - columns.start)
        return self.value[key]


@pytest.mark.parametrize('orbs', [(0, 2, 2, 5), (1, 4, 4, 6)])
@pytest.mark.parametrize('disk_raw', [False, True])
@pytest.mark.parametrize('disk_z', [False, True])
def test_prepared_mo_geometry_matches_dense_with_bounded_owned_panels(
        case, tmp_path, monkeypatch, orbs, disk_raw, disk_z):
    """Wrong packing, metric factors, full reads, or borrowed raw seeds fail."""
    obj, coeff, rng, scratch = case
    prepare = getattr(_cderi_vjp, '_prepare_outcore_mo_geometry', None)
    assert callable(prepare), 'MP2 source geometry must be separately preparable'
    with config_update('pyscfad_moleintor_opt', True):
        lov, reference = jax.vjp(lambda m, a: _dense_transform(m, a, coeff, orbs),
                                 obj.mol, obj.auxmol)
    seeds = [rng.normal(size=lov.shape) * scale for scale in (1., -.37, 0.)]
    initial_files = set(scratch.iterdir())
    coordinate_calls = []
    original = _cderi_vjp._int3c_mo_deriv_coords_vjp_from_z_reader

    def observed(*args, **kwargs):
        coordinate_calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_cderi_vjp, '_int3c_mo_deriv_coords_vjp_from_z_reader', observed)
    for seed in seeds:
        expected = reference(seed)
        with ExitStack() as raw:
            if disk_raw:
                handle = raw.enter_context(h5py.File(tmp_path / 'raw.h5', 'w'))
                raw_lov = handle.create_dataset('lov', data=numpy.asarray(lov))
                raw_bar = handle.create_dataset('bar', data=seed)
            else:
                raw_lov, raw_bar = numpy.array(lov), seed.copy()
            readers = _ColumnReader(raw_bar), _ColumnReader(raw_lov)
            before = len(coordinate_calls)
            contribution = prepare(obj, coeff, readers[0], orbs, readers[1], force_disk=disk_z)
            assert len(coordinate_calls) == before
            assert max(readers[0].widths + readers[1].widths) == 1
            assert contribution.mo_panel_columns == lov.shape[1]
            assert contribution.retained_panel_bytes == (0 if disk_z else lov.size * 8)
            assert contribution.disk_panels == disk_z
        # Raw files are closed before either callback is used.
        try:
            shape = (3, obj.mol.nao * (obj.mol.nao + 1) // 2)
            packed = numpy.empty(shape)
            contribution.fill_packed_density_aux_block(0, 3, packed)
            physical = contribution.read_density_aux_block(0, 3)
            expected_packed = lib.pack_tril(physical)
            diagonal = numpy.arange(obj.mol.nao)
            expected_packed[:, diagonal * (diagonal + 3) // 2] *= .5
            numpy.testing.assert_allclose(packed, expected_packed, atol=2e-11, rtol=2e-11)
            for _ in range(2):
                actual = contribution.contract()
                numpy.testing.assert_allclose(actual.mol.coords, expected[0].coords,
                                              atol=2e-9, rtol=2e-10)
                numpy.testing.assert_allclose(actual.auxmol.coords, expected[1].coords,
                                              atol=2e-9, rtol=2e-10)
        finally:
            contribution.close()
        contribution.close()
        assert set(scratch.iterdir()) <= initial_files


@pytest.mark.parametrize('active', ['mol', 'auxmol', 'coeff', 'all'])
def test_deferred_transform_keeps_immediate_coeff_bar_and_activity(case, monkeypatch, active):
    """An inactive geometry input cannot acquire a deferred source gradient."""
    from pyscfad.df import _ooc_cotangent
    obj, coeff, rng, _ = case
    orbs = (0, 2, 2, 5)
    registrations = []

    def accept(source_token, source, mol, auxmol, mo, slices, saved, seed, *, geometry_activity):
        assert source_token == obj._fast_exchange_cache_token
        assert source == obj._get_cderi_source()
        assert slices == orbs
        registrations.append(geometry_activity)
        return True

    monkeypatch.setattr(_ooc_cotangent, 'defer_mo_geometry', accept, raising=False)
    primals = (obj.mol, obj.auxmol, coeff)
    indices = {'mol': (0,), 'auxmol': (1,), 'coeff': (2,), 'all': (0, 1, 2)}[active]

    def call(values, dense):
        current = list(primals)
        for index, value in zip(indices, values):
            current[index] = value
        if dense:
            return _dense_transform(*current, orbs)
        return dfmp2._outcore_nr_e2(*current, obj._get_cderi_source(), 256, orbs, 's2',
                                   source_token=obj._fast_exchange_cache_token)

    selected = tuple(primals[i] for i in indices)
    with config_update('pyscfad_moleintor_opt', True):
        output, ref = jax.vjp(lambda *values: call(values, True), *selected)
    seed = np.asarray(rng.normal(size=output.shape))
    _, pullback = jax.vjp(lambda *values: call(values, False), *selected)
    actual, expected = pullback(seed), ref(seed)
    for index, got, wanted in zip(indices, actual, expected):
        if index == 2:
            numpy.testing.assert_allclose(got, wanted, atol=2e-10, rtol=2e-10)
        else:
            numpy.testing.assert_array_equal(got.coords, numpy.zeros_like(wanted.coords))
    assert registrations == ([] if active == 'coeff' else
                              [(0 in indices, 1 in indices)])


def test_failed_mo_preparation_closes_z_scratch(case, monkeypatch):
    obj, coeff, rng, scratch = case
    prepare = getattr(_cderi_vjp, '_prepare_outcore_mo_geometry', None)
    assert callable(prepare)
    orbs = (0, 2, 2, 5)
    shape = (obj.get_naoaux(), 6)
    seed, lov = rng.normal(size=shape), rng.normal(size=shape)
    initial_files = set(scratch.iterdir())

    def failed_solve(*args, **kwargs):
        raise RuntimeError('injected MO solve failure')

    monkeypatch.setattr(_cderi_vjp.scipy.linalg, 'solve_triangular', failed_solve)
    with pytest.raises(RuntimeError, match='injected MO solve failure'):
        prepare(obj, coeff, seed, orbs, lov, force_disk=True)
    assert set(scratch.iterdir()) <= initial_files


@pytest.mark.parametrize('orbs', [(0, 0, 1, 4), (1, 4, 4, 4)])
def test_empty_mo_slice_has_zero_geometry_readers(case, orbs):
    obj, coeff, _, _ = case
    zero = numpy.zeros((obj.get_naoaux(), 0))
    contribution = _cderi_vjp._prepare_outcore_mo_geometry(
        obj, coeff, zero, orbs, zero, force_disk=True)
    try:
        assert contribution.mo_panel_columns == contribution.retained_panel_bytes == 0
        numpy.testing.assert_array_equal(contribution.lowbar, 0.)
        numpy.testing.assert_array_equal(contribution.read_density_aux_block(0, 2),
                                          numpy.zeros((2, obj.mol.nao, obj.mol.nao)))
        packed = numpy.full((2, obj.mol.nao * (obj.mol.nao + 1) // 2), numpy.nan)
        contribution.fill_packed_density_aux_block(0, 2, packed)
        numpy.testing.assert_array_equal(packed, numpy.zeros_like(packed))
    finally:
        contribution.close()
