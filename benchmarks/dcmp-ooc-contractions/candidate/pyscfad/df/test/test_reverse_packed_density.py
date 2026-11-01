"""Packed coordinate adjoints retain both AO legs and diagonal weights."""

import jax
import numpy
import pytest

from pyscfad import config_update, df, gto
from pyscfad import numpy as np
from pyscfad.df import _cderi_vjp, _int3c_cross_opt, df_jk, incore


def _molecules():
    mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    auxmol = gto.Mole(atom='H .1 .2 .3; H -.2 .1 1.1',
                      basis='weigend', verbose=0)
    auxmol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol, auxmol


def _explicit_coords(mol, auxmol, density):
    ints = numpy.asarray(_int3c_cross_opt.int3c_cross(
        mol, auxmol, intor='int3c2e_ip1', comp=3, aosym='s1'))
    per_aux_ao = numpy.einsum('puv,xuvp->pux', density, ints)
    mol_coords = numpy.zeros((mol.natm, 3), dtype=density.dtype)
    aux_coords = numpy.zeros((auxmol.natm, 3), dtype=density.dtype)
    for atom, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        mol_coords[atom] = -per_aux_ao[:, p0:p1].sum(axis=(0, 1))
    for atom, (_, _, p0, p1) in enumerate(auxmol.aoslice_by_atom()):
        aux_coords[atom] = per_aux_ao[p0:p1].sum(axis=(0, 1))
    return mol_coords, aux_coords


def _dense_factor_values(mol, auxmol, factors):
    """Independent dense integrals and tensor contractions for all three bars."""
    packed = incore.cholesky_eri(mol, auxmol=auxmol)
    if packed.shape[1] == mol.nao ** 2:
        fitted = packed.reshape(-1, mol.nao, mol.nao)
    else:
        rows, cols = numpy.tril_indices(mol.nao)
        fitted = np.zeros((packed.shape[0], mol.nao, mol.nao))
        fitted = fitted.at[:, rows, cols].set(packed).at[:, cols, rows].set(packed)
    density = factors @ factors.T
    return (np.einsum('Lij,ij,Lpq->pq', fitted, density, fitted),
            np.einsum('Lpi,ij,Ljq->pq', fitted, density, fitted))


@pytest.mark.parametrize('transpose', (False, True))
def test_pack_symmetric_keeps_diagonal_once_and_sums_off_diagonal(transpose):
    """Conjugation, half/double diagonals, or wrong triangle order corrupt bars."""
    matrices = numpy.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                            [[-2., 1., 4.], [3., -5., 8.], [6., 10., -9.]]])
    if transpose:
        matrices = matrices.swapaxes(1, 2)
        assert not matrices.flags.c_contiguous
    storage = numpy.full((4, 6), numpy.nan)
    out = storage[1:3]
    result = _cderi_vjp._pack_tril_symmetric_sum(matrices, out=out)
    assert result is out
    numpy.testing.assert_array_equal(
        result, [[1., 6., 5., 10., 14., 9.],
                 [-2., 4., -5., 10., 18., -9.]])
    assert numpy.isnan(storage[[0, 3]]).all()
    numpy.testing.assert_array_equal(
        _cderi_vjp._pack_tril_symmetric_sum(matrices), result)


@pytest.mark.parametrize('dtype', (numpy.float32, numpy.complex128))
def test_symmetric_pack_rejects_non_double_input(dtype):
    """A ctypes double kernel must never reinterpret unsupported data."""
    with pytest.raises(TypeError, match='float64'):
        _cderi_vjp._pack_tril_symmetric_sum(numpy.ones((2, 3, 3), dtype=dtype))


@pytest.mark.parametrize('bad_output', ('dtype', 'shape', 'strided', 'readonly'))
def test_symmetric_pack_rejects_unsafe_output(bad_output):
    output = numpy.empty((2, 6))
    if bad_output == 'dtype':
        output = output.astype(numpy.float32)
    elif bad_output == 'shape':
        output = numpy.empty((2, 5))
    elif bad_output == 'strided':
        output = numpy.empty((2, 12))[:, ::2]
    else:
        output.flags.writeable = False
    with pytest.raises((TypeError, ValueError)):
        _cderi_vjp._pack_tril_symmetric_sum(numpy.ones((2, 3, 3)), out=output)


@pytest.mark.parametrize('offset', (0, 3))
def test_symmetric_pack_rejects_overlapping_output_without_mutation(offset):
    """Packing into a source view must not overwrite entries still to be read."""
    density = numpy.arange(1., 10.).reshape(1, 3, 3)
    before = density.copy()
    output = density.reshape(-1)[offset:offset + 6].reshape(1, 6)
    with pytest.raises(ValueError, match='overlap'):
        _cderi_vjp._pack_tril_symmetric_sum(density, out=output)
    numpy.testing.assert_array_equal(density, before)


def test_packed_reader_matches_explicit_derivative_with_one_row_subranges(monkeypatch):
    """The callback must bypass dense reads without losing shell/atom offsets."""
    mol, auxmol = _molecules()
    rng = numpy.random.default_rng(710)
    one_leg = rng.normal(size=(auxmol.nao, mol.nao, mol.nao))
    physical = one_leg + one_leg.swapaxes(1, 2)
    expected = _explicit_coords(mol, auxmol, physical)
    rows, cols = numpy.tril_indices(mol.nao)
    diagonal = rows == cols
    reads, native_bytes = [], []

    def forbidden_dense(*args):
        raise AssertionError('packed path expanded a dense physical density')

    def fill_packed(p0, p1, out):
        reads.append((p0, p1))
        assert out.shape == (p1 - p0, len(rows))
        assert out.dtype == numpy.float64 and out.flags.c_contiguous
        out[:] = physical[p0:p1, rows, cols]
        out[:, diagonal] *= .5

    native = _int3c_cross_opt.int3c_cross_bwd

    def bounded_native(*args, **kwargs):
        native_bytes.append(args[-1].nbytes)
        return native(*args, **kwargs)

    monkeypatch.setattr(_int3c_cross_opt, 'int3c_cross_bwd', bounded_native)
    actual = _cderi_vjp._int3c_mo_deriv_coords_vjp_from_z_reader(
        mol, auxmol, None, None, (0, mol.nao, 0, mol.nao),
        z_dtype=numpy.float64, block_memory_mb=1, z_aux_block_max_rows=1,
        read_density_aux_block=forbidden_dense,
        fill_packed_density_aux_block=fill_packed)
    for bar, reference in zip(actual, expected):
        numpy.testing.assert_allclose(bar.coords, reference, atol=3e-10, rtol=3e-11)
    assert reads == [(p, p + 1) for p in range(auxmol.nao)]
    shell_rows = int(numpy.diff(auxmol.ao_loc).max())
    assert max(native_bytes) <= 8 * len(rows) * shell_rows
    numpy.testing.assert_allclose(
        actual[0].coords.sum(axis=0) + actual[1].coords.sum(axis=0), 0, atol=3e-10)


@pytest.mark.parametrize('kind', ('complex', 'float32', 'asymmetric'))
def test_packed_interface_preserves_unsupported_dense_fallback(monkeypatch, kind):
    """Complex/float32 and nonsymmetric physical bars must retain exact fallback."""
    mol, auxmol = _molecules()
    rng = numpy.random.default_rng(711)
    density = rng.normal(size=(auxmol.nao, mol.nao, mol.nao))
    if kind != 'asymmetric':
        density = density + density.swapaxes(1, 2)
    if kind == 'complex':
        density = density + 1j * density * .3
    elif kind == 'float32':
        density = density.astype(numpy.float32)
    expected = _explicit_coords(mol, auxmol, density)

    def forbidden_packed(*args, **kwargs):
        raise AssertionError('unsupported density entered real symmetric native path')

    monkeypatch.setattr(_int3c_cross_opt, 'int3c_cross_bwd', forbidden_packed)
    # Symmetry is a producer contract of the optional filler; asymmetric
    # producers retain the existing dense reader instead of supplying it.
    filler = None if kind == 'asymmetric' else forbidden_packed
    actual = _cderi_vjp._int3c_mo_deriv_coords_vjp_from_z_reader(
        mol, auxmol, None, None, (0, mol.nao, 0, mol.nao),
        z_dtype=density.dtype, block_memory_mb=1, z_aux_block_max_rows=1,
        read_density_aux_block=lambda p0, p1: density[p0:p1],
        fill_packed_density_aux_block=filler)
    tolerance = 2e-5 if kind == 'float32' else 3e-10
    for bar, reference in zip(actual, expected):
        numpy.testing.assert_allclose(bar.coords, reference, atol=tolerance,
                                      rtol=tolerance)


@pytest.mark.parametrize(('with_j', 'with_k'), ((True, True), (True, False),
                                               (False, True)))
@pytest.mark.parametrize('factor_response', (False, True))
def test_factorized_packed_reverse_matches_density_pullback(
        tmp_path, monkeypatch, with_j, with_k, factor_response):
    """Packed J/K assembly must preserve molecular, auxiliary and factor bars."""
    monkeypatch.setenv('PYSCFAD_FACTOR_PANEL_MEMORY_MB', '0')
    monkeypatch.setenv('PYSCFAD_SOLVE_MEMORY_MB', '.001')
    monkeypatch.setenv('PYSCFAD_DF_OCCUPIED_VJP', '1')
    mol, _ = _molecules()
    obj = df.DF(mol, auxbasis='weigend', incore=False)
    obj._cderi_to_save = str(tmp_path / 'packed_fitted.h5')
    with config_update('pyscfad_moleintor_opt', True):
        obj.build()
    rng = numpy.random.default_rng(712)
    factors = np.asarray(rng.normal(size=(mol.nao, 3)))
    seeds = tuple(np.asarray(rng.normal(size=(mol.nao, mol.nao)))
                  if enabled else np.zeros((mol.nao, mol.nao))
                  for enabled in (with_j, with_k))

    with config_update('pyscfad_moleintor_opt', True):
        _, reference_pullback = jax.vjp(
            _dense_factor_values, obj.mol, obj.auxmol, factors)
        expected_mol, expected_auxmol, expected_factors = reference_pullback(seeds)
    coordinate = _cderi_vjp._int3c_mo_deriv_coords_vjp_from_z_reader
    packed_calls = []

    def require_packed(*args, **kwargs):
        packed_calls.append(kwargs.get('fill_packed_density_aux_block') is not None)
        return coordinate(*args, **kwargs)

    monkeypatch.setattr(_cderi_vjp, '_int3c_mo_deriv_coords_vjp_from_z_reader',
                        require_packed)
    _, pullback = jax.vjp(
        lambda obj_, factors_: df_jk.get_jk_from_occ(
            obj_, factors_, with_j=with_j, with_k=with_k,
            _factor_response=factor_response), obj, factors)
    actual_obj, actual_factors = pullback(seeds)
    assert packed_calls and all(packed_calls)
    numpy.testing.assert_allclose(actual_obj.mol.coords, expected_mol.coords,
                                  atol=2e-8, rtol=2e-10)
    numpy.testing.assert_allclose(actual_obj.auxmol.coords, expected_auxmol.coords,
                                  atol=2e-8, rtol=2e-10)
    numpy.testing.assert_allclose(
        actual_factors, expected_factors if factor_response else 0.,
        atol=2e-9, rtol=2e-10)
