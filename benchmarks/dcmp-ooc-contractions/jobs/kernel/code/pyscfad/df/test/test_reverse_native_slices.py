"""Independent checks for bounded native coordinate contractions."""

import numpy
import pytest

from pyscfad import gto
from pyscfad.df import _cderi_vjp, _int3c_cross_opt


def _molecules():
    mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    # Independent centres and unequal atom counts expose auxiliary mapping bugs.
    auxmol = gto.Mole(atom='H .1 .2 .3; H -.2 .1 1.1',
                      basis='weigend', verbose=0)
    auxmol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol, auxmol


def _explicit_ip1_reference(mol, auxmol, physical_density):
    """Contract an independently formed density with all explicit ip1 values."""
    ints = numpy.asarray(_int3c_cross_opt.int3c_cross(
        mol, auxmol, intor='int3c2e_ip1', comp=3, aosym='s1'))
    per_aux_ao = numpy.einsum('puv,xuvp->pux', physical_density, ints)
    mol_coords = numpy.zeros((mol.natm, 3), dtype=physical_density.dtype)
    aux_coords = numpy.zeros((auxmol.natm, 3), dtype=physical_density.dtype)
    for atom, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        mol_coords[atom] = -per_aux_ao[:, p0:p1].sum(axis=(0, 1))
    for atom, (_, _, p0, p1) in enumerate(auxmol.aoslice_by_atom()):
        aux_coords[atom] = per_aux_ao[p0:p1].sum(axis=(0, 1))
    return mol_coords, aux_coords


def _assert_coords(actual, expected):
    for result, reference in zip(actual, expected):
        numpy.testing.assert_allclose(result.coords, reference,
                                      atol=3e-10, rtol=3e-11)
    numpy.testing.assert_allclose(
        actual[0].coords.sum(axis=0) + actual[1].coords.sum(axis=0),
        0, atol=3e-10)


@pytest.mark.parametrize('i0', [0, 1])
@pytest.mark.parametrize('slice_kind', ['within_atom', 'across_atoms', 'last_atom'])
def test_native_auxiliary_slice_preserves_pair_and_atom_offsets(i0, slice_kind):
    """Wrong global/local aux offsets or shifted packed rows change gradients."""
    mol, auxmol = _molecules()
    boundary = int(auxmol.aoslice_by_atom()[1, 0])
    a0, a1 = {
        'within_atom': (1, boundary - 1),
        'across_atoms': (boundary - 1, boundary + 1),
        'last_atom': (boundary + 1, auxmol.nbas),
    }[slice_kind]
    p0, p1 = map(int, auxmol.ao_loc[[a0, a1]])
    i1 = mol.nbas - 1
    ao0, ao1 = map(int, mol.ao_loc[[i0, i1]])
    pair0, pair1 = ao0 * (ao0 + 1) // 2, ao1 * (ao1 + 1) // 2
    seed = numpy.random.default_rng(913).normal(size=(pair1 - pair0, p1 - p0))
    full = numpy.zeros((mol.nao * (mol.nao + 1) // 2, auxmol.nao))
    full[pair0:pair1, p0:p1] = seed
    rows, cols = numpy.tril_indices(mol.nao)
    physical = numpy.zeros((auxmol.nao, mol.nao, mol.nao))
    physical[:, rows, cols] = full.T
    physical[:, cols, rows] += full.T
    expected = _explicit_ip1_reference(mol, auxmol, physical)
    full_bars = _int3c_cross_opt.int3c_cross_bwd(
        'int3c2e', 1, 's2ij', None, None, (mol, auxmol), full)
    _assert_coords(full_bars, expected)
    slices = (i0, i1, 0, mol.nbas, mol.nbas + a0, mol.nbas + a1)
    actual = _int3c_cross_opt.int3c_cross_bwd(
        'int3c2e', 1, 's2ij', slices, None, (mol, auxmol), seed)
    _assert_coords(actual, expected)


@pytest.mark.parametrize('use_density_reader', [False, True])
@pytest.mark.parametrize('kc,lc', [(2, 5), (5, 2)])
def test_mo_coordinate_reverse_avoids_derivative_tensor(
        monkeypatch, use_density_reader, kc, lc):
    """The native path must retain both AO legs, diagonal weights, and row caps."""
    mol, auxmol = _molecules()
    rng = numpy.random.default_rng(914)
    coeff = rng.normal(size=(mol.nao, kc + lc))
    z = rng.normal(size=(auxmol.nao, kc * lc))
    density = numpy.einsum('uk,pkl,vl->puv', coeff[:, :kc],
                           z.reshape(auxmol.nao, kc, lc), coeff[:, kc:])
    physical = density + density.swapaxes(1, 2)
    expected = _explicit_ip1_reference(mol, auxmol, physical)
    original = _int3c_cross_opt.int3c_cross

    def no_derivative_tensor(*args, **kwargs):
        assert kwargs.get('comp', 1) == 1, 'full derivative tensor materialized'
        return original(*args, **kwargs)

    monkeypatch.setattr(_int3c_cross_opt, 'int3c_cross', no_derivative_tensor)
    native_bwd = _int3c_cross_opt.int3c_cross_bwd
    packed_bytes = []

    def bounded_native(*args, **kwargs):
        packed_bytes.append(args[-1].nbytes)
        return native_bwd(*args, **kwargs)

    monkeypatch.setattr(_int3c_cross_opt, 'int3c_cross_bwd', bounded_native)
    reads = []

    def read_z(p0, p1):
        assert not use_density_reader
        reads.append((p0, p1))
        return z[p0:p1]

    def read_density(p0, p1):
        reads.append((p0, p1))
        # This density already contains BOTH AO derivative legs.
        return physical[p0:p1]

    actual = _cderi_vjp._int3c_mo_deriv_coords_vjp_from_z_reader(
        mol, auxmol, coeff, read_z, (0, kc, kc, kc + lc),
        z_dtype=numpy.float64, block_memory_mb=1,
        z_aux_block_max_rows=1,
        read_density_aux_block=read_density if use_density_reader else None)
    _assert_coords(actual, expected)
    assert reads == [(p, p + 1) for p in range(auxmol.nao)]
    # A one-row reader cap permits one indivisible auxiliary shell in the
    # native cotangent, never an unbounded full-auxiliary packed allocation.
    shell_rows = int(numpy.diff(auxmol.ao_loc).max())
    npair = mol.nao * (mol.nao + 1) // 2
    assert max(packed_bytes) <= 8 * npair * shell_rows


@pytest.mark.parametrize('unsupported', ['complex', 'asymmetric_density'])
def test_coordinate_reverse_preserves_explicit_fallback(monkeypatch, unsupported):
    """Native real symmetric packed bars cannot represent these contractions."""
    mol, auxmol = _molecules()
    rng = numpy.random.default_rng(915)
    coeff = rng.normal(size=(mol.nao, 5))
    z = rng.normal(size=(auxmol.nao, 6))
    if unsupported == 'complex':
        z = z + 1j * rng.normal(size=z.shape)
    density = numpy.einsum('uk,pkl,vl->puv', coeff[:, :2],
                           z.reshape(auxmol.nao, 2, 3), coeff[:, 2:])
    if unsupported == 'complex':
        density = density + density.swapaxes(1, 2)
    expected = _explicit_ip1_reference(mol, auxmol, density)

    def forbidden_native(*args, **kwargs):
        raise AssertionError('unsupported contraction sent to real packed VJP')

    monkeypatch.setattr(_int3c_cross_opt, 'int3c_cross_bwd', forbidden_native)
    actual = _cderi_vjp._int3c_mo_deriv_coords_vjp_from_z_reader(
        mol, auxmol, coeff, lambda p0, p1: z[p0:p1], (0, 2, 2, 5),
        z_dtype=z.dtype, z_aux_block_max_rows=1,
        read_density_aux_block=(lambda p0, p1: density[p0:p1])
        if unsupported == 'asymmetric_density' else None)
    _assert_coords(actual, expected)
