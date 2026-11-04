"""Packed coordinate seeds preserve both AO legs and bounded reader ownership."""
import jax
import numpy as np
import pytest
from pyscfad import gto
from pyscfad.df import addons, _cderi_vjp as v
from pyscfad.lno import _df_shared
from pyscfad.lno.test import test_shared_local_df_reverse as shared


@pytest.mark.parametrize('backend', ['gemm', 'tiled', 'auto'])
@pytest.mark.parametrize('dims', [(1, 7, 2, 5), (3, 129, 7, 3), (2, 257, 5, 5),
                                  (1, 535, 3, 7), (0, 7, 2, 5), (2, 7, 0, 5)])
def test_packed_seed_matches_nonsymmetric_dense_reference(monkeypatch, backend, dims):
    monkeypatch.setenv('PYSCFAD_DF_MO_DENSITY_BACKEND', backend)
    nq, nao, nk, nl = dims
    rng = np.random.default_rng(901)
    # Noncontiguous inputs, unequal dimensions, partial AO tiles, and a
    # single auxiliary row exercise both contraction orders and scheduling.
    ck = rng.normal(size=(nao, 2*nk))[:, ::2]
    cl = np.asfortranarray(rng.normal(size=(nao, nl)))
    z = rng.normal(size=(nq, nk, 2*nl))[:, :, ::2]
    original = tuple(a.copy() for a in (ck, cl, z))
    dense = np.einsum('uk,pkl,vl->puv', ck, z, cl, optimize=True)
    rows, cols = np.tril_indices(nao)
    expected = dense[:, rows, cols] + dense[:, cols, rows]
    expected[:, rows == cols] *= .5
    storage = np.full((nq+2, len(rows)), np.nan)
    out = storage[1:-1]
    workspace = {}
    assert v._int3c_mo_packed_density(ck, cl, z, out, workspace=workspace) is out
    np.testing.assert_allclose(out, expected, atol=2e-11, rtol=2e-12)
    assert np.isnan(storage[[0, -1]]).all()
    v._int3c_mo_packed_density(ck, cl, -.4*z, out, accumulate=True, workspace=workspace)
    np.testing.assert_allclose(out, .6*expected, atol=2e-11, rtol=2e-12)
    for array, before in zip((ck, cl, z), original):
        np.testing.assert_array_equal(array, before)
    if 'tiles' in workspace:
        assert workspace['tiles'].size <= nq*nao*nao


@pytest.mark.parametrize('backend', ['gemm', 'tiled'])
def test_direct_geometry_reads_once_and_matches_legacy(monkeypatch, backend):
    mol = gto.Mole(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    aux = addons.make_auxmol(mol, 'weigend')
    rng = np.random.default_rng(19)
    coeff = rng.normal(size=(mol.nao, 5))
    seed = rng.normal(size=(aux.nao, 6))
    monkeypatch.setenv('PYSCFAD_DF_MO_DENSITY_BACKEND', 'legacy')
    expected = v._int3c_mo_deriv_coords_vjp(mol, aux, coeff, seed, (0, 2, 2, 5))
    monkeypatch.setenv('PYSCFAD_DF_MO_DENSITY_BACKEND', backend)
    def forbidden(*args, **kwargs):
        pytest.fail('Packed geometry must bypass the full symmetric density')
    monkeypatch.setattr(v, '_int3c_mo_physical_density', forbidden)
    reads = []
    def reader(begin, end):
        reads.append((begin, end))
        return seed[begin:end]
    for scale in (1., -.3):
        reads.clear()
        actual = v._int3c_mo_deriv_coords_vjp(mol, aux, coeff,
            lambda a,b: scale*reader(a,b), (0, 2, 2, 5), z_aux_block_max_rows=1)
        assert reads == [(a, a+1) for a in range(aux.nao)]
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got.coords, scale*want.coords, atol=2e-10, rtol=2e-11)


@pytest.mark.parametrize('backend', ['gemm', 'tiled'])
def test_shared_packed_geometry_preserves_coordinate_and_orbital_gradients(monkeypatch, backend):
    mol, coeff = shared._problem(7)
    monkeypatch.setenv('PYSCFAD_DF_MO_DENSITY_BACKEND', 'legacy')
    expected, ref = jax.value_and_grad(shared._energy, argnums=(0, 1))(mol, coeff)
    monkeypatch.setenv('PYSCFAD_DF_MO_DENSITY_BACKEND', backend)
    def forbidden(*args, **kwargs):
        pytest.fail('Shared geometry must fill the packed buffer directly')
    monkeypatch.setattr(v, '_int3c_mo_physical_density', forbidden)
    with _df_shared.reverse_scope(shared._mf(mol), shared.ATOMS) as scope:
        value, bars = jax.value_and_grad(shared._energy, argnums=(0, 1))(mol, coeff)
        geometry = scope.finish()
        assert scope.receipt['geometry_passes'] == 1
        assert not scope._packing_workspace
    np.testing.assert_allclose(value, expected, atol=1e-12)
    np.testing.assert_allclose(bars[0].coords+geometry, ref[0].coords, atol=2e-9, rtol=2e-8)
    np.testing.assert_allclose(bars[1], ref[1], atol=2e-10, rtol=2e-9)


def test_old_native_library_retains_gemm_packing(monkeypatch):
    monkeypatch.setattr(v, '_half_to_packed_tiles', None)
    monkeypatch.setenv('PYSCFAD_DF_MO_DENSITY_BACKEND', 'auto')
    rng = np.random.default_rng(7)
    c = rng.normal(size=(513, 2)); z = rng.normal(size=(1, 2, 2))
    actual = v._int3c_mo_packed_density(c, c, z)
    dense = np.einsum('uk,pkl,vl->puv', c, z, c)
    np.testing.assert_allclose(actual, v._pack_tril_symmetric_sum(dense), atol=1e-12)


def test_output_alias_and_backend_validation(monkeypatch):
    c = np.ones((2, 3)); z = np.ones((2, 3, 3))
    with pytest.raises(ValueError, match='overlap'):
        v._int3c_mo_packed_density(c, c, z, out=c)
    monkeypatch.setenv('PYSCFAD_DF_MO_DENSITY_BACKEND', 'typo')
    with pytest.raises(ValueError, match='BACKEND'):
        v._use_packed_mo_density()
