"""Forward DF factors and resident-array iteration contracts."""

import h5py
import numpy
import pytest
from pyscf import lib

from pyscfad import df, gto
from pyscfad.df import _df_jk_opt


def _df_with_integrals():
    mol = gto.Mole(atom='He 0 0 0', basis='cc-pvdz', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    obj = df.DF(mol)
    rng = numpy.random.default_rng(713)
    obj._cderi = rng.normal(size=(9, mol.nao * (mol.nao + 1) // 2))
    obj.blockdim = 4
    return obj


def test_resident_loop_converts_once_before_slicing(monkeypatch):
    from pyscfad.df import df as df_module
    obj = _df_with_integrals()
    resident = obj._cderi
    conversions = []
    original = numpy.asarray

    def record(value, *args, **kwargs):
        conversions.append(value)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(df_module.numpy, 'asarray', record)
    blocks = list(obj.loop())
    assert len(conversions) == 1
    assert conversions[0] is resident
    numpy.testing.assert_array_equal(numpy.vstack(blocks), resident)


def test_loop_preserves_non_numpy_resident_and_disk(tmp_path):
    from pyscfad import numpy as np
    obj = _df_with_integrals()
    resident = np.asarray(obj._cderi)
    obj._cderi = resident
    blocks = list(obj.loop(to_numpy=False))
    assert all(type(block) is type(resident) for block in blocks)
    numpy.testing.assert_array_equal(numpy.vstack(blocks), resident)
    path = tmp_path / 'cderi.h5'
    with h5py.File(path, 'w') as handle:
        handle['j3c'] = numpy.asarray(resident)
    obj.attach_outcore_cderi(str(path))
    numpy.testing.assert_array_equal(numpy.vstack(list(obj.loop())), resident)


@pytest.mark.parametrize('batch', (False, True))
@pytest.mark.parametrize('case', ('psd', 'occupied', 'stale', 'indefinite', 'nonsymmetric', 'zero'))
def test_factor_dispatch_matches_independent_dense_jk(monkeypatch, case, batch):
    obj = _df_with_integrals()
    nao = obj.mol.nao
    rng = numpy.random.default_rng(718)
    coeff = rng.normal(size=(nao, 2))
    occ = numpy.asarray([2., .3])
    density = (coeff * occ) @ coeff.T
    if case == 'indefinite':
        density = numpy.diag(numpy.linspace(-1., 2., nao))
    elif case == 'zero':
        density[:] = 0
    elif case == 'nonsymmetric':
        density[0, 1] += .2
    if batch:
        density = numpy.stack([density, .6 * density]).reshape(1, 2, nao, nao)
        coeff = numpy.stack([coeff, numpy.sqrt(.6) * coeff]).reshape(1, 2, nao, 2)
        occ = numpy.broadcast_to(occ, (1, 2, 2))
    if case in ('occupied', 'stale', 'indefinite', 'nonsymmetric'):
        density = lib.tag_array(density, mo_coeff=coeff, mo_occ=occ)
    if case == 'stale':
        density.mo_coeff = coeff * 2
    if case == 'occupied':
        def fail_cholesky(*args, **kwargs):
            raise AssertionError('valid occupied factors must precede Cholesky')
        monkeypatch.setattr(_df_jk_opt, 'dpstrf', fail_cholesky)
    factors = _df_jk_opt._density_factors(density)
    if case in ('indefinite', 'nonsymmetric'):
        assert factors is None
    else:
        assert len(factors) == (2 if batch else 1)
        for x, d in zip(factors, density.reshape(-1, nao, nao)):
            assert x.flags.f_contiguous
            numpy.testing.assert_allclose(x @ x.T, d, atol=2e-13, rtol=2e-13)
    # JAX's custom-VJP boundary drops NumPy array attributes. Test the raw
    # forward body too, so stale-tag stripping and occupied priority are
    # checked where these attributes are available.
    B = lib.unpack_tril(obj._cderi)
    dense = numpy.asarray(density).reshape(-1, nao, nao)
    expected_j = numpy.einsum('Luv,xuv,Lij->xij', B, dense, B).reshape(density.shape)
    expected_k = numpy.einsum('Liu,xuv,Lvj->xij', B, dense, B).reshape(density.shape)
    J, K = _df_jk_opt.get_jk.__wrapped__(obj, density)
    numpy.testing.assert_allclose(J, expected_j, atol=2e-11, rtol=2e-12)
    numpy.testing.assert_allclose(K, expected_k, atol=2e-11, rtol=2e-12)


def test_cached_occupied_forward_precedes_cholesky(monkeypatch):
    obj = _df_with_integrals()
    nao = obj.mol.nao
    coeff = numpy.eye(nao)
    occ = numpy.zeros(nao)
    occ[:2] = [2., .3]
    density = (coeff * occ) @ coeff.T
    _df_jk_opt.set_fast_exchange_dm_data(obj, coeff, occ, numpy.eye(nao))

    def fail_cholesky(*args, **kwargs):
        raise AssertionError('cached occupied path must precede Cholesky')

    monkeypatch.setattr(_df_jk_opt, 'dpstrf', fail_cholesky)
    J, K = _df_jk_opt.get_jk(obj, density)
    B = lib.unpack_tril(obj._cderi)
    numpy.testing.assert_allclose(J, numpy.einsum('Luv,uv,Lij->ij', B, density, B))
    numpy.testing.assert_allclose(K, numpy.einsum('Liu,uv,Lvj->ij', B, density, B))


def test_unsupported_and_small_scale_stale_factors():
    coeff = numpy.asarray([[1.], [2.]]) * 1e-20
    density = coeff @ coeff.T
    stale = lib.tag_array(density, mo_coeff=2 * coeff, mo_occ=numpy.ones(1))
    factors = _df_jk_opt._density_factors(stale)
    numpy.testing.assert_allclose(factors[0] @ factors[0].T, density, atol=1e-54)
    assert _df_jk_opt._density_factors(density, hermi=0) is None
    assert _df_jk_opt._density_factors(density.astype(complex)) is None
    assert _df_jk_opt._density_factors(numpy.full((2, 2), numpy.nan)) is None
    zero = _df_jk_opt._density_factors(numpy.zeros((2, 2)))
    assert zero[0].shape == (2, 0)


@pytest.mark.parametrize('attribute', ('mo_coeff', 'mo_occ'))
def test_ragged_orbital_metadata_falls_back_to_density_factorization(attribute):
    density = lib.tag_array(numpy.eye(2), mo_coeff=numpy.eye(2),
                            mo_occ=numpy.ones(2))
    setattr(density, attribute, [[1.], [1., 2.]])
    factors = _df_jk_opt._density_factors(density)
    assert len(factors) == 1
    numpy.testing.assert_allclose(factors[0] @ factors[0].T, numpy.eye(2))
    tagged = _df_jk_opt._tag_dm(density)
    reconstructed = ((tagged.mo_coeff[0] * tagged.mo_occ[0])
                     @ tagged.mo_coeff[0].T)
    numpy.testing.assert_allclose(reconstructed, numpy.eye(2))
