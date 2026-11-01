"""Compact exchange-seed panels match unrestricted DF geometry responses."""
import importlib
import importlib.util

import jax
import numpy as np
import pytest
from pyscf import df as pyscf_df, lib
from pyscfad import config_update, df, gto
from pyscfad.df import _df_jk_factorized as factorized


def compact_module():
    name = 'pyscfad.df._df_jk_compact'
    assert importlib.util.find_spec(name) is not None, 'compact geometry preparation missing'
    return importlib.import_module(name)


@pytest.fixture
def case(tmp_path):
    mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    obj = df.DF(mol, auxbasis='weigend', incore=False)
    obj._cderi_to_save = str(tmp_path / 'source.h5')
    with config_update('pyscfad_moleintor_opt', True):
        obj.build()
    rng = np.random.default_rng(942)
    s = rng.normal(size=(mol.nao, 2)) * .3
    q = rng.normal(size=(mol.nao, 2)) * .2
    g = rng.normal(size=(mol.nao, mol.nao))
    g = (g + g.T) * .5
    return obj, s, q, g


@pytest.mark.parametrize('signs', [(1., 1.), (-1., -1.), (1., -1.)])
@pytest.mark.parametrize(('with_j', 'with_k'), [(True, True), (False, True), (True, False)])
@pytest.mark.parametrize('disk', [False, True])
def test_compact_preparation_matches_unrestricted_metric_density_and_coords(
        case, tmp_path, monkeypatch, signs, with_j, with_k, disk):
    compact = compact_module()
    obj, s, q, g = case
    h = (q * np.asarray(signs)) @ q.T
    h = (h + h.T) * .5
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    monkeypatch.setattr(lib.param, 'TMPDIR', str(scratch))
    if disk:
        monkeypatch.setenv('PYSCFAD_SOLVE_MEMORY_MB', '.001')
    with config_update('pyscfad_moleintor_opt', True):
        expected, _ = factorized._prepare_backward(with_j, with_k, False, (obj, s), (g, h))
        actual = compact.prepare_compact(with_j, with_k, (obj, s), (g, h), force_disk=disk)
        assert actual is not None
        try:
            assert actual.compact_exchange_rank == (2 if with_k else 0)
            assert actual.disk_panels == (disk and with_k)
            np.testing.assert_allclose(actual.lowbar, expected.lowbar, atol=3e-11, rtol=3e-11)
            # Multiple reader ranges check the compact panel indexing as well as total values.
            for begin, end in ((0, 3), (3, obj.get_naoaux())):
                np.testing.assert_allclose(actual.read_density_aux_block(begin, end),
                                           expected.read_density_aux_block(begin, end),
                                           atol=3e-11, rtol=3e-11)
                shape = (end-begin, obj.mol.nao * (obj.mol.nao+1) // 2)
                got, want = np.empty(shape), np.empty(shape)
                actual.fill_packed_density_aux_block(begin, end, got)
                expected.fill_packed_density_aux_block(begin, end, want)
                np.testing.assert_allclose(got, want, atol=3e-11, rtol=3e-11)
            got, want = actual.contract(), expected.contract()
            for x, y in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want), strict=True):
                np.testing.assert_allclose(x, y, atol=3e-10, rtol=3e-10)
        finally:
            actual.close()
            expected.close()
        actual.close()
    assert not list(scratch.iterdir())


def test_compact_zero_seed_uses_no_exchange_panel(case):
    compact = compact_module()
    obj, s, _, g = case
    zero = np.zeros_like(g)
    with config_update('pyscfad_moleintor_opt', True):
        actual = compact.prepare_compact(False, True, (obj, s), (zero, zero), force_disk=True)
        assert actual is not None
        try:
            assert actual.compact_exchange_rank == 0
            assert not actual.disk_panels
            np.testing.assert_array_equal(actual.lowbar, np.zeros_like(actual.lowbar))
            np.testing.assert_array_equal(actual.read_density_aux_block(0, 2), np.zeros((2, obj.mol.nao, obj.mol.nao)))
        finally:
            actual.close()


def test_compact_rejects_full_rank_even_with_small_physical_eigenvalues(case):
    compact = compact_module()
    obj, s, _, g = case
    # Every direction exceeds the matrix's floating-point resolution. A loose
    # low-rank tolerance would silently delete most of this physical seed.
    h = np.diag([1., 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 5e-15])
    assert compact.prepare_compact(True, True, (obj, s), (g, h)) is None


@pytest.mark.parametrize('invalid', ['float32', 'complex', 'nonfinite', 'asymmetric'])
def test_compact_keeps_unsupported_inputs_on_the_existing_path(case, invalid):
    compact = compact_module()
    obj, s, q, g = case
    h = q @ q.T
    if invalid == 'float32':
        s = s.astype(np.float32)
    elif invalid == 'complex':
        h = h.astype(np.complex128)
    elif invalid == 'nonfinite':
        h[0, 0] = np.nan
    elif invalid == 'asymmetric':
        h[0, 1] += .1
    assert compact.prepare_compact(True, True, (obj, s), (g, h)) is None


def test_compact_uses_packed_native_projection_and_cleans_failed_scratch(case, tmp_path, monkeypatch):
    compact = compact_module()
    obj, s, q, g = case
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    monkeypatch.setattr(lib.param, 'TMPDIR', str(scratch))
    def forbidden(*args, **kwargs):
        raise AssertionError('compact geometry unpacked a full AO factor block')
    monkeypatch.setattr(lib, 'unpack_tril', forbidden)
    with config_update('pyscfad_moleintor_opt', True):
        actual = compact.prepare_compact(True, True, (obj, s), (g, q @ q.T), force_disk=True)
        assert actual is not None
        actual.close()
        original = compact._ao2mo.nr_e2
        def failed_projection(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError('projection failure')
        monkeypatch.setattr(compact._ao2mo, 'nr_e2', failed_projection)
        with pytest.raises(RuntimeError, match='projection failure'):
            compact.prepare_compact(True, True, (obj, s), (g, q @ q.T), force_disk=True)
    assert not list(scratch.iterdir())


def test_compact_coordinate_derivative_matches_independent_finite_difference(case):
    compact = compact_module()
    obj, s, q, g = case
    h = (q * np.array([1., -1.])) @ q.T
    h = (h + h.T) * .5
    with config_update('pyscfad_moleintor_opt', True):
        contribution = compact.prepare_compact(True, True, (obj, s), (g, h))
        assert contribution is not None
        try:
            bar = contribution.contract()
        finally:
            contribution.close()
    mol, aux = obj.mol.to_pyscf(), obj.auxmol.to_pyscf()
    d = s @ s.T
    def energy():
        b = lib.unpack_tril(pyscf_df.incore.cholesky_eri(mol, auxmol=aux))
        j = np.einsum('Lij,ij,Lpq->pq', b, d, b)
        k = np.einsum('Lpi,ij,Ljq->pq', b, d, b)
        return np.sum(j*g) + np.sum(k*h)
    step = 2e-4
    for target, expected in ((mol, bar.mol), (aux, bar.auxmol)):
        coordinates = target.atom_coords()
        plus, minus = coordinates.copy(), coordinates.copy()
        plus[1, 2] += step
        minus[1, 2] -= step
        target.set_geom_(plus, unit='Bohr')
        ep = energy()
        target.set_geom_(minus, unit='Bohr')
        em = energy()
        target.set_geom_(coordinates, unit='Bohr')
        np.testing.assert_allclose(expected.coords[1, 2], (ep-em)/(2*step), atol=3e-7, rtol=3e-7)


@pytest.mark.parametrize('exchange_rank', [1, 3])
def test_compact_projection_preserves_unequal_factor_dimensions(case, exchange_rank):
    compact = compact_module()
    obj, s, _, g = case
    q = np.random.default_rng(949).normal(size=(obj.mol.nao, exchange_rank)) * .2
    signs = np.ones(exchange_rank)
    signs[-1] = -1
    h = (q*signs) @ q.T
    h = (h+h.T)*.5
    with config_update('pyscfad_moleintor_opt', True):
        expected, _ = factorized._prepare_backward(True, True, False, (obj, s), (g, h))
        actual = compact.prepare_compact(True, True, (obj, s), (g, h))
        assert actual is not None
        try:
            assert actual.compact_exchange_rank == exchange_rank
            np.testing.assert_allclose(actual.lowbar, expected.lowbar, atol=3e-11, rtol=3e-11)
            np.testing.assert_allclose(actual.read_density_aux_block(0, obj.get_naoaux()),
                                       expected.read_density_aux_block(0, obj.get_naoaux()),
                                       atol=3e-11, rtol=3e-11)
        finally:
            actual.close()
            expected.close()


def test_compact_traced_input_retains_generic_fallback(case):
    compact = compact_module()
    obj, s, q, g = case
    @jax.jit
    def traced(factors):
        assert compact.prepare_compact(True, True, (obj, factors), (g, q@q.T)) is None
        return jax.numpy.sum(factors)
    np.testing.assert_allclose(traced(s), s.sum())
