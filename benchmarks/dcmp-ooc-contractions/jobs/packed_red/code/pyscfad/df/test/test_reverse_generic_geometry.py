"""Thin geometry work must preserve the independent AO-density transpose."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import lib

from pyscfad import config_update, df, gto
from pyscfad.df import _df_jk_opt, _df_jk_factorized, incore


@pytest.fixture
def problem(tmp_path, monkeypatch):
    monkeypatch.setenv('PYSCFAD_DF_OCCUPIED_VJP', '1')
    monkeypatch.setenv('PYSCFAD_FACTOR_PANEL_MEMORY_MB', '4')
    mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='sto-3g', verbose=0, max_memory=1000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    obj = df.DF(mol, auxbasis='weigend')
    obj.incore = False
    obj._cderi_to_save = str(tmp_path / 'factors.h5')
    with config_update('pyscfad_moleintor_opt', True):
        obj.build()
    rng = np.random.default_rng(9401)
    factors = rng.normal(size=(mol.nao, 2)) * .2
    density = factors @ factors.T
    seeds = [tuple(rng.normal(size=density.shape) for _ in range(2))
             for _ in range(2)]
    return obj, density, factors, seeds


def _dense_values(mol, auxmol, density):
    packed = incore.cholesky_eri(mol, auxmol=auxmol)
    if packed.shape[1] == mol.nao ** 2:
        b = packed.reshape(-1, mol.nao, mol.nao)
    else:
        i, j = np.tril_indices(mol.nao)
        b = jnp.zeros((packed.shape[0], mol.nao, mol.nao))
        b = b.at[:, i, j].set(packed).at[:, j, i].set(packed)
    return (jnp.einsum('Lij,ij,Lpq->pq', b, density, b),
            jnp.einsum('Lpi,ij,Ljq->pq', b, density, b))


def _spill_observer(monkeypatch):
    paths = []
    original = _df_jk_opt.tempfile.mkstemp
    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get('prefix', '').startswith('pyscfad_dfjk_eri_bar_'):
            paths.append(result[1])
        return result
    monkeypatch.setattr(_df_jk_opt.tempfile, 'mkstemp', observe)
    return paths


@pytest.mark.parametrize(('with_j', 'with_k'), [(True, True), (True, False), (False, True)])
def test_repeated_geometry_and_full_density_bars_without_ao_spill(
        problem, monkeypatch, with_j, with_k):
    obj, density, factors, seeds = problem
    with config_update('pyscfad_moleintor_opt', True):
        _, reference = jax.vjp(_dense_values, obj.mol, obj.auxmol, jnp.asarray(density))
    paths = _spill_observer(monkeypatch)
    expected_first = None
    for bars in (seeds[0], seeds[1], seeds[0]):
        bars = tuple(bar if active else np.zeros_like(bar)
                     for bar, active in zip(bars, (with_j, with_k)))
        with config_update('pyscfad_moleintor_opt', True):
            expected = reference(bars)
        actual_obj, actual_dm = _df_jk_opt.get_jk_bwd(
            1, with_j, with_k, 1e-13,
            (obj, density, (True, False, True, True)), bars)
        for actual, wanted in zip((actual_obj.mol.coords, actual_obj.auxmol.coords,
                                   actual_dm), (expected[0].coords, expected[1].coords,
                                                expected[2])):
            np.testing.assert_allclose(actual, wanted, atol=2e-9, rtol=2e-10)
        assert actual_obj._cderi is None
        if expected_first is None:
            expected_first = np.asarray(actual_dm).copy()
    np.testing.assert_allclose(actual_dm, expected_first, atol=2e-12, rtol=2e-12)
    # Sbar alone cannot reconstruct this nonzero virtual-virtual block.
    q, _ = np.linalg.qr(factors, mode='complete')
    outside = q[:, factors.shape[1]:]
    assert np.linalg.norm(outside.T @ actual_dm @ outside) > 1e-6
    assert paths == [], 'eligible geometry wrote full AO-factor adjoints'


@pytest.mark.parametrize(('mol_active', 'aux_active'), [(True, False), (False, True), (True, True)])
def test_geometry_only_preserves_inactive_bars(problem, monkeypatch, mol_active, aux_active):
    obj, density, _, seeds = problem
    with config_update('pyscfad_moleintor_opt', True):
        _, reference = jax.vjp(_dense_values, obj.mol, obj.auxmol, jnp.asarray(density))
        expected = reference(seeds[0])
    paths = _spill_observer(monkeypatch)
    actual, dm_bar = _df_jk_opt.get_jk_bwd(
        1, True, True, 1e-13,
        (obj, density, (False, False, mol_active, aux_active)), seeds[0])
    assert dm_bar is None and actual._cderi is None
    for bar, wanted, active in zip((actual.mol.coords, actual.auxmol.coords),
                                   (expected[0].coords, expected[1].coords),
                                   (mol_active, aux_active)):
        if active:
            np.testing.assert_allclose(bar, wanted, atol=2e-9, rtol=2e-10)
        else:
            assert bar is None
    assert paths == []


@pytest.mark.parametrize('mode', ['disabled', 'asymmetric', 'indefinite', 'callback'])
def test_ineligible_geometry_keeps_generic_route(problem, monkeypatch, mode):
    obj, density, _, seeds = problem
    if mode == 'disabled':
        monkeypatch.setenv('PYSCFAD_DF_OCCUPIED_VJP', '0')
    elif mode == 'asymmetric':
        density = density.copy()
        density[0, 1] += .1
    elif mode == 'indefinite':
        density = density - np.eye(density.shape[0])
    def forbidden(*args, **kwargs):
        raise AssertionError('ineligible generic geometry entered factorized reverse')
    monkeypatch.setattr(_df_jk_factorized, '_backward', forbidden)
    with config_update('pyscfad_moleintor_opt', True):
        _, reference = jax.vjp(_dense_values, obj.mol, obj.auxmol, jnp.asarray(density))
        expected = reference(seeds[0])
    callbacks = []
    def coordinate_callback(*args, **kwargs):
        callbacks.append(1)
        return _df_jk_opt._cderi_mol_aux_vjp_from_block_fn(*args, **kwargs)
    actual, dm_bar = _df_jk_opt.get_jk_bwd(
        1, True, True, 1e-13, (obj, density, (True, False, True, True)), seeds[0],
        coordinate_vjp=coordinate_callback if mode == 'callback' else None)
    for bar, wanted in zip((actual.mol.coords, actual.auxmol.coords, dm_bar),
                           (expected[0].coords, expected[1].coords, expected[2])):
        np.testing.assert_allclose(bar, wanted, atol=2e-9, rtol=2e-10)
    assert bool(callbacks) == (mode == 'callback')


def test_stale_tags_cannot_change_geometry(problem, monkeypatch):
    obj, density, factors, seeds = problem
    # Invalid tags may be discarded and correct factors recovered from D.
    tagged = lib.tag_array(density, mo_coeff=factors * 3, mo_occ=np.ones(2))
    with config_update('pyscfad_moleintor_opt', True):
        _, reference = jax.vjp(_dense_values, obj.mol, obj.auxmol, jnp.asarray(density))
        expected = reference(seeds[0])
    actual, dm_bar = _df_jk_opt.get_jk_bwd(
        1, True, True, 1e-13, (obj, tagged, (True, False, True, True)), seeds[0])
    for bar, wanted in zip((actual.mol.coords, actual.auxmol.coords, dm_bar),
                           (expected[0].coords, expected[1].coords, expected[2])):
        np.testing.assert_allclose(bar, wanted, atol=2e-9, rtol=2e-10)
