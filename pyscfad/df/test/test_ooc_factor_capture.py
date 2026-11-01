"""Capture geometry factors while the forward density's orbital frame is live."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad import config_update, df, gto
from pyscfad.df import _df_jk_opt as jk, _df_jk_factorized as factorized


@pytest.fixture
def case(tmp_path):
    mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='6-31g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    obj = df.DF(mol, auxbasis='weigend', incore=False)
    obj._cderi_to_save = str(tmp_path / 'capture.h5')
    with config_update('pyscfad_moleintor_opt', True):
        obj.build()
    rng = np.random.default_rng(271)
    coeff = np.linalg.qr(rng.normal(size=(mol.nao, mol.nao)))[0]
    occ = np.zeros(mol.nao)
    occ[:2] = 2
    s = coeff[:, :2] * np.sqrt(2.)
    jk.set_fast_exchange_dm_data(obj, coeff, occ, np.eye(mol.nao))
    return obj, s, s @ s.T, rng


def assert_tree_close(actual, expected):
    for x, y in zip(jax.tree_util.tree_leaves(actual),
                    jax.tree_util.tree_leaves(expected), strict=True):
        np.testing.assert_allclose(x, y, atol=2e-10, rtol=2e-10)


def test_forward_capture_retains_exact_factors_after_orbital_cache_changes(case, monkeypatch):
    obj, s, dm, rng = case
    seeds = tuple(rng.normal(size=dm.shape) for _ in range(2))
    monkeypatch.setenv('PYSCFAD_DF_OOC_CAPTURE_FACTORS', '0')
    _, ordinary = jax.vjp(jk.get_jk, obj, dm)
    monkeypatch.setenv('PYSCFAD_DF_OOC_CAPTURE_FACTORS', '1')
    _, captured = jax.vjp(jk.get_jk, obj, dm)
    # Simulate the extra SCF eigensolver replacing the shared orbital cache.
    coeff = np.linalg.qr(rng.normal(size=(obj.mol.nao, obj.mol.nao)))[0]
    occ = np.zeros(obj.mol.nao)
    occ[:2] = 2
    jk.set_fast_exchange_dm_data(obj, coeff, occ, np.eye(obj.mol.nao))
    seen_residuals, seen_factors = [], []
    residual = jk._jk_residual
    backward = factorized._backward
    def record_residual(res):
        seen_residuals.append(res)
        return residual(res)
    def record_backward(with_j, with_k, response, res, bars, **kwargs):
        seen_factors.append(np.array(res[1]))
        return backward(with_j, with_k, response, res, bars, **kwargs)
    monkeypatch.setattr(jk, '_jk_residual', record_residual)
    monkeypatch.setattr(factorized, '_backward', record_backward)
    actual = captured(seeds)
    assert len(seen_residuals[0]) == 4, 'forward geometry factors were not retained'
    np.testing.assert_array_equal(seen_residuals[0][3], s)
    np.testing.assert_array_equal(seen_factors[0], s)
    assert_tree_close(actual, ordinary(seeds))
    assert any(len(res) == 3 for res in seen_residuals)
    assert_tree_close(captured(tuple(2 * seed for seed in seeds)),
                      jax.tree_util.tree_map(lambda x: 2 * x, actual))


def test_capture_omits_nonbeneficial_or_inactive_geometry(case, monkeypatch):
    obj, s, dm, _ = case
    monkeypatch.setenv('PYSCFAD_DF_OOC_CAPTURE_FACTORS', '1')
    assert hasattr(jk, '_capture_geometry_factors'), 'guarded capture helper missing'
    capture = jk._capture_geometry_factors
    assert capture(obj, dm, 1, (True, False, False, False)) is None
    assert capture(obj, np.eye(obj.mol.nao), 1, (True, False, True, True)) is None
    assert capture(obj, dm.astype(np.float32), 1, (True, False, True, True)) is None
    assert capture(obj, dm, 0, (True, False, True, True)) is None
    @jax.jit
    def traced(d):
        assert capture(obj, d, 1, (True, False, True, True)) is None
        return jnp.sum(d)
    np.testing.assert_allclose(traced(dm), dm.sum())


def test_four_part_residual_preserves_traced_density_and_factor_bars(case, monkeypatch):
    obj, s, dm, rng = case
    obj._cderi = rng.normal(size=(4, obj.mol.nao * (obj.mol.nao + 1) // 2))
    obj._prefer_cderi_to_save = False
    seeds = tuple(rng.normal(size=dm.shape) for _ in range(2))
    activity = (True, True, False, False)
    @jax.jit
    def explicit(jbar, kbar):
        return jk.get_jk_bwd(1, True, True, 1e-13,
                            (obj, dm, activity), (jbar, kbar))
    @jax.jit
    def retained(jbar, kbar):
        return jk.get_jk_bwd(1, True, True, 1e-13,
                            (obj, dm, activity, s), (jbar, kbar))
    assert_tree_close(retained(*seeds), explicit(*seeds))


def test_complete_scf_capture_coalesces_only_exact_root_factors(case, monkeypatch):
    from pyscfad import scf
    from pyscfad.df import _ooc_cotangent
    obj, _, _, _ = case
    def outputs(inputs):
        mol, source = inputs
        mf = scf.RHF(mol).density_fit(with_df=source)
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-9
        mf.kernel()
        return mf.e_tot, mf.mo_energy, mf.mo_coeff
    with config_update('pyscfad_moleintor_opt', True), \
         config_update('pyscfad_scf_implicit_diff', True):
        reference = None
        for enabled in (False, True):
            monkeypatch.setenv('PYSCFAD_DF_OOC_CAPTURE_FACTORS', str(int(enabled)))
            values, pullback = jax.vjp(outputs, (obj.mol, obj))
            seed = tuple(np.ones_like(x) * .01 for x in values)
            with _ooc_cotangent.geometry_scope(obj, enabled=True, force_disk=True) as scope:
                bars, = pullback(seed)
                actual = bars[0], scope.finish(bars[1])
            assert scope.stats['consumers'] == 3
            assert scope.stats['seed_groups'] == (2 if enabled else 3)
            assert scope.stats['geometry_passes'] == 1
            if reference is None:
                reference = actual
            else:
                assert_tree_close(actual, reference)
