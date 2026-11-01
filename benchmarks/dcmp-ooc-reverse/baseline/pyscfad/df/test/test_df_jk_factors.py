"""Exact density factors must preserve orbital and molecular response."""

import jax
import numpy
import pytest

from pyscfad import config_update, df, gto
from pyscfad import numpy as np
from pyscfad.df import _df_jk_opt, df_jk


@pytest.fixture(autouse=True)
def clear_factor_memory_environment(monkeypatch):
    monkeypatch.delenv('PYSCFAD_FACTOR_PANEL_MEMORY_MB', raising=False)
    monkeypatch.delenv('PYSCFAD_SOLVE_MEMORY_MB', raising=False)


@pytest.mark.parametrize(('name', 'default'), [
    ('PYSCFAD_FACTOR_PANEL_MEMORY_MB', 256.0),
    ('PYSCFAD_SOLVE_MEMORY_MB', 64.0),
])
@pytest.mark.parametrize('value', (None, '', 'invalid', '-1', 'nan', 'inf'))
def test_memory_budget_missing_or_invalid_uses_default(monkeypatch, name, default, value):
    from pyscfad.df import _df_jk_factorized
    if value is not None:
        monkeypatch.setenv(name, value)
    assert _df_jk_factorized._memory_mb(name, default) == default


@pytest.mark.parametrize(('budget', 'in_memory'), [
    ('0', False), ('0.00025', False), ('0.0005', True), ('4096', True),
])
def test_panel_environment_selects_ram_or_disk(tmp_path, monkeypatch, budget, in_memory):
    from pyscf import lib
    from pyscfad.df import _df_jk_factorized
    monkeypatch.setattr(lib.param, 'TMPDIR', str(tmp_path))
    monkeypatch.setenv('PYSCFAD_FACTOR_PANEL_MEMORY_MB', budget)
    # The two panels together occupy 384 bytes.
    with _df_jk_factorized._factor_panels(4, 3, 2, True) as (T, R):
        assert isinstance(T, numpy.ndarray) == in_memory
        assert isinstance(R, numpy.ndarray) == in_memory
        T[:] = 1
        R[:] = 2
        numpy.testing.assert_array_equal(T[:], numpy.ones((4, 6)))
        numpy.testing.assert_array_equal(R[:], numpy.full((4, 6), 2))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('fail', (False, True))
def test_large_factor_scratch_closes_after_success_or_failure(tmp_path, monkeypatch, fail):
    from pyscf import lib
    from pyscfad.df import _df_jk_factorized
    monkeypatch.setattr(lib.param, 'TMPDIR', str(tmp_path))
    monkeypatch.setenv('PYSCFAD_FACTOR_PANEL_MEMORY_MB', '0')

    def use_panels():
        with _df_jk_factorized._factor_panels(4, 3, 2, True) as (T, R):
            handle = T.file
            T[:] = 1
            R[:] = 2
            if fail:
                raise RuntimeError('reverse failed')
        return handle

    if fail:
        with pytest.raises(RuntimeError, match='reverse failed'):
            use_panels()
    else:
        assert not use_panels().id.valid
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(('with_j', 'with_k'), [(True, True), (True, False),
                                               (False, True)])
@pytest.mark.parametrize('disk_panels', (False, True))
def test_occupied_factor_pullback_matches_density_pullback(
        tmp_path, monkeypatch, with_j, with_k, disk_panels):
    from pyscfad.df import _df_jk_factorized
    if disk_panels:
        monkeypatch.setenv('PYSCFAD_FACTOR_PANEL_MEMORY_MB', '0')
        monkeypatch.setenv('PYSCFAD_SOLVE_MEMORY_MB', '.001')
    mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='sto-3g', verbose=0, max_memory=1000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    obj = df.DF(mol, auxbasis='weigend')
    obj._cderi_to_save = str(tmp_path / 'fitted.h5')
    with config_update('pyscfad_moleintor_opt', True):
        obj.build()
    rng = numpy.random.default_rng(8202)
    factors = np.asarray(rng.normal(size=(mol.nao, 3)))
    seeds = tuple(np.asarray(rng.normal(size=(mol.nao, mol.nao)))
                  for _ in range(2))
    if not with_j:
        seeds = (np.zeros_like(seeds[0]), seeds[1])
    if not with_k:
        seeds = (seeds[0], np.zeros_like(seeds[1]))

    def density_call(obj_, factors_):
        # Zero seeds select each component independently, including the
        # legacy primitive whose disabled component is an integer scalar.
        return _df_jk_opt.get_jk(obj_, factors_ @ factors_.T)

    expected, ref_pullback = jax.vjp(density_call, obj, factors)
    ref_obj_bar, ref_factors_bar = ref_pullback(seeds)
    solve_columns = []
    original_solve = _df_jk_factorized.scipy.linalg.solve_triangular

    def record_solve(a, b, *args, **kwargs):
        if b.ndim == 2:
            solve_columns.append(b.shape[1])
        return original_solve(a, b, *args, **kwargs)

    monkeypatch.setattr(_df_jk_factorized.scipy.linalg, 'solve_triangular', record_solve)
    actual, pullback = jax.vjp(
        lambda obj_, factors_: df_jk.get_jk_from_occ(
            obj_, factors_, with_j=with_j, with_k=with_k), obj, factors)
    obj_bar, factors_bar = pullback(seeds)
    if with_k:
        assert solve_columns == ([1] * (mol.nao * 3) if disk_panels else [mol.nao * 3])
    for x, y, enabled in zip(actual, expected, (with_j, with_k)):
        if enabled:
            numpy.testing.assert_allclose(x, y, atol=2e-11, rtol=2e-11)
    numpy.testing.assert_allclose(factors_bar, ref_factors_bar,
                                  atol=2e-9, rtol=2e-10)
    numpy.testing.assert_allclose(obj_bar.mol.coords, ref_obj_bar.mol.coords,
                                  atol=2e-8, rtol=2e-10)
    numpy.testing.assert_allclose(obj_bar.auxmol.coords, ref_obj_bar.auxmol.coords,
                                  atol=2e-8, rtol=2e-10)


def test_factor_forward_matches_independent_tensor_contractions():
    mol = gto.Mole(atom='H 0 0 0; H 0 0 .74', basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    obj = df.DF(mol, auxbasis='weigend')
    with config_update('pyscfad_moleintor_opt', True):
        obj.build()
    S = np.asarray([[.4], [.7]])
    packed = numpy.asarray(obj._cderi)
    from pyscf import lib
    B = lib.unpack_tril(packed)
    D = numpy.asarray(S @ S.T)
    expected_j = numpy.einsum('puv,uv,pij->ij', B, D, B)
    expected_k = numpy.einsum('piu,uv,pvj->ij', B, D, B)
    J, K = df_jk.get_jk_from_occ(obj, S)
    numpy.testing.assert_allclose(J, expected_j, atol=2e-12, rtol=2e-12)
    numpy.testing.assert_allclose(K, expected_k, atol=2e-12, rtol=2e-12)


def test_implicit_scf_factor_paths_preserve_nonstationary_gradient(tmp_path, monkeypatch):
    from pyscfad import scf
    mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='sto-3g', verbose=0, max_memory=1000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    obj = df.DF(mol, auxbasis='weigend')
    obj._cderi_to_save = str(tmp_path / 'scf_fitted.h5')
    with config_update('pyscfad_moleintor_opt', True):
        obj.build()

    def objective(mol_):
        mf = scf.RHF(mol_).density_fit(auxbasis='weigend')
        mf.with_df.attach_outcore_cderi(obj._cderi_to_save)
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
        mf.chkfile = None
        mf.kernel()
        # Nonstationary orbital contribution exercises implicit response.
        return mf.e_tot + .03 * np.sum(mf.mo_energy[:5]**2)

    monkeypatch.setenv('PYSCFAD_DF_OCCUPIED_VJP', '0')
    with (config_update('pyscfad_moleintor_opt', True),
          config_update('pyscfad_scf_implicit_diff', True),
          config_update('pyscfad_scf_first_order_custom', False)):
        expected_value, expected_gradient = jax.value_and_grad(objective)(mol)
    calls = []
    original = df_jk.get_jk_from_occ

    def counted(*args, **kwargs):
        calls.append(args[1].shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(df_jk, 'get_jk_from_occ', counted)
    monkeypatch.setenv('PYSCFAD_DF_OCCUPIED_VJP', '1')
    with (config_update('pyscfad_moleintor_opt', True),
          config_update('pyscfad_scf_implicit_diff', True),
          config_update('pyscfad_scf_first_order_custom', False)):
        actual_value, actual_gradient = jax.value_and_grad(objective)(mol)
    numpy.testing.assert_allclose(actual_value, expected_value, atol=2e-11, rtol=0)
    numpy.testing.assert_allclose(actual_gradient.coords, expected_gradient.coords,
                                  atol=2e-9, rtol=2e-9)
    assert len(calls) >= 2
    assert all(shape == (mol.nao, 5) for shape in calls)
