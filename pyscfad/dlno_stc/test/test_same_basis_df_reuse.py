"""Explicit identical correlation fits reuse the reference factor source."""
import numpy as np
import pytest
from pyscfad import gto, scf
from pyscfad.dlno_stc import prepare


@pytest.fixture(scope='module')
def reference():
    mol = gto.Mole(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = scf.RHF(mol).density_fit(auxbasis='weigend')
    mf.kernel()
    return mf


@pytest.mark.parametrize('canonical', [False, True])
def test_identical_explicit_basis_uses_cached_source(reference, monkeypatch, canonical):
    mf = reference
    operation = (prepare.prepare_canonical_system_inputs if canonical
                 else prepare.prepare_system_inputs)
    expected = operation(mf, [0], [1])
    source = mf.with_df._get_cderi_source()
    def unexpected(*args, **kwargs):
        raise AssertionError('identical explicit basis rebuilt local factors')
    monkeypatch.setattr(prepare.lno_df, 'get_local_Lov', unexpected)
    actual = operation(mf, [0], [1], correlation_auxbasis='weigend')
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-12, rtol=0)
    assert mf.with_df._get_cderi_source() is source


def test_explicit_basis_compatibility_is_conservative(reference):
    from copy import copy
    from pyscfad.df.addons import make_auxmol
    from pyscfad.dlno_stc._df_basis import can_reuse_scf_df
    mf = copy(reference)
    mf.with_df = copy(reference.with_df)
    mf.with_df._cderi = reference.with_df._cderi
    mf.with_df.auxmol = reference.with_df.auxmol
    assert can_reuse_scf_df(mf, 'weigend')
    assert not can_reuse_scf_df(mf, 'cc-pvdz-ri')
    original_auxmol = mf.with_df.auxmol
    mf.with_df.auxmol = copy(original_auxmol)
    mf.with_df.auxmol.coords = np.asarray(mf.mol.coords) + 0.1
    assert not can_reuse_scf_df(mf, 'weigend')
    mf.with_df.auxmol = make_auxmol(mf.mol, 'cc-pvdz-ri')
    assert not can_reuse_scf_df(mf, 'weigend')
    mf.with_df.auxbasis = None
    assert not can_reuse_scf_df(mf, 'weigend')
    mf.with_df._cderi = None
    assert not can_reuse_scf_df(mf, None)


def test_exact_screening_reuses_identical_explicit_fit(reference, monkeypatch):
    from pyscfad.dlno_stc import _pairs, build_domain_selections, domain
    from pyscfad.lno import df as lno_df
    static = build_domain_selections(reference, frozen=0, pair_energy_model='all')
    common = domain.rebuild_domain_data(reference, static)
    expected = _pairs._exact_boys_pair_scores(reference, common, static)
    def unexpected(*args, **kwargs):
        raise AssertionError('exact screening rebuilt identical fitting factors')
    monkeypatch.setattr(lno_df, 'get_local_Lov', unexpected)
    actual = _pairs._exact_boys_pair_scores(
        reference, common, static, correlation_auxbasis='weigend')
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def test_disk_source_is_reused(reference, tmp_path, monkeypatch):
    from copy import copy
    import h5py
    mf = copy(reference)
    mf.with_df = copy(reference.with_df)
    mf.with_df._cderi = reference.with_df._cderi
    mf.with_df.auxmol = reference.with_df.auxmol
    path = str(tmp_path / 'factors.h5')
    with h5py.File(path, 'w') as handle:
        handle['j3c'] = np.asarray(reference.with_df._get_cderi_source())
    mf.with_df.attach_outcore_cderi(path)
    expected = prepare.prepare_canonical_system_inputs(mf, [0], [1])
    def unexpected(*args, **kwargs):
        raise AssertionError('disk source was replaced by direct factors')
    monkeypatch.setattr(prepare.lno_df, 'get_local_Lov', unexpected)
    actual = prepare.prepare_canonical_system_inputs(
        mf, [0], [1], correlation_auxbasis='weigend')
    np.testing.assert_allclose(actual['B'], expected['B'], atol=1e-12, rtol=0)
    assert mf.with_df._get_cderi_source() == path


def test_concrete_basis_route_reuses_cache_inside_vjp(reference, monkeypatch):
    import jax
    from pyscfad.dlno_stc.domain import (select_system_boys_reference,
                                         select_system_virtual_anchor_columns)
    from pyscfad.dlno_stc._df_basis import can_reuse_scf_df
    reuse = can_reuse_scf_df(reference, "weigend")
    assert reuse
    boys = select_system_boys_reference(reference, [0])
    anchors = select_system_virtual_anchor_columns(reference, [1])
    def unexpected(*args, **kwargs):
        raise AssertionError('traced identical basis rebuilt local factors')
    monkeypatch.setattr(prepare.lno_df, 'get_local_Lov', unexpected)
    # The driver fixes this numerical route before duplicated pytree leaves
    # become distinct tracers; the scientific basis identity stays explicit.
    result, pullback = jax.vjp(
        lambda mf: prepare.prepare_system_inputs(
            mf, [0], [1], boys_reference=boys, virtual_anchor_columns=anchors,
            correlation_auxbasis=None if reuse else 'weigend')['B'], reference)
    assert result.shape[1:] == (1, 1)
    pullback(np.ones(result.shape))


def test_high_level_identical_basis_gradient_uses_cache(reference, monkeypatch):
    from pyscfad import dlno_stc
    from pyscfad.dlno_stc.test.test_backend import native
    from pyscfad.dlno_stc.test.test_correlation_auxbasis import _controls
    native()
    def factory(mol):
        mf = scf.RHF(mol).density_fit(auxbasis='weigend')
        mf.conv_tol = 1e-12
        mf.kernel()
        return mf
    options = dict(scope='system', frozen=0, controls=_controls(), include_hf=True)
    expected, expected_bar = dlno_stc.value_and_grad(reference.mol, factory, **options)
    def unexpected(*args, **kwargs):
        raise AssertionError('high-level same-basis gradient rebuilt local factors')
    monkeypatch.setattr(prepare.lno_df, 'get_local_Lov', unexpected)
    actual, actual_bar = dlno_stc.value_and_grad(
        reference.mol, factory, correlation_auxbasis='weigend', **options)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual_bar.coords, expected_bar.coords, atol=1e-10, rtol=0)
