"""Global MP2 plus local corrections, with real screening and DF integrals."""
import numpy as np
import pytest

from pyscfad import gto, scf, config_update
from pyscfad.dlno_stc import build_domain_selections, ccsd


def build_mf(mol):
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.chkfile = None
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope='module')
def reference():
    mol = gto.Mole(atom='H 0 0 0; H 0 0 .75; H 0 0 6; H 0 0 6.75',
                   basis='sto-3g', verbose=0, max_memory=6000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = build_mf(mol)
    domains = build_domain_selections(mf, frozen=0, mp2_laplace_points=12)
    assert not np.all(domains.strong_mask)
    return mf, domains


def select(mf, domains, kind='V', **kwargs):
    return ccsd.build_static_selections(mf, domains, thresh_occ=1e-4, thresh_vir=1e-5,
                                      matrix_kind=kind, baseline_kind='global', **kwargs)


def forbidden(*args, **kwargs):
    raise AssertionError('global baseline must not evaluate weak-pair completion')


@pytest.mark.parametrize('kind', ['V', 'D', 'W'])
def test_global_baseline_accounting_and_restart(reference, monkeypatch, tmp_path, kind):
    from pyscfad.mp.dfmp2 import MP2
    from pyscfad.dlno_stc import _global_mp2
    mf, domains = reference
    selected = select(mf, domains, kind)
    original_mask = domains.strong_mask.copy()
    monkeypatch.setattr(ccsd, '_run_weak', forbidden)
    options = dict(return_components=True, mp2_aux_svd_tol=None)
    if kind == 'V':
        options['checkpoint_dir'] = str(tmp_path / 'checkpoint')
    energy, parts = ccsd.kernel(mf, selected, **options)
    expected_mp2 = MP2(mf).kernel(with_t2=False)[0]
    np.testing.assert_allclose(parts.e_mp2_base, expected_mp2, atol=2e-10, rtol=0)
    np.testing.assert_allclose(energy, parts.e_corr, atol=1e-13, rtol=0)
    assert parts.e_mp2_weak == 0. and parts.baseline_kind == 'global'
    assert parts.baseline_standard_error == 0.
    assert len(parts.fragments) == 2
    assert all(row.nocc_lis == 1 for row in parts.fragments)
    np.testing.assert_array_equal(domains.strong_mask, original_mask)
    if kind == 'V':
        assert all(row.matrix_energy_diagnostic is None for row in parts.fragments)
        saved = ccsd.load_static_selections(options['checkpoint_dir'], mf.mol,
            selection_options=dict(baseline_kind='global', matrix_kind='V'))
        assert saved.baseline_kind == 'global'
        # Restart after completed groups must neither recompute nor double-count
        # the global baseline, and must retain the absent V energy diagnostic.
        monkeypatch.setattr(_global_mp2, 'accumulate', forbidden)
        monkeypatch.setattr(ccsd, '_run_group', forbidden)
        resumed, resumed_parts = ccsd.kernel(mf, selected, resume=True, **options)
        np.testing.assert_allclose(resumed, energy, atol=1e-14)
        np.testing.assert_allclose(resumed_parts.e_mp2_base, parts.e_mp2_base)
        assert all(row.matrix_energy_diagnostic is None for row in resumed_parts.fragments)


def test_local_mp2_cancellation(reference, monkeypatch):
    mf, domains = reference
    original = ccsd._solve_fragment

    def mp2_only(*args, **kwargs):
        emp2, _, _ = original(*args, **kwargs)
        return emp2, emp2, emp2 * 0

    monkeypatch.setattr(ccsd, '_solve_fragment', mp2_only)
    monkeypatch.setattr(ccsd, '_run_weak', forbidden)
    energy, parts = ccsd.kernel(mf, select(mf, domains), return_components=True)
    np.testing.assert_allclose(energy, parts.e_mp2_base, atol=1e-14)


def test_full_lis_limit_matches_canonical_ccsdt(reference):
    from pyscf import scf as pyscf_scf
    from pyscf.cc.dfccsd import RCCSD
    mf, _ = reference
    domains = build_domain_selections(mf, frozen=0, boys_grouping='system', mp2_laplace_points=12)
    selected = ccsd.build_static_selections(mf, domains, thresh_occ=0., thresh_vir=0.,
        matrix_kind='V', baseline_kind='global', triples_laplace_points=12)
    energy = ccsd.kernel(mf, selected, ccsd_t=True, mp2_aux_svd_tol=None)
    host = pyscf_scf.RHF(mf.mol.to_pyscf()).density_fit(auxbasis=mf.with_df.auxbasis)
    for name in ('mo_coeff', 'mo_energy', 'mo_occ', 'e_tot'):
        setattr(host, name, np.asarray(getattr(mf, name)))
    host.converged = True
    cc = RCCSD(host)
    cc.conv_tol = 1e-11
    cc.conv_tol_normt = 1e-9
    ecc = cc.kernel()[0]
    assert cc.converged
    np.testing.assert_allclose(energy, ecc + cc.ccsd_t(), atol=2e-8, rtol=0)


@pytest.mark.parametrize('frozen,auxbasis', [(1, None), (0, 'cc-pvdz-ri')])
def test_global_baseline_respects_active_space_and_fitting_basis(reference, frozen, auxbasis):
    from pyscf import df as pyscf_df
    from pyscfad.dlno_stc import _global_mp2
    mf, _ = reference
    domains = build_domain_selections(mf, frozen=frozen, correlation_auxbasis=auxbasis,
                                     mp2_laplace_points=12)
    selected = select(mf, domains, correlation_auxbasis=auxbasis)
    controls, _ = ccsd._validate_run(selected, None, None)
    state = ccsd.driver._initialize(mf, domains, controls, with_grad=False,
                                    correlation_auxbasis=auxbasis)
    _global_mp2.accumulate(state, selected, controls, with_grad=False)
    mol = mf.mol.to_pyscf()
    aux = pyscf_df.addons.make_auxmol(mol, mf.with_df.auxbasis if auxbasis is None else auxbasis)
    ao = pyscf_df.incore.cholesky_eri(mol, auxmol=aux, aosym='s1').reshape(-1, mol.nao, mol.nao)
    co = np.asarray(mf.mo_coeff)[:, domains.active_occ_indices]
    cv = np.asarray(mf.mo_coeff)[:, domains.active_vir_indices]
    b = np.einsum('Puv,ui,va->Pia', ao, co, cv)
    g = np.einsum('Pia,Pjb->ijab', b, b)
    e = np.asarray(mf.mo_energy)
    gaps = e[domains.active_occ_indices, None] - e[None, domains.active_vir_indices]
    t = g / (gaps[:, None, :, None] + gaps[None, :, None, :])
    expected = np.einsum('ijab,ijab->', t, 2 * g - g.swapaxes(2, 3))
    np.testing.assert_allclose(state['baseline_energy'], expected, atol=2e-10, rtol=0)


def test_global_v_nuclear_response(reference):
    mf, domains = reference
    selected = select(mf, domains)
    options = dict(mp2_aux_svd_tol=None)
    with config_update('pyscfad_scf_implicit_diff', True):
        energy, grad = ccsd.value_and_grad(mf.mol, build_mf, selected, **options)
    h = 2e-4
    direction = np.zeros_like(mf.mol.coords)
    direction[1, 2] = 1.
    values = []
    for sign in (1, -1):
        mol = mf.mol.set_geom_(np.asarray(mf.mol.atom_coords()) + sign * h * direction,
                               unit='Bohr', inplace=False)
        values.append(ccsd.kernel(build_mf(mol), selected, **options))
    np.testing.assert_allclose(np.sum(grad.coords * direction),
                               (values[0] - values[1]) / (2 * h), atol=2e-7, rtol=2e-5)
    assert np.isfinite(energy)
