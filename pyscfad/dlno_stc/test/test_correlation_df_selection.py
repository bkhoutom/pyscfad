"""Fitting choices remain consistent through selection and saved metadata."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import h5py
import pytest

from pyscfad.dlno_stc import _restart, ccsd
from pyscfad.dlno_stc._df_basis import basis_identity, validate_selection_basis
from pyscfad.dlno_stc.test.test_domain_grouping import grouping_fixture


def _legacy_node(node):
    """The pre-override schema has no fitting field at either selection level."""
    node = deepcopy(node)
    def strip(value):
        if isinstance(value, dict):
            if value.get('kind') == 'dataclass':
                value['fields'] = [pair for pair in value['fields']
                                   if pair[0] != 'correlation_basis_identity']
            for item in value.values():
                strip(item)
        elif isinstance(value, list):
            for item in value:
                strip(item)
    strip(node)
    return node


@pytest.mark.parametrize('cc', [False, True])
def test_default_selections_keep_legacy_checkpoint_and_digest(tmp_path, cc):
    static = grouping_fixture()
    if cc:
        static = ccsd.CCSDSelections(static, ())
    encoded = _restart._encode_value(static)
    assert encoded == _legacy_node(encoded)
    with h5py.File(tmp_path / 'static.h5', 'w') as handle:
        encoded = _restart._encode_value(static, _restart._DatasetWriter(handle))
        restored = _restart._decode_value(_legacy_node(encoded), _restart._DatasetReader(handle))
        assert restored.correlation_basis_identity is None
        assert _restart.scientific_digest(restored) == _restart.scientific_digest(static)
        broken = deepcopy(encoded)
        broken['fields'].pop(0)
        with pytest.raises(_restart.RestartCorruptionError):
            _restart._decode_value(broken, _restart._DatasetReader(handle))


def test_saved_fit_is_checked_without_binding_multipole_domains_to_a_fit():
    mf = SimpleNamespace(with_df=SimpleNamespace(auxbasis='cc-pvdz-jkfit'))
    domains = grouping_fixture()
    validate_selection_basis(mf, domains, 'cc-pvdz-ri')
    exact = replace(domains, pair_energy_model='exact')
    for static in (exact, ccsd.CCSDSelections(domains, ())):
        validate_selection_basis(mf, static, None)
        validate_selection_basis(mf, static, 'cc-pvdz-jkfit')
        with pytest.raises(ValueError, match='correlation_auxbasis'):
            validate_selection_basis(mf, static, 'cc-pvdz-ri')
        saved = replace(static, correlation_basis_identity=basis_identity(mf, 'cc-pvdz-ri'))
        validate_selection_basis(mf, saved, 'cc-pvdz-ri')
        with pytest.raises(ValueError, match='correlation_auxbasis'):
            validate_selection_basis(mf, saved, None)
        assert _restart.scientific_digest(saved) != _restart.scientific_digest(static)


@pytest.fixture(scope='module')
def selection_water_reference():
    from pyscfad import config_update, gto, scf
    with config_update('pyscfad_moleintor_opt', True):
        mol = gto.Mole(atom='O 0 0 0; H .1 0 1; H 0 1 .1',
                       basis='sto-3g', verbose=0, max_memory=2000)
        mol.build(trace_exp=False, trace_ctr_coeff=False)
        mf = scf.RHF(mol).density_fit(auxbasis='weigend')
        mf.conv_tol = 1e-11
        mf.kernel()
        yield mf


def test_exact_pair_screening_matches_independent_pyscf_ri_oracle(selection_water_reference):
    import numpy as np
    from pyscf import df as pyscf_df
    from pyscfad.dlno_stc import _pairs, build_domain_selections, domain

    mf = selection_water_reference
    jk = mf.with_df
    jk_source, jk_auxmol = jk._cderi, jk.auxmol
    static = build_domain_selections(mf, frozen=1, pair_energy_model='all')
    common = domain.rebuild_domain_data(mf, static)
    basis = 'cc-pvdz-ri'
    pymol = mf.mol.to_pyscf()
    auxmol = pyscf_df.addons.make_auxmol(pymol, basis)
    assert auxmol.nao != jk_auxmol.nao
    # The oracle transforms dense PySCF AO factors and all canonical pairs at
    # once, independently of the production local-direct/tiled pair route.
    factors = pyscf_df.incore.cholesky_eri(pymol, auxmol=auxmol, aosym='s1')
    co = np.asarray(mf.mo_coeff)[:, static.active_occ_indices]
    cv = np.asarray(mf.mo_coeff)[:, static.active_vir_indices]
    lov = np.einsum('Puv,ui,va->Pia', factors.reshape(auxmol.nao, pymol.nao, pymol.nao), co, cv)
    g = np.einsum('Pia,Pjb->iajb', lov, lov)
    energies = np.asarray(mf.mo_energy)
    gaps = energies[static.active_occ_indices, None] - energies[None, static.active_vir_indices]
    t = g / (gaps[:, :, None, None] + gaps[None, None, :, :])
    labels = [int(indices[0]) for indices in static.frag_lolist]
    rotation = (co.T @ np.asarray(common.s1e) @ np.asarray(common.iao_coeff))[:, labels]
    target_g = np.einsum('ip,iajb,jq->paqb', rotation, g, rotation)
    target_t = np.einsum('ip,iajb,jq->paqb', rotation, t, rotation)
    directed = 4 * np.einsum('paqb,paqb->pq', target_g, target_t)
    expected = directed + directed.T
    np.fill_diagonal(expected, np.diag(directed))
    actual = _pairs._exact_boys_pair_scores(mf, common, static, correlation_auxbasis=basis)
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-11)
    jk_scores = _pairs._exact_boys_pair_scores(mf, common, static)
    assert np.max(np.abs(actual - jk_scores)) > 1e-8
    assert mf.with_df is jk and jk._cderi is jk_source and jk.auxmol is jk_auxmol
    assert jk.auxbasis == 'weigend'


def test_public_selection_callback_inherits_ri_on_detached_reference(
        selection_water_reference, monkeypatch):
    from pyscfad import dlno_stc
    from pyscfad.dlno_stc import driver

    mf = selection_water_reference
    references = []
    def choose(reference):
        references.append(reference)
        assert reference is not mf
        assert reference._stc_correlation_auxbasis == 'cc-pvdz-ri'
        # Invoke the real builder without repeating the keyword. Full domains
        # keep this focused on selection routing, with no correlation reverse.
        return dlno_stc.build_domain_selections(reference, frozen=1,
            pair_energy_model='exact', force_full_domains=True)
    def initialize(current, selected, controls, **options):
        assert current is mf and options['correlation_auxbasis'] == 'cc-pvdz-ri'
        return selected
    monkeypatch.setattr(driver, '_initialize_scf', lambda *args, **kwargs: (mf, lambda bar: (bar,)))
    monkeypatch.setattr(driver, '_initialize', initialize)
    monkeypatch.setattr(driver, '_run_domain', lambda initialize, *args, **kwargs: initialize())
    selected = dlno_stc.value_and_grad(mf.mol, lambda mol: mf, controls={},
        build_static=choose, correlation_auxbasis='cc-pvdz-ri')
    assert selected.correlation_basis_identity == basis_identity(mf, 'cc-pvdz-ri')
    assert len(references) == 1
    for reference in (mf, *references):
        assert not hasattr(reference, '_stc_correlation_auxbasis')
        assert not hasattr(reference, '_stc_scf_data')
        assert not hasattr(reference, '_stc_selection_response')
    assert mf.with_df.auxbasis == 'weigend'


def test_real_standalone_cc_selection_records_only_explicit_ri_identity(selection_water_reference):
    from pyscfad.dlno_stc import build_domain_selections

    mf = selection_water_reference
    domains = build_domain_selections(mf, frozen=1, pair_energy_model='all')
    options = dict(matrix_kind='W', thresh_occ=1e-5, thresh_vir=1e-5,
                   baseline_controls={}, matrix_controls={})
    default = ccsd.build_static_selections(mf, domains, **options)
    explicit = ccsd.build_static_selections(mf, domains,
        correlation_auxbasis='cc-pvdz-ri', **options)
    assert default.correlation_basis_identity is None
    assert domains.correlation_basis_identity is None
    assert explicit.correlation_basis_identity == basis_identity(mf, 'cc-pvdz-ri')
    assert explicit.correlation_basis_identity != basis_identity(mf, None)
    assert default.deferred_selection == explicit.deferred_selection
    assert _restart.scientific_digest(default) != _restart.scientific_digest(explicit)
    assert not hasattr(mf, '_stc_correlation_auxbasis')
    assert mf.with_df.auxbasis == 'weigend'
