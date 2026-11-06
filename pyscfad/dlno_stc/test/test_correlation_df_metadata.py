"""Correlation fitting identity stays separate from the SCF fitting source."""
from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import df as pyscf_df
from pyscfad import gto
from pyscfad.dlno_stc import _mpi, _workflow
from pyscfad.dlno_stc._restart import RestartManager, RestartMismatchError


@pytest.fixture
def reference():
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 1.4'
    mol.basis = 'sto-3g'
    mol.verbose = 0
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    auxbasis = 'weigend'
    with_df = SimpleNamespace(mol=mol, auxbasis=auxbasis,
        auxmol=pyscf_df.addons.make_auxmol(mol.to_pyscf(), auxbasis),
        max_memory=10000., _cderi=np.ones((2, 3)))
    return SimpleNamespace(mol=mol, with_df=with_df, max_memory=10000.,
        mo_occ=np.array([2., 0.]), mo_coeff=np.eye(2), mo_energy=np.array([-1., 1.]))


def test_restart_rejects_changed_correlation_fit_with_identical_jk(reference, tmp_path):
    # Omitting correlation identity would silently reuse bars from another RI fit.
    payload = _workflow.run_payload(reference, 0, driver='test',
                                    correlation_auxbasis='weigend')
    manager = RestartManager(tmp_path, method='test', scientific_payload=payload)
    manager.close()
    changed = _workflow.run_payload(reference, 0, driver='test',
                                    correlation_auxbasis='def2-svp-jkfit')
    assert payload['df_source'] == changed['df_source']
    assert payload['auxbasis'] == changed['auxbasis'] == 'weigend'
    with pytest.raises(RestartMismatchError):
        RestartManager(tmp_path, resume=True, method='test', scientific_payload=changed)


@pytest.mark.parametrize('guard', ['packet', 'domain', 'system'])
def test_memory_uses_correlation_dimension_without_replacing_jk(reference, monkeypatch, guard):
    from pyscfad.lno import _df_direct
    # Bypass unrelated native scratch forecasts; retain real molecule/basis sizes.
    monkeypatch.setattr(_df_direct, '_local_direct_workspace_mb', lambda *a, **k: 0.)
    monkeypatch.setattr(_workflow, '_native_workspace_mb', lambda *a, **k: 0.)
    fragment = SimpleNamespace(strong_occ_metric_keep=(0,),
        strong_virtual=SimpleNamespace(metric_keep=(0,)), extended_atoms=(0, 1))
    static = SimpleNamespace(fragments=(fragment,))
    jk_auxmol = reference.with_df.auxmol
    corr_auxmol = pyscf_df.addons.make_auxmol(reference.mol.to_pyscf(), 'def2-svp-jkfit')
    assert corr_auxmol.nao != jk_auxmol.nao
    def estimate(**kwargs):
        if guard == 'packet':
            return _workflow.check_memory(reference, 1, 1, **kwargs)
        if guard == 'domain':
            return _workflow.check_domain_memory(reference, static, 0, with_grad=False, **kwargs)
        return _workflow.check_system_memory(reference, 1, 1, with_grad=False, **kwargs)
    inherited = estimate()
    explicit = estimate(correlation_auxbasis='def2-svp-jkfit')
    assert (explicit > inherited) == (corr_auxmol.nao > jk_auxmol.nao)
    if guard == 'packet':
        assert explicit == 64 * corr_auxmol.nao / 1e6
    assert reference.with_df.auxmol is jk_auxmol
    assert reference.with_df.auxbasis == 'weigend'


def test_mpi_correlation_config_is_installed_and_mismatch_rejected(reference):
    # The service signature must continue to describe only the SCF/JK fit.
    signature = _mpi._reference_signature(reference)
    config = {'correlation_auxbasis': {'H': 'def2-svp-jkfit'}}
    state = {'mf': reference}
    _mpi._install_correlation_config(state, config)
    assert state['correlation_auxbasis'] == {'H': 'def2-svp-jkfit'}
    assert _mpi._reference_signature(reference) == signature
    with pytest.raises(ValueError, match='correlation auxiliary basis'):
        _mpi._install_correlation_config(state, {'correlation_auxbasis': 'weigend'})


def test_packet_replay_rejects_changed_correlation_fit(reference):
    from pyscfad.dlno_stc import mp2
    from pyscfad.dlno_stc._selection import build_domain_groups
    from pyscfad.dlno_stc.test.test_domain_grouping import grouping_fixture
    static = grouping_fixture()
    groups = build_domain_groups(static)
    group = groups[0]
    budget = dict(total_tolerance=None, original_target_count=len(static.fragments),
                  group_count=len(groups), effective_tolerance=None)
    inputs = {'foo': np.zeros((2, 2)), 'fvv': np.zeros((1, 1)), 'B': np.zeros((3, 2, 1))}
    metadata = mp2._packet_metadata(reference, static, 0, group, inputs, groups, budget,
                                   correlation_auxbasis='def2-svp-jkfit')
    assert metadata['auxiliary_order']['auxbasis'] == 'def2-svp-jkfit'
    assert metadata['correlation_auxbasis'] == 'def2-svp-jkfit'
    mp2._validate_packet_identity(reference, static, 0, group, inputs, metadata, {}, groups,
                                  correlation_auxbasis='def2-svp-jkfit')
    with pytest.raises(ValueError, match='auxiliary_order|molecular_fingerprint|correlation_auxbasis'):
        mp2._validate_packet_identity(reference, static, 0, group, inputs, metadata, {}, groups,
                                      correlation_auxbasis='weigend')


def test_cc_mpi_owners_use_root_correlation_setting(reference, monkeypatch):
    import concurrent.futures
    from typing import NamedTuple
    from pyscfad.dlno_stc import _mpi_ccsd, ccsd
    from pyscfad.dlno_stc._selection import DomainGroup
    from pyscfad.dlno_stc.test.test_mpi_core import _World
    from pyscfad.lib import _compilation_cache

    class Common(NamedTuple):
        s1e: object

    reference.e_tot = np.array(-1.)
    common = Common(np.eye(2))
    static = SimpleNamespace(mp2_static=object())
    groups = (DomainGroup((0,), 0, (0,)), DomainGroup((1,), 1, (0,)))
    def initialize():
        state = dict(mf=reference, common=common, weak_pairs=(), mf_bar=None, common_bar=None,
                     cc_fragment_indices=(0, 1), cc_triples_controls=None,
                     correlation_auxbasis={'H': 'def2-svp-jkfit'})
        return state, static, {}, {}

    # Isolate scheduling from molecular response and native CC work; run real
    # broadcasts, root metadata construction, config installation and reduction.
    monkeypatch.setattr(_mpi_ccsd, 'build_domain_groups', lambda *a, **k: groups)
    monkeypatch.setattr(_mpi_ccsd, '_reference_signature', lambda mf: {'auxbasis': 'weigend'})
    monkeypatch.setattr(_mpi_ccsd, '_build_cc_worker', lambda *a: reference)
    monkeypatch.setattr(_mpi_ccsd, '_validate_shared_reference', lambda *a, **k: None)
    monkeypatch.setattr(_compilation_cache, 'configure_compilation_cache', lambda: None)
    monkeypatch.setattr(ccsd, '_clear_jax_caches', lambda *a, **k: None)
    monkeypatch.setattr(ccsd, '_run_weak', lambda *a, **k: None)
    monkeypatch.setattr(ccsd, '_finish', lambda state, *a, **k: float(state['energy']))
    seen = {}
    def run_group(state, static, group, *args, **kwargs):
        seen[group.representative_index] = state['correlation_auxbasis']
        state['energy'] += group.representative_index + 1
        return []
    monkeypatch.setattr(ccsd, '_run_group', run_group)
    world = _World(2)
    def owner(rank):
        return _mpi_ccsd._run_jobs(initialize, None, world.communicator(rank), with_grad=False,
            include_hf=False, merge_identical=True, verbose_imp=0, ccsd_t=False,
            return_components=False, include_baseline=True, fragment_indices=None,
            scf_service=None, vvvv_direct=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(owner, range(2)))
    assert result == [3., None]
    assert seen == {0: {'H': 'def2-svp-jkfit'}, 1: {'H': 'def2-svp-jkfit'}}
    assert reference.with_df.auxbasis == 'weigend'
