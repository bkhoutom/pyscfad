"""Memory forecasts warn; only verified impossible storage stops STC."""
from types import SimpleNamespace
import warnings

import numpy as np
import pytest


def observation(**kwargs):
    from pyscfad.dlno_stc._memory import MemoryObservation
    return MemoryObservation(**dict(rss_mb=50., **kwargs))


def test_soft_budget_and_over_limit_forecast_only_warn():
    from pyscfad.dlno_stc._memory import assess_memory
    info = observation(enforced_limit_mb=100., enforced_source='cgroup v2 /job',
                       headroom_mb=1., allowance_mb=100., allowance_source='cgroup v2 /job',
                       rank=3)
    with pytest.warns(RuntimeWarning) as caught:
        assess_memory('domain', 200., configured_mb=10., observation=info)
    text = str(caught[0].message)
    for field in ('domain', 'rank=3', 'estimate=200.0', 'RSS=50.0',
                  'configured=10.0', 'limit=100.0', 'headroom=1.0', 'cgroup v2'):
        assert field in text


def test_only_unavoidable_storage_over_verified_limit_stops():
    from pyscfad.dlno_stc._memory import assess_memory
    info = observation(enforced_limit_mb=1., enforced_source='cgroup v1 /job')
    with pytest.raises(MemoryError, match='unavoidable'):
        assess_memory('system', 200., unavoidable_bytes=1_000_001, observation=info)
    with warnings.catch_warnings(record=True):
        assess_memory('system', 200., unavoidable_bytes=1_000_000, observation=info)


def test_missing_limits_never_prove_failure():
    from pyscfad.dlno_stc._memory import assess_memory
    with pytest.warns(RuntimeWarning, match='unknown'):
        assess_memory('packet', 100., configured_mb=1., unavoidable_bytes=10**15,
                      observation=observation())


def test_reclaimable_usage_and_host_headroom_do_not_stop():
    from pyscfad.dlno_stc._memory import assess_memory
    info = observation(enforced_limit_mb=100., headroom_mb=0., host_available_mb=0.,
                       allowance_mb=100.)
    with pytest.warns(RuntimeWarning):
        assess_memory('domain', 1000., unavoidable_bytes=10_000_000, observation=info)


def test_peak_is_not_charged_rss_twice():
    from pyscfad.dlno_stc._memory import assess_memory
    info = observation(allowance_mb=100.)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assess_memory('matrix', 80., estimate_kind='peak', observation=info)
    assert not caught
    with pytest.warns(RuntimeWarning, match='peak=130.0'):
        assess_memory('matrix', 80., estimate_kind='workspace', observation=info)


def test_warning_lifetime_is_one_stage_rank_per_calculation(monkeypatch):
    from pyscfad.dlno_stc import _memory
    monkeypatch.setattr(_memory, 'observe_memory', lambda: observation())

    @_memory.memory_calculation
    def calculation(*, comm=None):
        _memory.assess_memory('domain', 100., configured_mb=1.)
        _memory.assess_memory('domain', 200., configured_mb=1.)
        _memory.assess_memory('system', 100., configured_mb=1.)

    comm = SimpleNamespace(Get_rank=lambda: 7)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        calculation(comm=comm)
        calculation(comm=comm)
    assert len(caught) == 4
    assert all('rank=7' in str(item.message) for item in caught)


def proc_files(tmp_path, *, version=2, root='/allocation', group='/allocation/job/rank'):
    proc = tmp_path / 'proc'
    (proc / 'self').mkdir(parents=True)
    (proc / 'self/status').write_text('VmRSS:\t50000 kB\n')
    (proc / 'meminfo').write_text('MemTotal: 8000000 kB\nMemAvailable: 2000000 kB\n')
    mount = tmp_path / 'memory cgroup'
    leaf = mount / 'job/rank'
    leaf.mkdir(parents=True)
    escaped = str(mount).replace(' ', r'\040')
    kind = 'cgroup2 cgroup rw' if version == 2 else 'cgroup cgroup rw,memory'
    (proc / 'self/mountinfo').write_text(f'42 1 0:1 {root} {escaped} rw - {kind}\n')
    (proc / 'self/cgroup').write_text(f'0::{group}\n' if version == 2
                                     else f'5:cpu,memory:{group}\n')
    return proc, mount, leaf


@pytest.mark.parametrize('version', [1, 2])
def test_actual_cgroup_and_tighter_ancestor_are_used(tmp_path, version):
    from pyscfad.dlno_stc._memory import observe_memory
    proc, mount, leaf = proc_files(tmp_path, version=version)
    limit, usage = (('memory.max', 'memory.current') if version == 2 else
                    ('memory.limit_in_bytes', 'memory.usage_in_bytes'))
    (mount / limit).write_text('800000000')
    (mount / usage).write_text('700000000')
    (leaf.parent / limit).write_text('400000000')
    (leaf.parent / usage).write_text('399000000')
    (leaf / limit).write_text('max' if version == 2 else str(2**63 - 4096))
    for path in (mount, leaf.parent, leaf):
        if version == 2:
            (path / 'memory.swap.max').write_text('0')
        else:
            (path / 'memory.use_hierarchy').write_text('1')
            (path / 'memory.memsw.limit_in_bytes').write_text((path / limit).read_text())
    info = observe_memory(proc_root=proc, environ={})
    assert info.rss_mb == 51.2
    assert info.enforced_limit_mb == 400.
    assert info.headroom_mb == 1.
    assert info.host_available_mb == 2048.
    assert str(leaf.parent) in info.enforced_source


def test_unreadable_observations_are_unknown(tmp_path):
    from pyscfad.dlno_stc._memory import observe_memory
    info = observe_memory(proc_root=tmp_path, environ={})
    assert info.rss_mb is None
    assert info.enforced_limit_mb is None
    assert info.headroom_mb is None
    assert info.host_available_mb is None


@pytest.mark.parametrize('extra,expected', [
    ({'OMPI_COMM_WORLD_LOCAL_SIZE': '4', 'OMPI_COMM_WORLD_SIZE': '64'}, 419.4304),
    ({'MPI_LOCALNRANKS': '8', 'PMI_SIZE': '64'}, 209.7152),
    ({'SLURM_TASKS_PER_NODE': '2(x2),4', 'SLURM_NODEID': '2'}, 419.4304),
])
def test_slurm_uses_local_ranks_not_global_size(tmp_path, extra, expected):
    from pyscfad.dlno_stc._memory import observe_memory
    info = observe_memory(proc_root=tmp_path,
                          environ=dict(SLURM_MEM_PER_NODE='1600', **extra))
    assert info.allowance_mb == expected
    assert 'Slurm' in info.allowance_source
    assert info.enforced_limit_mb is None


def test_slurm_cpu_memory_and_uncertain_rank_mapping(tmp_path):
    from pyscfad.dlno_stc._memory import observe_memory
    env = dict(SLURM_MEM_PER_CPU='100', SLURM_CPUS_ON_NODE='8',
               OMPI_COMM_WORLD_LOCAL_SIZE='4')
    assert observe_memory(proc_root=tmp_path, environ=env).allowance_mb == 209.7152
    env.pop('OMPI_COMM_WORLD_LOCAL_SIZE')
    assert observe_memory(proc_root=tmp_path, environ=env).allowance_mb is None


@pytest.fixture
def workflow_mf(monkeypatch):
    from pyscfad.dlno_stc import _memory, _workflow
    from pyscfad.lno import df as lno_df
    from pyscfad.lno import _df_direct
    mol = SimpleNamespace(max_memory=1., nao=4, to_pyscf=lambda: None)
    aux = SimpleNamespace(nao=10)
    mf = SimpleNamespace(mol=mol, max_memory=1., with_df=SimpleNamespace(auxmol=aux, auxbasis=None),
                         mo_coeff=np.zeros((4, 4)), mo_occ=np.ones(4), mo_energy=np.ones(4))
    fragment = SimpleNamespace(strong_occ_metric_keep=(0, 1),
                               strong_virtual=SimpleNamespace(metric_keep=(0, 1, 2)),
                               extended_atoms=(0,))
    static = SimpleNamespace(fragments=(fragment,))
    monkeypatch.setattr(_workflow.pyscf_df.addons, 'make_auxmol', lambda *a: aux)
    monkeypatch.setattr(lno_df, 'make_local_mol', lambda *a: mol)
    monkeypatch.setattr(_df_direct, '_local_direct_int3c_block_mb', lambda: 0.)
    monkeypatch.setattr(_workflow, '_native_workspace_mb', lambda *a, **k: 2.)
    monkeypatch.setattr(_memory, 'observe_memory', lambda: observation())
    return mf, static


@pytest.mark.parametrize('guard', ['packet', 'domain', 'system'])
def test_workflow_guards_continue_over_soft_budget(workflow_mf, guard):
    from pyscfad.dlno_stc import _workflow
    mf, static = workflow_mf
    mf.max_memory = 1e-6
    with pytest.warns(RuntimeWarning):
        if guard == 'packet':
            result = _workflow.check_memory(mf, 2, 3)
        elif guard == 'domain':
            result = _workflow.check_domain_memory(mf, static, 0, with_grad=True)
        else:
            result = _workflow.check_system_memory(mf, 2, 3, with_grad=True)
    assert result > mf.max_memory


@pytest.mark.parametrize('guard', ['domain', 'system'])
def test_workflow_known_new_factor_larger_than_cgroup_stops(workflow_mf, monkeypatch, guard):
    from pyscfad.dlno_stc import _memory, _workflow
    mf, static = workflow_mf
    monkeypatch.setattr(_memory, 'observe_memory', lambda: observation(enforced_limit_mb=.0004))
    # One required float64 B[10,2,3] occupies 480 bytes, above the 400-byte limit.
    with pytest.raises(MemoryError, match='unavoidable'):
        if guard == 'domain':
            _workflow.check_domain_memory(mf, static, 0, with_grad=True)
        else:
            _workflow.check_system_memory(mf, 2, 3, with_grad=True)


def test_system_resident_arrays_are_not_added_twice(workflow_mf, monkeypatch):
    from pyscfad.dlno_stc import _memory, _workflow
    mf, _ = workflow_mf
    monkeypatch.setattr(_memory, 'observe_memory', lambda: observation())
    with pytest.warns(RuntimeWarning) as caught:
        estimate = _workflow.check_system_memory(mf, 2, 3, with_grad=False)
    # Existing mo_coeff/occ/energy occupy 192 bytes; the estimator already counts them.
    assert f'peak={estimate + 50. - .000192:.1f}' in str(caught[0].message)


def test_system_original_estimate_still_triggers_configured_warning(workflow_mf, monkeypatch):
    from pyscfad.dlno_stc import _memory, _workflow
    mf, _ = workflow_mf
    mf.max_memory = 1e9
    monkeypatch.setattr(_memory, 'observe_memory', lambda: observation())
    estimate = _workflow.check_system_memory(mf, 2, 3, with_grad=False)
    # The budget lies between the original estimate and normalized new workspace.
    mf.max_memory = estimate - .0001
    with pytest.warns(RuntimeWarning):
        _workflow.check_system_memory(mf, 2, 3, with_grad=False)


def test_unknown_rss_does_not_claim_a_measured_peak():
    from pyscfad.dlno_stc._memory import assess_memory, MemoryObservation
    with pytest.warns(RuntimeWarning, match='peak=unknown'):
        assess_memory('domain', 100., configured_mb=1.,
                      observation=MemoryObservation())


def test_invalid_cgroup_path_cannot_supply_false_limit(tmp_path):
    from pyscfad.dlno_stc._memory import observe_memory
    proc, mount, _ = proc_files(tmp_path, group='/other/job')
    (mount / 'memory.max').write_text('1')
    assert observe_memory(proc_root=proc, environ={}).enforced_limit_mb is None


def test_conflicting_local_rank_mapping_is_advisory_only(tmp_path):
    from pyscfad.dlno_stc._memory import observe_memory
    info = observe_memory(proc_root=tmp_path, environ=dict(SLURM_MEM_PER_NODE='1600',
        OMPI_COMM_WORLD_LOCAL_SIZE='4', SLURM_TASKS_PER_NODE='8'))
    assert info.allowance_mb is None
    assert info.enforced_limit_mb is None


def test_unrelated_subtree_mount_cannot_supply_a_proof_limit(tmp_path):
    from pyscfad.dlno_stc._memory import observe_memory
    proc, mount, _ = proc_files(tmp_path, root='/unrelated', group='/')
    (mount / 'memory.max').write_text('1000000')
    (mount / 'memory.swap.max').write_text('0')
    info = observe_memory(proc_root=proc, environ={})
    assert info.enforced_limit_mb is None
    assert info.allowance_source.startswith('host')


@pytest.mark.parametrize('hierarchy', ['0', ''])
def test_v1_nonhierarchical_or_unknown_ancestor_is_not_enforced(tmp_path, hierarchy):
    from pyscfad.dlno_stc._memory import observe_memory
    proc, _, leaf = proc_files(tmp_path, version=1)
    (leaf / 'memory.limit_in_bytes').write_text('10000000')
    (leaf / 'memory.memsw.limit_in_bytes').write_text('10000000')
    (leaf.parent / 'memory.limit_in_bytes').write_text('1000000')
    (leaf.parent / 'memory.memsw.limit_in_bytes').write_text('1000000')
    if hierarchy:
        (leaf.parent / 'memory.use_hierarchy').write_text(hierarchy)
    info = observe_memory(proc_root=proc, environ={})
    assert info.enforced_limit_mb == 10.
    assert info.allowance_mb == 10.


@pytest.mark.parametrize('swap,proof', [('1000000000', 1100.), ('max', None), ('', None)])
def test_v2_swap_capacity_prevents_false_ram_only_stop(tmp_path, swap, proof):
    from pyscfad.dlno_stc._memory import observe_memory, assess_memory
    proc, _, leaf = proc_files(tmp_path)
    (leaf / 'memory.max').write_text('100000000')
    if swap:
        (leaf / 'memory.swap.max').write_text(swap)
    info = observe_memory(proc_root=proc, environ={})
    assert info.allowance_mb == 100.
    assert info.enforced_limit_mb == proof
    with pytest.warns(RuntimeWarning):
        assess_memory('domain', 200., unavoidable_bytes=200_000_000, observation=info)
    if proof is not None:
        with pytest.raises(MemoryError):
            assess_memory('domain', 1200., unavoidable_bytes=1_100_000_001, observation=info)


def test_v1_ram_limit_without_memsw_capacity_is_advisory(tmp_path):
    from pyscfad.dlno_stc._memory import observe_memory
    proc, _, leaf = proc_files(tmp_path, version=1)
    (leaf / 'memory.limit_in_bytes').write_text('1000000')
    info = observe_memory(proc_root=proc, environ={})
    assert info.allowance_mb == 1.
    assert info.enforced_limit_mb is None


def test_exact_byte_limit_does_not_round_into_an_impossible_array():
    from pyscfad.dlno_stc._memory import assess_memory
    assess_memory('domain', 0., unavoidable_bytes=249,
                  observation=observation(enforced_limit_mb=249 / 1e6))
