"""Memory admission must survive cold caches, DIIS growth and tight budgets."""
from types import SimpleNamespace
import numpy as np
import pytest
from pyscfad.cc import _native_ccsd_forward as forward
from pyscfad.cc import _native_ccsd_memory as memory
from pyscfad.cc import _native_ccsd_solver as solver
from .test_native_residual import make_case

cpp=forward._ccsd_kernels
pytestmark=pytest.mark.skipif(cpp is None or not hasattr(cpp,'ladder_memory'),
                             reason='rebuild CCSD memory metadata')


def test_fragment36_fits_ten_gb_after_history_fills(monkeypatch):
    # Actual settled state from the fragment36 memory audit. Shapes/metadata
    # only: this regression test must not allocate multi-GB tensors.
    cc=SimpleNamespace(max_memory=10000,iterative_damping=1.)
    a=SimpleNamespace(shape=(36,152));b=SimpleNamespace(nbytes=239542272)
    er=SimpleNamespace(vvvv=SimpleNamespace(flags=SimpleNamespace(c_contiguous=True)))
    state=dict(populated_bytes=2266036112,scratch_capacity_bytes=718627008)
    monkeypatch.setattr(forward.lib,'current_memory',lambda:(7310.459,0))
    assert forward._plan_for(cc,a,b,er,resident=True,existing=state)[1]==152
    # Already resident capacity is not charged a second time. Conversely,
    # allocated-but-unpopulated storage must not receive the same RSS credit.
    cold=dict(state,populated_bytes=1068324560)
    ladder=cpp.ladder_memory(36,152)
    warm_bound=memory.residual_memory(36,152,152,ladder,resident=True,existing=state)
    cold_bound=memory.residual_memory(36,152,152,ladder,resident=True,existing=cold)
    cc.max_memory=7310.459+(warm_bound['incremental_bytes']+cold_bound['incremental_bytes'])/2e6
    assert forward._plan_for(cc,a,b,er,resident=True,existing=state)[1]==152
    plan=forward._plan_for(cc,a,b,er,resident=True,existing=cold)
    assert plan is None or plan[1]<152
    reserve=cold_bound['incremental_bytes']-warm_bound['incremental_bytes']
    assert forward._plan_for(cc,a,b,er,resident=True,existing=state,
                             reserve_bytes=reserve) is None or \
        forward._plan_for(cc,a,b,er,resident=True,existing=state,
                         reserve_bytes=reserve)[1]<152


def test_workspace_population_and_reuse():
    cc,er,a,b=make_case(3,7)
    ws=forward._resident_workspace(cc,a,b,er)
    cold=ws.memory_info()
    assert not cold['amplitudes_populated']
    assert cold['allocated_bytes']-cold['populated_bytes']==2*b.nbytes
    forward._native_update(cc,a,b,er,forward._native_plan(cc,a,b,er))
    warm=ws.memory_info()
    assert warm['amplitudes_populated']
    assert 0<warm['scratch_populated_bytes']<=warm['scratch_capacity_bytes']
    assert warm['populated_bytes']==(warm['integral_bytes']+warm['amplitude_bytes']+
                                     warm['index_bytes']+warm['scratch_populated_bytes'])
    assert warm['populated_bytes']<=warm['allocated_bytes']
    forward._native_update(cc,a,b,er,forward._native_plan(cc,a,b,er))
    assert ws.memory_info()==warm


def test_tight_budget_selects_partial_block_and_preserves_equations(monkeypatch):
    cc,er,a,b=make_case(4,19,.1)
    expected=forward._reference_update_amps(cc,a,b,er)
    ladder=cpp.ladder_memory(4,19)
    low=memory.residual_memory(4,19,1,ladder)['incremental_bytes']
    high=memory.residual_memory(4,19,19,ladder)['incremental_bytes']
    assert high>low
    monkeypatch.setattr(forward.lib,'current_memory',lambda:(100.,0))
    monkeypatch.setattr(forward,'_resident_workspace',lambda *args:None)
    cc.max_memory=100.+(low+high)/2e6
    plan=forward._native_plan(cc,a,b,er)
    assert plan is not None and 1<=plan[1]<19
    result=forward._native_update(cc,a,b,er,plan,profile=True)
    for actual,reference in zip(result[:2],expected):
        np.testing.assert_allclose(actual,reference,atol=2e-13,rtol=1e-12)


def test_fallback_releases_unused_workspace(monkeypatch):
    cc,er,a,b=make_case()
    assert forward._resident_workspace(cc,a,b,er) is not None
    cc.max_memory=0
    def reference(*args):
        assert er._native_residual_workspace is None
        return 'fallback'
    monkeypatch.setattr(forward,'_reference_update_amps',reference)
    assert forward.update_amps(cc,a,b,er)=='fallback'


def test_solver_reserves_only_unfilled_diis_and_restores_policy(monkeypatch):
    cc,er,a,b=make_case()
    update=cc.update_amps;seen=[];size=a.nbytes+b.nbytes
    def checked(*args):
        seen.append(cc._native_ccsd_future_bytes)
        return update(*args)
    monkeypatch.setattr(cc,'update_amps',checked)
    solver.kernel(cc,er,a,b,max_cycle=6,tol=-1,tolnormt=-1)
    assert seen==[7*size,6*size,4*size,2*size,0,0]
    assert not hasattr(cc,'_native_ccsd_future_bytes')
    cc._native_ccsd_future_bytes=123
    with pytest.raises(RuntimeError):
        with solver._memory_reservation(cc,456):
            raise RuntimeError('restore after failure')
    assert cc._native_ccsd_future_bytes==123


@pytest.mark.parametrize('o,v,block',[(1,1,1),(9,2,1),(3,19,7),(36,152,152),(60,500,500)])
def test_shared_arena_and_contiguous_vvvv_copy(o,v,block):
    ladder=cpp.ladder_memory(o,v)
    info=memory.residual_memory(o,v,block,ladder,resident=True)
    assert info['peak_bytes']==max(info['phase_bytes'].values())
    arena=list(info['arena_bytes'].values())
    assert arena==sorted(arena)
    copied=memory.residual_memory(o,v,block,ladder,resident=True,vvvv_copy_bytes=123456)
    assert copied['peak_bytes']==info['peak_bytes']+123456
    # A resident arena can be larger than the next call's requirements; its
    # excess capacity remains live, and is credited only if populated.
    extra=10*max(arena)
    retained=memory.residual_memory(o,v,block,ladder,resident=True,
        existing=dict(scratch_capacity_bytes=extra,populated_bytes=extra))
    assert all(x==extra for x in retained['arena_bytes'].values())
    assert retained['incremental_bytes']>=0


def test_memory_bound_tracks_short_lifetimes_and_older_extensions():
    ladder=cpp.ladder_memory(36,152)
    current=memory.residual_memory(36,152,152,ladder,resident=True)
    legacy=memory.residual_memory(36,152,152,ladder,resident=True,short_lifetimes=False)
    assert current['peak_bytes']<legacy['peak_bytes']-1_000_000_000
    assert current['arena_bytes']==legacy['arena_bytes']
