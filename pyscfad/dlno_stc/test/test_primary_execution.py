"""Concrete primary execution preserves spaces and restores thread budgets."""
import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from pyscfad import gto
from pyscfad.dlno_stc.test import _weak_reference
from pyscfad.dlno_stc import _weak_numpy
from pyscfad.dlno_stc import _pairs, _primary_response, _selection, driver
from pyscfad.dlno_stc.test.test_boys_prepare import boys_water_dimer
from pyscfad.lib._threading import dense_blas_scope


def pools(api):
    return {p['filepath']:p['num_threads'] for p in threadpool_info() if p['user_api']==api}


@pytest.mark.parametrize('nvir', [0, 1, 3])
def test_reference_fock_matches_live_finishing(nvir):
    occupied=np.eye(4)[:, :1]
    virtual=np.eye(4)[:, 1:1+nvir]
    fock=np.array([[-1., .1, 0., 0.], [.1, .5, .2, .1],
                   [0., .2, 1., .3], [0., .1, .3, 1.5]])
    got=_pairs._finish_weak_screen_reference(occupied, virtual, fock, 0)
    want=_weak_reference._finish_weak_screen(occupied, virtual, fock, 0)
    np.testing.assert_allclose(got.occupied_energy,want.occupied_energy,atol=2e-14)
    np.testing.assert_allclose(got.virtual_energy,want.virtual_energy,atol=2e-14)
    # Compare spectral projectors; eigenvector signs are not physical outputs.
    for i in range(nvir):
        np.testing.assert_allclose(np.outer(got.virtual_coeff[:,i],got.virtual_coeff[:,i]),
                                   np.outer(want.virtual_coeff[:,i],want.virtual_coeff[:,i]),atol=2e-13)
    assert all(isinstance(x,np.ndarray) for x in got)


@pytest.mark.parametrize('occupied_energy,virtual_energy', [(1., .5),(1.,1.),(np.nan,.5),(-1.,np.inf)])
def test_reference_fock_rejects_invalid_gaps(occupied_energy,virtual_energy):
    with pytest.raises((RuntimeError,ValueError),match='denominator|finite|infs|NaNs'):
        _pairs._finish_weak_screen_reference(np.eye(2)[:,:1],np.eye(2)[:,1:],
                                            np.diag([occupied_energy,virtual_energy]),7)


@pytest.mark.parametrize('operation', ['metric','preprojection','finish'])
@pytest.mark.parametrize('failure', [False,True])
def test_primary_blas_limit_preserves_openmp_and_restores(monkeypatch,operation,failure):
    # Observe a real dense eigensolve inside each helper, including its error path.
    if operation=='finish':
        _pairs._finish_weak_screen_reference(np.eye(3)[:,:1],np.eye(3)[:,1:],np.diag([-1.,.5,1.]),0)
    owner=_selection.scipy_linalg
    original=owner.eigh
    seen=[]
    def checked(*a,**k):
        seen.append((pools('blas'),pools('openmp')))
        if failure:
            raise RuntimeError('injected eigensolver failure')
        return original(*a,**k)
    monkeypatch.setattr(owner,'eigh',checked)
    def call():
        if operation=='metric':
            return _selection._reference_metric_space(np.eye(3)[:,:2],np.eye(3),1e-10)
        if operation=='finish':
            return _pairs._finish_weak_screen_reference(np.eye(3)[:,:1],np.eye(3)[:,1:],np.diag([-1.,.5,1.]),0)
        mol=gto.Mole(atom='H 0 0 0; H 0 0 1',basis='sto-3g',verbose=0)
        mol.build(trace_exp=False,trace_ctr_coeff=False)
        return _selection._reference_pao_preprojection(
            mol,np.eye(2),np.arange(2),np.eye(2),parent_atoms=[0,1],support_atoms=[0,1],
            representation_atoms=[0,1],completeness_threshold=.98,overlap_threshold=1e-4)
    with threadpool_limits(limits=2),dense_blas_scope():
        before=pools('blas'),pools('openmp')
        if failure:
            with pytest.raises(RuntimeError,match='injected'):
                call()
        else:
            call()
        assert (pools('blas'),pools('openmp'))==before
    assert seen and all(set(b.values())=={1} and o==before[1] for b,o in seen)


def test_concrete_finishing_does_not_replace_live_response(boys_water_dimer,monkeypatch):
    mf,_,reference=boys_water_dimer
    reference_finish=_pairs._finish_weak_screen_reference
    live_finish=_weak_numpy.target
    calls={'reference':0,'live':0}
    def concrete(*a,**k):
        calls['reference']+=1
        return reference_finish(*a,**k)
    def live(*a,**k):
        calls['live']+=1
        return live_finish(*a,**k)
    monkeypatch.setattr(_pairs,'_finish_weak_screen_reference',concrete)
    monkeypatch.setattr(_weak_numpy,'target',live)
    _selection.build_domain_selections(mf,frozen=reference.frozen,
        target_coeff=reference.target_reference_coeff)
    assert calls['reference']>0 and calls['live']==0
    calls['reference']=0
    driver._select_domain_reference(mf,lambda current:
        _selection.build_domain_selections(current,frozen=reference.frozen))
    assert calls['reference']==0 and calls['live']>0
