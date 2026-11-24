"""Dense DIIS must reproduce packed RCCSD, including full-history rollover."""
import types
import numpy as np
import pytest
from pyscf import lib
from pyscf.cc import ccsd
from pyscfad.cc import _native_ccsd_solver as solver
from .test_native_residual import make_case

cpp=solver.cpp
pytestmark=pytest.mark.skipif(cpp is None or not hasattr(cpp,'weighted_dot'),reason='rebuild native solver kernels')


def symmetric(rng,o,v):
    t1=rng.normal(size=(o,v));t2=rng.normal(size=(o,o,v,v))
    return t1,(t2+t2.transpose(1,0,3,2))*.5


@pytest.mark.parametrize('layout',['c','fortran','strided'])
def test_dense_metric_combination_and_energy(layout):
    rng=np.random.default_rng(621)
    o,v=4,19
    pairs=[symmetric(rng,o,v) for _ in range(3)]
    if layout=='fortran':pairs=[tuple(np.asfortranarray(a) for a in pair) for pair in pairs]
    if layout=='strided':pairs=[(a[:,::-1],b[:,:,::-1,::-1]) for a,b in pairs]
    a,b=pairs[:2]
    av,bv=ccsd.amplitudes_to_vector(*a),ccsd.amplitudes_to_vector(*b)
    np.testing.assert_allclose(cpp.weighted_dot(*a,*b),av@bv,atol=2e-11,rtol=2e-13)
    np.testing.assert_allclose(cpp.difference_norm(*a,*b),np.linalg.norm(av-bv),atol=2e-12,rtol=2e-13)
    coeff=np.array([.3,-.2,.9])
    expected=[sum(c*p[i] for c,p in zip(coeff,pairs)) for i in range(2)]
    result=solver.DenseDIIS.combine(coeff,pairs)
    for x,y in zip(result,expected):np.testing.assert_allclose(x,y,atol=2e-13)
    cc,eris,t1,t2=make_case(o,v)
    if layout=='fortran':
        t1,t2=np.asfortranarray(t1),np.asfortranarray(t2)
        eris.fock,eris.ovvo=np.asfortranarray(eris.fock),np.asfortranarray(eris.ovvo)
    np.testing.assert_allclose(solver.energy(cc,t1,t2,eris),ccsd.energy(cc,t1,t2,eris),atol=2e-13,rtol=2e-13)


@pytest.mark.parametrize('space',[2,3,6])
@pytest.mark.parametrize('dependent',[False,True])
def test_dense_diis_matches_packed_history(space,dependent):
    rng=np.random.default_rng(765);o,v=3,7
    dense=solver.DenseDIIS(space);reference=lib.diis.DIIS(incore=True);reference.space=space
    base=symmetric(rng,o,v)
    for step in range(2*space+5):
        a,b=base if dependent else symmetric(rng,o,v)
        x1,x2=a.copy(),b.copy()
        packed=ccsd.amplitudes_to_vector(x1,x2)
        expected=reference.update(packed)
        actual=dense.update(x1,x2)
        np.testing.assert_allclose(ccsd.amplitudes_to_vector(*actual),expected,atol=3e-11,rtol=2e-11)
        # The next call cannot observe changed input buffers in stored history.
        x1.fill(9);x2.fill(8)
    assert dense.count==space


@pytest.mark.parametrize('diis,damping,start,gate',[(True,1.,0,1e9),(True,.7,2,1e9),(False,1.,0,1e9),(True,1.,0,0.)])
def test_dense_solver_matches_iteration_path(diis,damping,start,gate,monkeypatch):
    cc,eris,t1,t2=make_case(3,7,.1)
    cc.diis,cc.iterative_damping,cc.diis_start_cycle,cc.diis_start_energy_diff=diis,damping,start,gate
    assert solver._supported(cc,t1,t2,eris)
    baseline=[];actual=[]
    def record(target):
        return lambda state:target.append((state['t1new'].copy(),state['t2new'].copy()))
    cc.energy=types.MethodType(ccsd.energy,cc)
    expected=ccsd.kernel(cc,eris,t1.copy(),t2.copy(),max_cycle=10,tol=-1,tolnormt=-1,callback=record(baseline))
    cc.energy=types.MethodType(solver.energy,cc)
    monkeypatch.setattr(cc,'amplitudes_to_vector',lambda *a,**kw:pytest.fail('dense solve packed amplitudes'))
    monkeypatch.setattr(cc,'vector_to_amplitudes',lambda *a,**kw:pytest.fail('dense solve unpacked amplitudes'))
    result=solver.kernel(cc,eris,t1.copy(),t2.copy(),max_cycle=10,tol=-1,tolnormt=-1,callback=record(actual))
    assert len(actual)==len(baseline)==10
    for x,y in zip(actual,baseline):
        for a,b in zip(x,y):np.testing.assert_allclose(a,b,atol=2e-11,rtol=2e-10)
    np.testing.assert_allclose(result[1],expected[1],atol=1e-12)
    # Forced iterations past convergence can amplify roundoff through nearly
    # dependent DIIS errors; keep a 1e-9 absolute amplitude check.
    for a,b in zip(result[2:],expected[2:]):np.testing.assert_allclose(a,b,atol=1e-9,rtol=2e-10)


def test_dense_solver_convergence_and_fallback(monkeypatch):
    cc,eris,t1,t2=make_case(3,7)
    cc.energy=types.MethodType(ccsd.energy,cc)
    expected=ccsd.kernel(cc,eris,t1,t2,max_cycle=80,tol=1e-12,tolnormt=1e-10)
    cycles=cc.cycles
    cc.energy=types.MethodType(solver.energy,cc)
    actual=solver.kernel(cc,eris,t1,t2,max_cycle=80,tol=1e-12,tolnormt=1e-10)
    assert expected[0] and actual[0]
    assert cc.cycles==cycles
    np.testing.assert_allclose(actual[1],expected[1],atol=1e-12)
    for key,value in [('cpp_kernels',False),('max_memory',0),('diis_file','unused'),('diis_space',1),('diis',lib.diis.DIIS(incore=True))]:
        with monkeypatch.context() as patch:
            patch.setattr(cc,key,value)
            patch.setattr(ccsd,'kernel',lambda *a,**kw:'fallback')
            assert solver.kernel(cc,eris,t1,t2)=='fallback'


def test_solver_input_validation():
    a=np.ones((2,3));b=np.ones((2,2,3,3))
    with pytest.raises(ValueError,match='shapes'):cpp.weighted_dot(a,b,a[:1],b)
    with pytest.raises(ValueError,match='shapes'):cpp.ccsd_energy(a,b,np.eye(4),b)
    with pytest.raises(ValueError):cpp.dense_combine(np.ones(2),[a],[b])
    with pytest.raises(TypeError):cpp.weighted_dot(a.astype(np.float32),b,a,b)
    with pytest.raises(TypeError):cpp.dense_combine(np.ones(1),[a.astype(np.float32)],[b])
