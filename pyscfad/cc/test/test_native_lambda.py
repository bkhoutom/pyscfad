"""Native lambda equations, dense response metric and bounded cache policy."""
import copy
import numpy as np
import pytest
from pyscf import lib
from pyscf.cc import ccsd
from threadpoolctl import threadpool_limits
from .test_native_residual import make_case
from pyscfad.cc import _native_ccsd_lambda as reference
from pyscfad.cc import _native_lambda_cpp as compiled
from pyscfad.cc import _dfccsd_native as native

cpp=compiled.cpp
pytestmark=pytest.mark.skipif(cpp is None or not hasattr(cpp,'lambda_prepare'),reason='rebuild lambda kernels')


def arrays(er,t1,t2):
    return [t1,t2,er.fock,er.mo_energy,er.oooo,er.ovoo,er.oovv,er.ovvo,er.ovvv,np.ascontiguousarray(er.vvvv)]


@pytest.mark.parametrize('o,v,block',[(1,1,1),(3,7,1),(3,7,3),(3,7,7),(4,19,7)])
@pytest.mark.parametrize('resident,cache',[(True,True),(False,True),(True,False),(False,False)])
@pytest.mark.parametrize('layout',['c','fortran','negative'])
def test_lambda_update(o,v,block,resident,cache,layout):
    cc,er,t1,t2=make_case(o,v,.13);cc.max_memory=10000
    # Negative strides preserve the values and integral symmetries.
    def layout_of(x):
        if layout=='negative' and x.ndim>1:return np.ascontiguousarray(x[...,::-1])[...,::-1]
        return np.array(x,order='F' if layout=='fortran' else 'C',copy=True)
    for name in native.ERI_NAMES:setattr(er,name,layout_of(getattr(er,name)))
    t1,t2=map(layout_of,(t1,t2));l1,l2=t1*.7,t2*.9
    with lib.with_omp_threads(2),threadpool_limits(1,user_api='blas'):
        im=reference._make_intermediates_df(cc,t1,t2,er)
        expected=reference.update_lambda(cc,t1,t2,l1,l2,er,im)
        w,report=cpp.lambda_prepare(arrays(er,t1,t2),block,2**20,resident,cache,True)
        got=cpp.lambda_update(w,l1,l2,.13,True)
        for a,b in zip(got[:2],expected):np.testing.assert_allclose(a,b,atol=2e-13,rtol=1e-12)
        values=w.intermediates()
        for name,value in values.items():
            if value is None:continue
            target=2*im.wVOov+im.wvOOv if name=='mixed' else getattr(im,name)
            np.testing.assert_allclose(value,target,atol=2e-13,rtol=1e-12)
        expected2=reference.update_lambda(cc,t1,t2,l1*1.1,l2*.8,er,im)
        got2=cpp.lambda_update(w,l1*1.1,l2*.8,.13)
        for a,b in zip(got2,expected2):np.testing.assert_allclose(a,b,atol=2e-13,rtol=1e-12)
        assert set(got[2]['sections'])=={'vvvv','fock_and_density','ovvv','ovov','oooo','finish'}
        info=w.memory_info()
        assert info['working_bytes']==t2.nbytes
        for a,b in zip(got[:2],expected):np.testing.assert_allclose(a,b,atol=2e-13,rtol=1e-12)
        assert info['populated_bytes']<=info['allocated_bytes']
        bound=compiled.memory_bound(o,v,block,cpp.ladder_memory(o,v,2**20),resident_ovvv=resident,cache_wvvov=cache)
        assert info['allocated_bytes']<=bound['peak_bytes']
        assert info['scratch_capacity_bytes']<=bound['scratch_bytes']


@pytest.mark.parametrize('shift',[0.,.13])
@pytest.mark.parametrize('space',[0,3])
def test_dense_response_and_diis(shift,space,monkeypatch):
    cc,er,t1,t2=make_case(3,7,shift);cc.max_memory=10000;cc.conv_tol_normt=1e-11;cc.max_cycle=50
    cc.diis=bool(space);cc.diis_space=space or 3
    rng=np.random.default_rng(179);bars=(.7,rng.normal(size=t1.shape)*.01,rng.normal(size=t2.shape)*.01)
    old=copy.copy(cc);old.cpp_kernels=False
    with lib.with_omp_threads(2),threadpool_limits(1,user_api='blas'):
        action=native.response_action(old,er,t1,t2)
        source=native.energy_amplitude_cotangent(er,t1,bars[0])+native.pack_cotangent(*bars[1:])
        vec=rng.normal(size=ccsd.amplitudes_to_vector(t1,t2).shape)*.01
        im=compiled.prepare(cc,t1,t2,er,response=True)
        rhs=cpp.lambda_source(t1,er.fock,er.ovov,er.ovvo,*bars)
        got=cpp.lambda_response(im.workspace,*ccsd.vector_to_amplitudes(vec,10,3),*rhs,shift)
        np.testing.assert_allclose(ccsd.amplitudes_to_vector(*got),action(vec)+source,atol=1e-12,rtol=1e-12)
        assert im.workspace.memory_info()['working_bytes']==2*t2.nbytes+t1.nbytes
        expected=native.solve_response(old,er,t1,t2,*bars)
        assert expected[0]
        count=[];pack=ccsd.amplitudes_to_vector
        def counted(*args,**kwargs):count.append(1);return pack(*args,**kwargs)
        def forbidden(*args,**kwargs):raise AssertionError('packed response conversion in iteration')
        monkeypatch.setattr(ccsd,'amplitudes_to_vector',counted)
        monkeypatch.setattr(native,'pack_cotangent',forbidden)
        monkeypatch.setattr(native,'inverse_metric',forbidden)
        actual=native.solve_response(cc,er,t1,t2,*bars)
        assert actual[0] and len(count)==1
        np.testing.assert_allclose(actual[1],expected[1],atol=1e-11,rtol=1e-11)


def test_memory_rejection_and_cache_selection(monkeypatch):
    cc,er,a,b=make_case(3,7)
    monkeypatch.setattr(lib,'current_memory',lambda:(100.,0.))
    cc.max_memory=100
    assert compiled.prepare(cc,a,b,er) is None
    cc.max_memory=10000
    assert compiled.prepare(cc,a,b,er,reserve_bytes=10**12) is None
    im=compiled.prepare(cc,a,b,er)
    assert im.plan['resident_ovvv'] and im.plan['cache_wvvov']
    cc.cpp_kernels=False
    assert compiled.prepare(cc,a,b,er) is None
