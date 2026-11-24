"""Validate parallel tensor updates, alias guards, and blocked CCSD equations."""
import numpy as np
import pytest
from pyscfad.cc import _native_kernels, _native_ccsd_forward as forward

cpp = _native_kernels._ccsd_kernels
pytestmark = pytest.mark.skipif(cpp is None or not hasattr(cpp, 'tau4'),
                                reason='rebuild CCSD extension for tensor kernels')


@pytest.mark.parametrize('shape', [(3, 2, 4, 5), (7, 5, 13, 17), (2, 0, 3, 4)])
@pytest.mark.parametrize('reverse', [False, True])
def test_tensor_operations(shape, reverse):
    rng = np.random.default_rng(713)
    i,j,a,b = shape
    x = rng.normal(size=(a,i,b,j)).transpose(1,3,0,2)
    y = rng.normal(size=shape)
    if reverse:
        x,y = x[...,::-1],y[...,::-1]
    destination = np.empty((i,j,a*2,b))[:,:,::2]
    destination[:] = y
    cpp.axpy4(destination, x, -.7)
    np.testing.assert_allclose(destination, y-.7*x, atol=1e-14)
    np.testing.assert_allclose(cpp.linear4(x,y,2.,-.5), 2*x-.5*y, atol=1e-14)
    left,right = rng.normal(size=(i,a)),rng.normal(size=(j,b))
    expected=.5*x-2*np.einsum('ia,jb->ijab',left,right)
    np.testing.assert_allclose(cpp.tau4(x,left,right,.5,-2.),expected,atol=1e-14)
    before = destination.copy()
    cpp.axpy4(destination,destination,.5)
    np.testing.assert_allclose(destination,before*1.5,atol=1e-14)


def test_tensor_validation():
    a=np.ones((3,3,4,4))
    with pytest.raises(ValueError,match='overlap'):
        cpp.axpy4(a,a.transpose(1,0,2,3),1.)
    overlap=np.lib.stride_tricks.as_strided(a,shape=a.shape,strides=(0,0,0,8))
    with pytest.raises(ValueError,match='overlapping output'):
        cpp.axpy4(overlap,a.copy(),1.)
    a.flags.writeable=False
    with pytest.raises(ValueError,match='writable'):
        cpp.axpy4(a,a,1.)
    with pytest.raises(ValueError,match='shapes'):
        cpp.linear4(a,np.ones((2,2,2,2)),1.,1.)
    with pytest.raises(ValueError,match='outer-product'):
        cpp.tau4(a,np.ones((3,5)),np.ones((3,4)),1.,1.)
    with pytest.raises(TypeError):
        cpp.linear4(a.astype(np.float32),a,1.,1.)


@pytest.mark.parametrize('blocked', [False, True])
def test_complete_update_with_resident_integral_views(blocked):
    from pyscfad.cc import _dfccsd_native as native
    import sys
    rng=np.random.default_rng(795)
    no,nv,nx=3,7,11
    n=no+nv
    factors=rng.normal(size=(nx,n,n))*.04
    factors=(factors+factors.transpose(0,2,1))*.5
    eri=np.einsum('xpq,xrs->pqrs',factors,factors)
    fock=np.diag(np.r_[-np.linspace(2,1,no),np.linspace(.5,1.5,nv)])
    oo,ov,vv=factors[:,:no,:no],factors[:,:no,no:],factors[:,no:,no:]
    from pyscf import lib
    data=dict(fock=fock,mo_energy=fock.diagonal().copy(),oooo=eri[:no,:no,:no,:no],
        ovoo=eri[:no,no:,:no,:no],ovov=eri[:no,no:,:no,no:],oovv=eri[:no,:no,no:,no:],
        ovvo=eri[:no,no:,no:,:no],ovvv=lib.pack_tril(eri[:no,no:,no:,no:].reshape(no*nv,nv,nv)).reshape(no,nv,-1),
        Lov=ov,Lvv=lib.pack_tril(vv))
    options=dict(max_memory=1000,max_cycle=50,conv_tol=1e-9,conv_tol_normt=1e-7,
        iterative_damping=1.,diis_space=3,diis_start_cycle=0,diis_start_energy_diff=1e9,
        level_shift=0.,verbose=0,stdout=sys.stdout,diis=True,threads=2,vvvv_direct=False,cpp_kernels=True)
    cc,eris=native.host_objects(options,[data[k] for k in native.ERI_NAMES],no)
    if blocked:cc.max_memory=0
    t1=rng.normal(size=(no,nv))*.01
    t2=rng.normal(size=(no,no,nv,nv))*.01
    t2=(t2+t2.transpose(1,0,3,2))*.5
    arrays=[getattr(eris,k) for k in native.ERI_NAMES]
    saved=[a.copy() for a in arrays]
    for a in arrays:a.flags.writeable=False
    expected=forward._reference_update_amps(cc,t1,t2,eris)
    actual=forward.update_amps(cc,t1,t2,eris)
    for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,atol=2e-13,rtol=1e-12)
    for a,b in zip(arrays,saved):np.testing.assert_array_equal(a,b)
