"""Differential checks for the complete native resident RCCSD update."""
import sys
import numpy as np
import pytest
from pyscf import lib
from pyscfad.cc import _dfccsd_native as native, _native_ccsd_forward as forward

cpp = forward._ccsd_kernels
pytestmark = pytest.mark.skipif(cpp is None or not hasattr(cpp, 'residual'),
                               reason='rebuild CCSD extension for native residual')


def make_case(o=3, v=7, shift=0.):
    rng = np.random.default_rng(937)
    n, nx = o+v, 11
    factors = rng.normal(size=(nx,n,n))*.04
    factors = (factors+factors.transpose(0,2,1))*.5
    eri = np.einsum('xpq,xrs->pqrs', factors, factors)
    energy = np.r_[-np.linspace(2,1,o), np.linspace(.5,1.5,v)]
    fock = rng.normal(size=(n,n))*.003
    fock = (fock+fock.T)*.5
    np.fill_diagonal(fock, energy)
    data = dict(fock=fock, mo_energy=energy, oooo=eri[:o,:o,:o,:o],
        ovoo=eri[:o,o:,:o,:o], ovov=eri[:o,o:,:o,o:], oovv=eri[:o,:o,o:,o:],
        ovvo=eri[:o,o:,o:,:o],
        ovvv=lib.pack_tril(eri[:o,o:,o:,o:].reshape(o*v,v,v)).reshape(o,v,-1),
        Lov=factors[:,:o,o:], Lvv=lib.pack_tril(factors[:,o:,o:]))
    options = dict(max_memory=1000, max_cycle=50, conv_tol=1e-9, conv_tol_normt=1e-7,
        iterative_damping=1., diis_space=3, diis_start_cycle=0, diis_start_energy_diff=1e9,
        level_shift=shift, verbose=0, stdout=sys.stdout, diis=True, threads=2,
        vvvv_direct=False, cpp_kernels=True)
    cc, eris = native.host_objects(options, [data[k] for k in native.ERI_NAMES], o)
    t1 = rng.normal(size=(o,v))*.01
    t2 = rng.normal(size=(o,o,v,v))*.01
    t2 = (t2+t2.transpose(1,0,3,2))*.5
    return cc, eris, t1, t2


@pytest.mark.parametrize('o,v,block,shift', [(1,1,1,0.), (2,5,1,.2), (3,7,3,0.), (3,7,7,.17), (4,19,7,.1)])
@pytest.mark.parametrize('layout', ['strided', 'fortran'])
def test_native_complete_residual(o,v,block,shift,layout,monkeypatch):
    cc, eris, t1, t2 = make_case(o,v,shift)
    # Physical ERIs start as slices of a larger tensor. Fortran layout covers
    # a different zero-copy GEMM plan; all inputs must remain read-only.
    if layout == 'fortran':
        for name in native.ERI_NAMES:
            setattr(eris, name, np.asfortranarray(getattr(eris,name)))
        t1, t2 = np.asfortranarray(t1), np.asfortranarray(t2)
    arrays = [t1,t2]+[getattr(eris,k) for k in native.ERI_NAMES]+[eris.vvvv]
    saved = [a.copy() for a in arrays]
    for a in arrays: a.flags.writeable=False
    expected = forward._reference_update_amps(cc,t1,t2,eris)
    reference_ladder = cc._add_vvvv(t1,t2,eris,t2sym='jiba')
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1,user_api='blas'):
        seed = cpp.ladder_residual(t1,t2,eris.vvvv,2**20)
    np.testing.assert_allclose(seed,reference_ladder,atol=2e-13,rtol=1e-12)
    with threadpool_limits(1,user_api='blas'):
        actual = cpp.residual(t1,t2,eris.fock,eris.mo_energy,eris.oooo,eris.ovoo,
            eris.oovv,eris.ovvo,eris.ovvv,seed,shift,block,True)
    for a,b in zip(actual[:2],expected): np.testing.assert_allclose(a,b,atol=2e-13,rtol=1e-12)
    assert set(actual[2]['sections']) == {'fock','ovvv','ovov','oooo','finish'}
    assert actual[2]['contractions']['icjb,cjba->ia']['calls'] == (v+min(32,block)-1)//min(32,block)
    for expression in ('ijac,bc->ijab','ki,kjab->ijab'):
        stats=actual[2]['contractions'][expression]
        assert stats['direct_calls']==1 and stats['result_bytes']==0
    for expression in ('jb,ijba->ia','jbki,kjba->ia','ijkl,klab->ijab'):
        stats=actual[2]['contractions'][expression]
        assert stats['direct_calls']==(v+block-1)//block
        assert stats['result_bytes']==0
    assert actual[2]['resident_reuse']['oooo_scratch_bytes']<=forward._residual_scratch_bytes(o,v)
    assert set(actual[2]['finish_detail'])=={'contractions','symmetry_and_denominators'}
    # Verify driver dispatch really exercises C++, with no Python equations.
    monkeypatch.setattr(forward,'_reference_update_amps',lambda *a:pytest.fail('unexpected fallback'))
    got = forward.update_amps(cc,t1,t2,eris)
    for a,b in zip(got,expected):np.testing.assert_allclose(a,b,atol=2e-13,rtol=1e-12)
    for a,b in zip(arrays,saved): np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('expr', ['icjb,cjba->ia','aikc,kcjb->aijb','bkic,jkca->bija',
    'jkca,ckib->jaib','kica,ckjb->ijab','kc,kcji->ij','kc,icjk->ij',
    'ijcd,cdbk->ijbk','biac,jc->bija','ia,ja->ij','ia,ib->ab','ijac,bc->ijab'])
@pytest.mark.parametrize('layout', ['c','fortran','negative'])
def test_contraction_plans(expr,layout):
    rng=np.random.default_rng(812)
    lhs,out=expr.split('->');specs=lhs.split(',')
    dims={c:2+i%3 for i,c in enumerate(sorted(set(lhs)-{','}))}
    arrays=[rng.normal(size=tuple(dims[c] for c in spec)) for spec in specs]
    if layout=='fortran':arrays=[np.asfortranarray(a) for a in arrays]
    if layout=='negative':arrays=[a[...,::-1] for a in arrays]
    originals=[a.copy() for a in arrays]
    np.testing.assert_allclose(cpp._contract(expr,*arrays),np.einsum(expr,*arrays),atol=2e-12,rtol=1e-12)
    for a,b in zip(arrays,originals):np.testing.assert_array_equal(a,b)


def test_native_validation_and_fallback(monkeypatch):
    cc,eris,t1,t2=make_case()
    assert forward._native_plan(cc,t1,t2,eris) is not None
    for key,value in [('cpp_kernels',False),('direct',True),('async_io',True),('max_memory',0)]:
        with monkeypatch.context() as patch:
            patch.setattr(cc,key,value)
            patch.setattr(forward,'_reference_update_amps',lambda *a:'fallback')
            assert forward.update_amps(cc,t1,t2,eris)=='fallback'
    with monkeypatch.context() as patch:
        patch.setattr(eris,'vvvv',None)
        assert forward._native_plan(cc,t1,t2,eris) is None
    with pytest.raises(ValueError,match='shapes'):
        cpp.ladder_residual(t1,t2[:2],eris.vvvv,2**20)
    with pytest.raises(ValueError,match='dimensions'):
        cpp._contract('ij,jk->ik',np.zeros((2,3)),np.zeros((4,5)))
    with pytest.raises(ValueError):
        cpp._contract('ii,ij->j',np.ones((3,3)),np.ones((3,2)))
    with pytest.raises(TypeError):
        cpp._contract('ij,jk->ik',np.ones((2,3),np.float32),np.ones((3,4)))


@pytest.mark.parametrize('m,n,k', [(7,11,18000), (350,9,150), (9,350,150)])
@pytest.mark.parametrize('layout', ['c','fortran'])
def test_parallel_gemm_partition(m,n,k,layout):
    from threadpoolctl import threadpool_limits
    rng=np.random.default_rng(543)
    a,b=rng.normal(size=(m,k)),rng.normal(size=(k,n))
    if layout=='fortran':a,b=np.asfortranarray(a),np.asfortranarray(b)
    with threadpool_limits(1,user_api='blas'):
        expected=a@b
        actual=cpp._contract('ik,kj->ij',a,b)
    np.testing.assert_allclose(actual,expected,atol=5e-11,rtol=1e-12)


def test_iteration_layout_cache_refreshes_amplitudes():
    from threadpoolctl import threadpool_limits
    cc,eris,t1,t2=make_case(3,7,.1)
    for scale in (1., .91, 1.07):
        # Reuse the same addresses while changing their contents.
        t1 *= scale
        t2 *= scale
        expected=forward._reference_update_amps(cc,t1,t2,eris)
        plan=forward._native_plan(cc,t1,t2,eris)
        with threadpool_limits(1,user_api='blas'):
            actual=forward._native_update(cc,t1,t2,eris,plan,profile=True)
        for a,b in zip(actual[:2],expected):
            np.testing.assert_allclose(a,b,atol=2e-13,rtol=1e-12)
        assert actual[2]['ovov_layout_cache']['bytes'] <= t2.nbytes
        assert actual[2]['ovov_layout_cache']['end_bytes'] == 0


@pytest.mark.parametrize('block', [1,7,19])
@pytest.mark.parametrize('layout', ['c','fortran','negative'])
def test_selective_ovov_cache_and_tau_lifetimes(block,layout):
    cc,eris,t1,t2=make_case(4,19,.1)
    for name in native.ERI_NAMES:
        value=np.array(getattr(eris,name),order='F' if layout=='fortran' else 'C',copy=True)
        if layout=='negative' and value.ndim>1:value=value[...,::-1]
        setattr(eris,name,value)
    expected=forward._reference_update_amps(cc,t1,t2,eris)
    plan=forward._native_plan(cc,t1,t2,eris)
    actual=forward._native_update(cc,t1,t2,eris,(plan[0],block),profile=True)
    for a,b in zip(actual[:2],expected):
        np.testing.assert_allclose(a,b,atol=2e-13,rtol=1e-12)
    cache=actual[2]['ovov_layout_cache'];tau=actual[2]['ovov_tau_temporaries']
    assert cache['end_bytes']==0
    assert cache['bytes']<=8*4*4*block*19
    assert cache['packed_bytes']<=t2.nbytes
    if layout=='c':
        assert cache['bytes']==8*4*4*block*19
        assert cache['packed_bytes']==t2.nbytes
        assert cache['hits']==(19+block-1)//block
    # Reuse across partial p blocks must not rebuild either full tensor.
    assert tau['half_builds']==tau['response_builds']==1
    assert tau['peak_bytes']==t2.nbytes*(1 if block==19 else 2)


@pytest.mark.skipif(not hasattr(cpp, 'ResidualWorkspace'), reason='rebuild resident workspaces')
@pytest.mark.parametrize('layout', ['c', 'fortran', 'negative'])
def test_resident_workspace_refresh_and_views(layout):
    from threadpoolctl import threadpool_limits
    cc,eris,t1,t2=make_case(3,19,.1)
    if layout=='fortran':
        eris.ovvv=np.asfortranarray(eris.ovvv)
        eris.ovoo=np.asfortranarray(eris.ovoo)
    elif layout=='negative':
        eris.ovvv=eris.ovvv[:,:,::-1]
        eris.ovoo=eris.ovoo[:,:,:,::-1]
    eris.ovvv.flags.writeable=eris.ovoo.flags.writeable=False
    cached=None
    for scale in (1., .91, 1.07):
        t1*=scale;t2*=scale
        expected=forward._reference_update_amps(cc,t1,t2,eris)
        plan=forward._native_plan(cc,t1,t2,eris)
        # Force partial virtual blocks in both residual sections.
        with threadpool_limits(1,user_api='blas'):
            actual=forward._native_update(cc,t1,t2,eris,(plan[0],7),profile=True)
        for a,b in zip(actual[:2],expected):
            np.testing.assert_allclose(a,b,atol=2e-13,rtol=1e-12)
        assert actual[2]['resident_reuse']['enabled']
        if hasattr(cpp, '_matrix_layout'):
            # Tau tiles stay as padded matrix views; only the integral is packed.
            contraction=actual[2]['contractions']['ijcd,cdbk->ijbk']
            assert contraction['packed_bytes']==t1.shape[0]*t1.shape[1]**3*8
            assert contraction['direct_calls']==3
            assert contraction['result_bytes']==0
            assert actual[2]['resident_reuse']['ovvv_scratch_bytes']<=forward._ovvv_scratch_bytes(*t1.shape)
            reuse=actual[2]['resident_reuse']
            assert 0<reuse['ovov_scratch_bytes']<=forward._ovov_scratch_bytes(*t1.shape)
            assert reuse['shared_scratch_bytes']==max(reuse['vvvv_scratch_bytes'],reuse['ovvv_scratch_bytes'],reuse['ovov_scratch_bytes'])
            assert 0<reuse['vvvv_scratch_bytes']<=cpp.ladder_scratch_bytes(*t1.shape)
        current=eris._native_residual_workspace[1]
        if cached is not None:assert current is cached
        cached=current
    eris.ovvv=eris.ovvv.copy()*1.01
    refreshed=forward._resident_workspace(cc,t1,t2,eris)
    assert refreshed is not cached
    # A context cannot be used with a different integral view accidentally.
    with threadpool_limits(1,user_api='blas'):
        seed=cpp.ladder_residual(t1,t2,eris.vvvv,2**20)
        with pytest.raises(ValueError,match='different integral views'):
            cpp.residual(t1,t2,eris.fock,eris.mo_energy,eris.oooo,eris.ovoo,
                eris.oovv,eris.ovvo,eris.ovvv,seed,.1,7,False,cached)
    eris._native_residual_workspace=None
    cc.max_memory=0
    assert forward._resident_workspace(cc,t1,t2,eris) is None


@pytest.mark.skipif(not hasattr(cpp, '_matrix_layout'), reason='rebuild leading-dimension adapter')
def test_matrix_leading_dimensions():
    tau=np.zeros((3,4,19,11))[:,:,2:9,:]
    for a,split,trans,ld in [(tau,2,'N',19*11),
                            (tau.transpose(2,3,0,1),2,'T',19*11),
                            (np.ones((6,31))[::2,2:9],1,'N',62),
                            (np.ones((1,19,1,11))[:,2:9],2,'N',11)]:
        info=cpp._matrix_layout(a,split)
        assert info==dict(packed_bytes=0,trans=trans,leading_dimension=ld)
    for a in (tau[:,:,:,::-1],tau[:,:,:,::2],np.broadcast_to(np.ones((1,5)),(3,5))):
        assert cpp._matrix_layout(a,a.ndim//2)['packed_bytes']>0


@pytest.mark.skipif(not hasattr(cpp, '_contract_accumulate'), reason='rebuild direct accumulation')
@pytest.mark.parametrize('m,n,k', [(7,11,18000),(350,9,150),(9,350,150)])
@pytest.mark.parametrize('layout', ['padded','transposed','inner_stride'])
def test_direct_gemm_padded_output(m,n,k,layout):
    from threadpoolctl import threadpool_limits
    rng=np.random.default_rng(745)
    a=rng.normal(size=(m,k+3))[:,1:k+1]
    b=rng.normal(size=(k,n+5))[:,2:n+2]
    if layout=='transposed':
        storage=rng.normal(size=(n+4,m+3));out=storage[1:n+1,2:m+2].T
    else:
        storage=rng.normal(size=(m+4,2*n+3))
        out=storage[1:m+1,1:n+1] if layout=='padded' else storage[1:m+1,1:2*n+1:2]
    saved=storage.copy();expected_storage=saved.copy()
    if layout=='transposed':expected=expected_storage[1:n+1,2:m+2].T
    elif layout=='padded':expected=expected_storage[1:m+1,1:n+1]
    else:expected=expected_storage[1:m+1,1:2*n+1:2]
    a.flags.writeable=b.flags.writeable=False
    with threadpool_limits(1,user_api='blas'):
        expected+=-.37*(a@b)
        report=cpp._contract_accumulate(out,'ik,kj->ij',a,b,-.37)
    np.testing.assert_allclose(storage,expected_storage,atol=5e-11,rtol=1e-12)
    assert report['packed_bytes']==0
    assert report['direct_calls']==(layout!='inner_stride')
    assert report['result_bytes']==(out.nbytes if layout=='inner_stride' else 0)


@pytest.mark.skipif(not hasattr(cpp, '_contract_accumulate'), reason='rebuild direct accumulation')
def test_direct_gemm_alias_falls_back():
    rng=np.random.default_rng(747)
    a=rng.normal(size=(5,5));b=rng.normal(size=(5,5))
    expected=a+.3*(a@b)
    report=cpp._contract_accumulate(a,'ik,kj->ij',a,b,.3)
    np.testing.assert_allclose(a,expected,atol=2e-13,rtol=1e-12)
    assert report['direct_calls']==0
    assert report['result_bytes']==a.nbytes


@pytest.mark.skipif(not hasattr(cpp, '_contract_accumulate'), reason='rebuild direct accumulation')
def test_layout_cache_survives_scratch_reuse():
    from threadpoolctl import threadpool_limits
    rng=np.random.default_rng(749)
    # All three matrices need packing. Only A fits in the layout cache;
    # B and the output temporary repeatedly share the scratch arena.
    a=rng.normal(size=(7,26))[:,::2]
    b=rng.normal(size=(13,18))[:,::2]
    out=rng.normal(size=(7,18))[:,::2]
    a.flags.writeable=b.flags.writeable=False
    with threadpool_limits(1,user_api='blas'):
        expected=out+3*.37*(a@b)
        report=cpp._contract_accumulate(out,'ik,kj->ij',a,b,.37,
                                       cache_bytes=a.nbytes,repeat=3)
    np.testing.assert_allclose(out,expected,atol=2e-13,rtol=1e-12)
    assert report['cache_bytes']==a.nbytes
    assert report['cache_hits']==2
    assert report['packed_bytes']==a.nbytes+3*b.nbytes
    assert report['direct_calls']==0
    assert report['result_bytes']==3*out.nbytes


@pytest.mark.skipif(not hasattr(cpp, 'residual_from_integrals'), reason='rebuild integrated ladder')
@pytest.mark.parametrize('resident', [False, True])
@pytest.mark.parametrize('o,v', [(1,1), (3,7), (4,19)])
def test_direct_ladder_expansion_and_buffer_lifetimes(o,v,resident):
    from threadpoolctl import threadpool_limits
    cc,eris,t1,t2=make_case(o,v,.13)
    t1,t2=np.asfortranarray(t1),np.asfortranarray(t2)
    workspace=cpp.ResidualWorkspace(eris.ovvv,eris.ovoo) if resident else None
    saved=None
    # Exercise one-panel workspaces and larger plans with partial tiles.
    for budget,scale in [(8*v*v,1.), (2**20,.91), (8*v*v,1.07)]:
        t1*=scale;t2*=scale
        originals=t1.copy(),t2.copy(),eris.vvvv.copy()
        expected=forward._reference_update_amps(cc,t1,t2,eris)
        with threadpool_limits(1,user_api='blas'):
            actual=cpp.residual_from_integrals(t1,t2,eris.fock,eris.mo_energy,
                eris.oooo,eris.ovoo,eris.oovv,eris.ovvo,eris.ovvv,eris.vvvv,
                budget,.13,min(v,3),True,workspace)
        for a,b in zip(actual[:2],expected):
            np.testing.assert_allclose(a,b,atol=2e-13,rtol=1e-12)
        for a,b in zip((t1,t2,eris.vvvv),originals):np.testing.assert_array_equal(a,b)
        if saved is not None:
            for old,copy in zip(saved[0],saved[1]):np.testing.assert_array_equal(old,copy)
        saved=actual[:2],tuple(a.copy() for a in actual[:2])
        reuse=actual[2]['resident_reuse']
        assert 0<reuse['vvvv_scratch_bytes']<=cpp.ladder_scratch_bytes(o,v)
        assert reuse['shared_scratch_bytes']>=max(reuse[k] for k in
            ('vvvv_scratch_bytes','ovvv_scratch_bytes','ovov_scratch_bytes'))
        assert actual[2]['sections']['vvvv']['wall_seconds']>0
        assert not np.shares_memory(actual[1],t2)
    with pytest.raises(ValueError,match='workspace'):
        cpp.residual_from_integrals(t1,t2,eris.fock,eris.mo_energy,eris.oooo,
            eris.ovoo,eris.oovv,eris.ovvo,eris.ovvv,eris.vvvv,8*v*v-1,.13,1)


@pytest.mark.skipif(not hasattr(cpp, '_finish_amplitudes'), reason='rebuild tiled final update')
@pytest.mark.parametrize('o,v', [(1,1), (3,7), (4,35), (2,65), (5,40)])
@pytest.mark.parametrize('threads', [1,4])
def test_tiled_symmetry_and_denominators(o,v,threads):
    from threadpoolctl import threadpool_limits
    rng=np.random.default_rng(997)
    r1=np.asfortranarray(rng.normal(size=(o,v)))
    r2=np.asfortranarray(rng.normal(size=(o,o,v,v)))
    storage=np.zeros(2*(o+v))
    eps=storage[::2];eps[:o]=-np.linspace(2,1,o);eps[o:]=np.linspace(.5,1.5,v)
    shift=.17
    expected1=r1/(eps[:o,None]-eps[None,o:]-shift)
    denom=(eps[:o,None,None,None]+eps[None,:o,None,None]
           -eps[None,None,o:,None]-eps[None,None,None,o:]-2*shift)
    expected2=(r2+r2.transpose(1,0,3,2))/denom
    originals=[a.copy() for a in (r1,r2,eps)]
    for a in (r1,r2,eps):a.flags.writeable=False
    with threadpool_limits(threads,user_api='openmp'):
        for _ in range(3):
            actual=cpp._finish_amplitudes(r1,r2,eps,shift)
            np.testing.assert_allclose(actual[0],expected1,atol=2e-14,rtol=2e-14)
            np.testing.assert_allclose(actual[1],expected2,atol=2e-14,rtol=2e-14)
            np.testing.assert_array_equal(actual[1],actual[1].transpose(1,0,3,2))
    for a,b in zip((r1,r2,eps),originals):np.testing.assert_array_equal(a,b)
