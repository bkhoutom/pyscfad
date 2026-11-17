"""Symmetric half-Laplace partition versus independent six-index equations."""
import itertools
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad import lib
from pyscfad.dlno_stc import controls, _stc_mp2 as native
from pyscfad.dlno_stc.mrcc_triples import run_mrcc
from pyscfad.dlno_stc.smooth_triples import make_solver
from pyscfad.dlno_stc.test.test_triples_stable import fixture, transform
from pyscfad.dlno_stc.test._triples_dressing_reference import symmetric_exp


def numerators(inputs):
    t1,t2,gv,go,gg,fov,foo,fvv,u=inputs
    raw=jnp.einsum('iabf,kjcf->ijkabc',gv,t2)-jnp.einsum('iajm,mkbc->ijkabc',go,t2)
    singles=jnp.einsum('iajb,kc->ijkabc',gg,t1)+jnp.einsum('ijab,kc->ijkabc',t2,fov)
    perms=list(itertools.permutations(range(3)))
    w=sum(raw.transpose((*p,*(i+3 for i in p))) for p in perms)
    l=sum(singles.transpose((*p,*(i+3 for i in p))) for p in perms)
    return w,l


def reference(inputs,node):
    w,l=numerators(inputs)
    o=symmetric_exp(node*inputs[6]/2)
    v=symmetric_exp(-node*inputs[7]/2)
    axes=(inputs[8].T@o,o,o,v,v,v)
    z=transform(w+.5*l,axes);w=transform(w,axes)
    # Virtual spin permutations commute with the physical occupied projection.
    z=(4*z+z.transpose(0,1,2,4,5,3)+z.transpose(0,1,2,5,3,4)
       -2*(z.transpose(0,1,2,3,5,4)+z.transpose(0,1,2,4,3,5)+z.transpose(0,1,2,5,4,3)))
    return -jnp.sum(w*z)/3


@pytest.mark.parametrize('selected',[1,2,3])
@pytest.mark.parametrize('spectrum',['dense','diagonal','degenerate'])
def test_energy_and_all_nine_bars(selected,spectrum):
    inputs=list(fixture(selected=selected,degenerate=spectrum=='degenerate'))
    if spectrum=='diagonal':
        inputs[6]=np.diag([-.8,-1.2,-.4]);inputs[7]=np.diag([1.1,.3,1.6,.7])
    inputs=tuple(map(jnp.asarray,inputs));nodes=np.array([0.,.37]);weights=np.array([.2,.8])
    expected,expected_bars=jax.value_and_grad(lambda x:sum(w*reference(x,t) for t,w in zip(nodes,weights)))(inputs)
    (actual,d),bars=run_mrcc(inputs,nodes,weights,with_grad=True,max_memory=2000.,rotate=True)
    forward,_=run_mrcc(inputs,nodes,weights,with_grad=False,max_memory=2000.,rotate=True)
    np.testing.assert_allclose([actual,forward[0]],expected,rtol=3e-11,atol=3e-13)
    for i,(got,want) in enumerate(zip(bars,expected_bars)):
        np.testing.assert_allclose(got,want,rtol=3e-10,atol=3e-12,err_msg=f'input {i}')
    assert d['connected_build_blocks']==len(nodes)*selected*6
    assert d['connected_gemm_calls']==12*d['connected_build_blocks']
    assert d['occupied_block_batch_size']<=128


def test_complete_partition_matches_asymmetric_and_selected_partition_differs():
    def run(x):return run_mrcc(x,np.array([.37]),np.array([1.]),with_grad=False,max_memory=2000.,rotate=True)[0][0]
    def asym(x):
        from pyscfad.dlno_stc.test.test_triples_stable import reference as asymmetric_reference
        return asymmetric_reference(tuple(map(jnp.asarray,x)),.37)
    full=list(fixture(selected=3));full[8]=np.eye(3)
    np.testing.assert_allclose(run(full),asym(full),rtol=3e-11,atol=3e-13)
    parts=[]
    for i in range(3):
        part=list(full);part[8]=np.eye(3)[:,i:i+1];parts.append(run(part))
    np.testing.assert_allclose(sum(parts),run(full),rtol=3e-11,atol=3e-13)
    partial=list(full);partial[8]=np.eye(3)[:,:1]
    assert abs(float(run(partial)-asym(partial)))>1e-7
    partial[6]=np.diag([-.8,-1.2,-.4])
    np.testing.assert_allclose(run(partial),asym(partial),rtol=3e-11,atol=3e-13)


@pytest.mark.parametrize('integral_transform',['df','rotate'])
def test_public_option_and_custom_vjp(integral_transform):
    assert controls.normalize_triples_controls(None,ccsd_t=True)=={'mode':'deterministic','partition':'mrcc'}
    with pytest.raises(ValueError, match='removed'):
        controls.normalize_triples_controls({'partition':'asymmetric'},ccsd_t=True)
    options=controls.normalize_triples_controls({'partition':'mrcc','integral_transform':integral_transform},ccsd_t=True)
    for bad in [{'partition':'wrong'},{'partition':'mrcc','mode':'stochastic','nsamples':16}]:
        with pytest.raises(ValueError,match='partition'):
            controls.normalize_triples_controls(bad,ccsd_t=True)
    x=tuple(map(jnp.asarray,fixture(selected=2)))
    solver=make_solver(options,(np.array([.37]),np.array([1.]),{}),with_grad=True)
    def energy(values):
        t1,t2,gv,go,gg,fov,foo,fvv,u=values
        eris=SimpleNamespace(ovvv=lib.pack_tril(gv.reshape(12,4,4)).reshape(3,4,-1),
            ovoo=go,ovov=gg,fock=jnp.block([[foo,fov],[fov.T,fvv]]))
        return solver(SimpleNamespace(max_memory=2000.),eris,u.T,t1=t1,t2=t2)[0]
    actual,bars=jax.value_and_grad(energy)(x)
    def packed_reference(values):
        values=list(values)
        values[2]=lib.unpack_tril(lib.pack_tril(values[2].reshape(12,4,4))).reshape(3,4,4,4)
        return reference(values,.37)
    expected,want=jax.value_and_grad(packed_reference)(x)
    np.testing.assert_allclose(actual,expected,rtol=3e-11,atol=3e-13)
    for a,b in zip(bars,want):np.testing.assert_allclose(a,b,rtol=3e-10,atol=3e-12)


def test_roundoff_diagonal_shortcut_does_not_drop_offdiagonal_response():
    x=list(fixture(selected=2));x[6]=np.diag([-.8,-.8,-.4]);x[7]=np.diag([.3,.3,1.1,1.6])
    x=tuple(map(jnp.asarray,x))
    (_,d),bars=run_mrcc(x,np.array([.37]),np.array([1.]),with_grad=True,max_memory=2000.,rotate=True)
    assert d['diagonal_occupied_points']==d['diagonal_virtual_points']==1
    for k in [6,7]:
        direction=np.zeros(x[k].shape);direction[0,1]=direction[1,0]=1.
        h=1e-5;p=list(x);m=list(x);p[k]+=h*direction;m[k]-=h*direction
        def energy(values):return run_mrcc(values,np.array([.37]),np.array([1.]),with_grad=False,max_memory=2000.,rotate=True)[0][0]
        finite=(energy(p)-energy(m))/(2*h)
        np.testing.assert_allclose(np.sum(bars[k]*direction),finite,rtol=2e-7,atol=2e-10)


def test_matches_mrcc_2024_folded_scalar_expression():
    """Independent translation of ccsd_pt.f's localcc>=2024 energy loop."""
    x=tuple(map(jnp.asarray,fixture(selected=2)))
    w,l=numerators(x);o=symmetric_exp(.37*x[6]/2);v=symmetric_exp(-.37*x[7]/2)
    axes=(x[8].T@o,o,o,v,v,v)
    z=np.asarray(transform(w+.5*l,axes));w=np.asarray(transform(w,axes))
    energy=0.
    orders=((0,1,2),(1,2,0),(2,0,1),(0,2,1),(2,1,0),(1,0,2))
    for s in range(2):
        for j in range(3):
            for k in range(j+1):
                deltaij=1+(j==k)
                for c in range(4):
                    for b in range(c,4):
                        for a in range(b+(b==c),4):
                            abc=(a,b,c);delta=1+(b==c)+(a==b)
                            wi=np.array([w[(s,j,k)+tuple(abc[t] for t in p)] for p in orders])
                            vi=np.array([z[(s,j,k)+tuple(abc[t] for t in p)] for p in orders])
                            w1,w2=wi[:3].sum(),wi[3:].sum();y,zv=vi[:3].sum(),vi[3:].sum()
                            energy-=(3-delta)*(3-deltaij)/6*(w1*(y-2*zv)+w2*(zv-2*y)+3*np.dot(wi,vi))
    np.testing.assert_allclose(energy,reference(x,.37),rtol=3e-12,atol=3e-13)


def test_multiple_batches_and_minimum_workspace_preserve_prepared_bars():
    rng = np.random.default_rng(584)
    no, nv = 9, 2
    shapes = [(no, nv), (no, no, nv, nv), (no, nv, nv, nv),
              (no, nv, no, no), (no, nv, no, nv), (no, nv)]
    x = [rng.normal(scale=.1, size=shape) for shape in shapes]
    x.extend((np.diag(-np.linspace(.3, 1.2, no)), np.diag([.2, 1.]),
              np.eye(no)[:, :3]))
    engine=native.StableTriplesDressing(tuple(x),full_external=True,mrcc_partition=True)
    prepared=engine.prepare(.37)
    results=[native.triples_mrcc_point(prepared,-.2,with_grad=True,
        workspace_limit_bytes=budget,block_batch_size=128) for budget in [0,2**28]]
    assert results[0]['occupied_block_batch_size']==1
    assert results[1]['occupied_block_batch_size']==128
    assert results[1]['occupied_block_batches']==2
    assert results[1]['connected_build_blocks']==135
    assert results[1]['numerator_cache_peak_bytes']==128*2*2**3*8
    np.testing.assert_allclose(results[0]['energy'],results[1]['energy'],rtol=3e-12,atol=3e-13)
    for a,b in zip(results[0]['cotangents'],results[1]['cotangents']):
        np.testing.assert_allclose(a,b,rtol=3e-10,atol=3e-12)


@pytest.mark.parametrize('no,nv',[(3,17),(2,46)])
def test_retained_t2_and_blocked_gathers_across_partial_tiles(no,nv):
    # 17**2 crosses the 256-column destination tile; 17 also exercises
    # the remainder of the 16-wide transpose. Use unrelated, nonsymmetric
    # T2/integrals so cached layouts cannot rely on physical symmetries.
    # 46**2 also crosses the direct-adjoint output and T2V reduction tiles.
    rng=np.random.default_rng(493);ns=2
    shapes=[(no,nv),(no,no,nv,nv),(no,nv,nv,nv),
            (no,nv,no,no),(no,nv,no,nv),(no,nv)]
    x=tuple(rng.normal(scale=.04,size=shape) for shape in shapes)
    x+= (np.diag(-np.linspace(.3,1.2,no)),np.diag(np.linspace(.2,1.5,nv)),
          rng.normal(size=(no,ns)))
    x=tuple(map(jnp.asarray,x));node=.23;weight=-.7
    expected,want=jax.value_and_grad(lambda values:weight*reference(values,node))(x)
    (actual,_),bars=run_mrcc(x,np.array([node]),np.array([weight]),
        with_grad=True,max_memory=3000.,rotate=True)
    np.testing.assert_allclose(actual,expected,rtol=3e-11,atol=3e-13)
    for got,ref in zip(bars,want):
        np.testing.assert_allclose(got,ref,rtol=3e-10,atol=3e-12)
