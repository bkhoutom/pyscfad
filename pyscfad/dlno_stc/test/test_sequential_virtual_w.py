"""Independent dense oracle for the occupied-conditioned W construction."""
import numpy as np
import pytest
from scipy.linalg import expm
from pyscfad.dlno_stc import backend
from pyscfad.lib._threading import blas_threads


def fixture():
    rng = np.random.default_rng(194)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    v, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    x = dict(foo=q @ np.diag([-1.3,-.9,-.7,-.4]) @ q.T,
             fvv=v @ np.diag(np.linspace(.2,1.7,7)) @ v.T,
             B=rng.normal(size=(11,4,7))*.1)
    x = {k: np.ascontiguousarray(a) for k,a in x.items()}
    c = dict(mode='deterministic', threads=4, laplace_roots=[0.,.3,1.1],
             laplace_weights=[.07,.4,.6])
    return x,c,q[:,:2]


def oracle(x,c,u,targets):
    # Original full-ED amplitudes, then project the partner index. Never
    # recompute amplitudes using a truncated occupied Fock matrix.
    t=np.zeros((len(targets),4,7,7))
    for beta,w in zip(c['laplace_roots'],c['laplace_weights']):
        b=np.einsum('ij,Pjb,ba->Pia',expm(beta*x['foo']),x['B'],expm(-beta*x['fvv']))
        t-=w*np.einsum('Pia,Pjb->ijab',b[:,targets],b)
    g=np.einsum('Pia,Pjb->ijab',x['B'][:,targets],x['B'])
    t=np.einsum('ijab,jk->ikab',t,u)
    g=np.einsum('ijab,jk->ikab',g,u)
    s=2*g-g.swapaxes(-1,-2)
    w=.5*(np.einsum('ikac,ikbc->ab',t,s)+np.einsum('ikca,ikcb->ab',t,s))
    return .5*(w+w.T)


@pytest.mark.parametrize('threads',[1,4])
@pytest.mark.parametrize('rank',[0,2,4])
def test_projected_virtual_w_matches_full_amplitude_oracle(threads,rank):
    x,c,u=fixture(); c['threads']=threads
    u=np.linalg.qr(np.random.default_rng(4).normal(size=(4,4)))[0][:,:rank]
    saved={k:a.copy() for k,a in x.items()};meta={'target_indices':[0,2]}
    f=backend.begin_sequential_matrices(x,meta,c)
    with blas_threads(1): original=backend.solve_matrices(x,meta,c)
    np.testing.assert_allclose(f['occupied'],original['occupied'],atol=2e-14)
    np.testing.assert_allclose(f['energy'],original['energy'],atol=2e-14)
    w=backend.finish_sequential_virtual(f,u)
    np.testing.assert_allclose(w,oracle(x,c,u,[0,2]),atol=2e-14)
    if rank==4:np.testing.assert_allclose(w,original['virtual'],atol=2e-14)
    for k in x:np.testing.assert_array_equal(x[k],saved[k])


def test_only_retained_projector_matters():
    x,c,u=fixture();q=np.array([[.6,-.8],[.8,.6]])
    outs=[]
    for v in [u,u@q]:
        f=backend.begin_sequential_matrices(x,{'target_indices':[1,3]},c)
        outs.append(backend.finish_sequential_virtual(f,v))
    np.testing.assert_allclose(*outs,atol=2e-14)


def test_reject_invalid_space_and_stochastic():
    x,c,u=fixture()
    with pytest.raises(NotImplementedError):
        backend.begin_sequential_matrices(x,{'target_indices':[0]},dict(c,mode='stochastic'))
    f=backend.begin_sequential_matrices(x,{'target_indices':[0]},c)
    with pytest.raises(ValueError,match='orthonormal'):
        backend.finish_sequential_virtual(f,2*u)


def test_molecular_full_occupied_limit_and_gradient_guard():
    from pyscfad import gto, scf, dlno_stc
    from pyscfad.dlno_stc import ccsd
    mol=gto.Mole(atom='O 0 0 0; H 0 0 1; H 0 1 0',basis='sto-3g',verbose=0,max_memory=10000)
    mol.build(trace_exp=False,trace_ctr_coeff=False)
    mf=scf.RHF(mol).density_fit(auxbasis='weigend');mf.kernel()
    domains=dlno_stc.build_domain_selections(mf,frozen=1,boys_grouping='atom',full_domain=True)
    controls=dict(mode='deterministic',threads=2)
    def plan(sequential):
        return ccsd.build_static_selections(mf,domains,thresh_occ=0.,thresh_vir=1e-5,
            matrix_controls=controls,baseline_controls=controls,sequential_virtual=sequential)
    options=dict(matrix_controls=controls,baseline_controls=controls,include_hf=True)
    a=ccsd.kernel(mf,static=plan(False),**options)
    b=ccsd.kernel(mf,static=plan(True),**options)
    np.testing.assert_allclose(a,b,atol=2e-10,rtol=0)
    # With a finite occupied cutoff, verify an actual molecular forward solve.
    finite=ccsd.build_static_selections(mf,domains,thresh_occ=3e-5,thresh_vir=1e-5,
        matrix_controls=controls,baseline_controls=controls,sequential_virtual=True)
    assert np.isfinite(ccsd.kernel(mf,static=finite,**options))
    with pytest.raises(NotImplementedError,match='forward energies only'):
        ccsd.value_and_grad(mol,lambda m: mf,static=plan(True),**options)


@pytest.mark.parametrize('option',['sequential_virtual','scale_lis_by_target_count'])
def test_checkpoint_default_compatibility_and_method_identity(option):
    from dataclasses import replace
    from pyscfad.dlno_stc.lis import LISSelections
    from pyscfad.dlno_stc import _restart as storage
    from pyscfad.dlno_stc._ccsd_restart_reference import _selection_check
    # Scalar-only selection scaffold tests the exact historical wire format.
    old = LISSelections(None, 3e-5, 1e-5, 1e-6, ())
    encoded = storage._encode_value(old)
    assert 'sequential_virtual' not in dict(encoded['fields'])
    restored = storage._decode_value(encoded, None)
    assert restored.sequential_virtual is False
    assert storage.scientific_digest(restored) == storage.scientific_digest(old)
    assert 'scale_lis_by_target_count' not in dict(encoded['fields'])
    new = replace(old, **{option: True})
    assert getattr(storage._decode_value(storage._encode_value(new), None), option)
    assert storage.scientific_digest(new) != storage.scientific_digest(old)
    with pytest.raises(storage.RestartMismatchError, match=option):
        _selection_check(new, old)
