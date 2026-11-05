"""Real external solver checks for the optional shared local amplitude frame."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import stc_ccsd as adapter


def test_shared_frame_control_defaults_on_and_preserves_explicit_overrides():
    defaults = adapter.normalize_controls(None)
    assert defaults['localize'] is True
    assert defaults['shared_local_frame'] is True
    assert adapter.normalize_controls(defaults) == defaults
    assert adapter.normalize_controls({'shared_local_frame': True})['shared_local_frame'] is True
    assert adapter.normalize_controls({'shared_local_frame': False})['shared_local_frame'] is False
    assert adapter.normalize_controls({'localize': False})['localize'] is False
    for invalid in [1, 'yes', None]:
        with pytest.raises(TypeError, match='shared_local_frame'):
            adapter.normalize_controls({'shared_local_frame': invalid})


@pytest.mark.parametrize('deferred',[False,True])
def test_shared_frame_requires_full_fock_triples(deferred):
    from pyscfad.dlno_stc import ccsd
    def selection(smooth):
        return SimpleNamespace(deferred_selection={'smooth':smooth} if deferred else None,
            lis_static=SimpleNamespace(smooth_frames=() if smooth else None))
    for shared in [False,True]:
        ccsd._validate_shared_frame_triples(selection(True),True,{'shared_local_frame':shared})
        ccsd._validate_shared_frame_triples(selection(False),False,{'shared_local_frame':shared})
    ccsd._validate_shared_frame_triples(selection(False),True,{'shared_local_frame':False})
    default_controls = ccsd._cc_solver_options('stc', None)['stc_cc_controls']
    ccsd._validate_shared_frame_triples(selection(True),True,default_controls)
    with pytest.raises(ValueError,match='smooth'):
        ccsd._validate_shared_frame_triples(selection(False),True,default_controls)
    assert ccsd._cc_solver_options('deterministic', None) == {}


def test_default_stc_matches_legacy_response_without_backrotation(monkeypatch):
    from pyscfad import gto, scf
    from pyscfad.lno.ccsd import RCCSD, FragmentAmplitudeResult, ccsd_fragment_energy
    pytest.importorskip('stc_cc_jax.cc.rccsd')
    from pyscfad import lib
    mol=gto.Mole(atom='H 0 0 0; H 0 0 0.8',basis='6-31g',verbose=0)
    mol.build(trace_exp=False,trace_ctr_coeff=False)
    mf=scf.RHF(mol).density_fit();mf.kernel()
    mcc=RCCSD(mf); eris=mcc.ao2mo();_,t1,t2=mcc.init_amps(eris)
    settings=dict(max_cycle=5,average=3,target_error=.001,
        sampled_terms=[0],nrun=2,init_nsamples=100,max_nsamples_var=200,
        min_nsamples=20,batch_size=256,grad_max_cycle=4,grad_average=2,vjp_nsamples=100)
    prj=jnp.array([[.8]])
    def objective(fock,projector,shared):
        changed=SimpleNamespace(**dict(vars(eris),fock=fock))
        controls = settings if shared else dict(settings, shared_local_frame=False)
        answer=adapter.make_solver(controls)(mcc,changed,t1,t2,prjlo=projector)
        if shared:
            assert isinstance(answer,FragmentAmplitudeResult)
            return ccsd_fragment_energy(answer.eris,answer.t1,answer.t2,answer.prjlo)
        a,b,_=answer
        return ccsd_fragment_energy(changed,a,b,projector)
    expected,expected_bar=jax.value_and_grad(lambda f,p:objective(f,p,False),(0,1))(eris.fock,prj)
    def forbidden(*a,**kw):
        raise AssertionError('shared path must not backrotate even for diagnostics')
    monkeypatch.setattr(adapter,'_backrotate_amplitudes',forbidden)
    frames=[]
    working=adapter._working_rotations
    def observe(*a,**kw):
        q=working(*a,**kw);frames.append(q);return q
    monkeypatch.setattr(adapter,'_working_rotations',observe)
    actual,actual_bar=jax.value_and_grad(lambda f,p:objective(f,p,True),(0,1))(eris.fock,prj)
    assert len(frames)==1, 'one fixed frame must serve the forward and adjoint'
    np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-13)
    for got,want in zip(actual_bar,expected_bar):
        np.testing.assert_allclose(got,want,rtol=1e-9,atol=1e-11)
    assert np.linalg.norm(np.asarray(actual_bar[1]))>1e-6
    result=adapter.make_solver(settings)(mcc,eris,t1,t2,prjlo=prj)
    assert result.diagnostics['shared_local_frame'] is True
    assert result.diagnostics['energy_frame']=='stc_local'
    assert result.eris.Lvv.ndim==2
    # Accessing the rotate-triples block is lazy, cached and numerically local.
    no,nv=result.t1.shape
    expected_vvv=jnp.einsum('Pia,Pbc->iabc',result.eris.Lov,lib.unpack_tril(result.eris.Lvv))
    cached=result.eris.ovvv
    assert result.eris.ovvv is cached
    np.testing.assert_allclose(lib.unpack_tril(cached.reshape(no*nv,-1)).reshape(no,nv,nv,nv),expected_vvv,atol=1e-13)
