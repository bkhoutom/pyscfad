"""OV NAF adjoints, native MP2/W response, and molecular ED pass counts."""
from copy import copy
import inspect

import jax
import numpy as np
import pytest

from pyscfad.dlno_stc._mp2_aux import compress_ov
from pyscfad.dlno_stc import driver, ccsd, backend
from pyscfad import dlno_stc
from pyscfad.dlno_stc.test.test_backend import native
from pyscfad.dlno_stc.test.test_full_kernel import grid
from pyscfad.dlno_stc.test.test_boys_prepare import boys_water_dimer
from pyscfad.dlno_stc.test.test_correlation_auxbasis import molecule, _mf, _controls, RIFIT


def factors(shape, spectrum=None):
    rng = np.random.default_rng(728)
    m, n = shape[0], np.prod(shape[1:])
    k = min(m, n)
    u = np.linalg.qr(rng.normal(size=(m, k)))[0]
    v = np.linalg.qr(rng.normal(size=(n, k)))[0]
    values = np.geomspace(.2, .002, k) if spectrum is None else np.asarray(spectrum)
    return ((u * values) @ v.T).reshape(shape)


@pytest.mark.parametrize('shape,spectrum', [((5,2,4), None), ((10,2,3), None),
    ((5,2,4), [.2,.2,.1,.01,.01])])
def test_anchored_factor_pullback_with_subspace_response(shape, spectrum):
    b = factors(shape, spectrum)
    out, pb, selection, info = compress_ov(b, with_grad=True)
    assert 0 < selection.rank < shape[0]
    rng = np.random.default_rng(46)
    seed, direction = rng.normal(size=out.shape), rng.normal(size=b.shape)
    analytic = np.sum(pb(seed) * direction)
    h = 1e-6
    def objective(delta):
        return np.sum(compress_ov(b+delta*direction, selection=selection)[0] * seed)
    finite = (objective(h)-objective(-h))/(2*h)
    np.testing.assert_allclose(analytic, finite, atol=2e-8, rtol=2e-7)
    frozen_basis_bar = selection.anchor @ seed.reshape(selection.rank,-1)
    assert abs(np.sum(frozen_basis_bar.reshape(b.shape)*direction)-finite) > 1e-4
    assert info['pullback_s'] > 0


@pytest.mark.parametrize('tol', [None, 0., 1e-12, 10.])
def test_disabled_full_rank_and_zero_rank(tol):
    b = factors((4,2,3))
    out, pb, selection, _ = compress_ov(b, tol, with_grad=True)
    seed = np.ones_like(out)
    if tol == 10.:
        assert out.shape == (0,2,3)
        np.testing.assert_array_equal(pb(seed), np.zeros_like(b))
    elif tol in (None, 0.):
        assert out is b
        np.testing.assert_array_equal(pb(seed), seed)
    else:
        np.testing.assert_array_equal(out, b)
        np.testing.assert_array_equal(pb(seed), seed)


@pytest.mark.parametrize('tol', [-1., float('nan'), float('inf'), True])
def test_invalid_mp2_cutoff(tol):
    with pytest.raises((ValueError,TypeError), match='mp2_aux_svd_tol'):
        compress_ov(np.ones((2,1,2)),tol)


def test_public_defaults():
    for api in (dlno_stc.kernel,dlno_stc.value_and_grad,
                driver.kernel,driver.value_and_grad,ccsd.kernel,ccsd.value_and_grad):
        assert inspect.signature(api).parameters['mp2_aux_svd_tol'].default == 1e-3


@pytest.mark.parametrize('solver', ['energy', 'W', 'D'])
def test_native_mp2_and_density_factor_gradients(solver):
    native()
    b = factors((5,2,4))
    out,pb,sel,_ = compress_ov(b,with_grad=True)
    x = dict(foo=np.diag([-1.,-.7]), fvv=np.diag([.2,.4,.6,.8]), B=out)
    ctrl=grid(threads=1); ctrl.pop('virtual_block_size')
    meta=dict(target_indices=(0,))
    occ_seed=np.array([[.7,.3],[.3,-.2]])
    vir_seed=np.random.default_rng(10).normal(size=(4,4)); vir_seed+=vir_seed.T
    def objective(x, reverse=False):
        if solver=='energy':
            result=backend.solve(x,meta,ctrl,with_grad=reverse)
            return result['energy'],result.get('cotangents')
        result=backend.solve_matrices(x,meta,ctrl,matrix_kind=solver)
        value=.6*result['energy']+np.sum(result['occupied']*occ_seed)+np.sum(result['virtual']*vir_seed)
        bars=backend.pullback_matrices(x,result,energy_bar=.6,occupied_bar=occ_seed,
                                      virtual_bar=vir_seed) if reverse else None
        return value,bars
    value,bars=objective(x,True)
    direction=np.random.default_rng(15).normal(size=b.shape)
    analytic=np.sum(pb(bars['B'])*direction)
    h=1e-6
    def moved(sign):
        return objective(dict(x,B=compress_ov(b+sign*h*direction,selection=sel)[0]))[0]
    np.testing.assert_allclose(analytic,(moved(1)-moved(-1))/(2*h),atol=2e-10,rtol=2e-6)


@pytest.mark.parametrize('cache_mode', ['direct','half'])
@pytest.mark.parametrize('return_frame', [False,True])
def test_ed_coordinate_response_and_one_integral_reverse(boys_water_dimer,monkeypatch,cache_mode,return_frame):
    from pyscfad.dlno_stc import build_domain_groups
    mf,common,static=boys_water_dimer
    monkeypatch.setenv('PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE',cache_mode)
    group=build_domain_groups(static)[0]
    fragment=static.fragments[group.representative_index]
    direction=np.random.default_rng(7).normal(size=mf.mol.coords.shape)
    ctrl=grid(threads=1); ctrl.pop('virtual_block_size')
    receipts=[]
    ranks=[]
    def evaluate(current,reverse):
        state=dict(mf=current,common=common,mp2_aux_svd_tol=1e-3)
        with driver._mp2_df_scope(state,fragment.extended_atoms,fragment.pao_center_atoms,
                                  with_grad=reverse) as shared:
            prepared=driver._prepare(state,static,group,ctrl,reverse,return_frame=return_frame)
            inputs,host,pullback=prepared[:3]
            info=state['mp2_auxiliary_compression'][group.representative_index]
            assert 0 < info['retained_naux'] < info['original_naux']
            ranks.append(info['retained_naux'])
            meta=dict(target_indices=group.target_indices)
            if return_frame:
                result=backend.solve_matrices(host,meta,ctrl)
                os=np.eye(inputs['foo'].shape[0])*.2
                vs=np.eye(inputs['fvv'].shape[0])*.3
                energy=float(.7*result['energy']+np.sum(result['occupied']*os)+np.sum(result['virtual']*vs))
                if reverse:
                    bars=backend.pullback_matrices(host,result,energy_bar=.7,occupied_bar=os,virtual_bar=vs)
                    frame=prepared[3]
                    seed=(bars,(np.zeros_like(frame.occupied_coeff),np.zeros_like(frame.virtual_coeff)))
            else:
                result=backend.solve(host,meta,ctrl,with_grad=reverse)
                energy=float(result['energy'])
                if reverse:seed=result['cotangents']
            if reverse:
                mfbar,_=pullback(seed)
                geom=shared.finish()
                gradient=np.asarray(mfbar.mol.coords)+geom
                receipts.append(dict(shared.receipt))
                return energy,gradient
            shared.finish()
            return energy
    _,gradient=evaluate(mf,True)
    h=2e-5
    def moved(sign):
        current=copy(mf)
        current.mol=mf.mol.set_geom_(np.asarray(mf.mol.atom_coords())+sign*h*direction,
                                      unit='Bohr',inplace=False)
        return evaluate(current,False)
    finite=(moved(1)-moved(-1))/(2*h)
    np.testing.assert_allclose(np.sum(gradient*direction),finite,atol=2e-8,rtol=3e-5)
    assert len(set(ranks))==1
    receipt=receipts[0]
    assert receipt['geometry_passes']==receipt['metric_passes']==1
    assert receipt['raw_source_passes']==1+int(cache_mode=='direct')
    print('ED NAF gradient',cache_mode,return_frame,'rank',ranks[0],
          'analytic',np.sum(gradient*direction),'finite',finite,'receipt',receipt)


def test_public_mp2_nuclear_gradient(molecule,monkeypatch):
    monkeypatch.setenv('PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE','half')
    options=dict(boys_grouping='system',frozen=1,controls=_controls(),correlation_auxbasis=RIFIT)
    energy,gradient=dlno_stc.value_and_grad(molecule,_mf,**options)
    direction=np.zeros((molecule.natm,3));direction[1,2]=1.;direction[2,1]=-.4
    h=1e-4
    def evaluate(delta):
        moved=molecule.set_geom_(np.asarray(molecule.atom_coords())+delta*direction,
                                  unit='Bohr',inplace=False)
        return dlno_stc.kernel(_mf(moved),**options)
    np.testing.assert_allclose(energy,evaluate(0),atol=2e-10,rtol=0)
    finite=(evaluate(h)-evaluate(-h))/(2*h)
    analytic=np.sum(np.asarray(gradient.coords)*direction)
    np.testing.assert_allclose(analytic,finite,atol=3e-7,rtol=2e-5)
    np.testing.assert_allclose(np.sum(gradient.coords,axis=0),0,atol=2e-8)
    print('public MP2 NAF coordinate gradient',analytic,finite,'error',abs(analytic-finite))


def test_mpi_installs_root_mp2_cutoff():
    from pyscfad.dlno_stc._mpi import _install_correlation_config
    state = {}
    _install_correlation_config(state, dict(mp2_aux_svd_tol=3e-3))
    assert state['mp2_aux_svd_tol'] == 3e-3
    _install_correlation_config(state, dict(mp2_aux_svd_tol=None))
    assert state['mp2_aux_svd_tol'] is None


def test_public_gradient_with_nonzero_discarded_weight(molecule, monkeypatch):
    # A larger orbital space exercises actual NAF truncation, not just the
    # auxiliary null space of the minimal-basis molecular checks.
    mol = molecule.copy()
    mol.basis = 'cc-pvdz'
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    diagnostics = []
    from pyscfad.dlno_stc import _mp2_aux
    original = _mp2_aux.compress_ov
    def counted(*args, **kwargs):
        result = original(*args, **kwargs)
        diagnostics.append(result[3])
        return result
    monkeypatch.setattr(_mp2_aux, 'compress_ov', counted)
    test_public_mp2_nuclear_gradient(mol, monkeypatch)
    assert len(diagnostics) == 4
    assert all(d['cutoff'] == 1e-3 and d['discarded_weight'] > 1e-5
               for d in diagnostics)
    assert len({d['retained_naux'] for d in diagnostics}) == 1
    print('nonzero molecular NAF truncation', diagnostics[0])
