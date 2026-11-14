"""Lambda-Boys objective, local Hessian, and implicit Fock response."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm
from pyscf import scf
from pyscfad import gto
from pyscfad.lo import boys
from pyscfad.lo.lambda_boys import LambdaBoys, lambda_boys, at_stationary_frame, _operators
from pyscfad.soscf.ciah import extract_rotation


@pytest.fixture(scope='module')
def reference():
    mol = gto.Mole(atom='O 0 0 0; H .12 -.74 .57; H -.05 .83 .62', basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = scf.RHF(mol.to_pyscf()).run(conv_tol=1e-13)
    return mol, jnp.asarray(mf.mo_coeff[:, 1:5]), jnp.asarray(mf.get_fock())


@pytest.mark.parametrize('weight', [0., 100.])
def test_objective_gradient_hessian_against_direct_expression(reference, weight):
    mol, coeff, fock = reference
    loc = LambdaBoys(mol, coeff, fock, lambda_=weight)
    rng = np.random.default_rng(61)
    a = rng.normal(size=(4, 4))*.1;u = expm(a-a.T)
    c = coeff @ u
    dip = boys.dipole_integral(mol, c)
    r2 = jnp.einsum('pi,pi->', c, mol.intor('int1e_r2') @ c)
    def objective(x):
        v = extract_rotation(x)
        d = v.T @ dip @ v
        f = v.T @ (c.T @ fock @ c) @ v
        off = f-jnp.diag(jnp.diag(f))
        return r2-jnp.einsum('aii,aii->', d, d)+weight*jnp.sum(off**2)
    x = jnp.zeros(6)
    g = jax.grad(objective)(x);h = jax.jacfwd(jax.grad(objective))(x)
    with loc._cache_dipoles():
        actual_g, hop, diagonal = loc.gen_g_hop(u)
        np.testing.assert_allclose(loc.cost_function(u), objective(x), atol=1e-11)
        np.testing.assert_allclose(actual_g, g, atol=2e-11)
        direction = rng.normal(size=6)
        np.testing.assert_allclose(hop(direction), h @ direction, atol=2e-10)
        np.testing.assert_allclose(diagonal, np.diag(h), atol=2e-10)
    if weight == 0:
        ordinary = boys.Boys(mol, coeff)
        with ordinary._cache_dipoles():
            np.testing.assert_allclose(loc.cost_function(u), ordinary.cost_function(u), atol=1e-12)


def test_forward_and_fock_response(reference):
    mol, coeff, fock = reference
    options = dict(init_guess=np.eye(4), conv_tol=1e-12, conv_tol_grad=1e-8,
                   gmres_options=dict(tol=1e-9, atol=1e-11, restart=20, maxiter=100))
    localized = lambda_boys(mol, coeff, fock, **options)
    loc = LambdaBoys(mol, coeff, fock)
    overlap = mol.intor('int1e_ovlp')
    rotation = np.asarray(coeff.T @ overlap @ localized)
    np.testing.assert_allclose(localized.T @ overlap @ localized, np.eye(4), atol=1e-12)
    assert np.linalg.norm(loc.get_grad(rotation)) < 1e-7
    assert loc.cost_function(rotation) <= loc.cost_function(np.eye(4))+1e-10
    pure = boys.boys(mol, coeff, **options)
    def offnorm(c):
        f = c.T @ fock @ c
        return float(jnp.sum((f-jnp.diag(jnp.diag(f)))**2))
    assert offnorm(localized) < offnorm(pure)
    dip = mol.intor('int1e_r')
    weights = jnp.arange(1.,5.)
    def observable(f, reuse=False):
        c = (at_stationary_frame(mol, localized, f, conv_tol_grad=1e-7,
                                gmres_options=options['gmres_options']) if reuse else
             lambda_boys(mol, localized, f, **{**options,'max_saddle_escapes':0}))
        return jnp.sum(weights*jnp.einsum('ui,uv,vi->i', c, dip[0], c))
    rng = np.random.default_rng(77)
    direction = rng.normal(size=fock.shape)*.01;direction = (direction+direction.T)/2
    derivative = float(jnp.sum(jax.grad(observable)(fock)*direction))
    attached = float(jnp.sum(jax.grad(lambda f:observable(f, True))(fock)*direction))
    h=1e-4
    fd=float((observable(fock+h*direction)-observable(fock-h*direction))/(2*h))
    np.testing.assert_allclose(derivative,fd,atol=2e-6,rtol=2e-4)
    np.testing.assert_allclose(attached,derivative,atol=2e-7,rtol=1e-4)
    assert abs(derivative)>1e-6


def test_options_and_invalid_weight(reference):
    from pyscfad.dlno_stc._selection import _resolve_boys_options
    mol, coeff, fock=reference
    assert _resolve_boys_options({'method':'lambda_boys'})=={'method':'lambda_boys'}
    with pytest.raises(ValueError, match='requires method'):
        _resolve_boys_options({'lambda_':100})
    for value in (-1.,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='finite and nonnegative'):
            lambda_boys(mol,coeff,fock,lambda_=value)


def test_geometry_response_with_live_fock_and_overlap(reference):
    mol, coeff, fock = reference
    from jax.scipy.linalg import solve_triangular
    from pyscfad.dlno_stc.domain import _rebuild_boys_targets
    from types import SimpleNamespace
    options=dict(init_guess=np.eye(4), conv_tol=1e-12, conv_tol_grad=1e-8,
                 gmres_options=dict(tol=1e-9,atol=1e-11,restart=20,maxiter=100))
    localized=lambda_boys(mol,coeff,fock,**options)
    # Fixed labels, moving AO overlap, and an explicitly geometry-dependent
    # Fock exercise the three external derivative paths without an SCF solve.
    reference_frame=SimpleNamespace(target_reference_coeff=np.asarray(localized),
        target_reference_coords=np.asarray(mol.atom_coords()),
        lo_kwargs={'method':'lambda_boys', **options})
    def observable(m, stationary=False):
        s=m.intor('int1e_ovlp');metric=coeff.T@s@coeff
        c=solve_triangular(jnp.linalg.cholesky(metric),coeff.T,lower=True).T
        live_fock=fock+.1*(m.intor('int1e_kin')-mol.intor('int1e_kin'))
        frame=_rebuild_boys_targets(SimpleNamespace(mol=m),c,reference_frame,
                                   _stationary=stationary,fock=live_fock)
        dip=m.intor('int1e_r')
        return jnp.sum(jnp.arange(1.,5.)*jnp.einsum('ui,uv,vi->i',frame,dip[0],frame))
    direction=np.random.default_rng(91).normal(size=mol.coords.shape)*.1
    grad=jax.grad(observable)(mol).coords
    attached=jax.grad(lambda m:observable(m,True))(mol).coords
    h=1e-4
    def shifted(scale):
        return mol.set_geom_(np.asarray(mol.atom_coords())+scale*direction,
                             unit='Bohr', inplace=False)
    fd=(float(observable(shifted(h)))-float(observable(shifted(-h))))/(2*h)
    np.testing.assert_allclose(np.sum(grad*direction),fd,rtol=4e-4,atol=3e-6)
    np.testing.assert_allclose(attached,grad,rtol=2e-4,atol=2e-6)


def test_stc_selection_and_stationary_replay_use_lambda_objective(reference):
    from pyscfad import scf as ad_scf
    from pyscfad.dlno_stc import build_domain_selections, driver
    mol, _, _=reference
    mf=ad_scf.RHF(mol).density_fit();mf.conv_tol=1e-12;mf.kernel()
    options={'method':'lambda_boys','conv_tol':1e-12,'conv_tol_grad':1e-8,
             'init_guess':np.eye(4)}
    static,data,data_pb,receipt=driver._select_domain_reference(mf,
        lambda reference:build_domain_selections(reference,frozen=1,
            force_full_domains=True,lo_kwargs=options))
    assert static.lo_kwargs['method']=='lambda_boys'
    common,pullback,pending=receipt.take(static)
    assert pending is None
    loc=LambdaBoys(mol,np.asarray(common.iao_coeff),np.asarray(common.fock))
    assert np.linalg.norm(loc.get_grad())<1e-7
    # Both the shared fresh-selection path and ordinary replay must solve
    # the same Fock-dependent localization problem.
    replay,_=driver._build_common_with_pullback(mf,static,data,data_pb)
    np.testing.assert_allclose(replay.iao_coeff,common.iao_coeff,atol=1e-7)
    assert callable(pullback)
