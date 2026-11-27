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


def test_molecular_full_occupied_forward_limit():
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


def dense_outputs(x, c, u, targets, matrix_kind="W"):
    no, nv = x['B'].shape[1:]
    t = np.zeros((len(targets), no, nv, nv))
    for beta, weight in zip(c['laplace_roots'], c['laplace_weights']):
        b = np.einsum('ij,Pjb,ba->Pia', expm(beta*x['foo']), x['B'], expm(-beta*x['fvv']))
        t -= weight*np.einsum('Pia,Pjb->ijab', b[:, targets], b)
    g = np.einsum('Pia,Pjb->ijab', x['B'][:, targets], x['B'])
    g = 2*g-g.swapaxes(-1, -2)
    energy = np.sum(t*g)
    if matrix_kind == 'D':
        g = 2*t-t.swapaxes(-1, -2)
    wo = np.einsum('ijab,ikab->jk', t, g)
    wo = (wo+wo.T)/2
    if u is None:
        return energy, wo
    tp, gp = (np.einsum('ijab,jk->ikab', a, u) for a in (t, g))
    wv = .5*(np.einsum('ikac,ikbc->ab', tp, gp)+np.einsum('ikca,ikcb->ab', tp, gp))
    return energy, wo, (wv+wv.T)/2


@pytest.mark.parametrize('matrix_kind', ['W', 'D'])
@pytest.mark.parametrize('threads', [1, 4, 32])
@pytest.mark.parametrize('rank', [0, 2, 4])
def test_sequential_native_reverse_inputs_and_rotation_fd(threads, rank, matrix_kind):
    x, c, _ = fixture(); c['threads'] = threads
    rng = np.random.default_rng(97)
    u = np.linalg.qr(rng.normal(size=(4, 4)))[0][:, :rank]
    targets = [0, 2]
    eb, ob, vb = .37, rng.normal(size=(4, 4)), rng.normal(size=(7, 7))
    forward = backend.begin_sequential_matrices(x, {'target_indices': targets}, c, matrix_kind=matrix_kind)
    forward['virtual'] = backend.finish_sequential_virtual(forward, u)
    expected_e, expected_o, expected_v = dense_outputs(x, c, u, targets, matrix_kind)
    for name, expected in [('energy', expected_e), ('occupied', expected_o), ('virtual', expected_v)]:
        np.testing.assert_allclose(forward[name], expected, atol=3e-14, rtol=2e-12)
    rotation_bars = []
    def pullback(bar):
        rotation_bars.append(bar.copy())
        return np.zeros_like(ob)
    bars = backend.pullback_sequential_matrices(x, forward, energy_bar=eb,
        occupied_bar=ob, virtual_bar=vb, rotation_pullback=pullback)
    def loss(inputs, rotation):
        e, o, v = dense_outputs(inputs, c, rotation, targets, matrix_kind)
        return eb*e + np.sum(ob*o) + np.sum(vb*v)
    step = 2e-5
    for key in x:
        direction = rng.normal(size=x[key].shape)
        if key != 'B': direction = (direction+direction.T)/2
        direction /= np.linalg.norm(direction)
        xp, xm = dict(x), dict(x)
        xp[key], xm[key] = x[key]+step*direction, x[key]-step*direction
        fd = (loss(xp, u)-loss(xm, u))/(2*step)
        np.testing.assert_allclose(np.sum(bars[key]*direction), fd, rtol=2e-7, atol=2e-10)
    direction = rng.normal(size=u.shape)
    # Oracle also defines the extension off the orthonormal manifold, allowing
    # every rotation entry to be checked without an extra QR derivative.
    fd = (loss(x, u+step*direction)-loss(x, u-step*direction))/(2*step)
    np.testing.assert_allclose(np.sum(rotation_bars[0]*direction), fd, rtol=2e-7, atol=2e-10)
    assert len(rotation_bars) == 1
    if rank == 4:
        plain = backend.solve_matrices(x, {'target_indices': targets}, c, matrix_kind=matrix_kind)
        reference = backend.pullback_matrices(x, plain, energy_bar=eb, occupied_bar=ob, virtual_bar=vb)
        for key in bars:
            np.testing.assert_allclose(bars[key], reference[key], atol=2e-13, rtol=2e-12)


@pytest.mark.parametrize('matrix_kind', ['W', 'D'])
@pytest.mark.parametrize('threads', [1, 4, 32])
def test_sequential_selected_occupied_chain_fd(threads, matrix_kind):
    import jax
    import jax.numpy as jnp
    from pyscfad.dlno_stc._retained_space import retained_basis
    x, c, _ = fixture(); c['threads'] = threads
    targets = [0, 2]
    forward = backend.begin_sequential_matrices(x, {'target_indices': targets}, c, matrix_kind=matrix_kind)
    u, tape = jax.vjp(retained_basis((0, 1)), jnp.asarray(forward['occupied']))
    forward['virtual'] = backend.finish_sequential_virtual(forward, u)
    rng = np.random.default_rng(78)
    vb = rng.normal(size=(7, 7))
    bars = backend.pullback_sequential_matrices(x, forward, energy_bar=.3,
        occupied_bar=np.zeros((4, 4)), virtual_bar=vb,
        rotation_pullback=lambda bar: np.asarray(tape(jnp.asarray(bar))[0]))
    def loss(inputs):
        _, wo = dense_outputs(inputs, c, None, targets, matrix_kind)
        _, u = np.linalg.eigh(wo)
        e, _, v = dense_outputs(inputs, c, u[:, :2], targets, matrix_kind)
        return .3*e+np.sum(vb*v)
    step = 2e-5
    for key in x:
        direction = rng.normal(size=x[key].shape)
        if key != 'B': direction = (direction+direction.T)/2
        direction /= np.linalg.norm(direction)
        xp, xm = dict(x), dict(x)
        xp[key], xm[key] = x[key]+step*direction, x[key]-step*direction
        fd = (loss(xp)-loss(xm))/(2*step)
        np.testing.assert_allclose(np.sum(bars[key]*direction), fd, rtol=3e-7, atol=3e-10)
    changed = dict(x, B=x['B']*1.001)
    with pytest.raises(ValueError, match='replay inputs'):
        backend.pullback_sequential_matrices(changed, forward, energy_bar=0.,
            occupied_bar=np.zeros((4, 4)), virtual_bar=vb,
            rotation_pullback=lambda bar: np.zeros((4, 4)))


@pytest.mark.parametrize('target_count', [1, 2])
def test_occupied_selector_target_metric_and_frame_response(target_count):
    """Unconstrained input bars, including the physical target complement."""
    import jax.numpy as jnp
    from types import SimpleNamespace
    from typing import NamedTuple
    from pyscfad.dlno_stc import lis
    from pyscfad.dlno_stc.domain import BoysEDFrame, TargetOccupiedData
    class Data(NamedTuple):
        s1e: object
        fragment_occupied_data: tuple
    rng = np.random.default_rng(196)
    parent = np.linalg.qr(rng.normal(size=(6, 4)))[0]
    targets = parent @ np.linalg.qr(rng.normal(size=(4, 4)))[0][:, :target_count]
    data = Data(jnp.eye(6), tuple(TargetOccupiedData(jnp.asarray(targets[:, i:i+1]))
                                 for i in range(target_count)))
    domains = SimpleNamespace(boys_grouping='atom', fragments=(None,)*target_count,
                              physical_groups=(tuple(range(target_count)),))
    frame = BoysEDFrame(jnp.asarray(parent), jnp.zeros((6, 2)), np.arange(6), np.arange(2), 0, ())
    w = rng.normal(size=(4, 4)); w = (w+w.T)/2
    target = lis._domain_targets(data, domains, 0, frame)
    external = lis._external_map(jnp.eye(4), target)
    columns = lis._range_columns(external)
    _, compact = lis._compact_mapped_density(external, w, columns)
    values = np.linalg.eigvalsh(compact)
    cutoff = np.sort(np.abs(values))[-2:].mean()
    selection = SimpleNamespace(full_occupied_space=False, occupied_map_columns=columns,
                                occupied_lno_keep=np.flatnonzero(np.abs(values)>cutoff))
    u, tape = lis.occupied_selection_rotation(None, domains, 0, frame, w,
        common=data, threshold=cutoff, selection=selection, with_grad=True)
    assert u.shape[1] == target_count+1
    np.testing.assert_allclose(u.T@u, np.eye(u.shape[1]), atol=2e-14)
    seed = rng.normal(size=(4, 4)); seed = (seed+seed.T)/2
    bw, bd, bc = tape(jnp.asarray(2*seed@u))
    def loss(block, common, coeff):
        rotation = lis.occupied_selection_rotation(None, domains, 0,
            frame._replace(occupied_coeff=coeff), block, common=common,
            threshold=cutoff, selection=selection)
        return np.sum(seed*(rotation@rotation.T))
    directions = [rng.normal(size=w.shape), rng.normal(size=(6, 6)),
                  rng.normal(size=targets.shape), rng.normal(size=parent.shape)]
    directions[0] = (directions[0]+directions[0].T)/2
    directions[1] = (directions[1]+directions[1].T)/2
    step = 1e-5
    errors = []
    for kind, direction in enumerate(directions):
        direction /= np.linalg.norm(direction)
        vals = []
        for sign in (1, -1):
            ww, dd, cc = w, data, parent
            if kind == 0: ww = w+sign*step*direction
            if kind == 1: dd = data._replace(s1e=data.s1e+sign*step*direction)
            if kind == 2:
                dd = data._replace(fragment_occupied_data=tuple(
                    TargetOccupiedData(targets[:, i:i+1]+sign*step*direction[:, i:i+1])
                    for i in range(target_count)))
            if kind == 3: cc = parent+sign*step*direction
            vals.append(loss(ww, dd, cc))
        analytic = [bw, bd.s1e, jnp.concatenate(tuple(x.iao_coeff for x in bd.fragment_occupied_data), axis=1), bc][kind]
        expected = (vals[0]-vals[1])/(2*step)
        actual = np.sum(analytic*direction)
        errors.append(abs(actual-expected))
        np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-7)
    assert max(errors) < 2e-8
