"""Scientific checks for the whole-LIS smooth CCSD/(T) path."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc.smooth import (
    AnchoredFrame, noncanonical_mp2,
    select_frame, rebuild_frame,
)
from pyscfad.dlno_stc.smooth_triples import make_grid, make_solver
from pyscfad.dlno_stc.test._triples_dressing_reference import symmetric_exp
from pyscfad.dlno_stc.test.test_triples import triples_fixture, triples_reference


def test_projector_internal_degeneracy():
    matrix = jnp.diag(jnp.array([-.8, .2, .2, 1.]))
    rng = np.random.default_rng(9)
    d = rng.normal(size=(4, 4)); d = (d + d.T) / 2
    from pyscfad.dlno_stc._retained_space import retained_projector
    projector = retained_projector((1, 2))
    _, tangent = jax.jvp(projector, (matrix,), (jnp.asarray(d),))
    step = 1e-5
    fd = (projector(matrix + step*d) - projector(matrix - step*d)) / (2*step)
    np.testing.assert_allclose(tangent, fd, atol=1e-8)
    assert np.isfinite(tangent).all()


def test_exponential_degenerate_derivative():
    a = jnp.eye(3) * .4
    direction = jnp.array([[.2, .5, .1], [.5, -.4, .3], [.1, .3, .7]])
    _, tangent = jax.jvp(symmetric_exp, (a,), (direction,))
    np.testing.assert_allclose(tangent, np.exp(.4) * direction, atol=1e-13)


def test_full_block_frame_skips_projectors_and_retains_moving_subspace(monkeypatch):
    from pyscfad.dlno_stc import smooth
    rng = np.random.default_rng(161)
    coeff = jnp.asarray(np.linalg.qr(rng.normal(size=(6, 4)))[0])
    overlap = jnp.eye(6)
    fock = coeff @ jnp.diag(jnp.array([-.8, -.8, .2, .2])) @ coeff.T
    frame = select_frame(coeff, fock)

    def forbidden(*args):
        raise AssertionError('a full block must not construct a spectral projector')

    monkeypatch.setattr(jnp.linalg, 'eigh', forbidden)
    dc = jnp.asarray(rng.normal(size=(6, 4)) * .1)
    ds = jnp.asarray(rng.normal(size=(6, 6)) * .1); ds = (ds + ds.T) / 2
    df = jnp.asarray(rng.normal(size=(6, 6))); df = (df + df.T) / 2

    def moving(x):
        metric = overlap + x * ds
        current = smooth.orthonormalize(coeff + x * dc, metric)
        return rebuild_frame(current, metric, frame)

    q, dq = jax.jvp(moving, (0.,), (1.,))
    step = 1e-5
    np.testing.assert_allclose(dq, (moving(step) - moving(-step)) / (2*step),
                               rtol=2e-7, atol=2e-10)
    np.testing.assert_allclose(q.T @ q, np.eye(4), atol=1e-13)
    np.testing.assert_allclose(q @ q.T, coeff @ coeff.T, atol=1e-13)
    np.testing.assert_allclose(dq.T @ q + q.T @ dq + q.T @ ds @ q, 0., atol=1e-12)
    assert np.linalg.norm(dq) > .01
    _, response = jax.jvp(lambda f: rebuild_frame(coeff, overlap, frame), (fock,), (df,))
    np.testing.assert_array_equal(response, 0.)


def test_block_frame_preserves_full_matrix_response():
    a = jnp.diag(jnp.array([-.8, -.799, .2, .5, .502]))
    identity = jnp.eye(5)
    frame = select_frame(identity, a)
    rng = np.random.default_rng(17)
    b = rng.normal(size=(5, 5)); b = jnp.asarray((b+b.T)/2)
    def in_frame(fock):
        q = rebuild_frame(identity, identity, frame)
        return jnp.sum((q.T @ b @ q) * symmetric_exp(-.6 * (q.T @ fock @ q)))
    def direct(fock):
        return jnp.sum(b * symmetric_exp(-.6*fock))
    actual, derivative = jax.value_and_grad(in_frame)(a)
    expected, reference = jax.value_and_grad(direct)(a)
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    np.testing.assert_allclose(derivative, reference, atol=1e-12)


def inputs_and_reference():
    x = {k: jnp.asarray(v) for k, v in triples_fixture().items()}
    x['U'] = jnp.eye(2)
    x['fov'] = jnp.array([[.03, -.02, .01], [-.04, .05, -.06]])
    foo, fvv = jnp.diag(x['eocc']), jnp.diag(x['evir'])
    frames = (AnchoredFrame(np.eye(2)), AnchoredFrame(np.eye(3)))
    grid = make_grid(x['eocc'], x['evir'])
    return x, foo, fvv, frames, grid


def evaluate(solver, x, foo, fvv):
    no = len(foo)
    fock = jnp.block([[foo, x['fov']], [x['fov'].T, fvv]])
    from pyscfad import lib
    eris = SimpleNamespace(Lov=x['Bvo'], Loo=x['Boo'], Lvv=lib.pack_tril(x['Bvv']), fock=fock)
    return solver(None, eris, x['U'].T, t1=x['t1'], t2=x['t2'])[0]


def canonical_reference(x, foo, fvv):
    eo, o = jnp.linalg.eigh(foo)
    ev, v = jnp.linalg.eigh(fvv)
    transformed = dict(t1=o.T @ x['t1'] @ v,
        t2=jnp.einsum('ip,jq,ar,bs,ijab->pqrs', o, o, v, v, x['t2'], optimize='optimal'),
        Bvo=jnp.einsum('ip,Pia,ar->Ppr', o, x['Bvo'], v),
        Boo=jnp.einsum('ip,Pij,jq->Ppq', o, x['Boo'], o),
        Bvv=jnp.einsum('ar,Pab,bs->Prs', v, x['Bvv'], v),
        eocc=eo, evir=ev, U=(jnp.eye(len(eo)) if x['U'].shape == (len(eo), len(eo))
                            else o.T @ x['U']), fov=o.T @ x['fov'] @ v)
    return triples_reference(**transformed)


def test_deterministic_laplace_energy_and_all_cotangents():
    x, foo, fvv, frames, grid = inputs_and_reference()
    # Off-diagonal Fock tests the noncanonical equation, not just diagonal scaling.
    foo = foo.at[0, 1].set(.02).at[1, 0].set(.02)
    fvv = fvv.at[0, 1].set(.03).at[1, 0].set(.03)
    solver = make_solver({'mode': 'deterministic'}, grid, with_grad=True)
    actual, bars = jax.value_and_grad(lambda x, fo, fv: evaluate(solver, x, fo, fv),
                                    argnums=(0, 1, 2))(x, foo, fvv)
    reference, expected = jax.value_and_grad(canonical_reference, argnums=(0, 1, 2))(x, foo, fvv)
    np.testing.assert_allclose(actual, reference, rtol=2e-8, atol=1e-12)
    for name in ('Boo', 'Bvv'):
        bars[0][name] = (bars[0][name] + bars[0][name].swapaxes(1, 2)) / 2
        expected[0][name] = (expected[0][name] + expected[0][name].swapaxes(1, 2)) / 2
    for values in (bars, expected):
        values[0]['t2'] = (values[0]['t2'] + values[0]['t2'].transpose(1, 0, 3, 2)) / 2
    for got, want in zip(jax.tree_util.tree_leaves(bars), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=2e-10)


def test_mp2_resolvent_offdiagonal_fock_response():
    x, foo, fvv, _, _ = inputs_and_reference()
    rhs = jnp.einsum('Pia,Pjb->iajb', x['Bvo'], x['Bvo'])
    def energy(fo, fv):
        fock = jnp.block([[fo, x['fov']], [x['fov'].T, fv]])
        return noncanonical_mp2(SimpleNamespace(fock=fock, ovov=rhs), jnp.eye(2))[0]
    _, tangent = jax.jvp(energy, (foo, fvv), (jnp.zeros_like(foo), jnp.ones_like(fvv)))
    step = 1e-5
    fd = (energy(foo, fvv+step) - energy(foo, fvv-step)) / (2*step)
    np.testing.assert_allclose(tangent, fd, rtol=1e-7, atol=1e-10)


def test_existing_deterministic_triples_vjp_reference():
    """Guard the conventional force reference against a missing/scaled VJP."""
    import sys
    from pyscfad import lib
    from pyscfad.lno import ccsd_t
    x, foo, fvv, _, _ = inputs_and_reference()
    def conventional(t1):
        eris = SimpleNamespace(ovvv=None, mo_energy=jnp.r_[x['eocc'], x['evir']],
            fock=jnp.block([[foo, x['fov']], [x['fov'].T, fvv]]),
            ovoo=jnp.einsum('Pia,Pjk->iajk', x['Bvo'], x['Boo']),
            ovov=jnp.einsum('Pia,Pjb->iajb', x['Bvo'], x['Bvo']),
            Lov=x['Bvo'], Lvv=lib.pack_tril(x['Bvv']))
        return ccsd_t.kernel(SimpleNamespace(max_memory=800000, stdout=sys.stdout, verbose=0),
                             eris, x['U'].T, t1=t1, t2=x['t2'], verbose=0)
    value, derivative = jax.value_and_grad(conventional)(x['t1'])
    expected, reference = jax.value_and_grad(lambda t1: triples_reference(**{**x, 't1': t1}))(x['t1'])
    np.testing.assert_allclose(value, expected, atol=1e-13)
    np.testing.assert_allclose(derivative, reference, atol=1e-12)


def test_partial_target_laplace_all_responses():
    """Noncommuting fragment projector, off-diagonal Fock and nonzero Fov."""
    x, foo, fvv, frames, grid = inputs_and_reference()
    x['U'] = jnp.array([[.6], [-.8]])
    foo = foo.at[0, 1].set(.02).at[1, 0].set(.02)
    fvv = fvv.at[0, 1].set(.03).at[1, 0].set(.03)
    solver = make_solver({'mode': 'deterministic'}, grid, with_grad=True)
    actual, bars = jax.value_and_grad(lambda x, fo, fv: evaluate(solver, x, fo, fv),
                                     argnums=(0, 1, 2))(x, foo, fvv)
    expected, refbars = jax.value_and_grad(canonical_reference, argnums=(0, 1, 2))(x, foo, fvv)
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-12)
    for collection in (bars, refbars):
        for name in ('Boo', 'Bvv'):
            collection[0][name] = (collection[0][name] + collection[0][name].swapaxes(1, 2))/2
        collection[0]['t2'] = (collection[0]['t2'] + collection[0]['t2'].transpose(1, 0, 3, 2))/2
    for got, want in zip(jax.tree_util.tree_leaves(bars), jax.tree_util.tree_leaves(refbars)):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=2e-10)


def test_truncated_anchored_frame_w_response_and_internal_degeneracy():
    from pyscfad.dlno_stc.lis import _smooth_density_space
    rng = np.random.default_rng(83)
    q = jnp.asarray(np.linalg.qr(rng.normal(size=(6, 6)))[0])
    matrix = q @ jnp.diag(jnp.array([0., .01, .01, .3, .3, .8])) @ q.T
    internal = q[:, :1]
    identity = jnp.eye(6)
    frame = AnchoredFrame(np.asarray(q[:, [0, 3, 4, 5]]))
    d = rng.normal(size=(6, 6)); d = jnp.asarray((d+d.T)/2)
    def project(w):
        c = _smooth_density_space(w, internal, (3, 4, 5), False, identity, identity, frame)
        return c @ c.T
    _, tangent = jax.jvp(project, (matrix,), (d,))
    step = 1e-5
    fd = (project(matrix+step*d)-project(matrix-step*d))/(2*step)
    np.testing.assert_allclose(tangent, fd, rtol=1e-6, atol=1e-7)
    assert float(jnp.linalg.norm(tangent)) > .1
    np.testing.assert_allclose(project(matrix), frame.anchor @ frame.anchor.T, atol=1e-13)


def test_partial_target_mp2_partition_and_response():
    from pyscfad.lno.ccsd import mp2_fragment_energy
    x, foo, fvv, _, _ = inputs_and_reference()
    u = jnp.array([[.6, -.8]])
    rhs = jnp.einsum('Pia,Pjb->iajb', x['Bvo'], x['Bvo'])
    fock = jnp.block([[foo, x['fov']], [x['fov'].T, fvv]])
    eris = SimpleNamespace(fock=fock, ovov=rhs)
    gaps = x['eocc'][:, None] - x['evir'][None, :]
    t2 = rhs.transpose(0, 2, 1, 3)/(gaps[:, None, :, None]+gaps[None, :, None, :])
    actual, bar = jax.value_and_grad(lambda prj: noncanonical_mp2(eris, prj)[0])(u)
    expected, refbar = jax.value_and_grad(lambda prj: mp2_fragment_energy(eris, t2, prj))(u)
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    np.testing.assert_allclose(bar, refbar, atol=1e-13)


def test_complementary_fragment_targets_recover_whole_laplace():
    x, foo, fvv, frames, grid = inputs_and_reference()
    foo = foo.at[0, 1].set(.02).at[1, 0].set(.02)
    solver = make_solver({'mode': 'deterministic'}, grid, with_grad=True)
    def fragments(fo):
        return sum(evaluate(solver, {**x, 'U': u}, fo, fvv)
                   for u in (jnp.array([[.6], [-.8]]), jnp.array([[.8], [.6]])))
    actual, bar = jax.value_and_grad(fragments)(foo)
    expected, refbar = jax.value_and_grad(lambda fo: evaluate(solver, x, fo, fvv))(foo)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(bar, refbar, atol=1e-11)


def test_retained_frame_checkpoint_roundtrip_and_old_selection_rejection(tmp_path):
    from pyscfad.dlno_stc import ccsd
    from pyscfad.dlno_stc._restart import RestartManager, scientific_digest
    frame = AnchoredFrame(np.eye(3))
    manager = RestartManager(tmp_path, resume=False, method='full-lis-frame', scientific_payload={})
    try:
        manager.save_static(frame)
        restored = manager.load_static(expected_type=AnchoredFrame)
        assert scientific_digest(restored) == scientific_digest(frame)
    finally:
        manager.close()
    previous = ccsd.CCSDSelections(None, (), schema_version=5)
    with pytest.raises(ValueError, match='incompatible CC selections'):
        ccsd._validate_run(previous, {}, {}, None)
