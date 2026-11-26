"""Armadillo oracle, independent finite differences, and backend isolation."""
import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits
from pyscfad.dlno_stc import _native_w as new, backend

old = pytest.importorskip('pyscfad.dlno_stc._stc_mp2')
pytestmark = pytest.mark.skipif(not new.available(), reason='rebuild CCSD kernels')


def fixture(degenerate=False):
    rng = np.random.default_rng(431)
    o, v, q = 3, 7, 8
    u = np.linalg.qr(rng.normal(size=(o, o)))[0]
    z = np.linalg.qr(rng.normal(size=(v, v)))[0]
    eo = np.array([-.9, -.9 if degenerate else -.7, -.3])
    ev = np.array([.1, .1 if degenerate else .2, .4, .4+1e-13, .8, 1., 1.2])
    arrays = [np.ascontiguousarray(a) for a in
              (u@np.diag(eo)@u.T, z@np.diag(ev)@z.T, rng.normal(size=(q,o,v))*.1)]
    controls = dict(mode='deterministic', threads=2, laplace_roots=[0., .2, .6],
                    laplace_weights=[.03, .4, .8])
    seeds = (.7, rng.normal(size=(o,o)), rng.normal(size=(v,v)))
    return arrays, controls, seeds


@pytest.mark.parametrize('threads', [1, 3])
@pytest.mark.parametrize('degenerate', [False, True])
def test_forward_reverse_and_directional_gradients(threads, degenerate):
    arrays, controls, seeds = fixture(degenerate)
    controls['threads'] = threads
    saved = [x.copy() for x in arrays]
    with threadpool_limits(1, user_api='blas'):
        reference = old.solve_matrices(*arrays, (0,2), controls)
        expected = old.pullback_matrices(*arrays, reference['replay'], *seeds)
    actual = new.solve_matrices(*arrays, (0,2), controls)
    bars = new.pullback_matrices(*arrays, actual['replay'], *seeds)
    for key in ('energy', 'occupied', 'virtual'):
        np.testing.assert_allclose(actual[key], reference[key], rtol=2e-11, atol=2e-13)
    rng = np.random.default_rng(125)
    for i, key in enumerate(('foo', 'fvv', 'B')):
        np.testing.assert_allclose(bars[key], expected[key], rtol=3e-10, atol=3e-13)
        direction = rng.normal(size=arrays[i].shape)
        if i < 2: direction = (direction+direction.T)*.5
        # Five-point differences remove the large cubic-direction error of
        # the quartic DF-factor objective without weakening the tolerance.
        h = 1e-4
        def objective(sign):
            shifted = arrays.copy();shifted[i] = arrays[i] + sign*h*direction
            f = new.solve_matrices(*shifted, (0,2), controls)
            return seeds[0]*f['energy'] + np.vdot(seeds[1],f['occupied']) + np.vdot(seeds[2],f['virtual'])
        numeric = (objective(-2)-8*objective(-1)+8*objective(1)-objective(2))/(12*h)
        np.testing.assert_allclose(np.vdot(bars[key],direction), numeric, rtol=3e-7, atol=1e-9)
    for a,b in zip(arrays,saved):np.testing.assert_array_equal(a,b)


def test_per_target_seeds_and_sum():
    arrays, controls, seeds = fixture()
    targets=(0,2)
    forward = new.solve_orbital_matrices(*arrays, targets, controls)
    summed = new.solve_matrices(*arrays, targets, controls)
    for key in ('energy', 'occupied', 'virtual'):
        np.testing.assert_allclose(sum(x[key] for x in forward), summed[key], atol=3e-14)
    energies=[.7,-.3];occupied=[seeds[1],-.2*seeds[1]];virtuals=[seeds[2],.9*seeds[2]]
    actual = new.pullback_orbital_matrices(*arrays, forward[0]['replay'], energies, occupied, virtuals)
    with threadpool_limits(1, user_api='blas'):
        f = old.solve_orbital_matrices(*arrays, targets, controls)
        reference = old.pullback_orbital_matrices(*arrays, f[0]['replay'], energies, occupied, virtuals)
    for key in actual:np.testing.assert_allclose(actual[key], reference[key], rtol=2e-10, atol=3e-13)


def test_runtime_scope_and_no_legacy_calls(monkeypatch):
    arrays, controls, seeds = fixture()
    import_module = backend.importlib.import_module
    def checked_import(name, *args, **kwargs):
        assert name != 'pyscfad.dlno_stc._stc_mp2'
        return import_module(name, *args, **kwargs)
    monkeypatch.setattr(backend.importlib, 'import_module', checked_import)
    kernel = new.cpp.deterministic_w
    calls=[]
    def checked(*args):
        assert all(x['num_threads']==1 for x in threadpool_info() if x['user_api']=='blas')
        calls.append(True)
        return kernel(*args)
    monkeypatch.setattr(new.cpp, 'deterministic_w', checked)
    inputs=dict(zip(('foo','fvv','B'),arrays))
    with threadpool_limits(3):
        before=threadpool_info()
        f=backend.solve_matrices(inputs,dict(target_indices=(0,2)),controls)
        backend.pullback_matrices(inputs,f,energy_bar=seeds[0],occupied_bar=seeds[1],virtual_bar=seeds[2])
        assert before==threadpool_info()
        changed=dict(inputs,B=inputs['B']+.001)
        with pytest.raises(ValueError,match='fingerprint'):
            backend.pullback_matrices(changed,f,energy_bar=1.,occupied_bar=seeds[1],virtual_bar=seeds[2])
        assert before==threadpool_info()
    assert len(calls)==2


def test_parallel_matrix_views():
    # Large enough to enter the shared parallel GEMM path for frame rotations.
    rng=np.random.default_rng(740)
    no,nv,nq=4,65,27
    u=np.linalg.qr(rng.normal(size=(nv,nv)))[0]
    arrays=[np.diag(np.linspace(-1.,-.4,no)),
            np.ascontiguousarray(u@np.diag(np.linspace(.1,1.2,nv))@u.T),
            rng.normal(size=(nq,no,nv))*.03]
    controls=dict(mode='deterministic',threads=3,laplace_roots=[.1,.4],laplace_weights=[.3,.7])
    seeds=(.3,rng.normal(size=(no,no)),rng.normal(size=(nv,nv)))
    with threadpool_limits(1,user_api='blas'):
        ref=old.solve_matrices(*arrays,(1,3),controls)
        ref_bars=old.pullback_matrices(*arrays,ref['replay'],*seeds)
    actual=new.solve_matrices(*arrays,(1,3),controls)
    bars=new.pullback_matrices(*arrays,actual['replay'],*seeds)
    for key in ('energy','occupied','virtual'):
        np.testing.assert_allclose(actual[key],ref[key],atol=2e-13,rtol=2e-11)
    for key in bars:
        np.testing.assert_allclose(bars[key],ref_bars[key],atol=3e-13,rtol=3e-10)
