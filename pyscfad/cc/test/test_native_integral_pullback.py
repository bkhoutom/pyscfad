"""DF pullbacks against independent contractions and finite differences."""
import numpy as np
import pytest
from threadpoolctl import threadpool_limits, threadpool_info
from pyscfad.cc import _native_integral_pullback as native
from pyscfad.cc._native_ccsd_eris import _G_Lbar_numpy

cpp = pytest.importorskip('pyscfad.cc._ccsd_kernels')


def unpack(packed, v):
    rows, cols = np.tril_indices(v)
    full = np.empty((packed.shape[0], v, v))
    full[:, rows, cols] = packed
    full[:, cols, rows] = packed
    return full


def ovvv_objective(args):
    lov, lvv, t1, theta, tau, tbar, fbar, wbar, tmpbar, woobar = args
    o, v = t1.shape
    packed = np.einsum('Qip,Qz->piz', lov, lvv)
    full = np.einsum('Qip,Qab->piab', lov, unpack(lvv, v))
    t = np.einsum('icjb,cjba->ia', theta, full)
    f = 2*np.einsum('kc,ckab->ab', t1, full) - np.einsum('kc,pkca->ap', t1, full)
    w = np.einsum('piac,jc->pija', full, t1)
    tmp = np.einsum('IJpd,pibd->IJbi', tau, full)
    woo = -np.einsum('kp,piz->kiz', t1, packed).reshape(o, -1)
    return sum(np.vdot(a, b) for a, b in zip((t, f, w, tmp, woo), (tbar, fbar, wbar, tmpbar, woobar)))


def ovvv_case(o, v, q):
    rng = np.random.default_rng(41)
    shapes = [(q,o,v), (q,v*(v+1)//2), (o,v), (o,v,o,v), (o,o,v,v),
              (o,v), (v,v), (v,o,o,v), (o,o,v,o), (o,o*v*(v+1)//2)]
    return [rng.normal(size=shape) for shape in shapes]


@pytest.mark.parametrize('shape', [(1,1,2), (2,4,5), (3,7,2), (3,29,13)])
@pytest.mark.parametrize('threads', [1, 2])
def test_ovvv_factor_directional_gradients(shape, threads):
    arrays = ovvv_case(*shape)
    saved = [a.copy() for a in arrays]
    # Include non-C inputs; the adapter owns aligned, contiguous conversion.
    arrays = [np.asfortranarray(a) for a in arrays]
    with threadpool_limits(threads):
        bars = native.ovvv(*arrays)
    rng = np.random.default_rng(75)
    for index in (0,1):
        direction = rng.normal(size=arrays[index].shape)
        plus, minus = arrays.copy(), arrays.copy()
        # The objective is linear in each DF factor separately, so a larger
        # step avoids cancellation without introducing truncation error.
        h = 2e-3
        plus[index] = arrays[index] + h*direction
        minus[index] = arrays[index] - h*direction
        finite = (ovvv_objective(plus)-ovvv_objective(minus))/(2*h)
        np.testing.assert_allclose(np.vdot(bars[index], direction), finite, rtol=2e-8, atol=2e-7)
    for a,b in zip(arrays,saved):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('shape', [(1,1,2), (3,5,4), (2,9,3), (3,37,7)])
@pytest.mark.parametrize('threads', [1, 2])
def test_vvvv_factor_gradient(shape, threads):
    r,v,q=shape
    rng=np.random.default_rng(31)
    tau=rng.normal(size=(r,v,v));factors=rng.normal(size=(q,v,v));bar=rng.normal(size=tau.shape)
    with threadpool_limits(threads):
        actual=native.vvvv(tau,factors,bar)
    np.testing.assert_allclose(actual,_G_Lbar_numpy(tau,factors,bar),rtol=2e-13,atol=2e-12)
    direction=rng.normal(size=factors.shape)
    def objective(l):return np.vdot(np.einsum('pac,Qab,Qcd->pdb',tau,l,l),bar)
    h=2e-6
    finite=(objective(factors+h*direction)-objective(factors-h*direction))/(2*h)
    np.testing.assert_allclose(np.vdot(actual,direction),finite,rtol=2e-8,atol=2e-7)


def test_cpp_blas_scope_restores_on_success_and_error(monkeypatch):
    real=cpp.dfccsd_ovvv_vjp
    seen=[]
    def checked(*arrays):
        assert all(p['num_threads']==1 for p in threadpool_info() if p['user_api']=='blas')
        seen.append(True)
        return real(*arrays)
    monkeypatch.setattr(cpp,'dfccsd_ovvv_vjp',checked)
    with threadpool_limits(2):
        before=threadpool_info()
        native.ovvv(*ovvv_case(2,3,4))
        assert threadpool_info()==before
        bad=ovvv_case(2,3,4);bad[3]=np.zeros((2,3,2,2))
        with pytest.raises(ValueError):native.ovvv(*bad)
        assert threadpool_info()==before
    assert len(seen)==2


def test_validation_and_empty_inputs():
    arrays=ovvv_case(2,3,4)
    arrays[0]=arrays[0].astype(np.float32)
    with pytest.raises(NotImplementedError):native.ovvv(*arrays)
    with pytest.raises(ValueError):native.vvvv(np.ones((2,3,4)),np.ones((5,3,3)),np.ones((2,3,4)))
    np.testing.assert_array_equal(native.vvvv(np.empty((0,3,3)),np.ones((2,3,3)),np.empty((0,3,3))),0.)
    a,b=native.ovvv(*ovvv_case(0,3,2))
    assert a.shape==(2,0,3)
    np.testing.assert_array_equal(b,0.)


def test_older_extension_fallback_signal(monkeypatch):
    monkeypatch.setattr(native,'cpp',None)
    with pytest.raises(NotImplementedError):native.ovvv(*ovvv_case(2,3,4))
    from pyscfad.cc._native_ccsd_eris import _G_Lbar
    rng=np.random.default_rng(17)
    a=rng.normal(size=(2,3,3));b=rng.normal(size=(4,3,3));c=rng.normal(size=a.shape)
    np.testing.assert_array_equal(_G_Lbar(a,b,c),_G_Lbar_numpy(a,b,c))
