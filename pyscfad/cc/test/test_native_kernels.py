"""Exercise non-symmetric amplitudes, partial panels and memory fallbacks."""
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from pyscf import lib
from pyscf.cc import ccsd
from threadpoolctl import threadpool_limits
from pyscfad.cc import _native_kernels as kernels

cpp = pytest.importorskip('pyscfad.cc._ccsd_kernels')


@pytest.mark.parametrize('v,block', [(1, 1), (7, 1), (7, 3), (7, 7), (19, 8)])
def test_packed_ladder(v, block):
    rng = np.random.default_rng(81)
    npair = v*(v+1)//2
    factors = rng.normal(size=(npair, 11))
    packed = factors @ factors.T
    t = rng.normal(size=(2, 3, v, v))
    idx = lib.square_mat_in_trilu_indices(v)
    full = packed[idx[:, :, None, None], idx[None, None, :, :]]
    expected = np.einsum('ijcd,acbd->ijab', t, full)
    for threads in (1, 2):
        with lib.with_omp_threads(threads), threadpool_limits(1, user_api='blas'):
            actual = cpp.ladder(packed, t, 8*block**2*v**2)
        np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=1e-13)
    cc = SimpleNamespace(max_memory=4000, async_io=False, verbose=0, stdout=sys.stdout)
    reference = ccsd._contract_vvvv_t2(cc, None, packed, t)
    np.testing.assert_allclose(actual, reference, atol=2e-11, rtol=1e-13)


def test_ladder_fallbacks_and_validation(monkeypatch):
    p, t = np.ones((6, 6)), np.ones((2, 3, 3))
    cc = SimpleNamespace(max_memory=1000, cpp_kernels=True)
    monkeypatch.setattr(lib, 'current_memory', lambda: (1000., 0.))
    assert kernels.contract_vvvv(cc, p, t) is None

    cc.max_memory = 2000
    assert kernels.contract_vvvv(cc, p, t, out=np.empty_like(t)) is None
    cc.cpp_kernels = False
    assert kernels.contract_vvvv(cc, p, t) is None
    with pytest.raises(ValueError, match='workspace'):
        cpp.ladder(p, t, 1)
    with pytest.raises(ValueError, match='inconsistent'):
        cpp.ladder(np.ones((5, 5)), t, 1000)
    with pytest.raises(TypeError):
        cpp.ladder(p.astype(np.float32), t, 1000)
    assert cpp.ladder(p, np.empty((0, 3, 3)), 1000).shape == (0, 3, 3)
    cc.cpp_kernels = True
    monkeypatch.setattr(kernels, '_ccsd_kernels', None)
    assert kernels.contract_vvvv(cc, p, t) is None


def test_ladder_remaining_total_memory(monkeypatch):
    """Large total budgets reach C++; output/input copies remain accounted for."""
    p = np.ones((6, 6))
    t = np.ones((2, 3, 3))
    seen = []
    def capture(pairs, tensor, workspace):
        assert pairs.flags.c_contiguous and tensor.flags.c_contiguous
        seen.append(workspace)
        return np.zeros_like(tensor)
    monkeypatch.setattr(kernels, '_ccsd_kernels', SimpleNamespace(ladder=capture))
    monkeypatch.setattr(lib, 'current_memory', lambda: (1000., 0.))
    cc = SimpleNamespace(max_memory=900000, cpp_kernels=True)
    kernels.contract_vvvv(cc, p, t)
    reserve = t.nbytes + 9*np.dtype(np.uintp).itemsize
    assert seen.pop() == 899000*1_000_000-reserve
    kernels.contract_vvvv(cc, p.T, t.swapaxes(1, 2))
    assert seen.pop() == 899000*1_000_000-reserve-p.nbytes-t.nbytes
    cc.max_memory = 1000 + (reserve+8*9-1)/1e6
    assert kernels.contract_vvvv(cc, p, t) is None
    assert not seen


def test_panel_scaling_and_budget():
    """Large LISs escape the old 64 MiB cap without filling a 900 GB budget."""
    roomy = 900000*1_000_000
    small = cpp.panel_shape(90, roomy, 32)
    large = cpp.panel_shape(400, roomy, 32)
    assert small == (32, 3, 12, 74_649_600)
    assert large == (32, 3, 3, 368_640_000)
    assert cpp.panel_shape(400, 1 << 30, 32) == large
    for available in (8*400**2, 64 << 20, 256 << 20):
        workers, a, c, allocated = cpp.panel_shape(400, available, 32)
        assert allocated <= available
        assert 1 <= workers <= 32 and a >= 1 and c >= 1
