"""Exercise the singles contraction with unequal axes and production strides."""
import numpy as np
import pytest
from pyscf import lib
from pyscf.cc import ccsd
from pyscfad.cc import _native_ccsd_forward as forward


@pytest.mark.parametrize('shape', [(3, 4, 2, 5, 6), (1, 3, 1, 4, 2), (2, 0, 3, 4, 5)])
@pytest.mark.parametrize('layout', ['production', 'strided', 'fortran'])
def test_ovvv_singles_layouts(shape, layout):
    i, c, j, b, a = shape
    rng = np.random.default_rng(167)
    theta = rng.normal(size=(j, i, c, b)).transpose(1, 2, 0, 3)
    vovv = rng.normal(size=(c, j, b, a))
    if layout == 'strided':
        theta = theta[..., ::-1]
        vovv = vovv[:, :, ::-1, :]
    elif layout == 'fortran':
        theta, vovv = np.asfortranarray(theta), np.asfortranarray(vovv)
    originals = theta.copy(), vovv.copy()
    expected = np.einsum('icjb,cjba->ia', theta, vovv)
    actual = forward.ovvv_singles(theta, vovv)
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
    np.testing.assert_array_equal(theta, originals[0])
    np.testing.assert_array_equal(vovv, originals[1])


def test_no_integral_copy_and_no_global_patch(monkeypatch):
    rng = np.random.default_rng(54)
    theta = rng.normal(size=(2, 3, 4, 5)).transpose(1, 2, 0, 3)
    vovv = rng.normal(size=(4, 2, 5, 6))
    dot = lib.dot
    def check(left, right):
        assert np.shares_memory(right, vovv)
        return dot(left, right)
    monkeypatch.setattr(lib, 'dot', check)
    result = forward._ContractionLib().einsum('icjb,cjba->ia', theta, vovv)
    np.testing.assert_allclose(result, np.einsum('icjb,cjba->ia', theta, vovv), atol=2e-13)
    assert ccsd.update_amps.__globals__['_add_ovvv_'] is ccsd._add_ovvv_
    assert ccsd._add_ovvv_.__globals__['lib'] is lib
    assert forward._reference_update_amps.__globals__['_add_ovvv_'] is forward._add_ovvv


def test_dispatch_fallbacks(monkeypatch):
    calls = []
    def fallback(*args, **kwargs):
        calls.append((args, kwargs))
        return 'fallback'
    monkeypatch.setattr(lib, 'einsum', fallback)
    proxy = forward._ContractionLib()
    a = np.ones((2, 2, 2, 2))
    assert proxy.einsum('icjb,cjba->ia', a, a, optimize=True) == 'fallback'
    assert proxy.einsum('icjb,cjba->ia', a.astype(complex), a) == 'fallback'
    assert proxy.einsum('ij,jk->ik', a[0, 0], a[0, 0]) == 'fallback'
    assert len(calls) == 3
    with pytest.raises(ValueError, match='dimensions'):
        forward.ovvv_singles(a, np.ones((3, 2, 2, 2)))
