"""The physical density must use the scoped NumPy BLAS backend."""

import numpy
import pytest

from pyscfad.df import _cderi_vjp


@pytest.mark.parametrize("kc,lc", [(2, 4), (4, 2)])
@pytest.mark.parametrize("dtype", [numpy.float64, numpy.complex128])
def test_physical_density_uses_numpy_blas(kc, lc, dtype, monkeypatch):
    rng = numpy.random.default_rng(79)

    def values(shape):
        out = rng.normal(size=shape)
        if numpy.issubdtype(dtype, numpy.complexfloating):
            out = out + 1j * rng.normal(size=shape)
        return numpy.asarray(out, dtype=dtype)

    mo_k = values((5, kc))
    mo_l = values((5, lc))
    z = values((3, kc, lc))[:, ::-1, :]
    # Independent three-factor contraction also catches conjugation errors.
    expected = numpy.einsum("uk,pkl,vl->puv", mo_k, z, mo_l)
    expected = expected + expected.swapaxes(1, 2)

    def forbidden_sequential_backend(*args, **kwargs):
        raise AssertionError("physical density used PySCF's sequential BLAS")

    monkeypatch.setattr(
        _cderi_vjp.pyscf_lib, "einsum", forbidden_sequential_backend
    )
    actual = _cderi_vjp._int3c_mo_physical_density(mo_k, mo_l, z)
    numpy.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
