"""Direct C ABI checks against independent NumPy values and finite differences."""
import ctypes
import numpy
import pytest

import pyscfadlib


def _reference(x, energies):
    v = numpy.einsum("iaL,jbL->ijab", x, x)
    d = (energies[:2, None, None, None] + energies[None, :2, None, None]
         - energies[None, None, 2:, None] - energies[None, None, None, 2:])
    t = v / d
    return numpy.array([numpy.sum(v*t), -numpy.sum(v.swapaxes(2, 3)*t)]), t


def _ptr(value):
    return ctypes.c_void_p() if value is None else value.ctypes.data_as(ctypes.c_void_p)


@pytest.mark.parametrize("weights", [(2., 1.), (-0.3, 0.8), (0., 0.)])
@pytest.mark.parametrize("with_t2", [False, True])
def test_native_mp2_adds_arbitrary_seed_vjp_and_writes_t2(weights, with_t2):
    native = getattr(pyscfadlib, "libmp_vjp", None)
    assert native is not None, "optional MP2 library not built/loaded"
    rng = numpy.random.default_rng(180)
    x = rng.normal(size=(2, 3, 4))
    energies = numpy.array([-1.3, -0.7, 0.2, 0.6, 1.0])
    x_initial = rng.normal(size=x.shape)
    e_initial = rng.normal(size=energies.shape)
    xbar, ebar = x_initial.copy(), e_initial.copy()
    energy = numpy.array([0.2, -0.4])
    t2 = numpy.empty((2, 2, 3, 3)) if with_t2 else None
    tbar = rng.normal(size=(2, 2, 3, 3)) if with_t2 else None
    native.MP2_contract_d_energy_vjp(
        _ptr(energy), _ptr(t2), _ptr(tbar), _ptr(xbar), _ptr(ebar), _ptr(ebar[2:]),
        ctypes.c_double(weights[0]), ctypes.c_double(weights[1]),
        _ptr(x), _ptr(energies), _ptr(energies[2:]),
        ctypes.c_int(2), ctypes.c_int(3), ctypes.c_int(4))
    expected_e, expected_t = _reference(x, energies)
    numpy.testing.assert_allclose(energy, expected_e + [0.2, -0.4], atol=1e-10)
    if with_t2:
        numpy.testing.assert_allclose(t2, expected_t, atol=1e-12)
    def objective(xx, ee):
        e, t = _reference(xx, ee)
        value = numpy.dot(e, weights)
        return value if tbar is None else value + numpy.sum(t*tbar)
    for _ in range(2):
        dx, de = rng.normal(size=x.shape), rng.normal(size=energies.shape)
        step = 1e-5
        finite_difference = (objective(x+step*dx, energies+step*de)
                             - objective(x-step*dx, energies-step*de)) / (2*step)
        actual = numpy.sum((xbar-x_initial)*dx) + numpy.dot(ebar-e_initial, de)
        numpy.testing.assert_allclose(actual, finite_difference, atol=1e-7, rtol=1e-8)
    # The legacy derivative symbol must keep the same additive ABI.
    legacy_x, legacy_e = x_initial.copy(), e_initial.copy()
    native.MP2_contract_d_vjp(
        _ptr(legacy_x), _ptr(legacy_e), _ptr(legacy_e[2:]),
        ctypes.c_double(weights[0]), ctypes.c_double(weights[1]),
        _ptr(x), _ptr(energies), _ptr(energies[2:]),
        ctypes.c_int(2), ctypes.c_int(3), ctypes.c_int(4))
    if not with_t2:
        numpy.testing.assert_allclose(legacy_x, xbar, atol=1e-12)
        numpy.testing.assert_allclose(legacy_e, ebar, atol=1e-12)
