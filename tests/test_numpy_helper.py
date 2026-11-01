"""Triangular helper values and AD must agree for generic batch shapes."""

import jax
import jax.numpy as jnp
import numpy
import pytest

from pyscfad import lib


@pytest.mark.parametrize('axis', [0, -1])
def test_pack_unpack_batched_values_and_derivatives(axis):
    """Catch lost batch axes and non-transposable scatters in cached ladder AD."""
    a = jnp.arange(54., dtype=float).reshape(2, 3, 3, 3)
    if axis == 0:
        a = jnp.moveaxis(a, (-2, -1), (0, 1))
    i, j = numpy.tril_indices(3)

    def pack_ref(x):
        return x[i, j] if axis == 0 else x[..., i, j]

    def roundtrip(x):
        return lib.unpack_tril(lib.pack_tril(x, axis=axis),
                               filltriu=lib.SYMMETRIC, axis=axis)

    def roundtrip_ref(x):
        lookup = numpy.empty((3, 3), dtype=int)
        lookup[i, j] = lookup[j, i] = numpy.arange(6)
        p = pack_ref(x)
        return p[lookup] if axis == 0 else p[..., lookup]

    numpy.testing.assert_array_equal(lib.pack_tril(a, axis=axis), pack_ref(a))
    value, pull = jax.vjp(jax.jit(roundtrip), a)
    value_ref, pull_ref = jax.vjp(roundtrip_ref, a)
    numpy.testing.assert_array_equal(value, value_ref)
    cot = jnp.arange(a.size, dtype=float).reshape(a.shape) / 7
    numpy.testing.assert_array_equal(pull(cot)[0], pull_ref(cot)[0])
    _, dot = jax.jvp(jax.jit(roundtrip), (a,), (cot,))
    numpy.testing.assert_array_equal(dot, roundtrip_ref(cot))


@pytest.mark.parametrize('fill,want', [
    (lib.PLAIN, [[1, 0, 0], [2, 3, 0], [4, 5, 6]]),
    (lib.SYMMETRIC, [[1, 2, 4], [2, 3, 5], [4, 5, 6]]),
    (lib.HERMITIAN, [[1, 2, 4], [2, 3, 5], [4, 5, 6]]),
    (lib.ANTIHERMI, [[1, -2, -4], [2, 3, -5], [4, 5, 6]]),
])
def test_unpack_fill_modes_and_transpose(fill, want):
    """Catch upper-triangle sign and diagonal mistakes in primal and AD rules."""
    packed = jnp.arange(1., 7.)
    numpy.testing.assert_array_equal(lib.unpack_tril(packed, filltriu=fill), want)
    cot = jnp.arange(9.).reshape(3, 3)
    # Independent linear reference from the six packed basis matrices.
    bases = []
    for k in range(6):
        basis = numpy.zeros((3, 3))
        i, j = tuple(zip(*numpy.tril_indices(3)))[k]
        basis[i, j] = 1
        if i != j and fill != lib.PLAIN:
            basis[j, i] = -1 if fill == lib.ANTIHERMI else 1
        bases.append(basis)
    actual = jax.grad(lambda p: jnp.vdot(lib.unpack_tril(p, filltriu=fill), cot))(packed)
    numpy.testing.assert_array_equal(actual, numpy.einsum('kij,ij->k', bases, cot))


@pytest.mark.parametrize('fill,want', [
    (lib.PLAIN, [[1+2j, 0, 0], [2+3j, 3+4j, 0], [4+5j, 5+6j, 6+7j]]),
    (lib.SYMMETRIC, [[1+2j, 2+3j, 4+5j],
                     [2+3j, 3+4j, 5+6j], [4+5j, 5+6j, 6+7j]]),
    (lib.HERMITIAN, [[1+2j, 2-3j, 4-5j],
                     [2+3j, 3+4j, 5-6j], [4+5j, 5+6j, 6+7j]]),
    (lib.ANTIHERMI, [[1+2j, -2+3j, -4+5j],
                     [2+3j, 3+4j, -5+6j], [4+5j, 5+6j, 6+7j]]),
])
def test_complex_unpack_fill_modes_and_derivatives(fill, want):
    """Packed complex diagonals survive, and only upper entries are conjugated."""
    packed = jnp.arange(1., 7.) + 1j * jnp.arange(2., 8.)

    def reference(p):
        if fill == lib.PLAIN:
            u01 = u02 = u12 = jnp.zeros_like(p[0])
        else:
            u01, u02, u12 = p[1], p[3], p[4]
            if fill in (lib.HERMITIAN, lib.ANTIHERMI):
                u01, u02, u12 = u01.conj(), u02.conj(), u12.conj()
            if fill == lib.ANTIHERMI:
                u01, u02, u12 = -u01, -u02, -u12
        return jnp.stack((jnp.stack((p[0], u01, u02)),
                          jnp.stack((p[1], p[2], u12)),
                          jnp.stack((p[3], p[4], p[5]))))

    fn = jax.jit(lambda p: lib.unpack_tril(p, filltriu=fill))
    numpy.testing.assert_array_equal(fn(packed), want)
    tangent = packed * (.25 + .125j)
    _, actual_dot = jax.jvp(fn, (packed,), (tangent,))
    _, expected_dot = jax.jvp(reference, (packed,), (tangent,))
    numpy.testing.assert_array_equal(actual_dot, expected_dot)
    cot = jnp.arange(9.).reshape(3, 3) + 1j * jnp.arange(9., 18.).reshape(3, 3)
    _, pull = jax.vjp(fn, packed)
    _, pull_ref = jax.vjp(reference, packed)
    numpy.testing.assert_array_equal(pull(cot)[0], pull_ref(cot)[0])
