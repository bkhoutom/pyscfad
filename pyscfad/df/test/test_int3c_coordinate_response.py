"""Coordinate response must avoid atom-batched three-center integral arrays."""

import jax
import jax.numpy as jnp
import numpy
import pytest

from pyscf.df.incore import aux_e2
from pyscfad import gto
from pyscfad.df import incore


def _molecules(cart):
    atom = 'O 0.1 0.0 0.0; H 0.0 -0.8 0.6; H 0.0 0.7 0.5'
    mol = gto.Mole(atom=atom, unit='Bohr', cart=cart, verbose=0,
                   basis={'O': [[0, [1., 1.]], [1, [.8, 1.]], [2, [.6, 1.]]],
                          'H': [[0, [.7, 1.]]]})
    aux = gto.Mole(atom=atom, unit='Bohr', cart=cart, verbose=0,
                   basis={'O': [[0, [1.3, 1.]], [1, [.9, 1.]]],
                          'H': [[0, [.4, 1.]], [1, [.3, 1.]]]})
    for obj in (mol, aux):
        obj.build(trace_exp=False, trace_ctr_coeff=False)
    return mol, aux


@pytest.mark.parametrize('cart', [False, True])
@pytest.mark.parametrize('intor,comp', [('int3c2e', 1),
                                       ('int3c2e_ip1', 3), ('int3c2e_ip2', 3)])
def test_coordinate_response_matches_finite_difference_and_adjoint(cart, intor, comp):
    mol, aux = _molecules(cart)
    rng = numpy.random.default_rng(910)
    coordinates = (mol.coords, aux.coords)
    directions = tuple(jnp.asarray(rng.normal(size=x.shape)) for x in coordinates)

    def integrals(coords, auxcoords):
        current, auxiliary = mol.copy(), aux.copy()
        current.coords, auxiliary.coords = coords, auxcoords
        return incore.int3c_cross(current, auxiliary, intor=intor, comp=comp)

    value, tangent = jax.jvp(integrals, coordinates, directions)
    h = 1e-4
    displaced = []
    for sign in (1., -1.):
        current = mol.to_pyscf().set_geom_(numpy.asarray(coordinates[0] + sign*h*directions[0]),
                                          unit='Bohr', inplace=False)
        auxiliary = aux.to_pyscf().set_geom_(numpy.asarray(coordinates[1] + sign*h*directions[1]),
                                            unit='Bohr', inplace=False)
        displaced.append(aux_e2(current, auxiliary, intor=intor, comp=comp, aosym='s1'))
    numpy.testing.assert_allclose(tangent, (displaced[0]-displaced[1])/(2*h),
                                  atol=3e-7, rtol=3e-6)

    seed = jnp.asarray(rng.normal(size=value.shape))
    _, pullback = jax.vjp(integrals, *coordinates)
    bars = pullback(seed)
    numpy.testing.assert_allclose(sum(jnp.vdot(b, d) for b, d in zip(bars, directions)),
                                  jnp.vdot(seed, tangent), atol=2e-10, rtol=2e-10)
    translation = tuple(jnp.broadcast_to(jnp.array([.3, -.2, .1]), x.shape)
                        for x in coordinates)
    _, translated = jax.jvp(integrals, coordinates, translation)
    numpy.testing.assert_allclose(translated, 0., atol=2e-11)


def test_coordinate_hessian_vector_matches_gradient_difference():
    mol, aux = _molecules(False)
    rng = numpy.random.default_rng(911)
    seed = jnp.asarray(rng.normal(size=(mol.nao, mol.nao, aux.nao)))

    def scalar_at(reference, auxiliary_reference):
        def scalar(coords):
            current, auxiliary = reference.copy(), auxiliary_reference.copy()
            current.coords = auxiliary.coords = coords
            return jnp.vdot(seed, incore.int3c_cross(current, auxiliary))
        return scalar

    direction = jnp.asarray(rng.normal(size=mol.coords.shape))
    gradient = jax.grad(scalar_at(mol, aux))
    _, hessian_vector = jax.jvp(gradient, (mol.coords,), (direction,))
    h = 1e-4
    gradients = []
    for sign in (1., -1.):
        coords = numpy.asarray(mol.coords + sign*h*direction)
        # Native integral values read the molecular environment; rebuild it
        # at each finite-difference geometry as well as setting the AD leaf.
        current = mol.set_geom_(coords, unit='Bohr', inplace=False)
        auxiliary = aux.set_geom_(coords, unit='Bohr', inplace=False)
        gradients.append(jax.grad(scalar_at(current, auxiliary))(current.coords))
    finite_difference = (gradients[0] - gradients[1])/(2*h)
    numpy.testing.assert_allclose(hessian_vector, finite_difference, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize('auxiliary', [False, True])
def test_coordinate_linearization_has_no_atom_batched_integral_residual(auxiliary):
    mol, aux = _molecules(False)
    obj = aux if auxiliary else mol
    fill = incore._int3c_fill_jvp_r0_ip2 if auxiliary else incore._int3c_fill_jvp_r0_ip1
    shape = (3, mol.nao, mol.nao, aux.nao)

    def retained(ints):
        def contract(coords):
            tangent = obj.copy()
            tangent.coords = coords
            return fill(obj, tangent, ints)
        return jax.linearize(contract, jnp.zeros_like(obj.coords))[1]

    residuals = jax.tree_util.tree_leaves(jax.eval_shape(retained,
                                                       jax.ShapeDtypeStruct(shape, jnp.float64)))
    largest = max(numpy.prod(x.shape) for x in residuals)
    assert largest <= numpy.prod(shape)
