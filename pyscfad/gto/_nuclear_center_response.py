"""Stream the linear nuclear-center integral response at a fixed geometry.

The first-order reverse tape retains only molecular metadata. Each nucleus's
AO derivative is generated and contracted on demand, rather than saving a
dense derivative for every nucleus during forward linearization. Higher-order
geometry/basis differentiation uses the differentiable integral rules instead.
"""

from functools import partial

import jax
import numpy
from jax._src.core import Primitive, ShapedArray
from jax.interpreters import ad, batching, mlir
from pyscf.gto import ATOM_OF

from ._moleintor_helper import _intor_impl, int1e_get_dr_order
from ._pyscf_moleintor import make_loc, _get_intor_and_comp


def nuclear_center_tangent(mol, tangent, intor_a, intor_b, hermi, shls_slice):
    """Return a streamed JVP, or None when the geometry itself is traced."""
    if any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(mol)):
        return None
    if shls_slice is None:
        shls_slice = (0, mol.nbas, 0, mol.nbas)
    shls_slice = tuple(shls_slice)
    ao_loc = make_loc(mol._bas, intor_a)
    i0, i1, j0, j1 = shls_slice[:4]
    if hermi == 1:
        assert (i0 == j0 and i1 == j1)
    _, comp = _get_intor_and_comp(intor_a)
    shape = (comp // 3, int(ao_loc[i1] - ao_loc[i0]),
             int(ao_loc[j1] - ao_loc[j0]))
    # Detach the small integral environment, never an SCF/DF object or a tape.
    reference = mol.to_pyscf().copy()
    return _center_response_p.bind(
        tangent, mol=reference, intor_a=intor_a, intor_b=intor_b,
        hermi=hermi, shls_slice=shls_slice, shape=shape, transpose=False)


def _contract(value, *, mol, intor_a, intor_b, hermi, shls_slice, shape, transpose):
    # with_rinv_at_nucleus mutates the integral origin temporarily. Each call
    # owns its environment, including under batched or compiled execution.
    mol = mol.copy()
    value = numpy.asarray(value)
    result = numpy.zeros((mol.natm, 3) if transpose else shape, dtype=value.dtype)
    ecp = 'ECP' in intor_a
    atoms = (sorted(set(mol._ecpbas[:, ATOM_OF])) if ecp else range(mol.natm))
    for atom in atoms:
        charge = 1 if ecp else -mol.atom_charge(atom)
        if charge == 0:
            continue
        with mol.with_rinv_at_nucleus(atom):
            derivative = _intor_impl(mol, intor_a, shls_slice=shls_slice).reshape(3, *shape)
            if hermi == 0:
                order = int1e_get_dr_order(intor_b)[0]
                ket = _intor_impl(mol, intor_b, shls_slice=shls_slice)
                ket = ket.reshape(3**order, 3, -1, *shape[-2:]).transpose(1, 0, 2, 3, 4)
                derivative += ket.reshape(3, *shape)
                del ket
        derivative *= charge
        if transpose:
            result[atom] = numpy.einsum('xyij,yij->x', derivative, value)
            if hermi == 1:
                result[atom] += numpy.einsum('xyij,yji->x', derivative, value)
        else:
            result += numpy.einsum('xyij,x->yij', derivative, value[atom])
        # Do not overlap the previous nucleus's buffer with the next allocation.
        del derivative
    if not transpose and hermi == 1:
        result += result.transpose(0, 2, 1)
    return result


def _abstract(value, *, mol, shape, transpose, **unused):
    return ShapedArray((mol.natm, 3) if transpose else shape, value.dtype)


def _transpose(seed, value, **options):
    return (_center_response_p.bind(seed, **{**options, 'transpose': not options['transpose']}),)


def _batch(args, dims, **options):
    value, = args
    dim, = dims
    if dim is None:
        return _center_response_p.bind(value, **options), None
    value = jax.numpy.moveaxis(value, dim, 0)
    return jax.lax.map(lambda row: _center_response_p.bind(row, **options), value), 0


def _lower(ctx, value, **options):
    aval, = ctx.avals_out
    def call(value):
        return jax.pure_callback(partial(_contract, **options),
                                 jax.ShapeDtypeStruct(aval.shape, aval.dtype), value)
    return mlir.lower_fun(call, multiple_results=False)(ctx, value)


_center_response_p = Primitive('nuclear_center_response')
_center_response_p.def_impl(_contract)
_center_response_p.def_abstract_eval(_abstract)
ad.deflinear2(_center_response_p, _transpose)
batching.primitive_batchers[_center_response_p] = _batch
mlir.register_lowering(_center_response_p, _lower)
