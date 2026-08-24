from functools import partial

import jax
import jax.numpy as jnp
from .util import fake_mol_by_atom

__all__ = ['dipole_op', 'quadrupole_op', 'octupole_op']


def _fake_mol_with_traced_centers(mol, atmlst):
    """Build an atom submolecule while retaining its coordinate response."""
    fake_mol = fake_mol_by_atom(mol, atmlst)
    coords = getattr(mol, 'coords', None)
    if atmlst is not None and coords is not None:
        # ``fake_mol_by_atom`` rebuilds the concrete integral environment but
        # its shallow copy otherwise retains the full parent coordinate leaf.
        # Slice that leaf so JAX scatters a local-moment cotangent back to the
        # corresponding parent atoms rather than to atoms 0..len(atmlst)-1.
        fake_mol.coords = coords[jnp.asarray(atmlst, dtype=int)]
    return fake_mol


def _contains_tracer(value):
    """Return whether ``value`` contains a JAX tracer."""
    return any(
        isinstance(leaf, jax.core.Tracer)
        for leaf in jax.tree_util.tree_leaves(value)
    )


def _raw_cartesian_moment_impl(fake_mol, rank):
    """Evaluate an AO Cartesian moment about the fixed global origin."""
    nao = fake_mol.nao
    intor = {2: 'int1e_rr', 3: 'int1e_rrr'}[rank]
    with fake_mol.with_common_origin((0.0, 0.0, 0.0)):
        moment = jnp.asarray(fake_mol.intor(intor))
    return moment.reshape((3,) * rank + (nao, nao))


def _raw_cartesian_moment_coordinate_jvp(fake_mol, coords_t, rank):
    """Coordinate JVP with the Cartesian operator axes kept in order.

    For ``int1e_rr[_r]_dr01`` libcint places the ket derivative axis after
    the two or three Cartesian operator axes.  The generic two-centre
    integral JVP treats it as the leading axis, which is only correct for a
    scalar operator.  Keep the reordering local to the multipole operators
    until the general integral rule supports arbitrary Cartesian rank.
    """
    nao = fake_mol.nao
    intor = {2: 'int1e_rr', 3: 'int1e_rrr'}[rank]
    ncart = 3 ** rank
    with fake_mol.with_common_origin((0.0, 0.0, 0.0)):
        bra = -jnp.asarray(fake_mol.intor(f'{intor}_dr10'))
        ket = -jnp.asarray(fake_mol.intor(f'{intor}_dr01'))
    bra = bra.reshape(3, ncart, nao, nao)
    ket = ket.reshape(ncart, 3, nao, nao).transpose(1, 0, 2, 3)

    tangent = jnp.zeros((ncart, nao, nao), dtype=bra.dtype)
    for atom, (_, _, p0, p1) in enumerate(fake_mol.aoslice_by_atom()):
        bra_atom = jnp.einsum(
            'x,xpuv->puv', coords_t[atom], bra[:, :, p0:p1, :]
        )
        ket_atom = jnp.einsum(
            'x,xpuv->puv', coords_t[atom], ket[:, :, :, p0:p1]
        )
        tangent = tangent.at[:, p0:p1, :].add(bra_atom)
        tangent = tangent.at[:, :, p0:p1].add(ket_atom)
    return tangent.reshape((3,) * rank + (nao, nao))


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _raw_cartesian_moment(fake_mol, rank):
    """Cartesian moment with a rank-aware nuclear-coordinate JVP."""
    return _raw_cartesian_moment_impl(fake_mol, rank)


@_raw_cartesian_moment.defjvp
def _raw_cartesian_moment_jvp(rank, primals, tangents):
    fake_mol, = primals
    fake_mol_t, = tangents
    primal = _raw_cartesian_moment(fake_mol, rank)

    # Preserve the pre-existing exponent/contraction-coefficient response by
    # asking the general integral rule for a JVP with its coordinate tangent
    # zeroed.  The coordinate part is supplied below with the correct
    # Cartesian component ordering.
    other_tangent = jnp.zeros_like(primal)
    if any(
        getattr(fake_mol, name, None) is not None
        for name in ('exp', 'ctr_coeff', 'r0')
    ):
        basis_tangent = fake_mol_t.copy(deep=False)
        if basis_tangent.coords is not None:
            basis_tangent.coords = jnp.zeros_like(basis_tangent.coords)
        _, other_tangent = jax.jvp(
            partial(_raw_cartesian_moment_impl, rank=rank),
            (fake_mol,),
            (basis_tangent,),
        )

    coordinate_tangent = jnp.zeros_like(primal)
    if fake_mol.coords is not None:
        coordinate_tangent = _raw_cartesian_moment_coordinate_jvp(
            fake_mol, fake_mol_t.coords, rank
        )
    return primal, other_tangent + coordinate_tangent


def _origin_zero_moments(fake_mol, order):
    """Evaluate raw Cartesian moments about the fixed global origin.

    PySCF stores the common origin in a NumPy ``_env`` buffer, so its
    ``with_common_origin`` context cannot consume a traced, geometry-dependent
    origin.  A concrete zero origin is safe inside a JAX trace; the moments
    can then be translated with ordinary JAX algebra.
    """
    nao = fake_mol.nao
    with fake_mol.with_common_origin((0.0, 0.0, 0.0)):
        overlap = jnp.asarray(fake_mol.intor('int1e_ovlp')).reshape(nao, nao)
        r = jnp.asarray(fake_mol.intor('int1e_r')).reshape(3, nao, nao)
        rr = None
        rrr = None
        # A plain PySCF Mole is not a registered JAX pytree, so it cannot be
        # passed through the custom-JVP wrapper merely because some unrelated
        # input (for example an orbital energy) is traced.  Use the identical
        # primal integral call when this molecule has no differentiable basis
        # or coordinate leaves.
        moment = _raw_cartesian_moment
        if not any(
            getattr(fake_mol, name, None) is not None
            for name in ('coords', 'exp', 'ctr_coeff', 'r0')
        ):
            moment = _raw_cartesian_moment_impl
        if order >= 2:
            rr = moment(fake_mol, 2)
        if order >= 3:
            rrr = moment(fake_mol, 3)
    return overlap, r, rr, rrr


def _translated_second_moment(fake_mol, origin):
    r"""Return :math:`(r-R)_x(r-R)_y` AO integrals for traced ``R``."""
    overlap, r, rr, _ = _origin_zero_moments(fake_mol, order=2)
    origin = jnp.asarray(origin)
    rr = rr - jnp.einsum('x,yuv->xyuv', origin, r)
    rr = rr - jnp.einsum('y,xuv->xyuv', origin, r)
    rr = rr + jnp.einsum(
        'x,y,uv->xyuv', origin, origin, overlap
    )
    return rr


def _translated_third_moment(fake_mol, origin):
    r"""Return :math:`\prod_{a=x,y,z}(r_a-R_a)` AO integrals."""
    overlap, r, rr, rrr = _origin_zero_moments(fake_mol, order=3)
    origin = jnp.asarray(origin)

    rrr = rrr - jnp.einsum('x,yzuv->xyzuv', origin, rr)
    rrr = rrr - jnp.einsum('y,xzuv->xyzuv', origin, rr)
    rrr = rrr - jnp.einsum('z,xyuv->xyzuv', origin, rr)
    rrr = rrr + jnp.einsum(
        'x,y,zuv->xyzuv', origin, origin, r
    )
    rrr = rrr + jnp.einsum(
        'x,z,yuv->xyzuv', origin, origin, r
    )
    rrr = rrr + jnp.einsum(
        'y,z,xuv->xyzuv', origin, origin, r
    )
    rrr = rrr - jnp.einsum(
        'x,y,z,uv->xyzuv', origin, origin, origin, overlap
    )
    return rrr

def dipole_op(mol, R=jnp.zeros((3,)), atmlst=None):
    fake_mol = _fake_mol_with_traced_centers(mol, atmlst)
    nao = fake_mol.nao
    with fake_mol.with_common_origin(R):
        r = jnp.asarray(fake_mol.intor('int1e_r')).reshape(3,nao,nao)
    return r

def quadrupole_op(mol, R=jnp.zeros((3,)), atmlst=None):
    fake_mol = _fake_mol_with_traced_centers(mol, atmlst)
    nao = fake_mol.nao
    if _contains_tracer((R, getattr(fake_mol, 'coords', None))):
        rr = _translated_second_moment(fake_mol, R)
    else:
        with fake_mol.with_common_origin(R):
            rr = jnp.asarray(fake_mol.intor('int1e_rr')).reshape(3,3,nao,nao)
    r2 = jnp.trace(rr)

    rr = rr * 3
    for x in range(3):
        rr = rr.at[x,x].add(-r2)
    rr = rr * 0.5
    return rr

def octupole_op(mol, R=jnp.zeros((3,)), atmlst=None):
    fake_mol = _fake_mol_with_traced_centers(mol, atmlst)
    nao = fake_mol.nao
    if _contains_tracer((R, getattr(fake_mol, 'coords', None))):
        rrr = _translated_third_moment(fake_mol, R)
    else:
        with fake_mol.with_common_origin(R):
            rrr = jnp.asarray(fake_mol.intor('int1e_rrr')).reshape(3,3,3,nao,nao)

    r2r_0 = jnp.trace(rrr, axis1=1, axis2=2)
    r2r_1 = jnp.trace(rrr, axis1=2, axis2=0)
    r2r_2 = jnp.trace(rrr, axis1=0, axis2=1)

    rrr = rrr * 5
    for x in range(3):
        rrr = rrr.at[:,x,x].add(-r2r_0)
        rrr = rrr.at[x,:,x].add(-r2r_1)
        rrr = rrr.at[x,x,:].add(-r2r_2)
    rrr = rrr * 0.5
    return rrr
