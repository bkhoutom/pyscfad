"""Differentiable multipole operators and weak-pair MP2 screening.

AO moments retain their coordinate response through the registered custom JVP.
Endpoint construction batches occupied modes over shared AO moments before
forming pair energies. Discrete topology selection uses the separate NumPy
implementation in :mod:`pyscfad.dlno.multipole_numpy`.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from pyscfad.lib._multipole import (
    coordinate_moment_jvp,
    pair_energy as _multipole_pair_energy,
    transition_moments,
)
from .tools import fake_mol_by_atom

__all__ = [
    'dipole_op', 'quadrupole_op', 'octupole_op',
    'pair_energy_multipole', 'pair_energy_multipole_cross',
]


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

    aoslices = fake_mol.aoslice_by_atom()
    ao_owner = np.repeat(
        np.arange(fake_mol.natm, dtype=np.int32),
        aoslices[:, 3] - aoslices[:, 2],
    )
    tangent = coordinate_moment_jvp(bra, ket, coords_t, ao_owner)
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


def _atom_list_key(atoms):
    """Convert one discrete atom selection to a hashable group key."""
    if atoms is None:
        return None
    return tuple(int(atom) for atom in np.asarray(atoms).ravel())


def _multipole_batch_shapes_compatible(mo_occ, e_vir, mo_vir):
    """Whether one endpoint can use rectangular batched contractions."""
    if len(mo_occ) == 0:
        return False
    occ_shapes = [tuple(jnp.shape(orbital)) for orbital in mo_occ]
    vir_shapes = [tuple(jnp.shape(orbital)) for orbital in mo_vir]
    energy_shapes = [tuple(jnp.shape(energy)) for energy in e_vir]
    if any(len(shape) != 1 for shape in occ_shapes):
        return False
    if any(len(shape) != 2 for shape in vir_shapes):
        return False
    if any(shape != occ_shapes[0] for shape in occ_shapes[1:]):
        return False
    if any(shape != vir_shapes[0] for shape in vir_shapes[1:]):
        return False
    if any(shape != energy_shapes[0] for shape in energy_shapes[1:]):
        return False
    nao = occ_shapes[0][0]
    nvir = vir_shapes[0][1]
    return (
        vir_shapes[0][0] == nao
        and len(energy_shapes[0]) == 1
        and energy_shapes[0][0] == nvir
    )


def _multipole_orbital_data_batch(
        mol, e_occ, mo_occ, e_vir, mo_vir, atmlst, order):
    """Build all one-endpoint transition multipoles from one AO integral set.

    Raw moments are evaluated once about the global origin.  Their AO-to-MO
    transition contractions are batched over occupied modes, after which the
    second and third moments are translated to each mode's charge center and
    made traceless.  Keeping translation at the MO-transition level avoids
    retaining one large AO moment tape per occupied mode in a gradient.
    """
    occupied = jnp.stack(
        [jnp.asarray(orbital).ravel() for orbital in mo_occ], axis=1
    )
    occupied_energy = jnp.stack(
        [jnp.asarray(energy).reshape(()) for energy in e_occ]
    )

    # The IAO weak-screen caller repeats the exact same PAO array for every
    # mode.  Avoid stacking copies in that common case, while retaining a
    # general rectangular batch for API callers with mode-dependent spaces.
    shared_virtual = all(
        orbital is mo_vir[0] for orbital in mo_vir[1:]
    )
    if shared_virtual:
        virtual = jnp.asarray(mo_vir[0])
    else:
        virtual = jnp.stack([jnp.asarray(space) for space in mo_vir])

    shared_virtual_energy = all(
        energy is e_vir[0] for energy in e_vir[1:]
    )
    if shared_virtual_energy:
        excitation_energy = (
            jnp.asarray(e_vir[0])[None, :] - occupied_energy[:, None]
        )
    else:
        excitation_energy = (
            jnp.stack([jnp.asarray(energy) for energy in e_vir])
            - occupied_energy[:, None]
        )

    fake_mol = _fake_mol_with_traced_centers(mol, atmlst)
    overlap, dipole, second, third = _origin_zero_moments(
        fake_mol, order=max(1, order - 1)
    )
    center, transition_dipole, transition_quadrupole, transition_octupole = (
        transition_moments(
            occupied, virtual, overlap, dipole, second, third, order
        )
    )

    return [
        (
            center[index],
            transition_dipole[index],
            excitation_energy[index],
            None if transition_quadrupole is None
            else transition_quadrupole[index],
            None if transition_octupole is None
            else transition_octupole[index],
        )
        for index in range(len(e_occ))
    ]


def _multipole_endpoint_data(
        mol, e_occ, mo_occ, e_vir, mo_vir, atmlst, order):
    """Build endpoint records exclusively through batched AO moments.

    Modes sharing an atom list and rectangular occupied/virtual shapes are
    evaluated together.  Heterogeneous public inputs are partitioned into
    compatible groups; a genuinely unique layout becomes a one-mode batch,
    rather than entering a second scalar implementation.
    """
    nmode = len(e_occ)
    lengths = {
        "mo_occ": len(mo_occ),
        "e_vir": len(e_vir),
        "mo_vir": len(mo_vir),
        "atmlst": len(atmlst),
    }
    mismatched = {
        name: length for name, length in lengths.items() if length != nmode
    }
    if mismatched:
        detail = ", ".join(
            f"{name}={length}" for name, length in mismatched.items()
        )
        raise ValueError(
            f"multipole endpoint inputs must all have length {nmode}; {detail}"
        )
    if order not in (2, 3, 4):
        raise ValueError("multipole order must be 2, 3, or 4")

    groups = {}
    for index in range(nmode):
        key = (
            _atom_list_key(atmlst[index]),
            tuple(jnp.shape(mo_occ[index])),
            tuple(jnp.shape(e_vir[index])),
            tuple(jnp.shape(mo_vir[index])),
        )
        groups.setdefault(key, []).append(index)

    records = [None] * nmode
    for key, indices in groups.items():
        atoms = key[0]
        group_e_occ = tuple(e_occ[index] for index in indices)
        group_mo_occ = tuple(mo_occ[index] for index in indices)
        group_e_vir = tuple(e_vir[index] for index in indices)
        group_mo_vir = tuple(mo_vir[index] for index in indices)
        if not _multipole_batch_shapes_compatible(
                group_mo_occ, group_e_vir, group_mo_vir):
            raise ValueError(
                "multipole occupied orbitals must be rank 1 and virtual "
                "spaces rank 2 with matching AO and virtual dimensions"
            )
        group_records = _multipole_orbital_data_batch(
            mol,
            group_e_occ,
            group_mo_occ,
            group_e_vir,
            group_mo_vir,
            atoms,
            order,
        )
        for index, record in zip(indices, group_records):
            records[index] = record
    return records


def _multipole_cross_from_data(left_data, right_data, order):
    """Contract two precomputed endpoint records into a rectangular block."""
    pair_energy = jnp.zeros(
        (len(left_data), len(right_data)), dtype=jnp.float64
    )
    for left_index, left in enumerate(left_data):
        for right_index, right in enumerate(right_data):
            pair_energy = pair_energy.at[left_index, right_index].set(
                _multipole_pair_energy(left, right, order)
            )
    return pair_energy


def pair_energy_multipole(
        mol,
        e_occ,
        mo_occ,
        e_vir,
        mo_vir,
        atmlst=None,
        order=4,
    ):
    """Multipole approximation to the MP2 pair energy.
    """
    nocc = len(e_occ)
    if atmlst is None:
        atmlst = [None,] * nocc
    orbital_data = _multipole_endpoint_data(
        mol, e_occ, mo_occ, e_vir, mo_vir, atmlst, order
    )
    e_mp2_pair = jnp.zeros((nocc, nocc), dtype=jnp.float64)
    for i in range(nocc):
        for j in range(i):
            e_mp2_pair = e_mp2_pair.at[i, j].set(
                _multipole_pair_energy(orbital_data[i], orbital_data[j], order)
            )

    e_mp2_pair = e_mp2_pair + e_mp2_pair.T
    return e_mp2_pair


def pair_energy_multipole_cross(
        mol,
        e_occ_left,
        mo_occ_left,
        e_vir_left,
        mo_vir_left,
        e_occ_right,
        mo_occ_right,
        e_vir_right,
        mo_vir_right,
        atmlst_left=None,
        atmlst_right=None,
        order=4,
    ):
    """Multipole MP2 energies for every left-right orbital pair.

    Unlike :func:`pair_energy_multipole`, this routine never forms pairs
    within either input set.  Callers can therefore pass two distinct
    fragment spaces without encountering the coincident-centroid pairs
    that occur between orbitals belonging to the same fragment.
    """
    nleft = len(e_occ_left)
    nright = len(e_occ_right)
    if atmlst_left is None:
        atmlst_left = [None,] * nleft
    if atmlst_right is None:
        atmlst_right = [None,] * nright

    left_data = _multipole_endpoint_data(
        mol,
        e_occ_left,
        mo_occ_left,
        e_vir_left,
        mo_vir_left,
        atmlst_left,
        order,
    )
    right_data = _multipole_endpoint_data(
        mol,
        e_occ_right,
        mo_occ_right,
        e_vir_right,
        mo_vir_right,
        atmlst_right,
        order,
    )
    return _multipole_cross_from_data(left_data, right_data, order)
