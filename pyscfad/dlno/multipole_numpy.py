"""Static NumPy implementation of the DLNO multipole pair model.

The routines in :mod:`pyscfad.dlno.mp2` intentionally use JAX because their
results may participate in a differentiated energy expression.  Domain
topology construction, on the other hand, is a discrete preprocessing step.
Sending every trial pair through JAX there grows the device allocation and
dispatch caches without providing a useful derivative.  This module mirrors
the same dipole--octupole formulas using only NumPy and PySCF integral calls.
"""

import numpy as np

from .util import fake_mol_by_atom


__all__ = [
    "multipole_orbital_data",
    "multipole_pair_energy",
    "multipole_pair_energy_cross",
    "multipole_pair_energy_matrix",
]


def _fake_mol(mol, atmlst):
    return fake_mol_by_atom(mol, atmlst)


def _dipole_op(mol, atmlst=None):
    fake_mol = _fake_mol(mol, atmlst)
    nao = fake_mol.nao
    with fake_mol.with_common_origin(np.zeros(3)):
        return np.asarray(fake_mol.intor("int1e_r")).reshape(3, nao, nao)


def _quadrupole_op(mol, center, atmlst=None):
    fake_mol = _fake_mol(mol, atmlst)
    nao = fake_mol.nao
    with fake_mol.with_common_origin(np.asarray(center)):
        rr = np.asarray(fake_mol.intor("int1e_rr")).reshape(
            3, 3, nao, nao
        )

    r2 = np.trace(rr, axis1=0, axis2=1)
    rr = rr * 3.0
    for axis in range(3):
        rr[axis, axis] -= r2
    return rr * 0.5


def _octupole_op(mol, center, atmlst=None):
    fake_mol = _fake_mol(mol, atmlst)
    nao = fake_mol.nao
    with fake_mol.with_common_origin(np.asarray(center)):
        rrr = np.asarray(fake_mol.intor("int1e_rrr")).reshape(
            3, 3, 3, nao, nao
        )

    r2r_0 = np.trace(rrr, axis1=1, axis2=2)
    r2r_1 = np.trace(rrr, axis1=2, axis2=0)
    r2r_2 = np.trace(rrr, axis1=0, axis2=1)
    rrr = rrr * 5.0
    for axis in range(3):
        rrr[:, axis, axis] -= r2r_0
        rrr[axis, :, axis] -= r2r_1
        rrr[axis, axis, :] -= r2r_2
    return rrr * 0.5


def _expectation(op, orbital):
    orbital = np.asarray(orbital).reshape(-1)
    op = np.asarray(op)
    op_orbital = op.reshape(-1, orbital.size, orbital.size) @ orbital
    return op_orbital @ orbital.conj()


def _transition(op, occupied, virtual):
    occupied = np.asarray(occupied).reshape(-1)
    virtual = np.asarray(virtual)
    op = np.asarray(op)
    op_occupied = op.reshape(-1, occupied.size, occupied.size) @ occupied
    return (op_occupied @ virtual.conj()).reshape(
        *op.shape[:-2], virtual.shape[1]
    )


def multipole_orbital_data(
    mol,
    e_occ,
    mo_occ,
    e_vir,
    mo_vir,
    atmlst=None,
    order=4,
):
    """Build static one-orbital data for Nagy's OS pair-increment model."""
    if order not in (2, 3, 4):
        raise ValueError("multipole order must be 2, 3, or 4")

    occupied = np.asarray(mo_occ).reshape(-1)
    virtual = np.asarray(mo_vir)
    occupied_energy = float(np.asarray(e_occ).reshape(()))
    virtual_energy = np.asarray(e_vir)

    dipole = _dipole_op(mol, atmlst=atmlst)
    center = _expectation(dipole, occupied)
    transition_dipole = _transition(dipole, occupied, virtual)

    transition_quadrupole = None
    transition_octupole = None
    if order > 2:
        quadrupole = _quadrupole_op(mol, center, atmlst=atmlst)
        transition_quadrupole = _transition(
            quadrupole, occupied, virtual
        )
    if order > 3:
        octupole = _octupole_op(mol, center, atmlst=atmlst)
        transition_octupole = _transition(octupole, occupied, virtual)

    return (
        center,
        transition_dipole,
        virtual_energy - occupied_energy,
        transition_quadrupole,
        transition_octupole,
    )


def multipole_pair_energy(left, right, order=4):
    """Contract records using Nagy's ``-8`` OS pair-increment convention."""
    if order not in (2, 3, 4):
        raise ValueError("multipole order must be 2, 3, or 4")

    Ri, mu_ai, e_ai, theta_ai, omega_ai = left
    Rj, mu_bj, e_bj, theta_bj, omega_bj = right
    Ri = np.asarray(Ri)
    Rj = np.asarray(Rj)
    mu_ai = np.asarray(mu_ai)
    mu_bj = np.asarray(mu_bj)
    e_ai = np.asarray(e_ai)
    e_bj = np.asarray(e_bj)

    displacement = Rj - Ri
    distance = np.linalg.norm(displacement)
    direction = displacement / distance

    aibj_2 = mu_ai.T @ mu_bj
    tmp_ai = direction @ mu_ai
    tmp_bj = direction @ mu_bj
    aibj_2 -= np.outer(tmp_ai, tmp_bj * 3.0)
    aibj_2 /= distance**3

    aibj = aibj_2
    if order > 2:
        theta_ai = np.asarray(theta_ai)
        theta_bj = np.asarray(theta_bj)
        theta_ai_flat = theta_ai.reshape(9, -1)
        theta_bj_flat = theta_bj.reshape(9, -1)
        RR = np.outer(direction, direction)

        tmp1_ai = RR.ravel() @ theta_ai_flat
        tmp1_bj = RR.ravel() @ theta_bj_flat
        aibj_3 = np.outer(tmp1_ai, tmp_bj * 5.0)
        aibj_3 -= np.outer(tmp_ai, tmp1_bj * 5.0)

        mu_R_ai = (
            mu_ai[:, None, :] * direction[None, :, None]
        ).reshape(9, -1)
        mu_R_bj = (
            mu_bj[:, None, :] * direction[None, :, None]
        ).reshape(9, -1)
        aibj_3 += (2.0 * mu_R_ai.T) @ theta_bj_flat
        aibj_3 -= theta_ai_flat.T @ (mu_R_bj * 2.0)
        aibj_3 /= distance**4
        aibj = aibj + aibj_3

    if order > 3:
        omega_ai = np.asarray(omega_ai)
        omega_bj = np.asarray(omega_bj)
        omega_ai_flat = omega_ai.reshape(27, -1)
        omega_bj_flat = omega_bj.reshape(27, -1)
        RR = np.outer(direction, direction)
        RRR = (
            direction[:, None, None]
            * direction[None, :, None]
            * direction[None, None, :]
        )

        RR9 = RR * 9.0
        omega_RR_bj = np.tensordot(
            omega_bj, RR9, axes=([1, 2], [0, 1])
        )
        omega_RR_ai = np.tensordot(
            RR9, omega_ai, axes=([0, 1], [0, 1])
        )
        aibj_4 = mu_ai.T @ omega_RR_bj
        aibj_4 += omega_RR_ai.T @ mu_bj

        omega_R3_ai = RRR.ravel() @ omega_ai_flat
        omega_R3_bj = RRR.ravel() @ omega_bj_flat
        aibj_4 -= np.outer(tmp_ai, omega_R3_bj * 21.0)
        aibj_4 -= np.outer(omega_R3_ai, tmp_bj * 21.0)
        aibj_4 += np.outer(tmp1_ai, tmp1_bj * 35.0)

        tmp2_ai = np.tensordot(theta_ai, direction, axes=([1], [0]))
        tmp2_bj = np.tensordot(theta_bj, direction, axes=([1], [0]))
        aibj_4 -= tmp2_ai.T @ (tmp2_bj * 20.0)
        aibj_4 += theta_ai_flat.T @ (theta_bj_flat * 2.0)
        aibj_4 /= 3.0 * distance**5
        aibj = aibj + aibj_4

    denominator = e_ai[:, None] + e_bj[None, :]
    return float(np.real(-8.0 * np.sum(aibj * aibj / denominator)))


def multipole_pair_energy_cross(left_data, right_data, order=4):
    """Return all left--right energies without forming within-set pairs."""
    pair_energy = np.empty((len(left_data), len(right_data)), dtype=float)
    for left_index, left in enumerate(left_data):
        for right_index, right in enumerate(right_data):
            pair_energy[left_index, right_index] = multipole_pair_energy(
                left, right, order=order
            )
    return pair_energy


def multipole_pair_energy_matrix(orbital_data, order=4):
    """Return the symmetric pair-energy matrix for one orbital list."""
    norb = len(orbital_data)
    pair_energy = np.zeros((norb, norb), dtype=float)
    for left in range(norb):
        for right in range(left):
            value = multipole_pair_energy(
                orbital_data[left], orbital_data[right], order=order
            )
            pair_energy[left, right] = pair_energy[right, left] = value
    return pair_energy
