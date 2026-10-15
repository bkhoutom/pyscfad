"""Numerical kernels and endpoint checks shared by DLNO multipole consumers.

Molecule construction and AO integral evaluation stay with the callers; these
kernels contain only differentiable contractions and moment translations.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def coordinate_moment_jvp(bra, ket, coords_t, ao_owner):
    """Contract canonical bra/ket derivatives with their atom tangents.

    ``bra`` and ``ket`` both have shape ``(3, ncart, nao, nao)`` and
    ``ao_owner`` maps each AO index to an atom in ``coords_t``.
    """
    ao_tangent = coords_t[ao_owner]
    return (
        jnp.einsum('ux,xpuv->puv', ao_tangent, bra)
        + jnp.einsum('vx,xpuv->puv', ao_tangent, ket)
    )


@partial(jax.jit, static_argnums=(6,))
def transition_moments(occupied, virtual, overlap, dipole, second, third, order):
    """Contract AO moments for a rectangular batch of occupied modes.

    ``occupied`` has shape ``(nao, nmode)``. ``virtual`` is either one shared
    ``(nao, nvir)`` space or a ``(nmode, nao, nvir)`` batch. The returned center,
    dipole, quadrupole, and octupole arrays have mode as their leading axis;
    moments above the requested order are ``None``.
    """
    if virtual.ndim == 2:
        overlap_transition_expr = 'ua,uv,vi->ia'
        dipole_transition_expr = 'ua,xuv,vi->ixa'
        quadrupole_transition_expr = 'ua,xyuv,vi->ixya'
        octupole_transition_expr = 'ua,xyzuv,vi->ixyza'
    else:
        overlap_transition_expr = 'iua,uv,vi->ia'
        dipole_transition_expr = 'iua,xuv,vi->ixa'
        quadrupole_transition_expr = 'iua,xyuv,vi->ixya'
        octupole_transition_expr = 'iua,xyzuv,vi->ixyza'
    center = jnp.einsum(
        'ui,xuv,vi->ix', occupied.conj(), dipole, occupied
    )
    transition_overlap = jnp.einsum(
        overlap_transition_expr, virtual.conj(), overlap, occupied
    )
    transition_dipole = jnp.einsum(
        dipole_transition_expr, virtual.conj(), dipole, occupied
    )

    transition_quadrupole = None
    transition_octupole = None
    if order > 2:
        raw_second = jnp.einsum(
            quadrupole_transition_expr,
            virtual.conj(),
            second,
            occupied,
        )
        translated_second = raw_second
        translated_second = translated_second - jnp.einsum(
            'ix,iya->ixya', center, transition_dipole
        )
        translated_second = translated_second - jnp.einsum(
            'iy,ixa->ixya', center, transition_dipole
        )
        translated_second = translated_second + jnp.einsum(
            'ix,iy,ia->ixya', center, center, transition_overlap
        )
        second_trace = jnp.trace(
            translated_second, axis1=1, axis2=2
        )
        identity = jnp.eye(3, dtype=translated_second.dtype)
        transition_quadrupole = 0.5 * (
            3.0 * translated_second
            - jnp.einsum('xy,ia->ixya', identity, second_trace)
        )

    if order > 3:
        raw_third = jnp.einsum(
            octupole_transition_expr,
            virtual.conj(),
            third,
            occupied,
        )
        translated_third = raw_third
        translated_third = translated_third - jnp.einsum(
            'ix,iyza->ixyza', center, raw_second
        )
        translated_third = translated_third - jnp.einsum(
            'iy,ixza->ixyza', center, raw_second
        )
        translated_third = translated_third - jnp.einsum(
            'iz,ixya->ixyza', center, raw_second
        )
        translated_third = translated_third + jnp.einsum(
            'ix,iy,iza->ixyza', center, center, transition_dipole
        )
        translated_third = translated_third + jnp.einsum(
            'ix,iz,iya->ixyza', center, center, transition_dipole
        )
        translated_third = translated_third + jnp.einsum(
            'iy,iz,ixa->ixyza', center, center, transition_dipole
        )
        translated_third = translated_third - jnp.einsum(
            'ix,iy,iz,ia->ixyza',
            center,
            center,
            center,
            transition_overlap,
        )

        trace_yz = jnp.trace(translated_third, axis1=2, axis2=3)
        trace_xz = jnp.trace(translated_third, axis1=1, axis2=3)
        trace_xy = jnp.trace(translated_third, axis1=1, axis2=2)
        identity = jnp.eye(3, dtype=translated_third.dtype)
        transition_octupole = 0.5 * (
            5.0 * translated_third
            - jnp.einsum('yz,ixa->ixyza', identity, trace_yz)
            - jnp.einsum('xz,iya->ixyza', identity, trace_xz)
            - jnp.einsum('xy,iza->ixyza', identity, trace_xy)
        )

    return (
        center, transition_dipole, transition_quadrupole, transition_octupole
    )


def endpoint_matches_reference(endpoint, reference, *, rtol=1e-9, atol=1e-9):
    """Check AO-moment replay against its original forward endpoint values.

    Compare each moment family's Cartesian RMS per virtual column. This
    scale is independent of the Cartesian axes and remains meaningful for
    components near zero by cancellation. No column or moment order borrows
    another's scale. The absolute tolerance is an RMS bound, preserving all
    noise patterns accepted by the former componentwise absolute bound.
    Centers and excitation energies retain componentwise checks.
    """
    if jax.tree_util.tree_structure(endpoint) != jax.tree_util.tree_structure(reference):
        return False
    for slot, (live, saved) in enumerate(zip(endpoint, reference)):
        if live is None:
            continue
        live, saved = np.asarray(live), np.asarray(saved)
        if (live.shape != saved.shape or not np.isfinite(live).all()
                or not np.isfinite(saved).all()):
            return False
        if slot in (1, 3, 4):
            axes = tuple(range(live.ndim - 1))
            with np.errstate(over='ignore', invalid='ignore'):
                difference = np.sqrt(np.mean(np.abs(live - saved)**2, axis=axes))
                scale = np.maximum(
                    np.sqrt(np.mean(np.abs(live)**2, axis=axes)),
                    np.sqrt(np.mean(np.abs(saved)**2, axis=axes)))
                bound = atol + rtol*scale
            if (not np.isfinite(difference).all() or not np.isfinite(bound).all()
                    or not np.all(difference <= bound)):
                return False
        elif not np.allclose(live, saved, rtol=rtol, atol=atol):
            return False
    return True


@partial(jax.jit, static_argnums=(2,))
def pair_energy(left, right, order):
    """Contract precomputed multipoles for one ordered orbital pair."""
    Ri, mu_ai, e_ai, theta_ai, omega_ai = left
    Rj, mu_bj, e_bj, theta_bj, omega_bj = right

    R = jnp.linalg.norm(Rj - Ri)
    R_bar = (Rj - Ri) / R

    aibj_2 = mu_ai.T @ mu_bj
    tmp_ai = R_bar @ mu_ai
    tmp_bj = R_bar @ mu_bj
    aibj_2 -= jnp.outer(tmp_ai, tmp_bj * 3)
    aibj_2 /= R**3

    aibj = aibj_2
    if order > 2:
        RR = jnp.outer(R_bar, R_bar)
        tmp1_ai = RR.ravel() @ theta_ai.reshape(9, -1)
        tmp1_bj = RR.ravel() @ theta_bj.reshape(9, -1)
        aibj_3 = jnp.outer(tmp_ai, tmp1_bj * 5)
        aibj_3 -= jnp.outer(tmp1_ai, tmp_bj * 5)

        mu_R_ai = jnp.einsum('xa,y->xya', mu_ai, R_bar).reshape(9, -1)
        mu_R_bj = jnp.einsum('xb,y->xyb', mu_bj, R_bar).reshape(9, -1)
        aibj_3 -= (2 * mu_R_ai.T) @ theta_bj.reshape(9, -1)
        aibj_3 += theta_ai.reshape(9, -1).T @ (mu_R_bj * 2)
        aibj_3 /= R**4
        aibj += aibj_3

    if order > 3:
        RR = jnp.outer(R_bar, R_bar)
        RRR = jnp.einsum('x,y,z->xyz', R_bar, R_bar, R_bar)

        aibj_4 = jnp.einsum('xa,xyzb,yz->ab', mu_ai, omega_bj, RR * 9)
        aibj_4 += jnp.einsum('xy,xyza,zb->ab', RR * 9, omega_ai, mu_bj)

        omega_R3_ai = RRR.ravel() @ omega_ai.reshape(27, -1)
        omega_R3_bj = RRR.ravel() @ omega_bj.reshape(27, -1)
        aibj_4 -= jnp.outer(tmp_ai, omega_R3_bj * 21)
        aibj_4 -= jnp.outer(omega_R3_ai, tmp_bj * 21)
        aibj_4 += jnp.outer(tmp1_ai, tmp1_bj * 35)

        tmp2_ai = jnp.einsum('xya,y->xa', theta_ai, R_bar)
        tmp2_bj = jnp.einsum('xyb,y->xb', theta_bj, R_bar)
        aibj_4 -= tmp2_ai.T @ (tmp2_bj * 20)
        aibj_4 += theta_ai.reshape(9, -1).T @ (
            theta_bj.reshape(9, -1) * 2
        )
        aibj_4 /= (3 * R**5)
        aibj += aibj_4

    aibj2 = aibj * aibj / (e_ai[:, None] + e_bj[None, :])
    return -4 * jnp.sum(aibj2)


# Compile the endpoint VJP together with its primal so pair loops dispatch one
# executable instead of a separate eager operation for each tensor contraction.
def _real_pair_energy(left, right, order):
    return jnp.real(pair_energy(left, right, order))


pair_energy_and_grad = jax.jit(
    jax.value_and_grad(_real_pair_energy, argnums=(0, 1)), static_argnums=(2,)
)
