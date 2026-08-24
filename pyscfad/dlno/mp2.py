from functools import partial
import jax.numpy as jnp
import numpy as np
from pyscf import lib
from pyscfad.lib import logger
from pyscfad.mp import dfmp2
from . import multipole as multipole_ops
from .multipole import *
from .util import einsum

WITH_T2 = True


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
        overlap_transition_expr = 'ua,uv,vi->ia'
        dipole_transition_expr = 'ua,xuv,vi->ixa'
        quadrupole_transition_expr = 'ua,xyuv,vi->ixya'
        octupole_transition_expr = 'ua,xyzuv,vi->ixyza'
    else:
        virtual = jnp.stack([jnp.asarray(space) for space in mo_vir])
        overlap_transition_expr = 'iua,uv,vi->ia'
        dipole_transition_expr = 'iua,xuv,vi->ixa'
        quadrupole_transition_expr = 'iua,xyuv,vi->ixya'
        octupole_transition_expr = 'iua,xyzuv,vi->ixyza'

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

    fake_mol = multipole_ops._fake_mol_with_traced_centers(mol, atmlst)
    overlap, dipole, second, third = multipole_ops._origin_zero_moments(
        fake_mol, order=max(1, order - 1)
    )
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
    translated_second = None
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


def _multipole_pair_energy(left, right, order):
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
        aibj_3 = jnp.outer(tmp1_ai, tmp_bj * 5)
        aibj_3 -= jnp.outer(tmp_ai, tmp1_bj * 5)

        mu_R_ai = einsum('xa,y->xya', mu_ai, R_bar).reshape(9, -1)
        mu_R_bj = einsum('xb,y->xyb', mu_bj, R_bar).reshape(9, -1)
        aibj_3 += (2 * mu_R_ai.T) @ theta_bj.reshape(9, -1)
        aibj_3 -= theta_ai.reshape(9, -1).T @ (mu_R_bj * 2)
        aibj_3 /= R**4
        aibj += aibj_3

    if order > 3:
        RR = jnp.outer(R_bar, R_bar)
        RRR = einsum('x,y,z->xyz', R_bar, R_bar, R_bar)

        aibj_4 = einsum('xa,xyzb,yz->ab', mu_ai, omega_bj, RR * 9)
        aibj_4 += einsum('xy,xyza,zb->ab', RR * 9, omega_ai, mu_bj)

        omega_R3_ai = RRR.ravel() @ omega_ai.reshape(27, -1)
        omega_R3_bj = RRR.ravel() @ omega_bj.reshape(27, -1)
        aibj_4 -= jnp.outer(tmp_ai, omega_R3_bj * 21)
        aibj_4 -= jnp.outer(omega_R3_ai, tmp_bj * 21)
        aibj_4 += jnp.outer(tmp1_ai, tmp1_bj * 35)

        tmp2_ai = einsum('xya,y->xa', theta_ai, R_bar)
        tmp2_bj = einsum('xyb,y->xb', theta_bj, R_bar)
        aibj_4 -= tmp2_ai.T @ (tmp2_bj * 20)
        aibj_4 += theta_ai.reshape(9, -1).T @ (
            theta_bj.reshape(9, -1) * 2
        )
        aibj_4 /= (3 * R**5)
        aibj += aibj_4

    aibj2 = aibj * aibj / (e_ai[:, None] + e_bj[None, :])
    return -8 * jnp.sum(aibj2)


def pair_energy_multipole(
        mol,
        e_occ,
        mo_occ,
        e_vir,
        mo_vir,
        atmlst=None,
        order=4,
    ):
    """Multipole approximation to the OS-MP2 pair energy.
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
    """Multipole OS-MP2 energies for every left-right orbital pair.

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


def kernel(mp, prj, mo_energy=None, mo_coeff=None, eris=None,
           with_t2=WITH_T2):
    if mo_energy is not None or mo_coeff is not None:
        assert (mp.frozen == 0 or mp.frozen is None)

    if eris is None:      eris = mp.ao2mo(mo_coeff)
    if mo_energy is None: mo_energy = eris.mo_energy
    if mo_coeff is None:  mo_coeff = eris.mo_coeff

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    naux = mp.with_df.get_naoaux()
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    # Try to compute Lov from density-fitting cderi using JAX-backed contractions
    try:
        Lov = _compute_lov_from_cderi_ao2mo(mp, mo_coeff)
    except Exception:
        # fallback to original loop if any issue arises
        Lov = np.empty((naux, nocc*nvir))
        p1 = 0
        for istep, qov in enumerate(mp.loop_ao2mo(mo_coeff, nocc)):
            logger.debug(mp, 'Load cderi step %d', istep)
            p0, p1 = p1, p1 + qov.shape[0]
            Lov[p0:p1] = qov

    ovov = (Lov.T @ Lov).reshape(nocc,nvir,nocc,nvir)
    oovv = ovov.transpose(0,2,1,3)
    denominator = (
        eia[:, None, :, None] + eia[None, :, None, :]
    )
    t2 = oovv / denominator

    ed_ij = einsum('pjab,qjab', t2, oovv) * 2
    ex_ij = -einsum('pjab,qjba', t2, oovv)

    if not with_t2:
        t2 = None

    m = prj.T.conj() @ prj
    ed = einsum('ij,ji', ed_ij, m).real
    ex = einsum('ij,ji', ex_ij, m).real

    emp2_ss = ed*0.5 + ex
    emp2_os = ed*0.5
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)
    return emp2, t2


def _compute_lov_from_cderi_ao2mo(mp, mo_coeff):
    """Compute ``Lov`` through the tested DF-MP2 transformation path."""
    nocc = mp.nocc
    nvir = mo_coeff.shape[1] - nocc
    lov = mp.loop_ao2mo(mo_coeff, nocc, with_t2=False)
    return lov.reshape((-1, nocc * nvir))


def kernel_dfmp2(fake_mf, prj, with_t2=False):
    """Evaluate the DLNO-projected DF-MP2 energy on ``fake_mf``."""
    _mp2 = dfmp2.MP2(fake_mf)
    return kernel(_mp2, prj, with_t2=with_t2)


from pyscfad.lno.mp2 import LNOMP2 as _LNOMP2  # noqa: E402


class DLNOMP2(_LNOMP2):
    """MP2 with domain-restricted LNO prescreening.

    A thin subclass of :class:`pyscfad.lno.mp2.LNOMP2` whose constructor
    accepts ``dlno_prescreen_data`` directly and enables
    ``use_dlno_prescreen`` by default.  See
    :class:`pyscfad.dlno.ccsd.DLNOCCSD` for the rationale.
    """

    def __init__(self, mf, thresh=1e-4, frozen=None,
                 dlno_prescreen_data=None, **kwargs):
        super().__init__(mf, thresh=thresh, frozen=frozen, **kwargs)
        self.use_dlno_prescreen = True
        self.dlno_prescreen_data = dlno_prescreen_data
