from functools import partial
import jax.numpy as jnp
import numpy as np
from pyscfad.lib import logger
from pyscfad.mp import dfmp2
from .multipole import *
from .util import einsum

WITH_T2 = True

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
    e_mp2_pair = jnp.zeros((nocc, nocc), dtype=jnp.float64)
    for i in range(nocc):
        lmo_i = jnp.asarray(mo_occ[i]).ravel()
        atmlst_i = atmlst[i]
        Di = dipole_op(mol, atmlst=atmlst_i)
        Ri = einsum('u,xuv,v->x', lmo_i.conj(), Di, lmo_i)

        pao_i = jnp.asarray(mo_vir[i])
        mu_ai = einsum('ua,xuv,v->xa', pao_i.conj(), Di, lmo_i)
        e_ai = jnp.asarray(e_vir[i]) - e_occ[i]

        theta_ai = None
        omega_ai = None
        if order > 2:
            Qi = quadrupole_op(mol, R=Ri, atmlst=atmlst_i)
            theta_ai = einsum('ua,xyuv,v->xya', pao_i.conj(), Qi, lmo_i)
        if order > 3:
            Oi = octupole_op(mol, R=Ri, atmlst=atmlst_i)
            omega_ai = einsum('ua,xyzuv,v->xyza', pao_i.conj(), Oi, lmo_i)

        for j in range(i):
            lmo_j = jnp.asarray(mo_occ[j]).ravel()
            atmlst_j = atmlst[j]
            Dj = dipole_op(mol, atmlst=atmlst_j)
            Rj = einsum('u,xuv,v->x', lmo_j.conj(), Dj, lmo_j)
            R = jnp.linalg.norm(Rj - Ri)
            R_bar = (Rj - Ri) / R

            pao_j = jnp.asarray(mo_vir[j])
            mu_bj = einsum('ua,xuv,v->xa', pao_j.conj(), Dj, lmo_j)

            aibj_2 = mu_ai.T @ mu_bj
            tmp_ai = R_bar @ mu_ai
            tmp_bj = R_bar @ mu_bj
            aibj_2 -= jnp.outer(tmp_ai, tmp_bj * 3)
            aibj_2 /= R**3

            aibj = aibj_2

            if order > 2:
                Qj = quadrupole_op(mol, R=Rj, atmlst=atmlst_j)
                theta_bj = einsum('ua,xyuv,v->xya', pao_j.conj(), Qj, lmo_j)
                RR = jnp.outer(R_bar, R_bar)

                tmp1_ai = RR.ravel() @ theta_ai.reshape(9, -1)
                tmp1_bj = RR.ravel() @ theta_bj.reshape(9, -1)
                aibj_3  = jnp.outer(tmp1_ai, tmp_bj * 5)
                aibj_3 -= jnp.outer(tmp_ai, tmp1_bj * 5)

                mu_R_ai = einsum('xa,y->xya', mu_ai, R_bar).reshape(9, -1)
                mu_R_bj = einsum('xb,y->xyb', mu_bj, R_bar).reshape(9, -1)
                aibj_3 += (2 * mu_R_ai.T) @ theta_bj.reshape(9, -1)
                aibj_3 -= theta_ai.reshape(9, -1).T @ (mu_R_bj * 2)
                aibj_3 /= R**4
                aibj += aibj_3

            if order > 3:
                Oj = octupole_op(mol, R=Rj, atmlst=atmlst_j)
                omega_bj = einsum('ua,xyzuv,v->xyza', pao_j.conj(), Oj, lmo_j)
                RR = jnp.outer(R_bar, R_bar)
                RRR = einsum('x,y,z->xyz', R_bar, R_bar, R_bar)

                aibj_4  = einsum('xa,xyzb,yz->ab', mu_ai, omega_bj, RR * 9)
                aibj_4 += einsum('xy,xyza,zb->ab', RR * 9, omega_ai, mu_bj)

                omega_R3_ai = RRR.ravel() @ omega_ai.reshape(27, -1)
                omega_R3_bj = RRR.ravel() @ omega_bj.reshape(27, -1)
                aibj_4 -= jnp.outer(tmp_ai, omega_R3_bj * 21)
                aibj_4 -= jnp.outer(omega_R3_ai, tmp_bj * 21)
                aibj_4 += jnp.outer(tmp1_ai, tmp1_bj * 35)

                tmp2_ai = einsum('xya,y->xa', theta_ai, R_bar)
                tmp2_bj = einsum('xyb,y->xb', theta_bj, R_bar)
                aibj_4 -= tmp2_ai.T @ (tmp2_bj * 20)
                aibj_4 += theta_ai.reshape(9, -1).T @ (theta_bj.reshape(9, -1) * 2)
                aibj_4 /= (3 * R**5)
                aibj += aibj_4

            e_bj = jnp.asarray(e_vir[j]) - e_occ[j]
            aibj2 = aibj * aibj / (e_ai[:,None] + e_bj[None,:])
            e_mp2_pair = e_mp2_pair.at[i, j].set(-8 * jnp.sum(aibj2))

    e_mp2_pair = e_mp2_pair + e_mp2_pair.T
    return e_mp2_pair


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
    t2 = oovv / lib.direct_sum('jb+ia->ijba', eia, eia)

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
    """Compute Lov = (naux, nocc*nvir) using pyscfad ao2mo helpers.

    Uses `_ao2mo.nr_e2` to transform density-fitting cderi to MO basis
    and reshape to (naux, nocc*nvir).
    """
    try:
        from pyscfad.ao2mo import _ao2mo
    except Exception:
        # older installs might expose via pyscf
        from pyscf.ao2mo import _ao2mo

    get_cderi = getattr(mp.with_df, '_get_cderi_source', None)
    cderi = get_cderi() if get_cderi is not None else mp.with_df._cderi
    mo = mo_coeff
    nmo = mo.shape[1]
    nocc = mp.nocc
    # ijslice selecting occupied and virtual blocks when forming Lpq for k-point or general
    ijslice = (0, nocc, nocc, nmo)

    # _ao2mo.nr_e2 accepts (eri_1d, mo, ijslice, aosym='s2') or packed cderi
    # produce Lpq shaped (naux, nmo, nmo)
    try:
        from pyscfad.df import addons as df_addons
        with df_addons.load(cderi, 'j3c') as eri1:
            eri1 = numpy.asarray(eri1)
            Lpq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2')
    except Exception:
        # fallback: try r_e2 with explicit parameters
        Lpq = _ao2mo.r_e2(cderi, mo, ijslice, [])

    # reshape to (naux, nocc, nvir)
    naux = Lpq.shape[0]
    nvir = nmo - nocc
    # depending on _ao2mo output layout, ensure correct reshape
    try:
        Lpq = Lpq.reshape(naux, nmo, nmo)
    except Exception:
        # already in desired shape
        pass

    # slice occupied/virtual blocks and flatten
    Lov = Lpq[:, :nocc, nocc:]
    return Lov.reshape((naux, nocc * nvir))


def kernel_dfmp2(fake_mf, prj, with_t2=False):
    # small convenience wrapper if needed
    _mp2 = dfmp2.DFMP2(fake_mf)
    return _mp2.kernel(prj, with_t2=with_t2)


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
