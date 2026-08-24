"""Legacy DLNO-prescreen diagnostic for the first-order SCF response.

This exercises the historical PAO-prescreened LNO implementation.  It does
not use the current IAO-DLNO-CCSD(T) solver in :mod:`pyscfad.dlno.ccsd`.
"""

import warnings
from pathlib import Path

import jax
import numpy as np

from pyscfad import config, gto, scf
from pyscfad.cc import dfccsd
from pyscfad.lno import LNOMP2
from pyscfad.lno import ccsd as lnoccsd
from pyscfad.dlno.mp2 import DLNOMP2
from pyscfad.dlno.prescreen import build_dlno_prescreen_data, rebuild_dlno_prescreen_data
from pyscfad.lno.tools import autofrag, map_lo_to_frag
from pyscfad.ops import stop_trace


warnings.filterwarnings(
    "ignore",
    message=r"Function mol\.dumps drops attribute .* because it is not JSON-serializable",
)

config.update("pyscfad_moleintor_opt", True)
config.update("pyscfad_scf_implicit_diff", True)
config.update("pyscfad_scf_first_order_custom", True)
config.update("pyscfad_ccsd_implicit_diff", True)
config.update("pyscfad_dfccsd_custom_response", True)

LO_TYPE = "iao"  # Change to "pm" or "boys" to try other localization types.
BUILD_THR = 1e-4
FINAL_THR = 1e-6
DOMAIN_THR = 1e-4


def build_mol():
    xyz_path = Path(__file__).with_name("water_dimer.xyz")
    atom = "\n".join(xyz_path.read_text().splitlines()[2:5])
    mol = gto.Mole(
        atom=atom,
        basis="sto3g",
        verbose=2,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def build_local_orbitals_and_fragments(mf):
    lo_coeff = lnoccsd.LNOCCSD(mf, thresh=BUILD_THR, frozen=0).get_lo(lo_type=LO_TYPE)
    frag_atmlist = stop_trace(autofrag)(mf.mol)
    frag_lolist = stop_trace(map_lo_to_frag)(
        mf.mol, lo_coeff, frag_atmlist, verbose=mf.mol.verbose
    )
    return lo_coeff, frag_lolist


def make_cc_solver(mf, dlno_data=None):
    mycc = lnoccsd.LNOCCSD(mf, thresh=BUILD_THR, frozen=0)
    if dlno_data is not None:
        # Reproduce the historical DLNO wrapper explicitly.  The current
        # pyscfad.dlno.ccsd solver instead uses IAO-MP2-selected LISs.
        mycc.use_dlno_prescreen = True
        mycc.dlno_prescreen_data = dlno_data
    mycc.thresh_occ = FINAL_THR
    mycc.thresh_vir = FINAL_THR
    mycc.lo_type = LO_TYPE
    mycc.no_type = "ie"
    mycc.ccsd_t = False
    return mycc


def make_local_mp2_solver(mf, dlno_data=None):
    if dlno_data is None:
        mymp = LNOMP2(mf, thresh=BUILD_THR, frozen=0)
    else:
        mymp = DLNOMP2(mf, thresh=BUILD_THR, frozen=0,
                       dlno_prescreen_data=dlno_data)
    mymp.thresh_occ = FINAL_THR
    mymp.thresh_vir = FINAL_THR
    mymp.lo_type = LO_TYPE
    mymp.no_type = "ie"
    return mymp


def _build_static_dlno_topology():
    """Build the DLNO topology once with a concrete (non-traced) SCF.

    ``build_dlno_prescreen_data`` internally calls ``np.asarray`` on PAO
    coefficient matrices, which would blow up on JAX tracers inside
    ``jax.value_and_grad``.  Only ``rebuild_dlno_prescreen_data`` is
    tracer-safe.
    """
    seed_mol = build_mol()
    seed_mf = scf.RHF(seed_mol).density_fit()
    seed_mf.kernel()
    seed_lo, seed_frag_lolist = build_local_orbitals_and_fragments(seed_mf)
    topology = build_dlno_prescreen_data(
        seed_mf,
        seed_lo,
        seed_frag_lolist,
        frozen=0,
        lmo_bp_domain_thr=0.9,
        pao_bp_domain_thr=0.9,
        domain_pao_thr=DOMAIN_THR,
        pair_energy_thr=DOMAIN_THR,
        multipole_order=2,
    )
    return seed_frag_lolist, topology


STATIC_FRAG_LOLIST, STATIC_TOPOLOGY = _build_static_dlno_topology()


def build_dlno_data(mf, lo_coeff):
    return rebuild_dlno_prescreen_data(
        mf, lo_coeff, STATIC_TOPOLOGY, frozen=0,
    )


def lno_total_energy(mol):
    mf = scf.RHF(mol).density_fit()
    ehf = mf.kernel()
    lo_coeff, _ = build_local_orbitals_and_fragments(mf)
    mycc = make_cc_solver(mf)
    mycc.kernel(orbloc=lo_coeff)
    return ehf + mycc.e_corr


def dlno_total_energy(mol):
    mf = scf.RHF(mol).density_fit()
    ehf = mf.kernel()
    lo_coeff, _ = build_local_orbitals_and_fragments(mf)
    dlno_data = build_dlno_data(mf, lo_coeff)
    mymp = make_local_mp2_solver(mf, dlno_data=dlno_data)
    mymp.kernel(frag_lolist=STATIC_FRAG_LOLIST, orbloc=lo_coeff)
    mycc = make_cc_solver(mf, dlno_data=dlno_data)
    mycc.kernel(frag_lolist=STATIC_FRAG_LOLIST, orbloc=lo_coeff)
    return ehf + mycc.e_corr_pt2corrected(mymp.e_corr)


def canonical_total_energy(mol):
    mf = scf.RHF(mol).density_fit()
    ehf = mf.kernel()

    mycc = dfccsd.RCCSD(mf, frozen=0)
    mycc.kernel()
    return ehf + mycc.e_corr


if __name__ == "__main__":
    mol = build_mol()
    e_can, g_can = jax.value_and_grad(canonical_total_energy)(mol)
    e_lno, g_lno = jax.value_and_grad(lno_total_energy)(mol)
    e_dlno, g_dlno = jax.value_and_grad(dlno_total_energy)(mol)

    g_can_arr = np.asarray(g_can.coords)
    g_lno_arr = np.asarray(g_lno.coords)
    g_dlno_arr = np.asarray(g_dlno.coords)
    g_diff = g_dlno_arr - g_lno_arr
    g_lno_can = g_lno_arr - g_can_arr
    g_dlno_can = g_dlno_arr - g_can_arr

    print()
    print("Testing DLNO-prescreened CCSD with the custom first-order CPHF SCF backend,")
    print("and comparing both LNO and DLNO against canonical DF-CCSD on the same system.")
    print()
    print("SCF backend: cphf")
    print(f"LO type: {LO_TYPE}")
    print(f"Canonical DF-CCSD energy:   {float(e_can): .12f}")
    print(f"LNO total energy:           {float(e_lno): .12f}")
    print(f"DLNO-prescreen total energy:{float(e_dlno): .12f}")
    print(f"LNO - canonical:            {float(e_lno - e_can): .6e}")
    print(f"DLNO - canonical:           {float(e_dlno - e_can): .6e}")
    print(f"DLNO - LNO energy diff:     {float(e_dlno - e_lno): .6e}")
    print("CCSD response backend:      custom DF-CCSD response")
    print(f"Max |LNO - canonical grad|: {np.max(np.abs(g_lno_can)): .6e}")
    print(f"Max |DLNO - canonical grad|:{np.max(np.abs(g_dlno_can)): .6e}")
    print(f"Max |DLNO - LNO grad diff|: {np.max(np.abs(g_diff)): .6e}")
    print(f"RMS |DLNO - LNO grad diff|: {np.sqrt(np.mean(g_diff**2)): .6e}")
