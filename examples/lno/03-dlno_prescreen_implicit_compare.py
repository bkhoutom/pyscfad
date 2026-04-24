import warnings
from pathlib import Path

import jax
import numpy as np

from pyscfad import config, gto, scf
from pyscfad.lno import LNOMP2
from pyscfad.lno import ccsd as lnoccsd
from pyscfad.lno.prescreen import build_dlno_prescreen_data, rebuild_dlno_prescreen_data
from pyscfad.lno.tools import autofrag, map_lo_to_frag
from pyscfad.ops import stop_trace


warnings.filterwarnings(
    "ignore",
    message=r"Function mol\.dumps drops attribute .* because it is not JSON-serializable",
)

config.update("pyscfad_scf_implicit_diff", True)
config.update("pyscfad_scf_first_order_custom", False)
config.update("pyscfad_ccsd_implicit_diff", True)

LO_TYPE = "iao"  # Change to "pm" or "boys" to try other localization types.
THRESH = 1e-4
DOMAIN_THR = 1e-6


def build_mol():
    xyz_path = Path(__file__).with_name("water_dimer.xyz")
    atom = "\n".join(xyz_path.read_text().splitlines()[2:])
    mol = gto.Mole(
        atom=atom,
        basis="cc-pvdz",
        verbose=2,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def build_local_orbitals_and_fragments(mf):
    lo_coeff = lnoccsd.LNOCCSD(mf, thresh=THRESH, frozen=0).get_lo(lo_type=LO_TYPE)
    frag_atmlist = stop_trace(autofrag)(mf.mol)
    frag_lolist = stop_trace(map_lo_to_frag)(
        mf.mol, lo_coeff, frag_atmlist, verbose=mf.mol.verbose
    )
    return lo_coeff, frag_lolist


def make_cc_solver(mf):
    mycc = lnoccsd.LNOCCSD(mf, thresh=THRESH, frozen=0)
    mycc.thresh_occ = THRESH
    mycc.thresh_vir = THRESH
    mycc.lo_type = LO_TYPE
    mycc.no_type = "ie"
    mycc.ccsd_t = False
    return mycc


def make_local_mp2_solver(mf):
    mymp = LNOMP2(mf, thresh=THRESH, frozen=0)
    mymp.thresh_occ = THRESH
    mymp.thresh_vir = THRESH
    mymp.lo_type = LO_TYPE
    mymp.no_type = "ie"
    return mymp


def build_dlno_data(mf, lo_coeff, frag_lolist):
    topology = stop_trace(build_dlno_prescreen_data)(
        mf,
        lo_coeff,
        frag_lolist,
        frozen=0,
        lmo_bp_domain_thr=0.9,
        pao_bp_domain_thr=0.9,
        domain_pao_thr=DOMAIN_THR,
        pair_energy_thr=DOMAIN_THR,
        multipole_order=2,
    )
    return rebuild_dlno_prescreen_data(mf, lo_coeff, topology, frozen=0)


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
    lo_coeff, frag_lolist = build_local_orbitals_and_fragments(mf)
    dlno_data = build_dlno_data(mf, lo_coeff, frag_lolist)
    mymp = make_local_mp2_solver(mf)
    mymp.use_dlno_prescreen = True
    mymp.dlno_prescreen_data = dlno_data
    mymp.kernel(frag_lolist=frag_lolist, orbloc=lo_coeff)
    mycc = make_cc_solver(mf)
    mycc.use_dlno_prescreen = True
    mycc.dlno_prescreen_data = dlno_data
    mycc.kernel(frag_lolist=frag_lolist, orbloc=lo_coeff)
    return ehf + mycc.e_corr_pt2corrected(mymp.e_corr)


if __name__ == "__main__":
    mol = build_mol()
    e_lno, g_lno = jax.value_and_grad(lno_total_energy)(mol)
    e_dlno, g_dlno = jax.value_and_grad(dlno_total_energy)(mol)

    g_lno_arr = np.asarray(g_lno.coords)
    g_dlno_arr = np.asarray(g_dlno.coords)
    g_diff = g_dlno_arr - g_lno_arr

    print()
    print("Testing whether DLNO-prescreened CCSD reproduces parent LNO-CCSD")
    print("when both use the standard implicit SCF derivative route.")
    print()
    print(f"LO type: {LO_TYPE}")
    print(f"LNO total energy:           {float(e_lno): .12f}")
    print(f"DLNO-prescreen total energy:{float(e_dlno): .12f}")
    print(f"Energy difference:          {float(e_dlno - e_lno): .6e}")
    print(f"Max |gradient diff|:        {np.max(np.abs(g_diff)): .6e}")
    print(f"RMS gradient diff:          {np.sqrt(np.mean(g_diff**2)): .6e}")
