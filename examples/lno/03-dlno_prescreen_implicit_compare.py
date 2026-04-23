import warnings
from pathlib import Path

import jax
import numpy as np

from pyscfad import config, gto, scf
from pyscfad.lno import ccsd as lnoccsd
from pyscfad.lno.prescreen import build_dlno_prescreen_data
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


def build_reference_data(mol):
    mf = scf.RHF(mol).density_fit()
    mf.kernel()

    helper = lnoccsd.LNOCCSD(mf, thresh=1e-4, frozen=0)
    lo_coeff = helper.get_lo(lo_type=LO_TYPE)
    frag_atmlist = stop_trace(autofrag)(mol)
    frag_lolist = stop_trace(map_lo_to_frag)(
        mol, lo_coeff, frag_atmlist, verbose=mol.verbose
    )

    dlno_data = build_dlno_prescreen_data(
        mf,
        lo_coeff,
        frag_lolist,
        frozen=0,
        lmo_bp_domain_thr=0.9,
        pao_bp_domain_thr=0.9,
        domain_pao_thr=1e-6,
        pair_energy_thr=1e-6,
        multipole_order=2,
    )
    return frag_lolist, dlno_data


def total_energy(mol, frag_lolist, dlno_data=None):
    mf = scf.RHF(mol).density_fit()
    ehf = mf.kernel()

    helper = lnoccsd.LNOCCSD(mf, thresh=1e-4, frozen=0)
    lo_coeff = helper.get_lo(lo_type=LO_TYPE)

    mycc = lnoccsd.LNOCCSD(mf, thresh=1e-4, frozen=0)
    mycc.thresh_occ = 1e-4
    mycc.thresh_vir = 1e-4
    mycc.lo_type = LO_TYPE
    mycc.no_type = "ie"
    mycc.ccsd_t = False
    if dlno_data is not None:
        mycc.use_dlno_prescreen = True
        mycc.dlno_prescreen_data = dlno_data
    mycc.kernel(frag_lolist=frag_lolist, orbloc=lo_coeff)
    return ehf + mycc.e_corr


if __name__ == "__main__":
    mol = build_mol()
    frag_lolist, dlno_data = build_reference_data(mol)

    e_lno, g_lno = jax.value_and_grad(lambda x: total_energy(x, frag_lolist))(mol)
    e_dlno, g_dlno = jax.value_and_grad(
        lambda x: total_energy(x, frag_lolist, dlno_data=dlno_data)
    )(mol)

    g_lno_arr = np.asarray(g_lno.coords)
    g_dlno_arr = np.asarray(g_dlno.coords)
    g_diff = g_dlno_arr - g_lno_arr

    print()
    print(f"LO type: {LO_TYPE}")
    print(f"LNO total energy:           {float(e_lno): .12f}")
    print(f"DLNO-prescreen total energy:{float(e_dlno): .12f}")
    print(f"Energy difference:          {float(e_dlno - e_lno): .6e}")
    print(f"Max |gradient diff|:        {np.max(np.abs(g_diff)): .6e}")
    print(f"RMS gradient diff:          {np.sqrt(np.mean(g_diff**2)): .6e}")
