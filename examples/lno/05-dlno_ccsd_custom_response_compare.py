"""Compare implicit and custom DF-CCSD response inside LNO/DLNO.

This example holds the SCF response backend fixed and toggles only
``pyscfad_dfccsd_custom_response``.  It exercises the generalized CCSD
backward pass because LNO/DLNO consume the impurity ``t1`` and ``t2``
amplitudes downstream.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import jax
import numpy as np

from pyscfad import config, gto, scf
from pyscfad.lno import LNOMP2
from pyscfad.lno import ccsd as lnoccsd
from pyscfad.dlno.prescreen import build_dlno_prescreen_data, rebuild_dlno_prescreen_data
from pyscfad.lno.tools import autofrag, map_lo_to_frag
from pyscfad.ops import stop_trace


warnings.filterwarnings(
    "ignore",
    message=r"Function mol\.dumps drops attribute .* because it is not JSON-serializable",
)


LO_TYPE = "iao"
BUILD_THR = 1e-4
FINAL_THR = 1e-6
DOMAIN_THR = 1e-4
USE_CUSTOM_SCF_RESPONSE = True


def configure(*, use_custom_ccsd_response: bool) -> None:
    config.update("pyscfad_moleintor_opt", True)
    config.update("pyscfad_scf_implicit_diff", True)
    config.update("pyscfad_scf_first_order_custom", USE_CUSTOM_SCF_RESPONSE)
    config.update("pyscfad_ccsd_implicit_diff", True)
    config.update("pyscfad_dfccsd_custom_response", use_custom_ccsd_response)


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


def make_cc_solver(mf):
    mycc = lnoccsd.LNOCCSD(mf, thresh=BUILD_THR, frozen=0)
    mycc.thresh_occ = FINAL_THR
    mycc.thresh_vir = FINAL_THR
    mycc.lo_type = LO_TYPE
    mycc.no_type = "ie"
    mycc.ccsd_t = False
    return mycc


def make_local_mp2_solver(mf):
    mymp = LNOMP2(mf, thresh=BUILD_THR, frozen=0)
    mymp.thresh_occ = FINAL_THR
    mymp.thresh_vir = FINAL_THR
    mymp.lo_type = LO_TYPE
    mymp.no_type = "ie"
    return mymp


def _build_static_dlno_topology():
    """Build the DLNO topology once with a concrete (non-traced) SCF.

    ``build_dlno_prescreen_data`` internally calls ``np.asarray`` on PAO
    coefficient matrices.  Inside ``jax.value_and_grad`` those are tracers
    and the conversion blows up, so the topology must be built outside the
    AD pass.  Only ``rebuild_dlno_prescreen_data`` (called inside the
    energy function) is tracer-safe.
    """
    # Use the implicit-diff config for the seed.  The topology depends only
    # on the converged SCF + LO partition, not on which CCSD response path
    # is later used to differentiate.
    configure(use_custom_ccsd_response=False)
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


def lno_total_energy(mol, *, use_custom_ccsd_response: bool):
    configure(use_custom_ccsd_response=use_custom_ccsd_response)
    mf = scf.RHF(mol).density_fit()
    ehf = mf.kernel()
    lo_coeff, _ = build_local_orbitals_and_fragments(mf)
    mycc = make_cc_solver(mf)
    mycc.kernel(orbloc=lo_coeff)
    return ehf + mycc.e_corr


def dlno_total_energy(mol, *, use_custom_ccsd_response: bool):
    configure(use_custom_ccsd_response=use_custom_ccsd_response)
    mf = scf.RHF(mol).density_fit()
    ehf = mf.kernel()
    lo_coeff, _ = build_local_orbitals_and_fragments(mf)
    dlno_data = build_dlno_data(mf, lo_coeff)
    mymp = make_local_mp2_solver(mf)
    mymp.use_dlno_prescreen = True
    mymp.dlno_prescreen_data = dlno_data
    mymp.kernel(frag_lolist=STATIC_FRAG_LOLIST, orbloc=lo_coeff)
    mycc = make_cc_solver(mf)
    mycc.use_dlno_prescreen = True
    mycc.dlno_prescreen_data = dlno_data
    mycc.kernel(frag_lolist=STATIC_FRAG_LOLIST, orbloc=lo_coeff)
    return ehf + mycc.e_corr_pt2corrected(mymp.e_corr)


def run_backend(label, energy_fn, *, use_custom_ccsd_response: bool):
    configure(use_custom_ccsd_response=use_custom_ccsd_response)
    mol = build_mol()
    t0 = time.perf_counter()
    energy, grad = jax.value_and_grad(
        lambda mm: energy_fn(
            mm, use_custom_ccsd_response=use_custom_ccsd_response
        )
    )(mol)
    elapsed = time.perf_counter() - t0
    return {
        "label": label,
        "energy": float(energy),
        "grad": np.asarray(grad.coords),
        "elapsed_s": elapsed,
    }


def compare(ref, trial):
    grad_diff = trial["grad"] - ref["grad"]
    return {
        "energy_diff": trial["energy"] - ref["energy"],
        "grad_max_abs": float(np.max(np.abs(grad_diff))),
        "grad_rms_abs": float(np.sqrt(np.mean(grad_diff**2))),
        "speedup": ref["elapsed_s"] / max(trial["elapsed_s"], 1e-12),
    }


def report(name, ref, trial):
    diff = compare(ref, trial)
    print()
    print(f"{name}: implicit CCSD response vs custom DF-CCSD response")
    print(f"Implicit energy: {ref['energy']:.15f}")
    print(f"Custom   energy: {trial['energy']:.15f}")
    print(f"Energy difference: {diff['energy_diff']:.6e}")
    print(f"Implicit time [s]: {ref['elapsed_s']:.3f}")
    print(f"Custom   time [s]: {trial['elapsed_s']:.3f}")
    print(f"Approximate speedup: {diff['speedup']:.3f}x")
    print(f"Max |gradient difference|: {diff['grad_max_abs']:.6e}")
    print(f"RMS |gradient difference|: {diff['grad_rms_abs']:.6e}")
    return diff


if __name__ == "__main__":
    lno_implicit = run_backend(
        "LNO implicit", lno_total_energy, use_custom_ccsd_response=False
    )
    lno_custom = run_backend(
        "LNO custom", lno_total_energy, use_custom_ccsd_response=True
    )
    dlno_implicit = run_backend(
        "DLNO implicit", dlno_total_energy, use_custom_ccsd_response=False
    )
    dlno_custom = run_backend(
        "DLNO custom", dlno_total_energy, use_custom_ccsd_response=True
    )

    print()
    print("Example: LNO/DLNO gradients with two CCSD response routes")
    print(f"SCF response backend held fixed: custom first-order CPHF = {USE_CUSTOM_SCF_RESPONSE}")
    print(f"LO type: {LO_TYPE}")

    lno_diff = report("LNO", lno_implicit, lno_custom)
    dlno_diff = report("DLNO", dlno_implicit, dlno_custom)

    custom_dlno_lno = dlno_custom["grad"] - lno_custom["grad"]
    print()
    print("Custom-response DLNO consistency check")
    print(f"DLNO - LNO energy: {dlno_custom['energy'] - lno_custom['energy']:.6e}")
    print(f"Max |DLNO - LNO gradient|: {np.max(np.abs(custom_dlno_lno)):.6e}")

    if (
        abs(lno_diff["energy_diff"]) > 1e-6
        or lno_diff["grad_max_abs"] > 1e-5
        or abs(dlno_diff["energy_diff"]) > 1e-6
        or dlno_diff["grad_max_abs"] > 1e-5
    ):
        raise RuntimeError("Custom DF-CCSD response did not reproduce implicit gradients.")
