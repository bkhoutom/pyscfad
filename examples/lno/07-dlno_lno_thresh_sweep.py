"""Sweep LNO truncation threshold and compare DLNO against LNO.

This script keeps the DLNO domain thresholds fixed and varies only the final
LNO-space truncation threshold used by both the parent LNO solver and the DLNO
prescreened variant.  It is intended to answer a narrow question:

    How well does DLNO track LNO as the final local-natural-orbital threshold
    is tightened or loosened?

Outputs:
- CSV summary next to this script
- PNG plot next to this script
"""
from __future__ import annotations

import csv
import time
import warnings
from pathlib import Path

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyscfad import config, gto, mp, scf
from pyscfad.lno import LNOCCSD, LNOMP2
from pyscfad.lno.prescreen import build_dlno_prescreen_data, rebuild_dlno_prescreen_data
from pyscfad.lno.tools import autofrag, map_lo_to_frag
from pyscfad.ops import stop_trace


warnings.filterwarnings(
    "ignore",
    message=r"Function mol\.dumps drops attribute .* because it is not JSON-serializable",
)

BASIS = "def2-svp"
LO_TYPE = "iao"
FROZEN = 0
DLNO_DOMAIN_PAO_THR = 1e-5
DLNO_PAIR_ENERGY_THR = 1e-5
LMO_BP_DOMAIN_THR = 0.9
PAO_BP_DOMAIN_THR = 0.9
MULTIPOLE_ORDER = 2
THRESHOLDS = [1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7]
ATOM = """
O 0.000000000000 0.000000000000 0.000000000000
H 0.756950327264 0.000000000000 0.585882276618
H -0.756950327264 0.000000000000 0.585882276618
O 4.000000000000 0.000000000000 0.000000000000
H 4.756950327264 0.000000000000 0.585882276618
H 3.243049672736 0.000000000000 0.585882276618
O 8.000000000000 0.000000000000 0.000000000000
H 8.756950327264 0.000000000000 0.585882276618
H 7.243049672736 0.000000000000 0.585882276618
"""

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "07-dlno_lno_thresh_sweep.csv"
PNG_PATH = HERE / "07-dlno_lno_thresh_sweep.png"


def build_mol(atom_spec=ATOM):
    mol = gto.Mole(atom=atom_spec, basis=BASIS, verbose=2)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def run_rhf(mol):
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    return mf


def build_local_orbitals_and_fragments(mf, *, thresh, lo_type=LO_TYPE):
    lo_coeff = LNOCCSD(mf, thresh=thresh, frozen=FROZEN).get_lo(lo_type=lo_type)
    frag_atmlist = stop_trace(autofrag)(mf.mol)
    frag_lolist = stop_trace(map_lo_to_frag)(
        mf.mol, lo_coeff, frag_atmlist, verbose=mf.mol.verbose
    )
    return lo_coeff, frag_lolist


def make_canonical_mp2_solver(mf):
    mymp = mp.dfmp2.MP2(mf, frozen=FROZEN)
    mymp.kernel(with_t2=False)
    return mymp


def make_local_mp2_solver(mf, *, thresh, lo_type=LO_TYPE):
    mymp = LNOMP2(mf, thresh=thresh, frozen=FROZEN)
    mymp.thresh_occ = mymp.thresh_vir = thresh
    mymp.lo_type, mymp.no_type = lo_type, "ie"
    return mymp


def make_cc_solver(mf, *, thresh, lo_type=LO_TYPE, ccsd_t=False):
    mycc = LNOCCSD(mf, thresh=thresh, frozen=FROZEN)
    mycc.thresh_occ = mycc.thresh_vir = thresh
    mycc.lo_type, mycc.no_type, mycc.ccsd_t = lo_type, "ie", ccsd_t
    return mycc


def build_dlno_data(mf, lo_coeff, frag_lolist):
    topology = stop_trace(build_dlno_prescreen_data)(
        mf,
        lo_coeff,
        frag_lolist,
        frozen=FROZEN,
        lmo_bp_domain_thr=LMO_BP_DOMAIN_THR,
        pao_bp_domain_thr=PAO_BP_DOMAIN_THR,
        domain_pao_thr=DLNO_DOMAIN_PAO_THR,
        pair_energy_thr=DLNO_PAIR_ENERGY_THR,
        multipole_order=MULTIPOLE_ORDER,
    )
    return rebuild_dlno_prescreen_data(mf, lo_coeff, topology, frozen=FROZEN)


def enable_dlno_prescreen(solver, dlno_data):
    solver.use_dlno_prescreen = True
    solver.dlno_prescreen_data = dlno_data
    return solver


def plot_results(rows):
    thresholds = np.asarray([row["lno_thresh"] for row in rows], dtype=float)
    energy_abs = np.asarray([abs(row["energy_diff"]) for row in rows], dtype=float)
    grad_norm = np.asarray([row["grad_norm_diff"] for row in rows], dtype=float)
    grad_max = np.asarray([row["grad_max_diff"] for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True)

    axes[0].plot(thresholds, energy_abs, marker="o", color="#0f766e")
    axes[0].set_ylabel(r"$|E_{DLNO}-E_{LNO}|$")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", alpha=0.25)

    axes[1].plot(thresholds, grad_norm, marker="o", label="norm", color="#1d4ed8")
    axes[1].plot(thresholds, grad_max, marker="s", label="max component", color="#dc2626")
    axes[1].set_ylabel(r"$||G_{DLNO}-G_{LNO}||$")
    axes[1].set_xlabel(r"$\tau_{\mathrm{LNO}}$")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.set_xscale("log")
    axes[-1].set_xticks(thresholds)
    axes[-1].set_xticklabels([f"{t:.0e}" for t in thresholds], rotation=45, ha="right")

    fig.suptitle("DLNO vs LNO Error vs LNO Threshold")
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    config.update("pyscfad_moleintor_opt", True)
    config.update("pyscfad_scf_implicit_diff", True)
    config.update("pyscfad_scf_first_order_custom", False)
    config.update("pyscfad_ccsd_implicit_diff", True)

    mol = build_mol()
    rows = []

    for threshold in THRESHOLDS:
        t0 = time.time()

        def lno_fn(mol):
            mf = run_rhf(mol)
            lo, _ = build_local_orbitals_and_fragments(mf, thresh=threshold)
            cc = make_cc_solver(mf, thresh=threshold, ccsd_t=False)
            cc.kernel(orbloc=lo)
            pt = make_canonical_mp2_solver(mf)
            return mf.e_tot + cc.e_corr_pt2corrected(pt.e_corr)

        def dlno_fn(mol):
            mf = run_rhf(mol)
            lo, frag_los = build_local_orbitals_and_fragments(mf, thresh=threshold)
            dlno_data = build_dlno_data(mf, lo, frag_los)
            pt = enable_dlno_prescreen(make_local_mp2_solver(mf, thresh=threshold), dlno_data)
            cc = enable_dlno_prescreen(make_cc_solver(mf, thresh=threshold, ccsd_t=False), dlno_data)
            pt.kernel(frag_lolist=frag_los, orbloc=lo)
            cc.kernel(frag_lolist=frag_los, orbloc=lo)
            return mf.e_tot + cc.e_corr_pt2corrected(pt.e_corr)

        e_lno, g_lno = jax.value_and_grad(lno_fn)(mol)
        e_dlno, g_dlno = jax.value_and_grad(dlno_fn)(mol)

        grad_diff = np.asarray(g_dlno.coords - g_lno.coords)
        row = {
            "lno_thresh": float(threshold),
            "energy_lno": float(e_lno),
            "energy_dlno": float(e_dlno),
            "energy_diff": float(e_dlno - e_lno),
            "grad_norm_diff": float(np.linalg.norm(grad_diff)),
            "grad_max_diff": float(np.max(np.abs(grad_diff))),
            "grad_rms_diff": float(np.sqrt(np.mean(grad_diff**2))),
            "elapsed_s": time.time() - t0,
        }
        rows.append(row)
        print(
            f"tau_LNO={threshold:.0e}  "
            f"dE={row['energy_diff']:+.3e}  "
            f"||dG||={row['grad_norm_diff']:.3e}  "
            f"max|dG|={row['grad_max_diff']:.3e}"
        )

    with CSV_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_results(rows)
    print(f"\nWrote CSV: {CSV_PATH}")
    print(f"Wrote plot: {PNG_PATH}")
    print()
    print("This sweep tests how the DLNO-vs-LNO energy and gradient differences change")
    print("as the final LNO truncation threshold is varied, while keeping the DLNO")
    print("prescreening setup fixed for each geometry and using the mixed MP2-correction")
    print("scheme from these examples.")


if __name__ == "__main__":
    main()
