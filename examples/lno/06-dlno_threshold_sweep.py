"""Sweep DLNO domain thresholds and compare against LNO.

For each threshold value it:
1. Builds a fixed DLNO topology at the reference geometry.
2. Rebuilds the current DLNO prescreen spaces on the differentiated path.
3. Compares DLNO and LNO total energies and nuclear gradients.

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

# Shared threshold sweep.  Keep the list short enough that gradients finish in
# a reasonable time but broad enough to show the onset of truncation error.
BASIS = "def2-svp"
LO_TYPE = "iao"
FROZEN = 0
LNO_THRESH = 1e-5
THRESHOLDS = [0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
LMO_BP_DOMAIN_THR = 0.9
PAO_BP_DOMAIN_THR = 0.9
MULTIPOLE_ORDER = 2
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
CSV_PATH = HERE / "06-dlno_threshold_sweep.csv"
PNG_PATH = HERE / "06-dlno_threshold_sweep.png"


def build_mol(atom_spec=ATOM):
    mol = gto.Mole(atom=atom_spec, basis=BASIS, verbose=2)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def run_rhf(mol):
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    return mf


def build_local_orbitals_and_fragments(mf, *, thresh=LNO_THRESH, lo_type=LO_TYPE):
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


def make_local_mp2_solver(mf, *, thresh=LNO_THRESH, lo_type=LO_TYPE):
    mymp = LNOMP2(mf, thresh=thresh, frozen=FROZEN)
    mymp.thresh_occ = mymp.thresh_vir = thresh
    mymp.lo_type, mymp.no_type = lo_type, "ie"
    return mymp


def make_cc_solver(mf):
    cc = LNOCCSD(mf, thresh=LNO_THRESH, frozen=FROZEN)
    cc.thresh_occ = cc.thresh_vir = LNO_THRESH
    cc.lo_type, cc.no_type, cc.ccsd_t = LO_TYPE, "ie", False
    cc.verbose = 0
    return cc


def build_dlno_topology(mf, lo_coeff, frag_lolist, *, domain_pao_thr, pair_energy_thr):
    return stop_trace(build_dlno_prescreen_data)(
        mf,
        lo_coeff,
        frag_lolist,
        frozen=FROZEN,
        lmo_bp_domain_thr=LMO_BP_DOMAIN_THR,
        pao_bp_domain_thr=PAO_BP_DOMAIN_THR,
        domain_pao_thr=domain_pao_thr,
        pair_energy_thr=pair_energy_thr,
        multipole_order=MULTIPOLE_ORDER,
    )


def build_dlno_data(mf, lo_coeff, topology):
    return rebuild_dlno_prescreen_data(mf, lo_coeff, topology, frozen=FROZEN)


def enable_dlno_prescreen(solver, dlno_data):
    solver.use_dlno_prescreen = True
    solver.dlno_prescreen_data = dlno_data
    return solver


def lno_total_energy(mol):
    mf = run_rhf(mol)
    lo_coeff, _ = build_local_orbitals_and_fragments(mf)
    mycc = make_cc_solver(mf)
    mycc.kernel(orbloc=lo_coeff)
    mymp = make_canonical_mp2_solver(mf)
    return mf.e_tot + mycc.e_corr_pt2corrected(mymp.e_corr)


def dlno_total_energy_threshold(mol, threshold):
    mf = run_rhf(mol)
    lo_coeff, frag_lolist = build_local_orbitals_and_fragments(mf)
    topology = build_dlno_topology(
        mf,
        lo_coeff,
        frag_lolist,
        domain_pao_thr=threshold,
        pair_energy_thr=threshold,
    )
    dlno_data = build_dlno_data(mf, lo_coeff, topology)
    mycc = enable_dlno_prescreen(make_cc_solver(mf), dlno_data)
    mycc.kernel(frag_lolist=frag_lolist, orbloc=lo_coeff)
    mymp = enable_dlno_prescreen(make_local_mp2_solver(mf), dlno_data)
    mymp.kernel(frag_lolist=frag_lolist, orbloc=lo_coeff)
    total = mf.e_tot + mycc.e_corr_pt2corrected(mymp.e_corr)
    return total, {
        "primary_sizes": np.asarray(
            [len(np.asarray(f["extended_primary_domain"]).ravel()) for f in topology["fragment_data"]],
            dtype=float,
        ),
        "strong_sizes": np.asarray(
            [len(np.asarray(f["strong_lmo_indices"]).ravel()) for f in topology["fragment_data"]],
            dtype=float,
        ),
    }


def plot_results(rows):
    thresholds = np.asarray([row["threshold"] for row in rows], dtype=float)
    energy_abs = np.asarray([abs(row["energy_diff"]) for row in rows], dtype=float)
    grad_norm = np.asarray([row["grad_norm_diff"] for row in rows], dtype=float)
    grad_max = np.asarray([row["grad_max_diff"] for row in rows], dtype=float)
    domain_mean = np.asarray([row["mean_primary_domain_atoms"] for row in rows], dtype=float)

    positive = thresholds[thresholds > 0]
    min_positive = positive.min() if positive.size else 1e-8
    plot_x = thresholds.copy()
    plot_x[plot_x == 0.0] = min_positive / 3.0

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 10.0), sharex=True)

    axes[0].plot(plot_x, energy_abs, marker="o", color="#0f766e")
    axes[0].set_ylabel(r"$|E_{DLNO}-E_{LNO}|$")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", alpha=0.25)

    axes[1].plot(plot_x, grad_norm, marker="o", label="norm", color="#1d4ed8")
    axes[1].plot(plot_x, grad_max, marker="s", label="max component", color="#dc2626")
    axes[1].set_ylabel(r"$||G_{DLNO}-G_{LNO}||$")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(frameon=False)

    axes[2].plot(plot_x, domain_mean, marker="o", color="#7c3aed")
    axes[2].set_ylabel("Mean primary domain atoms")
    axes[2].set_xlabel("domain_pao_thr = pair_energy_thr")
    axes[2].grid(True, which="both", alpha=0.25)

    for ax in axes:
        ax.set_xscale("log")
    axes[-1].set_xticks(plot_x)
    axes[-1].set_xticklabels(["0" if t == 0 else f"{t:.0e}" for t in thresholds], rotation=45, ha="right")

    fig.suptitle("DLNO vs LNO Error vs Domain Threshold")
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    config.update("pyscfad_moleintor_opt", True)
    config.update("pyscfad_scf_implicit_diff", True)
    config.update("pyscfad_scf_first_order_custom", True)
    config.update("pyscfad_ccsd_implicit_diff", True)

    mol = build_mol()

    print("Computing LNO reference once...")
    t0 = time.time()
    e_lno, g_lno = jax.value_and_grad(lno_total_energy)(mol)
    g_lno = np.asarray(g_lno.coords)
    print(f"  LNO energy = {float(e_lno):.12f}  (time {time.time() - t0:.1f}s)")

    rows = []
    for threshold in THRESHOLDS:
        print(f"Running threshold {threshold:.0e}..." if threshold else "Running threshold 0 (full-domain limit)...")
        t1 = time.time()
        (e_dlno, info_dlno), g_dlno = jax.value_and_grad(
            lambda x: dlno_total_energy_threshold(x, threshold),
            has_aux=True,
        )(mol)
        g_dlno = np.asarray(g_dlno.coords)
        energy_diff = float(e_dlno - e_lno)
        grad_diff = g_dlno - g_lno
        row = {
            "threshold": float(threshold),
            "energy_dlno": float(e_dlno),
            "energy_lno": float(e_lno),
            "energy_diff": energy_diff,
            "grad_norm_diff": float(np.linalg.norm(grad_diff)),
            "grad_max_diff": float(np.max(np.abs(grad_diff))),
            "mean_primary_domain_atoms": float(np.mean(info_dlno["primary_sizes"])),
            "max_primary_domain_atoms": int(np.max(info_dlno["primary_sizes"])),
            "mean_strong_lmos": float(np.mean(info_dlno["strong_sizes"])),
            "elapsed_s": time.time() - t1,
        }
        rows.append(row)
        print(
            f"  dE = {row['energy_diff']:+.3e}  "
            f"||dG|| = {row['grad_norm_diff']:.3e}  "
            f"max|dG| = {row['grad_max_diff']:.3e}  "
            f"mean domain atoms = {row['mean_primary_domain_atoms']:.1f}"
        )

    with CSV_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_results(rows)
    print(f"\nWrote CSV: {CSV_PATH}")
    print(f"Wrote plot: {PNG_PATH}")
    print()
    print("This sweep tests how closely DLNO tracks the parent LNO energy and gradient")
    print("as the DLNO domain/pair threshold is tightened or loosened, with only the")
    print("DLNO path using local DLNO-MP2 correction while the LNO reference stays fixed.")


if __name__ == "__main__":
    main()
