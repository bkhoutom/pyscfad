"""Sweep DLNO domain thresholds and compare against LNO.

This script uses the rebuilt-DLNO path from solvent_test.py.  For each
threshold value it:
1. Builds a fixed DLNO topology at the reference geometry.
2. Rebuilds the current DLNO prescreen spaces on the differentiated path.
3. Compares DLNO and LNO total energies and nuclear gradients.

Outputs:
- CSV summary next to this script
- PNG plot next to this script
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyscfad.lno.prescreen import build_dlno_prescreen_data, rebuild_dlno_prescreen_data
from solvent_test import (
    atom,
    build_local_orbitals_and_fragments,
    build_mol,
    lno_total_energy,
    make_lno_solver,
    make_mp2_solver,
    run_rhf,
)

# Shared threshold sweep.  Keep the list short enough that gradients finish in
# a reasonable time but broad enough to show the onset of truncation error.
THRESHOLDS = [0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
LMO_BP_DOMAIN_THR = 0.9
PAO_BP_DOMAIN_THR = 0.9
MULTIPOLE_ORDER = 2

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "06-dlno_threshold_sweep.csv"
PNG_PATH = HERE / "06-dlno_threshold_sweep.png"


def build_threshold_reference(mol_ref, threshold):
    mf = run_rhf(mol_ref)
    lo_coeff, frag_atmlist, frag_lolist = build_local_orbitals_and_fragments(mf)
    topology = build_dlno_prescreen_data(
        mf,
        lo_coeff,
        frag_lolist,
        frozen=0,
        lmo_bp_domain_thr=LMO_BP_DOMAIN_THR,
        pao_bp_domain_thr=PAO_BP_DOMAIN_THR,
        domain_pao_thr=threshold,
        pair_energy_thr=threshold,
        multipole_order=MULTIPOLE_ORDER,
    )
    primary_sizes = [len(np.asarray(f["extended_primary_domain"]).ravel()) for f in topology["fragment_data"]]
    strong_sizes = [len(np.asarray(f["strong_lmo_indices"]).ravel()) for f in topology["fragment_data"]]
    return {
        "frag_atmlist": frag_atmlist,
        "frag_lolist": frag_lolist,
        "dlno_topology": topology,
        "primary_sizes": primary_sizes,
        "strong_sizes": strong_sizes,
    }


def dlno_total_energy_threshold(mol, reference):
    mf = run_rhf(mol)
    lo_coeff, _, _ = build_local_orbitals_and_fragments(mf)
    dlno_data = rebuild_dlno_prescreen_data(mf, lo_coeff, reference["dlno_topology"], frozen=0)
    mycc = make_lno_solver(mf)
    mycc.verbose = 0
    mycc.use_dlno_prescreen = True
    mycc.dlno_prescreen_data = dlno_data
    mycc.kernel(frag_lolist=reference["frag_lolist"], orbloc=lo_coeff)
    mymp = make_mp2_solver(mf)
    return mf.e_tot + mycc.e_corr_pt2corrected(mymp.e_corr)


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
    mol = build_mol(atom)

    print("Computing LNO reference once...")
    t0 = time.time()
    ref0 = build_threshold_reference(mol, THRESHOLDS[0])
    e_lno, g_lno = jax.value_and_grad(lambda x: lno_total_energy(x, ref0["frag_lolist"]))(mol)
    g_lno = np.asarray(g_lno.coords)
    print(f"  LNO energy = {float(e_lno):.12f}  (time {time.time() - t0:.1f}s)")

    rows = []
    for threshold in THRESHOLDS:
        print(f"Running threshold {threshold:.0e}..." if threshold else "Running threshold 0 (full-domain limit)...")
        t1 = time.time()
        reference = build_threshold_reference(mol, threshold)
        e_dlno, g_dlno = jax.value_and_grad(lambda x: dlno_total_energy_threshold(x, reference))(mol)
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
            "mean_primary_domain_atoms": float(np.mean(reference["primary_sizes"])),
            "max_primary_domain_atoms": int(np.max(reference["primary_sizes"])),
            "mean_strong_lmos": float(np.mean(reference["strong_sizes"])),
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


if __name__ == "__main__":
    main()
