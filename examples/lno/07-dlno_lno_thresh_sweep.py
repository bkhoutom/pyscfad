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
from pathlib import Path

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyscfad import config
import solvent_test as st

THRESHOLDS = [1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7]

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "07-dlno_lno_thresh_sweep.csv"
PNG_PATH = HERE / "07-dlno_lno_thresh_sweep.png"


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
    config.update("pyscfad_scf_implicit_diff", True)
    config.update("pyscfad_scf_first_order_custom", False)
    config.update("pyscfad_ccsd_implicit_diff", True)

    mol = st.build_mol(st.atom)
    rows = []

    for threshold in THRESHOLDS:
        st.lno_thresh = float(threshold)
        t0 = time.time()
        reference = st.build_dlno_reference(mol)
        e_lno, g_lno = jax.value_and_grad(lambda x: st.lno_total_energy(x, reference["frag_lolist"]))(mol)
        e_dlno, g_dlno = jax.value_and_grad(lambda x: st.dlno_total_energy(x, reference))(mol)

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


if __name__ == "__main__":
    main()
