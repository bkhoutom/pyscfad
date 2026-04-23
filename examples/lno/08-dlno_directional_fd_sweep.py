"""Compare backprop and five-point directional finite differences vs lno_thresh.

For a fixed random direction in nuclear-coordinate space, this script compares
the directional derivative obtained from backpropagation against a five-point
central finite-difference stencil for both the parent LNO energy and the DLNO
prescreened energy.

The DLNO topology thresholds are kept fixed by ``solvent_test.py``.  Only the
final LNO truncation threshold is varied here.

Outputs:
- CSV summary next to this script
- PNG plot next to this script
"""
from __future__ import annotations

import csv
from pathlib import Path

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyscfad import config, gto
import solvent_test as st

THRESHOLDS = [1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7]
FD_H = 1e-2
SEED = 20260422

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "08-dlno_directional_fd_sweep.csv"
PNG_PATH = HERE / "08-dlno_directional_fd_sweep.png"


def mol_from_coords_bohr(symbols, coords_bohr):
    atom = [(sym, tuple(map(float, xyz))) for sym, xyz in zip(symbols, coords_bohr)]
    mol = gto.Mole(atom=atom, basis=st.BASIS, unit="Bohr", verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def five_point_directional_derivative(fn, symbols, base_coords, direction, h):
    e_m2 = float(fn(mol_from_coords_bohr(symbols, base_coords - 2 * h * direction)))
    e_m1 = float(fn(mol_from_coords_bohr(symbols, base_coords - h * direction)))
    e_p1 = float(fn(mol_from_coords_bohr(symbols, base_coords + h * direction)))
    e_p2 = float(fn(mol_from_coords_bohr(symbols, base_coords + 2 * h * direction)))
    return (e_m2 - 8 * e_m1 + 8 * e_p1 - e_p2) / (12 * h)


def plot_results(rows):
    thresholds = np.asarray([row["lno_thresh"] for row in rows], dtype=float)
    lno_abs = np.asarray([row["lno_abs_err"] for row in rows], dtype=float)
    dlno_abs = np.asarray([row["dlno_abs_err"] for row in rows], dtype=float)
    lno_rel = np.asarray([row["lno_rel_err"] for row in rows], dtype=float)
    dlno_rel = np.asarray([row["dlno_rel_err"] for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True)

    axes[0].plot(thresholds, lno_abs, marker="o", label="LNO", color="#1d4ed8")
    axes[0].plot(thresholds, dlno_abs, marker="s", label="DLNO", color="#dc2626")
    axes[0].set_ylabel("absolute directional error")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(thresholds, lno_rel, marker="o", label="LNO", color="#1d4ed8")
    axes[1].plot(thresholds, dlno_rel, marker="s", label="DLNO", color="#dc2626")
    axes[1].set_ylabel("relative directional error")
    axes[1].set_xlabel(r"$\tau_{\mathrm{LNO}}$")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.25)

    for ax in axes:
        ax.set_xscale("log")
    axes[-1].set_xticks(thresholds)
    axes[-1].set_xticklabels([f"{t:.0e}" for t in thresholds], rotation=45, ha="right")

    fig.suptitle("Directional derivative check vs LNO threshold")
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    config.update("pyscfad_scf_implicit_diff", True)
    config.update("pyscfad_scf_first_order_custom", False)
    config.update("pyscfad_ccsd_implicit_diff", True)

    base_mol = st.build_mol(st.atom)
    base_mol.verbose = 0
    symbols = [base_mol.atom_symbol(i) for i in range(base_mol.natm)]
    base_coords = np.asarray(base_mol.atom_coords())

    rng = np.random.default_rng(SEED)
    direction = rng.normal(size=base_coords.shape)
    direction /= np.linalg.norm(direction)

    rows = []
    for threshold in THRESHOLDS:
        st.lno_thresh = float(threshold)
        reference = st.build_dlno_reference(base_mol)

        def lno_fn(mol):
            return st.lno_total_energy(mol, reference["frag_lolist"])

        def dlno_fn(mol):
            return st.dlno_total_energy(mol, reference)

        e_lno, g_lno = jax.value_and_grad(lno_fn)(base_mol)
        e_dlno, g_dlno = jax.value_and_grad(dlno_fn)(base_mol)

        lno_back = float(np.sum(np.asarray(g_lno.coords) * direction))
        dlno_back = float(np.sum(np.asarray(g_dlno.coords) * direction))
        lno_fd = five_point_directional_derivative(lno_fn, symbols, base_coords, direction, FD_H)
        dlno_fd = five_point_directional_derivative(dlno_fn, symbols, base_coords, direction, FD_H)

        lno_abs = abs(lno_fd - lno_back)
        dlno_abs = abs(dlno_fd - dlno_back)
        lno_rel = lno_abs / max(abs(lno_back), 1e-16)
        dlno_rel = dlno_abs / max(abs(dlno_back), 1e-16)

        row = {
            "lno_thresh": float(threshold),
            "fd_h_bohr": float(FD_H),
            "lno_energy": float(e_lno),
            "dlno_energy": float(e_dlno),
            "lno_back": lno_back,
            "lno_fd5": lno_fd,
            "lno_abs_err": lno_abs,
            "lno_rel_err": lno_rel,
            "dlno_back": dlno_back,
            "dlno_fd5": dlno_fd,
            "dlno_abs_err": dlno_abs,
            "dlno_rel_err": dlno_rel,
        }
        rows.append(row)
        print(
            f"tau_LNO={threshold:.0e}  "
            f"LNO abs/rel=({lno_abs:.3e}, {lno_rel:.3e})  "
            f"DLNO abs/rel=({dlno_abs:.3e}, {dlno_rel:.3e})"
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
