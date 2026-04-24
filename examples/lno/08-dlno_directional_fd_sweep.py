"""Compare backprop and five-point directional finite differences vs lno_thresh.

For a fixed random direction in nuclear-coordinate space, this script compares
the directional derivative obtained from backpropagation against a five-point
central finite-difference stencil for LNO-CCSD and three DLNO-CCSD variants.

The DLNO topology thresholds are defined locally in this script.  Only the
final LNO truncation threshold is varied on the x-axis; the tight DLNO-MP2
correction uses its own fixed tight LNO threshold.

Outputs:
- CSV summary next to this script
- PNG plot next to this script
"""
from __future__ import annotations

import csv
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
THRESHOLDS = [1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7]
FD_H = 1e-2
SEED = 20260422
DLNO_CCSD_DOMAIN_PAO_THR = 1e-5
DLNO_CCSD_PAIR_ENERGY_THR = 1e-5
DLNO_MP2_LNO_THRESH = 1e-7
DLNO_MP2_DOMAIN_PAO_THR = 1e-7
DLNO_MP2_PAIR_ENERGY_THR = 1e-7
LMO_BP_DOMAIN_THR = 0.9
PAO_BP_DOMAIN_THR = 0.9
MULTIPOLE_ORDER = 4

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
CSV_PATH = HERE / "08-dlno_directional_fd_sweep.csv"
PNG_PATH = HERE / "08-dlno_directional_fd_sweep.png"


def build_mol(atom_spec=ATOM):
    mol = gto.Mole(atom=atom_spec, basis=BASIS, verbose=2)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def mol_from_coords_bohr(symbols, coords_bohr):
    atom = [(sym, tuple(map(float, xyz))) for sym, xyz in zip(symbols, coords_bohr)]
    mol = gto.Mole(atom=atom, basis=BASIS, unit="Bohr", verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def run_rhf(mol):
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    return mf


def build_local_orbitals_and_fragments(mf, *, thresh, lo_type=LO_TYPE):
    lo = LNOCCSD(mf, thresh=thresh, frozen=FROZEN).get_lo(lo_type=lo_type)
    frag_atoms = stop_trace(autofrag)(mf.mol)
    frag_los = stop_trace(map_lo_to_frag)(mf.mol, lo, frag_atoms, verbose=mf.mol.verbose)
    return lo, frag_los


def make_canonical_mp2_solver(mf):
    pt = mp.dfmp2.MP2(mf, frozen=FROZEN)
    pt.kernel(with_t2=False)
    return pt


def make_local_mp2_solver(mf, *, thresh, lo_type=LO_TYPE):
    pt = LNOMP2(mf, thresh=thresh, frozen=FROZEN)
    pt.thresh_occ = pt.thresh_vir = thresh
    pt.lo_type, pt.no_type = lo_type, "ie"
    return pt


def make_cc_solver(mf, *, thresh, lo_type=LO_TYPE, ccsd_t=False):
    cc = LNOCCSD(mf, thresh=thresh, frozen=FROZEN)
    cc.thresh_occ = cc.thresh_vir = thresh
    cc.lo_type, cc.no_type, cc.ccsd_t = lo_type, "ie", ccsd_t
    return cc


def build_dlno_topology(
    mf,
    lo,
    frag_los,
    *,
    domain_pao_thr,
    pair_energy_thr,
):
    return stop_trace(build_dlno_prescreen_data)(
        mf,
        lo,
        frag_los,
        frozen=FROZEN,
        lmo_bp_domain_thr=LMO_BP_DOMAIN_THR,
        pao_bp_domain_thr=PAO_BP_DOMAIN_THR,
        domain_pao_thr=domain_pao_thr,
        pair_energy_thr=pair_energy_thr,
        multipole_order=MULTIPOLE_ORDER,
    )


def build_current_dlno_data(mf, lo, frag_los, *, domain_pao_thr, pair_energy_thr):
    topology = build_dlno_topology(
        mf,
        lo,
        frag_los,
        domain_pao_thr=domain_pao_thr,
        pair_energy_thr=pair_energy_thr,
    )
    return rebuild_dlno_prescreen_data(mf, lo, topology, frozen=FROZEN)


def enable_dlno_prescreen(solver, dlno_data):
    solver.use_dlno_prescreen = True
    solver.dlno_prescreen_data = dlno_data
    return solver


def five_point_directional_derivative(fn, symbols, base_coords, direction, h):
    e_m2 = float(fn(mol_from_coords_bohr(symbols, base_coords - 2 * h * direction)))
    e_m1 = float(fn(mol_from_coords_bohr(symbols, base_coords - h * direction)))
    e_p1 = float(fn(mol_from_coords_bohr(symbols, base_coords + h * direction)))
    e_p2 = float(fn(mol_from_coords_bohr(symbols, base_coords + 2 * h * direction)))
    return (e_m2 - 8 * e_m1 + 8 * e_p1 - e_p2) / (12 * h)


def total_energy(mf, cc, pt=None):
    if pt is None:
        return mf.e_tot + cc.e_corr
    return mf.e_tot + cc.e_corr_pt2corrected(pt.e_corr)


def build_lno_inputs(mol, *, thresh):
    mf = run_rhf(mol)
    lo, frag_los = build_local_orbitals_and_fragments(mf, thresh=thresh)
    return mf, lo, frag_los


def build_dlno_inputs(
    mol,
    *,
    thresh,
    domain_pao_thr=DLNO_CCSD_DOMAIN_PAO_THR,
    pair_energy_thr=DLNO_CCSD_PAIR_ENERGY_THR,
):
    mf, lo, frag_los = build_lno_inputs(mol, thresh=thresh)
    dlno_data = build_current_dlno_data(
        mf,
        lo,
        frag_los,
        domain_pao_thr=domain_pao_thr,
        pair_energy_thr=pair_energy_thr,
    )
    return mf, lo, frag_los, dlno_data


def lno_total_energy(mol, *, thresh):
    mf, lo, _ = build_lno_inputs(mol, thresh=thresh)
    cc = make_cc_solver(mf, thresh=thresh, ccsd_t=False)
    cc.kernel(orbloc=lo)
    pt = make_canonical_mp2_solver(mf)
    return total_energy(mf, cc, pt)


def dlno_total_energy(mol, *, thresh, correction):
    mf, lo, frag_los, dlno_data = build_dlno_inputs(mol, thresh=thresh)
    cc = enable_dlno_prescreen(make_cc_solver(mf, thresh=thresh, ccsd_t=False), dlno_data)
    cc.kernel(frag_lolist=frag_los, orbloc=lo)

    if correction == "none":
        return total_energy(mf, cc)
    if correction == "canonical":
        pt = make_canonical_mp2_solver(mf)
    elif correction == "dlno":
        mp2_lo, mp2_frag_los = build_local_orbitals_and_fragments(
            mf,
            thresh=DLNO_MP2_LNO_THRESH,
        )
        mp2_dlno_data = build_current_dlno_data(
            mf,
            mp2_lo,
            mp2_frag_los,
            domain_pao_thr=DLNO_MP2_DOMAIN_PAO_THR,
            pair_energy_thr=DLNO_MP2_PAIR_ENERGY_THR,
        )
        pt = enable_dlno_prescreen(
            make_local_mp2_solver(mf, thresh=DLNO_MP2_LNO_THRESH),
            mp2_dlno_data,
        )
        pt.kernel(frag_lolist=mp2_frag_los, orbloc=mp2_lo)
    else:
        raise ValueError(f"Unknown correction type: {correction}")

    return total_energy(mf, cc, pt)


def build_energy_functions(thresh):
    def lno_fn(mol):
        return lno_total_energy(mol, thresh=thresh)

    def dlno_bare_fn(mol):
        return dlno_total_energy(mol, thresh=thresh, correction="none")

    def dlno_can_fn(mol):
        return dlno_total_energy(mol, thresh=thresh, correction="canonical")

    def dlno_dlno_fn(mol):
        return dlno_total_energy(mol, thresh=thresh, correction="dlno")

    return lno_fn, dlno_bare_fn, dlno_can_fn, dlno_dlno_fn


def plot_results(rows):
    thresholds = np.asarray([row["lno_thresh"] for row in rows], dtype=float)
    lno_abs = np.asarray([row["lno_abs_err"] for row in rows], dtype=float)
    dlno_bare_abs = np.asarray([row["dlno_bare_abs_err"] for row in rows], dtype=float)
    dlno_can_abs = np.asarray([row["dlno_can_abs_err"] for row in rows], dtype=float)
    dlno_dlno_abs = np.asarray([row["dlno_dlno_abs_err"] for row in rows], dtype=float)
    lno_rel = np.asarray([row["lno_rel_err"] for row in rows], dtype=float)
    dlno_bare_rel = np.asarray([row["dlno_bare_rel_err"] for row in rows], dtype=float)
    dlno_can_rel = np.asarray([row["dlno_can_rel_err"] for row in rows], dtype=float)
    dlno_dlno_rel = np.asarray([row["dlno_dlno_rel_err"] for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True)

    axes[0].plot(thresholds, lno_abs, marker="o", label="LNO-CCSD+MP2", color="#1d4ed8")
    axes[0].plot(thresholds, dlno_bare_abs, marker="^", label="DLNO-CCSD", color="#0f766e")
    axes[0].plot(thresholds, dlno_can_abs, marker="s", label="DLNO-CCSD+MP2", color="#dc2626")
    axes[0].plot(thresholds, dlno_dlno_abs, marker="D", label="DLNO-CCSD+tight DLNO-MP2", color="#7c3aed")
    axes[0].set_ylabel("absolute directional error")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(thresholds, lno_rel, marker="o", label="LNO-CCSD+MP2", color="#1d4ed8")
    axes[1].plot(thresholds, dlno_bare_rel, marker="^", label="DLNO-CCSD", color="#0f766e")
    axes[1].plot(thresholds, dlno_can_rel, marker="s", label="DLNO-CCSD+MP2", color="#dc2626")
    axes[1].plot(thresholds, dlno_dlno_rel, marker="D", label="DLNO-CCSD+tight DLNO-MP2", color="#7c3aed")
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
    config.update("pyscfad_moleintor_opt", True)
    config.update("pyscfad_scf_implicit_diff", True)
    config.update("pyscfad_scf_first_order_custom", False)
    config.update("pyscfad_ccsd_implicit_diff", True)

    base_mol = build_mol()
    base_mol.verbose = 0
    symbols = [base_mol.atom_symbol(i) for i in range(base_mol.natm)]
    base_coords = np.asarray(base_mol.atom_coords())

    rng = np.random.default_rng(SEED)
    direction = rng.normal(size=base_coords.shape)
    direction /= np.linalg.norm(direction)

    rows = []
    for threshold in THRESHOLDS:
        lno_fn, dlno_bare_fn, dlno_canonical_mp2_fn, dlno_fn = build_energy_functions(threshold)

        e_lno, g_lno = jax.value_and_grad(lno_fn)(base_mol)
        e_dlno_bare, g_dlno_bare = jax.value_and_grad(dlno_bare_fn)(base_mol)
        e_dlno_can, g_dlno_can = jax.value_and_grad(dlno_canonical_mp2_fn)(base_mol)
        e_dlno_dlno, g_dlno_dlno = jax.value_and_grad(dlno_fn)(base_mol)

        lno_back = float(np.sum(np.asarray(g_lno.coords) * direction))
        dlno_bare_back = float(np.sum(np.asarray(g_dlno_bare.coords) * direction))
        dlno_can_back = float(np.sum(np.asarray(g_dlno_can.coords) * direction))
        dlno_dlno_back = float(np.sum(np.asarray(g_dlno_dlno.coords) * direction))
        lno_fd = five_point_directional_derivative(lno_fn, symbols, base_coords, direction, FD_H)
        dlno_bare_fd = five_point_directional_derivative(dlno_bare_fn, symbols, base_coords, direction, FD_H)
        dlno_can_fd = five_point_directional_derivative(dlno_canonical_mp2_fn, symbols, base_coords, direction, FD_H)
        dlno_dlno_fd = five_point_directional_derivative(dlno_fn, symbols, base_coords, direction, FD_H)

        lno_abs = abs(lno_fd - lno_back)
        dlno_bare_abs = abs(dlno_bare_fd - dlno_bare_back)
        dlno_can_abs = abs(dlno_can_fd - dlno_can_back)
        dlno_dlno_abs = abs(dlno_dlno_fd - dlno_dlno_back)
        lno_rel = lno_abs / max(abs(lno_back), 1e-16)
        dlno_bare_rel = dlno_bare_abs / max(abs(dlno_bare_back), 1e-16)
        dlno_can_rel = dlno_can_abs / max(abs(dlno_can_back), 1e-16)
        dlno_dlno_rel = dlno_dlno_abs / max(abs(dlno_dlno_back), 1e-16)

        row = {
            "lno_thresh": float(threshold),
            "dlno_mp2_lno_thresh": float(DLNO_MP2_LNO_THRESH),
            "fd_h_bohr": float(FD_H),
            "lno_energy": float(e_lno),
            "dlno_bare_energy": float(e_dlno_bare),
            "dlno_can_energy": float(e_dlno_can),
            "dlno_dlno_energy": float(e_dlno_dlno),
            "lno_back": lno_back,
            "lno_fd5": lno_fd,
            "lno_abs_err": lno_abs,
            "lno_rel_err": lno_rel,
            "dlno_bare_back": dlno_bare_back,
            "dlno_bare_fd5": dlno_bare_fd,
            "dlno_bare_abs_err": dlno_bare_abs,
            "dlno_bare_rel_err": dlno_bare_rel,
            "dlno_can_back": dlno_can_back,
            "dlno_can_fd5": dlno_can_fd,
            "dlno_can_abs_err": dlno_can_abs,
            "dlno_can_rel_err": dlno_can_rel,
            "dlno_dlno_back": dlno_dlno_back,
            "dlno_dlno_fd5": dlno_dlno_fd,
            "dlno_dlno_abs_err": dlno_dlno_abs,
            "dlno_dlno_rel_err": dlno_dlno_rel,
        }
        rows.append(row)
        print(
            f"tau_LNO={threshold:.0e}  "
            f"LNO+MP2=({lno_abs:.3e}, {lno_rel:.3e})  "
            f"DLNO=({dlno_bare_abs:.3e}, {dlno_bare_rel:.3e})  "
            f"DLNO+MP2=({dlno_can_abs:.3e}, {dlno_can_rel:.3e})  "
            f"DLNO+tight DLNO-MP2=({dlno_dlno_abs:.3e}, {dlno_dlno_rel:.3e})"
        )

    with CSV_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_results(rows)
    print(f"\nWrote CSV: {CSV_PATH}")
    print(f"Wrote plot: {PNG_PATH}")
    print()
    print("This script checks the quality of the backpropagated directional derivative")
    print("by comparing it against a five-point finite-difference stencil for")
    print("LNO-CCSD+MP2, bare DLNO-CCSD, DLNO-CCSD+MP2, and")
    print("DLNO-CCSD+tight DLNO-MP2 as the final LNO truncation threshold is varied.")
    print(
        f"The DLNO-MP2 correction uses lno_thresh={DLNO_MP2_LNO_THRESH:.0e}, "
        f"domain_pao_thr={DLNO_MP2_DOMAIN_PAO_THR:.0e}, "
        f"and pair_energy_thr={DLNO_MP2_PAIR_ENERGY_THR:.0e}."
    )


if __name__ == "__main__":
    main()
