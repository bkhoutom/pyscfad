"""Compare two SCF-gradient backends inside an LNO-CCSD gradient calculation.

What this example does:
1. evaluates a PT2-corrected LNO-CCSD energy and gradient with standard
   implicit SCF backpropagation
2. evaluates the same quantity with the custom first-order occupied-virtual
   CPHF-style SCF response
3. reports the timing, energy difference, and gradient difference between the
   two routes

What we are comparing:
- implicit SCF backpropagation, which is general but relatively expensive
- custom first-order CPHF-style SCF response, which is intended to be more
  efficient for first-order properties
- the DF-CCSD response route is held fixed to the custom plug-in path for both
  calculations

Expected behavior:
- the two energies and gradients should agree closely
- small differences indicate that the custom first-order response is a good
  drop-in replacement for implicit backpropagation in this setting
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import jax
import numpy as np

from pyscfad import config, gto, mp, scf
from pyscfad.lno import LNOCCSD


warnings.filterwarnings(
    "ignore",
    message=r"Function mol\.dumps drops attribute .* because it is not JSON-serializable",
)


BASIS = "def2-svp"
FROZEN = 0
THRESH = 1e-6
ATOM = str(Path(__file__).resolve().with_name("water_dimer.xyz"))


def configure(*, use_custom_cphf: bool) -> None:
    config.update("pyscfad_moleintor_opt", True)
    config.update("pyscfad_scf_implicit_diff", True)
    config.update("pyscfad_scf_first_order_custom", use_custom_cphf)
    config.update("pyscfad_ccsd_implicit_diff", True)
    config.update("pyscfad_dfccsd_custom_response", True)


def make_mol():
    mol = gto.Mole(atom=ATOM, basis=BASIS)
    mol.verbose = 2
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def total_energy(mol, *, use_custom_cphf: bool):
    configure(use_custom_cphf=use_custom_cphf)

    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-10
    ehf = mf.kernel()

    mmp = mp.dfmp2.MP2(mf, frozen=FROZEN)
    mmp.kernel(with_t2=False)

    mycc = LNOCCSD(mf, thresh=THRESH, frozen=FROZEN)
    mycc.thresh_occ = THRESH
    mycc.thresh_vir = THRESH
    mycc.lo_type = "iao"
    mycc.no_type = "ie"
    mycc.ccsd_t = False
    mycc.kernel(frag_lolist=None)
    return ehf + mycc.e_corr_pt2corrected(mmp.e_corr)


def run_backend(*, use_custom_cphf: bool):
    label = "custom first-order CPHF" if use_custom_cphf else "implicit backprop"
    configure(use_custom_cphf=use_custom_cphf)
    mol = make_mol()

    t0 = time.perf_counter()
    e, g = jax.value_and_grad(
        lambda mm: total_energy(mm, use_custom_cphf=use_custom_cphf)
    )(mol)
    g = np.asarray(g.coords)
    e = float(e)
    elapsed = time.perf_counter() - t0
    return {
        "label": label,
        "energy": e,
        "grad": g,
        "elapsed_s": elapsed,
    }


def summarize_difference(ref, trial):
    grad_diff = trial["grad"] - ref["grad"]
    return {
        "energy_diff": trial["energy"] - ref["energy"],
        "grad_max_abs": float(np.max(np.abs(grad_diff))),
        "grad_rms_abs": float(np.sqrt(np.mean(grad_diff**2))),
        "speedup": ref["elapsed_s"] / max(trial["elapsed_s"], 1e-12),
    }


implicit = run_backend(use_custom_cphf=False)
custom = run_backend(use_custom_cphf=True)
diff = summarize_difference(implicit, custom)

print()
print("Comparing two SCF-gradient backends inside the same PT2-corrected LNO-CCSD calculation")
print("1. implicit SCF backprop")
print("2. custom first-order CPHF-style SCF response")
print("CCSD response backend held fixed: custom DF-CCSD response = True")
print("Note: the custom CPHF-style route is intended to be more efficient than implicit backprop for first-order properties.")
print()
print(f"Implicit energy: {implicit['energy']:.15f}")
print(f"Custom   energy: {custom['energy']:.15f}")
print(f"Energy difference (custom - implicit): {diff['energy_diff']:.6e}")
print()
print(f"Implicit time [s]: {implicit['elapsed_s']:.3f}")
print(f"Custom   time [s]: {custom['elapsed_s']:.3f}")
print(f"Approximate speedup (implicit/custom): {diff['speedup']:.3f}x")
print()
print(f"Max |gradient difference|: {diff['grad_max_abs']:.6e}")
print(f"RMS |gradient difference|: {diff['grad_rms_abs']:.6e}")
print()
if diff["grad_max_abs"] < 1e-5 and abs(diff["energy_diff"]) < 1e-6:
    print("Interpretation: the implicit and custom first-order CPHF results agree closely, so the custom route is reproducing the implicit gradient well on this example.")
else:
    print("Interpretation: the two backpropagation routes differ noticeably on this example and should be investigated further.")
print()
print("Implicit gradient:")
print(implicit["grad"])
print()
print("Custom first-order CPHF gradient:")
print(custom["grad"])
