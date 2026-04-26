"""Compare implicit and custom-response backpropagation for DF-CCSD.

This example evaluates the water-dimer DF-CCSD energy and nuclear gradient
twice in the def2-SVP basis:

1. with the general implicit CCSD differentiation route
2. with the custom DF-CCSD lambda-response backward pass

The SCF response backend is kept fixed between the two calculations, so the
reported differences isolate the CCSD response implementation.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import jax
import numpy as np

from pyscfad import config, gto, scf
from pyscfad.cc import dfccsd


warnings.filterwarnings(
    "ignore",
    message=r"Function mol\.dumps drops attribute .* because it is not JSON-serializable",
)


BASIS = "def2-svp"
ATOM = str((Path(__file__).resolve().parents[1] / "lno" / "water_dimer.xyz").resolve())
USE_CUSTOM_SCF_RESPONSE = True


def configure(*, use_custom_ccsd_response: bool) -> None:
    config.update("pyscfad_moleintor_opt", True)
    config.update("pyscfad_scf_implicit_diff", True)
    config.update("pyscfad_scf_first_order_custom", USE_CUSTOM_SCF_RESPONSE)
    config.update("pyscfad_ccsd_implicit_diff", True)
    config.update("pyscfad_dfccsd_custom_response", use_custom_ccsd_response)


def make_mol():
    mol = gto.Mole(atom=ATOM, basis=BASIS)
    mol.verbose = 2
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def run_backend(*, use_custom_ccsd_response: bool):
    label = (
        "custom DF-CCSD lambda response"
        if use_custom_ccsd_response
        else "general implicit CCSD differentiation"
    )
    configure(use_custom_ccsd_response=use_custom_ccsd_response)
    mol = make_mol()

    def energy_backend(mm):
        configure(use_custom_ccsd_response=use_custom_ccsd_response)
        mf = scf.RHF(mm).density_fit()
        mf.conv_tol = 1e-10
        mf.kernel()
        mycc = dfccsd.RCCSD(mf)
        mycc.conv_tol = 1e-7
        mycc.conv_tol_normt = 1e-5
        mycc.kernel()
        return mycc.e_tot

    t0 = time.perf_counter()
    e_tot, grad = jax.value_and_grad(energy_backend)(mol)
    elapsed = time.perf_counter() - t0
    return {
        "label": label,
        "energy": float(e_tot),
        "grad": np.asarray(grad.coords),
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


implicit = run_backend(use_custom_ccsd_response=False)
custom = run_backend(use_custom_ccsd_response=True)
diff = summarize_difference(implicit, custom)

print()
print("Example: water-dimer DF-CCSD gradient with two CCSD backpropagation routes")
print(f"Basis: {BASIS}")
print(f"SCF response backend held fixed: custom first-order CPHF = {USE_CUSTOM_SCF_RESPONSE}")
print("1. general implicit CCSD differentiation")
print("2. custom DF-CCSD lambda-response backward pass")
print()
print(f"Implicit CCSD energy: {implicit['energy']:.15f}")
print(f"Custom   CCSD energy: {custom['energy']:.15f}")
print(f"Energy difference (custom - implicit): {diff['energy_diff']:.6e}")
print()
print(f"Implicit CCSD time [s]: {implicit['elapsed_s']:.3f}")
print(f"Custom   CCSD time [s]: {custom['elapsed_s']:.3f}")
print(f"Approximate speedup (implicit/custom): {diff['speedup']:.3f}x")
print()
print(f"Max |gradient difference|: {diff['grad_max_abs']:.6e}")
print(f"RMS |gradient difference|: {diff['grad_rms_abs']:.6e}")
print()
if diff["grad_max_abs"] < 1e-5 and abs(diff["energy_diff"]) < 1e-6:
    print("Interpretation: the custom DF-CCSD response reproduces the implicit CCSD gradient closely on this example.")
else:
    print("Interpretation: the two CCSD backpropagation routes differ noticeably on this example and should be investigated further.")
print()
print("Implicit CCSD gradient:")
print(implicit["grad"])
print()
print("Custom DF-CCSD response gradient:")
print(custom["grad"])
