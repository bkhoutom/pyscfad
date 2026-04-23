"""Compare two SCF-gradient backends on the same RHF energy and gradient.

This example evaluates a density-fitted RHF energy and nuclear gradient twice:

1. with the standard implicit SCF backpropagation route
2. with the custom first-order occupied-virtual CPHF-style SCF response

The two energies and gradients should agree very closely for first-order
properties.  The custom CPHF-style route is intended to be more efficient than
the general implicit SCF backpropagation path, so this example reports the
differences explicitly.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import jax
import numpy as np

from pyscfad import config, gto, scf


warnings.filterwarnings(
    "ignore",
    message=r"Function mol\.dumps drops attribute .* because it is not JSON-serializable",
)


BASIS = "def2-svp"
ATOM = str((Path(__file__).resolve().parents[1] / "lno" / "water_dimer.xyz").resolve())


def configure(*, use_custom_cphf: bool) -> None:
    config.update("pyscfad_moleintor_opt", True)
    config.update("pyscfad_scf_implicit_diff", True)
    config.update("pyscfad_scf_first_order_custom", use_custom_cphf)


def make_mol():
    mol = gto.Mole(atom=ATOM, basis=BASIS)
    mol.verbose = 2
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def energy(mol):
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-10
    return mf.kernel()


def run_backend(*, use_custom_cphf: bool):
    label = "custom first-order CPHF" if use_custom_cphf else "implicit backprop"
    configure(use_custom_cphf=use_custom_cphf)
    mol = make_mol()

    def energy_backend(mm):
        configure(use_custom_cphf=use_custom_cphf)
        mf = scf.RHF(mm).density_fit()
        mf.conv_tol = 1e-10
        return mf.kernel()

    t0 = time.perf_counter()
    e, g = jax.value_and_grad(energy_backend)(mol)
    elapsed = time.perf_counter() - t0
    return {
        "label": label,
        "energy": float(e),
        "grad": np.asarray(g.coords),
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
print("Example: RHF energy and gradient with two SCF backpropagation routes")
print("What is being evaluated: density-fitted RHF on the water dimer")
print("1. implicit SCF backprop")
print("2. custom first-order occupied-virtual CPHF-style SCF response")
print("Note: the custom CPHF-style route is intended to be more efficient than a general implicit SCF backward pass for first-order properties.")
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
    print("Interpretation: the implicit and custom first-order CPHF results agree closely, so the custom route is reproducing the implicit SCF gradient well on this example.")
else:
    print("Interpretation: the two SCF backpropagation routes differ noticeably on this example and should be investigated further.")
print()
print("Implicit gradient:")
print(implicit["grad"])
print()
print("Custom first-order CPHF gradient:")
print(custom["grad"])
