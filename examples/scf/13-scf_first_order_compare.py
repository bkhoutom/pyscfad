"""Run an SCF first-order gradient with the custom CPHF-style SCF backprop."""

from __future__ import annotations

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


def configure() -> None:
    config.update("pyscfad_moleintor_opt", True)
    config.update("pyscfad_scf_implicit_diff", True)
    config.update("pyscfad_scf_first_order_custom", True)


def make_mol():
    mol = gto.Mole(atom=ATOM, basis=BASIS)
    mol.verbose = 2
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def energy(mol):
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-10
    return mf.kernel()


def run_backend():
    configure()
    mol = make_mol()

    def energy_backend(mm):
        configure()
        mf = scf.RHF(mm).density_fit()
        mf.conv_tol = 1e-10
        return mf.kernel()

    e, g = jax.value_and_grad(energy_backend)(mol)
    return float(e), np.asarray(g.coords)

e, g = run_backend()
print("SCF backend: cphf")
print(e)
print(g)
