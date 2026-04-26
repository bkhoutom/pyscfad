"""Second derivative check for the custom DF-CCSD response path.

The custom DF-CCSD gradient is differentiated once more to form a nuclear
Hessian-vector product.  The HVP is checked against a central finite
difference of custom-response gradients.
"""

from __future__ import annotations

import warnings

import jax
import numpy as onp

from pyscfad import config, gto, scf, numpy as np
from pyscfad.cc import dfccsd


warnings.filterwarnings(
    "ignore",
    message=r"Function mol\.dumps drops attribute .* because it is not JSON-serializable",
)


FD_STEP_BOHR = 1e-3
DIRECTION = np.asarray([[0.10, -0.03, 0.02], [-0.04, 0.05, -0.08]])
DIRECTION = DIRECTION / np.linalg.norm(DIRECTION)


def configure() -> None:
    # Higher-order tracing needs the fully JAX-visible in-core tensor path.
    config.update("pyscfad_moleintor_opt", False)
    config.update("pyscfad_scf_implicit_diff", True)
    config.update("pyscfad_scf_first_order_custom", True)
    config.update("pyscfad_ccsd_implicit_diff", True)
    config.update("pyscfad_dfccsd_custom_response", True)


def make_mol(coords_bohr=None):
    if coords_bohr is None:
        atom = "H 0 0 0; H 0 0 0.9"
        unit = "Angstrom"
    else:
        atom = [["H", tuple(coords_bohr[0])], ["H", tuple(coords_bohr[1])]]
        unit = "Bohr"

    mol = gto.Mole(
        atom=atom,
        basis="sto-3g",
        unit=unit,
        verbose=0,
    )
    mol.incore_anyway = True
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def energy(mol):
    configure()
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-10
    mf.kernel()

    mycc = dfccsd.RCCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-8
    mycc.kernel()
    return mycc.e_tot


def gradient(mol):
    return jax.grad(energy)(mol).coords


def grad_dot_direction(mol):
    return np.vdot(gradient(mol), DIRECTION)


def finite_difference_hvp(mol):
    coords = np.asarray(mol.coords)
    direction = np.asarray(DIRECTION)
    grad_plus = gradient(make_mol(coords + FD_STEP_BOHR * direction))
    grad_minus = gradient(make_mol(coords - FD_STEP_BOHR * direction))
    return (grad_plus - grad_minus) / (2 * FD_STEP_BOHR)


if __name__ == "__main__":
    configure()
    mol = make_mol()

    e_tot = energy(mol)
    grad = gradient(mol)
    hvp = jax.grad(grad_dot_direction)(mol).coords
    hvp_fd = finite_difference_hvp(mol)
    hvp_diff = hvp - hvp_fd

    print()
    print("Example: higher-order DF-CCSD derivative with custom CCSD response")
    print("System: H2/STO-3G")
    print("Quantity: nuclear Hessian-vector product")
    print(f"Energy: {float(e_tot):.15f}")
    print()
    print("Gradient:")
    print(onp.asarray(grad))
    print()
    print("AD Hessian-vector product:")
    print(onp.asarray(hvp))
    print()
    print("Finite-difference Hessian-vector product:")
    print(onp.asarray(hvp_fd))
    print()
    hvp_diff = onp.asarray(hvp_diff)
    print(f"Max |HVP difference|: {onp.max(onp.abs(hvp_diff)):.6e}")
    print(f"RMS |HVP difference|: {onp.sqrt(onp.mean(hvp_diff**2)):.6e}")

    if onp.max(onp.abs(hvp_diff)) > 1e-4:
        raise RuntimeError("Custom DF-CCSD higher-order derivative check failed.")
