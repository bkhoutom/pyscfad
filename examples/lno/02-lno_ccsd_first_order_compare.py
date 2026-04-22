"""Run an LNO-CCSD first-order gradient with custom SCF backprop."""

from __future__ import annotations

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


def configure() -> None:
    config.update("pyscfad_moleintor_opt", True)
    config.update("pyscfad_scf_implicit_diff", True)
    config.update("pyscfad_scf_first_order_custom", True)
    config.update("pyscfad_ccsd_implicit_diff", True)


def make_mol():
    mol = gto.Mole(atom=ATOM, basis=BASIS)
    mol.verbose = 2
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def energy(mol):
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


def run_backend():
    configure()
    mol = make_mol()

    def energy_backend(mm):
        configure()
        mf = scf.RHF(mm).density_fit()
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

    e, g = jax.value_and_grad(energy_backend)(mol)
    return float(e), np.asarray(g.coords)

e, g = run_backend()
print("SCF backend: cphf")
print(e)
print(g)
