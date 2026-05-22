"""PySCF DF-CCSD(T) gradient for the water cluster benchmark."""

import os
import time

import numpy as np

from pyscf import gto, lib, scf
from pyscf.cc import dfccsd
from pyscf.grad import ccsd_t as ccsd_t_grad
from pyscf.lib import numpy_helper


N_WATER = 6
BASIS = "dz"
MAX_MEMORY_MB = 60000
VERBOSE = 4
FROZEN = 0
EINSUM_BACKEND = None


def patch_pyscf_einsum():
    global EINSUM_BACKEND
    try:
        import opt_einsum

        stable_einsum = opt_einsum.contract
        EINSUM_BACKEND = "opt_einsum.contract"
    except ImportError:
        stable_einsum = np.einsum
        EINSUM_BACKEND = "numpy.einsum"

    # Some development PySCF checkouts have a lib.einsum wrapper that can
    # fail in the CCSD lambda equations for multi-operand contractions.
    lib.einsum = stable_einsum
    numpy_helper.einsum = stable_einsum


def water_cluster_atom(nwater=None):
    if nwater is None:
        nwater = N_WATER
    monomer = [
        ("O", (0.000000, 0.000000, 0.000000)),
        ("H", (0.758602, 0.000000, 0.504284)),
        ("H", (-0.758602, 0.000000, 0.504284)),
    ]
    atom = []
    for iw in range(nwater):
        shift = np.array((3.5 * iw, 0.0, 0.0))
        for symbol, xyz in monomer:
            atom.append((symbol, tuple(np.array(xyz) + shift)))
    return atom


def make_mol():
    mol = gto.Mole(
        atom=water_cluster_atom(),
        basis=BASIS,
        unit="Angstrom",
        verbose=VERBOSE,
        max_memory=MAX_MEMORY_MB,
    )
    mol.incore_anyway = True
    mol.build()
    return mol


def log(message=""):
    print(message, flush=True)


def timed(label, fn):
    start = time.perf_counter()
    out = fn()
    elapsed = time.perf_counter() - start
    log(f"{label} elapsed seconds: {elapsed:.3f}")
    return out, elapsed


def run_ccsd_t_gradient(mol):
    def run_scf():
        mf = scf.RHF(mol).density_fit()
        mf.max_memory = MAX_MEMORY_MB
        mf.conv_tol = 1e-10
        mf.max_cycle = 100
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("SCF did not converge")
        return mf

    mf, scf_time = timed("SCF", run_scf)

    def run_ccsd():
        mycc = dfccsd.RCCSD(mf, frozen=FROZEN)
        mycc.max_memory = MAX_MEMORY_MB
        mycc.conv_tol = 1e-8
        mycc.conv_tol_normt = 1e-7
        mycc.max_cycle = 100
        eris = mycc.ao2mo()
        mycc.kernel(eris=eris)
        if not mycc.converged:
            raise RuntimeError("DF-CCSD did not converge")
        return mycc, eris

    (mycc, eris), ccsd_time = timed("DF-CCSD", run_ccsd)

    et, triples_time = timed("CCSD(T) correction", lambda: mycc.ccsd_t(eris=eris))

    grad, grad_time = timed(
        "CCSD(T) gradient",
        lambda: ccsd_t_grad.Gradients(mycc).kernel(eris=eris),
    )

    e_tot = mycc.e_tot + et
    timings = {
        "scf": scf_time,
        "ccsd": ccsd_time,
        "triples_energy": triples_time,
        "gradient": grad_time,
        "total": scf_time + ccsd_time + triples_time + grad_time,
    }
    return e_tot, grad, timings


def main():
    patch_pyscf_einsum()
    mol = make_mol()
    nocc = mol.nelectron // 2 - int(FROZEN)
    nvir = mol.nao_nr() - mol.nelectron // 2

    log("PySCF water-cluster DF-CCSD(T) gradient")
    log(f"waters = {N_WATER}")
    log(f"basis = {BASIS}")
    log(f"frozen = {FROZEN}")
    log(f"orbitals = {nocc} occupied / {nvir} virtual")
    log(f"einsum backend = {EINSUM_BACKEND}")
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        log(f"{key} = {os.environ.get(key, 'unset')}")
    log()

    start = time.perf_counter()
    e_tot, grad, timings = run_ccsd_t_gradient(mol)
    elapsed = time.perf_counter() - start

    log()
    log(f"DF-CCSD(T) total energy: {float(e_tot):.15f}")
    log("DF-CCSD(T) nuclear gradient, Eh/Bohr:")
    print(grad, flush=True)
    log(f"Gradient norm: {np.linalg.norm(grad):.15e}")
    log(f"Phase-sum elapsed seconds: {timings['total']:.3f}")
    log(f"Total elapsed seconds: {elapsed:.3f}")


if __name__ == "__main__":
    main()
