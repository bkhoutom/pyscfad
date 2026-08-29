"""(H2O)24/cc-pVDZ IAO-local-MP2 energy and gradient with MPI.

Run with 1--4 MPI ranks, for example

    mpirun -np 2 .venv/bin/python -u examples/lno/water.py
"""

from pathlib import Path

import numpy
from mpi4py import MPI

from pyscfad import config, gto, scf
from pyscfad.df.mpi_outcore import build_cderi
from pyscfad.dlno.iao_mp2 import IAOFragmentMP2Thresholds
from pyscfad.dlno.iao_mp2_mpi import IAOFragmentMP2


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

BASIS = "cc-pvdz"
AUXBASIS = "cc-pvdz-ri"
HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parents[1] / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)
CDERI = str(OUTPUT / "h2o24_ccpvdz_cderi.h5")

config.update("pyscfad_moleintor_opt", True)

mol = gto.Mole(
    atom=str(HERE / "water_24.xyz"),
    basis=BASIS,
    max_memory=2000,
    verbose=4 if rank == 0 else 0,
)
mol.build(trace_exp=False, trace_ctr_coeff=False)

build_cderi(
    mol,
    CDERI,
    auxbasis=AUXBASIS,
    comm=comm,
    max_memory=mol.max_memory,
    overwrite=True,
    progress=True,
)


def build_mf(mol_, *, mo_coeff_init=None, mo_energy_init=None,
             mo_occ_init=None, e_tot_init=None):
    mf = scf.RHF(mol_).density_fit(auxbasis=AUXBASIS)
    mf.with_df.attach_outcore_cderi(CDERI)
    if mo_coeff_init is None:
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("DF-RHF did not converge")
    else:
        mf.mo_coeff = mo_coeff_init
        mf.mo_energy = mo_energy_init
        mf.mo_occ = mo_occ_init
        mf.e_tot = e_tot_init
        mf.converged = True
    return mf


energy, mol_bar = IAOFragmentMP2.value_and_grad(
    mol,
    build_mf=build_mf,
    frozen=24,
    thresholds=IAOFragmentMP2Thresholds(pair_energy=1e-4),
    pair_energy_model="multipole",
    include_hf=True,
    comm=comm,
    parallel_scf_jk=True,
    progress=True,
)

if rank == 0:
    print(f"Total energy = {float(energy):+.12f} Eh")
    print("Gradient (Eh/Bohr):")
    print(numpy.asarray(mol_bar.coords))
