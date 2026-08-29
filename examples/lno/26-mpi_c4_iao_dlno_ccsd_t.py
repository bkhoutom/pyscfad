"""MPI IAO-DLNO-CCSD(T) energy and gradient for C4H10.

Run from the repository root with, for example,

    mpirun -np 4 .venv/bin/python -u examples/lno/26-mpi_c4_iao_dlno_ccsd_t.py
"""

from pathlib import Path

import numpy
from mpi4py import MPI

from pyscfad import config, gto, scf
from pyscfad.df.mpi_outcore import build_cderi
from pyscfad.dlno.ccsd_mpi import DLNOCCSD
from pyscfad.dlno.iao_mp2 import IAOFragmentMP2Thresholds


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
here = Path(__file__).resolve().parent
output = here.parents[1] / "output"
output.mkdir(parents=True, exist_ok=True)
cderi = output / "c4_ccpvdz_cderi.h5"

config.update("pyscfad_moleintor_opt", True)
config.update("pyscfad_ccsd_implicit_diff", True)

mol = gto.Mole(
    atom=str(here / "c4h10.xyz"),
    basis="cc-pvdz",
    max_memory=4000,
    verbose=4 if rank == 0 else 0,
)
mol.build(trace_exp=False, trace_ctr_coeff=False)

build_cderi(
    mol,
    cderi,
    auxbasis="cc-pvdz-ri",
    comm=comm,
    overwrite=True,
    progress=True,
)


def build_mf(mol_, *, mo_coeff_init=None, mo_energy_init=None,
             mo_occ_init=None, e_tot_init=None):
    mf = scf.RHF(mol_).density_fit(auxbasis="cc-pvdz-ri")
    mf.with_df.attach_outcore_cderi(str(cderi))
    if mo_coeff_init is None:
        mf.kernel()
    else:
        mf.mo_coeff = mo_coeff_init
        mf.mo_energy = mo_energy_init
        mf.mo_occ = mo_occ_init
        mf.e_tot = e_tot_init
        mf.converged = True
    return mf


energy, gradient = DLNOCCSD.value_and_grad(
    mol,
    build_mf=build_mf,
    frozen=4,
    thresholds=IAOFragmentMP2Thresholds(pair_energy=1e-4),
    thresh_occ=1e-3,
    thresh_vir=1e-4,
    ccsd_t=True,
    parallel_scf_jk=True,
    comm=comm,
    progress=True,
)

if rank == 0:
    print(f"\nIAO-DLNO-CCSD(T) energy = {float(energy):.12f} Eh")
    print("Gradient (Eh/bohr):")
    print(numpy.asarray(gradient.coords))
