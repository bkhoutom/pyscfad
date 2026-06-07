'''DLNO-CCSD(T) energy + nuclear gradient for water trimer / cc-pvdz (MPI).

Each rank owns its own CDERI file (cderi_trimer_{rank}.h5) and processes a
round-robin subset of LNO fragments.  Rank 0 runs the full SCF + LO +
prescreen setup; all ranks share the canonical MO state via broadcast.

Run:
    mpirun -np 3 python 15-mpi_dlno_ccsd_t_trimer.py
'''
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')

import numpy
from mpi4py import MPI
from pyscfad import config, gto, scf
from pyscfad.dlno.ccsd_mpi import DLNOCCSD

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

config.update('pyscfad_moleintor_opt', True)
config.update('pyscfad_scf_implicit_diff', True)
config.update('pyscfad_scf_first_order_custom', True)

mol = gto.Mole(atom='water_trimer.xyz', basis='ccpvdz', max_memory=4000)
mol.verbose = 4 if rank == 0 else 0
mol.build(trace_exp=False, trace_ctr_coeff=False)

CDERI_PATH = f'cderi_trimer_{rank}.h5'

# Each rank builds its own CDERI file eagerly before any JAX trace.
_df = scf.RHF(mol).density_fit().with_df
_df.max_memory = mol.max_memory
_df._cderi_to_save = CDERI_PATH
_df.build()
del _df


def build_mf(mol, *, mo_coeff_init=None, mo_energy_init=None,
             mo_occ_init=None, e_tot_init=None):
    mf = scf.RHF(mol).density_fit()
    mf.with_df.max_memory = mol.max_memory
    mf.with_df.attach_outcore_cderi(CDERI_PATH)
    if mo_coeff_init is None:
        # Rank 0: run canonical SCF under jax.vjp
        mf.kernel()
    else:
        # Non-root: adopt rank-0's canonical MO state; skip SCF
        mf.mo_coeff = mo_coeff_init
        mf.mo_energy = mo_energy_init
        mf.mo_occ = mo_occ_init
        mf.e_tot = e_tot_init
    return mf


thr = 1e-4

e_dlno, jac_dlno = DLNOCCSD.value_and_grad(
    mol,
    build_mf=build_mf,
    ccsd_t=True,
    domain_pao_thr=thr,
    pair_energy_thr=thr,
    thresh_occ=1e-3,
    thresh_vir=1e-4,
)

if rank == 0:
    e_dlno = float(e_dlno)
    g_dlno = numpy.asarray(jac_dlno.coords)
    print(f'DLNO-CCSD(T) energy = {e_dlno:.10f}')
    print('DLNO-CCSD(T) gradient (Hartree/Bohr):')
    print(g_dlno)
