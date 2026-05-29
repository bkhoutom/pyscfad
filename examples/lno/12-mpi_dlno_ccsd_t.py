'''DLNO-CCSD(T) gradient for water trimer with MPI parallelization.

run with:
mpirun -n 2 python 12-mpi_dlno_ccsd_t.py
'''
from mpi4py import MPI
import warnings
import jax
import numpy
from pyscfad import gto, scf, mp
from pyscfad import config
from pyscfad.lno.ccsd_mpi import LNOCCSD
from pyscfad.lno.prescreen import (
    build_dlno_prescreen_data,
    rebuild_dlno_prescreen_data,
)
from pyscfad.lno.tools import autofrag, map_lo_to_frag
from pyscfad.ops import stop_trace

config.update('pyscfad_moleintor_opt', True)
config.update('pyscfad_scf_implicit_diff', True)
config.update('pyscfad_scf_first_order_custom', True)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank != 0:
    warnings.filterwarnings('ignore',
                            message='Function mol.dumps drops attribute coords.*')

mol = gto.Mole(atom='water_trimer.xyz', basis='ccpvdz')
mol.verbose = 4 if rank == 0 else 0
mol.build(trace_exp=False, trace_ctr_coeff=False)

frozen = 3
thresh_occ = 1e-5
thresh_vir = 1e-6
domain_thr = 1e-4


def _make_lnoccsd(mf):
    cc = LNOCCSD(mf, frozen=frozen)
    cc.thresh_occ = thresh_occ
    cc.thresh_vir = thresh_vir
    cc.lo_type = 'iao'
    cc.ccsd_t = True
    return cc


# Build the DLNO topology once on every rank using a concrete (untraced) SCF
# so we can pass it as static metadata into the gradient pass below.
_mf0 = scf.RHF(mol).density_fit()
_mf0.kernel()
lo_coeff0 = _make_lnoccsd(_mf0).get_lo(lo_type='iao')
frag_lolist = stop_trace(map_lo_to_frag)(
    mol, lo_coeff0, stop_trace(autofrag)(mol), verbose=mol.verbose,
)
topology = build_dlno_prescreen_data(
    _mf0, lo_coeff0, frag_lolist, frozen=frozen,
    lmo_bp_domain_thr=0.9, pao_bp_domain_thr=0.9,
    domain_pao_thr=domain_thr, pair_energy_thr=domain_thr,
    multipole_order=2,
)


def energy(mol):
    mf = scf.RHF(mol).density_fit()
    ehf = mf.kernel()
    cc = _make_lnoccsd(mf)
    cc.profile_fragments = True

    lo_coeff = cc.get_lo(lo_type='iao')
    cc.use_dlno_prescreen = True
    cc.dlno_prescreen_data = rebuild_dlno_prescreen_data(
        mf, lo_coeff, topology, frozen=frozen,
    )
    cc.kernel(frag_lolist=frag_lolist, orbloc=lo_coeff)

    if rank == 0:
        mmp = mp.dfmp2.MP2(mf, frozen=frozen)
        mmp.max_memory = 32000
        mmp.kernel(with_t2=False)
        return ehf + cc.e_corr_pt2corrected(mmp.e_corr)
    return cc.e_corr - cc.e_corr_pt2


e, jac = jax.value_and_grad(energy)(mol)
e = numpy.asarray(e)
grad = numpy.asarray(jac.coords)

etot = numpy.zeros_like(e) if rank == 0 else None
grad_tot = numpy.zeros_like(grad) if rank == 0 else None
comm.Reduce([e, MPI.DOUBLE], [etot, MPI.DOUBLE], op=MPI.SUM, root=0)
comm.Reduce([grad, MPI.DOUBLE], [grad_tot, MPI.DOUBLE], op=MPI.SUM, root=0)

if rank == 0:
    print(f'DLNO-CCSD(T) energy: {etot}')
    print(f'DLNO-CCSD(T) gradient:\n{grad_tot}')
