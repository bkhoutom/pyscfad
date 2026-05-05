'''LNO-CCSD(T) with MPI parallelization.

run with:
mpirun -n 2 python 11-mpi_lno_ccsd_t.py
'''
from mpi4py import MPI
import jax
import numpy
from pyscfad import gto, scf, mp
from pyscfad.cc import dfccsd
from pyscfad import config
from pyscfad.lno.ccsd_mpi import LNOCCSD

config.update('pyscfad_moleintor_opt', True)
config.update('pyscfad_scf_implicit_diff', True)
config.update('pyscfad_ccsd_implicit_diff', True)

atom = 'water_dimer.xyz'
basis = 'ccpvdz'

mol = gto.Mole(atom=atom, basis=basis)
mol.verbose = 4
mol.build(trace_exp=False, trace_ctr_coeff=False)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
frozen = 2
thresh_occ = 1e-3
thresh_vir = 1e-4


def build_system_metadata(mol, mfcc, orbloc, ehf):
    nelec = tuple(int(x) for x in mol.nelec)
    nelectron = int(sum(nelec))
    nao = int(mol.nao_nr())
    nmo_total = int(len(mfcc.mo_occ))
    nocc_total = int(numpy.count_nonzero(numpy.asarray(mfcc.mo_occ) > 0))
    nvir_total = nmo_total - nocc_total
    return {
        'atom': atom,
        'basis': basis,
        'nproc': size,
        'natm': int(mol.natm),
        'nelectron': nelectron,
        'nelec': nelec,
        'charge': int(mol.charge),
        'spin': int(mol.spin),
        'nao': nao,
        'nmo': nmo_total,
        'nocc_total': nocc_total,
        'nvir_total': nvir_total,
        'frozen': frozen,
        'lo_type': mfcc.lo_type,
        'nlo': int(orbloc.shape[1]),
        'thresh_occ': mfcc.thresh_occ,
        'thresh_vir': mfcc.thresh_vir,
        'ccsd_t': mfcc.ccsd_t,
        'ehf': ehf,
    }


def print_system_metadata(metadata):
    print('LNO-CCSD(T) MPI example')
    print(f"Molecule: {metadata['atom']}")
    print(f"Basis: {metadata['basis']}")
    print(f"MPI processes: {metadata['nproc']}")
    print(
        'System: '
        f"natm={metadata['natm']}, "
        f"nelectron={metadata['nelectron']} "
        f"(alpha={metadata['nelec'][0]}, beta={metadata['nelec'][1]}), "
        f"charge={metadata['charge']}, spin={metadata['spin']}"
    )
    print(
        'Orbital space: '
        f"nao={metadata['nao']}, nmo={metadata['nmo']}, "
        f"nocc={metadata['nocc_total']}, nvir={metadata['nvir_total']}"
    )
    print(f"Frozen orbitals from input: {metadata['frozen']}")
    print(
        'LNO setup: '
        f"lo_type={metadata['lo_type']}, nlo={metadata['nlo']}, "
        f"thresh_occ={metadata['thresh_occ']}, "
        f"thresh_vir={metadata['thresh_vir']}, "
        f"ccsd_t={metadata['ccsd_t']}"
    )
    print(f"RHF energy: {numpy.asarray(metadata['ehf'])}")


def energy(mol):
    mf = scf.RHF(mol).density_fit()
    ehf = mf.kernel()

    mfcc = LNOCCSD(mf, frozen=frozen)
    mfcc.thresh_occ = thresh_occ
    mfcc.thresh_vir = thresh_vir
    mfcc.lo_type = 'iao'
    mfcc.ccsd_t = True
    orbloc = mfcc.get_lo()
    mfcc.kernel(frag_lolist=None, orbloc=orbloc)
    metadata = build_system_metadata(mol, mfcc, orbloc, ehf)

    if rank == 0:
        mmp = mp.dfmp2.MP2(mf, frozen=frozen)
        mmp.kernel(with_t2=False)
        ecc_pt2corrected = mfcc.e_corr_pt2corrected(mmp.e_corr)
        etot = ehf + ecc_pt2corrected
    else:
        etot = mfcc.e_corr - mfcc.e_corr_pt2
    return etot, metadata

(e, metadata), jac = jax.value_and_grad(energy, has_aux=True)(mol)
e = numpy.asarray(e)
grad = numpy.asarray(jac.coords)

if rank == 0:
    etot = numpy.zeros_like(e)
    grad_tot = numpy.zeros_like(grad)
else:
    etot = None
    grad_tot = None

comm.Reduce([e, MPI.DOUBLE], [etot, MPI.DOUBLE],
            op=MPI.SUM, root=0)

comm.Reduce([grad, MPI.DOUBLE], [grad_tot, MPI.DOUBLE],
            op=MPI.SUM, root=0)

if rank == 0:
    print_system_metadata(metadata)
    print(f'LNO-CCSD(T) energy: {etot}')
    print(f'LNO-CCSD(T) gradient:\n{grad_tot}')
