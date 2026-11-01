import sys
import time
import pyscf
from pyscf import gto, scf
from pyscf.mp import dfmp2
print(f'Python: {sys.executable}', flush=True)
print(f'PySCF {pyscf.__version__}: {pyscf.__file__}', flush=True)
print(f'OpenMP threads={pyscf.lib.num_threads()}', flush=True)
MAX_MEMORY = 1000
FROZEN = 1
ATOM = 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587'
mol = gto.Mole()
mol.atom = ATOM
mol.basis = 'cc-pvdz'
mol.verbose = 4
mol.max_memory = MAX_MEMORY
mol.build()

def energy(mol):
    start = time.perf_counter()
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
    (mf.conv_tol, mf.conv_tol_grad, mf.max_cycle) = (1e-10, 1e-08, 250)
    mf.kernel()
    scf_seconds = time.perf_counter() - start
    if not mf.converged:
        raise RuntimeError('SCF did not converge')
    print(f'SCF forward (including DF build): {scf_seconds:.6f} s', flush=True)
    start = time.perf_counter()
    mp = dfmp2.MP2(mf, frozen=FROZEN)
    mp.kernel(with_t2=False)
    print(f'MP2 forward (excluding SCF): {time.perf_counter() - start:.6f} s', flush=True)
    return mp.e_tot
t0 = time.perf_counter()
e = energy(mol)
total_seconds = time.perf_counter() - t0
print(f'Total energy wall time: {total_seconds:.6f} s')
print(total_seconds)
