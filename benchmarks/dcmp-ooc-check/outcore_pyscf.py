import os
import sys
import time
import pyscf
from pyscf import gto, scf
from pyscf.mp import dfmp2
from threadpoolctl import ThreadpoolController

print(f'Python: {sys.executable}', flush=True)
print(f'PySCF {pyscf.__version__}: {pyscf.__file__}', flush=True)
print(f'OpenMP threads={pyscf.lib.num_threads()}', flush=True)

MAX_MEMORY = 4000  # MB, advisory budget
FROZEN = 2
CDERI_PATH = os.path.join(os.environ['PYSCF_TMPDIR'], 'dcmp-cderi.h5')
SCIPY_DF_THREADS = int(os.environ.get('SCIPY_DF_THREADS', pyscf.lib.num_threads()))
if SCIPY_DF_THREADS < 1:
    raise ValueError('SCIPY_DF_THREADS must be positive')
controller = ThreadpoolController()
scipy_paths = [pool['filepath'] for pool in controller.info()
               if pool['user_api'] == 'blas' and '/scipy.libs/' in pool['filepath']]
if len(scipy_paths) != 1:
    raise RuntimeError(f'Expected one SciPy BLAS library, found {scipy_paths}')
scipy_blas = controller.select(filepath=scipy_paths)
print(f'SciPy DF threads={SCIPY_DF_THREADS}; {scipy_paths[0]}', flush=True)

ATOM = 'input.xyz'

mol = gto.Mole()
mol.atom = ATOM
mol.basis = 'def2-svp'
mol.verbose = 4
mol.max_memory = MAX_MEMORY
mol.build()

def energy(mol):
    start = time.perf_counter()
    mf = scf.RHF(mol).density_fit(auxbasis='weigend')
    # A string destination forces PySCF to build and read AO DF factors on disk.
    mf.with_df._cderi_to_save = CDERI_PATH
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-10, 1e-8, 250
    # This large SciPy solve runs outside the native OpenMP kernels.
    # Restore single-threaded BLAS before SCF and MP2. The timer includes DF.
    df_start = time.perf_counter()
    with scipy_blas.limit(limits=SCIPY_DF_THREADS):
        mf.with_df.build()
    assert isinstance(mf.with_df._cderi, str)
    print(f'Out-of-core AO DF: {mf.with_df._cderi}; '
          f'{os.path.getsize(CDERI_PATH) / 2**30:.3f} GiB', flush=True)
    df_seconds = time.perf_counter() - df_start
    print(f'DF forward: {df_seconds:.6f} s', flush=True)
    mf.kernel()
    scf_seconds = time.perf_counter() - start
    if not mf.converged:
        raise RuntimeError('SCF did not converge')
    print(f'SCF forward (excluding DF build): {scf_seconds - df_seconds:.6f} s', flush=True)
    print(f'SCF forward (including DF build): {scf_seconds:.6f} s', flush=True)

    start = time.perf_counter()
    mp = dfmp2.MP2(mf, frozen=FROZEN)
    mp.kernel(with_t2=False)
    print(f'MP2 forward (excluding SCF): {time.perf_counter() - start:.6f} s',
          flush=True)
    return mp.e_tot

t0 = time.perf_counter()
e = energy(mol)
total_seconds = time.perf_counter() - t0
print(f'Total energy wall time: {total_seconds:.6f} s')
print(total_seconds)

# Save the complete result for the independent numerical comparison.
import json
from pathlib import Path
payload = {'energy': float(e)}
Path('pyscf.json').write_text(json.dumps(payload, indent=2) + '\n')
