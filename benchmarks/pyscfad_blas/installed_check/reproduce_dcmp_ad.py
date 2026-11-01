import sys
import time
import jax
import pyscf
import pyscfad
import pyscfadlib
from pyscfad import config, gto, scf
from pyscfad.mp import dfmp2
print(f'Python: {sys.executable}', flush=True)
print(f'PySCFAD {pyscfad.__version__}: {pyscfad.__file__}', flush=True)
print(f'PySCFADlib {pyscfadlib.__version__}: {pyscfadlib.__file__}', flush=True)
print(f'PySCF {pyscf.__version__}: {pyscf.__file__}', flush=True)
print(f'JAX {jax.__version__}; x64={jax.config.x64_enabled}; OpenMP threads={pyscf.lib.num_threads()}', flush=True)
MAX_MEMORY = 1000
FROZEN = 1
config.update('pyscfad_moleintor_opt', True)
config.update('pyscfad_scf_implicit_diff', True)
ATOM = 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587'
mol = gto.Mole()
mol.atom = ATOM
mol.basis = 'cc-pvdz'
mol.verbose = 4
mol.max_memory = MAX_MEMORY
mol.build(trace_exp=False, trace_ctr_coeff=False)

def scf_state(mol):
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
    (mf.conv_tol, mf.conv_tol_grad, mf.max_cycle) = (1e-10, 1e-08, 250)
    mf.kernel()
    return ((mf.mol, mf.with_df, mf.e_tot, mf.mo_coeff, mf.mo_energy), mf.converged)

def mp2_energy(state):
    (mol, with_df, ehf, mo_coeff, mo_energy) = state
    mf = scf.RHF(mol).density_fit(with_df=with_df)
    (mf.e_tot, mf.mo_coeff, mf.mo_energy) = (ehf, mo_coeff, mo_energy)
    mf.mo_occ = mf.get_occ()
    mf.converged = True
    mp = dfmp2.MP2(mf, frozen=FROZEN)
    mp.kernel(with_t2=False)
    return mp.e_tot
t0 = time.perf_counter()
start = time.perf_counter()
(state, scf_pullback, converged) = jax.vjp(scf_state, mol, has_aux=True)
jax.block_until_ready((state, scf_pullback))
scf_seconds = time.perf_counter() - start
if not converged:
    raise RuntimeError('SCF did not converge')
print(f'SCF forward (including DF build): {scf_seconds:.6f} s', flush=True)
start = time.perf_counter()
(e, mp2_pullback) = jax.vjp(mp2_energy, state)
jax.block_until_ready((e, mp2_pullback))
print(f'MP2 forward (excluding SCF): {time.perf_counter() - start:.6f} s', flush=True)
start = time.perf_counter()
(state_bar,) = mp2_pullback(jax.numpy.ones_like(e))
jax.block_until_ready(state_bar)
print(f'MP2 pullback: {time.perf_counter() - start:.6f} s', flush=True)
del mp2_pullback
start = time.perf_counter()
(jac,) = scf_pullback(state_bar)
jax.block_until_ready(jac)
print(f'SCF pullback (including DF pullback): {time.perf_counter() - start:.6f} s', flush=True)
grad = jac.coords
print(grad[:10])
total_seconds = time.perf_counter() - t0
print(f'Total energy + gradient wall time: {total_seconds:.6f} s')
print(total_seconds)
