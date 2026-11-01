import jax
import time
from pyscfad import config, gto, scf
from pyscfad.mp import dfmp2

MAX_MEMORY = 60000  # MB, advisory budget
FROZEN = 24

config.update('pyscfad_moleintor_opt', True)
config.update('pyscfad_scf_implicit_diff', True)

ATOM = """
P   -0.7861   -2.9882   -3.3568
O    0.3621   -0.4750   -0.0755
O   -0.6006   -2.9501    2.3130
O   -0.2284   -2.5042   -2.0976
O   -1.4907    1.6935    0.6571
O    0.1337   -2.7862   -4.4471
O   -2.0169   -2.2960   -3.6547
O   -1.0677   -4.4001   -3.2784
N    0.7577    1.5581    1.0208
N   -0.3824    3.5106    0.5641
N    0.6130    5.4794    0.3914
C   -0.6870   -1.7330    1.5756
C    0.0339   -1.8383    0.2125
C   -0.0291   -0.5231    2.2697
C    0.8080    0.1401    1.1447
C   -0.8302   -2.4810   -0.8985
C    1.8821    2.2875    1.1077
C   -0.3874    2.2126    0.7600
C    1.8936    3.6200    0.8982
C    0.6979    4.2198    0.6148
H   -1.7498   -1.5163    1.4273
H    0.9614   -2.4165    0.3100
H    0.6059   -0.8358    3.1032
H   -0.7921    0.1400    2.6852
H    1.8361   -0.2234    1.2890
H   -1.0695   -3.5012   -0.5805
H   -1.7655   -1.9154   -0.9635
H   -1.1504   -2.8067    3.1115
H    2.7592    1.8678    1.3230
H    2.7406    4.1403    0.9520
H   -0.2224    5.8834    0.2712
H    1.3879    6.0016    0.3430
H    0.3417   -1.9730   -4.7110
H   -2.5588   -2.5912   -4.2812
"""

mol = gto.Mole()
mol.atom = ATOM
mol.basis = 'cc-pvtz'
mol.verbose = 4
mol.max_memory = MAX_MEMORY
mol.build(trace_exp=False, trace_ctr_coeff=False)

def scf_state(mol):
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvtz-jkfit')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-10, 1e-8, 250
    mf.kernel()
    # e_tot is static in some SCF pytrees; expose it explicitly for the HF gradient.
    return (mf.mol, mf.with_df, mf.e_tot, mf.mo_coeff, mf.mo_energy), mf.converged


def mp2_energy(state):
    mol, with_df, ehf, mo_coeff, mo_energy = state
    mf = scf.RHF(mol).density_fit(with_df=with_df)
    mf.e_tot, mf.mo_coeff, mf.mo_energy = ehf, mo_coeff, mo_energy
    mf.mo_occ = mf.get_occ()
    mf.converged = True
    mp = dfmp2.MP2(mf, frozen=FROZEN)
    mp.kernel(with_t2=False)
    return mp.e_tot


t0 = time.perf_counter()
start = time.perf_counter()
state, scf_pullback, converged = jax.vjp(scf_state, mol, has_aux=True)
jax.block_until_ready((state, scf_pullback))
scf_seconds = time.perf_counter() - start
if not converged:
    raise RuntimeError('SCF did not converge')
print(f'SCF forward (including DF build): {scf_seconds:.6f} s', flush=True)

start = time.perf_counter()
e, mp2_pullback = jax.vjp(mp2_energy, state)
jax.block_until_ready((e, mp2_pullback))
print(f'MP2 forward (excluding SCF): {time.perf_counter() - start:.6f} s', flush=True)

start = time.perf_counter()
state_bar, = mp2_pullback(jax.numpy.ones_like(e))
jax.block_until_ready(state_bar)
print(f'MP2 pullback: {time.perf_counter() - start:.6f} s', flush=True)
del mp2_pullback  # Release the spent MP2 residuals before the SCF reverse pass.

start = time.perf_counter()
# Include all molecular and DF cotangents from MP2, not only orbital response.
jac, = scf_pullback(state_bar)
jax.block_until_ready(jac)
print(f'SCF pullback (including DF pullback): {time.perf_counter() - start:.6f} s',
      flush=True)

grad = jac.coords
print(grad[:10])
total_seconds = time.perf_counter() - t0
print(f'Total energy + gradient wall time: {total_seconds:.6f} s')
print(total_seconds)
