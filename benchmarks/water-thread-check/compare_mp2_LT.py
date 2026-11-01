"""Stochastic full-system LT-MP2 energy/gradient benchmark for (H2O)2."""
import json
import os
from pathlib import Path
import socket
import sys
from time import perf_counter

import jax
from jaxlib.cpu import _lapack
import numpy as np
import pyscf
import pyscfad
import pyscfadlib
from pyscfad import config, dlno_stc, gto, scf
from pyscfad.dlno_stc import _stc_mp2
from threadpoolctl import ThreadpoolController

ATOM = 'water_2.xyz'
BASIS = 'ccpvtz'
FROZEN = None
MAX_MEMORY = 120000  # MB; same advisory budget as the supplied driver.
CONTROLS = {'mode': 'stochastic', 'energy_tolerance': 3e-4, 'global_seed': 71}

NTHREADS = int(os.environ.get('SLURM_CPUS_PER_TASK',
                             os.environ.get('OMP_NUM_THREADS', '1')))
DF_THREADS = int(os.environ.get('SCIPY_DF_THREADS', str(NTHREADS)))
if not 1 <= DF_THREADS <= NTHREADS:
    raise ValueError('Require 1 <= SCIPY_DF_THREADS <= allocated CPU count')
pyscf.lib.num_threads(NTHREADS)
config.update('pyscfad_moleintor_opt', True)
config.update('pyscfad_scf_implicit_diff', True)
config.update('pyscfad_scf_df_precond', True)

# Discover the SciPy library used by JAX CPU LAPACK before creating a controller.
_lapack.initialize()
controller = ThreadpoolController()
scipy_paths = [p['filepath'] for p in controller.info()
               if p['user_api'] == 'blas' and '/scipy.libs/' in p['filepath']]
if len(scipy_paths) != 1:
    raise RuntimeError(f'Expected one SciPy BLAS library: {scipy_paths}')
scipy_blas = controller.select(filepath=scipy_paths)
phase_seconds = {}


def check_pools(scipy_threads):
    """Check the actual limits, including the STC extension's OpenBLAS."""
    for pool in controller.info():
        if pool['user_api'] == 'blas':
            expected = scipy_threads if pool['filepath'] in scipy_paths else 1
            if pool['num_threads'] != expected:
                raise RuntimeError(f'Unexpected BLAS thread count: {pool}')


def _build_df(mol):
    # Preserve density_fit's automatic auxiliary-basis choice from the input.
    with_df = scf.RHF(mol).density_fit().with_df
    with_df.build()
    return with_df


@jax.custom_vjp
def build_df(mol):
    with scipy_blas.limit(limits=DF_THREADS):
        result = _build_df(mol)
        jax.block_until_ready(result)
    return result


def _df_forward(mol):
    # Keep DF on the AD tape; save its real pullback, not detached factors.
    with scipy_blas.limit(limits=DF_THREADS):
        check_pools(DF_THREADS)
        start = perf_counter()
        result, pullback = jax.vjp(_build_df, mol)
        jax.block_until_ready((result, pullback))
        phase_seconds['df_forward'] = perf_counter() - start
    print(f"DF forward ({DF_THREADS} SciPy threads): "
          f"{phase_seconds['df_forward']:.6f} s", flush=True)
    return result, pullback


def _df_reverse(pullback, cotangent):
    with scipy_blas.limit(limits=DF_THREADS):
        check_pools(DF_THREADS)
        start = perf_counter()
        result = pullback(cotangent)
        jax.block_until_ready(result)
        phase_seconds['df_reverse'] = perf_counter() - start
    print(f"DF reverse ({DF_THREADS} SciPy threads): "
          f"{phase_seconds['df_reverse']:.6f} s", flush=True)
    return result


build_df.defvjp(_df_forward, _df_reverse)


def build_mf(mol):
    with_df = build_df(mol)
    check_pools(1)
    mf = scf.RHF(mol).density_fit(with_df=with_df)
    # Preserve the original driver's SCF convergence settings.
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('SCF did not converge')
    return mf


def main():
    print(f'Python: {sys.executable}', flush=True)
    print(f'PySCF {pyscf.__version__}: {pyscf.__file__}', flush=True)
    print(f'PySCFAD {pyscfad.__version__}: {pyscfad.__file__}', flush=True)
    print(f'PySCFADlib {pyscfadlib.__version__}: {pyscfadlib.__file__}', flush=True)
    print(f'STC native library: {_stc_mp2.__file__}', flush=True)
    print(f'Node: {socket.gethostname()}; CPU affinity: '
          f'{sorted(os.sched_getaffinity(0))}', flush=True)
    print(f'JAX {jax.__version__}; x64={jax.config.x64_enabled}; '
          f'OpenMP={pyscf.lib.num_threads()}; SciPy DF={DF_THREADS}', flush=True)
    print(f'Input={ATOM}; basis={BASIS}; frozen={FROZEN}; controls={CONTROLS}',
          flush=True)
    print('Initial thread pools:', json.dumps(controller.info()), flush=True)
    check_pools(1)
    if not jax.config.x64_enabled:
        raise RuntimeError('This benchmark requires JAX_ENABLE_X64=True')

    mol = gto.Mole(atom=ATOM, basis=BASIS, verbose=4, max_memory=MAX_MEMORY)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    phase_seconds.clear()
    start = perf_counter()
    energy, gradient = dlno_stc.value_and_grad(
        mol, build_mf, scope='system', frozen=FROZEN,
        controls=CONTROLS, include_hf=True,
    )
    jax.block_until_ready((energy, gradient))
    total_seconds = perf_counter() - start
    check_pools(1)
    coordinates = np.asarray(gradient.coords)
    if not np.isfinite(float(energy)) or not np.isfinite(coordinates).all():
        raise RuntimeError('Nonfinite energy or gradient')
    print(f'SCF + stochastic LT-MP2 energy + gradient: {total_seconds:.6f} s')
    print('Stochastic LT-MP2 total energy:', energy)
    print('Stochastic LT-MP2 nuclear gradient (Eh/bohr):')
    print(np.array2string(coordinates, precision=12, threshold=coordinates.size))
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    result = {
        'job_id': job_id, 'node': socket.gethostname(), 'input': ATOM,
        'basis': BASIS, 'frozen': FROZEN, 'controls': CONTROLS,
        'openmp_threads': NTHREADS, 'scipy_df_threads': DF_THREADS,
        'energy_hartree': float(energy), 'gradient_hartree_per_bohr': coordinates.tolist(),
        'total_seconds': total_seconds, 'df_phase_seconds': phase_seconds,
        'thread_pools_after_run': controller.info(),
    }
    Path(f'mp2_results.{job_id}.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
