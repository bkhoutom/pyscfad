"""Timeout-protected acceptance of CC's shared MPI SCF/J/K service."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from pyscfad.dlno_stc.test.test_domain_mpi import SOURCE_ROOT


def launch(tmp_path, mode='enabled', ranks=2, *, vvvv_direct=False):
    pytest.importorskip('mpi4py')
    pytest.importorskip('pyscfad.dlno_stc._stc_mp2')
    launcher = shutil.which('mpiexec')
    if launcher is None:
        pytest.skip('mpiexec is unavailable')
    output = tmp_path / f'cc-scf-{mode}-{ranks}-{int(vvvv_direct)}.json'
    command = [sys.executable, '-m', 'pyscfad.dlno_stc.test.ccsd_mpi_scf_smoke',
               '--mode', mode, '--output', str(output)]
    if vvvv_direct:
        command.append('--vvvv-direct')
    if mode != 'serial':
        command = [launcher, '-n', str(ranks), *command]
    env = dict(os.environ, OMP_NUM_THREADS='1', OMP_THREAD_LIMIT='1',
               OMP_MAX_ACTIVE_LEVELS='1', OMP_DYNAMIC='FALSE', OMP_PROC_BIND='FALSE',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MKL_DYNAMIC='FALSE',
               BLIS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', PYSCFAD_DENSE_BLAS_THREADS='1',
               JAX_PLATFORMS='cpu', XLA_FLAGS='--xla_cpu_multi_thread_eigen=false')
    env['PYTHONPATH'] = str(SOURCE_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    for name in ('OMP_PLACES', 'GOMP_CPU_AFFINITY', 'KMP_AFFINITY', 'KMP_HW_SUBSET'):
        env.pop(name, None)
    timeout = 120 if mode.endswith('failure') or mode == 'invalid-option' else 300
    try:
        result = subprocess.run(command, cwd=SOURCE_ROOT, env=env, capture_output=True,
                                text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f'CC MPI SCF {mode} hung beyond {timeout}s: {exc.stdout!r} {exc.stderr!r}')
    (tmp_path / f'cc-scf-{mode}-{ranks}.log').write_text(result.stdout + '\n' + result.stderr)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    assert 'CC MPI SCF smoke PASS' in result.stdout
    return json.loads(output.read_text())


def test_cc_root_only_default_and_parallel_scf_opt_in_match_serial(tmp_path):
    reference = launch(tmp_path, 'serial', 1)
    for mode, ranks, direct in (('enabled', 1, False), ('default', 2, False),
                                ('disabled', 2, False),
                                ('enabled', 2, False), ('enabled', 2, True)):
        result = launch(tmp_path, mode, ranks, vvvv_direct=direct)
        np.testing.assert_allclose(result['energy'], reference['energy'], atol=2e-10, rtol=0)
        np.testing.assert_allclose(result['gradient'], reference['gradient'], atol=2e-8, rtol=0)


def test_cc_energy_kernel_uses_forward_only_service_and_worker_skeleton(tmp_path):
    reference = launch(tmp_path, 'serial', 1)
    result = launch(tmp_path, 'kernel', 2)
    np.testing.assert_allclose(result['energy'], reference['energy'], atol=2e-10, rtol=0)


@pytest.mark.parametrize('mode', ['group-failure', 'response-failure', 'invalid-option'])
def test_cc_parallel_scf_failure_is_collective_and_cleans_registry(tmp_path, mode):
    result = launch(tmp_path, mode)
    assert all(row['failed'] and row['registry_clean'] and row['active_executor_clean']
               for row in result['observations'])
