"""Bounded real-MPI public driver checks; numerical detail stays in the smoke."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[3]


def run_smoke(tmp_path, ranks, mode='acceptance', *, serial=False):
    pytest.importorskip('mpi4py')
    pytest.importorskip('pyscfad.dlno_stc._stc_mp2')
    launcher = shutil.which('mpiexec')
    if launcher is None:
        pytest.skip('mpiexec is unavailable')
    output = tmp_path / f'{mode}-{ranks}-{serial}.json'
    command = [sys.executable, '-m', 'pyscfad.dlno_stc.test.domain_mpi_smoke',
               '--mode', mode, '--output', str(output)]
    command = command + ['--serial'] if serial else [launcher, '-n', str(ranks), *command]
    env = dict(os.environ, OMP_NUM_THREADS='1', OMP_THREAD_LIMIT='1',
               OMP_MAX_ACTIVE_LEVELS='1', OMP_DYNAMIC='FALSE', OMP_PROC_BIND='FALSE',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MKL_DYNAMIC='FALSE',
               BLIS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', PYSCFAD_DENSE_BLAS_THREADS='1',
               JAX_PLATFORMS='cpu', XLA_FLAGS='--xla_cpu_multi_thread_eigen=false')
    env['PYTHONPATH'] = str(SOURCE_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    for name in ('OMP_PLACES', 'GOMP_CPU_AFFINITY', 'KMP_AFFINITY', 'KMP_HW_SUBSET'):
        env.pop(name, None)
    timeout = 60 if mode in ('worker-failure', 'signed-reference',
                                  'zero-send-allocation', 'missing-auxmol') else 240
    try:
        completed = subprocess.run(command, cwd=SOURCE_ROOT, env=env, capture_output=True,
                                   text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f'{mode} on {ranks} ranks hung past {timeout} s: {exc.stdout!r} {exc.stderr!r}')
    assert completed.returncode == 0, completed.stdout + '\n' + completed.stderr
    assert 'Domain-owner MPI smoke PASS' in completed.stdout
    return json.loads(output.read_text())


def compare_groups(actual, expected, *, sampled=False, adaptive=False):
    assert len(actual) == len(expected) == 2
    for result, reference in zip(actual, expected):
        for key in ('fragment_id', 'targets', 'seed', 'tolerance'):
            assert result[key] == reference[key], (key, result, reference)
        if adaptive:
            assert np.isfinite(result['energy'])
            continue
        assert result['count'] == reference['count']
        np.testing.assert_allclose(result['energy'], reference['energy'], atol=2e-10, rtol=0)
        for name in ('foo', 'fvv', 'B'):
            np.testing.assert_allclose(result['bars'][name], reference['bars'][name],
                                       atol=2e-10, rtol=2e-10)


def test_deterministic_domain_only_serial_one_two_four_rank_parity(tmp_path):
    reference = run_smoke(tmp_path, 1, serial=True)
    for ranks in (1, 2, 4):
        result = run_smoke(tmp_path, ranks)
        for name in ('energy', 'correlation_energy', 'supplied_energy'):
            np.testing.assert_allclose(result[name], reference[name], atol=2e-10, rtol=0)
        for name in ('gradient', 'correlation_gradient'):
            np.testing.assert_allclose(result[name], reference[name], atol=2e-8, rtol=0)
        compare_groups(result['groups'], reference['groups'])


@pytest.mark.parametrize('mode,ranks', [('owners', 4), ('unmerged', 2), ('own-vjp', 2),
                                      ('worker-failure', 2), ('signed-reference', 2),
                                      ('zero-send-allocation', 2), ('missing-auxmol', 2)])
def test_domain_ownership_frames_and_synchronized_failures(tmp_path, mode, ranks):
    run_smoke(tmp_path, ranks, mode)


@pytest.mark.parametrize('mode', ['sampled', 'adaptive'])
def test_stochastic_global_streams_and_budgets(tmp_path, mode):
    reference = run_smoke(tmp_path, 1, mode, serial=True)
    result = run_smoke(tmp_path, 2, mode)
    compare_groups(result['groups'], reference['groups'], sampled=mode == 'sampled',
                   adaptive=mode == 'adaptive')
    assert np.isfinite(result['energy'])
    assert np.isfinite(result['gradient']).all()
    if mode == 'sampled':
        np.testing.assert_allclose(result['energy'], reference['energy'], atol=2e-10, rtol=0)
        np.testing.assert_allclose(result['gradient'], reference['gradient'], atol=2e-8, rtol=0)
