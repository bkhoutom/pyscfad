"""Numerical single-pass owner-local CC selection and collective validation."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from pyscfad.dlno_stc.test.test_domain_mpi import SOURCE_ROOT


def launch(tmp_path, *, ranks=3, serial=False, gradient=False,
           include_baseline=False, invalid=False, parallel_scf_jk=False):
    pytest.importorskip('mpi4py')
    pytest.importorskip('pyscfad.dlno_stc._stc_mp2')
    launcher = shutil.which('mpiexec')
    if launcher is None:
        pytest.skip('mpiexec is unavailable')
    output = tmp_path / f'single-pass-{serial}-{gradient}-{include_baseline}-{invalid}-{parallel_scf_jk}.json'
    command = [sys.executable, '-m', 'pyscfad.dlno_stc.test.ccsd_single_pass_mpi_smoke',
               '--output', str(output)]
    for option, enabled in (('serial', serial), ('gradient', gradient),
                           ('include-baseline', include_baseline), ('invalid', invalid),
                           ('parallel-scf-jk', parallel_scf_jk)):
        if enabled:
            command.append('--' + option)
    if not serial:
        command = [launcher, '-n', str(ranks), *command]
    env = dict(os.environ, OMP_NUM_THREADS='1', OMP_THREAD_LIMIT='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYSCFAD_DENSE_BLAS_THREADS='1', JAX_PLATFORMS='cpu',
               PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE='half',
               PYSCFAD_LNO_HALF_DF_CACHE_MB='1000',
               XLA_FLAGS='--xla_cpu_multi_thread_eigen=false')
    env['PYTHONPATH'] = str(SOURCE_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    for key in ('OMP_PLACES', 'GOMP_CPU_AFFINITY', 'KMP_AFFINITY', 'KMP_HW_SUBSET'):
        env.pop(key, None)
    try:
        result = subprocess.run(command, cwd=SOURCE_ROOT, env=env,
                                capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f'single-pass CC MPI hung beyond 300s: {exc.stdout!r} {exc.stderr!r}')
    output.with_suffix('.log').write_text(result.stdout + '\n' + result.stderr)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    return json.loads(output.read_text())


def compare(actual, reference, *, gradient=False):
    for name in ('energy', 'baseline', 'weak'):
        np.testing.assert_allclose(actual[name], reference[name], atol=3e-10, rtol=0)
    if gradient:
        np.testing.assert_allclose(actual['gradient'], reference['gradient'], atol=2e-8, rtol=0)
    for actual_row, expected in zip(actual['fragments'], reference['fragments'], strict=True):
        for name in ('index', 'dimensions', 'samples', 'target'):
            assert actual_row[name] == expected[name]
        np.testing.assert_allclose(actual_row['energies'], expected['energies'], atol=3e-10, rtol=0)


@pytest.mark.parametrize('parallel_scf_jk', [False, True])
def test_single_pass_mpi_partial_cc_keeps_full_baseline(tmp_path, parallel_scf_jk):
    # Run MPI first so the unsupported-single-pass guard is the initial failure.
    actual = launch(tmp_path, ranks=2, include_baseline=True, parallel_scf_jk=parallel_scf_jk)
    reference = launch(tmp_path, serial=True, include_baseline=True)
    compare(actual, reference)
    assert abs(actual['weak']) > 1e-10


def test_single_pass_mpi_gradient_subset_and_empty_owner(tmp_path):
    actual = launch(tmp_path, gradient=True)
    reference = launch(tmp_path, serial=True, gradient=True)
    compare(actual, reference, gradient=True)
    assert actual['baseline'] == actual['weak'] == 0.
    assert actual['observations'][2]['groups'] == []


def test_single_pass_mpi_rejects_stale_schema_scientific_controls_and_proxy_budgets(tmp_path):
    result = launch(tmp_path, ranks=2, invalid=True)
    assert all(not row['groups'] for row in result['observations'])
