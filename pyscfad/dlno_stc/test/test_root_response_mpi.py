"""Root response retains fusion/caching with one or multiple MPI ranks."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize('method', ['mp2', 'ccsd'])
def test_root_response_fuses_geometry_with_mpi_owners(tmp_path, method):
    pytest.importorskip('mpi4py')
    pytest.importorskip('pyscfad.dlno_stc._stc_mp2')
    launcher = shutil.which('mpiexec')
    if launcher is None:
        pytest.skip('mpiexec is unavailable')
    env = dict(os.environ, OMP_NUM_THREADS='1', OMP_THREAD_LIMIT='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYSCFAD_DENSE_BLAS_THREADS='1', JAX_PLATFORMS='cpu')
    env['PYTHONPATH'] = str(SOURCE_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    for name in ('PYSCFAD_SCF_OCCUPIED_CACHE', 'PYSCFAD_SCF_RESPONSE_FACTOR_CACHE_MB'):
        env.pop(name, None)
    results = []
    for ranks in (1, 2):
        output = tmp_path / f'{method}-{ranks}.json'
        command = [launcher, '-n', str(ranks), sys.executable, '-m',
                   'pyscfad.dlno_stc.test.root_response_mpi_smoke',
                   '--method', method, '--output', str(output)]
        result = subprocess.run(command, cwd=SOURCE_ROOT, env=env,
                                capture_output=True, text=True, timeout=180)
        (tmp_path / f'{method}-{ranks}.log').write_text(result.stdout + '\n' + result.stderr)
        assert result.returncode == 0, result.stdout + '\n' + result.stderr
        assert 'root response smoke PASS' in result.stdout
        results.append(json.loads(output.read_text())['results'][1])
    np.testing.assert_allclose(results[1]['energy'], results[0]['energy'], atol=2e-11, rtol=0)
    np.testing.assert_allclose(results[1]['gradient'], results[0]['gradient'], atol=2e-8, rtol=0)
