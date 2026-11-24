"""Native DF kernels preserve the caller's policy during and after work."""
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from pyscf import lib
from threadpoolctl import threadpool_info

from pyscfad.df import _cderi_vjp, _int3c_cross_opt
from pyscfad.df.test.test_reverse_native_slices import _molecules


@pytest.fixture(scope='module')
def parallel_policy_probe(tmp_path_factory):
    """Observe real GNU OpenMP worker regions without changing their policy."""
    if not sys.platform.startswith('linux'):
        pytest.skip('The worker observer uses GNU OpenMP symbol interposition')
    if not hasattr(_int3c_cross_opt.libcgto, 'GOMP_parallel'):
        pytest.skip('The native library does not use GNU OpenMP')
    runtime = next((pool['filepath'] for pool in threadpool_info()
                    if Path(pool['filepath']).name.startswith('libgomp.so')), None)
    if runtime is None:
        pytest.skip('The GNU OpenMP runtime path is unavailable')
    compiler = shutil.which('cc')
    if compiler is None:
        pytest.skip('A C compiler is needed for the worker observer')
    library = tmp_path_factory.mktemp('omp_policy') / 'probe.so'
    source = Path(__file__).with_name('_omp_dynamic_probe.c')
    subprocess.run([compiler, '-shared', '-fPIC', '-O2', str(source),
                    '-o', str(library), '-ldl'], check=True, capture_output=True)
    return library, runtime


@pytest.mark.parametrize('dynamic', ['FALSE', 'TRUE'])
@pytest.mark.parametrize('kernel', ['integral', 'packing'])
def test_submission_dynamic_policy_reaches_native_workers(
        parallel_policy_probe, dynamic, kernel):
    """A kernel that overrides OMP_DYNAMIC even temporarily changes its workers."""
    probe_library, runtime_library = parallel_policy_probe
    code = '''
import ctypes
import os
import pytest
from pyscfad.df import _int3c_cross_opt
from pyscfad.df.test.test_native_dynamic_teams import test_native_team_policy_restores_caller

expected = int(os.environ['OMP_DYNAMIC'] == 'TRUE')
runtime = _int3c_cross_opt.libcgto
runtime.omp_get_dynamic.restype = ctypes.c_int
assert runtime.omp_get_dynamic() == expected
probe = ctypes.CDLL(os.environ['POLICY_PROBE'])
probe.policy_probe_count.argtypes = [ctypes.c_int]
probe.policy_probe_count.restype = ctypes.c_int
with pytest.MonkeyPatch.context() as patch:
    probe.policy_probe_start()
    test_native_team_policy_restores_caller(expected, os.environ['POLICY_KERNEL'], patch)
    probe.policy_probe_stop()
counts = [probe.policy_probe_count(i) for i in (0, 1)]
assert sum(counts) > 0, ('no workers observed', counts)
assert counts[1-expected] == 0, ('kernel changed dynamic policy', expected, counts)
'''
    env = dict(os.environ, OMP_DYNAMIC=dynamic, OMP_NUM_THREADS='4',
               OMP_THREAD_LIMIT='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               POLICY_PROBE=str(probe_library), POLICY_KERNEL=kernel)
    # ctypes loads native libraries locally; expose the same OpenMP runtime
    # globally so the observer can forward to it through RTLD_NEXT.
    env['LD_PRELOAD'] = ' '.join(filter(None, [str(probe_library), runtime_library,
                                             env.get('LD_PRELOAD')]))
    result = subprocess.run([sys.executable, '-c', code], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('dynamic', [0, 1])
@pytest.mark.parametrize('kernel', ['integral', 'packing'])
def test_native_team_policy_restores_caller(dynamic, kernel, monkeypatch):
    runtime = (_int3c_cross_opt.libcgto if kernel == 'integral'
               else _cderi_vjp.libao2mo_vjp)
    runtime.omp_get_dynamic.restype = ctypes.c_int
    runtime.omp_set_dynamic.argtypes = [ctypes.c_int]
    runtime.omp_get_max_threads.restype = ctypes.c_int
    previous = runtime.omp_get_dynamic()
    try:
        with lib.with_omp_threads(4):
            runtime.omp_set_dynamic(dynamic)
            maximum = runtime.omp_get_max_threads()
            if kernel == 'integral':
                mol, aux = _molecules()
                seed = np.ones((mol.nao*(mol.nao+1)//2, aux.nao), order='F')
                bars = _int3c_cross_opt.int3c_cross_bwd(
                    'int3c2e', 1, 's2ij', None, None, (mol, aux), seed)
                assert all(np.isfinite(b.coords).all() for b in bars)
            else:
                monkeypatch.setenv('PYSCFAD_DF_MO_DENSITY_BACKEND', 'tiled')
                rng = np.random.default_rng(71)
                c = rng.normal(size=(129, 7))
                seed = rng.normal(size=(2, 7, 7))
                result = _cderi_vjp._int3c_mo_packed_density(c, c, seed)
                assert np.isfinite(result).all()
            assert runtime.omp_get_dynamic() == dynamic
            assert runtime.omp_get_max_threads() == maximum
    finally:
        runtime.omp_set_dynamic(previous)
