"""Check real runtime policy after benchmark startup in an isolated process."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('dynamic', [None, ('TRUE', 'FALSE')])
def test_runtime_defaults_and_caller_settings_survive_benchmark_startup(dynamic):
    if not hasattr(os, 'sched_getaffinity') or len(os.sched_getaffinity(0)) < 2:
        pytest.skip('requires Linux affinity and at least two available CPUs')
    code = r'''
import ctypes, json, os, sys
sys.path.insert(0, sys.argv[1])
from benchmark_affinity import configure_cpu_workload, check_worker_affinity
before = {key: os.environ.get(key) for key in ('OMP_DYNAMIC', 'MKL_DYNAMIC')}
cpus = configure_cpu_workload(2)
assert {key: os.environ.get(key) for key in before} == before
import jax, jax.numpy as jnp
from pyscf import lib
jax.block_until_ready(jnp.ones((4, 4)) @ jnp.ones((4, 4)))
omp = ctypes.CDLL('libgomp.so.1')
try:
    mkl = ctypes.CDLL('libmkl_rt.so.2')
except OSError:
    sys.exit(77)
assert bool(omp.omp_get_dynamic()) == (before['OMP_DYNAMIC'] == 'TRUE')
assert omp.omp_get_thread_limit() == 2
assert 1 <= omp.omp_get_max_threads() <= 2
assert bool(mkl.MKL_Get_Dynamic()) == (before['MKL_DYNAMIC'] != 'FALSE')
assert 1 <= mkl.MKL_Get_Max_Threads() <= 2
from pyscfad.lib._threading import fixed_threads, blas_threads
from threadpoolctl import ThreadpoolController
controller = ThreadpoolController()
assert any(pool.internal_api == 'mkl' for pool in controller.lib_controllers)
initial_omp, initial_mkl = omp.omp_get_dynamic(), mkl.MKL_Get_Dynamic()
try:
    with fixed_threads():
        assert omp.omp_get_dynamic() == 0
        assert mkl.MKL_Get_Dynamic() == initial_mkl
        with blas_threads(1), fixed_threads():
            assert mkl.MKL_Get_Max_Threads() == 1
            assert mkl.MKL_Get_Dynamic() == initial_mkl
        raise RuntimeError('check restoration')
except RuntimeError as exc:
    assert str(exc) == 'check restoration'
assert omp.omp_get_dynamic() == initial_omp
assert mkl.MKL_Get_Dynamic() == initial_mkl
assert set(os.sched_getaffinity(0)) == set(cpus)
info = check_worker_affinity(cpus)
for task in __import__('pathlib').Path('/proc/self/task').iterdir():
    try:
        assert set(os.sched_getaffinity(int(task.name))) == set(cpus)
    except ProcessLookupError:
        pass
print(json.dumps(info))
'''
    env = dict(os.environ)
    for key in ('OMP_DYNAMIC', 'MKL_DYNAMIC'):
        env.pop(key, None)
    if dynamic is not None:
        env.update(zip(('OMP_DYNAMIC', 'MKL_DYNAMIC'), dynamic))
    result = subprocess.run([sys.executable, '-c', code, str(Path(__file__).parent)],
                            env=env, capture_output=True, text=True, timeout=60)
    if result.returncode == 77:
        pytest.skip('requires the MKL runtime')
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(json.loads(result.stdout.splitlines()[-1])['cpu_affinity']) == 2
