"""Check real runtime policy after benchmark startup in an isolated process."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_auto_policy_reaches_openmp_mkl_and_preserves_jax_affinity():
    if not hasattr(os, 'sched_getaffinity') or len(os.sched_getaffinity(0)) < 32:
        pytest.skip('requires Linux affinity and at least 32 available CPUs')
    code = r'''
import ctypes, json, os, sys
sys.path.insert(0, sys.argv[1])
from benchmark_affinity import configure_cpu_workload, check_worker_affinity
cpus = configure_cpu_workload(32)
import jax, jax.numpy as jnp
from pyscf import lib
jax.block_until_ready(jnp.ones((4, 4)) @ jnp.ones((4, 4)))
omp = ctypes.CDLL('libgomp.so.1')
try:
    mkl = ctypes.CDLL('libmkl_rt.so.2')
except OSError:
    sys.exit(77)
assert omp.omp_get_dynamic() != 0
assert omp.omp_get_thread_limit() == 32
assert 1 <= omp.omp_get_max_threads() <= 32
assert mkl.MKL_Get_Dynamic() != 0
assert 1 <= mkl.MKL_Get_Max_Threads() <= 32
assert set(os.sched_getaffinity(0)) == set(cpus)
info = check_worker_affinity(cpus)
for task in __import__('pathlib').Path('/proc/self/task').iterdir():
    try:
        assert set(os.sched_getaffinity(int(task.name))) == set(cpus)
    except ProcessLookupError:
        pass
print(json.dumps(info))
'''
    env = dict(os.environ, PYSCFAD_BENCHMARK_THREAD_POLICY='auto')
    result = subprocess.run([sys.executable, '-c', code, str(Path(__file__).parent)],
                            env=env, capture_output=True, text=True, timeout=60)
    if result.returncode == 77:
        pytest.skip('requires the MKL runtime')
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(json.loads(result.stdout.splitlines()[-1])['cpu_affinity']) == 32
