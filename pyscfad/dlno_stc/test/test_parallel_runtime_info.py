"""Probe actual native/OpenMP worker masks without changing their placement."""
import json
import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux affinity diagnostics')
@pytest.mark.parametrize('requested', [4, 40])
def test_actual_worker_masks_and_loaded_runtime(requested, tmp_path):
    worker = r'''
import ctypes, json, os, shutil, sys
from pathlib import Path
from pyscfad.dlno_stc import _stc_mp2 as native
expected = min(int(os.environ['OMP_NUM_THREADS']), 32)
before = os.sched_getaffinity(0)
info = native.parallel_runtime_info()
assert info['threads'] == expected
assert [w['thread'] for w in info['workers']] == list(range(expected))
for w in info['workers']:
    assert w['affinity'] == sorted(before)
    assert w['cpu'] in w['affinity']
paths = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
         if 'libgomp' in line and line.split()[-1].startswith('/')}
assert paths
for path in paths:
    external = native.parallel_runtime_info(path)
    assert external['threads'] == expected
    assert [w['thread'] for w in external['workers']] == list(range(expected))
    assert all(w['affinity'] == sorted(before) and w['cpu'] in w['affinity']
               for w in external['workers'])
# A separately loaded GNU runtime has its own team allocation. The optional
# path must query and execute that runtime, rather than the extension runtime.
copy_path = Path(sys.argv[1]) / 'libgomp-probe.so'
shutil.copyfile(sorted(paths)[0], copy_path)
other = ctypes.CDLL(str(copy_path), mode=ctypes.RTLD_LOCAL)
other.omp_set_num_threads.argtypes = [ctypes.c_int]
other.omp_set_num_threads(2)
separate = native.parallel_runtime_info(str(copy_path))
assert separate['threads'] == 2
assert all(w['affinity'] == sorted(before) and w['cpu'] in w['affinity']
           for w in separate['workers'])
assert native.parallel_runtime_info()['threads'] == expected
assert os.sched_getaffinity(0) == before
try:
    native.parallel_runtime_info('/tmp/not-a-loaded-openmp-runtime.so')
except RuntimeError as error:
    assert 'already loaded' in str(error)
else:
    raise AssertionError('unloaded runtime was accepted')
print(json.dumps(info))
'''
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(requested), OMP_THREAD_LIMIT=str(requested), OMP_DYNAMIC='FALSE',
               OMP_PROC_BIND='FALSE', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    result = subprocess.run([sys.executable, '-c', worker, str(tmp_path)], env=env,
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)['threads'] == min(requested, 32)
