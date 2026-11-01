"""Measure the two local plain-PySCF drivers with actual DF pool diagnostics."""
import contextlib
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import re
import runpy
import socket
import subprocess
import sys
import time

import pyscf
from pyscf.df.df import DF
from threadpoolctl import threadpool_info

root = Path(__file__).resolve().parent
spec = json.loads((root / 'input.json').read_text())
label = sys.argv[1]
assert label in ('original', 'optimized')
assert os.environ.get('SLURM_JOB_ID'), 'Use submit.sh'
assert Path(sys.prefix) == Path(spec['environment'])
assert Path(pyscf.__file__).is_relative_to(Path(sys.prefix))
assert len(os.sched_getaffinity(0)) == 16
source = root / ('original.py' if label == 'original' else 'pyscf_scf.py')
handle = ctypes.CDLL(str(Path(pyscf.__file__).parent / 'lib/libcvhf.so'))
handle.MKL_Get_Max_Threads.restype = ctypes.c_int
assert handle.MKL_Get_Max_Threads() == 1
version = ctypes.create_string_buffer(256)
handle.MKL_Get_Version_String(version, len(version))
libraries = [Path(pyscf.__file__).parent / 'lib' / name
             for name in ('libcvhf.so', 'libmp.so')]
linkage = {p.name: subprocess.check_output(['ldd', str(p)], text=True)
           for p in libraries}
assert all('not found' not in text and 'libmkl_intel_lp64' in text
           and 'libmkl_sequential' in text for text in linkage.values())

df_calls = []
build = DF.build
def measured_build(self, *args, **kwargs):
    pools = threadpool_info()
    start = time.perf_counter()
    result = build(self, *args, **kwargs)
    df_calls.append(dict(seconds=time.perf_counter()-start, threadpools=pools))
    return result

capture = io.StringIO()
class Tee:
    def write(self, text):
        sys.__stdout__.write(text)
        return capture.write(text)
    def flush(self):
        sys.__stdout__.flush()

DF.build = measured_build
try:
    with contextlib.redirect_stdout(Tee()):
        calculation = runpy.run_path(str(source))
finally:
    DF.build = build

energy = float(calculation['e'])
assert abs(energy-spec['reference_energy']) < 1e-8
assert len(df_calls) == 1, df_calls
for pool in df_calls[0]['threadpools']:
    if pool['user_api'] == 'blas':
        expected = 16 if label == 'optimized' and '/scipy.libs/' in pool['filepath'] else 1
        assert pool['num_threads'] == expected, (pool, expected)
phases = {name: float(seconds) for name,seconds in re.findall(
    r'^(SCF forward \(including DF build\)|MP2 forward \(excluding SCF\)): ([\d.]+) s$',
    capture.getvalue(), re.M)}
assert len(phases) == 2
result = dict(label=label, job_id=os.environ['SLURM_JOB_ID'],
    hostname=socket.gethostname(), affinity=sorted(os.sched_getaffinity(0)),
    python=sys.executable, pyscf_path=pyscf.__file__, mkl=version.value.decode(),
    linkage=linkage, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    native_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in libraries},
    energy=energy, total_seconds=float(calculation['total_seconds']),
    phases=phases, df_calls=df_calls, threadpools_after=threadpool_info())
assert all(p['num_threads'] == 1 for p in result['threadpools_after']
           if p['user_api'] == 'blas')
(root / f'{label}.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2), flush=True)
