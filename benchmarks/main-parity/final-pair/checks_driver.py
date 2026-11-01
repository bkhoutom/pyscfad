"""Compare matching MKL libraries and full dCMP gradients in fresh processes."""
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys

root = Path(__file__).resolve().parent
source_root = Path('/resnick/groups/sharmagroup/bkhou/src')
variant = sys.argv[1]
assert variant in ('baseline', 'candidate', 'candidate-precond', 'main')
mode = sys.argv[2] if len(sys.argv) > 2 else 'benchmark'
repo = source_root / ('pyscfad-main' if variant == 'main' else
                      'pyscfad' if variant == 'baseline' else 'pyscfad-perf-integrated')
if variant == 'main':
    native = repo/'pyscfadlib/pyscfadlib'
    pyscf_parent = Path(sys.prefix)/'lib/python3.11/site-packages'
elif variant == 'baseline':
    pyscf_parent = source_root/'pyscfad/benchmarks/pyscfad_mkl/candidate'
    native = pyscf_parent/'pyscfadlib'
else:
    pyscf_parent = source_root/'pyscfad/benchmarks/pyscfad_mkl/candidate'
    native = source_root/'pyscfad/benchmarks/main-parity/final-native/pyscfadlib'
sys.meta_path[:] = [f for f in sys.meta_path if not
    getattr(f, '__module__', type(f).__module__).startswith(
        ('_editable_skbc_pyscfadlib', '__editable___pyscfad_'))]
sys.path[:0] = [str(native.parent), str(repo), str(pyscf_parent)]

import jax
import numpy as np
import pyscf
import pyscfad
import pyscfadlib
from pyscfad import config
from threadpoolctl import threadpool_info

assert Path(pyscf.__file__).resolve().parent == pyscf_parent/'pyscf'
assert Path(pyscfad.__file__).resolve().parent == repo/'pyscfad'
assert Path(pyscfadlib.__file__).resolve().parent == native
for name, module in tuple(sys.modules.items()):
    if name.startswith('pyscfadlib') and getattr(module, '__file__', None):
        assert Path(module.__file__).resolve().is_relative_to(native), (name, module.__file__)
if variant != 'main':
    config.update('pyscfad_scf_df_precond', variant == 'candidate-precond')

libraries = [pyscf_parent/'pyscf/lib/libcvhf.so', pyscf_parent/'pyscf/lib/libmp.so',
             *sorted(native.glob('*.so'))]
dependencies = {}
for path in libraries:
    output = subprocess.check_output(['ldd', str(path)], text=True)
    assert 'not found' not in output and 'flexiblas' not in output.lower(), output
    if path.name.startswith(('libcvhf', 'libmp', 'libcc_vjp', 'pyscfad_lapack')):
        assert all(s in output for s in ('libmkl_intel_lp64', 'libmkl_sequential', 'libmkl_core')), output
        assert 'libiomp5' not in output, output
    dependencies[str(path)] = output
handle = ctypes.CDLL(str(libraries[0]))
handle.MKL_Get_Max_Threads.restype = ctypes.c_int
assert handle.MKL_Get_Max_Threads() == 1
version = ctypes.create_string_buffer(256)
handle.MKL_Get_Version_String(version, 256)
metadata = dict(integral_schedule=os.environ.get('PYSCFAD_INTEGRAL_VJP_SCHEDULE', 'static'), variant=variant, mode=mode, job_id=os.environ.get('SLURM_JOB_ID'),
    hostname=socket.gethostname(), affinity=sorted(os.sched_getaffinity(0)),
    python=sys.executable, versions={n:importlib.metadata.version(n) for n in
        ('jax', 'jaxlib', 'numpy', 'scipy', 'pyscf', 'nanobind')},
    pyscf_path=pyscf.__file__, pyscfad_path=pyscfad.__file__,
    pyscfadlib_path=pyscfadlib.__file__, mkl=version.value.decode(),
    preconditioner=getattr(config, 'scf_df_precond', 'upstream automatic'),
    dependencies=dependencies,
    native_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in libraries},
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (repo/'pyscfad/mp/dfmp2.py', repo/'pyscfad/scf/hf.py',
         repo/'pyscfad/df/_df_jk_opt.py', repo/'pyscfad/_src/_config.py')})
print(json.dumps(metadata, indent=2), flush=True)

if mode == 'benchmark':
    result = runpy.run_path(str(root/'dcmp.py'), init_globals={'DF_THREADS': 16})
    jax.block_until_ready((result['e'], result['grad']))
    energy, gradient = float(result['e']), np.asarray(result['grad'])
    reference = json.loads((root/'reference.json').read_text())
    energy_error = abs(energy-reference['energy'])
    gradient_error = float(np.max(np.abs(gradient-np.asarray(reference['gradient']))))
    assert gradient.shape == (34, 3) and np.isfinite(gradient).all()
    assert energy_error < 1e-8 and gradient_error < 1e-7, (energy_error, gradient_error)
    metadata.update(energy=energy, gradient=gradient.tolist(),
        energy_error=energy_error, gradient_max_error=gradient_error,
        total_seconds=result['total_seconds'], phase_seconds=result['phase_seconds'],
        phase_threadpools=result['phase_threadpools'])
elif mode == 'regressions':
    import pytest
    os.environ['PYSCFAD_REQUIRE_STC_EXTENSION'] = '1'
    tests = ['pyscfadlib/test', 'pyscfad/mp/test',
        'pyscfad/scipy/test/test_linalg.py::test_eigh',
        'tests/test_scf.py::test_rhf_nuc_grad', 'tests/test_scf.py::test_df_rhf_nuc_grad',
        'pyscfad/df/test/test_scf_precond.py', 'pyscfad/df/test/test_df_jk_lowrank.py',
        'pyscfad/dlno_stc/test/test_selected_kernel.py',
        'pyscfad/dlno_stc/test/test_full_kernel.py',
        'pyscfad/dlno_stc/test/test_single_rank.py',
        'pyscfad/lno/test/test_ccsd_t_factor.py::test_factor_direct_energy_and_vjp_match_dense',
        'pyscfad/lno/test/test_ccsd_t_factor.py::test_factor_block_auto_and_numeric_preserve_energy_and_all_cotangents']
    # Independently reproduced on the unchanged fork in job 5765640:
    # conventional (non-DF) nuclear AD leaves mf._eri=None. These are outside
    # the DF-MP2 change; retain their names in the evidence rather than hiding them.
    baseline_failures = ['test_nuc_grad', 'test_nuc_grad_n2']
    metadata['known_baseline_failures'] = {
        'job': 5765640, 'reason': 'Conventional MP2 consumes absent traced SCF ERI cache',
        'tests': ['pyscfad/mp/test/test_mp2.py::'+name for name in baseline_failures]}
    deselect = ['--deselect=pyscfad/mp/test/test_mp2.py::'+name
                for name in baseline_failures]
    code = pytest.main(['-q', '--import-mode=importlib', '--rootdir='+str(repo), *deselect,
                       *[str(repo/p) for p in tests]])
    metadata['pytest_exit_code'] = code
    (root/f'{variant}-{mode}.json').write_text(json.dumps(metadata, indent=2)+'\n')
    raise SystemExit(code)
elif mode in ('dlno-unfrozen', 'dlno-frozen'):
    import pytest
    frozen = 'None' if mode == 'dlno-unfrozen' else '1'
    test = str(repo/'pyscfad/dlno/test/test_mp2_grad.py') + \
        '::test_full_domain_value_and_grad_matches_canonical_dfmp2[' + frozen + ']'
    raise SystemExit(pytest.main(['-q', '--import-mode=importlib', test]))
else:
    raise ValueError(mode)
metadata['threadpools_after'] = threadpool_info()
(root/f'{variant}-{mode}.json').write_text(json.dumps(metadata, indent=2)+'\n')
print(json.dumps(metadata, indent=2), flush=True)
