"""Select a saved native package, then run its checks or the local dCMP input."""
import ctypes
import json
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys

root = Path(__file__).resolve().parent
variant, mode = sys.argv[1:3]
assert variant in ('original', 'candidate', 'installed')
assert mode in ('linkage', 'tests', 'gradient-test', 'smoke', 'benchmark')
if variant != 'installed':
    # This environment's editable finder otherwise overrides PYTHONPATH.
    sys.meta_path[:] = [finder for finder in sys.meta_path
                       if type(finder).__module__ != '_editable_skbc_pyscfadlib']
    sys.path.insert(0, str(root / variant))
    expected = root / variant / 'pyscfadlib'
else:
    expected = root.parents[1] / 'pyscfadlib/pyscfadlib'

import numpy as np
import pyscf
import pyscfad
import pyscfadlib
from pyscfadlib import lib
from threadpoolctl import threadpool_info

assert Path(pyscfadlib.__file__).resolve().parent == expected, pyscfadlib.__file__
assert Path(pyscfad.__file__).resolve().is_relative_to(root.parents[1] / 'pyscfad')
for name in ('libao2mo_vjp', 'libcc_vjp', 'libcgto_ad', 'libcgto_vjp',
             'libcvhf_vjp', 'libnp_helper_vjp'):
    assert Path(getattr(lib, name)._name).resolve().parent == expected, name
for name, module in tuple(sys.modules.items()):
    if name.startswith('pyscfadlib') and getattr(module, '__file__', None):
        assert Path(module.__file__).resolve().is_relative_to(expected), (name, module.__file__)

metadata = dict(variant=variant, mode=mode, hostname=socket.gethostname(),
                affinity=sorted(os.sched_getaffinity(0)),
                pyscf_path=pyscf.__file__, pyscfad_path=pyscfad.__file__,
                pyscfadlib_path=pyscfadlib.__file__, threadpools=threadpool_info())

if mode == 'linkage':
    dependencies = {}
    for library in sorted(expected.glob('*.so')):
        output = subprocess.check_output(['ldd', str(library)], text=True)
        assert 'not found' not in output, (library, output)
        if variant != 'original':
            assert 'flexiblas' not in output.lower(), (library, output)
            if library.name.startswith(('libcgto_ad', 'libcc_vjp', 'libcvhf_vjp', 'pyscfad_lapack')):
                assert '/envs/pyscfad_env/lib/libopenblas.so.0' in output, (library, output)
        dependencies[library.name] = output
    if variant != 'original':
        handle = ctypes.CDLL('/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env/lib/libopenblas.so.0')
        handle.openblas_get_config.restype = ctypes.c_char_p
        config = handle.openblas_get_config().decode()
        assert '0.3.34' in config and 'USE64BITINT' not in config, config
        metadata['blas_config'] = config
    metadata['dependencies'] = dependencies
elif mode == 'gradient-test':
    import pytest
    frozen = sys.argv[3]
    assert frozen in ('None', '1')
    test = root.parents[1] / 'pyscfad/dlno/test/test_mp2_grad.py'
    raise SystemExit(pytest.main(['-q', '--import-mode=importlib',
        str(test) + '::test_full_domain_value_and_grad_matches_canonical_dfmp2[' + frozen + ']']))
elif mode == 'tests':
    import pytest
    repo = root.parents[1]
    os.environ['PYSCFAD_REQUIRE_STC_EXTENSION'] = '1'
    raise SystemExit(pytest.main(['-q', '--import-mode=importlib',
        str(repo / 'pyscfadlib/test'),
        str(repo / 'pyscfad/scipy/test/test_linalg.py') + '::test_eigh',
        str(repo / 'tests/test_scf.py') + '::test_rhf_nuc_grad',
        str(repo / 'tests/test_scf.py') + '::test_df_rhf_nuc_grad',
        str(repo / 'pyscfad/dlno_stc/test/test_selected_kernel.py'),
        str(repo / 'pyscfad/dlno_stc/test/test_full_kernel.py'),
        str(repo / 'pyscfad/dlno_stc/test/test_single_rank.py'),
        str(repo / 'pyscfad/lno/test/test_ccsd_t_factor.py') + '::test_factor_direct_energy_and_vjp_match_dense',
        str(repo / 'pyscfad/lno/test/test_ccsd_t_factor.py') + '::test_factor_block_auto_and_numeric_preserve_energy_and_all_cotangents']))
elif mode == 'smoke':
    os.chdir(root / 'smoke')
    runpy.run_path('verify.py')
else:
    print(json.dumps(metadata, indent=2), flush=True)
    result = runpy.run_path(str(root / 'driver.py'))
    energy = float(result['e'])
    gradient = np.asarray(result['grad'])
    assert abs(energy - (-1381.31940656203)) < 1e-8, energy
    assert np.isfinite(gradient).all()
    metadata.update(energy=energy, gradient=gradient.tolist(),
                    total_seconds=result['total_seconds'],
                    threadpools_after=threadpool_info())
output_path = root / f'{variant}-{mode}.json'
output_path.write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata, indent=2), flush=True)
