"""Verify this fork's active editable installation and modern LP64 linkage."""
import ctypes
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

import pyscf
import pyscfad
import pyscfadlib
from threadpoolctl import threadpool_info

root = Path(__file__).resolve().parent
prefix = Path('/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env')
assert Path(sys.prefix) == prefix, sys.prefix
assert Path(pyscfad.__file__).resolve().is_relative_to(root / 'pyscfad')
native = root / 'pyscfadlib/pyscfadlib'
assert Path(pyscfadlib.__file__).resolve().parent == native
blas = prefix / 'lib/libopenblas.so.0'
handle = ctypes.CDLL(str(blas))
handle.openblas_get_config.restype = ctypes.c_char_p
config = handle.openblas_get_config().decode()
assert '0.3.34' in config and 'USE64BITINT' not in config, config
dependencies = {}
for library in sorted(native.glob('*.so')):
    output = subprocess.check_output(['ldd', str(library)], text=True)
    assert 'not found' not in output and 'flexiblas' not in output.lower(), output
    if library.name.startswith(('libcgto_ad', 'libcc_vjp', 'libcvhf_vjp', 'pyscfad_lapack')):
        assert str(blas) in output, output
    dependencies[library.name] = output
assert len(dependencies) == 9, dependencies
print(json.dumps(dict(python=sys.executable, pyscf_path=pyscf.__file__,
                      pyscfad_path=pyscfad.__file__, pyscfadlib_path=pyscfadlib.__file__,
                      blas_config=config, dependencies=dependencies,
                      versions={name: metadata.version(name) for name in
                                ('pyscf', 'pyscfad', 'pyscfadlib', 'jax', 'jaxlib',
                                 'numpy', 'scipy', 'nanobind')},
                      threadpools=threadpool_info()), indent=2))
