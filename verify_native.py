"""Verify the normal editable fork and installed PySCF use LP64 sequential MKL.

Deploy this file beside the source checkout's submit.sh before running it.
NumPy and SciPy retain their own BLAS libraries, reported separately below.
"""
import ctypes
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

import numpy
import scipy.linalg
import pyscf
import pyscfad
import pyscfadlib
from jaxlib.cpu import _lapack
from threadpoolctl import threadpool_info

root = Path(__file__).resolve().parent
prefix = Path('/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env')
mkl_lib = Path('/resnick/software9/spack/opt/spack/linux-rhel9-skylake_avx512/'
               'none-none/intel-oneapi-mkl-2024.2.2-vmnjeo2f64yeg54h3dt3bvwmd2pt7m54/'
               'mkl/2024.2/lib')
assert Path(sys.prefix).resolve() == prefix, sys.prefix
assert Path(pyscfad.__file__).resolve().parent == root / 'pyscfad', pyscfad.__file__
native = root / 'pyscfadlib/pyscfadlib'
assert Path(pyscfadlib.__file__).resolve().parent == native, pyscfadlib.__file__
pyscf_dir = prefix / 'lib/python3.11/site-packages/pyscf'
assert Path(pyscf.__file__).resolve().parent == pyscf_dir, pyscf.__file__
for name, module in tuple(sys.modules.items()):
    path = getattr(module, '__file__', None)
    if name.startswith('pyscfadlib') and path:
        assert Path(path).resolve().is_relative_to(native), (name, path)

extension = 'pyscfad_lapack' + sysconfig.get_config_var('EXT_SUFFIX')
expected_native = {
    'libao2mo_vjp.so', 'libcc_vjp.so', 'libcgto_ad.so', 'libcgto_vjp.so',
    'libcvhf_vjp.so', 'libmp_vjp.so', 'libnp_helper_ad.so',
    'libnp_helper_vjp.so', 'libvjp_util.so', extension,
}
assert {p.name for p in native.glob('*.so')} == expected_native
mp2_symbols = ('MP2_contract_d_vjp', 'MP2_contract_d_energy_vjp')
assert pyscfadlib.libmp_vjp is not None, 'Native MP2 library was not loaded'
for symbol in mp2_symbols:
    assert hasattr(pyscfadlib.libmp_vjp, symbol), symbol
native_mkl = expected_native - {
    'libnp_helper_ad.so', 'libnp_helper_vjp.so', 'libvjp_util.so',
}
pyscf_core = ('libcgto.so', 'libcvhf.so', 'libmp.so', 'libao2mo.so', 'libnp_helper.so')
mkl_sonames = ('libmkl_intel_lp64.so.2', 'libmkl_sequential.so.2', 'libmkl_core.so.2')
dependencies, runtime_paths = {}, {}
libraries = [native / name for name in sorted(expected_native)]
libraries += [pyscf_dir / 'lib' / name for name in pyscf_core]
for library in libraries:
    assert library.is_file(), library
    output = subprocess.check_output(['ldd', str(library)], text=True)
    assert 'not found' not in output, (library, output)
    assert not any(s in output.lower() for s in (
        'flexiblas', 'libiomp5', 'libmkl_intel_ilp64',
        'libmkl_intel_thread', 'libmkl_gnu_thread')), (library, output)
    dynamic = subprocess.check_output(['readelf', '-d', str(library)], text=True)
    rpath = '\n'.join(line for line in dynamic.splitlines()
                      if '(RPATH)' in line or '(RUNPATH)' in line)
    assert str(prefix / 'lib') in rpath, (library, rpath)
    if library.parent != native or library.name in native_mkl:
        assert all(str(mkl_lib / name) in output for name in mkl_sonames), (library, output)
        assert str(mkl_lib) in rpath, (library, rpath)
        assert 'openblas' not in output.lower(), (library, output)
    if library.parent == native and library.name != extension:
        assert str(prefix / 'lib/libgomp.so.1') in output, (library, output)
    dependencies[str(library)] = output
    runtime_paths[str(library)] = rpath

cintad = native / 'thirdparty/lib/libcintad.so'
cintad_sha256 = hashlib.sha256(cintad.read_bytes()).hexdigest()
assert cintad_sha256 == '282c2d83c13ef1117be972872670e2c08a70adc51892f876f1ce0b542c463d0c'
handle = ctypes.CDLL(str(pyscf_dir / 'lib/libcvhf.so'))
handle.MKL_Get_Version_String.argtypes = [ctypes.c_char_p, ctypes.c_int]
handle.MKL_Get_Version_String.restype = None
version = ctypes.create_string_buffer(256)
handle.MKL_Get_Version_String(version, len(version))
mkl_version = version.value.decode()
assert 'Version 2024.2.2' in mkl_version, mkl_version
handle.MKL_Get_Max_Threads.argtypes = []
handle.MKL_Get_Max_Threads.restype = ctypes.c_int
mkl_threads = handle.MKL_Get_Max_Threads()
assert mkl_threads == 1, mkl_threads

# STC is independently compiled against the environment's Armadillo/OpenBLAS.
from pyscfad.dlno_stc import _stc_mp2
stc_path = Path(_stc_mp2.__file__).resolve()
assert stc_path.parent == root / 'pyscfad/dlno_stc', stc_path
assert {name for name in dir(_stc_mp2) if name.startswith('triples_')} == {
    'triples_asymmetric_energy', 'triples_asymmetric_energy_vjp'}
assert 'asymmetric-only' in _stc_mp2.TRIPLES_VERSION
stc_dependencies = subprocess.check_output(['ldd', str(stc_path)], text=True)
assert 'not found' not in stc_dependencies, stc_dependencies
for name in ('libarmadillo.so.15', 'libgomp.so.1', 'libstdc++.so.6'):
    assert str(prefix / 'lib' / name) in stc_dependencies, stc_dependencies
assert 'libblas.so.3' in stc_dependencies, stc_dependencies
assert not any(name in stc_dependencies.lower() for name in
               ('flexiblas', 'libmkl_', 'libiomp5')), stc_dependencies
openblas_path = (prefix / 'lib/libblas.so.3').resolve()
openblas = ctypes.CDLL(str(openblas_path))
openblas.openblas_get_config.restype = ctypes.c_char_p
openblas_config = openblas.openblas_get_config().decode()
assert 'OpenBLAS 0.3.34' in openblas_config, openblas_config
assert 'USE64BITINT' not in openblas_config, openblas_config

# This initializes JAX's CPU LAPACK bindings to SciPy without running a calculation.
_lapack.initialize()
pools = threadpool_info()
numpy_pools = [p for p in pools if '/numpy.libs/' in p['filepath']]
scipy_pools = [p for p in pools if '/scipy.libs/' in p['filepath']]
assert numpy_pools and scipy_pools, pools
assert {p['filepath'] for p in numpy_pools}.isdisjoint(p['filepath'] for p in scipy_pools)
print(json.dumps(dict(
    status='ok', python=sys.executable, pyscf_path=pyscf.__file__,
    pyscfad_path=pyscfad.__file__, pyscfadlib_path=pyscfadlib.__file__,
    native_library_count=len(expected_native),
    native_mp2_symbols=mp2_symbols,
    native_blas=dict(version=mkl_version, interface='LP64', threading='sequential',
                     max_threads=mkl_threads, directory=str(mkl_lib)),
    retained_libcintad=dict(path=str(cintad), sha256=cintad_sha256),
    dependencies=dependencies, runtime_paths=runtime_paths,
    native_sha256={name: hashlib.sha256((native / name).read_bytes()).hexdigest()
                   for name in sorted(expected_native)},
    versions={name: metadata.version(name) for name in
              ('pyscf', 'pyscfad', 'pyscfadlib', 'jax', 'jaxlib', 'numpy', 'scipy', 'nanobind')},
    numpy_threadpools=numpy_pools, scipy_threadpools=scipy_pools,
    all_loaded_threadpools=pools,
    stc=dict(path=str(stc_path), sha256=hashlib.sha256(stc_path.read_bytes()).hexdigest(),
             triples_version=_stc_mp2.TRIPLES_VERSION,
             dependencies=stc_dependencies, blas_path=str(openblas_path),
             blas_config=openblas_config, blas_interface='LP64'),
    scope='MKL checks cover PySCF core and BLAS-using PySCFAD support libraries. '
          'NumPy and SciPy keep their distinct bundled BLAS libraries; '
          'Ordinary JAX CPU LAPACK uses SciPy bindings; the custom generalized '
          'SCF eigensolver uses native MKL. STC Armadillo/OpenBLAS is checked separately.'
), indent=2))
