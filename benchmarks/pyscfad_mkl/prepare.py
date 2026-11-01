"""Snapshot installed distribution-owned files and fork sources, without installation."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PREFIX = Path('/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env')
FORK = ROOT.parents[1]
MKL = Path('/resnick/software9/spack/opt/spack/linux-rhel9-skylake_avx512/none-none/intel-oneapi-mkl-2024.2.2-vmnjeo2f64yeg54h3dt3bvwmd2pt7m54/mkl/2024.2/lib')
assert Path(sys.prefix) == PREFIX
dist = metadata.distribution('pyscf')
assert dist.version == '2.14.0'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(directory, predicate=lambda p: True):
    return {str(p.relative_to(directory)): digest(p)
            for p in sorted(directory.rglob('*'))
            if p.is_file() and predicate(p)}


if (ROOT / 'input.json').exists():
    raise SystemExit('Snapshot already exists; retain it as the reproducible input.')
owned = []
for entry in dist.files:
    rel = Path(str(entry))
    if '__pycache__' in rel.parts or rel.suffix == '.pyc':
        continue
    if rel.parts[0] not in ('pyscf', 'pyscf-2.14.0.dist-info'):
        raise RuntimeError(f'Unexpected distribution-owned path: {rel}')
    if rel.name in ('RECORD', 'INSTALLER', 'REQUESTED', 'direct_url.json'):
        continue
    for variant in ('baseline', 'candidate'):
        target = ROOT / variant / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dist.locate_file(entry), target)
    owned.append(str(rel))

native = FORK / 'pyscfadlib/pyscfadlib'
ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache')
for variant in ('baseline', 'candidate'):
    shutil.copytree(native, ROOT / variant / 'pyscfadlib', symlinks=True, ignore=ignore)
shutil.copytree(FORK / 'pyscfad', ROOT / 'python/pyscfad',
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.so', '*.a', '.pytest_cache'))
for variant in ('baseline', 'candidate'):
    # PySCF has no licensed SMD Fortran source in its source distribution.
    assert (ROOT / variant / 'pyscf/lib/libsolvent.so').is_file()
    assert (ROOT / variant / 'pyscfadlib/thirdparty/lib/libcintad.so').is_file()

spec = {
    'environment': str(PREFIX),
    'fork': str(FORK),
    'fork_commit': subprocess.check_output(['git', '-C', str(FORK), 'rev-parse', 'HEAD'], text=True).strip(),
    'mkl_lib': str(MKL),
    'mkl_version': '2024.2.2',
    'blas_abi': 'LP64',
    'threading': 'sequential MKL; GNU OpenMP in native outer loops',
    'cpu_target': 'x86-64-v3 (AVX2), generic tuning',
    'versions': {name: metadata.version(name) for name in
                 ('pyscf', 'pyscfad', 'pyscfadlib', 'jax', 'jaxlib', 'numpy', 'scipy', 'nanobind')},
    'pyscf_owned_files': owned,
    'native_files': [p.name for p in sorted(native.glob('*.so'))],
    'live_native_sha256': {p.name: digest(p) for p in sorted(native.glob('*.so'))},
    'native_source_sha256': hashes(native, lambda p: p.suffix in ('.py', '.c', '.cc', '.h', '.txt', '.in')),
    'python_source_sha256': hashes(ROOT / 'python/pyscfad'),
    'retained_libcintad_sha256': digest(native / 'thirdparty/lib/libcintad.so'),
    'retained_pyscf_libcint_sha256': digest(ROOT / 'baseline/pyscf/lib/deps/lib/libcint.so'),
    'retained_solvent_sha256': digest(ROOT / 'baseline/pyscf/lib/libsolvent.so'),
    'mkl_sha256': {name: digest(MKL / name) for name in
                  ('libmkl_intel_lp64.so', 'libmkl_sequential.so', 'libmkl_core.so')},
    'water': {
        'atom': [['O', [0.0, 0.0, 0.0]], ['H', [0.0, -1.43233673, 1.10715266]],
                 ['H', [0.0, 1.43233673, 1.10715266]]],
        'unit': 'Bohr', 'basis': 'def2-svp', 'auxbasis': 'def2-svp-jkfit',
        'frozen': 1, 'conv_tol': 1e-11, 'conv_tol_grad': 1e-8, 'max_cycle': 100,
        'fd_step_bohr': 1e-3, 'fd_atom': 1, 'fd_axis': 2,
        'energy_tolerance': 1e-8, 'gradient_tolerance': 2e-6,
    },
    'prior_dcmp_fork_seconds': {'openblas': 285.210, 'mkl': 279.559},
    'prior_timing_caveat': 'One common-main-env sample each; not this oldenv-specific candidate.',
}
(ROOT / 'input.json').write_text(json.dumps(spec, indent=2) + '\n')
print('Snapshots prepared; no live packages changed.')
