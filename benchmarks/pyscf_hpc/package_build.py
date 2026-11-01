#!/usr/bin/env python3
"""Package a verified build and a rollback wheel without changing the environment."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('variant', choices=('stock', 'native', 'integrals'))
args = parser.parse_args()
dist = importlib.metadata.distribution('pyscf')
assert dist.version == '2.14.0'
if args.variant != 'stock':
    summary = json.loads((root / 'summary.json').read_text())
    assert summary['status'] == 'ok', 'Numerical validation must pass before packaging.'

stage = root / 'wheel_staging' / args.variant
if stage.exists():
    raise SystemExit(f'Packaging stage already exists: {stage}')
stage.mkdir(parents=True)

# Respect distribution ownership: pyscf-properties also installs under pyscf/.
for entry in dist.files:
    relative = Path(str(entry))
    if '__pycache__' in relative.parts or relative.suffix == '.pyc':
        continue
    if relative.parts[0] == 'pyscf':
        source = root / args.variant / relative
    elif relative.parts[0] == 'pyscf-2.14.0.dist-info':
        if relative.name in ('RECORD', 'INSTALLER', 'REQUESTED', 'direct_url.json'):
            continue
        source = Path(dist.locate_file(entry))
    else:
        raise RuntimeError(f'Unexpected package ownership: {relative}')
    target = stage / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

build_tag = None
if args.variant != 'stock':
    # Include the integral library's versioned SONAME aliases if it was rebuilt.
    if args.variant == 'integrals':
        for source in (root / args.variant / 'pyscf/lib/deps/lib').glob('libcint.so*'):
            shutil.copy2(source, stage / 'pyscf/lib/deps/lib' / source.name)
        shutil.copy2(root / args.variant / 'pyscf/lib/deps/LIBCINT_LICENSE',
                     stage / 'pyscf/lib/deps/LIBCINT_LICENSE')
    info = {
        'upstream_version': '2.14.0', 'variant': args.variant,
        'cpu_target': 'x86-64-v3 (AVX2), generic tuning',
        'compiler': '/usr/bin/gcc 11.5.0', 'build_type': 'Release',
        'blas': '/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env/lib/libopenblas.so',
        'blas_version': '0.3.34', 'blas_integer_interface': 'LP64',
        'integral_engine': 'libcint 6.1.3',
        'benchmark_directory': str(root), 'build_job': '5736126',
        'unchanged_library': 'libsolvent.so (licensed Fortran sources unavailable)',
        'summary_sha256': hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),
        'source_archives': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (root/'sources').glob('*.tar.gz')},
    }
    (stage/'pyscf/HPC_BUILD.json').write_text(json.dumps(info, indent=2)+'\n')
    (stage/'pyscf-2.14.0.dist-info/WHEEL').write_text(
        'Wheel-Version: 1.0\nGenerator: caltech-pyscf-native-rebuild\n'
        'Root-Is-Purelib: false\nTag: py3-none-linux_x86_64\n')
    build_tag = '1caltechavx2'

(root/'wheels').mkdir(exist_ok=True)
command = [sys.executable, '-m', 'wheel', 'pack', str(stage), '-d', str(root/'wheels')]
if build_tag:
    command += ['--build-number', build_tag]
subprocess.run(command, check=True)
