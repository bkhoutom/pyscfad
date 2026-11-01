"""Package validated candidates and exact rollback snapshots; never install them."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parent
spec = json.loads((ROOT / 'input.json').read_text())
result = json.loads((ROOT / 'results/summary.json').read_text())
assert result['status'] == 'ok'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


metadata = dict(spec)
metadata.update(build_job=os.environ['SLURM_JOB_ID'],
                compiler=subprocess.check_output(['/usr/bin/gcc', '--version'], text=True).splitlines()[0],
                build_type='Release', validation_summary_sha256=digest(ROOT / 'results/summary.json'),
                intended_environment=spec['environment'], installed=False,
                preserved='Original libcint, libcintad, libxc, xcfun, licensed libsolvent.so; no Python algorithm edits.')
metadata['candidate_native_sha256'] = {name: digest(ROOT / 'candidate/pyscfadlib' / name)
                                       for name in spec['native_files']}
metadata['baseline_native_sha256'] = {name: digest(ROOT / 'baseline/pyscfadlib' / name)
                                      for name in spec['native_files']}
metadata['current_live_native_unchanged'] = {
    name: digest(Path(spec['fork']) / 'pyscfadlib/pyscfadlib' / name) == checksum
    for name, checksum in spec['live_native_sha256'].items()}
assert all(metadata['current_live_native_unchanged'].values())
site = Path(spec['environment']) / 'lib/python3.11/site-packages'
metadata['live_pyscf_unchanged'] = all(
    digest(site / entry) == digest(ROOT / 'baseline' / entry)
    for entry in spec['pyscf_owned_files'])
assert metadata['live_pyscf_unchanged']

for variant, tag, output in [('baseline', '1rollbackmkl', 'rollback'),
                             ('candidate', '2caltechmklseq', 'wheels')]:
    stage = ROOT / 'wheel_staging' / variant
    if stage.exists():
        shutil.rmtree(stage)  # Disposable packaging scratch inside this condition.
    stage.mkdir(parents=True, exist_ok=False)
    for entry in spec['pyscf_owned_files']:
        rel = Path(entry)
        target = stage / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / variant / rel, target)
    if variant == 'candidate':
        (stage / 'pyscf/HPC_BUILD.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (stage / 'pyscf-2.14.0.dist-info/WHEEL').write_text(
        'Wheel-Version: 1.0\nGenerator: isolated-pyscf-mkl-candidate\n'
        'Root-Is-Purelib: false\nTag: py3-none-linux_x86_64\n')
    subprocess.run([sys.executable, '-m', 'wheel', 'pack', str(stage),
                    '-d', str(ROOT / output), '--build-number', tag], check=True)
    wheel_path = ROOT / output / f'pyscf-2.14.0-{tag}-py3-none-linux_x86_64.whl'
    with zipfile.ZipFile(wheel_path) as wheel:
        expected = set(spec['pyscf_owned_files']) | {'pyscf-2.14.0.dist-info/RECORD'}
        assert set(wheel.namelist()) == expected, 'Unexpected wheel file ownership'
        for entry in spec['pyscf_owned_files']:
            if entry.endswith('/WHEEL'):
                continue  # wheel pack records the requested build tag.
            assert hashlib.sha256(wheel.read(entry)).hexdigest() == digest(stage / entry)
    archive_path = ROOT / output / ('pyscfadlib-' + variant + '-cp311-linux_x86_64.tar.gz')
    with tarfile.open(archive_path, 'w:gz', dereference=True) as archive:
        for name in spec['native_files']:
            archive.add(ROOT / variant / 'pyscfadlib' / name, arcname=name)
    with tarfile.open(archive_path) as archive:
        assert sorted(archive.getnames()) == sorted(spec['native_files'])
        for name in spec['native_files']:
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest(
                ROOT / variant / 'pyscfadlib' / name)

artifacts = [p for folder in ('wheels', 'rollback') for p in (ROOT / folder).iterdir() if p.is_file()]
metadata['artifacts'] = {str(p.relative_to(ROOT)): {'sha256': digest(p), 'bytes': p.stat().st_size}
                         for p in sorted(artifacts)}
metadata['install_status'] = 'Prepared only. No environment or editable-source binaries were replaced.'
(ROOT / 'BUILD_METADATA.json').write_text(json.dumps(metadata, indent=2) + '\n')
(ROOT / 'SHA256SUMS').write_text(''.join(value['sha256'] + '  ' + name + '\n'
    for name, value in metadata['artifacts'].items()))
print(json.dumps(metadata['artifacts'], indent=2))
