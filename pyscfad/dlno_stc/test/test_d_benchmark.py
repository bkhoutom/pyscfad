"""Benchmark output reuse must not mix different density selections."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pyscfad.dlno_stc import BoysDomainThresholds


@pytest.mark.parametrize('saved, requested', [(None, 'D'), ('W', 'D'), ('D', 'W')])
def test_benchmark_rejects_output_from_other_density_before_scf(tmp_path, saved, requested):
    if not hasattr(os, 'sched_getaffinity'):
        pytest.skip('benchmark requires Linux CPU affinity')
    cpu = min(os.sched_getaffinity(0))
    settings = dict(domain_thresholds=asdict(BoysDomainThresholds()),
                    scf_jk='df', lis_eigenspace='domain', lis_space='domain',
                    triples_variance_allocation='occupied_count', smooth_frame_mode='full',
                    cc_conv_tol=1e-7, cc_conv_tol_normt=1e-5,
                    scf_jk_workspace_memory_mb=4000., lno_thresh_occ=1e-4,
                    lno_thresh_vir=1e-5, triples_mode='smooth', full_domain=True,
                    cpu_affinity=[cpu], threads=1)
    if saved is not None:
        settings['matrix_kind'] = saved
    path = tmp_path / 'settings.json'
    original = json.dumps(settings)
    path.write_text(original)
    repo = Path(__file__).resolve().parents[3]
    result = subprocess.run([
        sys.executable, str(repo / 'examples/dlno_stc/benchmark_lno_w_triples.py'),
        '--threads', '1', '--scf-jk', 'df', '--xyz-file', str(tmp_path / 'unused.xyz'),
        '--basis', 'sto-3g', '--frozen', '4', '--boys-grouping', 'system',
        '--output-dir', str(tmp_path), '--matrix-kind', requested,
    ], cwd=repo, env=dict(os.environ, PYSCFAD_BENCHMARK_CPU_LIST=str(cpu)),
        capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert 'matrix_kind differs' in result.stderr
    assert path.read_text() == original
