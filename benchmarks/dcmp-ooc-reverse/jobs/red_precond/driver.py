"""Regression-first check: current code must expose redundant DF reads."""
import os
from pathlib import Path
import pyscfad
import pytest

assert Path(pyscfad.__file__).resolve().is_relative_to(Path('code').resolve())
print('Package under test:', pyscfad.__file__, flush=True)
raise SystemExit(pytest.main([
    '-q', 'tests/test_reverse_precond_cache.py',
    'code/pyscfad/df/test/test_scf_precond.py',
    f'--junitxml=results.{os.environ["SLURM_JOB_ID"]}.xml',
]))
