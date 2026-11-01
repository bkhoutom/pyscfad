"""Check that current code exposes auxiliary-slice and repeated-transform costs."""
import os
from pathlib import Path
import pyscfad
import pytest

assert Path(pyscfad.__file__).resolve().is_relative_to(Path('code').resolve())
print('Package under test:', pyscfad.__file__, flush=True)
raise SystemExit(pytest.main([
    '-q', 'tests/test_reverse_native_slices.py', 'tests/test_reverse_saved_lov.py',
    f'--junitxml=results.{os.environ["SLURM_JOB_ID"]}.xml',
]))
