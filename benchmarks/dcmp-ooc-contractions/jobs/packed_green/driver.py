"""Check packed adjoints against independent dense derivatives."""
import os
from pathlib import Path
import pyscfad
import pytest

assert Path(pyscfad.__file__).resolve().is_relative_to(Path('code').resolve())
print('Package under test:', pyscfad.__file__, flush=True)
raise SystemExit(pytest.main([
    '-q', 'code/pyscfad/df/test/test_reverse_packed_density.py',
    f'--junitxml=results.{os.environ["SLURM_JOB_ID"]}.xml',
]))
