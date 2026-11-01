"""Expected-failure check for the new packed-adjoint API before implementation."""
import json
import os
from pathlib import Path
import pytest

status = pytest.main(['-q', 'test_reverse_packed_density.py', '--maxfail=2',
                      '--tb=short', '-k', 'pack_symmetric'])
Path(f"red-status.{os.environ['SLURM_JOB_ID']}.json").write_text(
    json.dumps({'pytest_exit_code': int(status), 'expected': 1}) + '\n')
assert status == 1, f'Expected missing-interface test failures, got {status}'
