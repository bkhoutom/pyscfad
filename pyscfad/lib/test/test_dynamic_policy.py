"""Native kernels inherit policy; DF-J/SGX-K select fixed teams in Python."""
from pathlib import Path
import re

import pytest


def test_no_library_dynamic_policy_setters():
    # Native code must inherit its calling scope's policy. The explicit
    # DF-J/SGX-K exception lives in pyscfad.sgx._threading and restores every
    # runtime on exit; the general threading helpers still preserve policy.
    root = Path(__file__).resolve().parents[3]
    if not (root / 'pyscfadlib/pyscfadlib/sgx').is_dir():
        pytest.skip('repository sources required for the native policy audit')
    violations = []
    for package in (root / 'pyscfad', root / 'pyscfadlib/pyscfadlib'):
        for path in package.rglob('*'):
            if path.suffix not in ('.c', '.C', '.cpp', '.h', '.inc'):
                continue
            if any(part.startswith('test') for part in path.relative_to(package).parts):
                continue
            source = re.sub(r'/\*.*?\*/|//[^\n]*', '', path.read_text(), flags=re.S)
            if re.search(r'\bomp_set_dynamic\s*\(', source):
                violations.append(str(path.relative_to(root)))
    assert not violations, f'Library overrides caller OpenMP policy: {violations}'
