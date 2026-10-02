"""Local input and cotangent exchange for external domain MP2 solvers.

Importing this package does not import MPI or an external solver.
"""

from .protocol import (check_replay, request_fingerprint, validate_inputs,
                       validate_result)
from .exchange import read_input, read_result, write_input, write_result

__all__ = [
    "check_replay", "request_fingerprint", "validate_inputs", "validate_result",
    "read_input", "read_result", "write_input", "write_result",
]
