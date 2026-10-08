"""Local input and cotangent exchange for external domain MP2 solvers.

Importing this package does not import MPI or an external solver.
"""

from .protocol import (check_replay, request_fingerprint, validate_inputs,
                       validate_result)
from .exchange import read_input, read_result, write_input, write_result

__all__ = [
    "check_replay", "request_fingerprint", "validate_inputs", "validate_result",
    "read_input", "read_result", "write_input", "write_result",
    "kernel", "value_and_grad",
]


def kernel(mf, static=None, *, scope="domain", frozen=None, controls, comm=None,
           lo_kwargs=None):
    """Run in-memory domain or whole-system STC correlation energy."""
    from .driver import kernel as weighted_kernel
    return weighted_kernel(mf, static, scope=scope, frozen=frozen,
                           controls=controls, comm=comm, lo_kwargs=lo_kwargs)


def value_and_grad(mol, build_mf, static=None, *, scope="domain", frozen=None,
                   controls, comm=None, include_hf=False, lo_kwargs=None):
    """Run domain or whole-system coordinate energy and gradient."""
    from .driver import value_and_grad as weighted_value_and_grad
    return weighted_value_and_grad(mol, build_mf, static, scope=scope, frozen=frozen,
                                   controls=controls,
                                   comm=comm, include_hf=include_hf, lo_kwargs=lo_kwargs)
