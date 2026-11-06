"""Local input and cotangent exchange for external domain MP2 solvers.

Importing this package does not import MPI or an external solver.
"""

from .protocol import (check_replay, request_fingerprint, validate_inputs,
                       validate_result)
from .exchange import read_input, read_result, write_input, write_result
from ._df_basis import basis_kwargs
from ._selection import (BoysDomainThresholds, DomainGroup, DomainSelections,
                         build_domain_groups, build_domain_selections)

__all__ = [
    "check_replay", "request_fingerprint", "validate_inputs", "validate_result",
    "read_input", "read_result", "write_input", "write_result",
    "kernel", "value_and_grad",
    "BoysDomainThresholds", "DomainSelections", "build_domain_selections",
    "DomainGroup", "build_domain_groups",
]


def kernel(mf, static=None, *, scope="domain", frozen=None, controls, comm=None,
           lo_kwargs=None, merge_identical=True, parallel_scf_jk=True,
           correlation_auxbasis=None):
    """Run domain or system correlation energy with an optional separate fit.

    ``correlation_auxbasis=None`` preserves the existing SCF fitting workflow.
    An explicit basis changes local correlation factors only; HF/Fock keep
    using ``mf.with_df`` and its existing storage policy.
    """
    from .driver import kernel as domain_kernel
    return domain_kernel(mf, static, scope=scope, frozen=frozen,
                         controls=controls, comm=comm, lo_kwargs=lo_kwargs,
                         merge_identical=merge_identical, parallel_scf_jk=parallel_scf_jk,
                         **basis_kwargs(correlation_auxbasis))


def value_and_grad(mol, build_mf, static=None, *, scope="domain", frozen=None,
                   controls, comm=None, include_hf=False, lo_kwargs=None, merge_identical=True,
                   build_static=None, parallel_scf_jk=True, correlation_auxbasis=None):
    """Run coordinate energy and gradient with one differentiated SCF.

    Domain calls accept fixed ``static`` selections or a ``build_static(mf)``
    callback that selects them from the returned SCF reference. The callback
    inherits ``correlation_auxbasis``; standalone selection must use the same
    basis. ``None`` preserves the existing SCF fitting workflow.
    """
    from .driver import value_and_grad as domain_value_and_grad
    return domain_value_and_grad(
        mol, build_mf, static, scope=scope, frozen=frozen, controls=controls,
        comm=comm, include_hf=include_hf, lo_kwargs=lo_kwargs,
        merge_identical=merge_identical, build_static=build_static,
        parallel_scf_jk=parallel_scf_jk, **basis_kwargs(correlation_auxbasis))
