"""Domain MP2 energies and forces, with fragment DLNO-CCSD/CCSD(T) in ``ccsd``.

Importing this package does not import MPI or an external solver.
"""

from ._df_basis import basis_kwargs
from ._selection import (BoysDomainThresholds, DomainGroup, DomainSelections,
                         build_domain_groups, build_domain_selections)

__all__ = [
    "kernel", "value_and_grad",
    "BoysDomainThresholds", "DomainSelections", "build_domain_selections",
    "DomainGroup", "build_domain_groups",
    "materialize_domain_selections",
]


def materialize_domain_selections(mf, static, *, common=None):
    """Resolve all ED labels for inspection; calculations prepare one at a time."""
    from ._ed_selection import materialize_domain_selections as materialize
    return materialize(mf, static, common=common)


def kernel(mf, static=None, *, boys_grouping=None, frozen=None, controls, comm=None,
           lo_kwargs=None, merge_identical=True, parallel_scf_jk=True,
           correlation_auxbasis=None, mp2_aux_svd_tol=1e-3):
    """Run domain or system correlation energy with an optional separate fit.

    ``correlation_auxbasis=None`` preserves the existing SCF fitting workflow.
    An explicit basis changes local correlation factors only; HF/Fock keep
    using ``mf.with_df`` and its existing storage policy.
    ``mp2_aux_svd_tol=1e-3`` compresses ED OV factors; ``None`` or zero disables it.
    ``boys_grouping="system"`` selects one full-system fragment and uses the domain
    executor. ``"atom"`` and ``"singleton"`` also select their fragments
    automatically. With ``static``, omit ``boys_grouping`` and use its saved
    settings. For explicit domain options use ``build_domain_selections``.
    """
    from .driver import kernel as domain_kernel
    return domain_kernel(mf, static, boys_grouping=boys_grouping, frozen=frozen,
                         controls=controls, comm=comm, lo_kwargs=lo_kwargs,
                         merge_identical=merge_identical, parallel_scf_jk=parallel_scf_jk,
                         mp2_aux_svd_tol=mp2_aux_svd_tol,
                         **basis_kwargs(correlation_auxbasis))


def value_and_grad(mol, build_mf, static=None, *, boys_grouping=None, frozen=None,
                   controls, comm=None, include_hf=False, lo_kwargs=None, merge_identical=True,
                   build_static=None, parallel_scf_jk=True, correlation_auxbasis=None,
                   mp2_aux_svd_tol=1e-3):
    """Run coordinate energy and gradient with one differentiated SCF.

    ``mp2_aux_svd_tol=1e-3`` includes the ED NAF subspace response at fixed rank;
    ``None`` or zero disables compression. No extra ED integral pass is needed.
    Domain calls accept fixed ``static`` selections or a ``build_static(mf)``
    callback that selects them from the returned SCF reference. The callback
    inherits ``correlation_auxbasis``; standalone selection must use the same
    basis. ``None`` preserves the existing SCF fitting workflow.
    Alternatively, explicit ``boys_grouping`` selects fragments automatically;
    ``"system"`` uses one full-system fragment. Do not combine it with
    ``static`` or ``build_static``.
    """
    from .driver import value_and_grad as domain_value_and_grad
    return domain_value_and_grad(
        mol, build_mf, static, boys_grouping=boys_grouping, frozen=frozen, controls=controls,
        comm=comm, include_hf=include_hf, lo_kwargs=lo_kwargs,
        merge_identical=merge_identical, build_static=build_static,
        mp2_aux_svd_tol=mp2_aux_svd_tol,
        parallel_scf_jk=parallel_scf_jk, **basis_kwargs(correlation_auxbasis))
