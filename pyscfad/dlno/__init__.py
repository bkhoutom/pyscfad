"""Domain-local correlation with IAO-defined fragments.

The current MP2 entry point is
:class:`pyscfad.dlno.iao_mp2.IAOFragmentMP2`.  It evaluates all strong
extended-domain MP2 rows and every unordered weak multipole pair on a fixed
discrete topology.  :class:`pyscfad.dlno.ccsd.DLNOCCSD` uses those same
strong domains to build MP2 selection densities and local interacting spaces
(LISs), then evaluates the unique corrected correlation energy

``sum_F [CCSD(T)_F(LIS_F) - MP2_F(LIS_F)] + IAO-DLNO-MP2``.

The LIS subtraction is conventional full-spin MP2; the final local-MP2 term
contains both strong and weak contributions and is added exactly once.  There
is no spin-opposite-scaled or domain-only correction switch.

The fixed-rank LIS construction is exposed by :mod:`pyscfad.dlno.iao_lis`.
The gauge-safe MPI implementation of the MP2 energy and gradient lives in
:mod:`pyscfad.dlno.iao_mp2_mpi`.  The Option A fragment-parallel CCSD(T)
energy-and-gradient driver lives in :mod:`pyscfad.dlno.ccsd_mpi`.  Every rank
closes an entire fragment LIS/CC pullback locally before reducing cotangents,
so no gauge-dependent LIS frame is communicated.  Fragment-owning ranks must
provide a real, accessible density-fitting CDERI source.

The solver subclasses are not re-exported here because
:mod:`pyscfad.lno.lno_base` imports :mod:`pyscfad.dlno.util` during its
own module initialization, which makes top-level imports of the solver
classes from this package circular.  Import the classes from their
submodules instead::

    from pyscfad.dlno.ccsd import DLNOCCSD
    from pyscfad.dlno.ccsd_mpi import DLNOCCSD as MPIDLNOCCSD
    from pyscfad.dlno.iao_mp2 import IAOFragmentMP2
    from pyscfad.dlno.iao_mp2_mpi import IAOFragmentMP2 as MPIIAOFragmentMP2
"""
from .util import (
    project_mo,
    orthogonalize,
    ao_index_by_atom,
    shell_index_by_atom,
    fake_mol_by_atom,
    unique,
    list_to_array,
    einsum,
)

__all__ = [
    "project_mo",
    "orthogonalize",
    "ao_index_by_atom",
    "shell_index_by_atom",
    "fake_mol_by_atom",
    "unique",
    "list_to_array",
    "einsum",
]
