"""Domain-Local Natural Orbitals (DLNO) for pyscfad.

The user-facing entry points are the solver subclasses
:class:`pyscfad.dlno.ccsd.DLNOCCSD` and :class:`pyscfad.dlno.mp2.DLNOMP2`,
which extend their plain LNO counterparts with the prescreen attributes,
and the prescreen orchestration in :mod:`pyscfad.dlno.prescreen`
(``build_dlno_prescreen_data`` / ``rebuild_dlno_prescreen_data``).  The
low-level primitives (PAO construction, domain selection, multipole pair
energies, etc.) live in the sibling modules ``dlno``, ``domain``, ``mp2``,
``multipole``, ``pao``, ``util``.

The solver subclasses are not re-exported here because
:mod:`pyscfad.lno.lno_base` imports :mod:`pyscfad.dlno.util` during its
own module initialization, which makes top-level imports of the solver
classes from this package circular.  Import the classes from their
submodules instead::

    from pyscfad.dlno.ccsd import DLNOCCSD
    from pyscfad.dlno.mp2 import DLNOMP2
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
