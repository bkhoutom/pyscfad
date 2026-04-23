"""Minimal dlno compatibility layer for pyscfad.

This package provides a small subset of the original `dlno` utilities
reimplemented using JAX-friendly operations where it makes sense.
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
