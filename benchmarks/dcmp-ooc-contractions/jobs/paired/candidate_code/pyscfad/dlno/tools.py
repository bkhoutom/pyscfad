"""Array and orbital helpers used by domain-local correlation.

Projection, atom indexing, and submolecule construction are shared with LNO
and imported from its tools module. Keeping those primitives below DLNO avoids
an import cycle between the base LNO solvers and this extension.
"""

from functools import reduce

import jax
import jax.numpy as jnp
import numpy as np

from pyscfad.lno.tools import (
    _is_traced,
    project_mo,
    ao_index_by_atom,
    fake_mol_by_atom,
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


def orthogonalize(mo1, mo2, s):
    """Project ``mo1`` out of ``mo2``.

    Notes
    -----
    ``mo1`` must be orthonormal with respect to ``s``.

    Dispatches to numpy on concrete inputs to keep eager prescreen builds
    out of the JAX compile pipeline.
    """
    if _is_traced(mo1, mo2, s):
        s = jnp.asarray(s)
        mo1 = jnp.asarray(mo1)
        mo2 = jnp.asarray(mo2)
        s12 = mo1.conj().T @ s @ mo2
        return mo2 - mo1 @ s12
    s_np = np.asarray(s)
    mo1_np = np.asarray(mo1)
    mo2_np = np.asarray(mo2)
    s12 = mo1_np.conj().T @ s_np @ mo2_np
    return mo2_np - mo1_np @ s12


def shell_index_by_atom(mol, atmlst):
    shlslices = mol.aoslice_by_atom()[:, :2]
    shls_lst = map(lambda x: np.arange(*x), shlslices[atmlst].reshape(-1, 2))
    shls = reduce(np.union1d, shls_lst)
    return shls


def unique(a):
    unique_arr = {}
    for i, arr in enumerate(a):
        arr_tuple = tuple(arr)
        if arr_tuple not in unique_arr:
            unique_arr[arr_tuple] = [i]
        else:
            unique_arr[arr_tuple].append(i)
    return unique_arr


def list_to_array(a):
    out = np.empty(len(a), dtype=object)
    out[:] = a
    return out


def einsum(expr, *args):
    for a in args:
        if isinstance(a, jax.core.Tracer):
            return jnp.einsum(expr, *args)
    return np.einsum(expr, *[np.asarray(a) for a in args])
