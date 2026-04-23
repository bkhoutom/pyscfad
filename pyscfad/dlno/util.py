from functools import reduce
import numpy as np
import jax.numpy as jnp

from pyscf import lib, gto


def project_mo(mo1, s21, s22):
    """Project ``mo1`` to basis ``2`` using a jax-friendly solver.

    This mirrors the original dlno.util.project_mo but uses ``jnp.linalg.solve``
    so it can participate in JAX tracing where inputs are jax arrays.
    """
    # solve s22 x = s21 @ mo1  => x = solve(s22, s21 @ mo1)
    rhs = jnp.asarray(s21) @ jnp.asarray(mo1)
    return jnp.linalg.solve(jnp.asarray(s22), rhs)


def orthogonalize(mo1, mo2, s):
    """Project ``mo1`` out of ``mo2``.

    Notes
    -----
    ``mo1`` must be orthonormal with respect to ``s``.
    """
    s = jnp.asarray(s)
    mo1 = jnp.asarray(mo1)
    mo2 = jnp.asarray(mo2)
    s12 = mo1.conj().T @ s @ mo2
    mo2 = mo2 - mo1 @ s12
    return mo2


def ao_index_by_atom(mol, atmlst):
    aoslices = mol.aoslice_by_atom()[:, 2:]
    ao_idx_lst = map(lambda x: np.arange(*x), aoslices[atmlst].reshape(-1, 2))
    ao_idx = reduce(np.union1d, ao_idx_lst)
    return ao_idx


def shell_index_by_atom(mol, atmlst):
    shlslices = mol.aoslice_by_atom()[:, :2]
    shls_lst = map(lambda x: np.arange(*x), shlslices[atmlst].reshape(-1, 2))
    shls = reduce(np.union1d, shls_lst)
    return shls


def fake_mol_by_atom(mol, atmlst=None):
    if atmlst is not None:
        fake_mol = mol.copy(deep=False)
        fake_mol._atom = [mol._atom[a] for a in atmlst]
        fake_mol._atm, fake_mol._bas, fake_mol._env = \
            fake_mol.make_env(fake_mol._atom, fake_mol._basis,
                              mol._env[:gto.PTR_ENV_START])
        fake_mol._built = True
    else:
        fake_mol = mol
    return fake_mol


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
    return jnp.einsum(expr, *args)
