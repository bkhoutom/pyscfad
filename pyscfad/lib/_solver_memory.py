"""Runtime reservations for eager iterative-solver workspace.

RSS does not include untouched pages of a newly allocated Krylov basis.
Integral/grid planners must therefore reserve its eventual storage as well.
The reservation is deliberately conservative: resident solver arrays can be
counted again, rather than subtracting unrelated allocator/cache growth.
"""
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np


_active = ContextVar('pyscfad_solver_memory', default=())


def gmres_workspace(n, dtype, restart=None, preconditioned=False):
    """Plan SciPy GMRES storage from the actual flattened RHS and options.

    Basis and small-array sizes follow SciPy's restarted GMRES. Working
    vectors include the RHS, solution, residual, overlapping old/new matvec
    outputs, the writable host-result copy and a vector-expression temporary.
    A separate preconditioned output is allowed when M is supplied. Operator
    internals and preconditioner factors belong to their own memory planners.
    """
    n = int(n)
    restart = min(n, 20 if restart is None else int(restart))
    if n <= 0 or restart <= 0:
        raise ValueError('GMRES dimension and restart must be positive')
    dtype = np.dtype(dtype)
    vector_bytes = n * dtype.itemsize
    basis_bytes = (restart + 1) * vector_bytes
    working_vectors = 7 + int(preconditioned)
    working_bytes = working_vectors * vector_bytes
    # Hessenberg matrix, Givens rotations, residual coefficients and y.
    small_bytes = (restart*(restart+1) + 2*restart + restart+1 + restart)*dtype.itemsize
    return dict(dimension=n, dtype=dtype.name, restart=restart,
                vector_bytes=vector_bytes, basis_bytes=basis_bytes,
                working_vectors=working_vectors, working_bytes=working_bytes,
                small_array_bytes=small_bytes,
                reserved_bytes=basis_bytes+working_bytes+small_bytes)


@contextmanager
def reserve_gmres(n, dtype, restart=None, preconditioned=False):
    plan = gmres_workspace(n, dtype, restart, preconditioned)
    token = _active.set(_active.get() + (plan,))
    try:
        yield plan
    finally:
        _active.reset(token)


def active_plans():
    return tuple(dict(plan) for plan in _active.get())


def reserved_bytes():
    return sum(plan['reserved_bytes'] for plan in _active.get())
