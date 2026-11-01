"""Thread limits for serial dense BLAS operations in molecular reverse passes."""

from contextlib import contextmanager
from contextvars import ContextVar
import operator
import os

from threadpoolctl import ThreadpoolController, threadpool_limits


_BLAS_CONTROLLER = ContextVar('pyscfad_blas_controller', default=None)


@contextmanager
def dense_blas_scope():
    """Discover loaded libraries once for one complete eager invocation.

    This scope changes no thread limits itself. Enter it after the invocation's
    BLAS libraries have loaded; its snapshot excludes libraries loaded later.
    A nested scope discovers a fresh snapshot and restores its parent's on
    exit. Outside every scope, limit helpers discover libraries at each call.

    Limits still affect process-wide state, so concurrent Python calls with
    competing policies remain unsupported.
    """
    token = _BLAS_CONTROLLER.set(ThreadpoolController())
    try:
        yield
    finally:
        _BLAS_CONTROLLER.reset(token)


@contextmanager
def blas_threads(count):
    """Temporarily set a positive integer BLAS limit, leaving OpenMP unchanged.

    Reuse the invocation's library snapshot when ``dense_blas_scope`` is
    active; otherwise discover currently loaded libraries. Nested limits and
    failures restore the previous limits.
    """
    try:
        count = operator.index(count)
        if count < 1:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError('BLAS thread count must be a positive integer') from exc
    controller = _BLAS_CONTROLLER.get()
    limit = threadpool_limits if controller is None else controller.limit
    with limit(limits=count, user_api='blas'):
        yield


@contextmanager
def dense_blas_threads():
    """Temporarily set BLAS threads, leaving OpenMP kernel limits unchanged.

    ``PYSCFAD_DENSE_BLAS_THREADS`` is a positive integer, defaulting to one.
    Set it to the allocated CPU count to parallelize isolated dense operations.
    Never enclose OpenMP kernels or complete molecular reverse passes in this
    context. Limits affect process-wide library state; concurrent Python calls
    with competing policies are unsupported. Nested scopes restore prior limits,
    including when the enclosed operation raises an exception.
    """
    value = os.environ.get('PYSCFAD_DENSE_BLAS_THREADS', '1')
    try:
        count = int(value)
        if count < 1:
            raise ValueError
    except ValueError as exc:
        raise ValueError('PYSCFAD_DENSE_BLAS_THREADS must be a positive integer') from exc
    with blas_threads(count):
        yield
