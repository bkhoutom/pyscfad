"""Thread limits for serial dense BLAS operations in molecular reverse passes."""

from contextlib import contextmanager
import os

from threadpoolctl import threadpool_limits


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
    # Discover loaded pools at each entry, including lazily imported BLAS users.
    with threadpool_limits(limits=count, user_api='blas'):
        yield
