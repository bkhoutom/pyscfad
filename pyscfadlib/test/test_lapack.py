import pytest

def test_import():
    try:
        from pyscfadlib import lapack as lp
    except ImportError as e:
        raise ImportError(e)

def test_scipy_lapack_backend():
    from pyscfadlib import pyscfad_lapack
    pyscfad_lapack.initialize()
    pyscfad_lapack.initialize()  # Initialization must be idempotent.
    assert pyscfad_lapack.lapack_backend == 'scipy.linalg.cython_lapack'
