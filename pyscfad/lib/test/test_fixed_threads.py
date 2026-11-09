"""Real runtime budgets and dynamic flags at eager compute boundaries."""
import ctypes

import numpy as np
import pytest
from pyscf import lib
from pyscf.scf import _vhf
from threadpoolctl import ThreadpoolController

from pyscfad.lib._threading import blas_threads, fixed_threads


def runtime():
    omp = _vhf.libcvhf
    omp.omp_get_dynamic.restype = ctypes.c_int
    omp.omp_set_dynamic.argtypes = [ctypes.c_int]
    return omp


@pytest.mark.parametrize('count', [1, 2, 3])
def test_configured_count_dynamic_flags_and_exception_restoration(count):
    omp = runtime()
    dynamic = omp.omp_get_dynamic()
    pools = ThreadpoolController()
    try:
        omp.omp_set_dynamic(1)
        with lib.with_omp_threads(count):
            before = pools.info()
            with pytest.raises(RuntimeError, match='deliberate failure'):
                with fixed_threads():
                    assert lib.num_threads() == count
                    assert omp.omp_get_dynamic() == 0
                    assert {p['num_threads'] for p in pools.info()
                            if p.get('threading_layer') != 'disabled'} == {count}
                    # Deliberately serial BLAS inside a native OpenMP kernel
                    # must not be widened by a nested fixed-team scope.
                    with blas_threads(1), fixed_threads():
                        assert lib.num_threads() == count
                        assert all(p['num_threads'] == 1 for p in pools.info()
                                   if p['user_api'] == 'blas')
                    raise RuntimeError('deliberate failure')
            assert omp.omp_get_dynamic() == 1
            assert pools.info() == before
    finally:
        omp.omp_set_dynamic(dynamic)


def test_eager_gmres_uses_configured_threads_and_restores_on_matvec_failure():
    from pyscfad._src.scipy.sparse.linalg import gmres
    omp = runtime()
    previous = omp.omp_get_dynamic()
    calls = []
    def matvec(x):
        calls.append((lib.num_threads(), omp.omp_get_dynamic()))
        return np.array([2., 3., 4.])*x
    try:
        omp.omp_set_dynamic(1)
        with lib.with_omp_threads(2):
            x, info = gmres(matvec, np.ones(3), tol=1e-12, atol=1e-14)
            assert info == 0
            np.testing.assert_allclose(x, 1/np.array([2., 3., 4.]), atol=1e-12)
            assert calls and set(calls) == {(2, 0)}
            assert omp.omp_get_dynamic() == 1
            def fail(x):
                assert omp.omp_get_dynamic() == 0
                raise RuntimeError('matvec failure')
            with pytest.raises(RuntimeError, match='matvec failure'):
                gmres(fail, np.ones(3), atol=1e-14)
            assert omp.omp_get_dynamic() == 1
    finally:
        omp.omp_set_dynamic(previous)
