"""DF-J/SGX-K fix teams locally across independently linked OpenMP runtimes."""
import ctypes

import numpy as np
import pytest
from pyscf import gto, lib, scf
from threadpoolctl import ThreadpoolController

from pyscfad.sgx import _dfj, _native, _sgx_grad, sgx_fit
from pyscfad.sgx._threading import fixed_openmp_teams


@pytest.fixture
def runtimes():
    _native.library()
    result = []
    for pool in ThreadpoolController().lib_controllers:
        if pool.user_api == 'openmp':
            dll = pool.dynlib
            dll.omp_get_dynamic.restype = ctypes.c_int
            dll.omp_set_dynamic.argtypes = [ctypes.c_int]
            result.append(dll)
    saved = [dll.omp_get_dynamic() for dll in result]
    try:
        yield result
    finally:
        for dll, value in zip(result, saved):
            dll.omp_set_dynamic(value)


def flags(runtimes):
    return [dll.omp_get_dynamic() for dll in runtimes]


def test_nested_scope_restores_individual_runtime_flags_on_exception(runtimes):
    for i, dll in enumerate(runtimes):
        dll.omp_set_dynamic(i % 2)
    before = flags(runtimes)
    with lib.with_omp_threads(3):
        pools = ThreadpoolController().info()
        with pytest.raises(RuntimeError, match='intentional'):
            with fixed_openmp_teams():
                assert flags(runtimes) == [0] * len(runtimes)
                with fixed_openmp_teams():
                    assert flags(runtimes) == [0] * len(runtimes)
                assert flags(runtimes) == [0] * len(runtimes)
                raise RuntimeError('intentional')
        assert flags(runtimes) == before
        assert ThreadpoolController().info() == pools


@pytest.mark.parametrize('backend', ['libcint', 'generated', 'mixed'])
def test_jk_kernel_policy_and_numerics(runtimes, monkeypatch, backend):
    if backend != 'libcint' and _native.library() is None:
        pytest.skip('generated SGX library is not installed')
    mol = gto.M(atom='O 0 0 0; H 0 -.75 .57; H 0 .75 .57',
                basis='sto-3g', verbose=0)
    mf = sgx_fit(scf.RHF(mol), grid_level=0, grid_sequence=False,
                 integral_backend='libcint' if backend == 'libcint' else 'generated',
                 mixed_precision=backend == 'mixed',
                 screening='per_pair' if backend == 'mixed' else 'static')
    dm = mf.get_init_guess()
    obj = mf.with_df
    calls = []
    original_j = _dfj.df_jk.get_j
    original_front = _sgx_grad._front_contract

    def checked(function, name):
        def call(*args, **kwargs):
            assert flags(runtimes) == [0] * len(runtimes)
            assert lib.num_threads() == 2
            calls.append(name)
            return function(*args, **kwargs)
        return call

    monkeypatch.setattr(_dfj.df_jk, 'get_j', checked(original_j, 'J'))
    monkeypatch.setattr(_sgx_grad, '_front_contract', checked(original_front, 'K'))
    results = []
    with lib.with_omp_threads(2):
        obj.build()
        for dynamic in (0, 1):
            for dll in runtimes:
                dll.omp_set_dynamic(dynamic)
            results.append(obj.get_jk(dm))
            assert flags(runtimes) == [dynamic] * len(runtimes)
            assert lib.num_threads() == 2
        assert {'J', 'K'} <= set(calls)
        for left, right in zip(*results):
            np.testing.assert_allclose(left, right, rtol=0, atol=2e-12)

        def fail(*args, **kwargs):
            assert flags(runtimes) == [0] * len(runtimes)
            raise RuntimeError('kernel failure')

        monkeypatch.setattr(_dfj.df_jk, 'get_j', fail)
        with pytest.raises(RuntimeError, match='kernel failure'):
            _dfj.get_j(obj, dm)
        assert flags(runtimes) == [1] * len(runtimes)
        monkeypatch.setattr(_sgx_grad, '_front_contract', fail)
        with pytest.raises(RuntimeError, match='kernel failure'):
            _sgx_grad.get_k(obj, dm, 1, 1e-13)
        assert flags(runtimes) == [1] * len(runtimes)
