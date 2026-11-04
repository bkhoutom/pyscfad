"""Scoped DF GEMM parallelism preserves samples and every native cotangent."""
import ctypes
import os

import numpy as np
import pytest

from pyscfad.dlno_stc.test.test_triples_asymmetric import binding_inputs
from pyscfad.dlno_stc.test.test_triples import triples_native


def controlled_native():
    native = triples_native()
    if not getattr(native, 'TRIPLES_OPENBLAS_THREAD_CONTROL', False):
        if os.environ.get('PYSCFAD_REQUIRE_STC_OPENBLAS_THREAD_CONTROL'):
            pytest.fail('Build with PYSCFAD_STC_OPENBLAS_THREAD_CONTROL=ON')
        pytest.skip('optional pthread OpenBLAS thread control is not built')
    runtime = ctypes.CDLL(native.__file__)
    runtime.openblas_get_num_threads.restype = ctypes.c_int
    runtime.openblas_get_num_threads.argtypes = []
    runtime.openblas_set_num_threads.argtypes = [ctypes.c_int]
    runtime.openblas_set_num_threads.restype = None
    runtime.omp_get_max_threads.restype = ctypes.c_int
    runtime.omp_get_max_threads.argtypes = []
    runtime.omp_set_num_threads.argtypes = [ctypes.c_int]
    runtime.omp_set_num_threads.restype = None
    return native, runtime


@pytest.mark.parametrize('samples', [0, 16385])
def test_dense_blas_threads_preserve_energy_and_every_cotangent(monkeypatch, samples):
    native, runtime = controlled_native()
    previous_blas = runtime.openblas_get_num_threads()
    previous_omp = runtime.omp_get_max_threads()
    try:
        runtime.openblas_set_num_threads(1)
        runtime.omp_set_num_threads(8)
        inputs = binding_inputs()
        options = dict(nsamples=samples, seed=19)
        monkeypatch.setenv('PYSCFAD_STC_DENSE_BLAS_THREADS', '1')
        reference = native.triples_asymmetric_energy_vjp(**inputs, **options)
        monkeypatch.setenv('PYSCFAD_STC_DENSE_BLAS_THREADS', '8')
        result = native.triples_asymmetric_energy_vjp(**inputs, **options)
        assert runtime.openblas_get_num_threads() == 1
        assert runtime.omp_get_max_threads() == 8
        for entry, expected in ((reference, 1), (result, 8)):
            assert entry['setup_threads'] == entry['sampling_threads'] == 8
            assert entry['packing_blas_threads'] == entry['finalize_blas_threads'] == expected
            assert entry['sampling_blas_threads'] == 1
        for name in ('energy', 'standard_error'):
            np.testing.assert_allclose(result[name], reference[name], rtol=2e-12, atol=2e-15)
        assert result['samples'] == reference['samples'] == samples
        assert result['pilot_samples'] == reference['pilot_samples']
        assert set(result['cotangents']) == set(reference['cotangents']) == set(inputs)
        for name in inputs:
            np.testing.assert_allclose(result['cotangents'][name], reference['cotangents'][name],
                                       rtol=2e-12, atol=2e-14, err_msg=name)
        # A following unconfigured call sees the restored embedding policy.
        monkeypatch.delenv('PYSCFAD_STC_DENSE_BLAS_THREADS')
        restored = native.triples_asymmetric_energy_vjp(**inputs, **options)
        assert restored['packing_blas_threads'] == restored['finalize_blas_threads'] == 1
        assert restored['sampling_blas_threads'] == 1
    finally:
        runtime.openblas_set_num_threads(previous_blas)
        runtime.omp_set_num_threads(previous_omp)


@pytest.mark.parametrize('value', ['', '0', '-1', '1.5', 'eight'])
def test_invalid_dense_blas_count_fails_clearly(monkeypatch, value):
    native = triples_native()
    if not hasattr(native, 'TRIPLES_OPENBLAS_THREAD_CONTROL'):
        pytest.skip('extension predates scoped BLAS support')
    monkeypatch.setenv('PYSCFAD_STC_DENSE_BLAS_THREADS', value)
    with pytest.raises(ValueError, match='positive integer'):
        native.triples_asymmetric_energy_vjp(**binding_inputs(), nsamples=2)


def test_explicit_dense_blas_control_requires_supported_build(monkeypatch):
    native = triples_native()
    if not hasattr(native, 'TRIPLES_OPENBLAS_THREAD_CONTROL'):
        pytest.skip('extension predates scoped BLAS support')
    if native.TRIPLES_OPENBLAS_THREAD_CONTROL:
        pytest.skip('this extension includes the optional feature')
    monkeypatch.setenv('PYSCFAD_STC_DENSE_BLAS_THREADS', '8')
    with pytest.raises(RuntimeError, match='PYSCFAD_STC_OPENBLAS_THREAD_CONTROL=ON'):
        native.triples_asymmetric_energy_vjp(**binding_inputs(), nsamples=2)
