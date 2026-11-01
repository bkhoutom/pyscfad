"""Dense BLAS scopes preserve numerical results and surrounding runtime settings."""
import numpy as np
import pytest
from scipy.linalg import solve_triangular
from threadpoolctl import threadpool_info, threadpool_limits


def _counts(api):
    return {p['filepath']: p['num_threads'] for p in threadpool_info()
            if p['user_api'] == api and p.get('threading_layer') != 'disabled'}


def test_dense_scope_real_matrix_and_exception_restoration(monkeypatch):
    from pyscfad.lib._threading import dense_blas_threads
    # Import both BLAS users and PySCF OpenMP before inventorying runtime state.
    from pyscf import lib
    lib.num_threads()
    rng = np.random.default_rng(45)
    low = np.tril(rng.normal(size=(128, 128))) + np.eye(128) * 30
    rhs = rng.normal(size=(128, 96))
    with threadpool_limits(limits=1, user_api='blas'):
        expected = solve_triangular(low, rhs, lower=True)
        before = _counts('blas')
        omp_before = _counts('openmp')
        assert before
        monkeypatch.setenv('PYSCFAD_DENSE_BLAS_THREADS', '2')
        with dense_blas_threads():
            assert set(_counts('blas').values()) == {2}
            assert _counts('openmp') == omp_before
            actual = solve_triangular(low, rhs, lower=True)
            product = low @ actual
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(product, rhs, rtol=1e-12, atol=1e-12)
        assert _counts('blas') == before
        with pytest.raises(RuntimeError, match='body failure'):
            with dense_blas_threads():
                raise RuntimeError('body failure')
        assert _counts('blas') == before
        assert _counts('openmp') == omp_before


def test_default_and_nested_scopes_restore_previous_counts(monkeypatch):
    from pyscfad.lib._threading import dense_blas_threads
    monkeypatch.delenv('PYSCFAD_DENSE_BLAS_THREADS', raising=False)
    with threadpool_limits(limits=2, user_api='blas'):
        with dense_blas_threads():
            assert set(_counts('blas').values()) == {1}
            monkeypatch.setenv('PYSCFAD_DENSE_BLAS_THREADS', '2')
            with dense_blas_threads():
                assert set(_counts('blas').values()) == {2}
            assert set(_counts('blas').values()) == {1}
        assert set(_counts('blas').values()) == {2}


@pytest.mark.parametrize('value', ['0', '-1', 'abc', '1.5'])
def test_invalid_thread_budget_rejected_before_body(monkeypatch, value):
    from pyscfad.lib._threading import dense_blas_threads
    monkeypatch.setenv('PYSCFAD_DENSE_BLAS_THREADS', value)
    with pytest.raises(ValueError, match='PYSCFAD_DENSE_BLAS_THREADS'):
        with dense_blas_threads():
            pytest.fail('invalid count entered body')
