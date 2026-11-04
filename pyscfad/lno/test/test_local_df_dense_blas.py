"""Local metric response threads must restore BLAS before native OpenMP work."""
import numpy
import pytest
from threadpoolctl import threadpool_info

from pyscfad.lib._threading import blas_threads
from pyscfad.lno import _df_direct
from pyscfad.lno.test.test_local_df_direct import (
    _assert_tree_allclose, _direct_local_problem,
)


def _blas_counts():
    return [pool['num_threads'] for pool in threadpool_info()
            if pool['user_api'] == 'blas'
            and pool.get('threading_layer') != 'disabled']


def test_metric_threads_preserve_all_bars_and_restore_before_native(monkeypatch):
    mol, local, aux, coeff, slices, fitted = _direct_local_problem()
    if not _blas_counts():
        pytest.skip('No threadpoolctl-managed BLAS provider is loaded')
    seed = numpy.random.default_rng(915).normal(size=fitted.shape)
    args = (mol.max_memory, slices, (local, aux, coeff, fitted), seed)
    solve = _df_direct.scipy_linalg.solve_triangular
    dot = _df_direct.numpy.dot
    native = _df_direct._local_direct_mo_coeff_vjp
    observed = []

    def checked_solve(*args, **kwargs):
        if args[0].shape == (aux.nao, aux.nao) and args[1].shape == fitted.shape:
            observed.append(('solve', _blas_counts()))
        return solve(*args, **kwargs)

    def checked_dot(*args, **kwargs):
        if args[0].shape == fitted.shape and args[1].shape == (fitted.shape[1], aux.nao):
            observed.append(('metric_dot', _blas_counts()))
        return dot(*args, **kwargs)

    def checked_native(*args, **kwargs):
        observed.append(('native', _blas_counts()))
        return native(*args, **kwargs)

    with blas_threads(1):
        monkeypatch.setenv('PYSCFAD_DENSE_BLAS_THREADS', '1')
        reference = _df_direct._local_direct_nr_e2_bwd(*args)
        monkeypatch.setenv('PYSCFAD_DENSE_BLAS_THREADS', '4')
        monkeypatch.setattr(_df_direct.scipy_linalg, 'solve_triangular', checked_solve)
        monkeypatch.setattr(_df_direct.numpy, 'dot', checked_dot)
        monkeypatch.setattr(_df_direct, '_local_direct_mo_coeff_vjp', checked_native)
        actual = _df_direct._local_direct_nr_e2_bwd(*args)
        assert _blas_counts() and all(count == 1 for count in _blas_counts())
    _assert_tree_allclose(actual, reference, rtol=2e-10, atol=2e-10)
    for phase, expected in [('solve', 4), ('metric_dot', 4), ('native', 1)]:
        counts = [values for name, values in observed if name == phase]
        assert len(counts) == 1
        assert counts[0] and all(value == expected for value in counts[0])


def test_metric_threads_restore_after_solve_failure(monkeypatch):
    mol, local, aux, coeff, slices, fitted = _direct_local_problem()
    if not _blas_counts():
        pytest.skip('No threadpoolctl-managed BLAS provider is loaded')
    def fail(*args, **kwargs):
        assert all(count == 4 for count in _blas_counts())
        raise RuntimeError('injected metric solve failure')
    monkeypatch.setenv('PYSCFAD_DENSE_BLAS_THREADS', '4')
    monkeypatch.setattr(_df_direct.scipy_linalg, 'solve_triangular', fail)
    with blas_threads(1):
        with pytest.raises(RuntimeError, match='injected metric solve failure'):
            _df_direct._local_direct_nr_e2_bwd(
                mol.max_memory, slices, (local, aux, coeff, fitted),
                numpy.ones_like(fitted))
        assert _blas_counts() and all(count == 1 for count in _blas_counts())
