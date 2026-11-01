"""Invocation-local discovery retains real BLAS limits and fresh fallbacks."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from threadpoolctl import ThreadpoolController, threadpool_limits

from pyscfad.lib import _threading


def _counts(controller, api):
    return {pool['filepath']: pool['num_threads'] for pool in controller.info()
            if pool['user_api'] == api and pool.get('threading_layer') != 'disabled'}


def test_invocation_reuses_discovery_but_reads_each_dynamic_budget(monkeypatch):
    assert hasattr(_threading, 'dense_blas_scope'), 'invocation scope missing'
    observed = ThreadpoolController()
    discoveries = []
    original = ThreadpoolController._load_libraries

    def discover(controller):
        discoveries.append(True)
        return original(controller)

    with threadpool_limits(limits=1, user_api='blas'):
        before = _counts(observed, 'blas')
        omp_before = _counts(observed, 'openmp')
        assert before
        monkeypatch.setattr(ThreadpoolController, '_load_libraries', discover)
        with _threading.dense_blas_scope():
            assert _counts(observed, 'blas') == before
            for count in (2, 1, 2):
                monkeypatch.setenv('PYSCFAD_DENSE_BLAS_THREADS', str(count))
                with _threading.dense_blas_threads():
                    assert set(_counts(observed, 'blas').values()) == {count}
                    assert _counts(observed, 'openmp') == omp_before
                    np.testing.assert_array_equal(np.eye(3) @ np.ones(3), np.ones(3))
                assert _counts(observed, 'blas') == before
        assert len(discoveries) == 1
        # Ordinary calls still discover pools separately; the scope left no cache.
        with _threading.dense_blas_threads():
            assert set(_counts(observed, 'blas').values()) == {2}
        with _threading.blas_threads(1):
            assert _counts(observed, 'blas') == before
        assert len(discoveries) == 3
        assert _counts(observed, 'blas') == before


def test_nested_limits_and_invocations_restore_after_exception(monkeypatch):
    assert hasattr(_threading, 'blas_threads'), 'explicit BLAS limit helper missing'
    observed = ThreadpoolController()
    with threadpool_limits(limits=1, user_api='blas'):
        before = _counts(observed, 'blas')
        omp_before = _counts(observed, 'openmp')
        monkeypatch.setenv('PYSCFAD_DENSE_BLAS_THREADS', '2')
        with pytest.raises(RuntimeError, match='outer failure'):
            with _threading.dense_blas_scope():
                with _threading.dense_blas_threads():
                    assert set(_counts(observed, 'blas').values()) == {2}
                    with pytest.raises(RuntimeError, match='inner failure'):
                        with _threading.dense_blas_scope():
                            with _threading.blas_threads(1):
                                assert _counts(observed, 'blas') == before
                                raise RuntimeError('inner failure')
                    assert set(_counts(observed, 'blas').values()) == {2}
                assert _counts(observed, 'blas') == before
                raise RuntimeError('outer failure')
        assert _counts(observed, 'blas') == before
        assert _counts(observed, 'openmp') == omp_before
        with _threading.blas_threads(2):
            assert set(_counts(observed, 'blas').values()) == {2}
        assert _counts(observed, 'blas') == before


@pytest.mark.parametrize('count', [0, -1, 1.5, '2', None])
def test_explicit_limit_rejects_nonpositive_or_noninteger_count(count):
    assert hasattr(_threading, 'blas_threads'), 'explicit BLAS limit helper missing'
    with pytest.raises(ValueError, match='positive integer'):
        with _threading.blas_threads(count):
            pytest.fail('invalid limit entered body')


def test_invalid_dynamic_budget_within_invocation_does_not_change_limits(monkeypatch):
    assert hasattr(_threading, 'dense_blas_scope'), 'invocation scope missing'
    observed = ThreadpoolController()
    before = _counts(observed, 'blas')
    with _threading.dense_blas_scope():
        monkeypatch.setenv('PYSCFAD_DENSE_BLAS_THREADS', '0')
        with pytest.raises(ValueError, match='PYSCFAD_DENSE_BLAS_THREADS'):
            with _threading.dense_blas_threads():
                pytest.fail('invalid budget entered body')
        assert _counts(observed, 'blas') == before


def test_late_library_load_is_excluded_until_nested_scope_or_fresh_fallback():
    assert hasattr(_threading, 'dense_blas_scope'), 'invocation scope missing'
    # A fresh interpreter loads this helper alone, before any BLAS client.
    # Importing NumPy later exercises a real late library load, not a fake pool.
    script = textwrap.dedent('''
        import importlib.util
        import sys
        from threadpoolctl import threadpool_info
        spec = importlib.util.spec_from_file_location('isolated_threading', sys.argv[1])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        def counts():
            return {p['num_threads'] for p in threadpool_info()
                    if p['user_api'] == 'blas' and p.get('threading_layer') != 'disabled'}
        assert not counts(), 'BLAS unexpectedly loaded before isolated test'
        with module.dense_blas_scope():
            import numpy
            assert counts() == {3}
            with module.blas_threads(1):
                assert counts() == {3}, 'old snapshot must exclude late library'
            with module.dense_blas_scope():
                with module.blas_threads(2):
                    assert counts() == {2}
            assert counts() == {3}
            with module.blas_threads(1):
                assert counts() == {3}, 'nested exit must restore outer snapshot'
        with module.blas_threads(1):
            assert counts() == {1}, 'ordinary fallback must discover late library'
        assert counts() == {3}
    ''')
    env = os.environ.copy()
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
        env[name] = '3'
    env['OMP_NUM_THREADS'] = '1'
    env['PYTHONNOUSERSITE'] = '1'
    result = subprocess.run(
        [sys.executable, '-c', script, str(Path(_threading.__file__).resolve())],
        env=env, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
