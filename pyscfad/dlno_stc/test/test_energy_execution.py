"""Energy execution avoids unused H storage and nested BLAS parallelism."""
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from pyscfad.dlno_stc import backend, driver
from pyscfad.dlno_stc.test.test_backend import fixture, native
from pyscfad.dlno_stc.test.test_full_kernel import grid


def _pools(api):
    return {p['filepath']: p['num_threads'] for p in threadpool_info()
            if p['user_api'] == api}


@pytest.mark.parametrize('scope', ['domain', 'system'])
@pytest.mark.parametrize('with_grad', [False, True])
def test_native_energy_blas_budget_and_restoration(monkeypatch, scope, with_grad):
    extension = native()
    name = 'solve_selected' if scope == 'domain' else 'solve_full'
    original = getattr(extension, name)
    seen = []

    def checked(*args, **kwargs):
        seen.append((_pools('blas'), _pools('openmp')))
        return original(*args, **kwargs)

    monkeypatch.setattr(extension, name, checked)
    inputs, controls = fixture(), grid()
    metadata = {'target_indices': (0,)}
    with threadpool_limits(limits=2):
        before = (_pools('blas'), _pools('openmp'))
        assert before[0]
        result = backend._solve_local(inputs, metadata, controls, with_grad=with_grad,
                                      aux_offsets=None, scope=scope)
        assert (_pools('blas'), _pools('openmp')) == before
        arguments = ((0,), controls) if scope == 'domain' else (controls,)
        expected = original(*(inputs[k] for k in ('foo', 'fvv', 'B')), *arguments,
                            with_grad=with_grad)
    assert set(seen[0][0].values()) == {2 if with_grad else 1}
    assert seen[0][1] == before[1]
    np.testing.assert_allclose(result['energy'], expected['energy'], atol=2e-13, rtol=0)
    if with_grad:
        for key in inputs:
            np.testing.assert_allclose(result['cotangents'][key], expected['cotangents'][key],
                                       atol=2e-12, rtol=0)


def test_native_energy_restores_threads_on_failure(monkeypatch):
    extension = native()

    def fail(*args, **kwargs):
        assert set(_pools('blas').values()) == {1}
        raise RuntimeError('native failure')

    monkeypatch.setattr(extension, 'solve_selected', fail)
    with threadpool_limits(limits=2):
        before = (_pools('blas'), _pools('openmp'))
        with pytest.raises(RuntimeError, match='native failure'):
            backend._solve_local(fixture(), {'target_indices': (0,)}, grid(),
                                 with_grad=False, aux_offsets=None)
        assert (_pools('blas'), _pools('openmp')) == before


@pytest.mark.parametrize('with_grad', [False, True])
@pytest.mark.parametrize('override', [None, 'direct', 'half'])
def test_energy_df_default_and_explicit_override(monkeypatch, with_grad, override):
    from pyscfad.lno import _df_shared, _df_half
    monkeypatch.delenv('PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE', raising=False)
    if override is not None:
        monkeypatch.setenv('PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE', override)
    shared = SimpleNamespace(receipt={}, half=None)

    @contextmanager
    def scope(*args, **kwargs):
        yield shared

    monkeypatch.setattr(_df_shared, 'reverse_scope', scope)
    monkeypatch.setattr(_df_half, 'HalfTransformedDFCache', lambda owner: object())
    state = {'mf': SimpleNamespace(mol=object())}
    with driver._mp2_df_scope(state, atoms=np.array([0]), with_grad=with_grad) as actual:
        assert actual is shared
        expected = override or ('half' if with_grad else 'direct')
        assert shared.receipt['cache_mode'] == expected
        assert (shared.half is not None) == (expected == 'half')
