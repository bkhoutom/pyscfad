"""Stage cleanup preserves live response data and keeps compilation opt-in."""

import weakref

import jax
import jax.numpy as jnp
import numpy
import pytest

from .test_boys_prepare import boys_water_dimer


@pytest.mark.parametrize('flag, allow_clear, expect_clear', [
    (None, True, False), ('0', True, False), ('1', False, False),
    ('1', True, True),
])
def test_cleanup_preserves_live_vjp_and_only_clears_opted_in_kernels(
        monkeypatch, flag, allow_clear, expect_clear):
    from pyscfad.dlno_stc import _cleanup

    if flag is None:
        monkeypatch.delenv('PYSCFAD_STC_CLEAR_JAX_CACHES', raising=False)
    else:
        monkeypatch.setenv('PYSCFAD_STC_CLEAR_JAX_CACHES', flag)
    monkeypatch.delenv('PYSCFAD_STC_TRIM_MEMORY', raising=False)
    kernel = jax.jit(lambda x: jnp.sin(x).sum())
    x = jnp.arange(16., dtype=jnp.float64) / 16
    kernel(x).block_until_ready()
    value, pullback = jax.vjp(lambda y: (jnp.cos(y) * y).sum(), x)
    expected = numpy.cos(x) - numpy.asarray(x) * numpy.sin(x)
    assert kernel._cache_size() > 0

    class Cycle:
        pass
    dead = Cycle()
    dead.self = dead
    ref = weakref.ref(dead)
    del dead
    trimmed = []

    def trim(pad):
        assert ref() is None, 'collect dead objects before returning their heap pages'
        assert value.is_ready(), 'synchronize before trimming'
        trimmed.append(pad)
        return 1

    monkeypatch.setattr(_cleanup, '_get_malloc_trim', lambda: trim)
    _cleanup.release_memory('test', value, clear_caches=allow_clear)
    assert trimmed == [0]
    assert (kernel._cache_size() == 0) is expect_clear
    numpy.testing.assert_allclose(pullback(jnp.array(1.))[0], expected,
                                  atol=1e-14, rtol=1e-14)


def test_heap_trimming_can_be_disabled_without_disabling_collection(monkeypatch):
    from pyscfad.dlno_stc import _cleanup

    monkeypatch.setenv('PYSCFAD_STC_TRIM_MEMORY', '0')
    monkeypatch.setattr(_cleanup, '_get_malloc_trim',
                        lambda: pytest.fail('disabled trim accessed libc'))

    class Cycle:
        pass
    dead = Cycle()
    dead.self = dead
    ref = weakref.ref(dead)
    del dead
    _cleanup.release_memory('test', jnp.array(1.))
    assert ref() is None


@pytest.mark.parametrize('platform, loader_error', [
    ('darwin', None), ('linux', OSError), ('linux', AttributeError),
])
def test_unavailable_heap_trimming_is_harmless(monkeypatch, platform, loader_error):
    from pyscfad.dlno_stc import _cleanup

    monkeypatch.setenv('PYSCFAD_STC_TRIM_MEMORY', '1')
    monkeypatch.setattr(_cleanup.sys, 'platform', platform)

    def unavailable(*args):
        if loader_error is None:
            pytest.fail('non-Linux platforms must not load malloc_trim')
        raise loader_error('unavailable')

    monkeypatch.setattr(_cleanup.ctypes, 'CDLL', unavailable)
    _cleanup._get_malloc_trim.cache_clear()
    try:
        value = jnp.array([1., 2.])
        _cleanup.release_memory('test', value)
        numpy.testing.assert_array_equal(value, [1., 2.])
        assert _cleanup._get_malloc_trim() is None
    finally:
        _cleanup._get_malloc_trim.cache_clear()


def test_mp2_stage_cleanup_preserves_energy_and_full_response(
        boys_water_dimer, monkeypatch):
    from pyscfad.dlno_stc import driver, build_domain_groups, build_domain_selections, _cleanup
    from .test_finite_workflow import build_mf

    mf, _, static = boys_water_dimer
    controls = dict(mode='deterministic', laplace_roots=[.2, .8],
                    laplace_weights=[.4, .6])
    monkeypatch.setenv('PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE', 'direct')
    monkeypatch.setenv('PYSCFAD_STC_TRIM_MEMORY', '0')
    monkeypatch.setenv('PYSCFAD_STC_CLEAR_JAX_CACHES', '0')
    expected, expected_bar = driver.value_and_grad(
        mf.mol, build_mf, static, controls=controls, include_hf=True)

    # Keep the real cleanup/cache clear; observe only the allocator boundary.
    stages, clears = [], []
    original_release = _cleanup.release_memory
    original_clear = jax.clear_caches
    current_stage = None

    def release(stage, *args, **kwargs):
        nonlocal current_stage
        current_stage = stage
        return original_release(stage, *args, **kwargs)

    def clear():
        clears.append(current_stage)
        original_clear()

    monkeypatch.setattr(driver, '_release_memory', release, raising=False)
    monkeypatch.setattr(jax, 'clear_caches', clear)
    monkeypatch.setattr(_cleanup, '_get_malloc_trim',
                        lambda: lambda pad: stages.append(current_stage))
    monkeypatch.setenv('PYSCFAD_STC_TRIM_MEMORY', '1')
    monkeypatch.setenv('PYSCFAD_STC_CLEAR_JAX_CACHES', '1')
    actual, actual_bar = driver.value_and_grad(
        mf.mol, build_mf,
        build_static=lambda reference: build_domain_selections(reference, frozen=2),
        controls=controls, include_hf=True)
    assert stages == (['after_selection']
                      + ['after_domain'] * len(build_domain_groups(static))
                      + ['before_scf_response'])
    assert clears == ['after_selection']
    numpy.testing.assert_allclose(actual, expected, atol=1e-11, rtol=0)
    numpy.testing.assert_allclose(actual_bar.coords, expected_bar.coords,
                                  atol=1e-9, rtol=0)
