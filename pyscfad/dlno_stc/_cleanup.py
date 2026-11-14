"""Release unused host memory at completed eager calculation boundaries."""

import ctypes
from functools import lru_cache
import gc
import os
import sys

import jax
from pyscfad.tools import resource_profile


@lru_cache(maxsize=1)
def _get_malloc_trim():
    """Return glibc's optional heap trimmer, without adding a libc dependency."""
    if not sys.platform.startswith('linux'):
        return None
    try:
        trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return None
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return trim


def _enabled(name, default):
    return os.environ.get(name, default).strip().lower() in ('1', 'true', 'yes', 'on')


def release_memory(stage, *live_results, clear_caches=False,
                   profile_prefix='stc.mp2', **metadata):
    """Collect dead objects and return unused heap pages, preserving live arrays.

    Call after dropping consumed inputs and pullbacks. Heap trimming defaults
    on where glibc provides it; PYSCFAD_STC_TRIM_MEMORY=0 disables it. Clearing
    compiled kernels requires both an eligible boundary (clear_caches=True)
    and PYSCFAD_STC_CLEAR_JAX_CACHES=1. Other boundaries preserve those kernels.
    """
    clear = clear_caches and _enabled('PYSCFAD_STC_CLEAR_JAX_CACHES', '0')
    trim_heap = _enabled('PYSCFAD_STC_TRIM_MEMORY', '1')
    phase = 'clear_jax_caches' if clear else 'release_memory'
    with resource_profile.section(f'{profile_prefix}.{phase}', stage=stage,
                                  trim_heap=trim_heap, **metadata):
        jax.block_until_ready(live_results)
        if clear:
            jax.clear_caches()
        gc.collect()
        if trim_heap:
            trim = _get_malloc_trim()
            if trim is not None:
                trim(0)
