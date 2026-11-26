"""Fixed OpenMP teams for DF-J/SGX-K, with caller policy restoration."""
import ctypes
from contextlib import contextmanager

from threadpoolctl import ThreadpoolController


@contextmanager
def fixed_openmp_teams():
    """Disable dynamic sizing in loaded OpenMP runtimes for this call only.

    Keep the configured thread budgets and MKL dynamic policy unchanged.
    Discover runtimes on each entry: the optional generated SGX library may
    be loaded inside an outer J/K call and use a different OpenMP runtime.
    Native SGX entry points enter this scope again after their library loads.
    """
    saved = []
    try:
        for runtime in ThreadpoolController().lib_controllers:
            if runtime.user_api != 'openmp':
                continue
            get_dynamic = runtime.dynlib.omp_get_dynamic
            set_dynamic = runtime.dynlib.omp_set_dynamic
            get_dynamic.argtypes = []
            get_dynamic.restype = ctypes.c_int
            set_dynamic.argtypes = [ctypes.c_int]
            set_dynamic.restype = None
            saved.append((set_dynamic, get_dynamic()))
            set_dynamic(0)
        yield
    finally:
        for set_dynamic, previous in reversed(saved):
            set_dynamic(previous)
