import os
import threading
import numpy

def load_library(libname):
    try:
        _loaderpath = os.path.dirname(__file__)
        return numpy.ctypeslib.load_library(libname, _loaderpath)
    except OSError:
        raise

class _FallbackCDLL:
    def __init__(self, *libs):
        self._libs = libs
        self._name = libs[0]._name

    def __getattr__(self, name):
        last_err = None
        for lib in self._libs:
            try:
                return getattr(lib, name)
            except AttributeError as err:
                last_err = err
        raise last_err


class _LazyCDLL:
    """Load compatibility kernels only when a legacy symbol is requested."""
    def __init__(self, libname):
        self._libname = libname
        self._library = None
        self._lock = threading.Lock()

    def __getattr__(self, name):
        with self._lock:
            if self._library is None:
                self._library = load_library(self._libname)
        return getattr(self._library, name)


libao2mo_vjp = load_library('libao2mo_vjp')
# Deterministic DLNO uses _ccsd_kernels for DF pullbacks and _stc_mp2 for
# Laplace triples. Keep the older canonical/LNO triples ABI available without
# loading its BLAS/OpenMP dependencies into that route at import time.
libcc_vjp = _LazyCDLL('libcc_vjp')
libcgto_ad = load_library('libcgto_ad')
libcgto_vjp = _FallbackCDLL(load_library('libcgto_vjp'), libcgto_ad)
libcvhf_vjp = load_library('libcvhf_vjp')
libnp_helper_vjp = load_library('libnp_helper_vjp')

# MP2 fusion is optional so existing native installations remain importable.
try:
    libmp_vjp = load_library('libmp_vjp')
except OSError:
    libmp_vjp = None
