import os
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

libao2mo_vjp = load_library('libao2mo_vjp')
libcc_vjp = load_library('libcc_vjp')
libcgto_ad = load_library('libcgto_ad')
libcgto_vjp = _FallbackCDLL(load_library('libcgto_vjp'), libcgto_ad)
libcvhf_vjp = load_library('libcvhf_vjp')
libnp_helper_vjp = load_library('libnp_helper_vjp')
