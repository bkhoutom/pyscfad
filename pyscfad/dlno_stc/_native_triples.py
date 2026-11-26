"""Deterministic triples on the RCCSD SciPy BLAS/OpenMP runtime.

Each native entry confines BLAS to one worker while C++ owns the OpenMP team.
Thread counts are restored by the scope; the caller's dynamic policy is untouched.
"""
from threadpoolctl import threadpool_limits
from pyscfad.cc._native_kernels import _ccsd_kernels as cpp


def _required():
    if cpp is None or not hasattr(cpp, 'triples_mrcc_point'):
        raise ImportError('rebuild pyscfad/cc/cpp for deterministic CCSD(T)')
    return cpp


def __getattr__(name):
    if name.startswith('TRIPLES_'):
        return getattr(_required(), name)
    raise AttributeError(name)


class StableTriplesDressing:
    def __init__(self, inputs, *, full_external=True, mrcc_partition=True):
        with threadpool_limits(1, user_api='blas'):
            self._engine = _required().StableTriplesDressing(
                inputs, full_external=full_external, mrcc_partition=mrcc_partition)

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def prepare(self, *args, **kwargs):
        with threadpool_limits(1, user_api='blas'):
            return self._engine.prepare(*args, **kwargs)

    def pullback(self, *args, **kwargs):
        with threadpool_limits(1, user_api='blas'):
            return self._engine.pullback(*args, **kwargs)


def triples_mrcc_point(*args, **kwargs):
    with threadpool_limits(1, user_api='blas'):
        return _required().triples_mrcc_point(*args, **kwargs)
