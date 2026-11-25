"""DF-factor pullbacks using the RCCSD lambda BLAS/OpenMP policy."""
import numpy as np
from threadpoolctl import threadpool_limits
from ._native_kernels import _ccsd_kernels as cpp


def _inputs(arrays):
    if any(np.asarray(a).dtype != np.float64 for a in arrays):
        raise NotImplementedError('native DF factor pullbacks require float64')
    return tuple(np.require(a, requirements=['C', 'A']) for a in arrays)


def ovvv(*arrays):
    if cpp is None or not hasattr(cpp, 'dfccsd_ovvv_vjp'):
        raise NotImplementedError('rebuild _ccsd_kernels for DF factor pullbacks')
    arrays = _inputs(arrays)
    with threadpool_limits(1, user_api='blas'):
        return cpp.dfccsd_ovvv_vjp(*arrays)


def vvvv(tau, factors, cotangent):
    if cpp is None or not hasattr(cpp, 'dfccsd_vvvv_vjp'):
        raise NotImplementedError('rebuild _ccsd_kernels for DF factor pullbacks')
    arrays = _inputs((tau, factors, cotangent))
    with threadpool_limits(1, user_api='blas'):
        return cpp.dfccsd_vvvv_vjp(*arrays)
