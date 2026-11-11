# Copyright 2026 The PySCFAD Authors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Optional C++ tensor kernels; PySCF remains the portable fallback."""
import numpy as np
from pyscf import lib
from threadpoolctl import threadpool_limits

try:
    from . import _ccsd_kernels
except ImportError:
    _ccsd_kernels = None


def contract_vvvv(cc, pairs, tensor, out=None):
    """Return None when unavailable or when the panel exceeds the RAM budget.

    All remaining total memory is available after reserving the output,
    index table, and any contiguous input copies. The C++ kernel selects
    shape-dependent GEMM tiles to limit RSS without a fixed byte cap; it
    does not grow panels merely to consume the available allowance.
    """
    if _ccsd_kernels is None or not getattr(cc, 'cpp_kernels', False):
        return None
    # PySCF's buffer argument is uncommon in this path. Preserve its exact
    # aliasing/buffer semantics through the existing implementation.
    if out is not None:
        return None
    v = tensor.shape[-1]
    remaining = max(0., cc.max_memory-lib.current_memory()[0])*1e6
    reserved = tensor.nbytes + v*v*np.dtype(np.uintp).itemsize
    reserved += sum(x.nbytes for x in (pairs, tensor) if not x.flags.c_contiguous)
    workspace = int(remaining) - reserved
    if workspace < 8*v*v:
        return None
    # One OpenMP team owns independent output panels. Sequential BLAS
    # inside each worker prevents nested teams and BLAS/OpenMP spin overlap.
    with threadpool_limits(1, user_api='blas'):
        return _ccsd_kernels.ladder(np.ascontiguousarray(pairs),
                                    np.ascontiguousarray(tensor), workspace)
