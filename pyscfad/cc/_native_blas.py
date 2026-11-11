# Copyright 2026 The PySCFAD Authors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Use NumPy's configured BLAS for the native response contractions.

PySCF wheels can link their private ddot to a sequential BLAS. Optimized
NumPy contractions instead use the threadpool controlled by the native
CCSD scope, without replacing any process-global PySCF functions.
"""
import numpy as np


def einsum(subscripts, *operands):
    return np.einsum(subscripts, *operands, optimize=True)
