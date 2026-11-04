"""Exact symmetric-pair adjoints for same-frame local DF factors.

Stored pairs are matrix entries, not doubled off-diagonal coordinates.
Pair multiplicities enter only the metric contraction. Consumers expand
bounded auxiliary slabs, never a full auxiliary-by-orbital-square adjoint.
"""
import time
import ctypes

import numpy
import scipy.linalg
from pyscf import lib
from pyscfad.df._cderi_vjp import libao2mo_vjp
from pyscfad.lib._threading import dense_blas_threads

from ._profile import _vjp_progress_section

_pack = getattr(libao2mo_vjp, 'AO2MOsymmetric_pair_pack', None)
_unpack = getattr(libao2mo_vjp, 'AO2MOsymmetric_pair_unpack', None)
if _pack is not None:
    _pack.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]*5
    _pack.restype = None
if _unpack is not None:
    _unpack.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]*3
    _unpack.restype = None


def _pair_threads(naux, width):
    # Small streamed slabs do not justify launching a full machine team.
    return min(max(1, lib.num_threads()), 16,
               max(1, naux*width*width // (512*1024)))


def same_orbital_pair(slices):
    k0, k1, l0, l1 = slices
    return k0 == l0 and k1 == l1


def pack_symmetric(array, width, *, order='C', weighted=False):
    """Project an arbitrary real seed onto symmetric matrices and pack it."""
    array = numpy.asarray(array)
    matrices = array.reshape(array.shape[0], width, width)
    count = width*(width+1)//2
    out = numpy.empty((array.shape[0], count), dtype=array.dtype, order=order)
    if _pack is not None and array.dtype == numpy.float64 and array.flags.c_contiguous:
        _pack(out.ctypes.data, array.ctypes.data, len(array), width,
              int(order == 'F'), int(weighted), _pair_threads(len(array), width))
        return out
    i, j = numpy.tril_indices(width)
    # Bound the two advanced-indexing temporaries independently of naux.
    rows = max(1, (16 * 1024**2) // max(8 * len(i), 1))
    for p0 in range(0, len(out), rows):
        block = matrices[p0:p0+rows]
        packed = .5 * (block[:, i, j] + block[:, j, i])
        if weighted:
            packed[:, i != j] *= 2.
        out[p0:p0+rows] = packed
    return out


def unpack_symmetric(pairs, width):
    pairs = numpy.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != width * (width + 1) // 2:
        raise ValueError('Symmetric local DF pair shape mismatch')
    if not width or not len(pairs):
        return numpy.empty((len(pairs), width, width), dtype=pairs.dtype)
    out = numpy.empty((len(pairs), width, width), dtype=pairs.dtype)
    if _unpack is not None and pairs.dtype == numpy.float64:
        pairs = numpy.ascontiguousarray(pairs)
        _unpack(out.ctypes.data, pairs.ctypes.data, len(pairs), width,
                _pair_threads(len(pairs), width))
    else:
        i, j = numpy.tril_indices(width)
        out[:, i, j] = pairs
        out[:, j, i] = pairs
    return out


def factor_pullback(low, fitted, seed, *, symmetric=False, receipt=None):
    """Return Z=L^-T seed and Lbar, retaining only unique pairs if symmetric."""
    started = time.perf_counter()
    with _vjp_progress_section('local DF adjoint packing'):
        if symmetric:
            width = int(numpy.sqrt(seed.shape[1]))
            if width * width != seed.shape[1]:
                raise ValueError('Symmetric local DF factors must be square')
            rhs = pack_symmetric(seed, width, order='F')
            primal = pack_symmetric(fitted, width, weighted=True)
        else:
            rhs, primal = seed, fitted
    packed = time.perf_counter()
    with _vjp_progress_section('local DF whitening backward'), dense_blas_threads():
        z = scipy.linalg.solve_triangular(low.T, rhs, lower=False,
            check_finite=False, overwrite_b=symmetric)
    solved = time.perf_counter()
    with _vjp_progress_section('local DF metric product backward'), dense_blas_threads():
        low_bar = -numpy.dot(z, primal.T)
    finished = time.perf_counter()
    if receipt is not None:
        for name, elapsed in (
                ('adjoint_pack_seconds', packed-started),
                ('whitening_seconds', solved-packed),
                ('metric_product_seconds', finished-solved)):
            receipt[name] = receipt.get(name, 0.) + elapsed
    return z, low_bar
