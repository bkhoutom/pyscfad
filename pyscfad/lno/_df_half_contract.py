"""Auxiliary-row contractions without flattening the resident half transform."""
from concurrent.futures import ThreadPoolExecutor
import os

import numpy
from pyscf import lib
from pyscfad.lib._threading import blas_threads


def coefficient_contractions(h, z, left, right, *, symmetric=False):
    """Return both coefficient bars for one auxiliary tile.

    H has layout (auxiliary, AO, orbital). Each worker consumes complete
    auxiliary rows, so its H matrices retain their original leading dimension.
    Only the much smaller Z row is packed when needed. BLAS is single threaded
    inside the worker team; its previous settings are restored on return.
    The caller must serialize this scope with other BLAS-policy changes.
    """
    count, nao, _ = h.shape
    if h.dtype != numpy.float64 or z.dtype != numpy.float64:
        raise TypeError('Half DF coefficient contractions require float64')
    if z.ndim != 3 or z.shape[0] != count:
        raise ValueError('Half DF cotangent has incompatible auxiliary dimension')
    kc, lc = z.shape[1:]
    if h[:, :, left].shape[2] != kc or h[:, :, right].shape[2] != lc:
        raise ValueError('Half DF orbital slices do not match the cotangent')
    if symmetric and (left != right or kc != lc):
        raise ValueError('Symmetric coefficient contractions require identical slices')
    threads = int(os.environ.get('PYSCFAD_DENSE_BLAS_THREADS', lib.num_threads()))
    if threads < 1:
        raise ValueError('PYSCFAD_DENSE_BLAS_THREADS must be positive')
    workers = min(threads, count)
    if not count or not kc or not lc:
        return numpy.zeros((nao, kc)), numpy.zeros((nao, lc))

    def contract(bounds):
        first, last = bounds
        occupied = numpy.zeros((nao, kc))
        virtual = None if symmetric else numpy.zeros((nao, lc))
        # Reuse one scratch allocation; no auxiliary-sized output or H copy.
        scratch = numpy.empty(nao * max(kc, lc))
        for p in range(first, last):
            seed = numpy.ascontiguousarray(z[p])
            out = scratch[:nao*kc].reshape(nao, kc)
            numpy.matmul(h[p, :, right], seed.T, out=out)
            occupied += out
            if not symmetric:
                out = scratch[:nao*lc].reshape(nao, lc)
                numpy.matmul(h[p, :, left], seed, out=out)
                virtual += out
        return occupied, virtual

    bounds = [(count*i//workers, count*(i+1)//workers) for i in range(workers)]
    with blas_threads(1):
        if workers == 1:
            parts = [contract(bounds[0])]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                parts = list(pool.map(contract, bounds))
        occupied, virtual = parts[0]
        # Fixed worker order makes the reduction reproducible for a given team.
        for other_occ, other_vir in parts[1:]:
            occupied += other_occ
            if not symmetric:
                virtual += other_vir
    if symmetric:
        occupied *= 2
        virtual = numpy.zeros((nao, lc))
    return occupied, virtual
