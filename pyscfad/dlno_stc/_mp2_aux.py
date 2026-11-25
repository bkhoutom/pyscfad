"""Host-owned OV NAF factors and their fixed-rank subspace response.

OV products are rectangular and carry no symmetric-pair weights. The returned
pullback acts only on resident factors; its result joins the existing ED DF
reverse, including the metric response and shared integral derivative pass.
"""
import time

import numpy as np
from scipy.linalg import solve_triangular

from pyscfad.lib._threading import dense_blas_threads
from ._aux_compress import (AuxiliarySelection, _auxiliary_eigensystem,
                            _matrix_pullback, normalize_tolerance)


DEFAULT_MP2_NAF_TOL = 1e-3


def compress_ov(factors, tol=DEFAULT_MP2_NAF_TOL, *, with_grad=False, selection=None):
    """Return ``(B_reduced, pullback, selection, diagnostics)``.

    A supplied selection replays its rank and anchored gauge, useful for
    fixed-selection finite differences. No division within a retained or
    discarded eigenvalue cluster is used. ``None`` or zero disables NAF.
    """
    tol = normalize_tolerance(tol, name='mp2_aux_svd_tol')
    array = np.asarray(factors)
    if array.ndim != 3 or np.iscomplexobj(array):
        raise ValueError('MP2 NAF requires real factors shaped (P, i, a)')
    naux, no, nv = array.shape
    info = dict(cutoff=tol, original_naux=naux, retained_naux=naux,
                forward_s=0., pullback_s=0., method='disabled')
    if tol is None or not array.size:
        return array, (lambda bar: bar) if with_grad else None, selection, info
    start = time.perf_counter()
    with dense_blas_threads():
        matrix = array.reshape(naux, no * nv)
        left, values, decomposition = _auxiliary_eigensystem(matrix, tol, selection)
        if selection is None:
            rank = int(np.count_nonzero(values > tol))
            selection = AuxiliarySelection(rank, left[:, :rank].copy())
        rank = selection.rank
        if not isinstance(rank, (int, np.integer)) or not 0 <= rank <= min(matrix.shape):
            raise ValueError('invalid retained MP2 auxiliary rank')
        info.update(decomposition, retained_naux=int(rank))
        residual = None
        if rank == naux:
            reduced = array
        elif rank == 0:
            reduced = np.empty((0, no, nv), dtype=array.dtype)
        else:
            gap = values[rank-1] - (values[rank] if rank < len(values) else 0.)
            resolution = np.finfo(values.dtype).eps * max(matrix.shape) * values[0]
            if gap <= resolution:
                raise ValueError('MP2 auxiliary retained/discarded boundary is degenerate')
            anchor = np.asarray(selection.anchor)
            if anchor.shape != (naux, rank):
                raise ValueError('fixed MP2 auxiliary anchor has incompatible dimensions')
            retained = left[:, :rank]
            projected = retained @ (retained.T @ anchor)
            chol = np.linalg.cholesky(projected.T @ projected)
            inverse = solve_triangular(chol.T, np.eye(rank), lower=False)
            frame = projected @ inverse
            # P-contiguous storage is shared with the borrowed scalar solver.
            reduced = np.asfortranarray(frame.T @ matrix).reshape(rank, no, nv)
            if with_grad:
                residual = (matrix, left, values, anchor, projected, chol, inverse, frame)
            info.update(boundary_gap=float(gap))
        info.update(forward_s=time.perf_counter()-start,
                    discarded_weight=float(np.sum(values[rank:])))
    if not with_grad:
        return reduced, None, selection, info

    def pullback(cotangent):
        before = time.perf_counter()
        bar = np.asarray(cotangent)
        if bar.shape != reduced.shape:
            raise ValueError('MP2 NAF cotangent has incompatible dimensions')
        with dense_blas_threads():
            if rank == naux:
                result = bar
            elif rank == 0:
                result = np.zeros_like(array)
            else:
                result = _matrix_pullback(residual, bar.reshape(rank, no*nv), xp=np)
                result = np.asfortranarray(result).reshape(array.shape)
        info['pullback_s'] += time.perf_counter()-before
        return result

    return reduced, pullback, selection, info
