"""Fragment-local auxiliary SVD compression with a cached subspace response.

The forward factor uses an anchored orthonormal frame, so its derivative remains
well defined when singular values repeat inside either side of the cutoff. This
is an eager custom VJP: the reduced auxiliary dimension must be concrete.
"""
from dataclasses import dataclass
import numbers
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_triangular

from pyscfad.tools import resource_profile


def normalize_tolerance(tol):
    """Return a positive finite real cutoff, or None for disabled compression."""
    if tol is None:
        return None
    if isinstance(tol, (bool, np.bool_)) or not isinstance(tol, numbers.Real):
        raise TypeError('lis_aux_svd_tol tolerance must be a nonnegative finite real number')
    value = float(tol)
    if not np.isfinite(value) or value < 0:
        raise ValueError('lis_aux_svd_tol tolerance must be a nonnegative finite real number')
    return value if value > 0 else None


@dataclass(frozen=True)
class AuxiliarySelection:
    """Fixed retained rank and auxiliary anchor for explicit selection replay."""
    rank: int
    anchor: object


def _packing(nmo, dtype):
    rows, cols = np.tril_indices(nmo)
    weights = np.sqrt(np.where(rows == cols, 1., 2.)).astype(dtype)
    return rows, cols, weights


def _unpack(matrix, nmo):
    rows, cols, weights = _packing(nmo, matrix.dtype)
    result = np.zeros((matrix.shape[0], nmo, nmo), dtype=matrix.dtype)
    values = matrix / weights
    result[:, rows, cols] = values
    result[:, cols, rows] = values
    return result


class AuxiliaryCompressor:
    """One fragment's fixed selection; full SVD factors live only in its VJP."""

    def __init__(self, tol, *, selection=None):
        self.tol = normalize_tolerance(tol)
        self.selection = selection
        self.diagnostics = {}

        @jax.custom_vjp
        def compress(factors):
            return self._forward(factors)[0]

        def forward(factors):
            return self._forward(factors)

        def backward(residual, cotangent):
            before = time.perf_counter()
            with resource_profile.section('lis_aux_svd_pullback'):
                gradient = _pullback(residual, cotangent)
                jax.block_until_ready(gradient)
            self.diagnostics['pullback_s'] = time.perf_counter() - before
            return (gradient,)

        compress.defvjp(forward, backward)
        self._compress = compress

    def __call__(self, factors):
        return self._compress(factors)

    def _forward(self, factors):
        before = time.perf_counter()
        naux, nmo, nmo1 = factors.shape
        if nmo != nmo1:
            raise ValueError('auxiliary factors must have square orbital-pair dimensions')
        if self.tol is None:
            self.diagnostics.update(original_naux=naux, retained_naux=naux,
                                    cutoff=0., forward_s=time.perf_counter() - before,
                                    pullback_s=0.)
            return factors, None

        with resource_profile.section('lis_aux_svd_forward', original_naux=naux):
            array = np.asarray(factors)
            if np.iscomplexobj(array):
                raise TypeError('auxiliary SVD compression requires real factors')
            rows, cols, weights = _packing(nmo, array.dtype)
            matrix = array[:, rows, cols] * weights
            left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
            eigenvalues = singular ** 2
            if self.selection is None:
                rank = int(np.count_nonzero(eigenvalues > self.tol))
                if rank == 0:
                    raise ValueError('auxiliary SVD tolerance discards every auxiliary direction')
                anchor = left[:, :rank].copy() if rank < naux else None
                if anchor is not None:
                    anchor.flags.writeable = False
                self.selection = AuxiliarySelection(rank, anchor)
            rank = self.selection.rank
            if not isinstance(rank, numbers.Integral) or not 0 < rank <= min(matrix.shape):
                raise ValueError('fixed auxiliary selection has an invalid retained rank')
            kept = float(eigenvalues[rank - 1])
            discarded = float(eigenvalues[rank]) if rank < len(singular) else 0.
            gap = kept - discarded
            discarded_weight = float(np.sum(eigenvalues[rank:]))
            total_weight = float(np.sum(eigenvalues))
            self.diagnostics.update(
                original_naux=naux, retained_naux=rank, cutoff=self.tol,
                boundary_kept_eigenvalue=kept,
                boundary_discarded_eigenvalue=discarded,
                boundary_gap=gap, discarded_weight=discarded_weight,
                relative_discarded_weight=discarded_weight / total_weight if total_weight else 0.,
                pullback_s=0.)
            if rank == naux:
                result, residual = factors, None
            else:
                # A genuine boundary degeneracy has no unique retained response.
                resolution = np.finfo(eigenvalues.dtype).eps * max(matrix.shape) * eigenvalues[0]
                if gap <= resolution:
                    raise ValueError('auxiliary retained/discarded boundary is degenerate')
                anchor = np.asarray(self.selection.anchor)
                if anchor.shape != (naux, rank):
                    raise ValueError('fixed auxiliary anchor has incompatible dimensions')
                retained = left[:, :rank]
                projected = retained @ (retained.T @ anchor)
                try:
                    chol = np.linalg.cholesky(projected.T @ projected)
                except np.linalg.LinAlgError as error:
                    raise ValueError('fixed auxiliary anchor no longer spans the retained space') from error
                inverse = solve_triangular(chol.T, np.eye(rank, dtype=array.dtype), lower=False)
                frame = projected @ inverse
                compressed = frame.T @ matrix
                result = jnp.asarray(_unpack(compressed, nmo))
                # No full factors are saved on the compressor itself.
                residual = tuple(jnp.asarray(value) for value in
                                 (matrix, left, eigenvalues, anchor, projected,
                                  chol, inverse, frame))
            self.diagnostics['forward_s'] = time.perf_counter() - before
            return result, residual


def _pullback(residual, cotangent):
    if residual is None:
        return cotangent
    matrix, left, eigenvalues, anchor, projected, chol, inverse, frame = residual
    rank = frame.shape[1]
    nmo = cotangent.shape[1]
    rows, cols, weights = _packing(nmo, cotangent.dtype)
    # Adjoint of symmetric unpacking: both off-diagonal output entries count.
    packed_bar = (cotangent[:, rows, cols]
                  + jnp.where(rows == cols, 0., cotangent[:, cols, rows])) / weights
    matrix_bar = frame @ packed_bar
    frame_bar = matrix @ packed_bar.T

    # Q = Y L^-T, LL^T = Y^T Y. Differentiate this exact forward gauge.
    projected_bar = frame_bar @ inverse.T
    inverse_bar = projected.T @ frame_bar
    chol_bar = -inverse @ inverse_bar.T @ inverse
    middle = chol.T @ chol_bar
    middle = jnp.tril(middle) - .5 * jnp.diag(jnp.diag(middle))
    gram_bar = inverse @ middle @ inverse.T
    projected_bar += projected @ (gram_bar + gram_bar.T)

    retained = left[:, :rank]
    discarded = left[:, rank:]
    # Cross-group projector response; never divide within either spectral group.
    mixing_bar = (projected_bar @ (anchor.T @ retained)
                  + anchor @ (projected_bar.T @ retained))
    if discarded.shape[1]:
        mixing = (discarded.T @ mixing_bar) / (
            eigenvalues[:rank][None, :] - eigenvalues[rank:][:, None])
        matrix_bar += (discarded @ mixing @ (retained.T @ matrix)
                       + retained @ mixing.T @ (discarded.T @ matrix))
    if left.shape[0] > left.shape[1]:
        # The thin SVD omits the left zero eigenspace. Apply its projector
        # without constructing an auxiliary-by-auxiliary identity/projector.
        null_mixing = (mixing_bar - left @ (left.T @ mixing_bar)) / eigenvalues[:rank]
        matrix_bar += null_mixing @ (retained.T @ matrix)

    # Adjoint of lower-triangle weighted packing (upper input entries unused).
    gradient = jnp.zeros((matrix.shape[0], nmo, nmo), dtype=matrix_bar.dtype)
    return gradient.at[:, rows, cols].set(matrix_bar * weights)


def make_compressor(tol, *, selection=None):
    """Make a fragment-local callback, optionally replaying a fixed selection."""
    return AuxiliaryCompressor(tol, selection=selection)
