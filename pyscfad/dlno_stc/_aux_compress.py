"""Fragment-local auxiliary compression with a cached subspace response.

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
from scipy.linalg import eigh, solve_triangular
from scipy.linalg.blas import get_blas_funcs

from pyscfad.lib._threading import dense_blas_threads
from pyscfad.tools import resource_profile


def normalize_tolerance(tol, *, name="lis_aux_svd_tol"):
    """Return a positive finite real cutoff, or None for disabled compression."""
    if tol is None:
        return None
    if isinstance(tol, (bool, np.bool_)) or not isinstance(tol, numbers.Real):
        raise TypeError(f'{name} tolerance must be a nonnegative finite real number')
    value = float(tol)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} tolerance must be a nonnegative finite real number')
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


def _auxiliary_eigensystem(matrix, tol, selection=None):
    """Return descending auxiliary eigenpairs without unused right vectors.

    The auxiliary Gram is smaller for wide factors. Its roundoff is absolute
    on the scale of the largest eigenvalue, so use the direct SVD when that
    uncertainty could change a cutoff decision or the replay boundary.
    Tall factors retain the thin SVD and its implicit left null space.
    """
    timings = dict(gram_s=0., eigensolve_s=0., svd_s=0.)
    naux, npair = matrix.shape
    method = 'svd_tall'
    if naux <= npair:
        start = time.perf_counter()
        # f2py expects column-major storage. A C-order factor's transpose is
        # already a Fortran-order view; SYRK's transpose flag avoids copying
        # the full factor while still forming matrix @ matrix.T.
        if matrix.flags.f_contiguous:
            operand, trans, layout = matrix, 0, 'F'
        elif matrix.flags.c_contiguous:
            operand, trans, layout = matrix.T, 1, 'C'
        else:
            operand, trans, layout = np.asfortranarray(matrix), 0, 'strided'
        timings.update(gram_copy_s=time.perf_counter() - start,
                       gram_copy_bytes=matrix.nbytes if layout == 'strided' else 0,
                       gram_input_order=layout, gram_trans=trans)
        syrk = get_blas_funcs('syrk', (operand,))
        multiply_start = time.perf_counter()
        gram = syrk(1., operand, trans=trans, lower=1)
        timings['gram_multiply_s'] = time.perf_counter() - multiply_start
        timings['gram_s'] = time.perf_counter() - start
        start = time.perf_counter()
        values, vectors = eigh(gram, lower=True, overwrite_a=True, driver='evd')
        values, vectors = values[::-1].copy(), vectors[:, ::-1].copy()
        timings['eigensolve_s'] = time.perf_counter() - start
        resolution = np.finfo(values.dtype).eps * max(matrix.shape) * abs(values[0])
        uncertain = values[-1] < -resolution
        if selection is None:
            uncertain = uncertain or np.any(np.abs(values - tol) <= resolution)
        elif isinstance(selection.rank, numbers.Integral) and 0 < selection.rank < naux:
            rank = selection.rank
            uncertain = uncertain or values[rank-1] - values[rank] <= 2 * resolution
        if not uncertain:
            # Roundoff can make a discarded zero eigenvalue slightly negative.
            return vectors, np.maximum(values, 0.), dict(method='gram_eigh', **timings)
        method = 'svd_cutoff_fallback'
    start = time.perf_counter()
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    timings['svd_s'] = time.perf_counter() - start
    return left, singular ** 2, dict(method=method, **timings)


class AuxiliaryCompressor:
    """One fragment's fixed selection; eigenpairs live only in its VJP."""

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

        with (resource_profile.section('lis_aux_svd_forward', original_naux=naux),
              dense_blas_threads()):
            array = np.asarray(factors)
            if np.iscomplexobj(array):
                raise TypeError('auxiliary SVD compression requires real factors')
            rows, cols, weights = _packing(nmo, array.dtype)
            matrix = array[:, rows, cols] * weights
            left, eigenvalues, decomposition = _auxiliary_eigensystem(
                matrix, self.tol, self.selection)
            self.diagnostics.update(decomposition)
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
            discarded = float(eigenvalues[rank]) if rank < len(eigenvalues) else 0.
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
    matrix_bar = _matrix_pullback(residual, packed_bar)

    # Adjoint of lower-triangle weighted packing (upper input entries unused).
    gradient = jnp.zeros((matrix.shape[0], nmo, nmo), dtype=matrix_bar.dtype)
    return gradient.at[:, rows, cols].set(matrix_bar * weights)


def _matrix_pullback(residual, matrix_cotangent, *, xp=jnp):
    """Adjoint of the anchored spectral projection, for any pair layout."""
    matrix, left, eigenvalues, anchor, projected, chol, inverse, frame = residual
    rank = frame.shape[1]
    matrix_bar = frame @ matrix_cotangent
    frame_bar = matrix @ matrix_cotangent.T

    # Q = Y L^-T, LL^T = Y^T Y. Differentiate this exact forward gauge.
    projected_bar = frame_bar @ inverse.T
    inverse_bar = projected.T @ frame_bar
    chol_bar = -inverse @ inverse_bar.T @ inverse
    middle = chol.T @ chol_bar
    middle = xp.tril(middle) - .5 * xp.diag(xp.diag(middle))
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

    return matrix_bar


def make_compressor(tol, *, selection=None):
    """Make a fragment-local callback, optionally replaying a fixed selection."""
    return AuxiliaryCompressor(tol, selection=selection)
