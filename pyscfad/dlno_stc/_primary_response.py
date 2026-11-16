"""Select primary spaces and retain their response from the same evaluation.

Only integer labels leave this eager selection boundary as metadata. Primal
arrays and their pullbacks belong to the current SelectionResponse and are
consumed before the strong-domain construction.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from pyscfad import scipy
from ._selection import (FixedPAOSubspaceSelection, _PAO_ORTH_THRESHOLD,
                         _pao_columns_on_atoms, _orthogonalize)
from .domain import _cholesky_metric_orthonormalize
from ._pairs import _finish_weak_screen, _SharedPAOPullback
from pyscfad.lno import tools
from pyscfad.lno.lno_base import SEMICANONICAL_DEG_THRESH
from pyscfad.lib._threading import blas_threads


def _keep(values, threshold):
    values = np.asarray(jax.device_get(jax.lax.stop_gradient(values)))
    if not np.isfinite(values).all():
        raise RuntimeError('nonfinite primary-space selection spectrum')
    return np.flatnonzero(values > threshold).astype(np.int32)


def _host_values(values):
    return np.asarray(jax.device_get(jax.lax.stop_gradient(values)))


def _roundoff(values):
    return 32 * np.finfo(values.real.dtype).eps * max(1, values.size) * max(
        1., float(np.max(np.abs(values), initial=0)))


def _near_cutoff(values, threshold):
    values = _host_values(values)
    return np.any(np.abs(values - threshold) <= _roundoff(values))


def _degenerate_spectrum(values, threshold):
    values = _host_values(values)
    return np.any(np.diff(values) <= max(threshold, _roundoff(values)))


@blas_threads(1)
def _basis_sensitive_completeness(values, keep, candidate, rhs, threshold):
    """Detect a degenerate group that an eigenvector-wise cutoff can split.

    A changed LAPACK workspace may rotate a repeated eigenspace. Its physical
    span is preserved only if every direction in that group is retained or
    every direction is rejected. Test the small Hermitian completeness
    operator, without assuming nested supports or symmetric ambient inputs.
    These host values select a computational route; the live arrays and their
    response are never replaced by detached values.
    """
    selected = _host_values(values)[keep]
    boundaries = np.r_[0, np.flatnonzero(np.diff(selected) > max(
        _PAO_ORTH_THRESHOLD, _roundoff(selected))) + 1, selected.size]
    groups = [(left, right) for left, right in zip(boundaries, boundaries[1:]) if right-left > 1]
    if not groups:
        return False
    coeff, product = map(_host_values, (candidate, rhs))
    for left, right in groups:
        metric = coeff[:, left:right].T.conj() @ product[:, left:right]
        spectrum = np.linalg.eigvalsh(.5 * (metric + metric.T.conj()))
        margin = _roundoff(spectrum)
        if spectrum[0] <= threshold + margin and spectrum[-1] >= threshold - margin:
            return True
    return False


def _eigh(matrix, threshold, *, deg_thresh=1e-9):
    matrix = (matrix + matrix.T.conj()) * .5
    # Rank cutoffs and eigenvector-response regularization are independent.
    # Match the fixed-selection replay's regularization at each stage.
    values, vectors = scipy.linalg.eigh(matrix, deg_thresh=deg_thresh)
    return vectors, _keep(jnp.real(values), threshold)


def _metric_space(coeff, overlap, threshold):
    if not coeff.shape[1]:
        return coeff, np.empty(0, dtype=np.int32)
    metric = coeff.T.conj() @ overlap @ coeff
    if coeff.shape[1] == 1:
        # Only the rank label is detached; normalization retains its S/C response.
        norm2 = jnp.real(metric[0, 0])
        keep = _keep(norm2[None], threshold)
        return (coeff / jnp.sqrt(norm2) if keep.size else coeff[:, :0]), keep
    vectors, keep = _eigh(metric, threshold)
    if keep.size == coeff.shape[1]:
        # The eigensolve has selected the full span. Its eigenvectors are an
        # arbitrary gauge; normalize directly and avoid their reverse rule.
        return _cholesky_metric_orthonormalize(coeff, overlap, metric=metric), keep
    rotation = vectors[:, keep]
    retained = coeff @ rotation
    if not keep.size:
        return retained, keep
    # Rotate the live Gram only when it is no larger than the AO overlap.
    # Its off-diagonal response is needed within degenerate retained spaces;
    # replacing it by the selected eigenvalues would drop that response.
    retained_metric = (rotation.T.conj() @ metric @ rotation
                       if metric.shape[0] <= overlap.shape[0] else None)
    return _cholesky_metric_orthonormalize(retained, overlap, metric=retained_metric), keep


def _select_preprojection_reference(common, products, *, columns, support, ao, thresholds,
                                    completeness_threshold=None):
    gram_all, overlap_pao = products
    gram = gram_all[np.ix_(columns, columns)]
    vectors, canonical = _eigh(gram, _PAO_ORTH_THRESHOLD, deg_thresh=_PAO_ORTH_THRESHOLD)
    empty = np.empty(0, dtype=np.int32)
    if not canonical.size:
        return jnp.zeros((len(ao), 0), dtype=gram.dtype), (canonical, empty, empty)
    transform = _cholesky_metric_orthonormalize(vectors[:, canonical], gram)
    tmp = overlap_pao[np.ix_(support, columns)] @ transform
    projected = jnp.linalg.solve(common.s1e[np.ix_(support, support)], tmp)
    vectors, overlap_keep = _eigh(tmp.T.conj() @ projected, thresholds.domain_pao,
                                 deg_thresh=_PAO_ORTH_THRESHOLD)
    if not overlap_keep.size:
        return jnp.zeros((len(ao), 0), dtype=gram.dtype), (canonical, overlap_keep, empty)
    rotation = vectors[:, overlap_keep]
    if np.array_equal(support, ao):
        candidate, rhs = projected @ rotation, tmp @ rotation
    else:
        rhs = overlap_pao[np.ix_(ao, columns)] @ (transform @ rotation)
        candidate = jnp.linalg.solve(common.s1e[np.ix_(ao, ao)], rhs)
    completeness = jnp.real(jnp.sum(candidate.conj() * rhs, axis=0))
    complete = _keep(completeness, thresholds.bp_pao if completeness_threshold is None
                     else completeness_threshold)
    return candidate[:, complete], (canonical, overlap_keep, complete)


def _select_preprojection(common, products, *, columns, support, ao, thresholds,
                          completeness_threshold=None):
    """Keep physical PAO rank labels outside the padded numerical kernels."""
    from . import _primary_kernels as kernels
    def reference():
        return _select_preprojection_reference(common, products, columns=columns,
            support=support, ao=ao, thresholds=thresholds,
            completeness_threshold=completeness_threshold)
    if not len(columns) or not len(support) or not len(ao):
        return reference()
    gram_all, overlap_pao = products
    padded_columns, padded_support = map(kernels.pad_indices, (columns, support))
    gram, values, vectors = kernels.canonical_pao(gram_all, padded_columns, len(columns))
    canonical = _keep(values[:len(columns)], _PAO_ORTH_THRESHOLD)
    if _near_cutoff(values[:len(columns)], _PAO_ORTH_THRESHOLD):
        return reference()
    empty = np.empty(0, dtype=np.int32)
    if not canonical.size:
        return jnp.zeros((len(ao), 0), dtype=gram.dtype), (canonical, empty, empty)
    transform, tmp, projected, values, vectors = kernels.project_pao(
        common.s1e, overlap_pao, gram, vectors,
        kernels.pad_indices(canonical, padded_columns.size), canonical.size,
        padded_columns, padded_support, len(columns), len(support))
    overlap_keep = _keep(values[:canonical.size], thresholds.domain_pao)
    if _near_cutoff(values[:canonical.size], thresholds.domain_pao):
        return reference()
    if not overlap_keep.size:
        return jnp.zeros((len(ao), 0), dtype=gram.dtype), (canonical, overlap_keep, empty)
    padded_keep = kernels.pad_indices(overlap_keep, padded_columns.size)
    if np.array_equal(support, ao):
        candidate, completeness, rhs = kernels.represent_pao_same(
            tmp, projected, vectors, padded_keep, overlap_keep.size)
    else:
        candidate, completeness, rhs = kernels.represent_pao_other(
            common.s1e, overlap_pao, transform, vectors, padded_keep, overlap_keep.size,
            padded_columns, kernels.pad_indices(ao), len(columns), len(ao))
    cutoff = thresholds.bp_pao if completeness_threshold is None else completeness_threshold
    complete = _keep(completeness[:overlap_keep.size], cutoff)
    if (_near_cutoff(completeness[:overlap_keep.size], cutoff)
            or _basis_sensitive_completeness(values, overlap_keep, candidate, rhs, cutoff)):
        return reference()
    return candidate[:len(ao), complete], (canonical, overlap_keep, complete)


def _select_target_reference(common, candidate, occupied_rhs, *, ao, thresholds, index):
    overlap, fock = common.s1e[np.ix_(ao, ao)], common.fock[np.ix_(ao, ao)]
    occupied = jnp.linalg.solve(overlap, occupied_rhs[ao])
    norm2 = jnp.real(jnp.einsum('ui,uv,vi->i', occupied.conj(), overlap, occupied))
    norm_keep = _keep(norm2, thresholds.metric_rank)
    empty = np.empty(0, dtype=np.int32)
    if not norm_keep.size:
        return None, (norm_keep, empty, empty)
    occupied = occupied[:, norm_keep] / jnp.sqrt(norm2[norm_keep])[None, :]
    span, span_keep = _metric_space(occupied, overlap, thresholds.metric_rank)
    virtual, metric_keep = _metric_space(
        _orthogonalize(span, candidate, overlap), overlap, thresholds.metric_rank)
    if not virtual.shape[1]:
        return None, (norm_keep, span_keep, metric_keep)
    return _finish_weak_screen(occupied, virtual, fock, index), (norm_keep, span_keep, metric_keep)


def _select_target_bucketed(common, candidate, occupied_rhs, *, ao, thresholds, index):
    """Return the screen, labels, and whether its original basis is required."""
    from . import _primary_kernels as kernels
    from ._pairs import WeakScreen
    empty = np.empty(0, dtype=np.int32)
    fallback = None, ((empty, empty, empty), True)
    coeff_dtype = np.result_type(common.s1e.dtype, candidate.dtype, occupied_rhs.dtype)
    # The original singleton semicanonicalizer preserves coefficient dtype;
    # the multiple-column branch may promote it with Fock. Preserve both paths.
    if (occupied_rhs.shape[1] != 1 or not candidate.shape[1]
            or np.result_type(coeff_dtype, common.fock.dtype) != coeff_dtype):
        return fallback
    nrow, ncol = candidate.shape
    padded, indices = kernels.pad_target(candidate, ao)
    overlap, fock, occupied, norm2 = kernels.project_occupied(
        common.s1e, common.fock, occupied_rhs, indices, nrow)
    norm_keep = _keep(norm2, thresholds.metric_rank)
    if _near_cutoff(norm2, thresholds.metric_rank):
        return fallback
    if not norm_keep.size:
        return None, ((norm_keep, empty, empty), False)
    occupied, span_norm2, candidate, metric, values, vectors = kernels.virtual_metric(
        occupied, norm2, padded, overlap, ncol, thresholds.metric_rank)
    span_keep = _keep(span_norm2[None], thresholds.metric_rank)
    metric_keep = _keep(values[:ncol], thresholds.metric_rank)
    if (_near_cutoff(span_norm2[None], thresholds.metric_rank)
            or _near_cutoff(values[:ncol], thresholds.metric_rank)):
        return fallback
    if not metric_keep.size:
        return None, ((norm_keep, span_keep, metric_keep), False)
    retained = metric_keep.size
    keep = np.pad(metric_keep, (0, padded.shape[1] - retained))
    occupied_energy, virtual_energy, virtual = kernels.finish_orbitals(
        occupied, candidate, overlap, fock, metric, vectors, keep, retained, ncol, nrow)
    virtual_energy, virtual = virtual_energy[:retained], virtual[:nrow, :retained]
    # The existing regularized eigh response depends on its eigenvector gauge
    # at degeneracy. Preserve the old basis and derivative convention there.
    if _degenerate_spectrum(virtual_energy, SEMICANONICAL_DEG_THRESH):
        return fallback
    gaps = np.asarray(jax.device_get(jax.lax.stop_gradient(virtual_energy[:, None] - occupied_energy)))
    if not np.isfinite(gaps).all() or np.any(gaps <= 0):
        return fallback
    screen = WeakScreen(occupied_energy, occupied[:nrow], virtual_energy, virtual)
    return screen, ((norm_keep, span_keep, metric_keep), False)


def _select_target(common, candidate, occupied_rhs, *, ao, thresholds, index):
    """Use padded workspaces except where the original frame is significant."""
    screen, (labels, original_required) = _select_target_bucketed(
        common, candidate, occupied_rhs, ao=ao, thresholds=thresholds, index=index)
    if original_required:
        return _select_target_reference(common, candidate, occupied_rhs,
                                        ao=ao, thresholds=thresholds, index=index)
    return screen, labels


def prepare_primary(common, products_preparation, mol, ao2pao, thresholds,
                    index, primary_bp, primary, cache):
    """Return fixed labels and one live orbital linearization, without replay."""
    products, products_pullback = products_preparation
    columns = _pao_columns_on_atoms(mol, primary_bp, ao2pao)
    support = tools.ao_index_by_atom(mol, np.asarray(primary_bp, dtype=np.int32)).astype(np.int32)
    ao = tools.ao_index_by_atom(mol, np.asarray(primary, dtype=np.int32)).astype(np.int32)
    key = tuple(map(tuple, (columns, support, ao)))
    if key not in cache:
        cache[key] = jax.vjp(partial(
            _select_preprojection, columns=columns, support=support, ao=ao,
            thresholds=thresholds), common, products.pao, has_aux=True)
    candidate, preprojection_pullback, (canonical, overlap_keep, complete) = cache[key]
    screen, target_pullback, (labels, original_required) = jax.vjp(partial(
        _select_target_bucketed, ao=ao, thresholds=thresholds, index=index),
        common, candidate, products.occupied[index], has_aux=True)
    if original_required:
        # A changed upstream PAO gauge can also alter a degenerate Fock
        # response. Restore both stages, sharing this original preparation
        # among every target on the same ordered support that needs it.
        original_key = ('original', key)
        if original_key not in cache:
            cache[original_key] = jax.vjp(partial(
                _select_preprojection_reference, columns=columns, support=support,
                ao=ao, thresholds=thresholds), common, products.pao, has_aux=True)
        candidate, preprojection_pullback, (canonical, overlap_keep, complete) = cache[original_key]
        screen, target_pullback, labels = jax.vjp(partial(
            _select_target_reference, ao=ao, thresholds=thresholds, index=index),
            common, candidate, products.occupied[index], has_aux=True)
    norm_keep, span_keep, metric_keep = labels
    selection = None
    preparation = None
    if screen is not None:
        selection = FixedPAOSubspaceSelection(
            columns, support, canonical, overlap_keep, complete, metric_keep)
        preparation = screen, _SharedPAOPullback(
            target_pullback, preprojection_pullback, products_pullback, index)
    return ao, norm_keep, span_keep, selection, preparation
