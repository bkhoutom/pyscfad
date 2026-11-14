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


def _keep(values, threshold):
    values = np.asarray(jax.device_get(jax.lax.stop_gradient(values)))
    if not np.isfinite(values).all():
        raise RuntimeError('nonfinite primary-space selection spectrum')
    return np.flatnonzero(values > threshold).astype(np.int32)


def _eigh(matrix, threshold, *, deg_thresh=1e-9):
    matrix = (matrix + matrix.T.conj()) * .5
    # Rank cutoffs and eigenvector-response regularization are independent.
    # Match the fixed-selection replay's regularization at each stage.
    values, vectors = scipy.linalg.eigh(matrix, deg_thresh=deg_thresh)
    return vectors, _keep(jnp.real(values), threshold)


def _metric_space(coeff, overlap, threshold):
    if not coeff.shape[1]:
        return coeff, np.empty(0, dtype=np.int32)
    vectors, keep = _eigh(coeff.T.conj() @ overlap @ coeff, threshold)
    if keep.size == coeff.shape[1]:
        # The eigensolve has selected the full span. Its eigenvectors are an
        # arbitrary gauge; normalize directly and avoid their reverse rule.
        return _cholesky_metric_orthonormalize(coeff, overlap), keep
    retained = coeff @ vectors[:, keep]
    if not keep.size:
        return retained, keep
    return _cholesky_metric_orthonormalize(retained, overlap), keep


def _select_preprojection(common, products, *, columns, support, ao, thresholds,
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


def _select_target(common, candidate, occupied_rhs, *, ao, thresholds, index):
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
    screen, target_pullback, (norm_keep, span_keep, metric_keep) = jax.vjp(partial(
        _select_target, ao=ao, thresholds=thresholds, index=index),
        common, candidate, products.occupied[index], has_aux=True)
    selection = None
    preparation = None
    if screen is not None:
        selection = FixedPAOSubspaceSelection(
            columns, support, canonical, overlap_keep, complete, metric_keep)
        preparation = screen, _SharedPAOPullback(
            target_pullback, preprojection_pullback, products_pullback, index)
    return ao, norm_keep, span_keep, selection, preparation
