"""Select primary spaces and retain their response from the same evaluation.

Only integer labels leave this eager selection boundary as metadata. Primal
arrays and their pullbacks belong to the current SelectionResponse and are
consumed before the strong-domain construction.
"""

import jax
import jax.numpy as jnp
import numpy as np

from pyscfad import scipy
from ._selection import FixedPAOSubspaceSelection, _pao_columns_on_atoms
from .domain import _cholesky_metric_orthonormalize
from ._pairs import _SharedPAOPullback
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


def prepare_primary(common, products_preparation, mol, ao2pao, thresholds,
                    index, primary_bp, primary, cache):
    """Return fixed labels and one live orbital linearization, without replay."""
    products, products_pullback = products_preparation
    columns = _pao_columns_on_atoms(mol, primary_bp, ao2pao)
    support = tools.ao_index_by_atom(mol, np.asarray(primary_bp, dtype=np.int32)).astype(np.int32)
    ao = tools.ao_index_by_atom(mol, np.asarray(primary, dtype=np.int32)).astype(np.int32)
    key = tuple(map(tuple, (columns, support, ao)))
    from . import _weak_numpy as host
    if key not in cache:
        cache[key] = host.preprojection(common, products.pao, columns=columns,
            support=support, ao=ao, thresholds=thresholds)
    candidate, preprojection_pullback, (canonical, overlap_keep, complete) = cache[key]
    screen, target_pullback, (norm_keep, span_keep, metric_keep) = host.target(
        common, candidate, products.occupied[index], ao=ao, thresholds=thresholds, index=index)
    selection = preparation = None
    if screen is not None:
        selection = FixedPAOSubspaceSelection(
            columns, support, canonical, overlap_keep, complete, metric_keep)
        preparation = screen, _SharedPAOPullback(
            target_pullback, preprojection_pullback, products_pullback, index)
    return ao, norm_keep, span_keep, selection, preparation
