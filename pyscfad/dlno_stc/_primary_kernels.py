"""Shape-bucketed algebra for singleton weak-screen target response.

Padding is confined to workspaces. Inactive AO rows have identity overlap and
zero coefficients; inactive virtual columns have identity normalization metric
and eigenvalues above the physical spectrum. Selection sees physical values
only, and all physical arrays retain their complete JAX response.
"""
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import numpy as np

from pyscfad import scipy
from pyscfad.lno.lno_base import SEMICANONICAL_DEG_THRESH
from ._selection import _PAO_ORTH_THRESHOLD


_BLOCK = 32


def _size(length):
    # Two size classes per doubling above the small 32-element grid. A fixed
    # spacing leaves almost every large molecular domain in a different class,
    # multiplying compiled forward/reverse kernels across their input axes.
    # Relative spacing bounds this growth while limiting padding to <50% per
    # axis for dimensions above 64. Physical ranks are still selected outside.
    block = max(_BLOCK, 1 << max(0, (int(length) - 1).bit_length() - 2))
    return ((length + block - 1) // block) * block


def pad_indices(indices, size=None):
    indices = np.asarray(indices, dtype=np.int32)
    size = _size(indices.size) if size is None else size
    return np.pad(indices, (0, size - indices.size))


def pad_target(candidate, ao):
    nrow, ncol = candidate.shape
    rows, cols = _size(nrow), _size(ncol)
    indices = pad_indices(ao, rows)
    padded = jnp.pad(candidate, ((0, rows - nrow), (0, cols - ncol)))
    return padded, indices


def _hermitian(matrix):
    return .5 * (matrix + matrix.T.conj())


def _local_overlap(overlap, indices, active):
    mask = jnp.arange(indices.size) < active
    return jnp.where(mask[:, None] & mask[None, :],
                     overlap[indices[:, None], indices[None, :]],
                     jnp.eye(indices.size, dtype=overlap.dtype))


def _local_product(product, rows, cols, active_rows, active_cols):
    mask = ((jnp.arange(rows.size) < active_rows)[:, None]
            & (jnp.arange(cols.size) < active_cols)[None, :])
    return jnp.where(mask, product[rows[:, None], cols[None, :]], 0)


def _normalize(coeff, metric, active_columns):
    active = jnp.arange(coeff.shape[1]) < active_columns
    gram = _hermitian(metric) + jnp.diag((~active).astype(metric.dtype))
    root = jnp.linalg.cholesky(gram)
    return solve_triangular(root, coeff.T.conj(), lower=True).T.conj()


def _spectral_padding(matrix, active):
    """Place dummy eigenvalues strictly above a Gershgorin spectral bound."""
    mask = jnp.arange(matrix.shape[0]) < active
    matrix = jnp.where(mask[:, None] & mask[None, :], matrix, 0)
    # The dummy spectrum is bookkeeping; its value cannot influence a physical
    # eigenpair. Stopping this bound does not stop any physical matrix response.
    ceiling = jax.lax.stop_gradient(1 + 2 * jnp.max(jnp.sum(jnp.abs(matrix), axis=1)))
    return matrix + jnp.diag(jnp.where(mask, 0, ceiling))


@jax.jit
def canonical_pao(gram_all, columns, active_columns):
    gram = _local_product(gram_all, columns, columns, active_columns, active_columns)
    values, vectors = scipy.linalg.eigh(
        _spectral_padding(_hermitian(gram), active_columns), deg_thresh=_PAO_ORTH_THRESHOLD)
    return gram, values, vectors


@jax.jit
def project_pao(overlap, overlap_pao, gram, vectors, canonical, retained,
                columns, support, active_columns, active_support):
    mask = jnp.arange(columns.size) < retained
    rotation = jnp.where(mask[None, :], vectors[:, canonical], 0)
    transform = _normalize(rotation, rotation.T.conj() @ gram @ rotation, retained)
    tmp = _local_product(overlap_pao, support, columns, active_support, active_columns) @ transform
    projected = jnp.linalg.solve(_local_overlap(overlap, support, active_support), tmp)
    metric = tmp.T.conj() @ projected
    values, vectors = scipy.linalg.eigh(
        _spectral_padding(_hermitian(metric), retained), deg_thresh=_PAO_ORTH_THRESHOLD)
    return transform, tmp, projected, values, vectors


@jax.jit
def represent_pao_same(tmp, projected, vectors, keep, retained):
    mask = jnp.arange(keep.size) < retained
    rotation = jnp.where(mask[None, :], vectors[:, keep], 0)
    candidate, rhs = projected @ rotation, tmp @ rotation
    return candidate, jnp.real(jnp.sum(candidate.conj() * rhs, axis=0)), rhs


@jax.jit
def represent_pao_other(overlap, overlap_pao, transform, vectors, keep, retained,
                        columns, ao, active_columns, active_rows):
    mask = jnp.arange(keep.size) < retained
    rotation = jnp.where(mask[None, :], vectors[:, keep], 0)
    rhs = _local_product(overlap_pao, ao, columns, active_rows, active_columns) @ (transform @ rotation)
    candidate = jnp.linalg.solve(_local_overlap(overlap, ao, active_rows), rhs)
    return candidate, jnp.real(jnp.sum(candidate.conj() * rhs, axis=0)), rhs


@jax.jit
def project_occupied(overlap, fock, rhs, indices, active_rows):
    mask = jnp.arange(indices.size) < active_rows
    pair_mask = mask[:, None] & mask[None, :]
    overlap = _local_overlap(overlap, indices, active_rows)
    fock = jnp.where(pair_mask, fock[indices[:, None], indices[None, :]], 0)
    occupied = jnp.linalg.solve(overlap, jnp.where(mask[:, None], rhs[indices], 0))
    norm2 = jnp.real(jnp.einsum('ui,uv,vi->i', occupied.conj(), overlap, occupied))
    return overlap, fock, occupied, norm2


@jax.jit
def virtual_metric(occupied, norm2, candidate, overlap, active_columns, threshold):
    occupied = occupied / jnp.sqrt(norm2)[None, :]
    span_norm2 = jnp.real((occupied.T.conj() @ overlap @ occupied)[0, 0])
    span = jnp.where(span_norm2 > threshold,
                     occupied / jnp.sqrt(jnp.where(span_norm2 > 0, span_norm2, 1)), 0)
    candidate = candidate - span @ (span.T.conj() @ overlap @ candidate)
    metric = candidate.T.conj() @ overlap @ candidate
    values, vectors = scipy.linalg.eigh(
        _spectral_padding(_hermitian(metric), active_columns), deg_thresh=1e-9)
    return occupied, span_norm2, candidate, metric, values, vectors


@jax.jit
def finish_orbitals(occupied, candidate, overlap, fock, metric, vectors, keep,
                    retained, original_columns, original_rows):
    active = jnp.arange(candidate.shape[1]) < retained

    def full_rank(_):
        return candidate, metric

    def truncated(_):
        rotation = jnp.where(active[None, :], vectors[:, keep], 0)
        coeff = candidate @ rotation
        gram = jax.lax.cond(original_columns <= original_rows,
            lambda _: rotation.T.conj() @ metric @ rotation,
            lambda _: coeff.T.conj() @ overlap @ coeff, operand=None)
        return coeff, gram

    coeff, gram = jax.lax.cond(retained == original_columns, full_rank, truncated, operand=None)
    virtual = _normalize(coeff, gram, retained)
    projected = _hermitian(virtual.T.conj() @ fock @ virtual)

    def singleton(_):
        return jnp.real(jnp.diag(projected)), virtual

    def multiple(_):
        energy, rotation = scipy.linalg.eigh(
            _spectral_padding(projected, retained), deg_thresh=SEMICANONICAL_DEG_THRESH)
        return jnp.real(energy), virtual @ rotation

    energy, virtual = jax.lax.cond(retained == 1, singleton, multiple, operand=None)
    occupied_energy = jnp.real(jnp.einsum('ui,uv,vi->i', occupied.conj(), fock, occupied))
    return occupied_energy, energy, virtual
