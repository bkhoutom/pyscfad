"""Smooth block frames within retained local interaction spaces.

Ranks, Fock clusters and AO reference anchors are chosen outside differentiation.
Only inter-cluster spectral gaps enter projector derivatives. Internal rotations
are fixed by projecting the reference frame, with differentiable normalization.
"""
from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np


@lru_cache(None)
def spectral_projector(keep):
    @jax.custom_jvp
    def project(matrix):
        _, vectors = jnp.linalg.eigh((matrix + matrix.T) / 2)
        selected = vectors[:, jnp.asarray(keep)]
        return selected @ selected.T

    @project.defjvp
    def derivative(primals, tangents):
        (matrix,), (direction,) = primals, tangents
        values, vectors = jnp.linalg.eigh((matrix + matrix.T) / 2)
        membership = jnp.zeros(values.shape).at[jnp.asarray(keep)].set(1.)
        numerator = membership[None, :] - membership[:, None]
        gaps = values[None, :] - values[:, None]
        factor = numerator / jnp.where(numerator != 0, gaps, 1.)
        rotated = vectors.T @ ((direction + direction.T) / 2) @ vectors
        selected = vectors[:, jnp.asarray(keep)]
        return selected @ selected.T, vectors @ (factor * rotated) @ vectors.T

    return project


def orthonormalize(candidate, overlap):
    metric = candidate.T @ overlap @ candidate
    chol = jnp.linalg.cholesky((metric + metric.T) / 2)
    return jnp.linalg.solve(chol, candidate.T).T


@dataclass(frozen=True)
class BlockFrame:
    anchor: np.ndarray
    clusters: tuple


def select_frame(coeff, fock, gap):
    coeff, fock = np.asarray(coeff), np.asarray(fock)
    values, rotation = np.linalg.eigh(coeff.T @ fock @ coeff)
    boundaries = np.r_[0, np.where(np.diff(values) >= gap)[0] + 1, len(values)]
    clusters = tuple(tuple(range(int(a), int(b)))
                     for a, b in zip(boundaries[:-1], boundaries[1:]))
    return BlockFrame(coeff @ rotation, clusters)


def rebuild_frame(coeff, overlap, fock, frame):
    # C C^T S is the AO projector: arbitrary HF rotations cancel here.
    candidate = coeff @ (coeff.T @ overlap @ jnp.asarray(frame.anchor))
    base = orthonormalize(candidate, overlap)
    projected = base.T @ fock @ base
    blocks = []
    for cluster in frame.clusters:
        projector = spectral_projector(cluster)(projected)
        block = projector[:, jnp.asarray(cluster)]
        blocks.append(base @ orthonormalize(block, jnp.eye(block.shape[0])))
    return jnp.concatenate(blocks, axis=1)


@jax.custom_jvp
def symmetric_exp(matrix):
    values, vectors = jnp.linalg.eigh((matrix + matrix.T) / 2)
    return (vectors * jnp.exp(values)) @ vectors.T


@symmetric_exp.defjvp
def _exp_jvp(primals, tangents):
    (matrix,), (direction,) = primals, tangents
    values, vectors = jnp.linalg.eigh((matrix + matrix.T) / 2)
    distance = jnp.abs(values[:, None] - values[None, :])
    safe = jnp.where(distance > 1e-7, distance, 1.)
    ratio = jnp.where(distance > 1e-7, jnp.expm1(distance) / safe,
                      1 + distance / 2 + distance**2 / 6 + distance**3 / 24)
    divided_difference = jnp.exp(jnp.minimum(values[:, None], values[None, :])) * ratio
    rotated = vectors.T @ ((direction + direction.T) / 2) @ vectors
    result = (vectors * jnp.exp(values)) @ vectors.T
    return result, vectors @ (divided_difference * rotated) @ vectors.T


def block_exp(fock, scale, clusters):
    result = jnp.zeros_like(fock)
    for cluster in clusters:
        ix = jnp.ix_(jnp.asarray(cluster), jnp.asarray(cluster))
        block = fock[ix] * scale
        value = jnp.exp(block) if len(cluster) == 1 else symmetric_exp(block)
        result = result.at[ix].set(value)
    return result


def noncanonical_mp2(eris, prj):
    """MP2 fragment energy with a full-Fock implicit resolvent derivative.

    Diagonalization is internal to the linear solve; AD differentiates the
    full Fock operator, never the eigenvectors used by the solver.
    """
    no = prj.shape[1]
    foo, fvv = eris.fock[:no, :no], eris.fock[no:, no:]
    rhs = jnp.asarray(eris.ovov).transpose(0, 2, 1, 3)

    def operator(t):
        return (jnp.einsum('ac,ijcb->ijab', fvv, t)
                + jnp.einsum('bc,ijac->ijab', fvv, t)
                - jnp.einsum('ki,kjab->ijab', foo, t)
                - jnp.einsum('kj,ikab->ijab', foo, t))

    def solve(_, b):
        eo, uo = jnp.linalg.eigh(foo)
        ev, uv = jnp.linalg.eigh(fvv)
        canonical = jnp.einsum('ip,jq,ar,bs,ijab->pqrs', uo, uo, uv, uv, b,
                                optimize='optimal')
        gap = (ev[None, None, :, None] + ev[None, None, None, :]
               - eo[:, None, None, None] - eo[None, :, None, None])
        return jnp.einsum('ip,jq,ar,bs,pqrs->ijab', uo, uo, uv, uv,
                          canonical / gap, optimize='optimal')

    t2 = jax.lax.custom_linear_solve(operator, -rhs, solve, symmetric=True)
    pair_energy = jnp.einsum('pjab,qjab->pq', t2, 2*rhs-rhs.swapaxes(2, 3))
    energy = jnp.sum(pair_energy * (prj.T @ prj))
    # The singles here are an iteration guess only.
    gap = jnp.diag(foo)[:, None] - jnp.diag(fvv)[None, :]
    t1 = eris.fock[:no, no:] / gap
    return energy, t1, t2
