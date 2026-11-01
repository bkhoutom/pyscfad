"""Physical retained/discarded D/W subspace response for truncated LIS ranks.

This is independent of the internal orbital gauge: full-Fock smooth frames have
no Fock cluster projectors, but a truncated LIS must still follow its D/W space.
"""
from functools import lru_cache

import jax
import jax.numpy as jnp


@lru_cache(None)
def retained_projector(keep):
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
