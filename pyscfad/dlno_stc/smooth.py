"""Anchored LIS frames and full occupied/virtual Fock matrix functions.

Ranks and AO anchors are chosen outside differentiation. Frame transport retains
the physical LIS and overlap response, without internal Fock eigenspace response.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import numpy as np


def orthonormalize(candidate, overlap):
    metric = candidate.T @ overlap @ candidate
    chol = jnp.linalg.cholesky((metric + metric.T) / 2)
    return solve_triangular(chol, candidate.T, lower=True).T


@dataclass(frozen=True)
class AnchoredFrame:
    anchor: np.ndarray


def select_frame(coeff, fock):
    """Choose a semicanonical AO anchor once, outside differentiation."""
    coeff, fock = np.asarray(coeff), np.asarray(fock)
    _, rotation = np.linalg.eigh(coeff.T @ fock @ coeff)
    return AnchoredFrame(coeff @ rotation)


def rebuild_frame(coeff, overlap, frame):
    """Transport the LIS/overlap response; the Fock-frame cotangent is zero.

    One identity Fock projector removes every Fock-dependent frame rotation.
    It does not freeze the physical LIS or its AO metric, which still determine
    the projected anchor's normalization.
    """
    # C C^T S is the AO projector: arbitrary HF rotations cancel here.
    candidate = coeff @ (coeff.T @ overlap @ jnp.asarray(frame.anchor))
    return orthonormalize(candidate, overlap)


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
