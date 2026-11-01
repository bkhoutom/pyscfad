"""Independent JAX matrix-function reference for native dressing tests only."""
import jax
import jax.numpy as jnp

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


@jax.jit
def dress(t1, t2, Bvo, Boo, Bvv, fov, foo, fvv, node):
    o = symmetric_exp(foo * (node / 2))
    v = symmetric_exp(fvv * (-node / 2))
    oi = symmetric_exp(foo * (-node / 2))
    vi = symmetric_exp(fvv * (node / 2))
    return (o @ t1 @ v,
            jnp.einsum('ip,jq,ar,bs,pqrs->ijab', o, o, v, v, t2,
                       optimize='optimal'),
            jnp.einsum('ij,Pjb,ba->Pia', o, Bvo, v, optimize='optimal'),
            jnp.einsum('ij,Pjk,kl->Pil', o, Boo, oi, optimize='optimal'),
            jnp.einsum('ab,Pbc,cd->Pad', v, Bvv, vi, optimize='optimal'),
            o @ fov @ v)

