"""Apply solver cotangents to saved local preparation."""

import math

import jax
from pyscfad import numpy as np

from .protocol import validate_full_result, validate_input_cotangents


def apply_selected_pullback(pullback, inputs, result, *, energy_bar=1.0):
    """Apply unit-seed bars to an already-created preparation VJP.

    ``inputs`` is the differentiable output returned alongside ``pullback`` by
    ``jax.vjp``. Cast host bars to those output dtypes before applying the
    scalar seed; the existing pullback preserves its primal tree and dtypes.
    This immediate path neither rebuilds preparation nor replays a packet.
    """
    seed = np.asarray(energy_bar)
    if seed.shape != ():
        raise ValueError("energy_bar must be a scalar")
    if not math.isfinite(float(seed)):
        raise ValueError("energy_bar must be finite")
    validate_full_result(result, inputs)
    cotangents = {
        name: np.asarray(result["cotangents"][name], dtype=value.dtype) * seed
        for name, value in inputs.items()
    }
    return pullback(cotangents)


def apply_input_pullback(pullback, inputs, cotangents):
    """Apply matrix VJP bars, which already include all output cotangent seeds."""
    validate_input_cotangents(cotangents, inputs)
    return pullback({name: np.asarray(cotangents[name], dtype=value.dtype)
                     for name, value in inputs.items()})


def fix_restart_mo_phases(mf):
    """Fix canonical phases inside SCF response, copied from native MP2.

    This small numerical helper intentionally has no native DLNO dependency.
    Near-tied maximal AO coefficients use the first row, preserving the
    existing restart gauge even under small converged-SCF roundoff.
    """
    import jax.numpy as jnp

    coeff = jnp.asarray(mf.mo_coeff)
    coeff_abs = jnp.abs(coeff)
    largest = jnp.max(coeff_abs, axis=0)
    near_largest = coeff_abs >= (largest[None, :] * (1.0 - 1e-10))
    pivot_rows = jnp.argmax(near_largest, axis=0)
    pivot = coeff[pivot_rows, jnp.arange(coeff.shape[1])]
    pivot_abs = jnp.abs(pivot)
    phase = jnp.where(pivot_abs > 0, jnp.conj(pivot) / pivot_abs,
                      jnp.ones_like(pivot))
    mf.mo_coeff = coeff * phase[None, :]
    return mf


def add_cotangent(left, right):
    """Add live numerical bars, preserving native MP2's None/float0 rules."""
    if left is None:
        return right
    if right is None:
        return left
    if hasattr(left, "dtype") and left.dtype == jax.dtypes.float0:
        return right
    if hasattr(right, "dtype") and right.dtype == jax.dtypes.float0:
        return left
    return left + right


def zero_mf_cotangent(mf, dtype):
    """Construct a live MF-shaped zero tree using the primal's own VJP."""
    _, pullback = jax.vjp(lambda current: np.zeros((), dtype=dtype), mf)
    mf_bar, = pullback(np.ones((), dtype=dtype))
    return mf_bar


def zero_term_cotangents(mf, common):
    """Construct live zero MF/common trees for one separately released term.

    Copied from native MP2's zero-tree helper; this retains molecule and DF
    leaf conventions instead of manufacturing detached numerical records.
    """
    _, pullback = jax.vjp(
        lambda current_mf, current_common: np.zeros((), dtype=current_common.s1e.dtype),
        mf, common,
    )
    return pullback(np.ones((), dtype=common.s1e.dtype))
