"""Replay local numerical preparation and apply imported cotangents."""

import math

import jax
from pyscfad import numpy as np

from .protocol import check_replay, validate_full_result


def pullback_inputs(prepare_fn, primals, saved_inputs, result, *,
                    energy_bar=1.0, replay_atol=1e-11):
    """Return VJP cotangents for ``primals`` from one domain's unit seed.

    ``prepare_fn`` has no file or MPI operations. The imported result must have
    already passed request/result validation at the exchange boundary.
    The caller may set replay_atol for measured numerical replay drift;
    the full-support path retains the strict default.
    """
    if not isinstance(primals, tuple):
        raise TypeError("primals must be a tuple")
    if not math.isfinite(float(energy_bar)):
        raise ValueError("energy_bar must be finite")
    if float(result.get("metadata", {}).get("cotangent_seed", 0.0)) != 1.0:
        raise ValueError("imported cotangents require cotangent_seed=1.0")

    replayed, vjp = jax.vjp(prepare_fn, *primals)
    check_replay(replayed, saved_inputs, atol=replay_atol)
    if set(result["cotangents"]) != set(replayed):
        raise ValueError("imported cotangent keys differ from replayed inputs")
    cotangents = {
        name: np.asarray(result["cotangents"][name]) * energy_bar
        for name in replayed
    }
    return vjp(cotangents)


def _apply_preparation_pullback(pullback, inputs, result, validate, energy_bar):
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
    validate(result, inputs)
    cotangents = {
        name: np.asarray(result["cotangents"][name], dtype=value.dtype) * seed
        for name, value in inputs.items()
    }
    return pullback(cotangents)


def apply_full_pullback(pullback, inputs, result, *, energy_bar=1.0):
    """Apply three bars to one saved whole-system preparation VJP."""
    return _apply_preparation_pullback(pullback, inputs, result,
                                       validate_full_result, energy_bar)


def apply_selected_pullback(pullback, inputs, result, *, energy_bar=1.0):
    """Apply exactly three bars to a fixed selected-target Boys preparation.

    Selected indices are discrete solver metadata. Their numerical packet
    has the same three-array validation contract as a whole-system result.
    """
    return _apply_preparation_pullback(pullback, inputs, result,
                                       validate_full_result, energy_bar)


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
