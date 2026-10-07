"""Replay local numerical preparation and apply imported cotangents."""

import math

import jax
from pyscfad import numpy as np

from .protocol import check_replay, validate_full_result, validate_weighted_result


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


def apply_weighted_pullback(pullback, inputs, result, *, energy_bar=1.0):
    """Apply all five bars to one saved weighted preparation VJP."""
    return _apply_preparation_pullback(pullback, inputs, result,
                                       validate_weighted_result, energy_bar)


def apply_full_pullback(pullback, inputs, result, *, energy_bar=1.0):
    """Apply three bars to one saved whole-system preparation VJP."""
    return _apply_preparation_pullback(pullback, inputs, result,
                                       validate_full_result, energy_bar)
