"""Imported input cotangents must reproduce a direct numerical pullback."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc.adjoint import pullback_inputs


def _prepare(x, y):
    return {
        "foo": jnp.array([[x + y, x - y], [x - y, y * y + 2.0]]),
        "fvv": jnp.array([[x * y + 3.0]]),
        "B": jnp.stack((x + 2.0 * y, x * y, x - y, y * y)).reshape(2, 2, 1),
    }


def _quadratic(inputs):
    return 0.5 * sum(jnp.sum(value * value) for value in inputs.values())


@pytest.mark.parametrize("seed", [0.0, 1.0, -2.5])
def test_imported_cotangents_reproduce_direct_vjp(seed):
    primals = (jnp.asarray(0.7), jnp.asarray(-0.2))
    saved = jax.tree_util.tree_map(np.asarray, _prepare(*primals))
    result = {
        "energy": float(_quadratic(saved)),
        "cotangents": saved,
        "metadata": {"cotangent_seed": 1.0},
    }
    actual = pullback_inputs(_prepare, primals, saved, result, energy_bar=seed)
    direct = jax.grad(lambda x, y: _quadratic(_prepare(x, y)), argnums=(0, 1))(*primals)
    np.testing.assert_allclose(np.asarray(actual), seed * np.asarray(direct), rtol=1e-12, atol=1e-12)


def test_changed_replay_packet_is_rejected():
    primals = (jnp.asarray(0.7), jnp.asarray(-0.2))
    saved = jax.tree_util.tree_map(np.asarray, _prepare(*primals))
    result = {"energy": 0.0, "cotangents": saved, "metadata": {"cotangent_seed": 1.0}}
    changed = (primals[0] + 0.01, primals[1])
    with pytest.raises(ValueError, match="replay|mismatch|differ"):
        pullback_inputs(_prepare, changed, saved, result)


@pytest.mark.parametrize("occupied_rotation", [
    np.diag([-1.0, 1.0]),
    np.array([[0.0, 1.0], [1.0, 0.0]]),
], ids=["sign_flip", "permutation"])
def test_changed_occupied_frame_is_rejected(occupied_rotation):
    primals = (jnp.asarray(0.7), jnp.asarray(-0.2))
    original = jax.tree_util.tree_map(np.asarray, _prepare(*primals))
    saved = {
        "foo": occupied_rotation.T @ original["foo"] @ occupied_rotation,
        "fvv": original["fvv"].copy(),
        "B": np.einsum("Pia,ij->Pja", original["B"], occupied_rotation),
    }
    # These packets have the same physical tensors in different occupied
    # frames. Cotangents from one frame cannot be applied to the other.
    assert not np.allclose(saved["B"], original["B"])
    result = {
        "energy": 0.0,
        "cotangents": saved,
        "metadata": {"cotangent_seed": 1.0},
    }
    with pytest.raises(ValueError, match="replay|mismatch|differ"):
        pullback_inputs(_prepare, primals, saved, result)


def test_explicit_finite_replay_tolerance_accepts_subnanohartree_drift():
    primals = (jnp.asarray(0.7), jnp.asarray(-0.2))
    exact = jax.tree_util.tree_map(np.asarray, _prepare(*primals))
    saved = {name: value.copy() for name, value in exact.items()}
    saved["B"][1, 1, 0] += 5e-10
    result = {
        "energy": 0.0,
        "cotangents": exact,
        "metadata": {"cotangent_seed": 1.0},
    }
    with pytest.raises(ValueError, match="replay mismatch"):
        pullback_inputs(_prepare, primals, saved, result)
    actual = pullback_inputs(
        _prepare, primals, saved, result, replay_atol=1e-9
    )
    expected = jax.vjp(_prepare, *primals)[1](exact)
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=1e-12, atol=1e-12
    )


def test_explicit_finite_replay_tolerance_still_rejects_changed_frame():
    primals = (jnp.asarray(0.7), jnp.asarray(-0.2))
    exact = jax.tree_util.tree_map(np.asarray, _prepare(*primals))
    changed = {name: value.copy() for name, value in exact.items()}
    changed["B"][1, 1, 0] += 1e-3
    result = {
        "energy": 0.0,
        "cotangents": exact,
        "metadata": {"cotangent_seed": 1.0},
    }
    with pytest.raises(ValueError, match="replay mismatch"):
        pullback_inputs(
            _prepare, primals, changed, result, replay_atol=1e-9
        )
