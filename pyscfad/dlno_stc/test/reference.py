"""Tiny deterministic and sampled test backends, never production STC solvers."""

import jax
import jax.numpy as jnp
import numpy as np


_KEYS = ("foo", "fvv", "B")
_IDENTITY = ("schema_version", "method", "energy_kind", "fragment_id",
             "target_index", "basis_frame", "B_axes", "request_fingerprint")


def _symmetric(a):
    return (a + a.T) / 2


def quadratic_energy(inputs):
    """Half squared norm, with symmetric Fock dependence."""
    foo = _symmetric(inputs["foo"])
    fvv = _symmetric(inputs["fvv"])
    return 0.5 * (jnp.sum(foo * foo) + jnp.sum(fvv * fvv) +
                  jnp.sum(inputs["B"] * inputs["B"]))


def target_mp2_energy(inputs, target_index):
    """Dense noncanonical target MP2 oracle in the supplied local frame.

    Amplitudes are solved in a semicanonical frame, then rotated back so
    the fixed target index retains its meaning in the original frame.
    """
    foo, fvv, B = inputs["foo"], inputs["fvv"], inputs["B"]
    if foo.shape[0] == 0:
        raise ValueError("no active occupied orbitals")
    if not 0 <= target_index < foo.shape[0]:
        raise ValueError("target_index outside occupied space")
    if fvv.shape[0] == 0:
        return jnp.sum(B) * 0.0
    eo, Uo = jnp.linalg.eigh(_symmetric(foo))
    ev, Uv = jnp.linalg.eigh(_symmetric(fvv))
    rotated_B = jnp.einsum("Pia,ip,aq->Ppq", B, Uo, Uv)
    g_semi = jnp.einsum("Pia,Pjb->ijab", rotated_B, rotated_B)
    denom = (eo[:, None, None, None] + eo[None, :, None, None]
             - ev[None, None, :, None] - ev[None, None, None, :])
    t_semi = g_semi / denom
    t = jnp.einsum("ip,jq,ar,bs,pqrs->ijab", Uo, Uo, Uv, Uv, t_semi)
    g = jnp.einsum("Pia,Pjb->ijab", B, B)
    return jnp.einsum("jab,jab->", t[target_index],
                      2 * g[target_index] - g[target_index].transpose(0, 2, 1))


def _result(energy, cotangents, metadata, *, backend, sample_count=0,
            diagnostics=None, seed_replay=None):
    result_metadata = {key: metadata[key] for key in _IDENTITY if key in metadata}
    result_metadata.update(cotangent_seed=1.0, backend=backend,
                           backend_version="1", actual_sample_count=int(sample_count),
                           derivative_convention="fixed_sample",
                           seed_replay=(
                               {"mode": "deterministic"}
                               if seed_replay is None else seed_replay
                           ))
    return {"energy": float(energy),
            "cotangents": {key: np.asarray(cotangents[key], dtype=np.float64)
                           for key in _KEYS},
            "metadata": result_metadata,
            "diagnostics": {} if diagnostics is None else diagnostics}


def reference_backend(inputs, metadata, controls, *, comm=None):
    """Return one deterministic result; a supplied communicator has root ownership."""
    if comm is not None and comm.Get_rank() != 0:
        return None
    replay = {"mode": "deterministic"}
    if "global_seed" in controls:
        replay["global_seed"] = controls["global_seed"]
    kind = metadata["energy_kind"]
    if kind == "quadratic_test_v1":
        energy_fn = quadratic_energy
    elif kind == "boys_target_mp2_v1":
        energy_fn = lambda x: target_mp2_energy(x, metadata["target_index"])
    elif kind == "whole_domain_mp2_test_v1":
        energy_fn = lambda x: sum(target_mp2_energy(x, f)
                                  for f in range(inputs["foo"].shape[0]))
    else:
        raise ValueError(f"unsupported energy_kind: {kind}")
    if kind != "quadratic_test_v1" and inputs["foo"].shape[0] == 0:
        raise ValueError("no active occupied orbitals")
    if inputs["fvv"].shape[0] == 0 and kind != "quadratic_test_v1":
        zeros = {key: np.zeros_like(inputs[key]) for key in _KEYS}
        return _result(0.0, zeros, metadata, backend="tiny_reference",
                       seed_replay=replay)
    jax_inputs = {key: jnp.asarray(inputs[key]) for key in _KEYS}
    energy, gradients = jax.value_and_grad(energy_fn)(jax_inputs)
    return _result(energy, gradients, metadata, backend="tiny_reference",
                   seed_replay=replay)


def sampled_backend(inputs, metadata, controls, *, comm=None):
    """Fixed logical block mean plus one deterministic quadratic term.

    Blocks are assigned by block number modulo communicator size. Each block
    contributes its unnormalized value and derivative, including ranks with
    zero blocks. This is a test estimator, not an STC approximation.
    """
    nblocks = controls["sample_blocks"]
    seed = controls.get("global_seed", 0)
    if comm is None:
        rank, size = 0, 1
    else:
        rank, size = comm.Get_rank(), comm.Get_size()
    blocks = range(rank, nblocks, size)
    coefficients = [1.0 + 0.25 * np.sin(seed + block) for block in blocks]
    local_count = np.array([len(coefficients)], dtype=np.int64)
    local_factor = sum(coefficients)
    quadratic = float(quadratic_energy(inputs))
    local_energy = np.array([local_factor * quadratic], dtype=np.float64)
    symmetric_inputs = {"foo": np.asarray(_symmetric(inputs["foo"])),
                        "fvv": np.asarray(_symmetric(inputs["fvv"])),
                        "B": np.asarray(inputs["B"])}
    local_gradients = {key: np.ascontiguousarray(local_factor * symmetric_inputs[key])
                       for key in _KEYS}
    if comm is None:
        total_count, total_energy = local_count, local_energy
        total_gradients = local_gradients
    else:
        total_count = np.empty_like(local_count) if rank == 0 else None
        total_energy = np.empty_like(local_energy) if rank == 0 else None
        total_gradients = {key: np.empty_like(local_gradients[key]) for key in _KEYS} if rank == 0 else {}
        comm.Reduce(local_count, total_count, root=0)
        comm.Reduce(local_energy, total_energy, root=0)
        for key in _KEYS:
            comm.Reduce(local_gradients[key],
                        total_gradients[key] if rank == 0 else None, root=0)
    if rank != 0:
        return None
    if total_count[0] == 0:
        raise ValueError("sample_blocks must be positive")
    gradients = {key: 0.125 * symmetric_inputs[key] + total_gradients[key] / total_count[0]
                 for key in _KEYS}
    energy = 0.125 * quadratic + total_energy[0] / total_count[0]
    return _result(energy, gradients, metadata, backend="sampled_test",
                   sample_count=total_count[0],
                   diagnostics={"logical_blocks": int(nblocks)},
                   seed_replay={"mode": "logical_blocks",
                                "global_seed": int(seed),
                                "sample_blocks": int(nblocks)})
