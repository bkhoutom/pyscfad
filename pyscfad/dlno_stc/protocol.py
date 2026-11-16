"""Host-side validation for the real, restricted local MP2 array contract."""

import hashlib
import itertools
import json
import math
import os

import numpy as np


ARRAY_NAMES = ("foo", "fvv", "B")
_BLOCK_ELEMENTS = 1 << 16


def _json_value(value):
    """Convert small metadata to a portable JSON value without stringifying it."""
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("metadata and controls keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata and controls must contain finite numbers")
        return value
    raise ValueError(f"metadata and controls cannot encode {type(value).__name__}")


def canonical_json(value):
    """Encode small metadata consistently for HDF5 and SHA-256."""
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _check_mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dictionary")


def _block_slices(shape, max_elements=_BLOCK_ELEMENTS):
    """Yield contiguous C-order slices containing at most max_elements values."""
    total = math.prod(shape)
    if total == 0:
        return
    if total <= max_elements:
        yield (slice(None),) * len(shape)
        return
    for axis in range(len(shape)):
        trailing = math.prod(shape[axis + 1:])
        if trailing <= max_elements:
            width = max(1, max_elements // trailing)
            for prefix in itertools.product(*(range(size) for size in shape[:axis])):
                for start in range(0, shape[axis], width):
                    yield prefix + (slice(start, min(start + width, shape[axis])),) + (slice(None),) * (len(shape) - axis - 1)
            return


def _check_array(value, name, shape=None, symmetric=False):
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise ValueError(f"{name} must be a float64 array")
    if np.dtype(value.dtype) != np.dtype("float64"):
        raise ValueError(f"{name} must have float64 dtype")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} shape {value.shape} does not match {shape}")
    for slices in _block_slices(value.shape):
        if not np.isfinite(np.asarray(value[slices])).all():
            raise ValueError(f"{name} must contain finite values")
    if symmetric:
        if len(value.shape) != 2 or value.shape[0] != value.shape[1]:
            raise ValueError(f"{name} shape must be square")
        # Full matrices are retained. Only numerical roundoff is tolerated.
        if not np.allclose(value, np.asarray(value).T, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{name} must be symmetric")


def _update_input_digest(digest, inputs):
    """Hash tensors in bounded blocks, using a stable byte convention."""
    for name in ARRAY_NAMES:
        array = inputs[name]
        digest.update(name.encode("ascii") + b"\0")
        digest.update(canonical_json({"shape": array.shape, "dtype": "float64"}).encode("ascii"))
        for slices in _block_slices(array.shape):
            block = np.asarray(array[slices], dtype="<f8", order="C")
            digest.update(block.tobytes(order="C"))


def matrix_request_fingerprint(inputs, metadata, controls, aux_offsets):
    """Identify the frame, native version and resolved options of a matrix call."""
    validate_full_inputs(inputs)
    _check_mapping(metadata, "matrix metadata")
    _check_mapping(controls, "matrix controls")
    digest = hashlib.sha256(b"dlno_stc_matrix_request_v2\0")
    digest.update(canonical_json({
        "metadata": metadata, "controls": controls, "matrix_kind": "W",
        "aux_offsets": aux_offsets,
    }).encode("utf-8"))
    _update_input_digest(digest, inputs)
    return digest.hexdigest()


def validate_full_inputs(inputs):
    """Check real tensors without any legacy packet metadata."""
    _check_mapping(inputs, "inputs")
    if set(inputs) != set(ARRAY_NAMES):
        raise ValueError("inputs must contain exactly " + ", ".join(ARRAY_NAMES))
    for name, value in inputs.items():
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise ValueError(f"{name} must be a float64 array")
    foo, fvv, B = (inputs[name] for name in ARRAY_NAMES)
    if len(foo.shape) != 2 or len(fvv.shape) != 2 or len(B.shape) != 3:
        raise ValueError("input shape ranks must be foo(oo), fvv(vv), B(Pia)")
    nocc, nvir = foo.shape[0], fvv.shape[0]
    if nocc == 0:
        raise ValueError("empty occupied space is a host-side no-work case")
    _check_array(foo, "foo", (nocc, nocc), symmetric=True)
    _check_array(fvv, "fvv", (nvir, nvir), symmetric=True)
    _check_array(B, "B", (B.shape[0], nocc, nvir))


def validate_full_result(result, inputs, *, with_grad=True):
    _check_mapping(result, "result")
    for name in ("energy", "energy_standard_error", "diagnostics"):
        if name not in result:
            raise ValueError(f"result missing {name}")
    for name in ("energy", "energy_standard_error"):
        value = np.asarray(result[name])
        if (value.shape != () or value.dtype != np.dtype("float64")
                or not np.isfinite(value)):
            raise ValueError(f"{name} must be a finite float64 scalar")
        if name == "energy_standard_error" and value < 0:
            raise ValueError("energy_standard_error must be nonnegative")
    _check_mapping(result["diagnostics"], "diagnostics")
    cotangents = result.get("cotangents")
    if not with_grad and (cotangents is None or
                          isinstance(cotangents, dict) and not cotangents):
        return
    _check_mapping(cotangents, "cotangents")
    if set(cotangents) != set(ARRAY_NAMES):
        raise ValueError("cotangents must contain exactly " + ", ".join(ARRAY_NAMES))
    for name in ARRAY_NAMES:
        _check_array(cotangents[name], name + " cotangent", inputs[name].shape,
                     symmetric=name in ("foo", "fvv"))


def validate_input_cotangents(cotangents, inputs):
    """Check already-seeded bars independently of the scalar energy contract."""
    _check_mapping(cotangents, "input cotangents")
    if set(cotangents) != set(ARRAY_NAMES):
        raise ValueError("input cotangents must contain exactly foo, fvv, and B")
    for name in ARRAY_NAMES:
        _check_array(cotangents[name], name + " cotangent", inputs[name].shape,
                     symmetric=name in ("foo", "fvv"))


def validate_matrix_result(result, inputs):
    """Validate matrix primals without requiring scalar unit-energy bars."""
    _check_mapping(result, "matrix result")
    for field in ("energy", "occupied", "virtual", "diagnostics", "replay"):
        if field not in result:
            raise ValueError(f"matrix result missing {field}")
    energy = np.asarray(result["energy"])
    if (energy.shape != () or energy.dtype != np.dtype("float64")
            or not np.isfinite(energy)):
        raise ValueError("matrix energy must be a finite float64 scalar")
    _check_array(result["occupied"], "occupied matrix", inputs["foo"].shape,
                 symmetric=True)
    _check_array(result["virtual"], "virtual matrix", inputs["fvv"].shape,
                 symmetric=True)
    _check_mapping(result["diagnostics"], "matrix diagnostics")
    diagnostics = result["diagnostics"]
    for name in ("energy_standard_error", "occupied_frobenius_se", "virtual_frobenius_se"):
        value = np.asarray(diagnostics.get(name))
        if (value.shape != () or value.dtype.kind not in "fi"
                or not np.isfinite(value) or value < 0):
            raise ValueError(f"matrix {name} must be finite and nonnegative")
    for name, source in (("occupied", "foo"), ("virtual", "fvv")):
        error = diagnostics.get(name+"_standard_error")
        _check_array(error, name+" standard error", inputs[source].shape)
        if np.any(error < 0):
            raise ValueError(f"{name} standard error must be nonnegative")
