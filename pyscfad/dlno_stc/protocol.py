"""Host-side validation for the real, restricted local MP2 array contract."""

import hashlib
import itertools
import json
import math
import os

import numpy as np


SCHEMA_VERSION = 1
ENERGY_KINDS = frozenset((
    "boys_target_mp2_v1", "quadratic_test_v1", "whole_domain_mp2_test_v1",
    "finite_three_tensor_v1",
))
ARRAY_NAMES = ("foo", "fvv", "B")
WEIGHTED_ARRAY_NAMES = (
    "foo", "fvv", "B", "target_projection", "partner_weight",
)
_IDENTITY_FIELDS = (
    "schema_version", "method", "energy_kind", "fragment_id", "target_index",
    "basis_frame", "B_axes",
)
_DESCRIPTIVE_FIELDS = (
    "units", "frozen", "orbital_order", "auxiliary_order",
    "checkpoint_id", "domain_options", "code_revision", "dirty_worktree",
)
_NON_FINGERPRINT_FIELDS = frozenset((
    "request_fingerprint", "timestamp", "created_at", "completed_at",
))
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


def _check_metadata(metadata, nocc=None):
    _check_mapping(metadata, "metadata")
    for field in _IDENTITY_FIELDS + _DESCRIPTIVE_FIELDS:
        if field not in metadata:
            raise ValueError(f"metadata missing {field}")
    if type(metadata["schema_version"]) is not int or metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if metadata["method"] != "mp2":
        raise ValueError("method must be mp2")
    if metadata["energy_kind"] not in ENERGY_KINDS:
        raise ValueError("unsupported energy_kind")
    fragment_id = metadata["fragment_id"]
    if (not isinstance(fragment_id, (int, np.integer))
            or isinstance(fragment_id, (bool, np.bool_)) or fragment_id < 0):
        raise ValueError("fragment_id must be a nonnegative integer")
    finite = metadata["energy_kind"] == "finite_three_tensor_v1"
    target_index = metadata["target_index"]
    if finite:
        if target_index is not None:
            raise ValueError("finite three-tensor packets have no local target_index")
        if metadata["basis_frame"] != "orthonormal_local":
            raise ValueError("finite three-tensor packets require an orthonormal_local frame")
    else:
        if (not isinstance(target_index, (int, np.integer))
                or isinstance(target_index, (bool, np.bool_)) or target_index < 0):
            raise ValueError("target_index must be a nonnegative integer")
        if nocc is not None and target_index >= nocc:
            raise ValueError("target_index is outside the occupied space")
        if metadata["basis_frame"] != "orthonormal_local":
            raise ValueError("basis_frame must be orthonormal_local")
    if metadata["B_axes"] != "Pia":
        raise ValueError("B_axes must be Pia")
    units = metadata["units"]
    if (not isinstance(units, dict) or units.get("energy") != "Eh"
            or units.get("length") != "bohr"):
        raise ValueError("units must specify energy in Eh and length in bohr")
    if not isinstance(metadata["domain_options"], dict):
        raise ValueError("domain_options must be a dictionary")
    if metadata["dirty_worktree"] is not None and not isinstance(metadata["dirty_worktree"], bool):
        raise ValueError("dirty_worktree must be boolean or unknown")
    canonical_json(metadata)


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


def validate_inputs(inputs, metadata):
    """Check three real float64 tensors in one orthonormal local frame."""
    _check_mapping(inputs, "inputs")
    if set(inputs) != set(ARRAY_NAMES):
        raise ValueError("inputs must contain exactly foo, fvv, and B")
    foo, fvv, B = (inputs[name] for name in ARRAY_NAMES)
    for name, value in inputs.items():
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise ValueError(f"{name} must be a float64 array")
    if len(foo.shape) != 2 or len(fvv.shape) != 2 or len(B.shape) != 3:
        raise ValueError("input shape ranks must be foo(oo), fvv(vv), B(Pia)")
    nocc, nvir = foo.shape[0], fvv.shape[0]
    if nocc == 0:
        raise ValueError("empty occupied space is a host-side no-work case")
    _check_metadata(metadata, nocc)
    _check_array(foo, "foo", (nocc, nocc), symmetric=True)
    _check_array(fvv, "fvv", (nvir, nvir), symmetric=True)
    _check_array(B, "B", (B.shape[0], nocc, nvir))


def request_fingerprint(inputs, metadata, controls):
    """Hash the problem and requested controls, streaming numeric blocks."""
    validate_inputs(inputs, metadata)
    _check_mapping(controls, "controls")
    problem = {key: value for key, value in metadata.items()
               if key not in _NON_FINGERPRINT_FIELDS}
    requested = {key: value for key, value in controls.items()
                 if key not in _NON_FINGERPRINT_FIELDS}
    digest = hashlib.sha256()
    digest.update(b"dlno_stc_request_v1\0")
    digest.update(canonical_json({"metadata": problem, "controls": requested}).encode("utf-8"))
    for name in ARRAY_NAMES:
        array = inputs[name]
        digest.update(name.encode("ascii") + b"\0")
        digest.update(canonical_json({"shape": array.shape, "dtype": "float64"}).encode("ascii"))
        for slices in _block_slices(array.shape):
            block = np.asarray(array[slices], dtype="<f8", order="C")
            digest.update(block.tobytes(order="C"))
    return digest.hexdigest()


def _check_digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("request_fingerprint must be a lowercase SHA-256 digest")


def _validate_result_structure(result):
    """Checks possible before request arrays are available (also used on write)."""
    _check_mapping(result, "result")
    for field in ("energy", "cotangents", "metadata", "diagnostics"):
        if field not in result:
            raise ValueError(f"result missing {field}")
    energy = np.asarray(result["energy"])
    if energy.shape != () or energy.dtype != np.dtype("float64") or not np.isfinite(energy):
        raise ValueError("energy must be a finite float64 scalar")
    cotangents = result["cotangents"]
    _check_mapping(cotangents, "cotangents")
    if set(cotangents) != set(ARRAY_NAMES):
        raise ValueError("cotangents must contain exactly foo, fvv, and B")
    for name in ARRAY_NAMES:
        _check_array(cotangents[name], name + " cotangent",
                     symmetric=name in ("foo", "fvv"))
    md = result["metadata"]
    _check_mapping(md, "result metadata")
    for field in _IDENTITY_FIELDS + (
        "request_fingerprint", "cotangent_seed", "backend", "backend_version",
        "actual_sample_count", "derivative_convention", "seed_replay",
    ):
        if field not in md:
            raise ValueError(f"result metadata missing {field}")
    _check_digest(md["request_fingerprint"])
    if type(md["cotangent_seed"]) not in (float, int) or md["cotangent_seed"] != 1.0:
        raise ValueError("cotangent_seed must be 1.0")
    if md["derivative_convention"] not in ("fixed_sample", "expected_energy_estimate"):
        raise ValueError("unsupported derivative_convention")
    count = md["actual_sample_count"]
    if not isinstance(count, (int, np.integer)) or isinstance(count, (bool, np.bool_)) or count < 0:
        raise ValueError("actual_sample_count must be nonnegative")
    if not isinstance(md["backend"], str) or not md["backend"]:
        raise ValueError("backend must be identified")
    if not isinstance(md["backend_version"], str) or not md["backend_version"]:
        raise ValueError("backend_version must be identified")
    if not isinstance(md["seed_replay"], dict) or not md["seed_replay"]:
        raise ValueError("seed_replay must be a nonempty dictionary")
    _check_mapping(result["diagnostics"], "diagnostics")
    canonical_json(md)
    canonical_json(result["diagnostics"])


def validate_result(result, inputs, metadata, *, controls=None,
                    expected_fingerprint=None):
    """Reject stale, wrong-frame, or malformed unit-seed cotangents.

    Supply the original controls or a trusted fingerprint returned by
    ``write_input``/``read_input``. The result's own echo is never trusted.
    """
    validate_inputs(inputs, metadata)
    _validate_result_structure(result)
    if controls is None and expected_fingerprint is None:
        raise ValueError("controls or a trusted expected_fingerprint is required")
    if controls is not None:
        computed = request_fingerprint(inputs, metadata, controls)
        if expected_fingerprint is not None and expected_fingerprint != computed:
            raise ValueError("trusted request fingerprint disagrees with controls")
        expected_fingerprint = computed
    _check_digest(expected_fingerprint)
    result_md = result["metadata"]
    for field in _IDENTITY_FIELDS:
        if result_md[field] != metadata[field]:
            raise ValueError(f"result {field} does not match request")
    if result_md["request_fingerprint"] != expected_fingerprint:
        raise ValueError("result request fingerprint does not match request")
    if controls is not None:
        blocks = controls.get("sample_blocks")
        if blocks is not None:
            if not isinstance(blocks, (int, np.integer)) or isinstance(blocks, (bool, np.bool_)) or blocks < 0:
                raise ValueError("sample_blocks must be a nonnegative integer")
            if blocks > 0 and result_md["actual_sample_count"] == 0:
                raise ValueError("actual_sample_count must be positive for a stochastic request")
        if "global_seed" in controls:
            if result_md["seed_replay"].get("global_seed") != controls["global_seed"]:
                raise ValueError("seed_replay.global_seed does not match request")
    for name in ARRAY_NAMES:
        _check_array(result["cotangents"][name], name + " cotangent",
                     inputs[name].shape, symmetric=name in ("foo", "fvv"))


def check_replay(replayed, saved, *, rtol=1e-9, atol=1e-11):
    """Compare reconstructed local tensors without copying a whole DF array."""
    _check_mapping(replayed, "replayed inputs")
    _check_mapping(saved, "saved inputs")
    if set(replayed) != set(ARRAY_NAMES) or set(saved) != set(ARRAY_NAMES):
        raise ValueError("replay mismatch: expected foo, fvv, and B")
    if not (math.isfinite(rtol) and math.isfinite(atol) and rtol >= 0 and atol >= 0):
        raise ValueError("replay tolerances must be finite and nonnegative")
    for name in ARRAY_NAMES:
        current, reference = replayed[name], saved[name]
        if tuple(current.shape) != tuple(reference.shape):
            raise ValueError(f"replay mismatch: {name} shape changed")
        if np.dtype(current.dtype) != np.dtype("float64") or np.dtype(reference.dtype) != np.dtype("float64"):
            raise ValueError(f"replay mismatch: {name} must be float64")
        for slices in _block_slices(reference.shape):
            a, b = np.asarray(current[slices]), np.asarray(reference[slices])
            if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a, b, rtol=rtol, atol=atol):
                raise ValueError(f"replay mismatch: {name} changed in its local frame")


def _validate_memory_inputs(inputs, names):
    """Check real tensors without any legacy packet metadata."""
    _check_mapping(inputs, "inputs")
    if set(inputs) != set(names):
        raise ValueError("inputs must contain exactly " + ", ".join(names))
    for name, value in inputs.items():
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise ValueError(f"{name} must be a float64 array")
    foo, fvv, B = (inputs[name] for name in ARRAY_NAMES)
    if len(foo.shape) != 2 or len(fvv.shape) != 2 or len(B.shape) != 3:
        raise ValueError("input shape ranks must be foo(oo), fvv(vv), B(Pia)")
    nocc, nvir = foo.shape[0], fvv.shape[0]
    if nocc == 0:
        if names == WEIGHTED_ARRAY_NAMES:
            raise ValueError("weighted inputs require a nonempty occupied space")
        raise ValueError("empty occupied space is a host-side no-work case")
    _check_array(foo, "foo", (nocc, nocc), symmetric=True)
    _check_array(fvv, "fvv", (nvir, nvir), symmetric=True)
    _check_array(B, "B", (B.shape[0], nocc, nvir))
    return nocc


def validate_full_inputs(inputs):
    """Validate the three-array real float64 whole-system boundary."""
    _validate_memory_inputs(inputs, ARRAY_NAMES)


def validate_weighted_inputs(inputs):
    """Validate five in-memory arrays; empty virtual spaces are valid."""
    nocc = _validate_memory_inputs(inputs, WEIGHTED_ARRAY_NAMES)
    _check_array(inputs["target_projection"], "target_projection", (1, nocc))
    _check_array(inputs["partner_weight"], "partner_weight",
                 (nocc, nocc), symmetric=True)


def _validate_memory_result(result, inputs, names, *, with_grad):
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
    if set(cotangents) != set(names):
        raise ValueError("cotangents must contain exactly " + ", ".join(names))
    for name in names:
        _check_array(cotangents[name], name + " cotangent", inputs[name].shape,
                     symmetric=name in ("foo", "fvv", "partner_weight"))


def validate_full_result(result, inputs, *, with_grad=True):
    """Check whole-system energy and its three unit-seed numerical bars."""
    validate_full_inputs(inputs)
    _validate_memory_result(result, inputs, ARRAY_NAMES, with_grad=with_grad)


def validate_weighted_result(result, inputs, *, with_grad=True):
    """Check weighted energy and its five unit-seed numerical bars."""
    validate_weighted_inputs(inputs)
    _validate_memory_result(result, inputs, WEIGHTED_ARRAY_NAMES, with_grad=with_grad)
