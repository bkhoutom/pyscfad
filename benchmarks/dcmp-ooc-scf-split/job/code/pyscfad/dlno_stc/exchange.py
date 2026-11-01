"""Atomic HDF5 exchange of one domain request and one combined result."""

import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

from .protocol import (
    ARRAY_NAMES, SCHEMA_VERSION, _check_digest, _validate_result_structure,
    canonical_json, request_fingerprint, validate_inputs, validate_result,
)


def _write_json(handle, name, value):
    handle.create_dataset(name, data=canonical_json(value),
                          dtype=h5py.string_dtype(encoding="utf-8"))


def _read_json(handle, name):
    try:
        encoded = handle[name][()]
    except KeyError as error:
        raise ValueError(f"missing /{name} JSON dataset") from error
    try:
        return json.loads(encoded.decode("utf-8") if isinstance(encoded, bytes)
                          else encoded)
    except (ValueError, TypeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid /{name} JSON dataset") from error


def _require_complete(handle, kind):
    if handle.attrs.get("complete", False) != True:
        raise ValueError(f"incomplete {kind} file")
    if handle.attrs.get("file_kind") != f"dlno_stc_{kind}":
        raise ValueError(f"wrong {kind} file kind")
    if handle.attrs.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported {kind} schema_version")


def _atomic_write(path, writer, *, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.",
                                             suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["complete"] = False
            writer(handle)
            handle.attrs["complete"] = True
            handle.flush()
        if overwrite:
            os.replace(temporary, path)
        else:
            # A hard link publishes a closed file atomically and fails if
            # another writer published the destination after our first check.
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_input(path, inputs, metadata, controls, *, overwrite=False):
    """Write a completed request and return its SHA-256 fingerprint."""
    validate_inputs(inputs, metadata)
    fingerprint = request_fingerprint(inputs, metadata, controls)

    def writer(handle):
        handle.attrs["file_kind"] = "dlno_stc_request"
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["request_fingerprint"] = fingerprint
        _write_json(handle, "metadata", metadata)
        _write_json(handle, "controls", controls)
        group = handle.create_group("inputs")
        for name in ARRAY_NAMES:
            group.create_dataset(name, data=inputs[name], dtype="float64")

    _atomic_write(path, writer, overwrite=overwrite)
    return fingerprint


def read_input(path):
    """Read a complete request, checking its arrays against the saved hash."""
    with h5py.File(path, "r") as handle:
        _require_complete(handle, "request")
        metadata = _read_json(handle, "metadata")
        controls = _read_json(handle, "controls")
        fingerprint = handle.attrs.get("request_fingerprint")
        _check_digest(fingerprint)
        try:
            group = handle["inputs"]
            inputs = {name: group[name][()] for name in ARRAY_NAMES}
        except KeyError as error:
            raise ValueError(f"request missing {error}") from error
    validate_inputs(inputs, metadata)
    if request_fingerprint(inputs, metadata, controls) != fingerprint:
        raise ValueError("request fingerprint does not match saved arrays or controls")
    metadata = {**metadata, "request_fingerprint": fingerprint}
    return inputs, metadata, controls


def write_result(path, result, *, overwrite=False):
    """Write one combined scalar energy and its unit-seed cotangents."""
    _validate_result_structure(result)

    def writer(handle):
        handle.attrs["file_kind"] = "dlno_stc_result"
        handle.attrs["schema_version"] = SCHEMA_VERSION
        _write_json(handle, "metadata", result["metadata"])
        _write_json(handle, "diagnostics", result["diagnostics"])
        handle.create_dataset("energy", data=result["energy"], dtype="float64")
        group = handle.create_group("cotangents")
        for name in ARRAY_NAMES:
            group.create_dataset(name, data=result["cotangents"][name],
                                 dtype="float64")

    _atomic_write(path, writer, overwrite=overwrite)


def read_result(path, inputs, metadata, *, controls=None,
                expected_fingerprint=None):
    """Read and validate result against the original request and frame.

    ``metadata`` may be the verified value returned by ``read_input``; its
    saved fingerprint is then a trusted match token when controls are absent.
    """
    validate_inputs(inputs, metadata)
    if expected_fingerprint is None and controls is None:
        expected_fingerprint = metadata.get("request_fingerprint")
    if controls is None and expected_fingerprint is None:
        raise ValueError("controls or a trusted expected_fingerprint is required")
    if controls is not None:
        computed = request_fingerprint(inputs, metadata, controls)
        if expected_fingerprint is not None and expected_fingerprint != computed:
            raise ValueError("trusted request fingerprint disagrees with controls")
        expected_fingerprint = computed
    _check_digest(expected_fingerprint)
    with h5py.File(path, "r") as handle:
        _require_complete(handle, "result")
        result_metadata = _read_json(handle, "metadata")
        if result_metadata.get("request_fingerprint") != expected_fingerprint:
            raise ValueError("result request fingerprint does not match request")
        diagnostics = _read_json(handle, "diagnostics")
        try:
            energy_dataset = handle["energy"]
            group = handle["cotangents"]
            if energy_dataset.shape != () or energy_dataset.dtype != np.dtype("float64"):
                raise ValueError("energy must be a float64 scalar")
            for name in ARRAY_NAMES:
                dataset = group[name]
                if dataset.shape != inputs[name].shape:
                    raise ValueError(f"{name} cotangent shape does not match request")
                if dataset.dtype != np.dtype("float64"):
                    raise ValueError(f"{name} cotangent must have float64 dtype")
            result = {
                "energy": energy_dataset[()],
                "cotangents": {name: group[name][()] for name in ARRAY_NAMES},
                "metadata": result_metadata,
                "diagnostics": diagnostics,
            }
        except KeyError as error:
            raise ValueError(f"result missing {error}") from error
    validate_result(result, inputs, metadata, controls=controls,
                    expected_fingerprint=expected_fingerprint)
    return result
