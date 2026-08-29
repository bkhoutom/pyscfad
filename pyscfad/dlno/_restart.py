"""Versioned, crash-safe checkpoints for progressive DLNO gradients.

This module intentionally serializes values, never live PySCF/JAX objects.
Cotangent pytrees are restored against a freshly constructed template so
that Python file handles, custom-VJP closures, pytree auxiliary data, and
JAX ``PyTreeDef`` implementation details never enter a checkpoint.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
import warnings

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

import h5py
import jax
import jax.numpy as jnp
import numpy
from jax.interpreters import ad as jax_ad


__all__ = [
    "RestartCorruptionError",
    "RestartError",
    "RestartManifestError",
    "RestartMismatchError",
    "RestartRecord",
    "RestartSerializationError",
    "RestartManager",
    "df_source_fingerprint",
    "scientific_digest",
]


_FORMAT_NAME = "pyscfad-dlno-restart"
_FORMAT_VERSION = 1
# Increment this whenever MP2/CC/(T) equations, cotangent conventions, or
# saved pytree semantics change without an accompanying payload/static-shape
# change.  It deliberately invalidates every older progressive cotangent.
_ALGORITHM_ABI = 1
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# Tests may monkeypatch this callable.  It is invoked after a temporary HDF5
# file has been closed and fsynced, but before it atomically replaces the
# previous generation.  Raising therefore emulates interruption safely.
_POST_SAVE_TEST_HOOK = None

# Integration tests may use this durable-event hook to stop a calculation at
# an exact checkpoint boundary.  Unlike ``_POST_SAVE_TEST_HOOK``, it runs only
# after the final path has been atomically replaced and the parent directory
# fsynced.
_CHECKPOINT_EVENT_HOOK = None


class RestartError(RuntimeError):
    """Base class for restart failures."""


class RestartManifestError(RestartError):
    """The run manifest is missing, malformed, or used incorrectly."""


class RestartMismatchError(RestartManifestError):
    """The checkpoint belongs to a scientifically different calculation."""


class RestartCorruptionError(RestartError):
    """A stage file is truncated, malformed, or fails an integrity hash."""


class RestartSerializationError(RestartError):
    """A value is unsafe or unsupported for durable serialization."""


@dataclasses.dataclass(frozen=True)
class RestartRecord:
    """Decoded scalar metadata and cotangent trees from one stage."""

    stage: str
    key: object
    scalars: object
    trees: dict[str, object]
    metadata: object
    path: Path


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _as_c_array(array):
    """Return a C-layout host array without promoting a scalar to rank one."""
    array = numpy.asarray(array)
    if array.ndim == 0:
        # numpy.ascontiguousarray intentionally promotes a 0-D array to
        # shape (1,), which changes a scalar cotangent's pytree contract.
        return numpy.array(array, copy=True)
    return numpy.ascontiguousarray(array)


def _array_digest(array):
    array = _as_c_array(array)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _static_registry():
    # Imports are deliberately lazy: the restart helper is imported by the
    # drivers whose dataclasses are registered here.
    from .iao_lis import (
        IAOLISFragmentStaticSelection,
        IAOFragmentLISStaticSelections,
    )
    from .iao_mp2 import IAOFragmentMP2Thresholds
    from .iao_mp2_grad import (
        FixedPAOSubspaceSelection,
        IAOMP2FragmentStaticSelection,
        IAOFragmentMP2StaticSelections,
    )

    classes = (
        IAOFragmentMP2Thresholds,
        FixedPAOSubspaceSelection,
        IAOMP2FragmentStaticSelection,
        IAOFragmentMP2StaticSelections,
        IAOLISFragmentStaticSelection,
        IAOFragmentLISStaticSelections,
    )
    return {
        f"{cls.__module__}:{cls.__qualname__}": cls
        for cls in classes
    }


def _class_tag(cls):
    return f"{cls.__module__}:{cls.__qualname__}"


def _host_array(value):
    if isinstance(value, jax.core.Tracer):
        raise RestartSerializationError("JAX tracers cannot be checkpointed")
    try:
        value = jax.device_get(value)
        array = numpy.asarray(value)
    except Exception as err:
        raise RestartSerializationError(
            f"cannot convert {type(value).__name__} to a host array"
        ) from err
    if array.dtype.hasobject:
        raise RestartSerializationError("object-dtype arrays are unsafe")
    if array.dtype == jax.dtypes.float0:
        return _as_c_array(array)
    if array.dtype.kind not in "biufc":
        raise RestartSerializationError(
            f"unsupported checkpoint array dtype {array.dtype}"
        )
    return _as_c_array(array)


class _DatasetWriter:
    def __init__(self, handle):
        self.group = handle.create_group("arrays", track_order=True)
        self.count = 0

    def add(self, value):
        array = _host_array(value)
        if array.dtype == jax.dtypes.float0:
            raise RestartSerializationError(
                "float0 must be represented as a pytree marker"
            )
        name = f"{self.count:08d}"
        self.count += 1
        self.group.create_dataset(name, data=array, track_times=False)
        return {
            "dataset": name,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": _array_digest(array),
        }


class _DatasetReader:
    def __init__(self, handle):
        try:
            self.group = handle["arrays"]
        except KeyError as err:
            raise RestartCorruptionError("checkpoint has no arrays group") from err

    def get(self, descriptor):
        try:
            name = descriptor["dataset"]
            dataset = self.group[name]
            array = numpy.asarray(dataset[()])
            expected_dtype = numpy.dtype(descriptor["dtype"])
            expected_shape = tuple(int(x) for x in descriptor["shape"])
            expected_hash = descriptor["sha256"]
        except Exception as err:
            raise RestartCorruptionError(
                "invalid checkpoint array descriptor"
            ) from err
        if array.dtype != expected_dtype or array.shape != expected_shape:
            raise RestartCorruptionError(
                f"dataset {name} has dtype/shape {array.dtype}/{array.shape}, "
                f"expected {expected_dtype}/{expected_shape}"
            )
        array = _as_c_array(array)
        if _array_digest(array) != expected_hash:
            raise RestartCorruptionError(
                f"dataset {name} failed its SHA-256 integrity check"
            )
        return array


def _encode_value(value, writer=None, *, require_registered_dataclass=True):
    """Encode a safe value into canonical JSON plus optional HDF5 arrays."""
    if value is None:
        return {"kind": "none"}
    if isinstance(value, (bool, numpy.bool_)):
        return {"kind": "bool", "value": bool(value)}
    if isinstance(value, (int, numpy.integer)):
        scalar = numpy.asarray(value)
        return {
            "kind": "int",
            "value": str(int(value)),
            "dtype": None if isinstance(value, int) else scalar.dtype.str,
        }
    if isinstance(value, (float, numpy.floating)):
        scalar = numpy.asarray(value)
        return {
            "kind": "float",
            "hex": float(value).hex(),
            "dtype": None if isinstance(value, float) else scalar.dtype.str,
        }
    if isinstance(value, (complex, numpy.complexfloating)):
        scalar = numpy.asarray(value)
        return {
            "kind": "complex",
            "real": float(value.real).hex(),
            "imag": float(value.imag).hex(),
            "dtype": None if isinstance(value, complex) else scalar.dtype.str,
        }
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "base64": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, os.PathLike):
        return {"kind": "path", "value": os.fspath(value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        tag = _class_tag(type(value))
        if require_registered_dataclass and tag not in _static_registry():
            raise RestartSerializationError(
                f"dataclass {tag} is not in the safe restart registry"
            )
        return {
            "kind": "dataclass",
            "class": tag,
            "fields": [
                [field.name, _encode_value(
                    getattr(value, field.name), writer,
                    require_registered_dataclass=require_registered_dataclass,
                )]
                for field in dataclasses.fields(value)
            ],
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [
                _encode_value(
                    item, writer,
                    require_registered_dataclass=require_registered_dataclass,
                )
                for item in value
            ],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [
                _encode_value(
                    item, writer,
                    require_registered_dataclass=require_registered_dataclass,
                )
                for item in value
            ],
        }
    if isinstance(value, (set, frozenset)):
        items = [
            _encode_value(
                item, writer,
                require_registered_dataclass=require_registered_dataclass,
            )
            for item in value
        ]
        items.sort(key=_canonical_json)
        return {
            "kind": "frozenset" if isinstance(value, frozenset) else "set",
            "items": items,
        }
    if isinstance(value, dict):
        items = [
            [
                _encode_value(
                    key, writer,
                    require_registered_dataclass=require_registered_dataclass,
                ),
                _encode_value(
                    item, writer,
                    require_registered_dataclass=require_registered_dataclass,
                ),
            ]
            for key, item in value.items()
        ]
        items.sort(key=lambda pair: _canonical_json(pair[0]))
        return {"kind": "dict", "items": items}
    if isinstance(value, jax.core.Tracer):
        raise RestartSerializationError("JAX tracers cannot be checkpointed")
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        array = _host_array(value)
        descriptor = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": _array_digest(array),
        }
        if writer is not None:
            descriptor = writer.add(array)
        return {"kind": "array", **descriptor}
    raise RestartSerializationError(
        f"unsupported checkpoint value {type(value).__module__}."
        f"{type(value).__qualname__}"
    )


def _decode_value(node, reader):
    try:
        kind = node["kind"]
    except Exception as err:
        raise RestartCorruptionError("malformed encoded value") from err
    if kind == "none":
        return None
    if kind == "bool":
        return bool(node["value"])
    if kind == "int":
        value = int(node["value"])
        return value if node.get("dtype") is None else numpy.asarray(
            value, dtype=numpy.dtype(node["dtype"])
        )[()]
    if kind == "float":
        value = float.fromhex(node["hex"])
        return value if node.get("dtype") is None else numpy.asarray(
            value, dtype=numpy.dtype(node["dtype"])
        )[()]
    if kind == "complex":
        value = complex(
            float.fromhex(node["real"]), float.fromhex(node["imag"])
        )
        return value if node.get("dtype") is None else numpy.asarray(
            value, dtype=numpy.dtype(node["dtype"])
        )[()]
    if kind == "str":
        return str(node["value"])
    if kind == "bytes":
        return base64.b64decode(node["base64"], validate=True)
    if kind == "path":
        return Path(node["value"])
    if kind in ("tuple", "list", "set", "frozenset"):
        items = [_decode_value(item, reader) for item in node["items"]]
        if kind == "tuple":
            return tuple(items)
        if kind == "list":
            return items
        if kind == "set":
            return set(items)
        return frozenset(items)
    if kind == "dict":
        return {
            _decode_value(key, reader): _decode_value(value, reader)
            for key, value in node["items"]
        }
    if kind == "array":
        if reader is None:
            raise RestartCorruptionError("array payload has no HDF5 reader")
        return reader.get(node)
    if kind == "dataclass":
        registry = _static_registry()
        tag = node.get("class")
        if tag not in registry:
            raise RestartCorruptionError(
                f"checkpoint requests unregistered dataclass {tag!r}"
            )
        cls = registry[tag]
        expected = tuple(field.name for field in dataclasses.fields(cls))
        fields = node.get("fields")
        if not isinstance(fields, list):
            raise RestartCorruptionError("malformed dataclass fields")
        names = tuple(pair[0] for pair in fields)
        if names != expected:
            raise RestartCorruptionError(
                f"dataclass {tag} fields changed: {names} != {expected}"
            )
        kwargs = {
            name: _decode_value(value, reader)
            for name, value in fields
        }
        try:
            return cls(**kwargs)
        except Exception as err:
            raise RestartCorruptionError(
                f"could not reconstruct safe dataclass {tag}"
            ) from err
    raise RestartCorruptionError(f"unknown encoded value kind {kind!r}")


def scientific_digest(value):
    """Return a stable scientific digest for nested safe values and arrays."""
    encoded = _encode_value(
        value, writer=None, require_registered_dataclass=True
    )
    return _sha256_bytes(_canonical_json(encoded).encode("utf8"))


def _stream_array_hash(array, *, target_bytes=8 * 1024**2):
    """Hash one numeric array in bounded row slabs."""

    dtype = numpy.dtype(array.dtype)
    shape = tuple(int(value) for value in array.shape)
    digest = hashlib.sha256()
    digest.update(dtype.str.encode("ascii"))
    digest.update(_canonical_json(list(shape)).encode("ascii"))
    if not shape:
        block = numpy.asarray(array[()])
        digest.update(_as_c_array(block).tobytes(order="C"))
        return digest.hexdigest()
    row_elements = max(int(numpy.prod(shape[1:], dtype=int)), 1)
    rows = max(int(target_bytes // max(dtype.itemsize * row_elements, 1)), 1)
    for start in range(0, shape[0], rows):
        block = numpy.asarray(array[start:min(start + rows, shape[0])])
        if block.dtype.hasobject or block.dtype.kind not in "biufc":
            raise RestartSerializationError(
                f"unsupported DF array dtype {block.dtype}"
            )
        digest.update(_as_c_array(block).tobytes(order="C"))
    return digest.hexdigest()


def _hdf5_scientific_fingerprint(path):
    """Hash the logical numeric contents of an out-of-core CDERI file."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"density-fitting integral file not found: {path}")
    datasets = []
    with h5py.File(path, "r") as handle:
        names = []
        handle.visititems(
            lambda name, item: names.append(name)
            if isinstance(item, h5py.Dataset) else None
        )
        for name in sorted(names):
            dataset = handle[name]
            datasets.append({
                "name": name,
                "dtype": dataset.dtype.str,
                "shape": list(dataset.shape),
                "sha256": _stream_array_hash(dataset),
            })
    if not datasets:
        raise RestartSerializationError(
            f"density-fitting integral file has no datasets: {path}"
        )
    return {
        "kind": "hdf5",
        # The path is deliberately excluded: an identical copied integral
        # file is scientifically interchangeable with the original.
        "datasets": datasets,
        "sha256": _sha256_bytes(
            _canonical_json(datasets).encode("utf8")
        ),
    }


def df_source_fingerprint(mf):
    """Return a content identity for the DF factors used by ``mf``.

    Both in-memory CDERI arrays and out-of-core HDF5 stores are hashed.  This
    prevents a saved orbital cotangent from being paired with a different
    integral file while still allowing an identical file to be moved.
    """

    with_df = getattr(mf, "with_df", None)
    if with_df is None:
        return {"kind": "none"}
    get_source = getattr(with_df, "_get_cderi_source", None)
    source = get_source() if callable(get_source) else getattr(
        with_df, "_cderi", None
    )
    if source is None:
        return {"kind": "none"}
    if isinstance(source, (str, bytes, os.PathLike)):
        return _hdf5_scientific_fingerprint(os.fsdecode(os.fspath(source)))
    if hasattr(source, "name") and isinstance(
        getattr(source, "name"), (str, bytes, os.PathLike)
    ):
        source_path = Path(os.fsdecode(os.fspath(source.name)))
        if source_path.is_file():
            return _hdf5_scientific_fingerprint(source_path)
    return {
        "kind": "in-memory",
        "sha256": scientific_digest(source),
    }


def _encode_path(path):
    encoded = []
    for key in path:
        if isinstance(key, jax.tree_util.GetAttrKey):
            encoded.append(["attr", key.name])
        elif isinstance(key, jax.tree_util.SequenceKey):
            encoded.append(["sequence", int(key.idx)])
        elif isinstance(key, jax.tree_util.DictKey):
            encoded.append([
                "dict",
                _encode_value(key.key, require_registered_dataclass=False),
            ])
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            encoded.append(["flat", int(key.key)])
        else:
            raise RestartSerializationError(
                f"unsupported JAX pytree path key {type(key).__name__}"
            )
    return encoded


def _path_id(encoded_path):
    return _canonical_json(encoded_path)


def _encode_tree(tree, writer):
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(
        tree, is_leaf=lambda value: value is None
    )
    entries = []
    seen = set()
    for path, leaf in leaves_with_path:
        encoded_path = _encode_path(path)
        identifier = _path_id(encoded_path)
        if identifier in seen:
            raise RestartSerializationError(
                f"pytree contains duplicate path {identifier}"
            )
        seen.add(identifier)
        if leaf is None:
            descriptor = {"kind": "none"}
        elif isinstance(leaf, jax_ad.Zero):
            descriptor = {"kind": "symbolic_zero"}
        elif hasattr(leaf, "dtype") and leaf.dtype == jax.dtypes.float0:
            descriptor = {
                "kind": "float0",
                "shape": list(getattr(leaf, "shape", ())),
            }
        else:
            array = _host_array(leaf)
            if array.dtype.kind in "fc" and not numpy.all(numpy.isfinite(array)):
                raise RestartSerializationError(
                    f"non-finite cotangent at pytree path {identifier}"
                )
            base = {"dtype": array.dtype.str, "shape": list(array.shape)}
            if not numpy.any(array):
                descriptor = {"kind": "zero", **base}
            else:
                descriptor = {"kind": "array", **writer.add(array)}
        entries.append({"path": encoded_path, "leaf": descriptor})
    return {"entries": entries}


def _template_array_metadata(leaf, identifier):
    if leaf is None:
        return None
    if isinstance(leaf, jax_ad.Zero):
        return None
    if not hasattr(leaf, "dtype") or not hasattr(leaf, "shape"):
        raise RestartCorruptionError(
            f"live template leaf {identifier} is not array-like"
        )
    return numpy.dtype(leaf.dtype), tuple(int(x) for x in leaf.shape)


def _array_like_template(array, template):
    if isinstance(template, jax.Array):
        return jnp.asarray(array)
    return array


def _decode_tree(node, reader, template):
    try:
        saved_entries = node["entries"]
    except Exception as err:
        raise RestartCorruptionError("malformed pytree descriptor") from err
    saved = {}
    for entry in saved_entries:
        identifier = _path_id(entry["path"])
        if identifier in saved:
            raise RestartCorruptionError(
                f"checkpoint pytree repeats path {identifier}"
            )
        saved[identifier] = entry["leaf"]

    template_items, treedef = jax.tree_util.tree_flatten_with_path(
        template, is_leaf=lambda value: value is None
    )
    template_ids = [_path_id(_encode_path(path)) for path, _ in template_items]
    if len(template_ids) != len(set(template_ids)):
        raise RestartCorruptionError("live pytree template has duplicate paths")
    if set(saved) != set(template_ids):
        missing = sorted(set(template_ids) - set(saved))[:5]
        extra = sorted(set(saved) - set(template_ids))[:5]
        raise RestartMismatchError(
            "checkpoint/live pytree paths differ; "
            f"missing={missing}, extra={extra}"
        )

    output = []
    for identifier, (_, template_leaf) in zip(template_ids, template_items):
        descriptor = saved[identifier]
        kind = descriptor.get("kind")
        if kind == "none":
            if template_leaf is not None:
                raise RestartMismatchError(
                    f"checkpoint path {identifier} is None but template is not"
                )
            output.append(None)
            continue
        if kind == "float0":
            metadata = _template_array_metadata(template_leaf, identifier)
            if (
                metadata is None
                or metadata[0] != numpy.dtype(jax.dtypes.float0)
                or metadata[1] != tuple(descriptor.get("shape", ()))
            ):
                raise RestartMismatchError(
                    f"float0 template mismatch at {identifier}"
                )
            output.append(template_leaf)
            continue
        if kind == "symbolic_zero":
            if isinstance(template_leaf, jax_ad.Zero):
                output.append(template_leaf)
            else:
                metadata = _template_array_metadata(template_leaf, identifier)
                if metadata is None:
                    raise RestartMismatchError(
                        f"symbolic-zero template mismatch at {identifier}"
                    )
                output.append(jnp.zeros_like(template_leaf))
            continue
        metadata = _template_array_metadata(template_leaf, identifier)
        if metadata is None:
            raise RestartMismatchError(
                f"array checkpoint has non-array template at {identifier}"
            )
        expected_dtype = numpy.dtype(descriptor.get("dtype"))
        expected_shape = tuple(int(x) for x in descriptor.get("shape", ()))
        if metadata != (expected_dtype, expected_shape):
            raise RestartMismatchError(
                f"pytree leaf {identifier} dtype/shape changed: "
                f"checkpoint={(expected_dtype, expected_shape)}, "
                f"live={metadata}"
            )
        if kind == "zero":
            value = jnp.zeros_like(template_leaf) if isinstance(
                template_leaf, jax.Array
            ) else numpy.zeros_like(template_leaf)
        elif kind == "array":
            array = reader.get(descriptor)
            value = _array_like_template(array, template_leaf)
        else:
            raise RestartCorruptionError(
                f"unknown pytree leaf kind {kind!r} at {identifier}"
            )
        output.append(value)
    return jax.tree_util.tree_unflatten(treedef, output)


def _fsync_file(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path):
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _durable_mkdir(path):
    """Create a directory tree and fsync every newly linked parent entry."""

    path = Path(path)
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    path.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        _fsync_directory(created.parent)


def _atomic_bytes(path, data):
    path = Path(path)
    _durable_mkdir(path.parent)
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_hdf5(path, build_metadata):
    path = Path(path)
    _durable_mkdir(path.parent)
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        with h5py.File(temporary, "x") as handle:
            writer = _DatasetWriter(handle)
            metadata = build_metadata(writer)
            digest = _sha256_bytes(_canonical_json(metadata).encode("utf8"))
            envelope = {"metadata": metadata, "sha256": digest}
            text = _canonical_json(envelope)
            dtype = h5py.string_dtype(encoding="utf-8")
            handle.create_dataset("metadata", data=text, dtype=dtype)
            handle.flush()
        _fsync_file(temporary)
        hook = _POST_SAVE_TEST_HOOK
        if hook is not None:
            hook(temporary, path)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_hdf5(path):
    try:
        handle = h5py.File(path, "r")
    except Exception as err:
        raise RestartCorruptionError(
            f"cannot open restart stage {path}"
        ) from err
    try:
        try:
            raw = handle["metadata"][()]
            if isinstance(raw, bytes):
                raw = raw.decode("utf8")
            envelope = json.loads(str(raw))
            metadata = envelope["metadata"]
            expected = envelope["sha256"]
        except Exception as err:
            raise RestartCorruptionError(
                f"restart stage {path} has malformed metadata"
            ) from err
        actual = _sha256_bytes(_canonical_json(metadata).encode("utf8"))
        if actual != expected:
            raise RestartCorruptionError(
                f"restart stage {path} failed its metadata hash"
            )
        return handle, metadata, _DatasetReader(handle)
    except Exception:
        handle.close()
        raise


class RestartManager:
    """Own and validate one directory of progressive restart records.

    ``initialize=True`` is used by a serial calculation or MPI root.  On a
    first run it creates ``run.json``; on resume it validates the existing
    manifest.  MPI workers use ``initialize=False`` after a barrier and open
    the root-created manifest even when the public ``resume`` flag is false.
    """

    def __init__(
        self,
        checkpoint_dir,
        *,
        resume=False,
        method,
        scientific_payload,
        initialize=True,
    ):
        self.resume = bool(resume)
        self.method = str(method)
        self._lease_fd = None
        if not self.method:
            raise ValueError("method must be a nonempty string")
        self.path = (
            None if checkpoint_dir is None
            else Path(checkpoint_dir).expanduser().resolve()
        )
        self.enabled = self.path is not None
        if not self.enabled:
            if self.resume:
                raise ValueError("resume=True requires checkpoint_dir")
            self.base_digest = scientific_digest({
                "abi": _ALGORITHM_ABI,
                "method": self.method,
                "payload": scientific_payload,
            })
            self._manifest = None
            return

        self.base_digest = scientific_digest({
            "abi": _ALGORITHM_ABI,
            "method": self.method,
            "payload": scientific_payload,
        })
        self._base_summary = _encode_value(scientific_payload)
        self.manifest_path = self.path / "run.json"
        if initialize:
            _durable_mkdir(self.path)
            self._acquire_writer_lease()
            try:
                self._initialize_root()
            except Exception:
                self.close()
                raise
        else:
            self._open_existing_manifest()

    def _acquire_writer_lease(self):
        """Hold a process-lifetime advisory lease for the root/serial writer."""

        if fcntl is None:
            return
        descriptor = os.open(
            self.path / ".writer.lock", os.O_CREAT | os.O_RDWR, 0o600
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as err:
            os.close(descriptor)
            raise RestartManifestError(
                "checkpoint directory is already in use by another writer: "
                f"{self.path}"
            ) from err
        self._lease_fd = descriptor

    def close(self):
        """Release this root/serial manager's single-writer lease."""

        descriptor = self._lease_fd
        self._lease_fd = None
        if descriptor is None:
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def static_digest(self):
        if not self.enabled:
            return None
        return self._manifest.get("static_digest")

    def _new_manifest(self):
        return {
            "format": _FORMAT_NAME,
            "format_version": _FORMAT_VERSION,
            "algorithm_abi": _ALGORITHM_ABI,
            "method": self.method,
            "base_digest": self.base_digest,
            "base_summary": self._base_summary,
            "static_digest": None,
        }

    def _initialize_root(self):
        if self.manifest_path.exists():
            if not self.resume:
                raise RestartManifestError(
                    f"checkpoint manifest already exists: {self.manifest_path}; "
                    "pass resume=True to reuse it"
                )
            self._open_existing_manifest()
            return
        if self.resume:
            raise RestartManifestError(
                f"resume requested but manifest is missing: {self.manifest_path}"
            )
        self.path.mkdir(parents=True, exist_ok=True)
        non_temporary = [
            child for child in self.path.iterdir()
            if not child.name.startswith(".")
        ]
        if non_temporary:
            raise RestartManifestError(
                f"checkpoint directory is nonempty but has no run.json: "
                f"{self.path}"
            )
        self._manifest = self._new_manifest()
        self._write_manifest()

    def _open_existing_manifest(self):
        if not self.manifest_path.is_file():
            raise RestartManifestError(
                f"checkpoint manifest is missing: {self.manifest_path}"
            )
        try:
            manifest = json.loads(self.manifest_path.read_text("utf8"))
        except Exception as err:
            raise RestartManifestError(
                f"checkpoint manifest is malformed: {self.manifest_path}"
            ) from err
        expected = {
            "format": _FORMAT_NAME,
            "format_version": _FORMAT_VERSION,
            "algorithm_abi": _ALGORITHM_ABI,
            "method": self.method,
            "base_digest": self.base_digest,
        }
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise RestartMismatchError(
                f"checkpoint manifest does not match this run: {mismatches}"
            )
        if "static_digest" not in manifest:
            raise RestartManifestError(
                "checkpoint manifest has no static_digest field"
            )
        self._manifest = manifest

    def refresh_manifest(self):
        """Reload and validate ``run.json`` after an MPI-root update."""
        if self.enabled:
            self._open_existing_manifest()
        return self

    def _write_manifest(self):
        _atomic_bytes(
            self.manifest_path,
            (_canonical_json(self._manifest) + "\n").encode("utf8"),
        )

    def bind_static(self, value):
        """Validate or establish the fixed-topology/LIS digest."""
        digest = scientific_digest(value)
        if not self.enabled:
            return digest
        current = self.static_digest
        if current is not None and current != digest:
            raise RestartMismatchError(
                "fixed topology/LIS selections differ from the checkpoint"
            )
        if current is None:
            self._manifest["static_digest"] = digest
            self._write_manifest()
        return digest

    @property
    def static_path(self):
        return None if not self.enabled else self.path / "static.h5"

    def save_static(self, value):
        """Atomically save a hardcoded-safe static selection dataclass."""
        digest = scientific_digest(value)
        if not self.enabled:
            return digest
        current = self.static_digest
        if current is not None and current != digest:
            raise RestartMismatchError(
                "refusing to replace checkpoint with different static data"
            )

        def build(writer):
            return {
                "format": _FORMAT_NAME,
                "format_version": _FORMAT_VERSION,
                "kind": "static",
                "method": self.method,
                "base_digest": self.base_digest,
                "static_digest": digest,
                "value": _encode_value(value, writer),
            }

        _atomic_hdf5(self.static_path, build)
        if current is None:
            self._manifest["static_digest"] = digest
            self._write_manifest()
        hook = _CHECKPOINT_EVENT_HOOK
        if hook is not None:
            hook("static", None, self.static_path)
        return digest

    def load_static(self, *, expected_type=None):
        """Load saved static selections, returning ``None`` when absent."""
        if not self.enabled or not self.static_path.is_file():
            return None
        handle, metadata, reader = _read_hdf5(self.static_path)
        try:
            self._validate_stage_header(
                metadata, kind="static", allow_unbound_static=True
            )
            value = _decode_value(metadata.get("value"), reader)
            digest = scientific_digest(value)
            if digest != metadata.get("static_digest"):
                raise RestartCorruptionError(
                    "decoded static selections fail their scientific digest"
                )
            current = self.static_digest
            if current is not None and current != digest:
                raise RestartMismatchError(
                    "static stage and run manifest have different digests"
                )
            if expected_type is not None and not isinstance(value, expected_type):
                raise RestartMismatchError(
                    f"static checkpoint has type {type(value).__name__}, "
                    f"expected {expected_type}"
                )
        finally:
            handle.close()
        if self.static_digest is None:
            self._manifest["static_digest"] = digest
            self._write_manifest()
        return value

    @staticmethod
    def _component(value, label):
        if isinstance(value, numpy.integer):
            value = int(value)
        text = str(value)
        if not _SAFE_COMPONENT.fullmatch(text):
            raise ValueError(
                f"{label} must contain only letters, digits, '.', '_', or '-'"
            )
        return text

    def record_path(self, stage, key=None):
        """Return the deterministic HDF5 path for a stage/key pair."""
        if not self.enabled:
            return None
        stage = self._component(stage, "stage")
        filename = "state.h5" if key is None else (
            self._component(key, "key") + ".h5"
        )
        return self.path / "records" / stage / filename

    def has_record(self, stage, key=None):
        path = self.record_path(stage, key)
        return path is not None and path.is_file()

    def save_record(
        self,
        stage,
        *,
        key=None,
        scalars=None,
        trees=None,
        metadata=None,
    ):
        """Atomically create or replace one progressive stage record."""
        if not self.enabled:
            return None
        stage = self._component(stage, "stage")
        path = self.record_path(stage, key)
        scalars = {} if scalars is None else scalars
        trees = {} if trees is None else dict(trees)
        metadata = {} if metadata is None else metadata
        for name in trees:
            self._component(name, "tree name")

        def build(writer):
            encoded_trees = {
                name: _encode_tree(trees[name], writer)
                for name in sorted(trees)
            }
            return {
                "format": _FORMAT_NAME,
                "format_version": _FORMAT_VERSION,
                "kind": "record",
                "method": self.method,
                "base_digest": self.base_digest,
                "static_digest": self.static_digest,
                "stage": stage,
                "key": _encode_value(key, writer),
                "scalars": _encode_value(scalars, writer),
                "metadata": _encode_value(metadata, writer),
                "trees": encoded_trees,
            }

        _atomic_hdf5(path, build)
        hook = _CHECKPOINT_EVENT_HOOK
        if hook is not None:
            hook(stage, key, path)
        return path

    def load_record(
        self,
        stage,
        *,
        key=None,
        templates=None,
        missing_ok=True,
        on_corrupt="raise",
    ):
        """Load a stage against live tree templates.

        With ``on_corrupt='miss'`` malformed records emit a warning and
        return ``None`` so the caller can recompute that stage.  Scientific
        mismatches are never downgraded to misses.
        """
        if on_corrupt not in ("raise", "miss"):
            raise ValueError("on_corrupt must be 'raise' or 'miss'")
        if not self.enabled:
            return None
        stage = self._component(stage, "stage")
        path = self.record_path(stage, key)
        if not path.is_file():
            if missing_ok:
                return None
            raise FileNotFoundError(path)
        try:
            return self._load_record(path, stage, key, templates or {})
        except RestartCorruptionError as err:
            if on_corrupt == "raise":
                raise
            warnings.warn(
                f"ignoring corrupt restart record {path}; stage will be "
                f"recomputed: {err}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    def _load_record(self, path, stage, key, templates):
        handle, payload, reader = _read_hdf5(path)
        try:
            self._validate_stage_header(payload, kind="record")
            if payload.get("stage") != stage:
                raise RestartCorruptionError(
                    f"record stage is {payload.get('stage')!r}, expected {stage!r}"
                )
            saved_key = _decode_value(payload.get("key"), reader)
            if saved_key != key:
                raise RestartCorruptionError(
                    f"record key is {saved_key!r}, expected {key!r}"
                )
            saved_trees = payload.get("trees")
            if not isinstance(saved_trees, dict):
                raise RestartCorruptionError("record has malformed trees")
            if set(saved_trees) != set(templates):
                raise RestartMismatchError(
                    "record/live tree names differ: "
                    f"checkpoint={sorted(saved_trees)}, "
                    f"live={sorted(templates)}"
                )
            trees = {
                name: _decode_tree(saved_trees[name], reader, templates[name])
                for name in sorted(saved_trees)
            }
            scalars = _decode_value(payload.get("scalars"), reader)
            metadata = _decode_value(payload.get("metadata"), reader)
        finally:
            handle.close()
        return RestartRecord(
            stage=stage,
            key=saved_key,
            scalars=scalars,
            trees=trees,
            metadata=metadata,
            path=path,
        )

    def _validate_stage_header(
        self, payload, *, kind, allow_unbound_static=False
    ):
        expected = {
            "format": _FORMAT_NAME,
            "format_version": _FORMAT_VERSION,
            "kind": kind,
            "method": self.method,
            "base_digest": self.base_digest,
        }
        mismatches = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise RestartMismatchError(
                f"restart stage does not match this run: {mismatches}"
            )
        saved_static = payload.get("static_digest")
        if (
            saved_static != self.static_digest
            and not (allow_unbound_static and self.static_digest is None)
        ):
            raise RestartMismatchError(
                "restart stage and manifest use different static selections"
            )
