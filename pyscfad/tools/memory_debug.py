# Copyright 2026 The PySCFAD Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in RSS tracing helpers for expensive tensor builds."""

import os
import resource

import numpy


ENABLED = os.environ.get("PYSCFAD_MEMORY_TRACE", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def current_rss_mb():
    """Return current process RSS in MiB."""
    try:
        with open("/proc/self/status", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.sys.platform == "darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


def _rank_label():
    for key in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "MV2_COMM_WORLD_RANK"):
        if key in os.environ:
            return os.environ[key]
    return "?"


def _shape(value):
    return getattr(value, "shape", None)


def _dtype_itemsize(value):
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    try:
        return numpy.dtype(dtype).itemsize
    except TypeError:
        return None


def array_nbytes(value):
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        try:
            return int(nbytes)
        except TypeError:
            pass

    shape = _shape(value)
    itemsize = _dtype_itemsize(value)
    if shape is None or itemsize is None:
        return None

    size = 1
    for dim in shape:
        try:
            size *= int(dim)
        except TypeError:
            return None
    return size * itemsize


def describe(value):
    shape = _shape(value)
    nbytes = array_nbytes(value)
    if shape is None:
        return "shape=? size=?"
    if nbytes is None:
        return f"shape={tuple(shape)} size=?"
    return f"shape={tuple(shape)} size={nbytes / 1024.0**2:.1f} MiB"


def trace(label, **arrays):
    if not ENABLED:
        return

    fields = [
        f"[memory-trace rank={_rank_label()} pid={os.getpid()}]",
        label,
        f"rss={current_rss_mb():.1f} MiB",
    ]
    for name, value in arrays.items():
        fields.append(f"{name}: {describe(value)}")
    print(" | ".join(fields), flush=True)
