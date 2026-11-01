# Copyright 2026 The PySCFAD Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Experimental, explicitly scoped resident DF cotangent accumulation.

Set ``PYSCFAD_DF_SHARED_COTANGENT=1`` before tracing the producer, then wrap
one complete reverse chain in ``reverse_scope()``. A scope asserts that each
registered factor's consumer cotangents pass unchanged to its producer: for
example MP2 -> SCF -> DF. Arbitrarily scaling/intercepting intermediate
pullback results inside a scope is unsupported. Ordinary public VJPs outside
a scope always return explicit cotangents, even with a live producer tape.

Only weak producer metadata persists between calls; writable buffers belong
to one reverse invocation. Repeated seeds require separate scopes. Nested
explicit scopes are rejected; nested implicit roots use the enclosing scope.
Successful exit requires every allocated buffer to have reached its producer.
This is an eager, first-order experiment, not a general JAX transformation.

Adapted from upstream PySCFAD PR #180's resident cotangent buffer. Unlike the
upstream global accumulator, buffers here cannot survive a reverse invocation.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import os
import weakref

import numpy

_producers = {}
_scope = ContextVar('pyscfad_df_cotangent_reverse_scope', default=None)
_suspended = ContextVar('pyscfad_df_cotangent_suspended', default=False)


def _enabled():
    return os.environ.get('PYSCFAD_DF_SHARED_COTANGENT', '0').lower() in (
        '1', 'true', 'yes', 'on')


def _owner(x):
    while isinstance(x.base, numpy.ndarray):
        x = x.base
    return x


class _Producer:
    def __init__(self, x):
        self.owner = weakref.ref(_owner(x))
        self.shape = x.shape
        self.strides = x.strides
        self.dtype = x.dtype


class ReverseScope:
    """Allocation receipts retained after the writable buffers are released."""
    def __init__(self, enabled):
        self.enabled = enabled
        self.buffers_created = 0
        self.buffer_bytes = 0
        self.peak_bytes = 0
        self.consumer_uses = 0
        self.producer_pops = 0
        self._live_bytes = 0
        self._buffers = {}


@contextmanager
def reverse_scope():
    """Own one complete reverse chain; reject nested or unfinished chains."""
    if _scope.get() is not None:
        raise RuntimeError('nested shared cotangent reverse scopes are unsupported')
    scope = ReverseScope(_enabled())
    token = _scope.set(scope)
    try:
        yield scope
        if scope._buffers:
            raise RuntimeError(
                'unconsumed shared cotangent buffers: reverse_scope must include '
                'the DF producer pullback')
    finally:
        scope._buffers.clear()
        scope._live_bytes = 0
        _scope.reset(token)


@contextmanager
def suspend():
    """Keep local bars explicit when a nested derivative does not reach the producer.

    In particular, an implicit root's Jacobian matvec can invoke another root
    whose external pullback returns bars discarded by the outer matvec. This
    marker is independent of the local implicit-phase flags and stays active
    through all such nested derivatives.
    """
    token = _suspended.set(True)
    try:
        yield
    finally:
        _suspended.reset(token)

def register(x):
    """Remember a resident producer, without allocating a cotangent buffer."""
    if (not _enabled() or not isinstance(x, numpy.ndarray)
            or x.dtype != numpy.float64 or not x.flags.c_contiguous
            or x.size == 0):
        return
    key = x.__array_interface__['data'][0]
    existing = _lookup(x)
    if existing is not None:
        return
    producer = _Producer(x)
    _producers[key] = producer

    def release():
        # A stale owner must never remove a newer allocation at this address.
        if _producers.get(key) is producer:
            del _producers[key]
    weakref.finalize(_owner(x), release)


def _lookup(x):
    if not isinstance(x, numpy.ndarray) or not x.flags.c_contiguous:
        return None
    producer = _producers.get(x.__array_interface__['data'][0])
    if (producer is None or producer.owner() is not _owner(x)
            or producer.shape != x.shape or producer.strides != x.strides
            or producer.dtype != x.dtype):
        return None
    return producer


def registered(x):
    """Whether this reverse invocation may redirect the factor cotangent."""
    scope = _scope.get()
    return (scope is not None and scope.enabled and not _suspended.get()
            and _lookup(x) is not None)


def get(x):
    """Return this invocation's additive buffer, or retain the explicit path."""
    scope = _scope.get()
    if scope is None or not scope.enabled or _suspended.get():
        return None
    producer = _lookup(x)
    if producer is None:
        return None
    data = scope._buffers.get(producer)
    if data is None:
        data = numpy.zeros(producer.shape, dtype=producer.dtype)
        scope._buffers[producer] = data
        scope.buffers_created += 1
        scope.buffer_bytes += data.nbytes
        scope._live_bytes += data.nbytes
        scope.peak_bytes = max(scope.peak_bytes, scope._live_bytes)
    scope.consumer_uses += 1
    return data


def pop(x):
    """Transfer ownership to the producer's destructive integral transpose."""
    scope = _scope.get()
    if scope is None or not scope.enabled or _suspended.get():
        return None
    producer = _lookup(x)
    data = scope._buffers.pop(producer, None)
    if data is not None:
        scope.producer_pops += 1
        scope._live_bytes -= data.nbytes
    return data
