"""Bounded, host-only factor caches for one implicit SCF root solve.

The optional budget covers the existing occupied/virtual preconditioner
factors, occupied panels, coefficient snapshots and row-validity masks.
Mandatory metric matrices, solver vectors and block scratch are excluded.
The full AO-pair DF tensor remains on disk. Nothing survives ``root_scope``.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import os

import numpy

_current = ContextVar('pyscfad_scf_response_cache', default=None)
_recorders = ContextVar('pyscfad_scf_response_cache_recorders', default=())


def _enabled():
    return os.environ.get('PYSCFAD_SCF_OCCUPIED_CACHE', '0').strip().lower() in (
        '1', 'true', 'yes', 'on')


def _budget_bytes():
    try:
        value = float(os.environ.get('PYSCFAD_SCF_RESPONSE_FACTOR_CACHE_MB', '1024'))
        if not numpy.isfinite(value) or value < 0:
            value = 1024.
    except ValueError:
        value = 1024.
    return int(value * 1024**2)


class _Ledger:
    def __init__(self, budget):
        self.budget = budget
        self.used = 0
        self.peak = 0

    def reserve(self, size):
        if self.used + size > self.budget:
            return False
        self.used += size
        self.peak = max(self.peak, self.used)
        return True


class _Panel:
    def __init__(self, scope, key, coeff, naux):
        self.scope = scope
        self.key = key
        self.coeff = numpy.array(coeff, copy=True)
        self.data = numpy.empty((naux, coeff.shape[1], coeff.shape[0]))
        self.valid = numpy.zeros(naux, dtype=bool)
        self.nbytes = self.data.nbytes + self.coeff.nbytes + self.valid.nbytes
        self.closed = False

    def _bounds(self, start, stop):
        return not self.closed and 0 <= start < stop <= len(self.valid)

    def lookup_block(self, start, stop):
        if not self._bounds(start, stop):
            return None
        stats = self.scope.stats
        if not self.valid[start:stop].all():
            stats['cache_miss_blocks'] += 1
            return None
        stats['cache_hit_blocks'] += 1
        stats['cache_hit_rows'] += stop - start
        return self.data[start:stop].reshape(-1, self.data.shape[2])

    def store_block(self, start, stop, values):
        if not self._bounds(start, stop):
            return
        target = self.data[start:stop].reshape(-1, self.data.shape[2])
        if values.shape != target.shape or values.dtype != target.dtype:
            return
        target[:] = values
        self.valid[start:stop] = True

    def close(self):
        if not self.closed:
            self.data = self.coeff = self.valid = None
            self.scope.ledger.used -= self.nbytes
            self.closed = True


class _Scope:
    def __init__(self, enabled, parent):
        requested = _enabled() and enabled
        # Nested solves share the reservation ledger, but do not populate
        # additional occupied panels or reuse their parent's panel.
        self.ledger = (parent.ledger if parent is not None and parent.ledger is not None
                       else _Ledger(_budget_bytes()) if requested else None)
        self.enabled = requested and parent is None
        self.panel = None
        self.invalidated = False
        self.preconditioner_bytes = 0
        self.stats = dict(enabled=self.enabled, requested=requested,
                          nested=parent is not None,
                          budget_bytes=self.ledger.budget if self.ledger else 0,
                          preconditioner_cache_bytes=0, occupied_panel_bytes=0,
                          coefficient_bytes=0, validity_bytes=0,
                          expected_occupied_bytes=0, expected_optional_bytes=0,
                          peak_optional_bytes=self.ledger.used if self.ledger else 0,
                          cache_hit_blocks=0, cache_hit_rows=0, cache_miss_blocks=0,
                          occupied_transform_calls=0, response_transform_calls=0,
                          rejected_reservations=0,
                          invalidations=0, finished=False, failed=False)

    def reserve(self, size):
        if not self.ledger.reserve(size):
            self.stats['rejected_reservations'] += 1
            return False
        self.stats['peak_optional_bytes'] = max(
            self.stats['peak_optional_bytes'], self.ledger.used)
        return True

    def close(self):
        if self.panel is not None:
            self.panel.close()
            self.panel = None
        if self.ledger is not None:
            self.stats['peak_optional_bytes'] = max(
                self.stats['peak_optional_bytes'], self.ledger.peak)
            self.ledger.used -= self.preconditioner_bytes
        self.stats['finished'] = True


@contextmanager
def record_response_caches():
    """Yield JSON-compatible receipts for every root completed in this scope."""
    receipts = []
    token = _recorders.set(_recorders.get() + (receipts,))
    try:
        yield receipts
    finally:
        _recorders.reset(token)


@contextmanager
def root_scope(enabled=True):
    """Keep host factors only through linearization, preconditioning and solve."""
    scope = _Scope(enabled, _current.get())
    token = _current.set(scope)
    try:
        yield scope
    except BaseException:
        scope.stats['failed'] = True
        raise
    finally:
        scope.close()
        _current.reset(token)
        for recorder in _recorders.get():
            recorder.append(dict(scope.stats))


def reserve_preconditioner(size):
    """Reserve optional B_ov storage; retain the ordinary policy when disabled."""
    scope = _current.get()
    if scope is None or scope.ledger is None:
        return True
    if size <= 0 or not scope.reserve(size):
        return False
    scope.preconditioner_bytes += size
    scope.stats['preconditioner_cache_bytes'] += size
    return True


def record_transforms(occupied):
    """Count native work in both enabled and ordinary root scopes."""
    scope = _current.get()
    if scope is not None:
        scope.stats['occupied_transform_calls'] += int(occupied)
        scope.stats['response_transform_calls'] += 1


def get_panel(dfobj, coeff):
    """Return a validated OOC occupied panel, or preserve the streaming path."""
    scope = _current.get()
    if scope is None or not scope.enabled or scope.invalidated:
        return None
    if (not isinstance(coeff, numpy.ndarray) or coeff.dtype != numpy.float64
            or coeff.ndim != 2 or not all(coeff.shape)
            or not numpy.isfinite(coeff).all()):
        return None
    # Function-local import avoids a lib -> df package import cycle.
    from pyscfad.df._ooc_cotangent import _source_key
    key = _source_key(dfobj)
    panel = scope.panel
    if panel is not None:
        if (key == panel.key and coeff.shape == panel.coeff.shape
                and numpy.array_equal(coeff, panel.coeff)):
            return panel
        panel.close()
        scope.panel = None
        scope.invalidated = True
        scope.stats['invalidations'] += 1
        return None
    if key is None:
        return None
    naux = dfobj.get_naoaux()
    occupied_bytes = naux * coeff.size * coeff.itemsize
    required = occupied_bytes + coeff.nbytes + naux
    scope.stats['expected_occupied_bytes'] = occupied_bytes
    scope.stats['expected_optional_bytes'] = required + scope.preconditioner_bytes
    if not naux or not scope.reserve(required):
        return None
    try:
        panel = _Panel(scope, key, coeff, naux)
    except MemoryError:
        scope.ledger.used -= required
        scope.invalidated = True
        return None
    scope.panel = panel
    scope.stats.update(occupied_panel_bytes=occupied_bytes,
                       coefficient_bytes=coeff.nbytes, validity_bytes=naux)
    return panel
