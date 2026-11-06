"""One host-memory ledger for retained DF response panels and block scratch.

DF.jk_workspace_memory (decimal MB) is the shared allowance. Ordinary DF
objects use max_memory. Every allocation is also bounded by process headroom
under precond_total_memory / mol.max_memory. Leave 20% of that headroom for
untracked native/allocator overhead. Legacy panel/solve MiB variables are
optional caps, not independent pools.
"""
from contextlib import contextmanager
import math
import os

from pyscf import lib


def override_bytes(name):
    """An absent/invalid legacy override imposes no additional limit."""
    try:
        value = float(os.environ[name])
    except (KeyError, ValueError):
        return None
    return int(value * 1024**2) if math.isfinite(value) and value >= 0 else None


class Reservation:
    def __init__(self, workspace, size=0, panels=False):
        self.workspace, self.size, self.panels = workspace, 0, panels
        self.resize(size)

    def resize(self, size):
        size = int(size)
        delta = size - self.size
        if size < 0:
            raise ValueError('negative DF workspace reservation')
        if delta > self.workspace.available(panels=self.panels):
            raise MemoryError('DF response workspace cannot accommodate '
                              f'{delta} additional bytes')
        self.workspace.used += delta
        if self.panels:
            self.workspace.panel_used += delta
        self.size = size
        self.workspace.peak = max(self.workspace.peak, self.workspace.used)

    def close(self):
        self.resize(0)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Workspace:
    def __init__(self, dfobj):
        total = getattr(dfobj, 'precond_total_memory', None)
        if total is None:
            total = dfobj.mol.max_memory
        budget = getattr(dfobj, 'jk_workspace_memory', None)
        if budget is None:
            budget = dfobj.max_memory
        if not all(math.isfinite(x) and x > 0 for x in (total, budget)):
            raise ValueError('DF response total/workspace memory must be positive decimal MB')
        self.total_bytes = int(total * 1e6)
        # The initial ceiling also accounts for allocated-but-not-yet-touched
        # arrays: subsequent reservations cannot all reuse the same RSS slack.
        self.limit = min(int(budget * 1e6), self._headroom())
        self.used = self.panel_used = self.peak = 0
        self.geometry_bytes = 0

    def _headroom(self):
        return int(.8 * max(0, self.total_bytes - lib.current_memory()[0] * 1e6))

    def available(self, *, panels=False, reserve=0, cap=None):
        size = min(self.limit - self.used, self._headroom()) - int(reserve)
        if panels:
            panel_cap = override_bytes('PYSCFAD_FACTOR_PANEL_MEMORY_MB')
            if panel_cap is not None:
                size = min(size, panel_cap - self.panel_used)
        if cap is not None:
            size = min(size, cap)
        return max(0, int(size))

    def lease(self, size=0, *, panels=False):
        return Reservation(self, size, panels)

    @contextmanager
    def geometry_buffer(self, size):
        previous = self.geometry_bytes
        with self.lease(max(0, size - previous)):
            self.geometry_bytes = max(size, previous)
            try:
                yield
            finally:
                self.geometry_bytes = previous

    def block(self, bytes_per_item, count, *, cap=None, alignment=1):
        """Use remaining memory; never conceal an impossible minimum block."""
        count = int(count)
        if count == 0:
            return 1
        width = min(count, self.available(cap=cap) // max(1, int(bytes_per_item)))
        if width < 1:
            raise MemoryError('DF response workspace cannot fit one block item '
                              f'({bytes_per_item} bytes needed; '
                              f'{self.available(cap=cap)} bytes available)')
        if width >= alignment and width < count:
            width = width // alignment * alignment
        return width

    def solve_width(self, naux, columns):
        cap = override_bytes('PYSCFAD_SOLVE_MEMORY_MB')
        # Preserve the legacy zero/tiny-cap request for one column, while
        # still requiring the actual shared workspace to fit that column.
        if cap is not None:
            cap = max(cap, 3 * 8 * naux)
        return self.block(3 * 8 * naux, columns, cap=cap, alignment=1024)


def workspace_for(dfobj):
    from . import _ooc_cotangent
    scope = _ooc_cotangent.metric_scope(dfobj)
    return scope.workspace if scope is not None else Workspace(dfobj)


def prepare_fixed_bytes(nao, naux, rank):
    # Metric/factor/adjoint and metric-product temporaries; AO densities,
    # seeds, packed columns and coefficient copies. Deliberately conservative.
    return 8 * (6 * naux**2 + 8 * nao**2 + 4 * nao * rank + 8 * naux)


def retained_fixed_bytes(nao, naux, rank):
    # Include shared metric/factor residuals conservatively until contraction.
    return 8 * (4 * naux**2 + 5 * nao**2 + 3 * nao * rank + 2 * naux)


def transform_row_bytes(nao, rank):
    # Current + prefetched packed B, unpacked B/layout scratch, t/r/products.
    return 8 * (nao * (nao + 1) + 2 * nao**2 + 4 * nao * rank)
