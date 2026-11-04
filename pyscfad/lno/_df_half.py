"""Resident half-transformed inputs for each ED/LIS forward and pullback.

Each new orbital frame streams raw AO integrals once, retaining H[P,mu,p]
= sum_nu (P|mu nu) C[nu,p] and the fitted MO-pair output. The same H supplies
both coefficient legs of the analytic reverse, including directions outside
the current orbital space. No raw or half-transformed integral is spooled.
All later frames must lie in the first ED span, and their
half transforms are rotations of H_ED. Only the first transform reads raw
integrals. Each consumer retains its ordinary analytic coefficient VJP;
geometry and metric adjoints are combined by the owning scope.
"""
import os
from types import SimpleNamespace

import numpy
import scipy.linalg
from pyscf import lib
from pyscfad import numpy as np
from pyscfad.lib._threading import dense_blas_threads


class HalfTransformedDFCache:
    def __init__(self, scope):
        self.scope = scope
        self.entries = []
        self.parent = None
        # As in SCF response, resident data uses the total process allowance;
        # the independent DF workspace only bounds temporary tiles in _rows.
        limit = float(os.environ.get('PYSCFAD_LNO_HALF_DF_CACHE_MB',
                                     scope.max_memory_mb))
        if not numpy.isfinite(limit) or limit < 0:
            raise ValueError('PYSCFAD_LNO_HALF_DF_CACHE_MB must be finite and nonnegative')
        self.limit_bytes = min(limit, scope.max_memory_mb) * 1e6

    def _find(self, coeff, slices):
        for entry in self.entries:
            if (entry.slices == tuple(slices) and entry.coeff.shape == coeff.shape
                    and numpy.array_equal(entry.coeff, coeff)):
                return entry
        return None

    def _rows(self, words, reserve_bytes=0):
        from ._df_direct import _local_direct_int3c_block_mb
        remaining = (self.scope.max_memory_mb - lib.current_memory()[0]) * 1e6
        budget = min(self.scope.workspace_mb * 1e6,
                     .5 * max(0., remaining - reserve_bytes),
                     _local_direct_int3c_block_mb() * 1e6)
        rows = int(budget // (8 * max(1, words)))
        if rows < 1:
            raise MemoryError('Insufficient user memory budget for one half-transformed DF tile')
        return min(self.scope.auxmol.nao, rows), budget

    def transform(self, coeff, slices, low):
        if self.scope.finished:
            raise RuntimeError('Cannot use a finished local DF reverse')
        coeff = numpy.asarray(coeff)
        if coeff.dtype != numpy.float64 or not numpy.isfinite(coeff).all():
            raise TypeError('Half-transformed DF inputs require finite real float64 orbitals')
        entry = self._find(coeff, slices)
        if entry is not None:
            self.scope.receipt['half_cache_hits'] += 1
            entry.consumed = False
            return entry.fitted

        from ._df_direct import _local_direct_raw_int3c_blocks
        scope = self.scope
        n, naux = scope.mol.nao, scope.auxmol.nao
        k0, k1, l0, l1 = slices
        kc, lc = k1-k0, l1-l0
        # Store an overlapping pair of orbital slices only once (LIS uses
        # the same full slice on both legs). ED uses disjoint occupied/virtual.
        columns = numpy.union1d(numpy.arange(k0, k1), numpy.arange(l0, l1))
        width = len(columns)
        parent = self.parent
        rotation = None
        if parent is not None and kc*lc:
            basis = parent.coeff[:, parent.columns]
            selected = coeff[:, columns]
            with dense_blas_threads():
                rotation = scipy.linalg.lstsq(basis, selected, check_finite=False)[0]
                error = numpy.linalg.norm(basis @ rotation - selected)
            if error > 1e-11 * max(1., numpy.linalg.norm(selected)):
                raise ValueError('Domain LIS leaves the retained ED orbital span; '
                                 'cannot reuse H_ED without a new integral pass')
            if parent.h is None:
                raise RuntimeError('Missing parent H_ED for domain LIS transformation')
        left = slice(int(numpy.searchsorted(columns, k0)),
                     int(numpy.searchsorted(columns, k0)) + kc)
        right = slice(int(numpy.searchsorted(columns, l0)),
                      int(numpy.searchsorted(columns, l0)) + lc)
        h_bytes = 8*naux*n*width if kc*lc else 0
        fitted_bytes = 8*naux*kc*lc
        retained = h_bytes + fitted_bytes + coeff.nbytes
        current = scope.receipt['half_resident_bytes']
        remaining = (scope.max_memory_mb - lib.current_memory()[0]) * 1e6
        # Include a second MO output during conversion to the AD backend,
        # and leave half the RSS headroom for CC tensors and response work.
        reserve = retained + fitted_bytes + coeff.nbytes
        if current + retained > self.limit_bytes or reserve > .5*remaining:
            raise MemoryError(
                'Resident half-transformed DF cache exceeds the user memory budget '
                f'(new H/output {retained/1e6:.1f} MB, live cache {current/1e6:.1f} MB). '
                'Increase the memory allowance or use PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE=direct; '
                'half mode does not spill integral inputs to disk.')
        npair = n*(n+1)//2
        # Integral tile, unpacked AO tile, half transform and MO tile.
        words = (3*npair + n*n + 2*n*width + 2*kc*lc
                 if rotation is None else
                 2*n*width + 2*kc*lc + rotation.size)
        rows, _ = self._rows(words, reserve + (0 if rotation is None else rotation.nbytes))
        h = numpy.empty((naux, n, width)) if kc*lc else None
        result = numpy.empty((naux, kc*lc), order='F')
        if kc*lc and rotation is not None:
            for p0 in range(0, naux, rows):
                p1 = min(p0+rows, naux)
                with dense_blas_threads():
                    numpy.matmul(parent.h[p0:p1], rotation, out=h[p0:p1])
                    result[p0:p1] = (coeff[:, k0:k1].T @ h[p0:p1, :, right]).reshape(p1-p0, -1)
        elif kc*lc:
            selected = numpy.ascontiguousarray(coeff[:, columns])
            for p0, p1, raw in _local_direct_raw_int3c_blocks(
                    scope.mol, scope.auxmol, rows):
                # Integral generation and unpacking use OpenMP; keep their
                # surrounding BLAS policy separate from these dense products.
                full = lib.unpack_tril(numpy.ascontiguousarray(raw))
                with dense_blas_threads():
                    numpy.matmul(full, selected, out=h[p0:p1])
                    result[p0:p1] = (coeff[:, k0:k1].T @ h[p0:p1, :, right]).reshape(p1-p0, -1)
                del full, raw
            del selected
        with dense_blas_threads():
            result = scipy.linalg.solve_triangular(
                low, result, lower=True, overwrite_b=True, check_finite=False)
        fitted = np.asarray(result)
        entry = SimpleNamespace(
            coeff=numpy.array(coeff, copy=True), slices=tuple(slices),
            left=left, right=right, h=h, fitted=fitted, consumed=False,
            columns=columns, nbytes=retained)
        self.entries.append(entry)
        if self.parent is None:
            self.parent = entry
        scope.receipt['half_builds' if parent is None else 'half_rotations'] += 1
        scope.receipt['half_resident_bytes'] += retained
        scope.receipt['half_peak_bytes'] = max(scope.receipt['half_peak_bytes'],
                                              scope.receipt['half_resident_bytes'])
        return fitted

    def coefficient_pullback(self, coeff, read_z, slices):
        if self.scope.finished:
            raise RuntimeError('Cannot use a finished local DF reverse')
        entry = self._find(numpy.asarray(coeff), slices)
        if entry is None:
            raise RuntimeError('Missing resident half-transformed DF forward inputs')
        k0, k1, l0, l1 = slices
        kc, lc = k1-k0, l1-l0
        bar = numpy.zeros_like(coeff)
        if kc*lc:
            n = self.scope.mol.nao
            rows, _ = self._rows(2*n*entry.h.shape[2] + 3*kc*lc,
                                 reserve_bytes=bar.nbytes)
            for p0 in range(0, self.scope.auxmol.nao, rows):
                p1 = min(p0+rows, self.scope.auxmol.nao)
                z = numpy.asarray(read_z(p0, p1)).reshape(p1-p0, kc, lc)
                h = entry.h[p0:p1]
                with dense_blas_threads():
                    bar[:, k0:k1] += numpy.einsum(
                        'pml,pkl->mk', h[:, :, entry.right], z, optimize=True)
                    bar[:, l0:l1] += numpy.einsum(
                        'pmk,pkl->ml', h[:, :, entry.left], z, optimize=True)
        entry.consumed = True
        self.scope.receipt['half_pullbacks'] += 1
        return bar

    def release_consumed(self, *, forward_only=False):
        # Called after each physical fragment, before the next LIS solve.
        # The ED entry stays live until its later, combined W/MP2 pullback.
        retained = []
        for entry in self.entries:
            if entry.consumed or forward_only and entry is not self.parent:
                self.scope.receipt['half_resident_bytes'] -= entry.nbytes
                self.scope.receipt['half_releases'] += 1
                entry.h = entry.fitted = None
                if entry is self.parent:
                    self.parent = None
            else:
                retained.append(entry)
        self.entries = retained

    def close(self):
        for entry in self.entries:
            entry.h = entry.fitted = None
        self.entries.clear()
        self.parent = None
        self.scope.receipt['half_resident_bytes'] = 0
        self.scope = None
