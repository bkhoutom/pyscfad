"""Resident ED/LIS half transforms preserve complete first-order response."""
import h5py
import jax
import numpy
import pytest

from pyscfad.lno import _df_direct, _df_shared
from pyscfad.lno._df_half import HalfTransformedDFCache
from pyscfad.lno.test import test_shared_local_df_reverse as shared


@pytest.mark.parametrize('dense_threads', [2, 3])
def test_half_transforms_scope_dense_threads_without_changing_integral_threads(
        monkeypatch, dense_threads):
    """Catch missing dense limits or limits leaking into OpenMP integral work."""
    import scipy.linalg
    from pyscf import lib
    from threadpoolctl import threadpool_info
    from pyscfad.lib._threading import blas_threads, dense_blas_scope

    monkeypatch.setenv('PYSCFAD_DENSE_BLAS_THREADS', str(dense_threads))
    mol, _ = shared._problem(4)
    rng = numpy.random.default_rng(91)
    seen = set()

    def pools(api):
        return {item['filepath']: item['num_threads']
                for item in threadpool_info() if item['user_api'] == api}

    with blas_threads(1), dense_blas_scope(), \
            _df_shared.reverse_scope(shared._mf(mol), shared.ATOMS) as scope:
        scope.half = cache = HalfTransformedDFCache(scope)
        low = scope.metric_factor()
        coeff = rng.normal(size=(scope.mol.nao, 4))
        child = coeff @ rng.normal(size=(4, 3))
        baseline_blas, baseline_omp = pools('blas'), pools('openmp')
        assert baseline_blas and set(baseline_blas.values()) == {1}

        def check(label, count):
            seen.add(label)
            assert pools('blas') == dict.fromkeys(baseline_blas, count), label
            assert pools('openmp') == baseline_omp, label

        def observe(func, label, count):
            def call(*args, **kwargs):
                check(label, count)
                return func(*args, **kwargs)
            return call

        raw = _df_direct._local_direct_raw_int3c_blocks
        def integral_blocks(*args, **kwargs):
            check('integrals', 1)
            for block in raw(*args, **kwargs):
                check('integrals', 1)
                yield block
                check('integrals', 1)

        def read_z(p0, p1, width):
            check('adjoint read', 1)
            return numpy.ones((p1-p0, width))

        with monkeypatch.context() as patch:
            for module, name in ((numpy, 'matmul'), (numpy, 'einsum'),
                                 (scipy.linalg, 'lstsq'),
                                 (scipy.linalg, 'solve_triangular')):
                patch.setattr(module, name,
                              observe(getattr(module, name), name, dense_threads))
            patch.setattr(lib, 'unpack_tril', observe(lib.unpack_tril, 'unpack', 1))
            patch.setattr(_df_direct, '_local_direct_raw_int3c_blocks', integral_blocks)
            cache.transform(coeff, (0, 2, 2, 4), low)
            check('after ED', 1)
            cache.transform(child, (0, 3, 0, 3), low)
            check('after LIS', 1)
            cache.coefficient_pullback(coeff, lambda p0, p1: read_z(p0, p1, 4),
                                       (0, 2, 2, 4))
            cache.coefficient_pullback(child, lambda p0, p1: read_z(p0, p1, 9),
                                       (0, 3, 0, 3))
            check('after pullbacks', 1)
        assert {'matmul', 'einsum', 'lstsq', 'solve_triangular',
                'integrals', 'unpack', 'adjoint read'} <= seen
        scope.finish()


@pytest.mark.parametrize('nmo', [4, 7])
def test_dependent_frames_use_one_pass_and_no_reverse_reads(monkeypatch, nmo):
    mol, coeff = shared._problem(nmo)
    expected, expected_bars = jax.value_and_grad(shared._energy, argnums=(0, 1))(mol, coeff)
    reads = []
    passes = 1
    read = _df_direct._generate_local_raw_int3c_blocks
    def raw(*args, **kwargs):
        reads.append(1)
        assert len(reads) <= passes, 'LIS rotations and backward must use retained H'
        yield from read(*args, **kwargs)
    monkeypatch.setattr(_df_direct, '_generate_local_raw_int3c_blocks', raw)
    def forbidden(*args, **kwargs):
        pytest.fail('Resident half inputs must not spool/read a raw-domain cache')
    monkeypatch.setattr(h5py.Dataset, '__array__', forbidden)
    rows = HalfTransformedDFCache._rows
    def small_tiles(self, *args, **kwargs):
        count, budget = rows(self, *args, **kwargs)
        return min(2, count), budget
    monkeypatch.setattr(HalfTransformedDFCache, '_rows', small_tiles)
    with _df_shared.reverse_scope(shared._mf(mol), shared.ATOMS) as scope:
        scope.half = cache = HalfTransformedDFCache(scope)
        value, pullback = jax.vjp(shared._energy, mol, coeff)
        assert reads == [1]*passes
        entries = list(cache.entries)
        assert len(entries) == 2
        ed, lis = entries
        assert ed.h.shape == (scope.auxmol.nao, scope.mol.nao, nmo)
        assert lis.h.shape == (scope.auxmol.nao, scope.mol.nao, 3)
        assert ed.fitted.shape == (scope.auxmol.nao, 2*(nmo-2))
        assert all(isinstance(entry.h, numpy.ndarray) for entry in entries)
        assert scope._file is None  # Neither H nor fitted MO inputs use disk.
        # Reusing a forward result does not rebuild either H or fitted B.
        assert cache.transform(ed.coeff, ed.slices, scope.low) is ed.fitted
        assert reads == [1]*passes
        bars = pullback(numpy.asarray(1.))
        assert reads == [1]*passes
        assert all(entry.consumed for entry in entries)
        assert scope.receipt['half_builds'] == passes
        assert scope.receipt['half_rotations'] == 1
        assert scope.receipt['half_pullbacks'] == 2
        assert scope.receipt['raw_source_passes'] == passes
        assert scope.receipt['raw_integral_builds'] == passes
        assert scope.receipt['half_peak_bytes'] == sum(entry.nbytes for entry in entries)
        assert scope.receipt['max_raw_rows'] <= 2
        _df_shared.release_consumed_inputs()
        assert cache.entries == [] and scope.receipt['half_resident_bytes'] == 0
        assert all(entry.h is None and entry.fitted is None for entry in entries)
        geometry = scope.finish()
        numpy.testing.assert_allclose(value, expected, atol=2e-12, rtol=0)
        numpy.testing.assert_allclose(bars[0].coords+geometry, expected_bars[0].coords,
                                      atol=2e-9, rtol=2e-8)
        # All AO directions, not just rotations inside ED/LIS.
        numpy.testing.assert_allclose(bars[1], expected_bars[1], atol=2e-10, rtol=2e-9)
        assert scope.receipt['geometry_passes'] == scope.receipt['metric_passes'] == 1
    assert scope.half is None and cache.scope is None
    assert scope._file is None and scope._handle is None


def test_completed_lis_released_while_ed_survives():
    mol, _ = shared._problem(4)
    rng = numpy.random.default_rng(18)
    with _df_shared.reverse_scope(shared._mf(mol), shared.ATOMS) as scope:
        scope.half = cache = HalfTransformedDFCache(scope)
        low = scope.metric_factor()
        ed_coeff = rng.normal(size=(scope.mol.nao, 4))
        ed = cache.transform(ed_coeff, (0, 2, 2, 4), low)
        entry = cache.entries[0]
        for _ in range(3):
            coeff = ed_coeff @ rng.normal(size=(4, 3))
            cache.transform(coeff, (0, 3, 0, 3), low)
            assert len(cache.entries) == 2
            cache.coefficient_pullback(coeff, lambda p0, p1: numpy.ones((p1-p0, 9)),
                                       (0, 3, 0, 3))
            _df_shared.release_consumed_inputs()
            assert cache.entries == [entry] and entry.fitted is ed
            assert scope.receipt['half_resident_bytes'] == entry.nbytes
        assert scope.receipt['raw_source_passes'] == 1
        scope.finish()
    assert entry.h is None and entry.fitted is None


def test_cap_accounts_for_ed_and_lis_together():
    mol, _ = shared._problem(4)
    with _df_shared.reverse_scope(shared._mf(mol), shared.ATOMS) as scope:
        scope.half = cache = HalfTransformedDFCache(scope)
        low = scope.metric_factor()
        coeff = numpy.ones((scope.mol.nao, 4))
        cache.transform(coeff, (0, 2, 2, 4), low)
        ed = cache.entries[0]
        # Either entry fits individually, but their combined storage does not.
        cache.limit_bytes = ed.nbytes * 1.5
        with pytest.raises(MemoryError, match='live cache'):
            cache.transform(2*coeff, (0, 2, 2, 4), low)
        assert scope.receipt['raw_source_passes'] == 1
        assert cache.entries == [ed] and ed.h is not None
        scope.finish()


def test_parent_reuse_rejects_outside_directions_before_reading(monkeypatch):
    mol, _ = shared._problem(4)
    rng = numpy.random.default_rng(987)
    with _df_shared.reverse_scope(shared._mf(mol), shared.ATOMS) as scope:
        scope.half = cache = HalfTransformedDFCache(scope)
        low = scope.metric_factor()
        ed = rng.normal(size=(scope.mol.nao, 4))
        cache.transform(ed, (0, 2, 2, 4), low)
        def forbidden(*args, **kwargs):
            pytest.fail('Parent reuse must not silently read another raw pass')
        monkeypatch.setattr(_df_direct, '_generate_local_raw_int3c_blocks', forbidden)
        with pytest.raises(ValueError, match='leaves the retained ED'):
            cache.transform(rng.normal(size=(scope.mol.nao, 3)), (0, 3, 0, 3), low)
        for _ in range(3):
            coeff = ed @ rng.normal(size=(4, 3))
            cache.transform(coeff, (0, 3, 0, 3), low)
            cache.release_consumed(forward_only=True)
            assert cache.entries == [cache.parent]
        assert scope.receipt['raw_source_passes'] == 1
        assert scope.receipt['half_rotations'] == scope.receipt['half_releases'] == 3
        scope.finish()


@pytest.mark.parametrize('cache_cap_mb', [None, '1'])
def test_resident_inputs_can_exceed_workspace_with_correct_full_response(
        monkeypatch, cache_cap_mb):
    mol, coeff = shared._problem(4)
    expected, expected_bars = jax.value_and_grad(shared._energy, argnums=(0, 1))(mol, coeff)
    if cache_cap_mb is None:
        monkeypatch.delenv('PYSCFAD_LNO_HALF_DF_CACHE_MB', raising=False)
    else:
        monkeypatch.setenv('PYSCFAD_LNO_HALF_DF_CACHE_MB', cache_cap_mb)
    holder = shared._mf(mol)
    holder.with_df.max_memory = .004  # Small tiles, but a larger resident H/B.
    seen = []
    rows = HalfTransformedDFCache._rows
    def record_rows(self, words, reserve_bytes=0):
        count, budget = rows(self, words, reserve_bytes)
        seen.append((count, words, budget))
        return count, budget
    monkeypatch.setattr(HalfTransformedDFCache, '_rows', record_rows)
    with _df_shared.reverse_scope(holder, shared.ATOMS) as scope:
        scope.half = cache = HalfTransformedDFCache(scope)
        value, pullback = jax.vjp(shared._energy, mol, coeff)
        assert scope.receipt['half_resident_bytes'] > holder.with_df.max_memory * 1e6
        assert cache.limit_bytes > holder.with_df.max_memory * 1e6
        assert scope._file is None  # Resident inputs do not spill to disk.
        bars = pullback(numpy.asarray(1.))
        assert seen and all(count*8*words <= budget <= 4000 for count, words, budget in seen)
        assert scope.receipt['raw_source_passes'] == scope.receipt['half_builds'] == 1
        _df_shared.release_consumed_inputs()
        assert scope.receipt['half_resident_bytes'] == 0
        geometry = scope.finish()
        numpy.testing.assert_allclose(value, expected, atol=2e-12, rtol=0)
        numpy.testing.assert_allclose(bars[0].coords+geometry, expected_bars[0].coords,
                                      atol=2e-9, rtol=2e-8)
        numpy.testing.assert_allclose(bars[1], expected_bars[1], atol=2e-10, rtol=2e-9)


@pytest.mark.parametrize('failure', ['cap', 'rss', 'source'])
def test_memory_or_source_failure_cleans_up_without_disk_fallback(monkeypatch, failure):
    mol, coeff = shared._problem(4)
    if failure == 'cap':
        monkeypatch.setenv('PYSCFAD_LNO_HALF_DF_CACHE_MB', '0')
    raw = _df_direct._generate_local_raw_int3c_blocks
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        if failure != 'source':
            pytest.fail('Memory admission must precede raw reads')
        yield next(raw(*args, **kwargs))
        raise RuntimeError('interrupted raw source')
    monkeypatch.setattr(_df_direct, '_generate_local_raw_int3c_blocks', fail)
    error = RuntimeError if failure == 'source' else MemoryError
    with pytest.raises(error, match='raw source|memory budget'):
        with _df_shared.reverse_scope(shared._mf(mol), shared.ATOMS) as scope:
            scope.half = cache = HalfTransformedDFCache(scope)
            if failure == 'rss':
                monkeypatch.setattr(_df_shared.pyscf_lib, 'current_memory',
                                    lambda: (mol.max_memory+1., 0.))
            jax.vjp(shared._energy, mol, coeff)
    assert scope.half is None and cache.scope is None and cache.entries == []
    assert scope._file is None and scope._handle is None
    assert _df_shared._ACTIVE.get() is None
    assert calls == ([1] if failure == 'source' else [])
