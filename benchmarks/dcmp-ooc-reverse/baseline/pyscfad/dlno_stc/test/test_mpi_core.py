"""Small collective protocol tests; no molecule, native solver, or MPI launch."""
import concurrent.futures
import threading

import jax
import numpy
import pytest

from pyscfad.dlno_stc import _mpi


class _World:
    def __init__(self, size):
        self.size = size
        self.condition = threading.Condition()
        self.calls = {}
        self.reduce_calls = 0

    def collect(self, step, kind, rank, value, root=0):
        with self.condition:
            entry = self.calls.setdefault(step, {'kind': kind, 'values': {}})
            assert entry['kind'] == kind, (step, entry['kind'], kind)
            entry['values'][rank] = value
            if len(entry['values']) == self.size:
                ordered = [entry['values'][i] for i in range(self.size)]
                if kind in ('allgather', 'gather'):
                    result = ordered
                elif kind in ('bcast', 'Bcast'):
                    result = ordered[root]
                elif kind == 'Reduce':
                    result = sum(ordered)
                    self.reduce_calls += 1
                entry['result'] = result
                self.condition.notify_all()
            else:
                assert self.condition.wait_for(lambda: 'result' in entry, timeout=10), (
                    'collective stranded', step, kind, rank)
            return entry['result']

    def communicator(self, rank):
        world = self

        class Comm:
            step = 0

            def Get_rank(self):
                return rank

            def Get_size(self):
                return world.size

            def _collect(self, kind, value, root=0):
                step = self.step
                self.step += 1
                return world.collect(step, kind, rank, value, root)

            def allgather(self, value):
                return self._collect('allgather', value)

            def bcast(self, value, root=0):
                return self._collect('bcast', value, root)

            def gather(self, value, root=0):
                result = self._collect('gather', value, root)
                return result if rank == root else None

            def Bcast(self, value, root=0):
                result = self._collect('Bcast', numpy.array(value, copy=True), root)
                value[...] = result

            def Reduce(self, send, receive, op=None, root=0):
                from mpi4py import MPI
                value = receive if send is MPI.IN_PLACE else send
                result = self._collect('Reduce', numpy.array(value, copy=True), root)
                if rank == root:
                    receive[...] = result

        return Comm()


def _reduce(trees, monkeypatch, chunk_bytes=16, root=0):
    monkeypatch.setattr(_mpi, '_TREE_REDUCE_CHUNK_BYTES', chunk_bytes)
    world = _World(len(trees))

    def run(rank):
        threading.current_thread().name = f'rank-{rank}'
        try:
            return _mpi.tree_sum_to_root(world.communicator(rank), trees[rank], root=root)
        except Exception as exc:
            return exc

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(trees)) as pool:
        results = list(pool.map(run, range(len(trees))))
    return results, world


def test_reducer_sums_chunks_and_mixed_zero_leaves(monkeypatch):
    trees = [
        {'a': numpy.arange(7, dtype=float), 'b': None,
         'c': numpy.zeros(3, dtype=jax.dtypes.float0), 'd': None},
        {'a': numpy.ones(7), 'b': numpy.array([2., 3.]),
         'c': None, 'd': None},
        {'a': None, 'b': numpy.zeros(2, dtype=jax.dtypes.float0),
         'c': numpy.zeros(3, dtype=jax.dtypes.float0), 'd': None},
    ]
    results, world = _reduce(trees, monkeypatch)
    assert not isinstance(results[0], Exception), repr(results)
    numpy.testing.assert_array_equal(results[0]['a'], numpy.arange(7) + 1)
    numpy.testing.assert_array_equal(results[0]['b'], [2., 3.])
    assert results[0]['c'].dtype == jax.dtypes.float0
    assert results[0]['d'] is None
    assert results[1:] == [None, None]
    assert world.reduce_calls == 5


@pytest.mark.parametrize('other', [
    {'a': numpy.zeros(4)}, {'a': numpy.zeros(3, dtype=numpy.float32)},
    {'missing': numpy.zeros(3)}, {'a': numpy.zeros(4, dtype=jax.dtypes.float0)},
])
def test_reducer_rejects_collective_schema_mismatch(monkeypatch, other):
    results, world = _reduce([{'a': numpy.zeros(3)}, other], monkeypatch)
    assert all(isinstance(result, RuntimeError) for result in results), results
    assert world.reduce_calls == 0


def test_reducer_synchronizes_host_staging_failure(monkeypatch):
    class BadArray:
        dtype = numpy.dtype('float64')
        shape = (4,)

        def __array__(self, *args, **kwargs):
            raise MemoryError('host staging allocation failed')

    results, world = _reduce([numpy.zeros(4), BadArray()], monkeypatch)
    assert all(isinstance(result, RuntimeError) for result in results), results
    assert all('host staging allocation failed' in str(result) for result in results)
    assert world.reduce_calls == 0


@pytest.mark.parametrize('failed_rank,trees', [
    (0, [None, numpy.ones(4)]),
    (1, [numpy.ones(4), None]),
])
def test_reducer_synchronizes_zero_allocation_failure(monkeypatch, failed_rank, trees):
    original_zeros = numpy.zeros

    def fail_zero(*args, **kwargs):
        if threading.current_thread().name == f'rank-{failed_rank}':
            raise MemoryError('zero allocation failed')
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(numpy, 'zeros', fail_zero)
    results, world = _reduce(trees, monkeypatch)
    assert all(isinstance(result, RuntimeError) for result in results), results
    assert all('zero allocation failed' in str(result) for result in results)
    assert world.reduce_calls == 0


def test_reducer_preserves_selected_root_construction_metadata(monkeypatch):
    @jax.tree_util.register_pytree_node_class
    class Tree:
        def __init__(self, value, identity):
            self.value, self.identity = value, identity

        def tree_flatten(self):
            return (self.value,), self.identity

        @classmethod
        def tree_unflatten(cls, identity, leaves):
            return cls(leaves[0], identity)

    identities = [object(), object()]
    trees = [Tree(numpy.ones(3), identities[0]), Tree(numpy.ones(3)*2, identities[1])]
    results, _ = _reduce(trees, monkeypatch, root=1)
    assert results[0] is None
    assert results[1].identity is identities[1]
    numpy.testing.assert_array_equal(results[1].value, 3.)


def test_reducer_checks_each_later_worker_zero_chunk(monkeypatch):
    original_zeros = numpy.zeros
    allocations = []

    def fail_second_zero(*args, **kwargs):
        if threading.current_thread().name == 'rank-1':
            allocations.append(args)
            if len(allocations) == 2:
                raise MemoryError('later chunk failed')
        return original_zeros(*args, **kwargs)

    trees = [numpy.ones(5), None]
    monkeypatch.setattr(numpy, 'zeros', fail_second_zero)
    results, world = _reduce(trees, monkeypatch)
    assert all(isinstance(result, RuntimeError) for result in results), results
    assert all('later chunk failed' in str(result) for result in results)
    assert world.reduce_calls == 1


def test_worker_constructs_only_requested_auxiliary_subtree(monkeypatch):
    from types import SimpleNamespace
    from pyscfad.df import addons
    calls = []
    auxmol = object()
    monkeypatch.setattr(addons, 'make_auxmol', lambda mol, basis: calls.append((mol, basis)) or auxmol)
    molecule = object()
    df = SimpleNamespace(auxmol=None, auxbasis='basis', _cderi=object())
    mf = SimpleNamespace(mol=molecule, with_df=df)
    canonical = {'mo_coeff': numpy.ones((1, 1))}
    result = _mpi._build_worker(lambda received: mf if received is canonical else None,
                                canonical, {'auxmol': None})
    assert result is mf and df._cderi is None
    assert df.auxmol is None and calls == []
    _mpi._build_worker(lambda received: mf, canonical, {'auxmol': {}})
    assert df.auxmol is auxmol and calls == [(molecule, 'basis')]


def test_scheduler_uneven_owners_and_idle_live_bars(monkeypatch):
    from mpi4py import MPI
    # Fake communicators run in Python threads; model valid MPI process mains.
    monkeypatch.setattr(MPI, 'Is_thread_main', lambda: True)
    from types import SimpleNamespace
    from typing import NamedTuple
    from pyscfad import numpy as jnp
    from pyscfad.dlno_stc import adjoint, backend, driver, _selection

    class MF(NamedTuple):
        mo_coeff: object
        mo_energy: object
        mo_occ: object
        e_tot: object

    class Common(NamedTuple):
        s1e: object

    mf = MF(jnp.ones((1, 1)), jnp.ones(1), jnp.ones(1)*2, jnp.array(-1.))
    common = Common(jnp.ones((1, 1))*2)
    static = SimpleNamespace(fragments=(object(), object()),
                             frag_lolist=(numpy.array([0]), numpy.array([1])))
    groups = (_selection.DomainGroup((0,), 0, (0,)),
              _selection.DomainGroup((1,), 1, (0,)))
    monkeypatch.setattr(_selection, 'build_domain_groups', lambda *args, **kwargs: groups)
    monkeypatch.setattr(_mpi, '_reference_signature', lambda mf: {'auxmol': None})
    monkeypatch.setattr(_mpi, '_validate_shared_reference', lambda *args, **kwargs: None)
    monkeypatch.setattr(_mpi, '_build_worker', lambda *args: mf)
    captured = {}

    def initialize():
        assert threading.current_thread().name == 'rank-0'
        return dict(mf=mf, common=common, static=static, weak_pairs=())

    def prepare(state, selected, group, controls, with_grad):
        def fn(m, c):
            return {'foo': m.mo_coeff * (group.representative_index+1) + c.s1e,
                    'fvv': jnp.ones((1, 1)), 'B': jnp.ones((1, 1, 1))}
        inputs, pullback = jax.vjp(fn, state['mf'], state['common'])
        return inputs, jax.tree_util.tree_map(numpy.asarray, inputs), pullback

    def solve(inputs, metadata, controls, *, comm, with_grad):
        assert comm is None
        assert controls['energy_tolerance'] == pytest.approx(.003 / numpy.sqrt(2))
        return {'energy': float(inputs['foo'].sum()), 'energy_standard_error': .1,
                'cotangents': {name: numpy.ones_like(value) for name, value in inputs.items()}}

    def weak(m, c, selected, pairs, comm, *, with_grad):
        captured[comm.Get_rank()] = (m, c)
        return jnp.array(0.), adjoint.zero_term_cotangents(m, c)

    monkeypatch.setattr(driver, '_prepare', prepare)
    monkeypatch.setattr(backend, 'solve', solve)
    monkeypatch.setattr(adjoint, 'apply_selected_pullback',
                        lambda pullback, inputs, result: pullback(result['cotangents']))
    monkeypatch.setattr(_mpi, 'weak_pair_contribution', weak)
    monkeypatch.setattr(driver, '_finish', lambda state, **kwargs:
                        (state['energy'], state['mf_bar'].mo_coeff + state['common_bar'].s1e))
    world = _World(4)

    def run(rank):
        threading.current_thread().name = f'rank-{rank}'
        return _mpi._run_domain_mpi(
            initialize, lambda canonical: mf, None,
            {'mode': 'stochastic', 'energy_tolerance': .003} if rank == 0 else None,
            world.communicator(rank), with_grad=True, parallel_scf_jk=False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(4)))
    numpy.testing.assert_allclose(results[0][0], 7.)
    numpy.testing.assert_allclose(results[0][1], 5.)
    assert results[1:] == [None, None, None]
    assert set(captured) == {0, 1, 2, 3}
    diagnostics = _mpi.LAST_RUN_DIAGNOSTICS
    assert [row['owner'] for row in diagnostics['groups']] == [0, 1]
    assert [len(row['groups']) for row in diagnostics['ranks']] == [1, 1, 0, 0]
    assert all('response' in row['stage_seconds'] for row in diagnostics['ranks'])
    assert diagnostics['conditional_energy_standard_error'] == pytest.approx(numpy.sqrt(.02))


@pytest.mark.parametrize('failure', ['thread-level', 'main-thread'])
def test_scheduler_synchronizes_thread_preflight_before_initialization(monkeypatch, failure):
    from mpi4py import MPI
    world = _World(2)
    initialized = []

    def thread_level():
        if threading.current_thread().name == 'rank-1' and failure == 'thread-level':
            return MPI.THREAD_SINGLE
        return MPI.THREAD_MULTIPLE

    def is_main():
        return not (threading.current_thread().name == 'rank-1' and failure == 'main-thread')

    monkeypatch.setattr(MPI, 'Query_thread', thread_level)
    monkeypatch.setattr(MPI, 'Is_thread_main', is_main)

    def initialize():
        initialized.append(True)
        raise AssertionError('initializer reached before thread preflight')

    def run(rank):
        threading.current_thread().name = f'rank-{rank}'
        try:
            return _mpi._run_domain_mpi(initialize, None, None, None,
                                        world.communicator(rank), with_grad=False, parallel_scf_jk=False)
        except Exception as exc:
            return exc

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, range(2)))
    expected = 'MPI_THREAD_FUNNELED' if failure == 'thread-level' else 'MPI main thread'
    assert all(isinstance(result, RuntimeError) and expected in str(result)
               for result in results), results
    assert initialized == []
    assert world.reduce_calls == 0


@pytest.mark.parametrize('with_grad,include_hf', [(False, False), (True, False), (True, True)])
def test_scheduler_zero_global_work_finishes_response_once(monkeypatch, with_grad, include_hf):
    from mpi4py import MPI
    from types import SimpleNamespace
    from typing import NamedTuple
    from pyscfad import numpy as jnp
    from pyscfad.dlno_stc import backend, driver, _selection

    monkeypatch.setattr(MPI, 'Is_thread_main', lambda: True)

    @jax.tree_util.register_pytree_node_class
    class MF:
        verbose = 0

        def __init__(self, coeff, energy, occupation, total):
            self.mo_coeff, self.mo_energy = coeff, energy
            self.mo_occ, self.e_tot = occupation, total

        def tree_flatten(self):
            return (self.mo_coeff, self.mo_energy, self.mo_occ, self.e_tot), None

        @classmethod
        def tree_unflatten(cls, unused, leaves):
            return cls(*leaves)

    class Common(NamedTuple):
        s1e: object

    mf, scf_pullback = jax.vjp(
        lambda coordinate: MF(jnp.ones((1, 1)), jnp.ones(1), jnp.zeros(1),
                              -2. + coordinate[0]), jnp.zeros(1))
    common, common_pullback = jax.vjp(lambda current: Common(current.mo_coeff*0 + 1), mf)
    static = SimpleNamespace(fragments=(), frag_lolist=())
    calls = {'initialize': 0, 'common': 0, 'scf': 0, 'prepare': 0, 'backend': 0}
    live_bars = {}
    monkeypatch.setattr(_selection, 'build_domain_groups', lambda *args, **kwargs: ())
    monkeypatch.setattr(_mpi, '_reference_signature', lambda current: {'auxmol': None})
    monkeypatch.setattr(_mpi, '_validate_shared_reference', lambda *args, **kwargs: None)
    monkeypatch.setattr(_mpi, '_build_worker', lambda *args: mf)

    def common_response(bar):
        assert threading.current_thread().name == 'rank-0'
        calls['common'] += 1
        assert all(numpy.all(numpy.asarray(leaf) == 0) for leaf in jax.tree_util.tree_leaves(bar))
        return common_pullback(bar)

    def scf_response(bar):
        assert threading.current_thread().name == 'rank-0'
        calls['scf'] += 1
        numpy.testing.assert_array_equal(bar.e_tot, float(include_hf))
        return scf_pullback(bar)

    def initialize():
        assert threading.current_thread().name == 'rank-0'
        calls['initialize'] += 1
        return dict(mf=mf, common=common, static=static, weak_pairs=(),
                    common_pullback=common_response, scf_pullback=scf_response)

    def forbidden_prepare(*args, **kwargs):
        calls['prepare'] += 1
        raise AssertionError('empty strong workload entered preparation')

    def forbidden_backend(*args, **kwargs):
        calls['backend'] += 1
        raise AssertionError('empty strong workload entered backend')

    original_add_bars = driver._add_bars

    def add_live_bars(state, mf_bar, common_bar):
        for tree in (state['mf_bar'], state['common_bar'], mf_bar, common_bar):
            leaves = jax.tree_util.tree_leaves(tree)
            assert leaves and all(numpy.all(numpy.asarray(leaf) == 0) for leaf in leaves)
        live_bars[threading.current_thread().name] = True
        original_add_bars(state, mf_bar, common_bar)

    monkeypatch.setattr(driver, '_prepare', forbidden_prepare)
    monkeypatch.setattr(backend, 'solve', forbidden_backend)
    monkeypatch.setattr(driver, '_add_bars', add_live_bars)
    world = _World(4)

    def run(rank):
        threading.current_thread().name = f'rank-{rank}'
        return _mpi._run_domain_mpi(
            initialize, lambda canonical: mf, None,
            {'mode': 'deterministic'} if rank == 0 else None,
            world.communicator(rank), with_grad=with_grad, include_hf=include_hf,
            parallel_scf_jk=False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(4)))
    assert results[1:] == [None, None, None]
    if with_grad:
        numpy.testing.assert_array_equal(results[0][0], -2. if include_hf else 0.)
        numpy.testing.assert_array_equal(results[0][1], [float(include_hf)])
        assert set(live_bars) == {'rank-0', 'rank-1', 'rank-2', 'rank-3'}
    else:
        numpy.testing.assert_array_equal(results[0], 0.)
        assert live_bars == {}
    assert calls == {'initialize': 1, 'common': int(with_grad), 'scf': int(with_grad),
                     'prepare': 0, 'backend': 0}
    diagnostics = _mpi.LAST_RUN_DIAGNOSTICS
    assert diagnostics['group_count'] == 0 and diagnostics['groups'] == []
    assert all(row['groups'] == [] for row in diagnostics['ranks'])
    assert diagnostics['conditional_energy_standard_error'] == 0.
