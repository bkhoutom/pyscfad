"""Host MPI scheduling of complete STC domain jobs and shared response.

Only numerical shared references cross MPI. The strong preparation packet and
its pullback stay together on their round-robin owner. MPI is imported lazily.
"""

from contextlib import contextmanager
from functools import partial
import gc
import math
import time

import jax
import numpy
from pyscfad import numpy as np
from pyscfad.tools import resource_profile

_TREE_REDUCE_CHUNK_BYTES = 64 * 1024**2
_TREE_REDUCE_MAX_COUNT = int(numpy.iinfo(numpy.int32).max)
LAST_RUN_DIAGNOSTICS = None


def _raise_if_any_rank_failed(comm, error):
    """Raise the same rank-labelled failure before the next typed collective."""
    failures = comm.allgather(
        None if error is None else f"rank {comm.Get_rank()}: {error}")
    failures = [failure for failure in failures if failure is not None]
    if failures:
        raise RuntimeError("STC domain MPI failed: " + "; ".join(failures))


def _error(exc, phase):
    return f"{phase}: {type(exc).__name__}: {exc}"


def _collective_call(comm, function, phase):
    result = None
    error = None
    try:
        result = function()
    except Exception as exc:
        error = _error(exc, phase)
    _raise_if_any_rank_failed(comm, error)
    return result


def _check_mpi_thread_support():
    """Require funneled support and calls from the thread initializing MPI."""
    from mpi4py import MPI

    if MPI.Query_thread() < MPI.THREAD_FUNNELED:
        raise RuntimeError('MPI_THREAD_FUNNELED support is required for domain MPI')
    if not MPI.Is_thread_main():
        raise RuntimeError('domain MPI must run on the MPI main thread')


def _leaf_metadata(leaf):
    if leaf is None:
        return ('none',)
    if getattr(leaf, 'dtype', None) == jax.dtypes.float0:
        return ('float0', tuple(leaf.shape))
    if hasattr(leaf, 'dtype') and hasattr(leaf, 'shape'):
        dtype, shape = numpy.dtype(leaf.dtype), tuple(leaf.shape)
    else:
        array = numpy.asarray(leaf)
        dtype, shape = array.dtype, array.shape
    if dtype.hasobject or dtype.kind not in 'iufc':
        raise TypeError(f"unsupported cotangent leaf dtype {dtype}")
    return ('numeric', shape, dtype.str)


def _tree_metadata(tree):
    items, treedef = jax.tree_util.tree_flatten_with_path(
        tree, is_leaf=lambda value: value is None)
    paths = tuple(jax.tree_util.keystr(path) for path, _ in items)
    if len(paths) != len(set(paths)):
        raise ValueError('cotangent tree contains duplicate stable leaf paths')
    leaves = {path: item[1] for path, item in zip(paths, items)}
    metadata = {path: _leaf_metadata(leaf) for path, leaf in leaves.items()}
    return paths, leaves, treedef, metadata


def _reduction_plans(paths, metadata_by_rank):
    reference_paths = set(paths)
    for rank, metadata in enumerate(metadata_by_rank):
        if set(metadata) != reference_paths:
            raise RuntimeError(
                f"cotangent leaf paths differ on rank {rank}; "
                f"root-only={sorted(reference_paths - set(metadata))[:5]}, "
                f"rank-only={sorted(set(metadata) - reference_paths)[:5]}")
    plans = {}
    for path in paths:
        descriptions = [metadata[path] for metadata in metadata_by_rank]
        numeric = [item for item in descriptions if item[0] == 'numeric']
        float0 = [item for item in descriptions if item[0] == 'float0']
        if numeric:
            reference = numeric[0]
            if (any(item != reference for item in numeric)
                    or any(item[1] != reference[1] for item in float0)):
                raise RuntimeError(
                    f"cotangent {path} has inconsistent shape or dtype: {descriptions}")
            plans[path] = reference
        elif float0:
            if any(item != float0[0] for item in float0):
                raise RuntimeError(f"cotangent {path} has inconsistent float0 shapes")
            plans[path] = float0[0]
        else:
            plans[path] = ('none',)
    return plans


def _collect_tree_schema(comm, tree, root=0):
    paths, leaves, treedef, metadata = _collective_call(
        comm, lambda: _tree_metadata(tree), 'cotangent schema staging')
    metadata_by_rank = comm.allgather(metadata)
    ordered_paths = tuple(metadata_by_rank[root])
    plans = _collective_call(
        comm, lambda: _reduction_plans(ordered_paths, metadata_by_rank),
        'cotangent schema validation')
    return paths, leaves, treedef, metadata, ordered_paths, plans


def tree_sum_to_root(comm, tree, *, root=0):
    """Sum leaves by stable paths, retaining root's pytree construction data.

    None and float0 contribute zeros at matching paths. Every allocation and
    conversion is checked collectively, including worker zero chunks, before
    peers enter a numeric Reduce. No complete tree is gathered or pickled.
    """
    from mpi4py import MPI

    rank = comm.Get_rank()
    paths, leaves, treedef, metadata, ordered_paths, plans = _collect_tree_schema(
        comm, tree, root)
    summed = {}
    for path in ordered_paths:
        plan = plans[path]
        if plan[0] == 'none':
            summed[path] = None
            continue
        if plan[0] == 'float0':
            value = _collective_call(
                comm, lambda: numpy.zeros(plan[1], dtype=jax.dtypes.float0)
                if rank == root else None, f'cotangent {path} float0 allocation')
            if rank == root:
                summed[path] = value
            continue
        shape, dtype = plan[1], numpy.dtype(plan[2])

        def stage_leaf():
            host = (numpy.array(numpy.asarray(leaves[path]), copy=True, order='C')
                    if metadata[path][0] == 'numeric' else None)
            result = ((host if host is not None else numpy.zeros(shape, dtype=dtype))
                      if rank == root else None)
            return (None if host is None else host.reshape(-1), result,
                    None if result is None else result.reshape(-1))

        flat, result, flat_result = _collective_call(
            comm, stage_leaf, f'cotangent {path} host staging')
        size = math.prod(shape)
        chunk_count = min(_TREE_REDUCE_MAX_COUNT,
                          max(1, _TREE_REDUCE_CHUNK_BYTES // dtype.itemsize))
        for start in range(0, size, chunk_count):
            stop = min(size, start + chunk_count)

            def stage_chunk():
                if rank == root:
                    return MPI.IN_PLACE, flat_result[start:stop]
                send = (numpy.zeros(stop - start, dtype=dtype) if flat is None
                        else flat[start:stop])
                return send, None

            send, receive = _collective_call(
                comm, stage_chunk, f'cotangent {path} chunk {start}:{stop} staging')
            comm.Reduce(send, receive, op=MPI.SUM, root=root)
        _collective_call(
            comm, lambda: summed.__setitem__(path, result) if rank == root else None,
            f'cotangent {path} output staging')
        del flat, result, flat_result
    return _collective_call(
        comm, lambda: jax.tree_util.tree_unflatten(treedef, [summed[path] for path in paths])
        if rank == root else None, 'cotangent output reconstruction')


def _broadcast_array_tree(comm, tree, root=0):
    """Broadcast concrete reference arrays using bounded, preflighted buffers."""
    rank = comm.Get_rank()
    from mpi4py import MPI  # noqa: F401 -- keep MPI initialization local

    def stage_root():
        if rank != root:
            return None
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        arrays = [numpy.array(numpy.asarray(leaf), copy=True, order='C') for leaf in leaves]
        if any(array.dtype.hasobject or array.dtype.kind not in 'biufc' for array in arrays):
            raise TypeError('shared reference must contain numerical arrays only')
        return arrays, treedef, [(array.shape, array.dtype.str) for array in arrays]

    staged = _collective_call(comm, stage_root, 'shared reference host staging')
    header = (staged[1], staged[2]) if rank == root else None
    treedef, descriptors = comm.bcast(header, root=root)
    arrays = _collective_call(comm, lambda: staged[0] if rank == root else [],
                              'shared reference output staging')
    for index, (shape, dtype_string) in enumerate(descriptors):
        dtype = numpy.dtype(dtype_string)
        array = _collective_call(
            comm, lambda: arrays[index] if rank == root else numpy.empty(shape, dtype=dtype),
            f'shared reference array {index} allocation')
        flat = _collective_call(comm, lambda: array.reshape(-1),
                                f'shared reference array {index} buffer staging')
        chunk_count = min(_TREE_REDUCE_MAX_COUNT,
                          max(1, _TREE_REDUCE_CHUNK_BYTES // dtype.itemsize))
        for start in range(0, flat.size, chunk_count):
            chunk = _collective_call(
                comm, lambda: flat[start:start + chunk_count],
                f'shared reference array {index} chunk staging')
            comm.Bcast(chunk, root=root)
        _collective_call(comm, lambda: arrays.append(array) if rank != root else None,
                          f'shared reference array {index} output staging')
    return _collective_call(comm, lambda: jax.tree_util.tree_unflatten(treedef, arrays),
                            'shared reference reconstruction')


def _freeze(value):
    """Comparable molecular/basis metadata, independent of object identity."""
    if isinstance(value, dict):
        return tuple((key, _freeze(value[key])) for key in sorted(value, key=str))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, numpy.ndarray):
        return (value.dtype.str, value.shape, _freeze(value.tolist()))
    if isinstance(value, numpy.generic):
        return value.item()
    return value


def _molecule_signature(mol):
    return {
        'definition': _freeze({
            'atoms': tuple(mol.atom_symbol(i) for i in range(mol.natm)),
            'charges': numpy.asarray(mol.atom_charges()),
            'basis': getattr(mol, '_basis', None),
            'ecp': getattr(mol, '_ecp', None),
            'pseudo': getattr(mol, '_pseudo', None),
            'cart': mol.cart, 'charge': mol.charge, 'spin': mol.spin,
        }),
        'coords': numpy.asarray(mol.atom_coords()).tolist(),
        'trace': {name: (None if getattr(mol, name, None) is None else
                         numpy.asarray(getattr(mol, name)).tolist())
                  for name in ('coords', 'exp', 'ctr_coeff', 'r0')},
    }


def _reference_signature(mf):
    with_df = getattr(mf, 'with_df', None)
    if with_df is None:
        raise ValueError('domain MPI requires a matching density-fitted worker SCF skeleton')
    auxmol = getattr(with_df, 'auxmol', None)
    return {'mol': _molecule_signature(mf.mol),
            'df_mol': _molecule_signature(with_df.mol),
            'auxbasis': _freeze(with_df.auxbasis),
            'auxmol': None if auxmol is None else _molecule_signature(auxmol)}


def _same_array(actual, expected, label):
    actual, expected = numpy.asarray(actual), numpy.asarray(expected)
    if (actual.shape != expected.shape or actual.dtype != expected.dtype
            or not numpy.isfinite(actual).all() or not numpy.isfinite(expected).all()
            or not numpy.allclose(actual, expected, rtol=1e-10, atol=1e-11)):
        raise ValueError(f'signed shared reference mismatch for {label}')


def _validate_molecule_signature(actual, expected, label):
    if actual['definition'] != expected['definition']:
        raise ValueError(f'{label} geometry/basis/charge/spin definition mismatch')
    _same_array(numpy.asarray(actual['coords']), numpy.asarray(expected['coords']),
                f'{label} geometry')
    for name in expected['trace']:
        left, right = actual['trace'][name], expected['trace'][name]
        if left is None or right is None:
            if left is not None or right is not None:
                raise ValueError(f'{label} trace leaf {name} differs')
        else:
            _same_array(numpy.asarray(left), numpy.asarray(right), f'{label}.{name}')


def _validate_shared_reference(mf, common, canonical, expected_common, signature, static,
                               *, with_grad):
    """Require actual signed values and molecular definitions before owner work."""
    from . import driver

    driver._validate_static(static)
    driver._validate_reference(mf, static)
    if with_grad:
        driver._coordinate_only(mf.mol)
    actual_signature = _reference_signature(mf)
    if actual_signature['auxbasis'] != signature['auxbasis']:
        raise ValueError('worker auxiliary basis definition differs from root')
    for name in ('mol', 'df_mol', 'auxmol'):
        actual, expected = actual_signature[name], signature[name]
        if actual is None or expected is None:
            if actual is not None or expected is not None:
                raise ValueError(f'worker {name} subtree differs from root')
        else:
            _validate_molecule_signature(actual, expected, name)
    for name, expected in canonical.items():
        _same_array(getattr(mf, name), expected, name)
    actual_paths, actual_leaves, _, _ = _tree_metadata(common)
    expected_paths, expected_leaves, _, _ = _tree_metadata(expected_common)
    if set(actual_paths) != set(expected_paths):
        raise ValueError('shared common array leaf paths differ from root')
    for path in expected_paths:
        _same_array(actual_leaves[path], expected_leaves[path], f'common{path}')
    for name, expected in (
        ('occupied_coeff', canonical['mo_coeff'][:, static.active_occ_indices]),
        ('virtual_coeff', canonical['mo_coeff'][:, static.active_vir_indices]),
        ('occupied_energy', canonical['mo_energy'][static.active_occ_indices]),
        ('virtual_energy', canonical['mo_energy'][static.active_vir_indices]),
    ):
        _same_array(getattr(common, name), expected, f'common.{name}')


def _build_worker(worker_initializer, canonical, signature):
    if not callable(worker_initializer):
        raise TypeError('domain MPI needs a worker MF skeleton or canonical-state builder')
    mf = worker_initializer(canonical)
    if mf is None or getattr(mf, 'with_df', None) is None:
        raise ValueError('domain MPI worker initializer must return a density-fitted MF skeleton')
    if signature['auxmol'] is not None and getattr(mf.with_df, 'auxmol', None) is None:
        from pyscfad.df.addons import make_auxmol
        mf.with_df.auxmol = make_auxmol(mf.mol, mf.with_df.auxbasis)
    mf.with_df._cderi = None
    return mf


def weak_pair_contribution(*args, **kwargs):
    """Lazy reexport used by the scheduler and focused tests."""
    from ._mpi_weak import weak_pair_contribution as contribution
    return contribution(*args, **kwargs)


@contextmanager
def _timed_section(name, totals, stage):
    start = time.perf_counter()
    try:
        with resource_profile.section(name):
            yield
    finally:
        totals[stage] = totals.get(stage, 0.0) + time.perf_counter() - start


def _run_domain_mpi(initializer, worker_initializer, static, controls, comm, *,
                    with_grad, include_hf=False, merge_identical=True,
                    parallel_scf_jk=True, scf_worker_initializer=None):
    """Schedule domains between distributed shared SCF/reference sessions."""
    from . import driver

    global LAST_RUN_DIAGNOSTICS
    LAST_RUN_DIAGNOSTICS = None
    _collective_call(comm, _check_mpi_thread_support, 'MPI thread preflight')
    rank = comm.Get_rank()

    def select_service():
        if not isinstance(parallel_scf_jk, bool):
            raise TypeError('parallel_scf_jk must be a bool')
        return parallel_scf_jk

    enabled = driver._root_call(comm, rank, select_service, 'SCF service option')
    enabled = comm.bcast(enabled, root=0)
    service = None
    start = time.perf_counter()
    try:
        if enabled:
            from ._mpi_scf import DomainSCFService
            service = DomainSCFService(comm, scf_worker_initializer)
        return _run_domain_jobs(initializer, worker_initializer, static, controls, comm,
                                with_grad=with_grad, include_hf=include_hf,
                                merge_identical=merge_identical, scf_service=service,
                                start=start)
    finally:
        # Shared failures after PAUSE leave no worker in serve; closing locally
        # releases the tokens carried by root's retained custom VJP closures.
        if service is not None:
            service.close_local()


def _run_domain_jobs(initializer, worker_initializer, static, controls, comm, *,
                     with_grad, include_hf, merge_identical, scf_service, start):
    """Run complete owner jobs while the optional DF service is paused."""
    from . import backend, driver
    from ._selection import build_domain_groups
    from ._workflow import allocate_domain_controls
    from .adjoint import apply_selected_pullback, zero_term_cotangents
    from .parallel import run_backend

    rank, size = comm.Get_rank(), comm.Get_size()
    totals, group_rows, phases = {}, [], {}
    phase_start = start
    with _timed_section('stc.domain.initialize', totals, 'initialize'):
        state = (driver._root_call(comm, rank, initializer, 'initial preparation')
                 if scf_service is None else scf_service.run_initial(initializer))

        def root_metadata():
            authoritative = state['static']
            groups = build_domain_groups(authoritative, merge_identical=merge_identical)
            return (authoritative, groups, state['weak_pairs'], controls,
                    allocate_domain_controls(controls, len(groups)),
                    _reference_signature(state['mf']))

        metadata = driver._root_call(comm, rank, root_metadata, 'shared domain selection')
        static, groups, pairs, controls, settings, signature = comm.bcast(metadata, root=0)
        shared_tree = driver._root_call(comm, rank, lambda: {
            'canonical': {name: getattr(state['mf'], name)
                          for name in ('mo_coeff', 'mo_energy', 'mo_occ', 'e_tot')},
            'common': state['common'],
        }, 'shared reference selection')
        payload = _broadcast_array_tree(comm, shared_tree)
        del shared_tree

        def setup():
            nonlocal state
            from pyscfad.lib._compilation_cache import configure_compilation_cache
            configure_compilation_cache()
            common = jax.tree_util.tree_map(np.asarray, payload['common'])
            if rank != 0:
                mf = _build_worker(worker_initializer, payload['canonical'], signature)
                state = {'mf': mf, 'common': common, 'static': static,
                         'scf_pullback': None, 'common_pullback': None, 'weak_pairs': pairs}
            _validate_shared_reference(state['mf'], state['common'], payload['canonical'],
                                       payload['common'], signature, static, with_grad=with_grad)
            state['energy'] = np.zeros((), dtype=common.s1e.dtype)
            state['variance'] = 0.0
            state['mf_bar'], state['common_bar'] = (
                zero_term_cotangents(state['mf'], state['common']) if with_grad else (None, None))
            jax.block_until_ready((state['energy'], state['mf_bar'], state['common_bar']))

        _collective_call(comm, setup, 'worker shared-reference setup')
        if with_grad:
            _collect_tree_schema(comm, (state['mf_bar'], state['common_bar']))
        else:
            _collect_tree_schema(comm, (state['mf'], state['common']))
        del payload, metadata
    phases['initialize'] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    error = None
    group_id = None
    try:
        for position in range(rank, len(groups), size):
            group = groups[position]
            group_id = group.representative_index
            row = {'global_position': position, 'fragment_id': group_id, 'owner': rank,
                   'target_indices': tuple(group.target_indices),
                   'fragment_indices': tuple(group.fragment_indices), 'replica_count': 1}
            group_start = time.perf_counter()
            with _timed_section('stc.domain.prepare', totals, 'prepare'):
                stage_start = time.perf_counter()
                prepared = driver._prepare(state, static, group, controls, with_grad)
                row['prepare'] = time.perf_counter() - stage_start
                row['input_shapes'] = {name: tuple(array.shape) for name, array in prepared[1].items()}
            metadata = {
                'fragment_id': group_id, 'target_indices': group.target_indices,
                'fragment_indices': group.fragment_indices,
                'global_target_orbitals': tuple(int(static.frag_lolist[i][0])
                                                for i in group.fragment_indices),
                'budget': {'total_tolerance': controls.get('energy_tolerance'),
                           'effective_tolerance': settings.get('energy_tolerance'),
                           'original_target_count': len(static.fragments), 'group_count': len(groups)},
            }
            with _timed_section('stc.domain.native', totals, 'native'):
                stage_start = time.perf_counter()
                result = run_backend(prepared[1], metadata, settings,
                                     partial(backend.solve, with_grad=with_grad), comm=None)
                row['native'] = time.perf_counter() - stage_start
            with _timed_section('stc.domain.input_pullback', totals, 'input_pullback'):
                stage_start = time.perf_counter()
                if with_grad:
                    driver._add_bars(state, *apply_selected_pullback(prepared[2], prepared[0], result))
                state['energy'] = state['energy'] + result['energy']
                state['variance'] += float(result['energy_standard_error']) ** 2
                jax.block_until_ready((state['energy'], state['mf_bar'], state['common_bar']))
                row['input_pullback'] = time.perf_counter() - stage_start
            row['energy'] = float(result['energy'])
            row['energy_standard_error'] = float(result['energy_standard_error'])
            row['elapsed'] = time.perf_counter() - group_start
            group_rows.append(row)
            del prepared, result
            gc.collect()
    except Exception as exc:
        error = _error(exc, f'strong group {group_id}')
    wait_start = time.perf_counter()
    _raise_if_any_rank_failed(comm, error)
    totals['strong_wait'] = time.perf_counter() - wait_start
    phases['strong'] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()

    def apply_weak():
        energy, bars = weak_pair_contribution(
            state['mf'], state['common'], static, pairs, comm, with_grad=with_grad)
        state['energy'] = state['energy'] + energy
        if with_grad:
            driver._add_bars(state, *bars)
        jax.block_until_ready((state['energy'], state['mf_bar'], state['common_bar']))

    with _timed_section('stc.domain.weak', totals, 'weak'):
        _collective_call(comm, apply_weak, 'weak contribution assembly')
    phases['weak'] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    reduced_scalars = tree_sum_to_root(comm, (state['energy'], numpy.float64(state['variance'])))
    bars = tree_sum_to_root(comm, (state['mf_bar'], state['common_bar'])) if with_grad else None

    def install_reduced():
        if rank == 0:
            state['energy'], variance = reduced_scalars
            state['variance'] = float(variance)
            if with_grad:
                state['mf_bar'], state['common_bar'] = bars

    _collective_call(comm, install_reduced, 'correlation result assembly')
    phases['reduce'] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    with _timed_section('stc.domain.response', totals, 'response'):
        finish = lambda: driver._finish(state, with_grad=with_grad, include_hf=include_hf)
        result = (driver._root_call(comm, rank, finish, 'final response')
                  if scf_service is None else scf_service.run_final(finish))
    phases['response'] = time.perf_counter() - phase_start
    from . import _mpi_weak
    local_diagnostics = _collective_call(
        comm, lambda: {'rank': rank, 'stage_seconds': totals,
                       'groups': group_rows, 'weak': _mpi_weak.LAST_DIAGNOSTICS,
                       'scf_service': ({'enabled': False} if scf_service is None
                                       else scf_service.diagnostics())},
        'diagnostic staging')
    rank_diagnostics = comm.gather(local_diagnostics, root=0)

    def finish_diagnostics():
        global LAST_RUN_DIAGNOSTICS
        if rank == 0:
            LAST_RUN_DIAGNOSTICS = {
                'size': size, 'group_count': len(groups), 'group_assignment': 'round_robin',
                'parallel_scf_jk': scf_service is not None,
                'ranks': rank_diagnostics,
                'groups': sorted((row for diagnostic in rank_diagnostics for row in diagnostic['groups']),
                                 key=lambda row: row['global_position']),
                'phase_elapsed': phases, 'elapsed': time.perf_counter() - start,
                'conditional_energy_standard_error': math.sqrt(state['variance']),
                'budget': {'total_tolerance': controls.get('energy_tolerance'),
                           'effective_tolerance': settings.get('energy_tolerance')},
            }

    _collective_call(comm, finish_diagnostics, 'final diagnostic assembly')
    return result if rank == 0 else None
