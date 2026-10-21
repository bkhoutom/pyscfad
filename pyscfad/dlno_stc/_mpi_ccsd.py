"""Complete CC group jobs on their owners, then close shared response on root.

Only fixed selections, numerical reference arrays, scalar records and completed
host cotangents cross MPI. Native matrix replay and Python pullbacks stay local.
"""
import gc

import jax
import numpy
from pyscfad import numpy as np

from . import driver
from ._mpi import (
    _broadcast_array_tree, _build_worker, _check_mpi_thread_support,
    _collect_tree_schema, _collective_call, _error, _raise_if_any_rank_failed,
    _reference_signature, _validate_shared_reference, tree_sum_to_root,
)
from ._selection import build_domain_groups
from ._workflow import allocate_domain_controls
from .adjoint import zero_term_cotangents


def _build_cc_worker(worker_initializer, canonical, signature):
    """Construct the worker reference used by the shared CC impurity solver."""
    mf = _build_worker(worker_initializer, canonical, signature)
    # Occupations are static metadata. PySCF's frozen-list orbital count
    # updates a mask derived from them, requiring a host NumPy array.
    mf.mo_occ = numpy.asarray(mf.mo_occ)
    # Keep a declared disk DF source in its original global auxiliary frame.
    # Without its placeholder, finite CC domains choose a local fitting model.
    source = getattr(mf.with_df, '_cderi_to_save', None)
    if source is not None and getattr(mf.with_df, '_prefer_cderi_to_save', False):
        mf.with_df.attach_outcore_cderi(source)
    return mf


def _run_mpi(initializer, worker_initializer, comm, *, with_grad,
             include_hf=False, merge_identical=True, verbose_imp=0,
             ccsd_t=False, return_components=False, include_baseline=True,
             fragment_indices=None):
    """Run round-robin group ownership with one root common/SCF response.

    ``initializer`` runs only on root and returns state, selections, baseline
    controls and matrix controls. Workers install its signed canonical state
    without SCF. This initial CC path uses root-only shared SCF/J/K work.
    """
    from . import ccsd

    rank, size = comm.Get_rank(), comm.Get_size()
    _collective_call(comm, _check_mpi_thread_support, 'CC MPI thread preflight')
    initialized = driver._root_call(comm, rank, initializer, 'CC initial preparation')
    state = initialized[0] if rank == 0 else None

    def root_metadata():
        _, static, baseline, matrix = initialized
        groups = build_domain_groups(static.mp2_static, merge_identical=merge_identical)
        indices = state['cc_fragment_indices']
        if not include_baseline:
            wanted = set(indices)
            groups = tuple(group for group in groups
                           if wanted.intersection(group.fragment_indices))
        options = dict(include_hf=include_hf, verbose_imp=verbose_imp, ccsd_t=ccsd_t,
                       triples_controls=state['cc_triples_controls'],
                       return_components=return_components, include_baseline=include_baseline,
                       fragment_indices=indices)
        return (static, baseline, matrix, groups, state['weak_pairs'], options,
                _reference_signature(state['mf']))

    metadata = driver._root_call(comm, rank, root_metadata, 'CC owner work selection')
    static, baseline, matrix, groups, pairs, options, signature = comm.bcast(metadata, root=0)
    shared = driver._root_call(comm, rank, lambda: {
        'canonical': {name: getattr(state['mf'], name)
                      for name in ('mo_coeff', 'mo_energy', 'mo_occ', 'e_tot')},
        'common': state['common'],
    }, 'CC shared reference selection')
    payload = _broadcast_array_tree(comm, shared)

    def setup():
        nonlocal state
        if rank != 0:
            mf = _build_cc_worker(worker_initializer, payload['canonical'], signature)
            state = dict(mf=mf, common=jax.tree_util.tree_map(np.asarray, payload['common']),
                         static=static.mp2_static, common_pullback=None,
                         scf_pullback=None, weak_pairs=pairs)
        _validate_shared_reference(state['mf'], state['common'], payload['canonical'],
                                   payload['common'], signature, static.mp2_static,
                                   with_grad=with_grad)
        state.update(energy=np.zeros((), dtype=state['common'].s1e.dtype),
                     variance=0., baseline_energy=0., weak_energy=0.)
        state['mf_bar'], state['common_bar'] = (
            zero_term_cotangents(state['mf'], state['common']) if with_grad else (None, None))

    _collective_call(comm, setup, 'CC worker reference setup')
    if with_grad:
        _collect_tree_schema(comm, (state['mf_bar'], state['common_bar']))
    del initialized, shared, payload, metadata
    allocated = allocate_domain_controls(baseline, len(groups))
    records, error, group_id = [], None, None
    try:
        for position in range(rank, len(groups), size):
            group = groups[position]
            group_id = group.representative_index
            records.extend(ccsd._run_group(
                state, static, group, allocated, matrix, with_grad=with_grad,
                verbose_imp=options['verbose_imp'], ccsd_t=options['ccsd_t'],
                triples_controls=options['triples_controls'],
                include_baseline=options['include_baseline'],
                fragment_indices=options['fragment_indices']))
            gc.collect()
    except Exception as exc:
        error = _error(exc, f'CC group {group_id}')
    _raise_if_any_rank_failed(comm, error)
    # Weak completion belongs to the separate baseline and is counted once.
    _collective_call(comm, lambda: ccsd._run_weak(state, static, with_grad=with_grad)
                     if rank == 0 and options['include_baseline'] else None,
                     'CC baseline weak completion')
    scalars = tree_sum_to_root(comm, (
        state['energy'], state['variance'], state['baseline_energy'], state['weak_energy']))
    bars = tree_sum_to_root(comm, (state['mf_bar'], state['common_bar'])) if with_grad else None
    gathered = comm.gather(records, root=0)

    def finish():
        state['energy'], variance, baseline_energy, weak_energy = scalars
        state.update(variance=float(variance), baseline_energy=float(baseline_energy),
                     weak_energy=float(weak_energy))
        if with_grad:
            state['mf_bar'], state['common_bar'] = bars
        records = [item for rows in gathered for item in rows]
        return ccsd._finish(state, static, records, with_grad=with_grad,
                            include_hf=options['include_hf'],
                            return_components=options['return_components'])

    return driver._root_call(comm, rank, finish, 'CC final shared response')
