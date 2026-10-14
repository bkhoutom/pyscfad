"""Owner-distributed compact weak endpoints and their bounded response.

Only compact numerical endpoint values cross MPI. AO preparations and reverse
mode tapes stay on their deterministic endpoint owner, one tape at a time.
"""
from functools import partial
import gc
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import _pairs
from .adjoint import add_cotangent, zero_term_cotangents


LAST_DIAGNOSTICS = {}


def _error(rank, phase, exc, index=None):
    subject = '' if index is None else f' endpoint/pair {index}'
    return f'rank {rank} weak {phase}{subject}: {type(exc).__name__}: {exc}'


def _exchange_endpoints(comm, prepared, indices, owners, check):
    """Allgather a small shape header and a typed compact numeric payload.

    Byte transport preserves each leaf's numerical dtype without pickling its
    values. Every conversion and receive allocation precedes a shared failure
    check, so Python allocation errors cannot strand peers in Allgatherv.
    """
    from mpi4py import MPI

    rank = comm.Get_rank()
    error = None
    try:
        header, chunks = [], []
        for index, endpoint in prepared.items():
            description = []
            for leaf in endpoint:
                if leaf is None:
                    description.append(None)
                    continue
                array = np.ascontiguousarray(jax.device_get(leaf))
                if array.dtype.kind not in 'biufc':
                    raise TypeError(f'unsupported weak endpoint dtype {array.dtype}')
                description.append((array.shape, array.dtype.str, array.nbytes))
                chunks.append(array.reshape(-1).view(np.uint8))
            header.append((index, tuple(description)))
        send = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint8)
    except Exception as exc:
        error = _error(rank, 'endpoint exchange staging', exc)
    check(comm, error)
    headers = comm.allgather(header)

    error = None
    try:
        counts, descriptions, seen = [], [], set()
        for source, source_header in enumerate(headers):
            count = 0
            for index, description in source_header:
                if index not in owners or owners[index] != source or index in seen:
                    raise ValueError(f'invalid weak endpoint owner/header for {index}')
                if len(description) != 5:
                    raise ValueError(f'invalid weak endpoint structure for {index}')
                seen.add(index)
                for record in description:
                    if record is not None:
                        shape, dtype, nbytes = record
                        dtype = np.dtype(dtype)
                        if (dtype.kind not in 'biufc'
                                or nbytes != int(np.prod(shape))*dtype.itemsize):
                            raise ValueError(f'invalid weak endpoint leaf for {index}')
                        count += nbytes
                descriptions.append((index, description))
            counts.append(count)
        if seen != set(indices):
            raise ValueError('shared weak endpoint IDs differ from the fixed graph')
        offsets = np.cumsum([0] + counts[:-1], dtype=np.int64)
        receive = np.empty(sum(counts), dtype=np.uint8)
        receive_spec = [receive, counts, offsets, MPI.BYTE]
        send_spec = [send, MPI.BYTE]
    except Exception as exc:
        error = _error(rank, 'endpoint exchange allocation', exc)
    check(comm, error)
    comm.Allgatherv(send_spec, receive_spec)

    error = None
    try:
        endpoints, offset = {}, 0
        for index, description in descriptions:
            leaves = []
            for record in description:
                if record is None:
                    leaves.append(None)
                else:
                    shape, dtype, nbytes = record
                    array = receive[offset:offset+nbytes].view(np.dtype(dtype)).reshape(shape)
                    leaves.append(jnp.asarray(array))
                    offset += nbytes
            endpoints[index] = tuple(leaves)
        jax.block_until_ready(endpoints)
    except Exception as exc:
        error = _error(rank, 'endpoint exchange reconstruction', exc)
    check(comm, error)
    return endpoints


def weak_pair_contribution(mf, common, static, pairs, comm, *, with_grad):
    """Return this rank's weak energy and optional MF/common cotangent trees.

    Pair positions and sorted incident endpoint positions have independent
    round-robin owners. Compact endpoint bars are summed to owners before any
    owner enters its independently sized local response loop. MPI never runs
    inside a differentiated function. Energy-only calls return ``None`` bars
    and skip cotangent preparation entirely.
    """
    from ._mpi import _raise_if_any_rank_failed, tree_sum_to_root

    check = _raise_if_any_rank_failed
    rank, size = comm.Get_rank(), comm.Get_size()
    diagnostics = dict(rank=rank, n_endpoint_builds=0, n_pairs=0,
                       n_endpoint_replays=0)
    LAST_DIAGNOSTICS.clear()
    LAST_DIAGNOSTICS.update(diagnostics)
    start = time.perf_counter()
    error = None
    try:
        _pairs.configure_compilation_cache()
        pairs = tuple(pairs)
        indices = sorted({int(index) for pair in pairs for index in pair})
        owners = {index: slot % size for slot, index in enumerate(indices)}
        owned_indices = tuple(index for index in indices if owners[index] == rank)
        bars = zero_term_cotangents(mf, common) if with_grad else None
        energy = jnp.zeros((), dtype=common.s1e.dtype)
        prepared = {}
        # Allocate the owner map before the ordered numerical reductions.
        owned_bars = dict.fromkeys(owned_indices)
    except Exception as exc:
        error = _error(rank, 'setup', exc)
    check(comm, error)
    diagnostics['setup_seconds'] = time.perf_counter() - start
    if not indices:
        LAST_DIAGNOSTICS.update(diagnostics)
        return energy, bars

    start, error, index = time.perf_counter(), None, None
    try:
        for index in owned_indices:
            prepared[index] = _pairs.build_weak_endpoint(mf, common, static, index)
            diagnostics['n_endpoint_builds'] += 1
        jax.block_until_ready(prepared)
    except Exception as exc:
        error = _error(rank, 'endpoint construction', exc, index)
    diagnostics['endpoint_build_work_seconds'] = time.perf_counter() - start
    check(comm, error)
    diagnostics['endpoint_build_seconds'] = time.perf_counter() - start
    start = time.perf_counter()
    endpoints = _exchange_endpoints(comm, prepared, indices, owners, check)
    del prepared
    diagnostics['endpoint_exchange_seconds'] = time.perf_counter() - start

    start, error, pair = time.perf_counter(), None, None
    try:
        endpoint_bars = ({index: jax.tree_util.tree_map(jnp.zeros_like, endpoints[index])
                          for index in indices} if with_grad else None)
        for pair in pairs[rank::size]:
            left, right = pair
            if with_grad:
                value, (left_bar, right_bar) = _pairs.pair_energy_and_grad(
                    endpoints[left], endpoints[right], static.thresholds.multipole_order)
                endpoint_bars[left] = jax.tree_util.tree_map(
                    add_cotangent, endpoint_bars[left], left_bar)
                endpoint_bars[right] = jax.tree_util.tree_map(
                    add_cotangent, endpoint_bars[right], right_bar)
                del left_bar, right_bar
            else:
                value = _pairs._multipole_pair_energy(
                    endpoints[left], endpoints[right], static.thresholds.multipole_order)
            energy = energy + value
            diagnostics['n_pairs'] += 1
        jax.block_until_ready((energy, endpoint_bars))
    except Exception as exc:
        error = _error(rank, 'pair contraction', exc, pair)
    diagnostics['pair_work_seconds'] = time.perf_counter() - start
    check(comm, error)
    diagnostics['pair_seconds'] = time.perf_counter() - start

    start = time.perf_counter()
    if with_grad:
        # All ranks visit this same order; no collectives in owner-only loops.
        for index in indices:
            reduced = tree_sum_to_root(comm, endpoint_bars[index], root=owners[index])
            if owners[index] == rank:
                owned_bars[index] = reduced
        del endpoint_bars, reduced
    diagnostics['endpoint_reduce_seconds'] = time.perf_counter() - start

    start, error, index = time.perf_counter(), None, None
    try:
        if with_grad:
            for index in owned_indices:
                original = endpoints.pop(index)
                endpoint_bar = owned_bars.pop(index)
                # Alignment must be inside the VJP to put its derivative in
                # the original frame used by the distributed pair bars.
                rebuilt, pullback = jax.vjp(
                    partial(_pairs._replay_weak_endpoint, static=static,
                            fragment_index=index, reference=original), mf, common)
                _pairs._check_endpoint_replay(rebuilt, original, index)
                contribution = pullback(endpoint_bar)
                bars = jax.tree_util.tree_map(add_cotangent, bars, contribution)
                jax.block_until_ready(bars)
                diagnostics['n_endpoint_replays'] += 1
                del rebuilt, pullback, original, contribution, endpoint_bar
        del endpoints, owned_bars
        gc.collect()
    except Exception as exc:
        error = _error(rank, 'endpoint response', exc, index)
    diagnostics['endpoint_response_work_seconds'] = time.perf_counter() - start
    check(comm, error)
    diagnostics['endpoint_response_seconds'] = time.perf_counter() - start
    LAST_DIAGNOSTICS.update(diagnostics)
    return jnp.real(energy), bars
