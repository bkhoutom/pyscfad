"""MRCC-style symmetric half-Laplace partition, deterministic energy and VJP.

Uses the current (2024+) closed-shell spin contraction, not the pre-2024
local scalar expression. Each point builds W once, then differentiates
<W, A(W + L/2)> with a single combined connected cotangent. The existing
full-Fock disconnected numerator, including T2 Fov, is retained.
"""
import importlib
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from pyscf.lib import current_memory


def run_mrcc(inputs, nodes, weights, *, with_grad, max_memory):
    from .smooth_triples import _dressing_cache_bytes
    native = importlib.import_module('pyscfad.dlno_stc._stc_mp2')
    start = time.perf_counter()
    inputs = tuple(np.asarray(x, dtype=np.float64) for x in inputs)
    nodes, weights = np.asarray(nodes), np.asarray(weights)
    if nodes.ndim != 1 or nodes.shape != weights.shape or not len(nodes):
        raise ValueError('MRCC triples requires matching nonempty nodes and weights')
    if not np.isfinite(weights).all():
        raise ValueError('MRCC triples requires finite quadrature weights')
    engine = native.StableTriplesDressing(inputs, full_external=True, mrcc_partition=True)
    no, nv = inputs[0].shape
    ns = inputs[8].shape[1]
    blocks = ns*no*(no+1)//2
    block_bytes = nv**3*8*(2 if with_grad else 1)
    batch = min(128, blocks)
    cap = os.environ.get('PYSCFAD_TRIPLES_NUMERATOR_CACHE_MB')
    if cap is not None:
        cap = float(cap)
        if not np.isfinite(cap) or cap < 0:
            raise ValueError('PYSCFAD_TRIPLES_NUMERATOR_CACHE_MB must be finite and nonnegative')
        batch = min(batch, max(1, int(cap*1024**2)//block_bytes))
    bars = tuple(np.zeros_like(x) for x in inputs) if with_grad else None
    energies, prepare, kernel, pullback, build, singles, reverse = ([] for _ in range(7))
    maxima = dict(resident_tensor_bytes=0, reverse_buffer_bytes=0,
        deterministic_workspace_bytes=0, numerator_cache_peak_bytes=0,
        rank_four_packing_bytes=0, sampling_threads=0, setup_threads=0,
        occupied_block_batch_size=0)
    calls = groups = builds = 0
    profile_grouped = os.environ.get('PYSCFAD_TRIPLES_PROFILE_GROUPED', '0') == '1'
    grouped_profile, grouped_spin, grouped_plan, grouped_jobs, grouped_commit = ([] for _ in range(5))
    for node, weight in zip(nodes, weights):
        cache = _dressing_cache_bytes(max_memory) if with_grad else 0
        tick = time.perf_counter()
        prepared = engine.prepare(float(node), cache_limit_bytes=cache)
        prepare.append(time.perf_counter()-tick)
        budget = int(max(0., max_memory-current_memory()[0])*1e6)
        tick = time.perf_counter()
        result = native.triples_mrcc_point(prepared, float(weight), with_grad=with_grad,
            workspace_limit_bytes=budget, block_batch_size=batch, profile_grouped=profile_grouped)
        kernel.append(time.perf_counter()-tick)
        del prepared
        energies.append(result['energy'])
        build.append(result['connected_build_seconds'])
        singles.append(result['singles_seconds'])
        reverse.append(result['connected_reverse_seconds'])
        grouped_profile.append(result['grouped_profile_worker_seconds'])
        grouped_spin.append(result['grouped_spin_wall_seconds'])
        grouped_plan.append(result['grouped_plan_wall_seconds'])
        grouped_jobs.append(result['grouped_jobs_wall_seconds'])
        grouped_commit.append(result['grouped_commit_wall_seconds'])
        for name in maxima:
            maxima[name] = max(maxima[name], result[name])
        calls += result['connected_gemm_calls']
        builds += result['connected_build_blocks']
        groups += result['grouped_adjoint_gemm_calls']
        tick = time.perf_counter()
        if with_grad:
            contribution = engine.pullback(float(node), result.pop('cotangents'), cache_limit_bytes=cache)
            for total, value in zip(bars, contribution):
                np.add(total, value, out=total)
            del contribution
        pullback.append(time.perf_counter()-tick if with_grad else 0.)
    diagnostics = dict(standard_error=0., samples=0, pilot_samples=0, target_met=True,
        adaptive=False, target_standard_error=0., mrcc_partition=True, mrcc_spin_expression=2024,
        asymmetric_dressing=False, one_sided_dressing=False, stable_dressing=True,
        native_dressing=True, streaming_quartic_memory=True, grouped_left_adjoint=True,
        shared_connected_numerator=True, laplace_points=len(nodes),
        laplace_nodes=nodes, laplace_weights=weights, point_weighted_energies=np.array(energies),
        point_weighted_variances=np.zeros(len(nodes)), point_prepare_seconds=np.array(prepare),
        point_pullback_seconds=np.array(pullback), point_kernel_seconds=np.array(kernel),
        point_connected_build_seconds=np.array(build), point_singles_seconds=np.array(singles),
        point_connected_reverse_seconds=np.array(reverse),
        grouped_profile_enabled=profile_grouped,
        point_grouped_profile_worker_seconds=np.array(grouped_profile),
        point_grouped_spin_wall_seconds=np.array(grouped_spin),
        point_grouped_plan_wall_seconds=np.array(grouped_plan),
        point_grouped_jobs_wall_seconds=np.array(grouped_jobs),
        point_grouped_commit_wall_seconds=np.array(grouped_commit),
        laplace_prepare_seconds=sum(prepare), laplace_pullback_seconds=sum(pullback),
        production_seconds=sum(kernel), sampling_seconds=sum(kernel),
        smooth_triples_seconds=time.perf_counter()-start,
        deterministic_blocks=builds, connected_build_blocks=builds, connected_gemm_calls=calls,
        connected_forward_flops=12*blocks*len(nodes)*nv**3*(nv+no),
        grouped_adjoint_gemm_calls=groups, quadrature_point_batches=len(nodes),
        dressing_cache_peak_bytes=engine.peak_cache_bytes,
        diagonal_occupied_points=sum(engine.occupied_diagonal_at(float(t)) for t in nodes),
        diagonal_virtual_points=sum(engine.virtual_diagonal_at(float(t)) for t in nodes),
        selected_occupied_frame=engine.selected_occupied_frame,
        **maxima)
    return (jnp.asarray(sum(energies)), jax.tree_util.tree_map(jnp.asarray, diagnostics)), bars
