"""Orbital-invariant Laplace (T) with physical fragment energy weights.

Each quadrature node dresses only external numerator legs. Internal contracted
legs are left undressed, using separate T2 forms for virtual and occupied
contractions. Physical projections are built directly with U.T and U.T O^2;
no inverse exponential is formed. Matrix-function derivatives retain the full
Fock and physical-projector response, including degenerate spectra.
Available cached CCSD integral blocks are reused and only missing blocks are
constructed from DF factors once per fragment, using the stable native numerator.
Deterministic evaluation uses the MRCC half-Laplace partition and grouped
adjoints in bounded occupied batches. Stochastic evaluation retains the
asymmetric selected functional. Prepared tensors and cotangents have at most
four orbital indices.
Semicanonical occupied and virtual factors use scalar axis scaling with full
matrix-function Fock pullbacks. Noncanonical blocks retain dense transforms.
"""
import importlib
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from pyscfad import lib
from pyscfad.tools import resource_profile
from pyscf.lib import current_memory, param
from .controls import DEFAULT_TRIPLES_TRACE_EXPONENT, _finite_control

NAMES = ('t1', 't2', 'Bvo', 'Boo', 'Bvv', 'fov')
INTEGRAL_NAMES = ('t1', 't2', 'ovvv', 'ovoo', 'ovov', 'fov')


def make_grid(eocc, evir, *, points=4):
    """Positive fitted triples quadrature, fixed during differentiation.

    Both 1/D and 1/D^2 enter the fit. Their measured errors are reported
    separately from stochastic uncertainty; the default is four points.
    """
    from .controls import _integer_control
    from .laplace import fit_grid
    points = _integer_control('triples_laplace_points', points)
    lo = 3 * float(np.min(evir) - np.max(eocc))
    hi = 3 * float(np.max(evir) - np.min(eocc))
    if lo <= 0:
        raise ValueError('smooth triples requires a positive occupied-virtual gap')
    delta = np.geomspace(lo, hi, 300)
    probability = (delta / hi)**2
    probability /= probability.sum()
    try:
        nodes, weights = fit_grid(delta, 100 * probability, points, .1)
    except RuntimeError:
        nodes, weights = fit_grid(delta, probability, points, .1)
    dense = np.geomspace(lo, hi, 10001)
    kernel = np.exp(-dense[:, None] * nodes)
    return nodes, weights, dict(kind='fitted', denominator_bounds=[lo, hi],
        points=points, lambda_gradient=.1,
        max_relative_energy_error=float(np.max(np.abs(dense * (kernel @ weights) - 1))),
        max_relative_derivative_error=float(np.max(np.abs(
            dense**2 * (kernel @ (weights * nodes)) - 1))))


def point_variance_fractions(foo, fvv, nodes, weights, *, exponent=DEFAULT_TRIPLES_TRACE_EXPONENT):
    """Trace-model variance shares, evaluated in log space.

    score_q = |w_q| [Tr exp(t_q F_oo/2) Tr exp(-t_q F_vv/2)]**exponent.
    The default is 5.0 for triples; the MP2 trace model uses 1.4.
    Spectra here choose sampling budgets only; no eigenvector AD is involved.
    """
    from scipy.special import logsumexp
    exponent = _finite_control('trace_exponent', exponent)
    eo, ev = np.linalg.eigvalsh(np.asarray(foo)), np.linalg.eigvalsh(np.asarray(fvv))
    half_time = np.asarray(nodes)[:, None] / 2
    log_trace = (logsumexp(half_time * (eo - eo.max()), axis=1)
                 + logsumexp(-half_time * (ev - ev.min()), axis=1)
                 - half_time[:, 0] * (ev.min() - eo.max()))
    log_mass = np.log(np.abs(weights)) + exponent * log_trace
    fractions = np.exp(log_mass - logsumexp(log_mass))
    if not np.isfinite(fractions).all() or np.any(fractions <= 0):
        raise ValueError('Laplace variance fractions must be finite and positive')
    return fractions / fractions.sum()


def _native(inputs, samples, seed, target=0., with_grad=True, *, pilot=None):
    native = importlib.import_module('pyscfad.dlno_stc._stc_mp2')
    values = [np.asarray(x, dtype=np.float64) for x in inputs]
    options = {} if pilot is None else {'pilot': pilot}
    function = getattr(native, 'triples_prepared_energy' + ('_vjp' if with_grad else ''))
    return function(tuple(values), nsamples=int(samples), seed=int(seed),
                    target_standard_error=float(target), **options)


def _make_dresser(inputs):
    native = importlib.import_module('pyscfad.dlno_stc._stc_mp2')
    return native.StableTriplesDressing(tuple(np.asarray(x, dtype=np.float64) for x in inputs))


def _dressing_cache_bytes(max_memory):
    """Use at most half current process headroom for one node's reverse tape."""
    available = max(0., float(max_memory) - current_memory()[0]) * .5e6
    cap = os.environ.get('PYSCFAD_TRIPLES_DRESSING_CACHE_MB')
    if cap is not None:
        cap = float(cap)
        if not np.isfinite(cap) or cap < 0:
            raise ValueError('PYSCFAD_TRIPLES_DRESSING_CACHE_MB must be finite and nonnegative')
        available = min(available, cap * 1024**2)
    return int(available)


def _integral_blocks(eris, no, nv):
    """Reuse materialized integrals in the current ERI frame, including AD.

    Inspect storage directly in the DF fallback path: probing a lazy property
    must not build a block just to decide whether it was already cached.
    """
    names = ('ovvv', 'ovoo', 'ovov')
    stored = getattr(eris, '__dict__', {})
    blocks = [stored.get(name) for name in names]
    if blocks[0] is not None:
        blocks[0] = lib.unpack_tril(blocks[0].reshape(no * nv, -1)).reshape(no, nv, nv, nv)
    if any(value is None for value in blocks):
        bvo = eris.Lov
        if blocks[0] is None:
            blocks[0] = jnp.einsum('Pia,Pbc->iabc', bvo, lib.unpack_tril(eris.Lvv))
        if blocks[1] is None:
            blocks[1] = jnp.einsum('Pia,Pjk->iajk', bvo, eris.Loo)
        if blocks[2] is None:
            blocks[2] = jnp.einsum('Pia,Pjb->iajb', bvo, bvo)
    return tuple(blocks)


def make_solver(controls, grid, *, with_grad):
    nodes, weights, _ = grid
    removed = {'dressing', 'dressing_backend', 'adaptive', 'integral_transform'} & controls.keys()
    if removed:
        raise ValueError('removed triples options: ' + ', '.join(sorted(removed)))
    stochastic = controls['mode'] == 'stochastic'
    partition = controls.get('partition', 'asymmetric' if stochastic else 'mrcc')
    if partition not in ('asymmetric', 'mrcc'):
        raise ValueError('triples partition must be asymmetric or mrcc')
    if partition == 'asymmetric' and not stochastic:
        raise ValueError('deterministic asymmetric triples were removed; use partition=mrcc')
    if partition == 'mrcc' and stochastic:
        raise ValueError('MRCC partition currently requires deterministic triples')
    if partition == 'mrcc' and not getattr(importlib.import_module(
            'pyscfad.dlno_stc._stc_mp2'), 'TRIPLES_MRCC_PARTITION', False):
        raise RuntimeError('Rebuild the STC native extension for the MRCC partition')
    if not stochastic and not getattr(importlib.import_module(
            'pyscfad.dlno_stc._stc_mp2'), 'TRIPLES_DETERMINISTIC_BLOCKS', False):
        raise RuntimeError('Rebuild the STC native extension for deterministic block Laplace triples')
    if not getattr(importlib.import_module('pyscfad.dlno_stc._stc_mp2'),
                   'TRIPLES_DIAGONAL_VIRTUAL_DRESSING', False):
        raise RuntimeError('Rebuild the STC native extension for diagonal virtual Laplace dressing')
    if not stochastic and not getattr(importlib.import_module(
            'pyscfad.dlno_stc._stc_mp2'), 'TRIPLES_QUADRATURE_STREAMING', False):
        raise RuntimeError('Rebuild the STC native extension for streaming Laplace triples')
    if not stochastic and not getattr(importlib.import_module(
            'pyscfad.dlno_stc._stc_mp2'), 'TRIPLES_GROUPED_ADJOINT', False):
        raise RuntimeError('Rebuild the STC native extension for grouped Laplace triples adjoints')
    native_options = {}
    trace_exponent = _finite_control('trace_exponent',
        controls.get('trace_exponent', DEFAULT_TRIPLES_TRACE_EXPONENT))
    seed = controls.get('global_seed', 71)
    # Independent, discarded pilots choose the initial production allocation.
    # Adaptive production continues its own stream using its measured SE.
    seeds = np.random.SeedSequence([seed, 0x4c41504c]).generate_state(2 * len(nodes), dtype=np.uint64)
    max_memory = 0.

    def run(inputs):
        if partition == 'mrcc':
            from .mrcc_triples import run_mrcc
            from .controls import DEFAULT_TRIPLES_WORKSPACE_GIB
            return run_mrcc(inputs, nodes, weights, with_grad=with_grad,
                max_memory=max_memory,
                workspace_gib=controls.get('workspace_gib', DEFAULT_TRIPLES_WORKSPACE_GIB))
        started = time.perf_counter()
        count = len(nodes)
        pilot_count = controls.get('pilot_samples', 10000)
        fractions = point_variance_fractions(inputs[6], inputs[7], nodes, weights,
                                             exponent=trace_exponent)
        target = controls.get('target_standard_error', 0.)
        point_targets = target**2 * fractions
        pilot_se = np.zeros(count)
        pilot_counts = np.zeros(count, dtype=np.int64)
        allocation = np.zeros(count, dtype=np.int64)
        phase = time.perf_counter()
        dresser = _make_dresser(inputs)
        diagonal_virtual_points = sum(dresser.virtual_diagonal_at(float(node)) for node in nodes)
        prepare = dresser.prepare
        prepare_seconds, pilot_seconds = time.perf_counter() - phase, 0.
        pilot_reverse_seconds = 0.
        prepared = None
        dressing_cache_peak_bytes = 0
        if not target:
            # Preserve legacy fixed-total-count allocation, which depends on
            # all pilot variances. Positive-target runs instead finish one
            # point at a time using its preassigned trace-model budget.
            phase = time.perf_counter()
            prepared = [prepare(float(node)) for node in nodes]
            jax.block_until_ready(prepared)
            prepare_seconds += time.perf_counter() - phase
            phase = time.perf_counter()
            # Pilot cotangents are discarded; only their variances allocate
            # the independent production streams.
            pilots = [_native(values, pilot_count, seeds[i], with_grad=False,
                              **native_options)
                      for i, values in enumerate(prepared)]
            # Pilots are energy-only. Retain the zero reverse-time field for
            # timing-report compatibility; only production computes VJPs.
            pilot_se = np.array([r['standard_error'] for r in pilots])
            pilot_counts[:] = pilot_count
            sigmas = np.abs(weights) * pilot_se * np.sqrt(pilot_count)
            sample_fractions = sigmas + max(float(sigmas.max()), 1e-300) * 1e-10
            sample_fractions /= sample_fractions.sum()
            allocation = np.maximum(2, np.ceil(controls['nsamples'] * sample_fractions).astype(np.int64))
            del pilots
            pilot_seconds += time.perf_counter() - phase
        energy, variance, samples = 0., 0., 0
        bars = tuple(np.zeros_like(x) for x in inputs) if with_grad else None
        native_seconds = 0.
        production_seconds, pullback_seconds = 0., 0.
        native_timings = {name: 0. for name in ('probability_setup_seconds',
            'amplitude_packing_seconds', 'vjp_reverse_seconds', 'projection_setup_seconds',
            'vjp_grouping_seconds', 'vjp_accumulation_seconds', 'vjp_finalize_seconds')}
        native_memory = {name: 0 for name in ('resident_tensor_bytes', 'proposal_bytes',
                                             'reverse_buffer_bytes',
                                             'deterministic_workspace_bytes')}
        deterministic_blocks = 0
        native_blas_threads = {name: 0 for name in ('packing_blas_threads',
                              'sampling_blas_threads', 'finalize_blas_threads')}
        setup_threads, sampling_threads = 0, 0
        point_energies, point_variances, point_samples, initial_errors = [], [], [], []
        for i, (node, weight) in enumerate(zip(nodes, weights)):
            phase = time.perf_counter()
            values = (prepare(float(node), cache_limit_bytes=(
                _dressing_cache_bytes(max_memory) if with_grad else 0))
                if prepared is None else prepared[i])
            dressing_cache_peak_bytes = max(dressing_cache_peak_bytes, dresser.cache_bytes)
            jax.block_until_ready(values)
            prepare_seconds += time.perf_counter() - phase
            native_target = 0.
            pilot = None
            number = int(allocation[i])
            if target:
                point_target = target * np.sqrt(fractions[i]) / abs(weight)
                if not np.isfinite(point_target) or point_target <= 0:
                    raise ValueError('triples point SE target is not representable')
                pilot = dict(samples=int(pilot_count), seed=int(seeds[i]),
                             allocation_target=float(point_target))
                number = 10000  # Minimum production count, independent of pilot size.
                # The native loop pools moments and cotangents, retaining every
                # production draw. A tiny inward rounding margin protects the
                # summed target from floating-point division/multiplication.
                native_target = point_target * (1 - 1e-12)
            phase = time.perf_counter()
            options = {} if pilot is None else {'pilot': pilot}
            result = _native(values, number, seeds[count + i], target=native_target,
                             with_grad=with_grad, **native_options, **options)
            # Native sampling has finished reading this node. Its returned
            # cotangents own their storage; dressing's reverse uses the original
            # inputs, so the full transformed tensors need not overlap it.
            if prepared is not None:
                prepared[i] = None
            del values
            native_wall = time.perf_counter() - phase
            if native_target and (not np.isfinite(result['standard_error']) or
                                  result['standard_error'] > native_target):
                raise RuntimeError('adaptive triples returned without meeting its point SE target')
            if pilot is not None:
                allocation[i] = int(result['pilot_samples'])
                pilot_counts[i] = int(result['independent_pilot_samples'])
                pilot_se[i] = result['independent_pilot_standard_error']
                pilot_time = result['independent_pilot_seconds']
                pilot_seconds += pilot_time
            else:
                pilot_time = 0.
            # Shared preparation and numerator pullback belong to the native call;
            # pilot_seconds contains only its discarded-pilot sampling phase.
            production_seconds += native_wall - pilot_time
            energy += weight * result['energy']
            variance += (weight * result['standard_error'])**2
            point_energies.append(weight * result['energy'])
            point_variances.append((weight * result['standard_error'])**2)
            point_samples.append(int(result['samples']))
            initial_errors.append(abs(weight) * result.get('pilot_standard_error', result['standard_error']))
            samples += int(result['samples'])
            deterministic_blocks += int(result.get('deterministic_blocks', 0))
            native_seconds += result.get('sampling_seconds', 0.)
            for name in native_timings:
                native_timings[name] += result.get(name, 0.)
            for name in native_memory:
                native_memory[name] = max(native_memory[name], int(result.get(name, 0)))
            for name in native_blas_threads:
                native_blas_threads[name] = max(native_blas_threads[name], int(result.get(name, 0)))
            setup_threads = max(setup_threads, int(result.get('setup_threads', 0)))
            sampling_threads = max(sampling_threads, int(result.get('sampling_threads', 0)))
            if with_grad:
                phase = time.perf_counter()
                transformed_bars = tuple(result['cotangents'])
                # The binding returns independent writable NumPy arrays, used
                # only here. Reuse them instead of allocating a weighted copy.
                for bar in transformed_bars:
                    np.multiply(bar, weight, out=bar)
                contribution = dresser.pullback(float(node), transformed_bars,
                    cache_limit_bytes=_dressing_cache_bytes(max_memory))
                dressing_cache_peak_bytes = max(dressing_cache_peak_bytes,
                                                 dresser.peak_cache_bytes)
                for a, b in zip(bars, contribution):
                    np.add(a, b, out=a)
                jax.block_until_ready(bars)
                del contribution, transformed_bars, bar
                pullback_seconds += time.perf_counter() - phase
            del result
        elapsed = time.perf_counter() - started
        diagnostics = dict(standard_error=np.sqrt(variance), samples=samples,
            pilot_samples=int(pilot_counts.sum()),
            laplace_points=count, sampling_seconds=native_seconds,
            smooth_triples_seconds=elapsed,
            laplace_prepare_seconds=prepare_seconds, pilot_seconds=pilot_seconds,
            pilot_reverse_seconds=pilot_reverse_seconds,
            production_seconds=production_seconds, laplace_pullback_seconds=pullback_seconds,
            overhead_seconds=elapsed-prepare_seconds-pilot_seconds-production_seconds-pullback_seconds,
            **native_timings, **native_memory, **native_blas_threads,
            setup_threads=setup_threads, sampling_threads=sampling_threads,
            target_standard_error=target,
            trace_exponent=trace_exponent,
            adaptive=bool(target),
            point_preparation_reused=bool(target),
            asymmetric_dressing=True,
            stable_dressing=True,
            native_dressing=True,
            deterministic_blocks=deterministic_blocks,
            diagonal_occupied_points=sum(dresser.occupied_diagonal_at(float(t)) for t in nodes),
            diagonal_virtual_points=diagonal_virtual_points,
            dressing_cache_peak_bytes=dressing_cache_peak_bytes,
            dressing_cache_hits=dresser.cache_hits,
            dressing_cache_rebuilds=dresser.cache_rebuilds,
            target_met=bool(not target or np.sqrt(variance) <= target),
            laplace_nodes=np.asarray(nodes), laplace_weights=np.asarray(weights),
            point_variance_fractions=fractions, point_variance_targets=point_targets,
            point_pilot_samples=pilot_counts,
            point_pilot_standard_errors=pilot_se,
            initial_production_samples=int(allocation.sum()),
            adaptive_samples=samples-int(allocation.sum()),
            point_initial_production_samples=np.asarray(allocation),
            point_initial_weighted_standard_errors=np.asarray(initial_errors),
            point_adaptive_samples=np.asarray(point_samples)-allocation,
            point_production_samples=np.asarray(point_samples),
            point_weighted_energies=np.asarray(point_energies),
            point_weighted_variances=np.asarray(point_variances))
        # Native numerator VJPs and dressing pullbacks run eagerly here,
        # during the forward call. These component times are inclusive parts
        # of solver.perturbative_triples, not additional elapsed work.
        resource_profile.checkpoint('stc.triples.native_timings',
            timings={name: value for name, value in diagnostics.items()
                     if name.endswith('_seconds')},
            blas_threads=native_blas_threads,
            with_grad=with_grad, seed=int(seed))
        return (jnp.asarray(energy), jax.tree_util.tree_map(jnp.asarray, diagnostics)), bars

    @jax.custom_vjp
    def evaluate(*inputs):
        return run(inputs)[0]

    def forward(*inputs):
        return run(inputs)

    def backward(bars, output_bar):
        if not resource_profile.enabled() or any(isinstance(value, jax.core.Tracer)
                for value in jax.tree_util.tree_leaves((bars, output_bar))):
            return tuple(output_bar[0] * value for value in bars)
        with resource_profile.section('stc.triples.pullback_scale', seed=int(seed)):
            scaled = tuple(output_bar[0] * value for value in bars)
            jax.block_until_ready(scaled)
        return scaled

    evaluate.defvjp(forward, backward)

    def solver(mcc, eris, prjlo, *, t1, t2, verbose=0):
        nonlocal max_memory
        max_memory = getattr(mcc, 'max_memory', param.MAX_MEMORY)
        no = t1.shape[0]
        # The identity frame projector and the physical fragment target are
        # different objects. A partial S retains U and its full response.
        u = jnp.eye(no, dtype=t1.dtype) if prjlo.shape == (no, no) else prjlo.T
        # Cached tensors retain the same physical integral response as DF
        # construction. Missing blocks are contracted once, outside quadrature.
        tensors = _integral_blocks(eris, no, t1.shape[1])
        inputs = (t1, t2, *tensors,
                  eris.fock[:no, no:], eris.fock[:no, :no], eris.fock[no:, no:], u)
        energy, diagnostics = evaluate(*inputs) if with_grad else run(inputs)[0]
        return energy, diagnostics

    return solver
