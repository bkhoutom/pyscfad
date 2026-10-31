"""Orbital-invariant Laplace (T) with physical fragment energy weights.

Each quadrature node dresses all six external numerator legs by half an
exponential. Reciprocal factors on contracted internal legs cancel exactly.
The existing native kernel then evaluates the dressed numerator with D=-1.
Full Fock-block response enters through stable matrix-function derivatives.

For a partial occupied target M=U U.T, the numerator weight is O^-1 M O.
The numerator factors this as L R.T, L=O^-1 U, R=O U, and contracts
directly over the physical selected space S. Whole-LIS targets use U=I.
"""
import importlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from pyscfad import lib
from .controls import DEFAULT_TRIPLES_TRACE_EXPONENT, _finite_control

NAMES = ('t1', 't2', 'Bvo', 'Boo', 'Bvv', 'fov')


def make_grid(eocc, evir, *, points=10):
    """Positive fitted triples quadrature, fixed during differentiation.

    Both 1/D and 1/D^2 enter the fit. Their measured errors are reported
    separately from stochastic uncertainty; the default is ten points.
    """
    from .controls import _integer_control
    from .laplace import fit_grid
    points = _integer_control('smooth_laplace_points', points)
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
    function = (native.triples_asymmetric_energy_vjp if with_grad
                else native.triples_asymmetric_energy)
    return function(*values[:5], *values[6:], nsamples=int(samples), seed=int(seed),
                    target_standard_error=float(target), fov=values[5], **options)


def _make_dresser(inputs):
    native = importlib.import_module('pyscfad.dlno_stc._stc_mp2')
    return native.TriplesDressing(tuple(np.asarray(x, dtype=np.float64) for x in inputs))


def make_solver(controls, grid, *, with_grad):
    nodes, weights, _ = grid
    removed = {'dressing', 'dressing_backend', 'adaptive'} & controls.keys()
    if removed:
        raise ValueError('removed triples options: ' + ', '.join(sorted(removed)))
    stochastic = controls['mode'] == 'stochastic'
    trace_exponent = _finite_control('trace_exponent',
        controls.get('trace_exponent', DEFAULT_TRIPLES_TRACE_EXPONENT))
    seed = controls.get('global_seed', 71)
    # Independent, discarded pilots choose the initial production allocation.
    # Adaptive production continues its own stream using its measured SE.
    seeds = np.random.SeedSequence([seed, 0x4c41504c]).generate_state(2 * len(nodes), dtype=np.uint64)

    def run(inputs):
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
        prepare = dresser.prepare
        prepare_seconds, pilot_seconds = time.perf_counter() - phase, 0.
        prepared = None
        if stochastic and not target:
            # Preserve legacy fixed-total-count allocation, which depends on
            # all pilot variances. Positive-target runs instead finish one
            # point at a time using its preassigned trace-model budget.
            phase = time.perf_counter()
            prepared = [prepare(float(node)) for node in nodes]
            jax.block_until_ready(prepared)
            prepare_seconds += time.perf_counter() - phase
            phase = time.perf_counter()
            pilots = [_native(values, pilot_count, seeds[i], with_grad=with_grad)
                      for i, values in enumerate(prepared)]
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
                                             'reverse_buffer_bytes')}
        setup_threads, sampling_threads = 0, 0
        point_energies, point_variances, point_samples, initial_errors = [], [], [], []
        for i, (node, weight) in enumerate(zip(nodes, weights)):
            phase = time.perf_counter()
            transformed = prepare(float(node)) if prepared is None else prepared[i]
            values = transformed if prepared is None else prepared[i]
            jax.block_until_ready(values)
            prepare_seconds += time.perf_counter() - phase
            native_target = 0.
            pilot = None
            number = int(allocation[i])
            if stochastic and target:
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
                             with_grad=with_grad, **options)
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
            # Shared preparation and DF pullback belong to the native call;
            # pilot_seconds contains only its discarded-pilot sampling phase.
            production_seconds += native_wall - pilot_time
            energy += weight * result['energy']
            variance += (weight * result['standard_error'])**2
            point_energies.append(weight * result['energy'])
            point_variances.append((weight * result['standard_error'])**2)
            point_samples.append(int(result['samples']))
            initial_errors.append(abs(weight) * result.get('pilot_standard_error', result['standard_error']))
            samples += int(result['samples'])
            native_seconds += result.get('sampling_seconds', 0.)
            for name in native_timings:
                native_timings[name] += result.get(name, 0.)
            for name in native_memory:
                native_memory[name] = max(native_memory[name], int(result.get(name, 0)))
            setup_threads = max(setup_threads, int(result.get('setup_threads', 0)))
            sampling_threads = max(sampling_threads, int(result.get('sampling_threads', 0)))
            if with_grad:
                phase = time.perf_counter()
                transformed_bars = tuple(np.asarray(result['cotangents'][name]) * weight
                                         for name in (*NAMES, 'L', 'R'))
                contribution = dresser.pullback(float(node), transformed_bars)
                for a, b in zip(bars, contribution):
                    np.add(a, b, out=a)
                jax.block_until_ready(bars)
                del contribution, transformed_bars
                pullback_seconds += time.perf_counter() - phase
            if prepared is not None:
                prepared[i] = None
            del result, values, transformed
        elapsed = time.perf_counter() - started
        diagnostics = dict(standard_error=np.sqrt(variance), samples=samples,
            pilot_samples=int(pilot_counts.sum()),
            laplace_points=count, sampling_seconds=native_seconds,
            smooth_triples_seconds=elapsed,
            laplace_prepare_seconds=prepare_seconds, pilot_seconds=pilot_seconds,
            production_seconds=production_seconds, laplace_pullback_seconds=pullback_seconds,
            overhead_seconds=elapsed-prepare_seconds-pilot_seconds-production_seconds-pullback_seconds,
            **native_timings, **native_memory,
            setup_threads=setup_threads, sampling_threads=sampling_threads,
            target_standard_error=target,
            trace_exponent=trace_exponent,
            adaptive=bool(stochastic and target),
            point_preparation_reused=bool(stochastic and target),
            asymmetric_dressing=True,
            native_dressing=True,
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
        return (jnp.asarray(energy), jax.tree_util.tree_map(jnp.asarray, diagnostics)), bars

    @jax.custom_vjp
    def evaluate(*inputs):
        return run(inputs)[0]

    def forward(*inputs):
        return run(inputs)

    def backward(bars, output_bar):
        return tuple(output_bar[0] * value for value in bars)

    evaluate.defvjp(forward, backward)

    def solver(mcc, eris, prjlo, *, t1, t2, verbose=0):
        no = t1.shape[0]
        # The identity frame projector and the physical fragment target are
        # different objects. A partial S retains U and its full response.
        u = jnp.eye(no, dtype=t1.dtype) if prjlo.shape == (no, no) else prjlo.T
        inputs = (t1, t2, eris.Lov, eris.Loo, lib.unpack_tril(eris.Lvv),
                  eris.fock[:no, no:], eris.fock[:no, :no], eris.fock[no:, no:], u)
        energy, diagnostics = evaluate(*inputs) if with_grad else run(inputs)[0]
        return energy, diagnostics

    return solver
