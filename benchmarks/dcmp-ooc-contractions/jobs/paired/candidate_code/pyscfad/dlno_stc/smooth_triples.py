"""Orbital-invariant Laplace (T) with physical fragment energy weights.

Each quadrature node dresses all six external numerator legs by half an
exponential. Reciprocal factors on contracted internal legs cancel exactly.
The existing native kernel then evaluates the dressed numerator with D=-1.
Full Fock-block response enters through stable matrix-function derivatives.

For a partial occupied target M=U U.T, the numerator weight is O^-1 M O.
The asymmetric path factors this as L R.T, L=O^-1 U, R=O U, and contracts
directly over S. The equal path retains the full nonsymmetric weight matrix.
"""
from functools import partial
import importlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from pyscfad import lib
from .smooth import block_exp
from .controls import DEFAULT_TRIPLES_TRACE_EXPONENT, _finite_control

NAMES = ('t1', 't2', 'Bvo', 'Boo', 'Bvv', 'fov')


def make_grid(eocc, evir, tolerance=1e-11, *, points=10):
    """Positive fitted triples quadrature, fixed during AD.

    The default is exactly ten points. Both 1/D and 1/D^2 enter the fit and
    their measured errors are recorded separately from stochastic uncertainty.
    ``points=None`` retains the older adaptive log-time Gaussian grid.
    """
    lo = 3 * float(np.min(evir) - np.max(eocc))
    hi = 3 * float(np.max(evir) - np.min(eocc))
    if lo <= 0:
        raise ValueError('smooth triples requires a positive occupied-virtual gap')
    if points is not None:
        from .controls import _integer_control
        from .laplace import fit_grid
        points = _integer_control('smooth_laplace_points', points)
        # Relative energy/derivative residuals over the full triples interval.
        # Scaling the objective avoids premature absolute solver termination.
        delta = np.geomspace(lo, hi, 300)
        probability = (delta / hi)**2
        probability /= probability.sum()
        try:
            nodes, weights = fit_grid(delta, 100 * probability, points, .1)
        except RuntimeError:
            # Retain the requested point count if the tighter objective does
            # not converge. The actual kernel errors below remain explicit.
            nodes, weights = fit_grid(delta, probability, points, .1)
        dense = np.geomspace(lo, hi, 10001)
        kernel = np.exp(-dense[:, None] * nodes)
        return nodes, weights, dict(kind='fitted', denominator_bounds=[lo, hi],
            points=points, lambda_gradient=.1,
            max_relative_energy_error=float(np.max(np.abs(dense * (kernel @ weights) - 1))),
            max_relative_derivative_error=float(np.max(np.abs(
                dense**2 * (kernel @ (weights * nodes)) - 1))))
    from scipy.special import roots_legendre
    # Integrate [0, small] directly, then [small, large] in logarithmic time.
    # This avoids spending many nodes on an almost constant integrand near 0.
    small, large = .05 / hi, -np.log(tolerance * .01) / lo
    x0, w0 = roots_legendre(4)
    first_nodes, first_weights = (x0 + 1) * small / 2, w0 * small / 2
    denominators = np.geomspace(lo, hi, 10001)
    half = np.log(large / small) / 2
    center = np.log(large * small) / 2
    for count in range(16, 97, 4):
        x, w = roots_legendre(count)
        log_nodes = np.exp(center + half * x)
        nodes = np.r_[first_nodes, log_nodes]
        weights = np.r_[first_weights, half * w * log_nodes]
        expval = np.exp(-denominators[:, None] * nodes)
        diagnostics = dict(denominator_bounds=[lo, hi], points=len(nodes),
            max_relative_energy_error=float(np.max(np.abs(denominators * (expval @ weights) - 1))),
            max_relative_derivative_error=float(np.max(np.abs(
                denominators**2 * (expval @ (weights * nodes)) - 1))))
        if max(diagnostics['max_relative_energy_error'],
               diagnostics['max_relative_derivative_error']) < 1e-9:
            return nodes, weights, diagnostics
    raise ValueError('Laplace quadrature failed its energy/derivative check')


def point_variance_fractions(foo, fvv, nodes, weights, *, exponent=DEFAULT_TRIPLES_TRACE_EXPONENT):
    """Trace-model variance shares, evaluated in log space.

    score_q = |w_q| [Tr exp(t_q F_oo/2) Tr exp(-t_q F_vv/2)]**exponent.
    The default 2.1 scales the empirical MP2 exponent 1.4 by 3/2 for triples.
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


def production_allocation(weighted_sigmas, fractions, target, *, minimum=10000):
    """Independent pilot estimates set counts for fixed weighted variance targets."""
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        counts = np.ceil((weighted_sigmas / (target * np.sqrt(fractions)))**2)
    if (not np.isfinite(counts).all() or np.any(counts < 0)
            or np.any(counts >= np.iinfo(np.int64).max)):
        raise ValueError('triples production sample budget exceeds int64 capacity')
    return np.maximum(minimum, counts.astype(np.int64))


@partial(jax.jit, static_argnames=('occupied_clusters', 'virtual_clusters'))
def dress(t1, t2, Bvo, Boo, Bvv, fov, foo, fvv, node, *,
          occupied_clusters, virtual_clusters):
    o = block_exp(foo, node / 2, occupied_clusters)
    v = block_exp(fvv, -node / 2, virtual_clusters)
    oi = block_exp(foo, -node / 2, occupied_clusters)
    vi = block_exp(fvv, node / 2, virtual_clusters)
    return (o @ t1 @ v,
            jnp.einsum('ip,jq,ar,bs,pqrs->ijab', o, o, v, v, t2,
                       optimize='optimal'),
            jnp.einsum('ij,Pjb,ba->Pia', o, Bvo, v, optimize='optimal'),
            jnp.einsum('ij,Pjk,kl->Pil', o, Boo, oi, optimize='optimal'),
            jnp.einsum('ab,Pbc,cd->Pad', v, Bvv, vi, optimize='optimal'),
            o @ fov @ v)


def _native(inputs, samples, seed, target=0., with_grad=True, *, pilot=None):
    native = importlib.import_module('pyscfad.dlno_stc._stc_mp2')
    values = [np.asarray(x, dtype=np.float64) for x in inputs]
    options = {} if pilot is None else {'pilot': pilot}
    if len(inputs) == 8:
        function = (native.triples_asymmetric_energy_vjp if with_grad
                    else native.triples_asymmetric_energy)
        return function(*values[:5], *values[6:], nsamples=int(samples), seed=int(seed),
                        target_standard_error=float(target), fov=values[5], **options)
    weighted = len(inputs) == 7
    function = (native.triples_weighted_energy_vjp if with_grad else native.triples_weighted_energy) if weighted else (
        native.triples_whole_energy_vjp if with_grad else native.triples_whole_energy)
    no, nv = values[0].shape
    extra = (values[6],) if weighted else ()
    return function(*values[:5], -np.ones(no) / 3, np.zeros(nv), *extra,
                    nsamples=int(samples), seed=int(seed),
                    target_standard_error=float(target), fov=values[5], **options)


def _make_dresser(inputs, occupied, virtual, asymmetric):
    native = importlib.import_module('pyscfad.dlno_stc._stc_mp2')
    return native.TriplesDressing(tuple(np.asarray(x, dtype=np.float64) for x in inputs),
        occupied.clusters, virtual.clusters, asymmetric)


def make_solver(controls, frames, grid, *, with_grad):
    nodes, weights, _ = grid
    occupied, virtual = frames
    dressing = controls.get('dressing', 'asymmetric')
    backend = controls.get('dressing_backend', 'native')
    if backend not in ('native', 'jax'):
        raise ValueError('triples dressing_backend must be native or jax')
    if dressing not in ('equal', 'asymmetric'):
        raise ValueError('triples dressing must be equal or asymmetric')
    dress_tensors = partial(dress, occupied_clusters=occupied.clusters,
                            virtual_clusters=virtual.clusters)
    @jax.jit
    def transform(*args):
        inputs, node = args[:-1], args[-1]
        values = dress_tensors(*inputs[:8], node)
        if len(inputs) == 9:
            foo, matrix = inputs[6], inputs[8]
            left = block_exp(foo, -node / 2, occupied.clusters)
            right = block_exp(foo, node / 2, occupied.clusters)
            if dressing == 'asymmetric':
                values = (*values, left @ matrix, right @ matrix)
            else:
                values = (*values, left @ matrix @ right)
        return values
    stochastic = controls['mode'] == 'stochastic'
    adaptive = bool(controls.get('adaptive', True))
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
        dresser = (_make_dresser(inputs, occupied, virtual, dressing == 'asymmetric')
                   if backend == 'native' else None)
        prepare = (dresser.prepare if dresser is not None else
                   lambda node: transform(*inputs, float(node)))
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
        bars = tuple((np if dresser is not None else jnp).zeros_like(x) for x in inputs) if with_grad else None
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
            if dresser is not None:
                transformed = prepare(float(node)) if prepared is None else prepared[i]
                pullback = lambda cotangents: dresser.pullback(float(node), cotangents)
            elif with_grad:
                # Build the primal and its pullback once, keeping only this
                # point's intermediates until its production has completed.
                transformed, pullback = jax.vjp(lambda *x: transform(*x, float(node)), *inputs)
            else:
                transformed = transform(*inputs, float(node)) if prepared is None else prepared[i]
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
                if adaptive:
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
                names = NAMES
                if len(inputs) == 9:
                    names = (*names, 'L', 'R') if dressing == 'asymmetric' else (*names, 'M')
                xp = np if dresser is not None else jnp
                transformed_bars = tuple(xp.asarray(result['cotangents'][name]) * weight for name in names)
                contribution = pullback(transformed_bars)
                if dresser is not None:
                    for a, b in zip(bars, contribution):
                        np.add(a, b, out=a)
                else:
                    bars = tuple(a + b for a, b in zip(bars, contribution))
                jax.block_until_ready(bars)
                del pullback, contribution, transformed_bars
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
            adaptive=bool(stochastic and target and adaptive),
            point_preparation_reused=bool(stochastic and target),
            asymmetric_dressing=bool(dressing == 'asymmetric' and len(inputs) == 9),
            native_dressing=bool(dresser is not None),
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
        inputs = (t1, t2, eris.Lov, eris.Loo, lib.unpack_tril(eris.Lvv),
                  eris.fock[:no, no:], eris.fock[:no, :no], eris.fock[no:, no:])
        if prjlo.shape != (no, no):
            inputs = (*inputs, prjlo.T if dressing == 'asymmetric' else prjlo.T @ prjlo)
        return evaluate(*inputs) if with_grad else run(inputs)[0]

    return solver
