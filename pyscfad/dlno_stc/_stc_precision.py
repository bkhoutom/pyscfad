"""Independent-replica precision of the final projected STC correction.

Pilots allocate a fixed, independent production ensemble. Its sample standard
error is empirical: finite pilots, iteration bias and approximation error are
not certified by an energy tolerance. No stochastic CC iteration is treated as
an independent observation here.
"""
import copy
import math
import warnings

import jax
import numpy


def enabled(controls):
    return controls is not None and controls.get('energy_tolerance') is not None


def validate_stages(controls, baseline, matrix, triples):
    """Exclude sources of random error not covered by the replica estimator."""
    if not enabled(controls):
        return
    for name, settings in (('baseline', baseline), ('matrix', matrix), ('triples', triples)):
        if settings is not None and settings.get('mode', 'deterministic') != 'deterministic':
            raise ValueError('STC molecular precision requires deterministic ' + name +
                             '; its stochastic uncertainty is not included')


def replica_controls(controls, fragment_index, stage, replica):
    """Stable separate forward/adjoint streams, independent of MPI ownership."""
    stage_id = {'pilot': 0, 'production': 1}[stage]
    result = copy.deepcopy(controls)
    seed = controls.get('seed', 0)
    grad_seed = controls.get('grad_seed')
    if grad_seed is None:
        grad_seed = seed
    for name, base, stream in (('seed', seed, 0), ('grad_seed', grad_seed, 1)):
        sequence = numpy.random.SeedSequence([base, 0x50524543, fragment_index,
                                              stage_id, replica, stream])
        result[name] = int(sequence.generate_state(1)[0])
    return result



def validate_seed_independence(controls, fragment_indices):
    """Reject collisions in either actual 32-bit adapter seed family.

    Hashing fragment identities does not mathematically prevent collisions.
    Check every possible pilot and production replica before launching work;
    adjoint keys have their own upstream stream, so their family is checked
    separately from forward keys. The legacy seed mapping stays unchanged.
    """
    if not enabled(controls):
        return
    candidates = controls['precision_pilot_replicas'] + controls['precision_max_replicas']
    if len(fragment_indices) * candidates > 2**32:
        raise ValueError('STC precision candidates exceed the 32-bit seed space; reduce fragments or replica limits')
    from .stc_ccsd import _fragment_seed
    observed = {'seed': {}, 'grad_seed': {}}
    for fragment in fragment_indices:
        for stage, count in (('pilot', controls['precision_pilot_replicas']),
                             ('production', controls['precision_max_replicas'])):
            for replica in range(count):
                config = replica_controls(controls, fragment, stage, replica)
                owner = (int(fragment), stage, replica)
                for family in observed:
                    actual = _fragment_seed(config[family], int(fragment))
                    previous = observed[family].get(actual)
                    if previous is not None:
                        raise ValueError(f'STC precision {family} seed collision {actual} between '
                                         f'{previous} and {owner}; choose another root seed or reduce '
                                         'precision_max_replicas')
                    observed[family][actual] = owner


def production_plan(pilot_energies, controls, nfragments):
    """Divide variance across all selected physical fragments, then use pilots."""
    if nfragments < 1:
        raise ValueError('STC molecular precision requires at least one selected fragment')
    target = float(controls['energy_tolerance']) / math.sqrt(nfragments)
    if not target > 0 or not math.isfinite(target) or target * target == 0:
        raise ValueError('STC molecular precision target is outside the finite variance range')
    energies = numpy.asarray(pilot_energies, dtype=float)
    if energies.size < 2 or not numpy.isfinite(energies).all():
        raise ValueError('STC precision pilots require at least two finite energies')
    variance = float(numpy.var(energies, ddof=1))
    requested = max(controls['precision_min_replicas'], math.ceil(variance / target**2))
    count = min(requested, controls['precision_max_replicas'])
    return dict(total_target_standard_error=float(controls['energy_tolerance']),
                selected_fragment_count=int(nfragments),
                fragment_target_standard_error=target,
                fragment_variance_fraction=1. / nfragments,
                pilot_replicas=len(energies), pilot_sample_variance=variance,
                requested_production_replicas=int(requested), production_replicas=int(count),
                capped=requested > count)


def _seed_record(config, diagnostics):
    """Preserve the actual adapter streams as well as their controller inputs."""
    cc = diagnostics.get('ccsd', {})
    return dict(seed_input=config['seed'], grad_seed_input=config['grad_seed'],
                seed=cc.get('seed'), grad_seed=cc.get('grad_seed'))


def run_replicas(evaluate, controls, fragment_index, nfragments, *, with_grad=False):
    """Average independently solved projected energies and optional cotangents.

    ``evaluate(config, with_grad)`` returns ``(energy, (parts, diagnostics), bars)``.
    It must consume a production VJP before returning; this function only retains
    one accumulated cotangent tree and scalar replica data, never VJP tapes.
    """
    pilot_energies, pilot_seeds, pilot_counts = [], [], []
    pilot_compression, production_compression = [], []
    for replica in range(controls['precision_pilot_replicas']):
        config = replica_controls(controls, fragment_index, 'pilot', replica)
        energy, (_, diagnostic), _ = evaluate(config, False)
        pilot_energies.append(float(energy))
        pilot_seeds.append(_seed_record(config, diagnostic))
        pilot_counts.append(diagnostic.get('ccsd', {}).get('nsamples_forward_production', 0))
        if 'auxiliary_compression' in diagnostic:
            pilot_compression.append(diagnostic['auxiliary_compression'])
    plan = production_plan(pilot_energies, controls, nfragments)
    count = plan['production_replicas']
    energies, parts, seeds, replica_diagnostics, triples_diagnostics = [], [], [], [], []
    sum_bars = None
    for replica in range(count):
        config = replica_controls(controls, fragment_index, 'production', replica)
        energy, (values, diagnostic), bars = evaluate(config, with_grad)
        energies.append(float(energy))
        parts.append([float(value) for value in values])
        seeds.append(_seed_record(config, diagnostic))
        cc = diagnostic.get('ccsd', {})
        replica_diagnostics.append({key: cc[key] for key in (
            'seed', 'grad_seed', 'converged', 'termination_mode', 'iterations',
            'averaged_count', 'nsamples_forward_production',
            'local_mean_iterate_stderr', 'projected_mean_iterate_stderr_iid') if key in cc})
        triples_diagnostics.append({key: value for key, value in diagnostic.items()
                                    if key not in ('ccsd', 'auxiliary_compression')})
        if 'auxiliary_compression' in diagnostic:
            production_compression.append(diagnostic['auxiliary_compression'])
        if with_grad:
            sum_bars = bars if sum_bars is None else jax.tree.map(lambda a, b: a + b, sum_bars, bars)
            jax.block_until_ready(sum_bars)
    values = numpy.asarray(parts)
    if not numpy.isfinite(energies).all() or not numpy.isfinite(values).all():
        raise ValueError('STC precision production returned nonfinite energies')
    se = float(numpy.std(energies, ddof=1) / math.sqrt(count))
    covariance = numpy.cov(values, rowvar=False, ddof=1)
    statistics = dict(plan, standard_error=se,
                      target_met=se <= plan['fragment_target_standard_error'],
                      pilot_correction_energies=pilot_energies,
                      production_correction_energies=energies,
                      production_component_energies=parts,
                      component_order=['mp2_lis', 'ccsd', 'triples'],
                      component_sample_covariance=covariance.tolist(),
                      ccsd_triples_covariance=float(covariance[1, 2]),
                      pilot_seeds=pilot_seeds, production_seeds=seeds,
                      variance_convention='independent_production_replica_sample_variance_over_count',
                      scope='sum_of_selected_projected_corrections',
                      limitation='Empirical SE; finite-pilot allocation is not a bound; excludes iteration and approximation bias and force uncertainty.')
    if statistics['capped'] or not statistics['target_met']:
        warnings.warn(f'STC precision fragment {fragment_index}: empirical SE {se:.6g} Eh, '
                      f'target {plan["fragment_target_standard_error"]:.6g} Eh; '
                      f'{count} production replicas, capped={statistics["capped"]}. '
                      'The requested precision is not certified.', RuntimeWarning, stacklevel=2)
    cc_diagnostic = dict(backend='stc', estimator='mean_of_independent_projected_replica_energies',
                         controls=copy.deepcopy(controls), precision=statistics,
                         replicas=replica_diagnostics,
                         nsamples_forward_production=sum(d.get('nsamples_forward_production', 0)
                                                         for d in replica_diagnostics),
                         nsamples_forward_pilot_replicas=sum(pilot_counts))
    diagnostic = {'ccsd': cc_diagnostic}
    if pilot_compression or production_compression:
        diagnostic['auxiliary_compression'] = dict(
            pilot=pilot_compression, production=production_compression)
    if any(triples_diagnostics):
        diagnostic['replica_triples_diagnostics'] = triples_diagnostics
    mean_bars = None if sum_bars is None else jax.tree.map(lambda value: value / count, sum_bars)
    return float(numpy.mean(energies)), (tuple(numpy.mean(values, axis=0)), diagnostic), mean_bars
