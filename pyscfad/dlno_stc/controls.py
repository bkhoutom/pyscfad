"""Shared defaults for high-level STC-MP2 drivers."""

from numbers import Real

import numpy

DEFAULT_ENERGY_TOLERANCE = 3e-4  # Hartree (0.3 mHa)
DEFAULT_TRIPLES_TRACE_EXPONENT = 5.0


def default_laplace_grid():
    """Return independent arrays for the original six-point fixed grid."""
    roots = numpy.array([0.003431, 0.023534, 0.088984, 0.275603, 0.757121, 1.906218])
    weights = numpy.array([0.009348, 0.035196, 0.107559, 0.293035, 0.729094, 1.690608])
    return roots * 2.6, weights * 2.6


def resolve_mp2_grid(grid=None):
    """Own and validate an MP2/W (roots, weights) pair; never a triples grid."""
    if grid is None:
        grid = default_laplace_grid()
    if not isinstance(grid, (tuple, list)) or len(grid) != 2:
        raise ValueError('mp2_grid must be a (roots, weights) pair')
    result = {}
    for name, raw in zip(('laplace_roots', 'laplace_weights'), grid):
        if raw is None or numpy.iscomplexobj(raw):
            raise ValueError(f'{name} must be a nonempty finite real vector')
        try:
            values = numpy.array(raw, dtype=numpy.float64, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{name} must be a nonempty finite real vector') from exc
        if values.ndim != 1 or not values.size or not numpy.isfinite(values).all():
            raise ValueError(f'{name} must be a nonempty finite real vector')
        result[name] = values
    if result['laplace_roots'].shape != result['laplace_weights'].shape or numpy.any(result['laplace_roots'] < 0):
        raise ValueError('Laplace roots/weights must match and roots must be nonnegative')
    return result


_MATRIX_COMMON = frozenset((
    'mode', 'laplace_roots', 'laplace_weights', 'global_seed', 'threads',
    'target_batch_size', 'memory_limit_mb', 'auxiliary_group_size',
))
_MATRIX_SAMPLING = {
    'matrix_tolerance': 1e-4,
    'domain_tau': 0.003,
    'virtual_domain_tau': 0.0003,
    'pair_pilot_samples': 100000,
    'min_samples': 64,
}


def _integer_control(name, value, minimum=1):
    if (not isinstance(value, (int, numpy.integer))
            or isinstance(value, (bool, numpy.bool_)) or value < minimum):
        adjective = 'nonnegative' if minimum == 0 else 'positive'
        raise ValueError(f'{name} must be a {adjective} integer')
    return int(value)


def _finite_control(name, value, *, positive=True):
    try:
        if isinstance(value, (bool, numpy.bool_)):
            raise ValueError
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be finite and positive') from exc
    if not numpy.isfinite(value) or (value <= 0 if positive else value < 0):
        adjective = 'positive' if positive else 'nonnegative'
        raise ValueError(f'{name} must be finite and {adjective}')
    return value


def _triples_integral_transform(controls):
    value = controls.get('integral_transform', 'df')
    if not isinstance(value, str) or value not in ('df', 'rotate'):
        raise ValueError('integral_transform must be df or rotate')
    return value


def normalize_triples_controls(controls, *, ccsd_t):
    """Validate triples independently of baseline and W selection settings.

    Omitted controls select deterministic evaluation. Sampling uses the
    asymmetric Laplace numerator and physical selected-space restriction.
    A positive energy tolerance enables independent pilots and adaptive
    production; otherwise nsamples sets a fixed production budget.
    """
    if not ccsd_t:
        if controls is not None:
            raise ValueError('triples_controls requires ccsd_t=True')
        return None
    if controls is None:
        return {'mode': 'deterministic', 'partition': 'mrcc'}
    if not isinstance(controls, dict):
        raise TypeError('triples_controls must be a dictionary')
    settings = controls.copy()
    unknown = set(settings) - {'mode', 'nsamples', 'global_seed', 'energy_tolerance',
                               'pilot_samples', 'fragment_energies', 'trace_exponent',
                               'integral_transform', 'variance_allocation', 'partition'}
    if unknown:
        raise ValueError('unsupported triples controls: '
                         + ', '.join(sorted(map(str, unknown))))
    mode = settings.setdefault('mode', 'deterministic')
    if mode not in ('deterministic', 'stochastic'):
        raise ValueError('triples mode must be deterministic or stochastic')
    partition = settings.get('partition', 'mrcc' if mode == 'deterministic' else 'asymmetric')
    if partition not in ('asymmetric', 'mrcc'):
        raise ValueError('triples partition must be asymmetric or mrcc')
    if mode == 'deterministic':
        if partition != 'mrcc':
            raise ValueError('deterministic asymmetric triples were removed; use partition=mrcc')
        settings['partition'] = 'mrcc'
    else:
        if partition != 'asymmetric':
            raise ValueError('MRCC partition currently requires deterministic triples')
        settings.pop('partition', None)
    if _triples_integral_transform(settings) == 'df':
        # Explicit default and omission share a checkpoint identity.
        settings.pop('integral_transform', None)
    if mode == 'deterministic':
        if set(settings) - {'mode', 'integral_transform', 'partition'}:
            raise ValueError('deterministic triples do not accept sampling controls')
        return settings
    allocation = settings.get('variance_allocation', 'mp2_energy')
    if allocation not in ('mp2_energy', 'occupied_count'):
        raise ValueError('variance_allocation must be mp2_energy or occupied_count')
    if allocation == 'occupied_count' and 'fragment_energies' in settings:
        raise ValueError('occupied_count allocation does not use fragment_energies')
    if 'nsamples' not in settings:
        raise ValueError('stochastic triples require integer nsamples >= 2')
    settings['nsamples'] = _integer_control('nsamples', settings['nsamples'], minimum=2)
    settings['global_seed'] = _integer_control('global_seed', settings.get('global_seed', 71), minimum=0)
    if settings['global_seed'] >= 2**64:
        raise ValueError('global_seed must fit in uint64')
    if 'pilot_samples' in settings:
        settings['pilot_samples'] = _integer_control(
            'pilot_samples', settings['pilot_samples'], minimum=2)
    if 'trace_exponent' in settings:
        settings['trace_exponent'] = _finite_control('trace_exponent', settings['trace_exponent'])
    if 'fragment_energies' in settings:
        raw = settings['fragment_energies']
        if numpy.iscomplexobj(raw):
            raise ValueError('fragment_energies must be finite real correlation energies')
        try:
            energies = numpy.asarray(raw, dtype=numpy.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('fragment_energies must be finite real correlation energies') from exc
        if energies.ndim != 1 or not energies.size or not numpy.isfinite(energies).all():
            raise ValueError('fragment_energies must be a nonempty finite real vector')
        settings['fragment_energies'] = tuple(float(x) for x in energies)
    if 'energy_tolerance' in settings:
        raw_tolerance = settings['energy_tolerance']
        if (isinstance(raw_tolerance, (bool, numpy.bool_))
                or not isinstance(raw_tolerance, Real) or raw_tolerance < 0):
            raise ValueError('energy_tolerance must be a finite nonnegative real scalar')
        try:
            tolerance = float(raw_tolerance)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('energy_tolerance must be representable as float64') from exc
        if not numpy.isfinite(tolerance) or (raw_tolerance != 0 and tolerance == 0.):
            raise ValueError('energy_tolerance must be finite and representable as float64')
        if tolerance:
            settings['energy_tolerance'] = tolerance
        else:
            # Zero and omission share one fixed-count checkpoint identity.
            del settings['energy_tolerance']
    return settings


def normalize_matrix_controls(controls):
    """Copy and validate the explicit native W interface controls.

    Sampling counts are chosen by the native precision-targeted allocation;
    energy-only fixed-count options do not belong to this matrix interface.
    The default grid and caller-supplied grid arrays are independently owned.
    """
    if not isinstance(controls, dict):
        raise TypeError('controls must be a dictionary')
    settings = controls.copy()
    allowed = _MATRIX_COMMON | _MATRIX_SAMPLING.keys()
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError('unsupported matrix controls: ' + ', '.join(sorted(map(str, unknown))))
    mode = settings.setdefault('mode', 'deterministic')
    if mode not in ('deterministic', 'stochastic'):
        raise ValueError('mode must be deterministic or stochastic')
    if mode == 'deterministic':
        invalid = settings.keys() & _MATRIX_SAMPLING.keys()
        if invalid:
            raise ValueError('deterministic matrices do not accept sampling controls: '
                             + ', '.join(sorted(invalid)))
    else:
        for name, value in _MATRIX_SAMPLING.items():
            settings.setdefault(name, value)
    settings.update(resolve_mp2_grid(
        None if not {'laplace_roots', 'laplace_weights'} & settings.keys() else
        (settings.get('laplace_roots'), settings.get('laplace_weights'))))
    weights = settings['laplace_weights']
    if mode == 'stochastic' and numpy.any(weights <= 0):
        raise ValueError('stochastic matrix Laplace weights must be strictly positive')
    for name, default in (('global_seed', 0), ('threads', 1),
                          ('auxiliary_group_size', 32)):
        settings[name] = _integer_control(name, settings.get(name, default),
                                         minimum=0 if name == 'global_seed' else 1)
    # Share the domain's Laplace factors across the physical fragment by
    # default. An explicit batch cap remains available for memory constraints.
    settings.setdefault('target_batch_size', None)
    if settings['target_batch_size'] is not None:
        settings['target_batch_size'] = _integer_control(
            'target_batch_size', settings['target_batch_size'])
    if settings.get('memory_limit_mb') is not None:
        settings['memory_limit_mb'] = _finite_control('memory_limit_mb', settings['memory_limit_mb'])
    if mode == 'stochastic':
        for name in ('pair_pilot_samples', 'min_samples'):
            settings[name] = _integer_control(name, settings[name], minimum=2)
        settings['matrix_tolerance'] = _finite_control('matrix_tolerance', settings['matrix_tolerance'])
        for name in ('domain_tau', 'virtual_domain_tau'):
            if name in settings:
                settings[name] = _finite_control(name, settings[name], positive=False)
    return settings


def estimate_matrix_memory(naux, nocc, nvir, ntarget, controls, *,
                           with_grad=True, auxiliary_group_count=None):
    """Conservative bytes newly needed by one staged matrix call and reverse.

    Counts include host/native layout copies, native factors, matrix output/SE
    arrays, per-worker accumulators and worst-case compact-domain caches. They
    exclude existing inputs, SCF/preparation tapes, allocator/BLAS overhead and
    the rest of the process. Add current process residency when enforcing a
    process budget. No allocation proportional to all targets is assumed for
    deterministic calls beyond the requested target batch. The default batch
    contains all targets in the physical fragment and is counted in full.
    """
    settings = normalize_matrix_controls(controls)
    dims = [_integer_control(name, value, minimum=0 if name in ('naux', 'nvir') else 1)
            for name, value in
            (('naux', naux), ('nocc', nocc), ('nvir', nvir), ('ntarget', ntarget))]
    naux, nocc, nvir, ntarget = dims
    if ntarget > nocc:
        raise ValueError('ntarget must not exceed nocc')
    factor = 8 * naux * nocc * nvir
    matrices = 8 * (nocc * nocc + nvir * nvir)
    threads = settings['threads']
    batch = ntarget if settings['target_batch_size'] is None else min(ntarget, settings['target_batch_size'])
    single_amplitude = 8 * nocc * nvir * nvir
    breakdown = {
        'input_copies': 2 * (factor + matrices),
        'factor_buffers': (8 if with_grad else 4) * factor + 8 * matrices,
        'amplitude_buffers': 0,
        'accumulators': 0,
        'compact_caches': 0,
        'atom_weights': 0,
        'proposal_caches': 0,
        'locks': 0,
        'reverse_updates': 0,
        'scratch': 0,
        'output_buffers': (8 if with_grad else 4) * matrices,
    }
    if naux == 0 or nvir == 0:
        breakdown = {name: value if name == 'output_buffers' else 0
                     for name, value in breakdown.items()}
        breakdown['total_bytes'] = breakdown['output_buffers']
        return breakdown
    # Selected amplitudes, reverse amplitudes, and single-target spin/Gram
    # cubes. Deterministic W uses no spin-amplitude cube in either pass.
    if settings['mode'] == 'deterministic':
        extra = 3 if with_grad else 0
        # W rotates the adapter's private input DF in place and reuses C for
        # its final gradient rotation. Neither rotation adds a full factor.
        breakdown['amplitude_buffers'] = ((2 if with_grad else 1) * batch + extra) * single_amplitude
        # Occupied W batches up to four spin-Gram columns per worker for
        # GEMM reuse. Reserve this panel in addition to existing scratch.
        panel = min(4, max(1, nocc // threads))
        breakdown['scratch'] = 8 * threads * ((6 + panel) * nvir * nvir + 2 * naux * nvir)
    else:
        points = 2
        ngroup = ((naux + settings['auxiliary_group_size'] - 1) // settings['auxiliary_group_size']
                  if auxiliary_group_count is None else
                  _integer_control('auxiliary_group_count', auxiliary_group_count))
        # Each point has propagated values/bars, scores and alias tables.
        breakdown['factor_buffers'] += points * ((2 if with_grad else 1) * factor
            + 8 * nocc * nvir * (8 * ngroup + 8))
        # Every resident point owns atom_weight. W's fixed B and dressed C
        # also keep atom_l1 for the occupied mixture and virtual reform.
        breakdown['atom_weights'] = 16 * points * ngroup * nocc * nvir
        # Occupied and virtual B/C aliases coexist. Reserve 16 bytes per
        # probability/index entry plus headers and normalization totals.
        alias_count = (4 * ngroup + 5) * nvir + (3 * ngroup + 10) * nocc
        breakdown['proposal_caches'] = (
            16 * nocc * nvir * (7 * ngroup + 15) + 72 * alias_count)
        # Default per-stream proposals retain all alias entries and vector
        # objects alongside the shared caches. Occupied and virtual phases
        # run sequentially; reserve the larger, including inline metadata
        # for at most twelve proposals. Construction scratch stays below.
        occupied_entries = ngroup * ((11 * batch + 3) * nvir + 4 * batch + 3)
        occupied_aliases = ngroup * (8 * batch + 6)
        virtual_entries = (ngroup * (3 * nocc * nvir + 3 * batch * nvir + 9 * nocc)
                           + 12 * nocc * nvir + 4 * nvir)
        virtual_aliases = ngroup * (3 * nocc + 6) + 12 * nvir
        breakdown['proposal_caches'] += max(
            16 * occupied_entries + 72 * occupied_aliases,
            16 * virtual_entries + 72 * virtual_aliases) + 12 * 1024
        if with_grad:
            # At most two factor channels are live. Each lock is padded to
            # one cache line, with up to 16 stripes per occupied/atom block.
            breakdown['locks'] = 2 * 64 * nocc * ngroup * min(16, nvir)
            chunk_limit = 160000
            chunk = max(1, min(chunk_limit, 480 * 1024**2 // (64 * threads)))
            events = threads * max(64, chunk)
            # Two 24-byte record queues coexist. A final global sort uses
            # pointer pairs; concurrent local sorts use uint32 index pairs
            # and one 65536-bin histogram per worker. Bound the larger case.
            sort_bytes = max(16 * events + 4 * 65536,
                             8 * events + 4 * 65536 * threads)
            breakdown['reverse_updates'] = 48 * events + sort_bytes + 128 * threads + 1024
        # At least two production replicas are retained to estimate matrix SE.
        breakdown['accumulators'] = (max(2, threads) + 2 * threads + 8) * matrices
        # Native point_data caps retained selected Gram caches at 1 GiB per
        # resident point; headers/domains are included before enforcing cap.
        compact_bound = batch * (single_amplitude + 8 * nocc * nvir
                                   + 512 * nocc + 512)
        breakdown['compact_caches'] = points * min(1024**3, compact_bound)
        # Uncached compact blocks, packed DF columns, pair-domain matches,
        # aliases/proposals and reverse blocks can coexist with the cache.
        breakdown['scratch'] = (4 * single_amplitude
            + 8 * threads * (6 * naux * nvir + 12 * nvir * nvir)
            + 8 * (16 * nocc * nocc * nvir + 16 * ngroup * nocc * nvir)
            + 1024 * nocc * nocc)
    breakdown['total_bytes'] = sum(breakdown.values())
    return breakdown


def check_solver_memory(estimate, limit_mb, *, stage='matrix solver',
                        estimate_kind='workspace', unavoidable_bytes=0,
                        warned=None, observation=None):
    """Warn on matrix forecasts; reject only a proven impossible allocation.

    The optional configured MB budget is advisory. ``estimate_kind`` states
    whether residency is already included in the forecast. Keep the original
    two positional arguments and estimated-MB return value for callers.
    """
    from ._memory import assess_memory

    total_mb = int(estimate['total_bytes']) / 1e6
    if limit_mb is not None:
        limit_mb = _finite_control('memory_limit_mb', limit_mb)
    assess_memory(stage, total_mb, configured_mb=limit_mb,
                  estimate_kind=estimate_kind, unavoidable_bytes=unavoidable_bytes,
                  warned=warned, observation=observation)
    return total_mb
