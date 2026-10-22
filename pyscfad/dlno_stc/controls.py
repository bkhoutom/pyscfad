"""Shared defaults for high-level STC-MP2 drivers."""

from numbers import Real

import numpy

DEFAULT_ENERGY_TOLERANCE = 3e-4  # Hartree (0.3 mHa)


def default_laplace_grid():
    """Return independent arrays for the original six-point fixed grid."""
    roots = numpy.array([0.003431, 0.023534, 0.088984, 0.275603, 0.757121, 1.906218])
    weights = numpy.array([0.009348, 0.035196, 0.107559, 0.293035, 0.729094, 1.690608])
    return roots * 2.6, weights * 2.6


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
    'resident_points': 3,
}
_MATRIX_D_SAMPLING = {
    'energy_tolerance': DEFAULT_ENERGY_TOLERANCE,
    'system_workload_cutoff': 0.0065,
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


def normalize_triples_controls(controls, *, ccsd_t):
    """Validate triples independently of baseline and D/W selection settings.

    Omitted controls preserve the existing deterministic correction. The
    stochastic interface uses fixed counts unless a positive total energy
    tolerance is supplied; then nsamples is the adaptive sampler pilot count.
    """
    if not ccsd_t:
        if controls is not None:
            raise ValueError('triples_controls requires ccsd_t=True')
        return None
    if controls is None:
        return {'mode': 'deterministic'}
    if not isinstance(controls, dict):
        raise TypeError('triples_controls must be a dictionary')
    settings = controls.copy()
    unknown = set(settings) - {'mode', 'nsamples', 'global_seed', 'energy_tolerance'}
    if unknown:
        raise ValueError('unsupported triples controls: '
                         + ', '.join(sorted(map(str, unknown))))
    mode = settings.setdefault('mode', 'deterministic')
    if mode not in ('deterministic', 'stochastic'):
        raise ValueError('triples mode must be deterministic or stochastic')
    if mode == 'deterministic':
        if set(settings) - {'mode'}:
            raise ValueError('deterministic triples do not accept sampling controls')
        return settings
    if 'nsamples' not in settings:
        raise ValueError('stochastic triples require integer nsamples >= 2')
    settings['nsamples'] = _integer_control('nsamples', settings['nsamples'], minimum=2)
    settings['global_seed'] = _integer_control('global_seed', settings.get('global_seed', 71), minimum=0)
    if settings['global_seed'] >= 2**64:
        raise ValueError('global_seed must fit in uint64')
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


def normalize_matrix_controls(controls, matrix_kind):
    """Copy and validate the explicit native D/W interface controls.

    Sampling counts are chosen by the native precision-targeted allocation;
    energy-only fixed-count options do not belong to this matrix interface.
    The default grid and caller-supplied grid arrays are independently owned.
    """
    if matrix_kind not in ('D', 'W'):
        raise ValueError('matrix_kind must be D or W')
    if not isinstance(controls, dict):
        raise TypeError('controls must be a dictionary')
    settings = controls.copy()
    allowed = _MATRIX_COMMON | _MATRIX_SAMPLING.keys() | _MATRIX_D_SAMPLING.keys()
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError('unsupported matrix controls: ' + ', '.join(sorted(map(str, unknown))))
    mode = settings.setdefault('mode', 'deterministic')
    if mode not in ('deterministic', 'stochastic'):
        raise ValueError('mode must be deterministic or stochastic')
    if matrix_kind == 'W':
        invalid = settings.keys() & _MATRIX_D_SAMPLING.keys()
        if 'resident_points' in settings:
            invalid = invalid | {'resident_points'}
        if invalid:
            raise ValueError('W targets matrix precision; unsupported controls: '
                             + ', '.join(sorted(invalid)))
    if mode == 'deterministic':
        invalid = settings.keys() & (_MATRIX_SAMPLING.keys() | _MATRIX_D_SAMPLING.keys())
        if invalid:
            raise ValueError('deterministic matrices do not accept sampling controls: '
                             + ', '.join(sorted(invalid)))
    else:
        for name, value in _MATRIX_SAMPLING.items():
            if matrix_kind == 'W' and name == 'resident_points':
                continue
            settings.setdefault(name, value)
        if matrix_kind == 'D':
            for name, value in _MATRIX_D_SAMPLING.items():
                settings.setdefault(name, value)
    if 'laplace_roots' not in settings and 'laplace_weights' not in settings:
        settings['laplace_roots'], settings['laplace_weights'] = default_laplace_grid()
    for name in ('laplace_roots', 'laplace_weights'):
        if name not in settings:
            raise ValueError(f'controls must supply {name}')
        raw = numpy.asarray(settings[name])
        if numpy.iscomplexobj(raw):
            raise ValueError(f'{name} must be a real finite vector')
        try:
            values = numpy.array(raw, dtype=numpy.float64, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{name} must be a real finite vector') from exc
        if values.ndim != 1 or not values.size or not numpy.isfinite(values).all():
            raise ValueError(f'{name} must be a nonempty finite vector')
        settings[name] = values
    roots, weights = settings['laplace_roots'], settings['laplace_weights']
    if roots.shape != weights.shape or numpy.any(roots < 0):
        raise ValueError('Laplace roots/weights must match and roots must be nonnegative')
    if mode == 'stochastic' and numpy.any(weights <= 0):
        raise ValueError('stochastic matrix Laplace weights must be strictly positive')
    for name, default in (('global_seed', 0), ('threads', 1),
                          ('target_batch_size', 1), ('auxiliary_group_size', 32)):
        settings[name] = _integer_control(name, settings.get(name, default),
                                         minimum=0 if name == 'global_seed' else 1)
    if settings.get('memory_limit_mb') is not None:
        settings['memory_limit_mb'] = _finite_control('memory_limit_mb', settings['memory_limit_mb'])
    if mode == 'stochastic':
        for name in ('pair_pilot_samples', 'min_samples'):
            settings[name] = _integer_control(name, settings[name], minimum=2)
        if 'resident_points' in settings:
            settings['resident_points'] = _integer_control('resident_points', settings['resident_points'])
            if settings['resident_points'] not in (2, 3):
                raise ValueError('resident_points must be 2 or 3')
        for name in ('matrix_tolerance', 'energy_tolerance', 'system_workload_cutoff'):
            if name in settings:
                settings[name] = _finite_control(name, settings[name])
        for name in ('domain_tau', 'virtual_domain_tau'):
            if name in settings:
                settings[name] = _finite_control(name, settings[name], positive=False)
    return settings


def estimate_matrix_memory(naux, nocc, nvir, ntarget, controls, *, matrix_kind,
                           with_grad=True, auxiliary_group_count=None):
    """Conservative bytes newly needed by one staged matrix call and reverse.

    Counts include host/native layout copies, native factors, matrix output/SE
    arrays, per-worker accumulators and worst-case compact-domain caches. They
    exclude existing inputs, SCF/preparation tapes, allocator/BLAS overhead and
    the rest of the process. Add current process residency when enforcing a
    process budget. No allocation proportional to all targets is assumed for
    deterministic calls: their native target batch is bounded explicitly.
    """
    settings = normalize_matrix_controls(controls, matrix_kind)
    dims = [_integer_control(name, value, minimum=0 if name in ('naux', 'nvir') else 1)
            for name, value in
            (('naux', naux), ('nocc', nocc), ('nvir', nvir), ('ntarget', ntarget))]
    naux, nocc, nvir, ntarget = dims
    if ntarget > nocc:
        raise ValueError('ntarget must not exceed nocc')
    factor = 8 * naux * nocc * nvir
    matrices = 8 * (nocc * nocc + nvir * nvir)
    threads = settings['threads']
    batch = min(ntarget, settings['target_batch_size'])
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
    # cubes. W's seed additionally needs bare and adjoint Gram cubes.
    if settings['mode'] == 'deterministic':
        extra = (4 if matrix_kind == 'W' else 2) if with_grad else 1
        breakdown['amplitude_buffers'] = ((2 if with_grad else 1) * batch + extra) * single_amplitude
        breakdown['scratch'] = 8 * threads * (6 * nvir * nvir + 2 * naux * nvir)
    else:
        points = min(len(settings['laplace_roots']), settings['resident_points']) if matrix_kind == 'D' else 2
        ngroup = ((naux + settings['auxiliary_group_size'] - 1) // settings['auxiliary_group_size']
                  if auxiliary_group_count is None else
                  _integer_control('auxiliary_group_count', auxiliary_group_count))
        # Each point has propagated values/bars, scores and alias tables.
        breakdown['factor_buffers'] += points * ((2 if with_grad else 1) * factor
            + 8 * nocc * nvir * (8 * ngroup + 8))
        # Every resident point owns atom_weight. W's fixed B and dressed C
        # also keep atom_l1 for the occupied mixture and virtual reform.
        breakdown['atom_weights'] = 8 * points * ngroup * nocc * nvir * (
            2 if matrix_kind == 'W' else 1)
        if matrix_kind == 'W':
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
            chunk_limit = 160000 if matrix_kind == 'W' else 260000
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


def check_solver_memory(estimate, limit_mb):
    """Reject the estimated new matrix working set above an optional MB cap."""
    total_mb = int(estimate['total_bytes']) / 1e6
    if limit_mb is not None:
        limit_mb = _finite_control('memory_limit_mb', limit_mb)
        if total_mb > limit_mb:
            raise MemoryError(f'estimated matrix solver memory {total_mb:.1f} MB '
                              f'exceeds configured {limit_mb:.1f} MB')
    return total_mb
