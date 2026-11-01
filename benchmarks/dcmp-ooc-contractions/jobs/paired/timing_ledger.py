"""Benchmark-only, phase-local accounting of nested SCF/DF reverse work."""
from contextlib import contextmanager
from functools import wraps
import time

import jax

events = []
_phase = 'outside'
_generic_depth = 0


@contextmanager
def phase(name):
    global _phase
    previous, _phase = _phase, name
    try:
        yield
    finally:
        _phase = previous


def add(event):
    events.append(dict(event, phase=_phase))


def install():
    from pyscfad.df import _df_jk_factorized, _df_jk_opt, _cderi_vjp
    for module in (_df_jk_factorized, _df_jk_opt, _cderi_vjp):
        module._reverse_timing_hook = add

    def wrap_df(original):
        @wraps(original)
        def measured(*args, **kwargs):
            global _generic_depth
            first = len(events)
            _generic_depth += 1
            start = time.perf_counter()
            try:
                result = original(*args, **kwargs)
                jax.block_until_ready(result)
                elapsed = time.perf_counter() - start
                nested = sum(e['df_seconds'] for e in events[first:]
                             if e['kind'] in ('factorized_df', 'generic_df'))
                add(dict(kind='generic_df', function=original.__name__,
                         df_seconds=elapsed - nested, inclusive_seconds=elapsed,
                         depth=_generic_depth))
                return result
            finally:
                _generic_depth -= 1
        return measured

    for name in ('_cderi_mol_aux_vjp_from_block_fn', '_cderi_mol_aux_vjp'):
        setattr(_df_jk_opt, name, wrap_df(getattr(_df_jk_opt, name)))

    original_precond = _df_jk_opt.gen_coulomb_precond

    @wraps(original_precond)
    def measured_precond(*args, **kwargs):
        start = time.perf_counter()
        apply = original_precond(*args, **kwargs)
        add(dict(kind='preconditioner_setup', seconds=time.perf_counter() - start))

        @wraps(apply)
        def measured_apply(*args, **kwargs):
            start = time.perf_counter()
            result = apply(*args, **kwargs)
            jax.block_until_ready(result)
            add(dict(kind='preconditioner_apply', seconds=time.perf_counter() - start))
            return result
        return measured_apply

    _df_jk_opt.gen_coulomb_precond = measured_precond


def summarize(phase_seconds):
    scf = [e for e in events if e['phase'] == 'scf_pullback']
    by_kind = {}
    for e in scf:
        by_kind.setdefault(e['kind'], []).append(e)

    def total(kind, key='seconds'):
        return sum(e[key] for e in by_kind.get(kind, []))

    fused = total('factorized_df', 'df_seconds') + total('generic_df', 'df_seconds')
    raw = phase_seconds['scf_pullback']
    residual = raw - fused
    density = total('coordinate_df', 'density_seconds')
    setup = total('factorized_df', 'density_setup_seconds')
    response = total('factorized_df', 'factor_response_prefix_seconds')
    matvec = total('implicit_matvec')
    precond_setup = total('preconditioner_setup')
    precond_apply = total('preconditioner_apply')
    counts = {k: len(v) for k, v in by_kind.items()}
    # This fixed dCMP driver has two orbital-factor VJPs and one generic
    # density VJP that takes the factorized geometry path. A change of path
    # requires reviewing the accounting rather than silently losing a timer.
    checks = {
        'three_factorized_geometry_calls': counts.get('factorized_df', 0) == 3,
        'three_coordinate_helpers': counts.get('coordinate_df', 0) == 3,
        'no_generic_fallback': counts.get('generic_df', 0) == 0,
        'implicit_response_observed': counts.get('implicit_matvec', 0) > 0,
        'one_preconditioner_setup': counts.get('preconditioner_setup', 0) == 1,
        'preconditioner_applied': counts.get('preconditioner_apply', 0) > 0,
        'positive_exclusive_times': 0 < density + setup < fused < raw,
        'df_substages_sum': all(abs(e['df_seconds'] - sum(e[k] for k in (
            'metric_transform_seconds', 'density_setup_seconds',
            'coordinate_seconds', 'metric_reverse_seconds'))) < 1e-6
            for e in by_kind.get('factorized_df', [])),
        'exclusive_bins_fit': response + matvec + precond_setup + precond_apply <= residual,
    }
    return {
        'definition': 'DF-construction reverse measured inside SCF; MP2 events excluded. '
                      'Conservative split assigns fused AO-adjoint assembly/packing to SCF.',
        'scf_forward_without_df_seconds': phase_seconds['scf_forward'],
        'scf_reverse_including_embedded_df_seconds': raw,
        'fused_df_construction_reverse_seconds': fused,
        'scf_reverse_excluding_fused_df_seconds': residual,
        'mixed_ao_adjoint_assembly_seconds': density + setup,
        'scf_reverse_without_df_conservative_seconds': residual + density + setup,
        'df_reverse_excluding_ao_adjoint_assembly_seconds': fused - density - setup,
        'scf_exclusive_breakdown_seconds': {
            'initial_factor_response_and_compressed_adjoint': response,
            'implicit_response_df_jk': matvec,
            'coulomb_preconditioner_setup': precond_setup,
            'coulomb_preconditioner_apply': precond_apply,
            'other_scf': residual - response - matvec - precond_setup - precond_apply,
        },
        'fused_df_breakdown_seconds': {
            'metric_and_triangular_transforms': total('factorized_df', 'metric_transform_seconds'),
            'density_setup': setup,
            'coordinate_total': total('factorized_df', 'coordinate_seconds'),
            'coordinate_density_assembly_and_packing': density,
            'coordinate_native_contraction': total('coordinate_df', 'native_seconds'),
            'coordinate_fallback_contraction': total('coordinate_df', 'fallback_seconds'),
            'metric_reverse': total('factorized_df', 'metric_reverse_seconds'),
        },
        'counts': counts,
        'checks': checks,
        'passed': all(checks.values()),
    }
