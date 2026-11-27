"""Serial CC progress using the existing safe STC checkpoint format.

Save accumulated, completed host contributions. Native replay records and
pullback closures are consumed before this boundary and are never serialized.
"""
from dataclasses import asdict

from ._ccsd_restart_reference import (CCReferenceManager, resume_selections,
                                     save_reference, validate_reference)
from ._workflow import run_payload
from ._df_basis import basis_kwargs


def open_checkpoint(state, static, groups, *, checkpoint_dir, resume, options):
    if checkpoint_dir is None:
        if resume:
            raise ValueError('resume=True requires checkpoint_dir')
        return None
    from pyscfad.lno.ccsd import RCCSD
    # Workspace changes only execution, so compatible completed domains remain
    # reusable across omitted, default, and explicitly enlarged workspace caps.
    options = options.copy()
    # Omit disabled NAF for compatibility with historical uncompressed runs.
    if state.get('mp2_aux_svd_tol'):
        options['mp2_aux_svd_tol'] = state['mp2_aux_svd_tol']
    if options.get('triples_controls') is not None:
        options['triples_controls'] = options['triples_controls'].copy()
        options['triples_controls'].pop('workspace_gib', None)
    if resume:
        # Validate the regenerated settings/labels, then identify this run
        # using its original numerical frame and quadrature. Regenerated BP
        # intermediates and roundoff must not change the checkpoint digest.
        static = resume_selections(checkpoint_dir, static)
    # The deterministic SciPy port preserves these numerical identities;
    # replay implementation versions are deliberately process-local.
    from . import _native_w
    matrix_settings = dict(static.controls_identity[2])
    if static.lis_static.matrix_kind == 'V':
        from ._v_matrices import matrix_kernel_version
        native_version = matrix_kernel_version
    elif matrix_settings.get('mode', 'deterministic') == 'stochastic':
        from . import _stc_mp2
        native_version = _stc_mp2.matrix_kernel_version
    else:
        native_version = _native_w.checkpoint_matrix_version
    # The combined adapter changes replay APIs, while retaining each provider's
    # calculation. Keep their checkpoint identities across these known versions.
    if native_version in (
            'src_code_rdm_v2:427459d9b91fc70f:standalone-7-d-w-orbital',
            'src_code_rdm_v2:427459d9b91fc70f:standalone-8-caller-dynamic'):
        native_version = 'src_code_rdm_v2:427459d9b91fc70f:standalone-6-d-w'
    if (static.lis_static.matrix_kind == 'W'
            and native_version == 'src_code_rdm_v2:427459d9b91fc70f:standalone-6-d-w'):
        native_version = 'src_code_rdm_v2:427459d9b91fc70f:standalone-5-w-only'
    payload = dict(reference=run_payload(state['mf'], static.mp2_static.frozen,
                                         driver='stc-global-hf-cc-v1',
                                         **basis_kwargs(state.get('correlation_auxbasis'))),
                   canonical={name: getattr(state['mf'], name)
                              for name in ('mo_coeff', 'mo_energy', 'mo_occ', 'e_tot')},
                   native_version=native_version,
                   cc_convergence=(RCCSD.conv_tol, RCCSD.conv_tol_normt, RCCSD.max_cycle),
                   controls=static.controls_identity, options=options, groups=groups,
                   cc_selection_schema=static.schema_version,

                   boys_grouping=static.mp2_static.boys_grouping,
                   physical_groups=static.mp2_static.physical_groups)
    if options.get('cc_solver') == 'stc':
        from .stc_ccsd import external_implementation_identity
        payload['external_stc_implementation'] = external_implementation_identity()
    if static.baseline_kind != 'domain':
        payload['baseline_kind'] = static.baseline_kind
        baseline_settings = dict(static.controls_identity[1])
        if baseline_settings.get('mode', 'deterministic') == 'stochastic':
            from . import _stc_mp2
            payload['global_mp2_version'] = dict(
                provider='stc_mp2_selected', native_version=_stc_mp2.matrix_kernel_version,
                energy_identity='boys_target_laplace_mp2_v1')
        else:
            # Keep deterministic global-baseline checkpoints compatible.
            payload['global_mp2_version'] = _native_w.checkpoint_matrix_version
    triples = options.get('triples_controls')
    if triples is not None:
        if triples.get('mode', 'deterministic') == 'stochastic':
            from . import _stc_mp2 as triples_provider
        else:
            from . import _native_triples as triples_provider
        payload['native_triples_version'] = triples_provider.TRIPLES_VERSION
    manager = CCReferenceManager(checkpoint_dir, resume=resume, method='stc-ccsd',
                             scientific_payload=payload)
    try:
        if resume:
            manager.bind_static(static)
            if manager.load_static(expected_type=type(static)) is None:
                raise FileNotFoundError('CC checkpoint has no saved LIS selections')
            manager.reference_report = validate_reference(manager, state, static)
        else:
            manager.save_static(static)
            save_reference(manager, state)
    except Exception:
        manager.close()
        raise
    return manager


def restore_progress(manager, state, *, with_grad):
    if manager is None or not manager.resume:
        return 0, False, []
    templates = {name: state[name] for name in ('mf_bar', 'common_bar')} if with_grad else {}
    record = manager.load_record('cc-progress', templates=templates)
    if record is None:
        return 0, False, []
    from .ccsd import FragmentComponents
    for name in ('energy', 'variance', 'baseline_energy', 'weak_energy'):
        state[name] = record.scalars[name]
    if with_grad:
        state.update(record.trees)
    return (int(record.scalars['next_group']), bool(record.scalars['weak_done']),
            [FragmentComponents(**item) for item in record.scalars['fragments']])


def save_progress(manager, state, records, next_group, weak_done, *, with_grad):
    if manager is None:
        return
    scalars = {name: float(state.get(name, 0.))
               for name in ('energy', 'variance', 'baseline_energy', 'weak_energy')}
    scalars.update(next_group=int(next_group), weak_done=bool(weak_done),
                   fragments=[asdict(item) for item in records])
    trees = {name: state[name] for name in ('mf_bar', 'common_bar')} if with_grad else {}
    manager.save_record('cc-progress', scalars=scalars, trees=trees)
