"""Serial CC progress using the existing safe STC checkpoint format.

Save accumulated, completed host contributions. Native replay records and
pullback closures are consumed before this boundary and are never serialized.
"""
from dataclasses import asdict, fields, is_dataclass

import numpy as np

from ._restart import RestartManager, RestartMismatchError, scientific_digest
from ._workflow import run_payload
from ._df_basis import basis_kwargs


def _selection_labels(value):
    """Keep discrete choices, not regenerated reference frames/diagnostics.

    Scientific scalar controls are checked separately. Numerical references
    are restored from static.h5, whose full integrity digest is still checked.
    """
    if is_dataclass(value):
        return (type(value).__name__, {
            field.name: _selection_labels(getattr(value, field.name))
            for field in fields(value)
            # BP construction intermediates can change at tied populations
            # without changing the resulting pair graph, effective fragment
            # domains, ranks or worklist checked below.
            if field.name not in ('controls_identity', 'mp2_grid_diagnostics',
                                  'compact_bp_domain', 'primary_bp_domain',
                                  'tight_bp_domain', 'pao_bp_domain', 'primary_domain')})
    if isinstance(value, dict):
        return {key: _selection_labels(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_selection_labels(item) for item in value)
    if isinstance(value, (float, complex, np.floating, np.complexfloating)):
        if not np.isfinite(value):
            raise RestartMismatchError('nonfinite numerical CC selection')
        return ('numeric',)
    if hasattr(value, 'shape'):
        array = np.asarray(value)
        if array.dtype.kind in 'fc':
            if not np.isfinite(array).all():
                raise RestartMismatchError('nonfinite numerical CC selection')
            return ('numeric-array', array.shape, array.dtype.kind)
    return value


def _selection_settings(static):
    domain, lis = static.mp2_static, static.lis_static
    return dict(thresholds=domain.thresholds, lo_type=domain.lo_type,
                lo_kwargs=domain.lo_kwargs, selection_options=static.selection_options,
                thresh_occ=lis.thresh_occ, thresh_vir=lis.thresh_vir,
                internal_rank_threshold=lis.internal_rank_threshold,
                labels=_selection_labels(static))


def _scf_settings(mf):
    result = dict(direct_scf_tol=mf.direct_scf_tol)
    backend = getattr(mf, '_sgx_backend', None)
    if backend is not None:
        result['sgx'] = {name: getattr(backend, name) for name in (
            'auxbasis', 'grid_level', 'fit_ovlp', 'integral_backend', 'accuracy')}
    return result


def _assert_close(saved, current, name, *, atol, rtol=0.):
    saved, current = np.asarray(saved), np.asarray(current)
    if (saved.shape != current.shape or not np.isfinite(current).all()
            or not np.isfinite(saved).all()
            or not np.allclose(saved, current, atol=atol, rtol=rtol)):
        raise RestartMismatchError(
            f'CC restart has incompatible {name}; rebuild the domain checkpoint')


def _validate_reference(record, mf):
    # The live SCF gradient path already fixes MO phases. Do not accept arbitrary
    # rotations/sign flips: saved MO cotangents belong to the original gauge.
    for name, tolerance in (('e_tot', 1e-7), ('mo_energy', 1e-6),
                            ('mo_coeff', 1e-5), ('mo_occ', 0.)):
        _assert_close(record.scalars[name], getattr(mf, name), 'SCF ' + name,
                      atol=tolerance)


def resume_selection(mf, static, groups, *, checkpoint_dir, options,
                     correlation_auxbasis=None):
    """Validate before constructing common-data response in the saved frame."""
    manager = open_checkpoint(dict(mf=mf, correlation_auxbasis=correlation_auxbasis),
                              static, groups, checkpoint_dir=checkpoint_dir,
                              resume=True, options=options)
    try:
        return manager.load_static(expected_type=type(static))
    finally:
        manager.close()


def open_checkpoint(state, static, groups, *, checkpoint_dir, resume, options):
    if checkpoint_dir is None:
        if resume:
            raise ValueError('resume=True requires checkpoint_dir')
        return None
    from . import _stc_mp2
    from pyscfad.lno.ccsd import RCCSD
    payload = dict(reference=run_payload(state['mf'], static.mp2_static.frozen,
                                         driver='stc-global-hf-cc-v2',
                                         **basis_kwargs(state.get('correlation_auxbasis'))),
                   scf_settings=_scf_settings(state['mf']),
                   selection_settings=_selection_settings(static),
                   native_version=_stc_mp2.matrix_kernel_version,
                   cc_convergence=(RCCSD.conv_tol, RCCSD.conv_tol_normt, RCCSD.max_cycle),
                   controls=static.controls_identity[1:], options=options, groups=groups,
                   cc_selection_schema=static.schema_version,

                   boys_grouping=static.mp2_static.boys_grouping,
                   physical_groups=static.mp2_static.physical_groups)
    if options.get('cc_solver') == 'stc':
        from .stc_ccsd import external_implementation_identity
        payload['external_stc_implementation'] = external_implementation_identity()
    triples = options.get('triples_controls')
    if triples is not None:
        payload['native_triples_version'] = _stc_mp2.TRIPLES_VERSION
    try:
        manager = RestartManager(checkpoint_dir, resume=resume, method='stc-ccsd',
                                 scientific_payload=payload)
    except RestartMismatchError as error:
        raise RestartMismatchError(
            'CC checkpoint scientific settings differ, or the checkpoint predates '
            'numerical SCF compatibility records. Legacy checkpoints cannot be '
            'migrated safely; use a new checkpoint directory. ' + str(error)) from error
    try:
        if resume:
            saved = manager.load_static(expected_type=type(static))
            if saved is None:
                raise FileNotFoundError('CC checkpoint has no saved LIS selections')
            reference = manager.load_record('cc-scf-reference', missing_ok=False)
            _validate_reference(reference, state['mf'])
            if scientific_digest(_selection_settings(saved)) != scientific_digest(_selection_settings(static)):
                raise RestartMismatchError('CC checkpoint domain selections differ')
            for name, original, rebuilt in zip(('roots', 'weights'), saved.mp2_grid, static.mp2_grid):
                _assert_close(original, rebuilt, 'MP2 quadrature ' + name,
                              atol=1e-9, rtol=1e-6)
        else:
            manager.save_static(static)
            manager.save_record('cc-scf-reference', scalars={
                name: np.asarray(getattr(state['mf'], name))
                for name in ('mo_coeff', 'mo_energy', 'mo_occ', 'e_tot')})
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
