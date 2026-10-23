"""Serial CC progress using the existing safe STC checkpoint format.

Save accumulated, completed host contributions. Native replay records and
pullback closures are consumed before this boundary and are never serialized.
"""
from dataclasses import asdict

from ._restart import RestartManager
from ._workflow import run_payload


def open_checkpoint(state, static, groups, *, checkpoint_dir, resume, options):
    if checkpoint_dir is None:
        if resume:
            raise ValueError('resume=True requires checkpoint_dir')
        return None
    from . import _stc_mp2
    from pyscfad.lno.ccsd import RCCSD
    payload = dict(reference=run_payload(state['mf'], static.mp2_static.frozen,
                                         driver='stc-global-hf-cc-v1'),
                   canonical={name: getattr(state['mf'], name)
                              for name in ('mo_coeff', 'mo_energy', 'mo_occ', 'e_tot')},
                   native_version=_stc_mp2.matrix_kernel_version,
                   cc_convergence=(RCCSD.conv_tol, RCCSD.conv_tol_normt, RCCSD.max_cycle),
                   controls=static.controls_identity, options=options, groups=groups,
                   cc_selection_schema=static.schema_version,
                   boys_grouping=static.mp2_static.boys_grouping,
                   physical_groups=static.mp2_static.physical_groups)
    triples = options.get('triples_controls')
    if triples is not None and triples['mode'] == 'stochastic':
        payload['native_triples_version'] = _stc_mp2.TRIPLES_VERSION
    manager = RestartManager(checkpoint_dir, resume=resume, method='stc-ccsd',
                             scientific_payload=payload)
    try:
        if resume:
            manager.bind_static(static)
            if manager.load_static(expected_type=type(static)) is None:
                raise FileNotFoundError('CC checkpoint has no saved LIS selections')
        else:
            manager.save_static(static)
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
