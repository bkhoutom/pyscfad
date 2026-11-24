"""Numerical reference checks for serial CC checkpoint reuse.

Scientific inputs and discrete selections remain exact. Reference arrays use
explicit tolerances, never rounded hashes. The original reference is retained
across resumes so that repeated restarts cannot accumulate tolerance drift.
"""
from dataclasses import fields, is_dataclass
from collections.abc import Mapping
import json
import os
from pathlib import Path
import uuid
import warnings

import jax
import numpy as np

from . import _restart as storage


ARRAY_ATOL = 1e-8
ARRAY_RTOL = 1e-8
ENERGY_ATOL = 1e-8  # absolute Eh, not relative to the extensive total energy
_NUMERICAL_SELECTION_FIELDS = {
    'DomainSelections': {'target_reference_coeff', 'pair_energy', 'weak_pair_energy',
                         'mp2_grid', 'mp2_grid_diagnostics'},
    'CCSDSelections': {'mp2_fragment_energies', 'mp2_grid', 'mp2_grid_diagnostics'},
    'LISSelections': {'triples_grid', 'smooth_frames'},
}
_BP_CONSTRUCTION_FIELDS = {
    'compact_bp_domain', 'primary_bp_domain', 'tight_bp_domain',
    'pao_bp_domain', 'primary_domain',
}


def _mapping(node):
    if node.get('kind') != 'dict':
        raise storage.RestartCorruptionError('checkpoint identity is not a dictionary')
    return {key['value']: value for key, value in node['items']}


def _manifest(path):
    try:
        manifest = json.loads((Path(path) / 'run.json').read_text())
    except (OSError, ValueError) as error:
        raise storage.RestartManifestError('missing or malformed CC run.json') from error
    expected = dict(format=storage._FORMAT_NAME, format_version=storage._FORMAT_VERSION,
                    equation_id=storage._EQUATION_ID, algorithm_abi=storage._ALGORITHM_ABI,
                    method='stc-ccsd')
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise storage.RestartMismatchError(f'CC checkpoint {name} differs')
    # Authenticate the summary against the identity used by every stage file.
    identity = storage._encode_value(dict(abi=storage._ALGORITHM_ABI,
        equations=storage._EQUATION_ID, method='stc-ccsd', payload=None))
    for key_value in identity['items']:
        if key_value[0]['value'] == 'payload':
            key_value[1] = manifest['base_summary']
    digest = storage._sha256_bytes(storage._canonical_json(identity).encode('utf8'))
    if digest != manifest.get('base_digest'):
        raise storage.RestartCorruptionError('CC checkpoint identity summary failed its digest')
    return manifest


def _saved_static(path, manifest):
    handle, metadata, reader = storage._read_hdf5(Path(path) / 'static.h5')
    try:
        for name in ('format', 'format_version', 'equation_id', 'method',
                     'base_digest', 'static_digest'):
            if metadata.get(name) != manifest.get(name):
                raise storage.RestartMismatchError(f'CC static checkpoint {name} differs')
        if metadata.get('kind') != 'static':
            raise storage.RestartCorruptionError('CC static checkpoint has the wrong kind')
        selected = storage._decode_value(metadata['value'], reader)
        if storage.scientific_digest(selected) != manifest.get('static_digest'):
            raise storage.RestartCorruptionError('CC static selections failed their digest')
        return selected
    finally:
        handle.close()


def _close(current, saved, name, *, atol=ARRAY_ATOL, rtol=ARRAY_RTOL):
    current, saved = np.asarray(current), np.asarray(saved)
    if (current.shape != saved.shape or current.dtype != saved.dtype
            or not np.isfinite(current).all() or not np.isfinite(saved).all()):
        raise storage.RestartMismatchError(f'restart reference {name}: shape/dtype/nonfinite mismatch')
    if not np.allclose(current, saved, atol=atol, rtol=rtol):
        error = float(np.max(np.abs(current - saved))) if current.size else 0.
        raise storage.RestartMismatchError(
            f'restart reference {name} differs (max abs {error:.3g}, atol={atol}, rtol={rtol})')
    return float(np.max(np.abs(current - saved))) if current.size else 0.


def _selection_check(current, saved, path='selection', numerical=False):
    """Only derived floating-point selection data may vary numerically."""
    if type(current) is not type(saved):
        raise storage.RestartMismatchError(f'{path}: selection type differs')
    if is_dataclass(saved):
        for field in fields(saved):
            if type(saved).__name__ == 'DomainSelections' and field.name in _BP_CONSTRUCTION_FIELDS:
                # Tied BP populations can change construction intermediates.
                # Effective fragment domains, ranks and the pair graph below
                # must still match; resumed work uses the original selections.
                continue
            if field.name == 'controls_identity':
                # Its first entry is the derived quadrature; seeds, modes and
                # sampling controls in the remaining entries stay exact.
                new, old = getattr(current, field.name), getattr(saved, field.name)
                _selection_check(new[0], old[0], f'{path}.controls_identity[0]', True)
                _selection_check(new[1:], old[1:], f'{path}.controls_identity[1:]')
                continue
            _selection_check(getattr(current, field.name), getattr(saved, field.name),
                             f'{path}.{field.name}',
                             numerical or field.name in
                             _NUMERICAL_SELECTION_FIELDS.get(type(saved).__name__, set()))
    elif isinstance(saved, dict):
        if current.keys() != saved.keys():
            raise storage.RestartMismatchError(f'{path}: selection keys differ')
        for key in saved:
            _selection_check(current[key], saved[key], f'{path}.{key}', numerical)
    elif isinstance(saved, (tuple, list)):
        if len(current) != len(saved):
            raise storage.RestartMismatchError(f'{path}: selection length differs')
        for index, (new, old) in enumerate(zip(current, saved)):
            _selection_check(new, old, f'{path}[{index}]', numerical)
    elif numerical and np.asarray(saved).dtype.kind in 'fc':
        _close(current, saved, path)
    elif storage.scientific_digest(current) != storage.scientific_digest(saved):
        raise storage.RestartMismatchError(f'{path}: fixed selection/settings differ')


def resume_selections(checkpoint_dir, current):
    """Use the saved labels, Boys anchors and grids before rebuilding a VJP."""
    saved = _saved_static(checkpoint_dir, _manifest(checkpoint_dir))
    _selection_check(current, saved)
    return saved


def prepare_resume(checkpoint_dir, resume):
    """Restart interrupted initialization without discarding committed work.

Before the first progress record, a crash can leave only the manifest/static
file. Such a run has no accumulated contributions to restore. Archive its
incomplete metadata under the writer lease and initialize a fresh reference.
An existing progress file, including a corrupt one, is never moved or reset.
"""
    if not resume:
        return False
    if checkpoint_dir is None:
        raise ValueError('resume=True requires checkpoint_dir')
    path = Path(checkpoint_dir).expanduser().resolve()
    if not path.exists():
        return False
    lease = storage.RestartManager(None, method='stc-ccsd', scientific_payload={})
    lease.path = path
    lease._acquire_writer_lease()
    try:
        progress = path / 'records/cc-progress/state.h5'
        if progress.exists():
            return True
        visible = {child.name for child in path.iterdir() if not child.name.startswith('.')}
        if not visible:
            return False
        if not (path / 'run.json').is_file():
            raise storage.RestartManifestError('nonempty checkpoint has no run.json')
        _manifest(path)
        if (path / 'static.h5').is_file() and (path / 'records/cc-reference/state.h5').is_file():
            return True
        # Do not reinterpret another method's data or unknown stage records as
        # an interrupted CC initialization.
        if visible - {'run.json', 'static.h5', 'records'}:
            raise storage.RestartManifestError('unexpected files in incomplete CC checkpoint')
        records = path / 'records'
        if records.exists() and any(child.name != 'cc-reference' for child in records.iterdir()
                                    if not child.name.startswith('.')):
            raise storage.RestartManifestError('unexpected records in incomplete CC checkpoint')
        archive = path / '.initialization-history' / uuid.uuid4().hex
        storage._durable_mkdir(archive)
        # Move the manifest last. An interruption of recovery itself therefore
        # leaves either another recognizable partial initialization or an empty
        # directory (apart from hidden history).
        for name in ('records', 'static.h5', 'run.json'):
            source = path / name
            if source.exists():
                os.replace(source, archive / name)
                storage._fsync_directory(archive)
                storage._fsync_directory(path)
        warnings.warn('Restarting incomplete CC checkpoint initialization; no '
                      'fragment progress had been committed. Metadata archived in '
                      f'{archive}', RuntimeWarning, stacklevel=2)
        return False
    finally:
        lease.close()


class CCReferenceManager(storage.RestartManager):
    """Permit a canonical-only hash mismatch pending numerical validation.

Only open_checkpoint constructs this manager and it validates the numerical
reference before returning it or restoring any accumulated contribution.
"""
    def _open_existing_manifest(self):
        manifest = _manifest(self.path)
        saved, current = _mapping(manifest['base_summary']), _mapping(self._base_summary)
        old_canonical, new_canonical = saved.pop('canonical'), current.pop('canonical')
        if saved != current:
            changed = sorted(k for k in saved.keys() | current.keys()
                             if saved.get(k) != current.get(k))
            raise storage.RestartMismatchError(f'CC checkpoint scientific settings differ: {changed}')
        old_canonical, new_canonical = _mapping(old_canonical), _mapping(new_canonical)
        if old_canonical['mo_occ'] != new_canonical['mo_occ']:
            raise storage.RestartMismatchError('CC checkpoint occupations differ')
        for name in ('mo_coeff', 'mo_energy', 'e_tot'):
            old, new = old_canonical[name], new_canonical[name]
            if old.get('kind') == 'array' and any(old.get(k) != new.get(k)
                                                for k in ('kind', 'shape', 'dtype')):
                raise storage.RestartMismatchError(f'CC checkpoint canonical {name} shape/dtype differs')
        self.canonical_changed = old_canonical != new_canonical
        self.base_digest = manifest['base_digest']
        self._base_summary = manifest['base_summary']
        super()._open_existing_manifest()


def _scf_settings(mf):
    backend = getattr(mf, '_sgx_backend', None)
    return dict(scf={name: getattr(mf, name, None)
                     for name in ('conv_tol', 'conv_tol_grad', 'direct_scf', 'direct_scf_tol')},
                sgx=None if backend is None else {
                    name: getattr(backend, name) for name in
                    ('auxbasis', 'grid_level', 'fit_ovlp', 'integral_backend', 'accuracy')})


def save_reference(manager, state):
    mf = state['mf']
    if not bool(mf.converged):
        raise storage.RestartMismatchError('cannot save unconverged SCF')
    canonical = {name: getattr(mf, name) for name in ('mo_coeff', 'mo_energy', 'mo_occ', 'e_tot')}
    scalars = dict(canonical=canonical, scf_settings=_scf_settings(mf))
    if not hasattr(state.get('common'), 'fock'):
        # Normal domain checkpoints already contain common.fock. Reference-
        # only callers must also retain it for zero-Fock SCF reconstruction.
        scalars['fock'] = np.asarray(mf.get_fock())
    manager.save_record('cc-reference', scalars=scalars,
        trees={'common': state['common']} if 'common' in state else {})


def _molecular_inputs(mol):
    return dict(coords_bohr=np.asarray(mol.atom_coords()),
                atom_symbols=tuple(mol.atom_symbol(i) for i in range(mol.natm)),
                charge=int(mol.charge), spin=int(mol.spin),
                basis=getattr(mol, '_basis', None), ecp=getattr(mol, '_ecp', None),
                cart=bool(getattr(mol, 'cart', False)))


def _validate_molecular_inputs(manifest, mol):
    saved = _mapping(_mapping(manifest['base_summary'])['reference'])
    for name, value in _molecular_inputs(mol).items():
        if storage._encode_value(value) != saved.get(name):
            raise storage.RestartMismatchError(f'checkpoint molecular input {name} differs')
    return saved


def load_selections(checkpoint_dir, mol, *, domain_options=None, selection_options=None):
    """Load authenticated choices and check requested settings without selection."""
    from ._selection import (BoysDomainThresholds, _normalize_boys_grouping,
                             _resolve_boys_options, _validate_boys_target_groups)

    for name, options in (('domain_options', domain_options), ('selection_options', selection_options)):
        if options is not None and not isinstance(options, Mapping):
            raise TypeError(f'{name} must be a mapping')
    manifest = _manifest(checkpoint_dir)
    _validate_molecular_inputs(manifest, mol)
    saved = _saved_static(checkpoint_dir, manifest)
    from .ccsd import CCSDSelections, CC_SELECTIONS_SCHEMA_VERSION
    if not isinstance(saved, CCSDSelections) or saved.schema_version != CC_SELECTIONS_SCHEMA_VERSION:
        raise storage.RestartMismatchError('incompatible saved CC selection schema')
    domain = saved.mp2_static
    domain_values = dict(frozen=domain.frozen, thresholds=domain.thresholds,
        lo_kwargs=domain.lo_kwargs or {}, boys_grouping=domain.boys_grouping,
        pair_energy_model=domain.pair_energy_model, frag_lolist=domain.frag_lolist)
    for name, value in (domain_options or {}).items():
        if name == 'full_domain':
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError('full_domain must be a bool')
            # Old static files record the effective domains, not the flag that
            # produced them. A request for full domains must actually be full.
            if value and (not np.all(domain.strong_mask) or any(
                    not np.array_equal(fragment.extended_ao_indices, np.arange(mol.nao))
                    or not np.array_equal(fragment.pao_center_atoms, np.arange(mol.natm))
                    for fragment in domain.fragments)):
                raise storage.RestartMismatchError('checkpoint full_domain setting differs')
            continue
        if name not in domain_values:
            raise ValueError(f'unsupported restart domain option: {name}')
        if name == 'thresholds':
            value = BoysDomainThresholds() if value is None else value
        elif name == 'lo_kwargs':
            value = _resolve_boys_options(value)
        elif name == 'boys_grouping':
            value = _normalize_boys_grouping(value)
        elif name == 'pair_energy_model':
            value = str(value).lower().replace('_', '-')
        elif name == 'frag_lolist':
            value = _validate_boys_target_groups(value, len(domain.active_occ_indices))
        if storage.scientific_digest(value) != storage.scientific_digest(domain_values[name]):
            raise storage.RestartMismatchError(f'checkpoint domain setting {name} differs')
    lis = saved.lis_static
    selection_values = dict(thresh_occ=lis.thresh_occ, thresh_vir=lis.thresh_vir,
        sequential_virtual=lis.sequential_virtual,
        scale_lis_by_target_count=lis.scale_lis_by_target_count,
        internal_rank_threshold=lis.internal_rank_threshold,
        triples_laplace_points=(saved.selection_options or {}).get('triples_laplace_points'))
    for name, value in (selection_options or {}).items():
        if name not in selection_values:
            raise ValueError(f'unsupported restart LIS option: {name}')
        if storage.scientific_digest(value) != storage.scientific_digest(selection_values[name]):
            raise storage.RestartMismatchError(f'checkpoint LIS setting {name} differs')
    return saved


def load_scf_reference(checkpoint_dir, mol):
    """Load authenticated numerical values without running either SCF solve."""
    if not (Path(checkpoint_dir) / 'run.json').exists():
        return _load_early_scf_reference(checkpoint_dir, mol)
    manifest = _manifest(checkpoint_dir)
    summary = _mapping(manifest['base_summary'])
    saved_inputs = _validate_molecular_inputs(manifest, mol)
    path = Path(checkpoint_dir) / 'records/cc-reference/state.h5'
    if not path.is_file():
        raise storage.RestartMismatchError(
            'checkpoint has no numerical SCF reference; cannot initialize SCF '
            'from saved orbitals. Legacy canonical hashes cannot recover a density.')
    handle, payload, reader = storage._read_hdf5(path)
    try:
        for name in ('format', 'format_version', 'equation_id', 'method',
                     'base_digest', 'static_digest'):
            if payload.get(name) != manifest.get(name):
                raise storage.RestartMismatchError(f'CC reference checkpoint {name} differs')
        if (payload.get('kind') != 'record' or payload.get('stage') != 'cc-reference'
                or storage._decode_value(payload.get('key'), reader) is not None):
            raise storage.RestartCorruptionError('invalid numerical SCF reference record')
        scalars = storage._decode_value(payload['scalars'], reader)
        canonical = scalars['canonical']
        if storage._encode_value(canonical) != summary['canonical']:
            raise storage.RestartCorruptionError(
                'numerical SCF reference does not match the original identity')
        entries = payload.get('trees', {}).get('common', {}).get('entries', ())
        def common_value(name):
            matches = [entry['leaf'] for entry in entries
                       if entry.get('path') == [['attr', name]]]
            if len(matches) > 1:
                raise storage.RestartCorruptionError(f'duplicate saved common.{name}')
            if not matches:
                return None
            leaf = matches[0]
            if leaf.get('kind') == 'array':
                return reader.get(leaf)
            if leaf.get('kind') == 'zero':
                return np.zeros(leaf['shape'], dtype=np.dtype(leaf['dtype']))
            raise storage.RestartCorruptionError(f'invalid saved common.{name}')
        boys_coeff = common_value('iao_coeff')
        fock = common_value('fock')
        if fock is None:
            fock = scalars.get('fock')
    finally:
        handle.close()
    _validate_canonical(canonical, mol)
    if fock is not None:
        _validate_fock(fock, mol)
    return dict(canonical=canonical, scf_settings=scalars['scf_settings'],
                auxbasis=storage._decode_value(saved_inputs['auxbasis'], None),
                scf_class=storage._decode_value(saved_inputs['scf_class'], None),
                boys_coeff=boys_coeff, fock=fock)


def load_boys_reference(checkpoint_dir, mf, static, *, with_identity=False):
    """Use the saved common-frame root, which may refine the initial anchors."""
    if not (Path(checkpoint_dir) / 'records/cc-reference/state.h5').is_file():
        # Legacy reference validation remains in open_checkpoint.
        return (None, False) if with_identity else None
    reference = load_scf_reference(checkpoint_dir, mf.mol)
    _validate_scf_backend(mf, reference)
    if not bool(mf.converged):
        raise storage.RestartMismatchError('checkpoint restart requires converged SCF')
    for name, value in reference['canonical'].items():
        _close(getattr(mf, name), value, name,
               atol=0. if name == 'mo_occ' else ENERGY_ATOL if name in ('e_tot', 'mo_energy') else ARRAY_ATOL,
               rtol=0. if name in ('e_tot', 'mo_energy', 'mo_occ') else ARRAY_RTOL)
    coeff = reference['boys_coeff']
    if coeff is not None and (coeff.shape != np.shape(static.target_reference_coeff)
                              or not np.isfinite(coeff).all()):
        raise storage.RestartCorruptionError('saved Boys frame shape or values differ')
    if with_identity:
        exact = storage.scientific_digest(reference['canonical']) == storage.scientific_digest({
            name: getattr(mf, name) for name in reference['canonical']})
        return coeff, exact
    return coeff


def _validate_canonical(canonical, mol):
    coeff, occupation = np.asarray(canonical['mo_coeff']), np.asarray(canonical['mo_occ'])
    energies, total = np.asarray(canonical['mo_energy']), np.asarray(canonical['e_tot'])
    if (coeff.ndim != 2 or coeff.shape[0] != mol.nao
            or occupation.shape != (coeff.shape[1],) or energies.shape != occupation.shape
            or total.shape != () or mol.spin != 0
            or not np.all((occupation == 0) | (occupation == 2))
            or np.sum(occupation) != mol.nelectron
            or any(not np.isfinite(x).all() for x in (coeff, occupation, energies, total))):
        raise storage.RestartCorruptionError('invalid saved RHF orbital shapes, occupations or values')


def scf_initial_guess(checkpoint_dir, mol):
    """Compatibility helper; the benchmark now restores response directly."""
    canonical = load_scf_reference(checkpoint_dir, mol)['canonical']
    coeff, occupation = canonical['mo_coeff'], canonical['mo_occ']
    return (coeff * occupation) @ coeff.conj().T


def _validate_scf_backend(mf, reference):
    if (storage.scientific_digest(_scf_settings(mf)) !=
            storage.scientific_digest(reference['scf_settings'])):
        raise storage.RestartMismatchError('checkpoint SCF/SGX settings differ')
    if (f'{type(mf).__module__}.{type(mf).__qualname__}' != reference['scf_class']
            or storage.scientific_digest(getattr(getattr(mf, 'with_df', None), 'auxbasis', None))
            != storage.scientific_digest(reference['auxbasis'])):
        raise storage.RestartMismatchError('checkpoint SCF class/auxiliary basis differs')


def restore_scf_reference(mf, reference):
    """Validate backend settings and attach response to the saved RHF root."""
    from pyscfad.scf._restart import restore_scf
    _validate_scf_backend(mf, reference)
    fock = reference.get('fock')
    if fock is None:
        raise storage.RestartMismatchError(
            'checkpoint has no saved AO Fock; cannot restore without rebuilding Fock')
    _validate_fock(fock, mf.mol)
    return restore_scf(mf, reference['canonical'], fock=fock)


def _validate_fock(fock, mol):
    value = np.asarray(fock)
    if (value.shape != (mol.nao, mol.nao) or np.iscomplexobj(value)
            or not np.isfinite(value).all()):
        raise storage.RestartCorruptionError('invalid saved AO Fock shape or values')


def save_early_scf_reference(checkpoint_dir, mf, *, reference_energy=None, fock=None):
    """Atomically save successful warm SCF before any domain selection.

    Called with a detached host reference. The hidden file is independent of
    the later CC manifest and survives incomplete-initialization recovery.
    """
    if not bool(mf.converged):
        raise storage.RestartMismatchError('cannot save unconverged SCF')
    if reference_energy is not None and (not np.isfinite(reference_energy)
            or abs(float(mf.e_tot) - reference_energy) > 1e-7):
        raise storage.RestartMismatchError('cold and warm SCF energies disagree')
    canonical = {name: np.asarray(getattr(mf, name))
                 for name in ('mo_coeff', 'mo_energy', 'mo_occ', 'e_tot')}
    _validate_canonical(canonical, mf.mol)
    # Builders pass their already computed final Fock through stop_trace.
    # Standalone callers outside a capture may construct it once when saving.
    fock = np.asarray(mf.get_fock() if fock is None else fock)
    _validate_fock(fock, mf.mol)
    value = dict(canonical=canonical, scf_settings=_scf_settings(mf),
                 auxbasis=getattr(getattr(mf, 'with_df', None), 'auxbasis', None),
                 scf_class=f'{type(mf).__module__}.{type(mf).__qualname__}',
                 inputs=_molecular_inputs(mf.mol), fock=fock)
    path = Path(checkpoint_dir).expanduser().resolve()
    storage._durable_mkdir(path)
    lease = storage.RestartManager(None, method='stc-ccsd', scientific_payload={})
    lease.path = path
    lease._acquire_writer_lease()
    try:
        destination = path / '.scf-reference.h5'
        if destination.exists():
            previous = _load_early_scf_reference(path, mf.mol)
            if storage.scientific_digest(previous) != storage.scientific_digest(value):
                raise storage.RestartMismatchError('refusing to replace a saved SCF reference')
            return
        storage._atomic_hdf5(destination, lambda writer: dict(
            format=storage._FORMAT_NAME, format_version=storage._FORMAT_VERSION,
            equation_id=storage._EQUATION_ID, algorithm_abi=storage._ALGORITHM_ABI,
            kind='converged-scf', value=storage._encode_value(value, writer)))
    finally:
        lease.close()


def _load_early_scf_reference(checkpoint_dir, mol):
    handle, payload, reader = storage._read_hdf5(Path(checkpoint_dir) / '.scf-reference.h5')
    try:
        expected = dict(format=storage._FORMAT_NAME, format_version=storage._FORMAT_VERSION,
                        equation_id=storage._EQUATION_ID, algorithm_abi=storage._ALGORITHM_ABI,
                        kind='converged-scf')
        if any(payload.get(k) != v for k, v in expected.items()):
            raise storage.RestartMismatchError('saved early SCF format/implementation differs')
        value = storage._decode_value(payload['value'], reader)
    finally:
        handle.close()
    if storage.scientific_digest(value['inputs']) != storage.scientific_digest(_molecular_inputs(mol)):
        raise storage.RestartMismatchError('saved early SCF molecular inputs differ')
    _validate_canonical(value['canonical'], mol)
    if value.get('fock') is not None:
        _validate_fock(value['fock'], mol)
    return value


def recover_legacy_energy(checkpoint_dir, energy):
    """Supply an old run's exact HF energy, authenticated by its saved hash.

Older checkpoints stored only canonical hashes. A completed result's hf_energy
can recover the scalar without guessing it or modifying any existing record.
The numerical and gauge checks in validate_reference still apply on resume.
"""
    path = Path(checkpoint_dir)
    manifest = _manifest(path)
    expected = _mapping(_mapping(manifest['base_summary'])['canonical'])['e_tot']
    value = np.asarray(energy, dtype=np.dtype(expected['dtype'])) if expected['kind'] == 'array' else energy
    if storage._encode_value(value) != expected:
        raise storage.RestartMismatchError('recovered HF energy does not match the original checkpoint hash')
    destination = path / 'legacy-reference-energy.json'
    data = dict(base_digest=manifest['base_digest'],energy=float(energy))
    if destination.exists():
        if json.loads(destination.read_text()) != data:
            raise storage.RestartMismatchError('a different legacy reference energy is already recorded')
        return
    storage._atomic_bytes(destination, (json.dumps(data, sort_keys=True)+'\n').encode())


def _legacy_reference(manager, state, static):
    """Recover only AO/Boys-frame progress; unknown canonical gauges are unsafe."""
    if not getattr(manager, 'canonical_changed', False):
        return {'mode': 'legacy-exact'}
    path = manager.path / 'legacy-reference-energy.json'
    if not path.is_file():
        raise storage.RestartMismatchError(
            'legacy checkpoint has no numerical SCF reference; recover its exact '
            'HF energy from the original result with recover_legacy_energy')
    data = json.loads(path.read_text())
    if data['base_digest'] != manager.base_digest:
        raise storage.RestartMismatchError('legacy reference energy belongs to another checkpoint')
    canonical = _mapping(_mapping(manager._manifest['base_summary'])['canonical'])
    expected = canonical['e_tot']
    energy = np.asarray(data['energy'], dtype=np.dtype(expected['dtype'])) if expected['kind'] == 'array' else data['energy']
    if storage._encode_value(energy) != expected:
        raise storage.RestartCorruptionError('legacy reference energy failed the original hash')
    energy_error = _close(np.asarray(state['mf'].e_tot), np.asarray(energy),
                          'HF energy', atol=ENERGY_ATOL, rtol=0.)
    common = state.get('common')
    if common is None:
        raise storage.RestartMismatchError('legacy numerical recovery requires the live Boys reference')
    saved_targets = np.asarray(static.mp2_static.target_reference_coeff)
    occupied = np.asarray(common.occupied_coeff)
    overlap = np.asarray(common.s1e)
    projected = occupied @ (occupied.T.conj() @ overlap @ saved_targets)
    space_error = _close(projected, saved_targets, 'legacy active occupied space')
    frame_error = _close(common.iao_coeff, saved_targets, 'legacy Boys orbital frame')
    # There is no original virtual/canonical reference to transport these bars.
    # Decode with normal integrity checks, then explicitly require them zero.
    templates = {k:state[k] for k in ('mf_bar','common_bar')} if 'mf_bar' in state else {}
    record = manager.load_record('cc-progress', templates=templates)
    if record is not None and record.trees:
        for name, tree in record.trees.items():
            for keys, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
                attrs = [key.name for key in keys if isinstance(key, jax.tree_util.GetAttrKey)]
                allowed = (name == 'mf_bar' and attrs and attrs[-1] == 'coords') or (
                    name == 'common_bar' and attrs and attrs[0] in
                    ('s1e', 'fock', 'pao_coeff', 'iao_coeff', 'fragment_occupied_data'))
                if not allowed and np.any(np.asarray(value)):
                    raise storage.RestartMismatchError(
                        f'legacy checkpoint has gauge-dependent {name}{attrs}; numerical reference is missing')
    warnings.warn('Recovered legacy CC checkpoint using authenticated HF energy, '
                  'saved Boys frame and zero canonical cotangents; original full '
                  'SCF arrays were not stored.', RuntimeWarning, stacklevel=3)
    return dict(mode='legacy-recovery', hf_energy_max_abs=energy_error,
                occupied_space_max_abs=space_error, boys_frame_max_abs=frame_error,
                canonical_cotangents_zero=True)


def validate_reference(manager, state, static):
    if not bool(getattr(state['mf'], 'converged', False)):
        raise storage.RestartMismatchError('checkpoint restart requires converged SCF')
    templates = {'common': state['common']} if 'common' in state else {}
    reference = manager.load_record('cc-reference', templates=templates)
    if reference is None:
        return _legacy_reference(manager, state, static)
    if storage.scientific_digest(reference.scalars['scf_settings']) != storage.scientific_digest(_scf_settings(state['mf'])):
        raise storage.RestartMismatchError('checkpoint SCF/SGX settings differ')
    canonical = reference.scalars['canonical']
    # The reference file must really contain the original canonical values.
    if storage._encode_value(canonical) != _mapping(manager._manifest['base_summary'])['canonical']:
        raise storage.RestartCorruptionError('numerical SCF reference does not match the original identity')
    errors = {}
    for name, saved in canonical.items():
        errors[name] = _close(getattr(state['mf'], name), saved, name,
               atol=0. if name == 'mo_occ' else ENERGY_ATOL if name in ('e_tot','mo_energy') else ARRAY_ATOL,
               rtol=0. if name in ('e_tot','mo_energy','mo_occ') else ARRAY_RTOL)
    if reference.trees:
        current = jax.tree_util.tree_leaves(state['common'])
        saved = jax.tree_util.tree_leaves(reference.trees['common'])
        errors['common'] = max((_close(new, old, f'common[{index}]')
                               for index, (new, old) in enumerate(zip(current, saved))), default=0.)
    return dict(mode='numerical-reference', max_abs_differences=errors)
