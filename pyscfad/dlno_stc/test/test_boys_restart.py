"""STC checkpoints serialize only their own fixed Boys selections."""
from importlib import import_module, util
from pathlib import Path
from types import SimpleNamespace
import json

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _restart():
    assert util.find_spec('pyscfad.dlno_stc._restart') is not None, 'STC restart utilities are missing'
    return import_module('pyscfad.dlno_stc._restart')


def _manager(path, *, resume=False, payload=None):
    return _restart().RestartManager(path, resume=resume, method='stc-boys-selected',
                                    scientific_payload={'x': np.arange(3.)} if payload is None else payload)


def test_ordinary_payload_digest_matches_native():
    pytest.importorskip('pyscfad.dlno._restart')
    restart = _restart()
    from pyscfad.dlno import _restart as native
    payload = {'coords': np.arange(6.).reshape(2, 3), 'frozen': (0, 5),
               'settings': [True, None, 1e-5, 2j, b'abc', Path('input')],
               'scalar': np.asarray(3.), 'ints': np.arange(2, dtype=np.int32)}
    assert restart.scientific_digest(payload) == native.scientific_digest(payload)
    assert restart.scientific_digest(payload) != restart.scientific_digest({**payload, 'frozen': (1, 5)})


def test_df_fingerprint_matches_native_and_ignores_hdf5_path(tmp_path):
    pytest.importorskip('pyscfad.dlno._restart')
    restart = _restart()
    from pyscfad.dlno import _restart as native
    for source in (None, np.arange(12.).reshape(3, 4)):
        mf = SimpleNamespace(with_df=SimpleNamespace(_cderi=source))
        assert restart.df_source_fingerprint(mf) == native.df_source_fingerprint(mf)
    paths = [tmp_path / name for name in ('a.h5', 'b.h5')]
    for path in paths:
        with h5py.File(path, 'w') as handle:
            handle['j3c/0'] = np.arange(12.).reshape(3, 4)
    first = SimpleNamespace(with_df=SimpleNamespace(_cderi=str(paths[0])))
    second = SimpleNamespace(with_df=SimpleNamespace(_cderi=str(paths[1])))
    assert restart.df_source_fingerprint(first) == native.df_source_fingerprint(first)
    assert restart.df_source_fingerprint(first) == restart.df_source_fingerprint(second)


def test_static_thresholds_and_cotangents_round_trip(tmp_path):
    restart = _restart()
    from pyscfad.dlno_stc._selection import BoysDomainThresholds
    manager = _manager(tmp_path)
    thresholds = BoysDomainThresholds()
    manager.save_static(thresholds)
    bars = {'foo': jnp.arange(4.).reshape(2, 2), 'scalar': np.asarray(3.),
            'none': None, 'float0': np.zeros((2,), dtype=jax.dtypes.float0)}
    manager.save_record('weak', key='0-1', scalars={'energy': -.125}, trees={'bars': bars})
    manager.close()
    resumed = _manager(tmp_path, resume=True)
    restored = resumed.load_static(expected_type=BoysDomainThresholds)
    assert restart.scientific_digest(restored) == restart.scientific_digest(thresholds)
    record = resumed.load_record('weak', key='0-1', templates={'bars': bars})
    assert record.scalars == {'energy': -.125}
    np.testing.assert_array_equal(record.trees['bars']['foo'], bars['foo'])
    assert record.trees['bars']['scalar'].shape == ()
    assert record.trees['bars']['float0'].dtype == jax.dtypes.float0
    resumed.close()


def test_native_records_and_manifest_identity_are_rejected(tmp_path):
    pytest.importorskip('pyscfad.dlno.domain')
    restart = _restart()
    from pyscfad.dlno.domain import DLNOThresholds
    from pyscfad.dlno import _restart as native
    with pytest.raises(restart.RestartSerializationError, match='registry'):
        restart.scientific_digest(DLNOThresholds())
    with pytest.raises(restart.RestartCorruptionError, match='unregistered'):
        restart._decode_value(native._encode_value(DLNOThresholds()), None)
    manager = _manager(tmp_path)
    manager.close()
    manifest_path = tmp_path / 'run.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['format'] = native._FORMAT_NAME
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(restart.RestartMismatchError, match='match'):
        _manager(tmp_path, resume=True)


def test_scientific_and_live_tree_mismatches_are_rejected(tmp_path):
    restart = _restart()
    manager = _manager(tmp_path)
    manager.save_record('bars', trees={'data': {'x': np.ones(2)}})
    with pytest.raises(restart.RestartMismatchError):
        manager.load_record('bars', templates={'data': {'x': np.ones(3)}})
    manager.close()
    with pytest.raises(restart.RestartMismatchError):
        _manager(tmp_path, resume=True, payload={'x': np.arange(4.)})


def test_full_static_records_round_trip_with_stc_class_identity(tmp_path):
    restart = _restart()
    from pyscfad.dlno_stc._selection import (BoysDomainThresholds, FixedPAOSubspaceSelection,
                                           FragmentDomainSelection, DomainSelections)
    idx = np.array([0], dtype=np.int32)
    pao = FixedPAOSubspaceSelection(*(idx.copy() for _ in range(6)))
    fragment = FragmentDomainSelection(0, idx, idx, idx, idx, idx, idx, idx, pao,
                                       idx, idx, idx, idx, idx, pao, 0)
    static = DomainSelections(None, BoysDomainThresholds(), idx, np.array([1], dtype=np.int32),
                              idx, idx, np.array([0, -1], dtype=np.int32), (idx,), (idx,),
                              np.ones((1, 1), dtype=bool), (fragment,),
                              target_reference_coeff=np.array([[1.], [0.]]),
                              target_reference_coords=np.zeros((1, 3)))
    manager = _manager(tmp_path)
    manager.save_static(static)
    restored = manager.load_static(expected_type=DomainSelections)
    assert type(restored) is DomainSelections
    assert type(restored.fragments[0]) is FragmentDomainSelection
    assert type(restored.fragments[0].strong_virtual) is FixedPAOSubspaceSelection
    assert restart.scientific_digest(restored) == restart.scientific_digest(static)
    assert all(tag.startswith('pyscfad.dlno_stc.') for tag in restart._static_registry())
    manager.close()


@pytest.mark.parametrize('field,value', [('equation_id', 'weighted-domain-v0'), ('algorithm_abi', -1), ('algorithm_abi', 1)])
def test_checkpoint_equation_mismatch_is_rejected(tmp_path, field, value):
    restart = _restart()
    manager = _manager(tmp_path)
    manager.close()
    path = tmp_path / 'run.json'
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(restart.RestartMismatchError):
        _manager(tmp_path, resume=True)


def test_stale_weighted_fields_cannot_be_decoded_as_stc_records():
    restart = _restart()
    from pyscfad.dlno_stc._selection import BoysDomainThresholds
    node = restart._encode_value(BoysDomainThresholds())
    node['fields'].append(['occupied_weight', {'kind': 'float', 'hex': float(1e-4).hex(), 'dtype': None}])
    with pytest.raises(restart.RestartCorruptionError, match='fields changed'):
        restart._decode_value(node, None)


def _grouped_selections():
    from dataclasses import replace
    from pyscfad.dlno_stc._selection import (
        BoysDomainThresholds, FixedPAOSubspaceSelection,
        FragmentDomainSelection, DomainSelections, DomainGroup,
    )
    idx = np.array([0], dtype=np.int32)
    partners = np.array([0, 1], dtype=np.int32)
    pao = FixedPAOSubspaceSelection(*(idx.copy() for _ in range(6)))
    first = FragmentDomainSelection(
        0, np.array([1], dtype=np.int32), idx, partners, idx, idx, idx,
        partners, pao, idx, idx, idx, idx, idx, pao, 0)
    second = replace(first, fragment_index=1, iao_indices=idx, target_index=1)
    static = DomainSelections(
        None, BoysDomainThresholds(), partners, np.array([2], dtype=np.int32),
        partners, idx, np.array([0, -1], dtype=np.int32),
        (np.array([1], dtype=np.int32), idx), (idx, idx),
        np.ones((2, 2), dtype=bool), (first, second),
        target_reference_coeff=np.eye(2), target_reference_coords=np.zeros((1, 3)))
    return static, (DomainGroup((0, 1), 0, (0, 1)),)


def test_group_worklist_round_trip_and_replay_identity(tmp_path):
    restart = _restart()
    assert callable(getattr(restart, 'grouping_payload', None)), 'group payload builder is missing'
    assert callable(getattr(restart, 'validate_grouping_payload', None)), 'group replay validation is missing'
    from pyscfad.dlno_stc._selection import DomainGroup, DomainSelections
    static, groups = _grouped_selections()
    payload = restart.grouping_payload(static, groups)
    manager = _manager(tmp_path, payload=payload)
    manager.save_static({'static': static, 'groups': groups})
    manager.save_record('group', key=0, metadata=payload, scalars={'energy': -.125})
    manager.close()
    resumed = _manager(tmp_path, resume=True, payload=payload)
    restored = resumed.load_static(expected_type=dict)
    assert type(restored['static']) is DomainSelections
    assert type(restored['groups'][0]) is DomainGroup
    assert restored['groups'][0].fragment_indices == (0, 1)
    assert restored['groups'][0].representative_index == 0
    assert restored['groups'][0].target_indices == (0, 1)
    assert tuple(int(x[0]) for x in restored['static'].frag_lolist) == (1, 0)
    np.testing.assert_array_equal(restored['static'].fragments[0].strong_virtual.metric_keep, [0])
    saved = resumed.load_record('group', key=0).metadata
    assert restart.validate_grouping_payload(saved, static, groups) is None
    resumed.close()


@pytest.mark.parametrize('change', [
    'membership', 'representative', 'targets', 'partners', 'virtual-rank', 'reference',
    'convention', 'fingerprint', 'metadata',
])
def test_group_replay_rejects_changed_scientific_identity(change):
    restart = _restart()
    assert callable(getattr(restart, 'grouping_payload', None)), 'group payload builder is missing'
    from dataclasses import replace
    static, groups = _grouped_selections()
    saved = restart.grouping_payload(static, groups)
    if change == 'membership':
        groups = (replace(groups[0], fragment_indices=(1, 0)),)
    elif change == 'representative':
        groups = (replace(groups[0], representative_index=1),)
    elif change == 'targets':
        groups = (replace(groups[0], target_indices=(1, 0)),)
    elif change == 'partners':
        fragments = (replace(static.fragments[0], strong_fragments=np.array([1, 0], dtype=np.int32)),
                     static.fragments[1])
        static = replace(static, fragments=fragments)
    elif change == 'virtual-rank':
        virtual = replace(static.fragments[0].strong_virtual, metric_keep=np.array([], dtype=np.int32))
        fragments = (replace(static.fragments[0], strong_virtual=virtual), static.fragments[1])
        static = replace(static, fragments=fragments)
    elif change == 'reference':
        static = replace(static, target_reference_coeff=-np.eye(2))
    elif change == 'convention':
        saved = {**saved, 'convention': 'weighted-domain-v0'}
    elif change == 'fingerprint':
        saved = {**saved, 'fingerprint': 'invalid'}
    elif change == 'metadata':
        saved = {**saved, 'metadata': {'static': static, 'groups': ()}}
    with pytest.raises(restart.RestartMismatchError, match='group'):
        restart.validate_grouping_payload(saved, static, groups)


def test_group_replay_rejects_ungrouped_or_missing_metadata():
    restart = _restart()
    assert callable(getattr(restart, 'validate_grouping_payload', None)), 'group replay validation is missing'
    static, groups = _grouped_selections()
    for saved in (None, {}, {'static': static, 'groups': groups}):
        with pytest.raises(restart.RestartMismatchError, match='group'):
            restart.validate_grouping_payload(saved, static, groups)
