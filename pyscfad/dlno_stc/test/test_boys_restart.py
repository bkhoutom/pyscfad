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


@pytest.mark.parametrize('domain_link', [False, True])
def test_full_static_records_round_trip_with_stc_class_identity(tmp_path, domain_link):
    restart = _restart()
    from pyscfad.dlno_stc._selection import (BoysDomainThresholds, FixedPAOSubspaceSelection,
                                           FragmentDomainSelection, DomainSelections)
    idx = np.array([0], dtype=np.int32)
    pao = FixedPAOSubspaceSelection(*(idx.copy() for _ in range(6)))
    fragment = FragmentDomainSelection(0, idx, idx, idx, idx, idx, idx, idx, pao,
                                       idx, idx, idx, idx, idx, pao, 0,
                                       virtual_anchor_columns=(0,))
    static = DomainSelections(None, BoysDomainThresholds(), idx, np.array([1], dtype=np.int32),
                              idx, idx, np.array([0, -1], dtype=np.int32), (idx,), (idx,),
                              np.ones((1, 1), dtype=bool), (fragment,),
                              target_reference_coeff=np.array([[1.], [0.]]),
                              target_reference_coords=np.zeros((1, 3)), domain_link=domain_link)
    encoded_fields = dict(restart._encode_value(static)['fields'])
    # Omitting the default preserves the serialized identity of older off-mode runs.
    assert ('domain_link' in encoded_fields) == domain_link
    manager = _manager(tmp_path)
    manager.save_static(static)
    restored = manager.load_static(expected_type=DomainSelections)
    assert type(restored) is DomainSelections
    assert restored.domain_link is domain_link
    assert type(restored.fragments[0]) is FragmentDomainSelection
    assert type(restored.fragments[0].strong_virtual) is FixedPAOSubspaceSelection
    assert restart.scientific_digest(restored) == restart.scientific_digest(static)
    assert all(tag.startswith('pyscfad.dlno_stc.') for tag in restart._static_registry())
    manager.close()


@pytest.mark.parametrize('field,value', [('equation_id', 'weighted-domain-v0'), ('algorithm_abi', -1), ('algorithm_abi', 1), ('algorithm_abi', 2)])
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


def test_old_domain_convention_is_rejected_before_decoding_arrays():
    from pyscfad.dlno_stc.test.test_domain_grouping import grouping_fixture
    restart = _restart()
    encoded = restart._encode_value(grouping_fixture())
    encoded['fields'] = [pair for pair in encoded['fields'] if pair[0] != 'domain_convention']
    with pytest.raises(restart.RestartMismatchError, match='domain/physical'):
        restart._decode_value(encoded, None)


def test_minus_eight_domain_checkpoint_requires_reselection(tmp_path):
    from dataclasses import replace
    from pyscfad.dlno_stc.test.test_domain_grouping import grouping_fixture
    restart = _restart()
    old = replace(grouping_fixture(),
                  domain_convention='individual-partners-pair8-pcd-aux-v1')
    manager = _manager(tmp_path)
    manager.save_static(old)
    manager.close()
    manager = _manager(tmp_path, resume=True)
    with pytest.raises(restart.RestartMismatchError, match='pair normalization'):
        manager.load_static(expected_type=type(old))
    manager.close()
