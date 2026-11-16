"""New ED labels retain exact grouping and durable replay semantics."""
from dataclasses import replace

import numpy as np
import pytest

from pyscfad.dlno_stc import _restart, build_domain_groups, driver
from pyscfad.dlno_stc._selection import BoysDomainThresholds, EDLocalPAOSelection
from pyscfad.dlno_stc.test.test_domain_grouping import grouping_fixture
from pyscfad.dlno_stc.test.test_boys_restart import _manager


def local_metadata_fixture():
    static = grouping_fixture()
    fragments = tuple(replace(f, strong_virtual=EDLocalPAOSelection(
        f.strong_virtual.parent_columns, f.extended_ao_indices,
        np.arange(3), np.arange(2))) for f in static.fragments)
    return replace(static, fragments=fragments)


def test_local_ed_grouping_distinguishes_retained_parent_labels():
    static = local_metadata_fixture()
    assert len(build_domain_groups(static)) == 1
    changed = replace(static.fragments[1], strong_virtual=replace(
        static.fragments[1].strong_virtual, completeness_keep=np.array([0, 2])))
    groups = build_domain_groups(replace(
        static, fragments=(static.fragments[0], changed) + static.fragments[2:]))
    assert groups[0].fragment_indices == (0, 2, 3, 4, 5, 6, 7)
    assert groups[1].fragment_indices == (1,)


def test_local_ed_grouping_requires_matching_ao_support():
    static = local_metadata_fixture()
    changed = replace(static.fragments[0], strong_virtual=replace(
        static.fragments[0].strong_virtual, support_ao_indices=np.array([0, 1])))
    with pytest.raises(ValueError, match="support"):
        build_domain_groups(replace(static, fragments=(changed,) + static.fragments[1:]))


def test_local_ed_checkpoint_roundtrip(tmp_path):
    static = local_metadata_fixture()
    manager = _manager(tmp_path)
    manager.save_static(static)
    manager.close()
    manager = _manager(tmp_path, resume=True)
    restored = manager.load_static(expected_type=type(static))
    manager.close()
    assert _restart.scientific_digest(restored) == _restart.scientific_digest(static)
    assert build_domain_groups(restored) == build_domain_groups(static)


def test_legacy_threshold_record_gets_default_local_rank_cutoff():
    encoded = _restart._encode_value(BoysDomainThresholds())
    encoded['fields'] = [pair for pair in encoded['fields'] if pair[0] != 'ed_pao_rank']
    assert _restart._decode_value(encoded, None).ed_pao_rank == 1e-6


@pytest.mark.parametrize('validate', [build_domain_groups, driver._validate_static])
def test_norm_based_selection_requires_rebuilding(validate):
    old = replace(grouping_fixture(), domain_convention='individual-partners-pair4-pcd-aux-v2')
    with pytest.raises(ValueError, match='domain convention'):
        validate(old)


def test_norm_based_checkpoint_selection_requires_rebuilding(tmp_path):
    import h5py
    old = replace(grouping_fixture(), domain_convention='individual-partners-pair4-pcd-aux-v2')
    with h5py.File(tmp_path / 'old.h5', 'w') as handle:
        encoded = _restart._encode_value(old, _restart._DatasetWriter(handle))
        with pytest.raises(_restart.RestartMismatchError, match='[Dd]omain'):
            _restart._decode_value(encoded, _restart._DatasetReader(handle))
