"""Final finite preparation uses the verified selected Boys frame."""
import numpy as np
from pyscfad.dlno_stc import build_domain_groups
from pyscfad.dlno_stc.prepare import prepare_inputs, prepare_boys_group_inputs
from pyscfad.dlno_stc.test.test_boys_prepare import boys_water_dimer


def test_finite_group_packet_matches_saved_representative(boys_water_dimer):
    mf, common, static = boys_water_dimer
    groups = build_domain_groups(static)
    assert len(groups) < len(static.fragments)
    for group in groups:
        representative = static.fragments[group.representative_index]
        assert len(representative.extended_ao_indices) < mf.mol.nao
        singleton = prepare_inputs(mf, common, static, group.representative_index)
        grouped = prepare_boys_group_inputs(mf, common, static, group)
        assert set(grouped) == {'foo', 'fvv', 'B'}
        assert grouped['foo'].shape[0] == len(representative.strong_fragments)
        for name in grouped:
            np.testing.assert_array_equal(grouped[name], singleton[name])
        for local, original in zip(group.target_indices, group.fragment_indices):
            assert representative.strong_fragments[local] == original
