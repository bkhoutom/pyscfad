"""Full and finite public exchange entry points share one implementation."""
from pyscfad.dlno_stc import finite, mp2


def test_finite_wrappers_share_grouped_exchange():
    assert finite.export_finite_run is mp2.export_run
    assert finite.load_finite_disk_packet is mp2.load_disk_packet
    assert finite.load_finite_static_from_run is mp2.load_static_from_run


def test_sixteen_targets_four_jobs_allocate_export_once(tmp_path, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    import numpy as np
    from pyscfad.dlno_stc.test.test_domain_grouping import grouping_fixture
    from pyscfad.dlno_stc.test.test_full_kernel import grid
    from pyscfad.dlno_stc.exchange import read_input
    static = grouping_fixture()
    template = static.fragments[0]
    fragments = tuple(replace(template, fragment_index=i, iao_indices=np.array([i]),
        strong_fragments=np.arange(4*(i//4), 4*(i//4+1)),
        strong_occ_metric_keep=np.arange(4), target_index=i%4,
        extended_atoms=np.array([i//4])) for i in range(16))
    static = replace(static, active_occ_indices=np.arange(16), fragments=fragments,
        target_reference_coords=np.zeros((4, 3)),
        frag_lolist=tuple(np.array([i]) for i in range(16)), frag_atmlist=(None,)*16,
        strong_mask=np.equal.outer(np.arange(16)//4, np.arange(16)//4))
    mf = SimpleNamespace(with_df=SimpleNamespace(auxbasis='test'))
    from pyscfad.dlno_stc import driver
    monkeypatch.setattr(driver, '_validate_reference', lambda *args: None)
    payload_calls = []
    def payload(mf, frozen):
        payload_calls.append(frozen)
        return {'frozen': frozen}
    monkeypatch.setattr(mp2, '_run_payload', payload)
    monkeypatch.setattr(mp2, 'check_memory', lambda *args: None)
    monkeypatch.setattr(mp2, 'rebuild_domain_data', lambda *args: None)
    monkeypatch.setattr(mp2, 'prepare_boys_group_inputs', lambda *args: {
        'foo': -np.eye(4), 'fvv': np.eye(2), 'B': np.ones((3, 4, 2))})
    controls = dict(grid(), energy_tolerance=.0003, production_samples=32)
    for merge, count, tolerance in [(True, 4, 1.5e-4), (False, 16, 7.5e-5)]:
        folder = tmp_path / str(merge)
        rows = mp2.export_run(mf, static, folder, controls=controls, merge_identical=merge)
        assert len(rows) == count
        for i in range(count):
            _, md, effective = read_input(folder/f'fragment_{i:04d}'/'input.h5')
            assert effective['energy_tolerance'] == tolerance
            assert effective['production_samples'] == 32
            assert md['budget'] == dict(total_tolerance=.0003,
                original_target_count=16, group_count=count, effective_tolerance=tolerance)
            assert md['fragment_id'] == md['representative_index']
            assert md['group_id'] == i
    assert controls['energy_tolerance'] == .0003
    assert len(payload_calls) == 2  # One DF/source identity per exported run.


def test_stochastic_disk_uses_representative_seed_identity(tmp_path):
    import numpy as np
    from pyscfad.dlno_stc import backend
    from pyscfad.dlno_stc.exchange import read_input, write_input
    from pyscfad.dlno_stc.test.test_protocol import packet, finite_metadata
    from pyscfad.dlno_stc.test.test_full_kernel import grid
    inputs, metadata = packet(), finite_metadata()
    metadata.update(fragment_id=4, representative_index=4, group_id=1,
                    budget=dict(total_tolerance=.0003, effective_tolerance=1.5e-4,
                                original_target_count=16, group_count=4))
    controls = dict(grid(), mode='stochastic', energy_tolerance=1.5e-4,
                    production_samples=64, pilot_samples=32,
                    min_production_samples=2, global_seed=17,
                    system_workload_cutoff=2.)
    direct = backend.solve(inputs, {'fragment_id': 4, 'target_indices': [1]}, controls)
    path = tmp_path/'input.h5'
    write_input(path, inputs, metadata, controls)
    saved, md, effective = read_input(path)
    disk = backend.solve(saved, md, effective)
    assert direct['diagnostics']['replicas'][0]['actual_sample_count'] > 0
    assert direct['energy'] == disk['energy']
    assert direct['energy_standard_error'] == disk['energy_standard_error']
    for key in inputs:
        np.testing.assert_array_equal(direct['cotangents'][key], disk['cotangents'][key])
    assert direct['diagnostics']['replicas'][0]['actual_sample_count'] == disk['diagnostics']['replicas'][0]['actual_sample_count']


def test_disk_rejects_invalid_topology_and_basis_response_before_work(tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace
    import pytest
    from pyscfad.dlno_stc.test.test_domain_grouping import grouping_fixture
    static = grouping_fixture()
    mask = static.strong_mask.copy()
    mask[0, 1] = False
    invalid = replace(static, strong_mask=mask)
    coordinate_only = SimpleNamespace(exp=None, ctr_coeff=None, r0=None)

    def forbidden(*args):
        pytest.fail('invalid replay reached SCF or packet loading')

    with pytest.raises(ValueError, match='strong_mask'):
        mp2.export_run(None, invalid, tmp_path, controls={})
    with pytest.raises(ValueError, match='strong_mask'):
        mp2.imported_value_and_grad(coordinate_only, forbidden, invalid, forbidden)
    with pytest.raises(NotImplementedError, match='coordinates only'):
        mp2.imported_value_and_grad(SimpleNamespace(exp=1), forbidden, static, forbidden)
