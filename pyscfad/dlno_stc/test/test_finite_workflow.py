"""One grouped selected-target packet contract through disk and molecular VJP."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
from pyscfad import scf
from pyscfad.dlno_stc import build_domain_groups
from pyscfad.dlno_stc import mp2
from pyscfad.dlno_stc.finite import export_finite_run, load_finite_static_from_run
from pyscfad.dlno_stc.test.test_boys_prepare import boys_water_dimer
from pyscfad.dlno_stc.test.test_full_kernel import grid
from pyscfad.dlno_stc.test.reference import reference_backend


def build_mf(mol):
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    return mf


def test_grouped_export_deserialize_solve_and_replay(boys_water_dimer, tmp_path, monkeypatch):
    mf, _, static = boys_water_dimer
    controls = dict(grid(), energy_tolerance=.0003)
    rows = export_finite_run(mf, static, tmp_path, controls=controls)
    groups = build_domain_groups(static)
    assert len(rows) == len(groups) < len(static.fragments)
    loaded = load_finite_static_from_run(mf, tmp_path, frozen=2, controls=controls)
    assert type(loaded) is type(static)
    assert controls['energy_tolerance'] == .0003
    for i, group in enumerate(groups):
        folder = tmp_path / f'fragment_{i:04d}'
        mp2.solve_request(folder/'input.h5', folder/'result.h5', reference_backend)
        inputs, metadata, effective, result = mp2.load_disk_packet(tmp_path, i)
        assert tuple(metadata['target_indices']) == group.target_indices
        assert metadata['energy_kind'] == 'finite_boys_target_mp2_v1'
        assert metadata['budget'] == dict(total_tolerance=.0003,
            original_target_count=8, group_count=len(groups),
            effective_tolerance=.0003/len(groups))
        assert effective['energy_tolerance'] == .0003/len(groups)
        assert set(inputs) == set(result['cotangents']) == {'foo','fvv','B'}
    with pytest.raises(ValueError, match='controls'):
        load_finite_static_from_run(mf, tmp_path, frozen=2,
                                   controls=dict(controls, energy_tolerance=.001))
    monkeypatch.setattr(mp2, 'allocate_domain_controls', lambda *a: pytest.fail('replay reallocates'))
    value, grad = mp2.imported_value_and_grad(mf.mol, build_mf, loaded,
        lambda i: mp2.load_disk_packet(tmp_path, i), include_hf=False)
    assert np.isfinite(value)
    assert np.isfinite(np.asarray(grad.coords)).all()
    # Native selected quadrature and dense reference use the same converged
    # MP2 denominator in this check (explicit accurate quadrature).
    from pyscfad.dlno_stc.driver import value_and_grad
    nodes, weights = np.polynomial.legendre.leggauss(48)
    accurate = dict(mode='deterministic', laplace_roots=(nodes+1)*15,
                    laplace_weights=weights*15)
    direct, direct_grad = value_and_grad(mf.mol, build_mf, loaded,
                                       controls=accurate, include_hf=False)
    np.testing.assert_allclose(value, direct, atol=1e-9, rtol=0)
    np.testing.assert_allclose(grad.coords, direct_grad.coords, atol=1e-8, rtol=0)


def test_packet_replay_rejects_changed_identity(boys_water_dimer):
    mf, common, static = boys_water_dimer
    groups = build_domain_groups(static)
    group = groups[0]
    inputs = mp2.prepare_boys_group_inputs(mf, common, static, group)
    budget = dict(total_tolerance=.0003, original_target_count=8,
                  group_count=len(groups), effective_tolerance=.0003/len(groups))
    controls = dict(energy_tolerance=budget['effective_tolerance'])
    metadata = mp2._packet_metadata(mf, static, 0, group, inputs, groups, budget)
    for field, replacement in [('target_indices', [99]), ('candidate_order', []),
            ('frame_convention', 'semicanonical'), ('virtual_anchor_parent_columns', []),
            ('global_target_orbitals', []), ('grouping_fingerprint', '0'*64),
            ('frozen', 0), ('molecular_fingerprint', '0'*64)]:
        changed = deepcopy(metadata)
        changed[field] = replacement
        with pytest.raises(ValueError, match=field):
            mp2._validate_packet_identity(mf, static, 0, group, inputs, changed, controls, groups)
    with pytest.raises(ValueError, match='budget'):
        mp2._validate_packet_identity(mf, static, 0, group, inputs, metadata,
                                     dict(energy_tolerance=.0003), groups)


def test_empty_strong_worklist_export_load_and_response(boys_water_dimer, tmp_path):
    from pyscfad.dlno_stc import kernel
    mf, _, static = boys_water_dimer
    static = replace(static, frozen=int(np.count_nonzero(mf.mo_occ)),
                     active_occ_indices=np.array([], dtype=int), fragments=(),
                     frag_lolist=(), frag_atmlist=(), strong_mask=np.zeros((0, 0), dtype=bool))
    controls = dict(grid(), energy_tolerance=.0003)
    messages = []
    assert mp2.export_run(mf, static, tmp_path, controls=controls,
                          reporter=messages.append) == []
    assert messages and 'Fragment' in messages[0]
    restored = mp2.load_static_from_run(mf, tmp_path, frozen=static.frozen, controls=controls)

    def forbidden(_):
        pytest.fail('empty run loaded a numerical packet')

    value, bar = mp2.imported_value_and_grad(mf.mol, build_mf, restored, forbidden,
                                            include_hf=False)
    assert float(value) == 0 and float(kernel(mf, static, controls=controls)) == 0
    np.testing.assert_array_equal(bar.coords, np.zeros_like(mf.mol.coords))
    assert controls['energy_tolerance'] == .0003
