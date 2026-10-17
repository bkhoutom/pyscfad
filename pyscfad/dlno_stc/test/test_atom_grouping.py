"""Atom grouping preserves baseline retained virtual spaces and target labels."""
from dataclasses import replace

import numpy as np
import pytest

from pyscfad import dlno_stc
from pyscfad.dlno_stc import _selection, _pairs, domain
from pyscfad.lno import tools
from pyscfad.dlno_stc.test.test_boys_prepare import boys_water_dimer
from pyscfad.dlno_stc.test.test_domain_grouping import grouping_fixture


def test_atom_groups_match_lowdin_assignment_and_permuted_record_labels(
    boys_water_dimer, monkeypatch,
):
    mf, _common, baseline = boys_water_dimer
    targets = np.asarray(baseline.target_reference_coeff)
    labels = (4, 0, 5, 1, 6, 2, 7, 3)
    singletons = tuple(np.array([label], dtype=np.int32) for label in labels)
    groups = _selection._boys_atom_groups(mf.mol, targets, singletons)

    # Independently compute the stated population formula in Boys-column order.
    overlap = np.asarray(mf.mol.intor_symmetric("int1e_ovlp"))
    eigenvalues, vectors = np.linalg.eigh(overlap)
    lowdin = (vectors * np.sqrt(eigenvalues)) @ vectors.T @ targets
    atom_groups = tools.autofrag(mf.mol, H2heavy=True)
    populations = np.array([
        np.sum(abs(lowdin[tools.ao_index_by_atom(mf.mol, atoms)]) ** 2, axis=0)
        for atoms in atom_groups
    ])
    ownership = np.argmax(populations, axis=0)
    expected = tuple(tuple(labels.index(int(column)) for column in
                           np.flatnonzero(ownership == group))
                     for group in range(len(atom_groups))
                     if np.any(ownership == group))
    assert tuple(tuple(group) for group in groups) == expected
    assert sorted(int(i) for group in groups for i in group) == list(range(8))
    assert all(group.dtype == np.int32 for group in groups)

    # Empty atom groups disappear, and invalid assignments are never narrowed.
    mapped = tools.map_lo_to_frag(mf.mol, targets, atom_groups, verbose=0)
    monkeypatch.setattr(tools, "map_lo_to_frag", lambda *args, **kwargs:
                        [np.array([], dtype=int)] + mapped)
    assert tuple(tuple(group) for group in _selection._boys_atom_groups(
        mf.mol, targets, singletons)) == expected
    for invalid in ([np.array([0, 0]), np.arange(1, 8)],
                    [np.arange(7)], [np.arange(7), np.array([2**32])]):
        monkeypatch.setattr(tools, "map_lo_to_frag",
                            lambda *args, result=invalid, **kwargs: result)
        with pytest.raises(ValueError, match="partition|indices"):
            _selection._boys_atom_groups(mf.mol, targets, singletons)


def test_atom_strong_graph_uses_any_member_and_exact_weak_complement():
    orbital = np.eye(5, dtype=bool)
    orbital[0, 1] = orbital[1, 0] = True
    groups = (np.array([2, 0]), np.array([4, 1]), np.array([3]))
    effective = _selection._expand_group_strong_mask(orbital, groups)
    expected = np.ones((5, 5), dtype=bool)
    expected[3, :] = expected[:, 3] = False
    expected[3, 3] = True
    np.testing.assert_array_equal(effective, expected)
    np.testing.assert_array_equal(effective, effective.T)
    assert np.all(effective[orbital])
    static = replace(grouping_fixture(), strong_mask=effective,
                     fragments=grouping_fixture().fragments[:5])
    assert _pairs.weak_pair_indices(static) == ((0, 3), (1, 3), (2, 3), (3, 4))
    singletons = tuple(np.array([i]) for i in range(5))
    np.testing.assert_array_equal(
        _selection._expand_group_strong_mask(orbital, singletons), orbital)
    for invalid in ((np.array([0, 1]), np.array([1, 2, 3, 4])),
                    (np.array([0, 1]), np.array([2, 3])),
                    (np.array([0, 1]), np.array([2, 3, 2**32]))):
        with pytest.raises(ValueError, match="partition|indices"):
            _selection._expand_group_strong_mask(orbital, invalid)


def _assert_saved_selection_equal(left, right):
    """Compare all concrete fields, including individual endpoint selections."""
    from dataclasses import fields, is_dataclass
    if is_dataclass(left):
        for field in fields(left):
            _assert_saved_selection_equal(getattr(left, field.name),
                                          getattr(right, field.name))
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_saved_selection_equal(first, second)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    else:
        assert left == right


def test_atom_public_builder_shares_strong_frames_and_preserves_endpoints(
    boys_water_dimer, monkeypatch,
):
    mf, _common, baseline = boys_water_dimer
    labels = (4, 0, 5, 1, 6, 2, 7, 3)
    singleton_groups = tuple(np.array([label]) for label in labels)
    targets = baseline.target_reference_coeff
    expected_physical = _selection._boys_atom_groups(mf.mol, targets, singleton_groups)
    baseline_permuted = dlno_stc.build_domain_selections(
        mf, frozen=2, target_coeff=targets, frag_lolist=singleton_groups)
    explicit_singleton = dlno_stc.build_domain_selections(
        mf, frozen=2, target_coeff=targets, frag_lolist=singleton_groups,
        boys_grouping="singleton")
    _assert_saved_selection_equal(baseline_permuted, explicit_singleton)
    # The pinned singleton dimer retains four local occupieds/two virtuals per ED.
    assert all(fragment.strong_occ_metric_keep.size == 4
               and fragment.strong_virtual.metric_keep.size == 2
               and fragment.extended_ao_indices.size == 7
               for fragment in explicit_singleton.fragments)

    captured = {}
    original_screen = _pairs.screen_pairs
    def inspect_screen(current_mf, common, static, **kwargs):
        captured["provisional"] = static
        result = original_screen(current_mf, common, static, **kwargs)
        captured["orbital_graph"] = result[3].copy()
        captured["pair_estimates"] = result[:3]
        return result
    monkeypatch.setattr(_pairs, "screen_pairs", inspect_screen)
    original_frame = domain.build_boys_ed_frame
    original_qr = domain.scipy_linalg.qr
    def reject_virtual_anchor_pivoting(*args, **kwargs):
        assert not kwargs.get("pivoting", False), "atom grouping restored virtual-anchor QR"
        return original_qr(*args, **kwargs)
    monkeypatch.setattr(domain.scipy_linalg, "qr", reject_virtual_anchor_pivoting)
    combined = dlno_stc.build_domain_selections(
        mf, frozen=2, target_coeff=targets, frag_lolist=singleton_groups,
        boys_grouping="atom")
    assert tuple(int(group[0]) for group in combined.frag_lolist) == labels
    _assert_saved_selection_equal(combined.compact_bp_domain,
                                  baseline_permuted.compact_bp_domain)
    expected_graph = _selection._expand_group_strong_mask(
        captured["orbital_graph"], expected_physical)
    np.testing.assert_array_equal(combined.strong_mask, expected_graph)
    for saved, screened in zip((combined.pair_energy, combined.weak_pair_energy,
                                combined.forced_strong_mask), captured["pair_estimates"]):
        np.testing.assert_array_equal(saved, screened)
    expected_weak = tuple((i, j) for i in range(8) for j in range(i + 1, 8)
                          if not expected_graph[i, j])
    assert _pairs.weak_pair_indices(combined) == expected_weak
    assert expected_weak  # This integration fixture also exercises surviving weak pairs.
    common = domain.rebuild_domain_data(mf, combined)
    for members in expected_physical:
        representative = combined.fragments[int(members[0])]
        primary_bp = np.unique(np.concatenate([
            baseline_permuted.primary_bp_domain[i] for i in members]))
        primary = np.unique(np.concatenate([
            baseline_permuted.primary_domain[i] for i in members]))
        slots = []
        for index in members:
            index = int(index)
            fragment = combined.fragments[index]
            provisional = captured["provisional"].fragments[index]
            np.testing.assert_array_equal(provisional.primary_bp_atoms, primary_bp)
            np.testing.assert_array_equal(provisional.primary_atoms, primary)
            assert provisional.iao_indices.size == 1
            assert provisional.weak_occ_norm_keep.size == 1
            for field in ("strong_fragments", "extended_atoms", "extended_ao_indices",
                          "pao_center_atoms", "strong_occ_metric_keep", "strong_virtual"):
                _assert_saved_selection_equal(getattr(fragment, field),
                                              getattr(representative, field))
            slot = fragment.target_index
            slots.append(slot)
            assert int(fragment.strong_fragments[slot]) == index
            assert labels[int(fragment.strong_fragments[slot])] == labels[index]
            assert original_frame(common, combined, index).target_index == slot
        assert len(set(slots)) == len(members)
    jobs = dlno_stc.build_domain_groups(combined)
    assert all(any(set(map(int, members)).issubset(job.fragment_indices)
                   for job in jobs) for members in expected_physical)
    with pytest.raises(ValueError, match="boys_grouping"):
        dlno_stc.build_domain_selections(mf, frozen=2, boys_grouping="invalid")


def test_atom_merged_and_unmerged_molecular_response_agree(boys_water_dimer):
    from pyscfad.dlno_stc.test import _boys_acceptance as accept

    mf, _common, baseline = boys_water_dimer
    static = dlno_stc.build_domain_selections(
        mf, frozen=2, target_coeff=baseline.target_reference_coeff,
        boys_grouping="atom",
    )
    groups = dlno_stc.build_domain_groups(static)
    assert len(groups) < len(static.fragments)
    assert _pairs.weak_pair_indices(static)
    unmerged_energy, unmerged_bar = accept.correlation_value_and_grad(mf.mol, static)
    merged_energy, merged_bar = accept.correlation_value_and_grad(mf.mol, static, groups)
    unmerged_grad = np.asarray(unmerged_bar.coords)
    merged_grad = np.asarray(merged_bar.coords)
    assert np.isfinite(merged_energy) and np.isfinite(unmerged_energy)
    assert np.isfinite(merged_grad).all() and np.isfinite(unmerged_grad).all()
    np.testing.assert_allclose(merged_energy, unmerged_energy, atol=1e-10, rtol=0)
    np.testing.assert_allclose(merged_grad, unmerged_grad, atol=1e-7, rtol=0)
