"""Exact saved-frame grouping and selected sums, with no orbital regrouping."""

from dataclasses import replace

import numpy as np
import pytest

from pyscfad import dlno_stc
from pyscfad.dlno_stc import prepare
from pyscfad.dlno_stc._selection import (
    BoysDomainThresholds, DomainSelections, FixedPAOSubspaceSelection,
    FragmentDomainSelection,
)
from pyscfad.dlno_stc.test.test_full_kernel import KEYS, grid
from pyscfad.dlno_stc.test.test_selected_kernel import solve
from pyscfad.dlno_stc.test.test_boys_prepare import boys_water_dimer


def grouping_fixture():
    """Equivalent strong frames with deliberately unequal weak records."""
    labels = (4, 0, 5, 1, 6, 2, 7, 3)
    partners = np.array([4, 0, 6, 2, 7, 1, 5, 3])
    virtual = FixedPAOSubspaceSelection(
        np.array([0, 1, 2]), np.array([0, 1, 2, 3]),
        np.array([0, 1, 2]), np.array([0, 1, 2]),
        np.array([0, 1, 2]), np.array([0, 1]),
    )
    fragments = tuple(FragmentDomainSelection(
        i, np.array([label]), None, partners.copy(), np.array([0, 1]),
        np.array([0, 1, 2, 3]), np.array([0, 1]), np.arange(8), virtual,
        np.array([i % 2]), np.array([i % 4]), np.array([i % 2]),
        np.array([i % 2]), np.array([0]), None,
        int(np.flatnonzero(partners == i)[0]), (0, 1),
    ) for i, label in enumerate(labels))
    return DomainSelections(
        0, BoysDomainThresholds(), np.arange(8), np.array([8, 9]),
        np.arange(8), np.array([0, 1, 2]), np.array([0, 1, 2, -1]),
        tuple(np.array([label]) for label in labels), (None,) * 8,
        np.ones((8, 8), dtype=bool), fragments,
        target_reference_coords=np.zeros((2, 3)),
    )


def test_grouping_exact_keys_and_partition():
    assert hasattr(dlno_stc, "build_domain_groups")
    assert hasattr(dlno_stc, "DomainGroup")
    static = grouping_fixture()
    fragments = list(static.fragments)
    # Four distinct reasons to refuse sharing otherwise identical support.
    fragments[1] = replace(fragments[1], extended_atoms=np.array([0]))
    fragments[3] = replace(fragments[3], strong_virtual=replace(
        fragments[3].strong_virtual, completeness_keep=np.array([0, 2]),
    ))
    fragments[5] = replace(fragments[5], virtual_anchor_columns=(1, 0))
    short = np.array([0, 5, 7])
    fragments[7] = replace(fragments[7], strong_fragments=short,
                           strong_occ_metric_keep=np.arange(3), target_index=2)
    static = replace(static, fragments=tuple(fragments))
    groups = dlno_stc.build_domain_groups(static)
    assert groups == (
        dlno_stc.DomainGroup((0, 2, 4, 6), 0, (1, 3, 0, 2)),
        dlno_stc.DomainGroup((1,), 1, (5,)),
        dlno_stc.DomainGroup((3,), 3, (7,)),
        dlno_stc.DomainGroup((5,), 5, (6,)),
        dlno_stc.DomainGroup((7,), 7, (2,)),
    )
    assert sorted(i for group in groups for i in group.fragment_indices) == list(range(8))
    for group in groups:
        candidate_labels = [int(static.frag_lolist[p][0]) for p in
                            static.fragments[group.representative_index].strong_fragments]
        assert len(set(group.target_indices)) == len(group.fragment_indices)
        for index, slot in zip(group.fragment_indices, group.target_indices):
            assert candidate_labels[slot] == int(static.frag_lolist[index][0])
    singles = dlno_stc.build_domain_groups(static, merge_identical=False)
    assert singles == tuple(dlno_stc.DomainGroup((i,), i, (fragment.target_index,))
                            for i, fragment in enumerate(static.fragments))
    empty = replace(static, active_occ_indices=np.array([], dtype=int),
                    frag_lolist=(), frag_atmlist=(), fragments=(),
                    strong_mask=np.zeros((0, 0), dtype=bool))
    assert dlno_stc.build_domain_groups(empty) == ()
    assert dlno_stc.build_domain_groups(empty, merge_identical=False) == ()


def test_grouping_preserves_order_and_rejects_invalid_saved_labels():
    assert hasattr(dlno_stc, "build_domain_groups")
    static = grouping_fixture()
    reordered = static.fragments[1].strong_fragments[::-1].copy()
    changed = replace(static.fragments[1], strong_fragments=reordered,
                      target_index=int(np.flatnonzero(reordered == 1)[0]))
    groups = dlno_stc.build_domain_groups(replace(
        static, fragments=(static.fragments[0], changed) + static.fragments[2:],
    ))
    assert groups[0].fragment_indices == (0, 2, 3, 4, 5, 6, 7)
    assert groups[1].fragment_indices == (1,)
    for field, value in (("strong_fragments", np.array([2**32], dtype=np.uint64)),
                         ("target_index", 2**32),
                         ("extended_ao_indices", np.array([2**32], dtype=np.uint64))):
        broken = replace(static.fragments[0], **{field: value})
        with pytest.raises(ValueError):
            dlno_stc.build_domain_groups(replace(
                static, fragments=(broken,) + static.fragments[1:],
            ))
    broken = replace(static.fragments[0], strong_occ_metric_keep=np.arange(7))
    with pytest.raises(RuntimeError, match="occupied|rank"):
        dlno_stc.build_domain_groups(replace(
            static, fragments=(broken,) + static.fragments[1:],
        ))
    with pytest.raises(ValueError, match="singleton"):
        dlno_stc.build_domain_groups(replace(
            static, frag_lolist=(np.array([2**32]),) + static.frag_lolist[1:],
        ))


def test_grouped_selected_energy_and_three_bars():
    rng = np.random.default_rng(936)
    occupied, virtual = rng.normal(size=(4, 4)), rng.normal(size=(3, 3))
    packet = {
        "foo": np.diag([-1.4, -1.1, -.8, -.6]) + .02 * (occupied + occupied.T),
        "fvv": np.diag([.3, .7, 1.]) + .02 * (virtual + virtual.T),
        "B": rng.normal(size=(5, 4, 3)) * .1,
    }
    targets = (3, 0, 2)  # Strict subset: occupied partner 1 remains available.
    controls = grid()
    grouped = solve(packet, targets, controls)
    singles = [solve(packet, (target,), controls) for target in targets]
    np.testing.assert_allclose(grouped["energy"], sum(x["energy"] for x in singles),
                               atol=1e-11, rtol=0)
    assert set(grouped["cotangents"]) == set(KEYS)
    for key in KEYS:
        np.testing.assert_allclose(grouped["cotangents"][key],
                                   sum(x["cotangents"][key] for x in singles),
                                   atol=1e-10, rtol=1e-10)
    # A simultaneous permutation changes packet axes and selected labels only.
    order = np.array([2, 0, 3, 1])
    permuted = dict(packet, foo=np.ascontiguousarray(packet["foo"][np.ix_(order, order)]),
                    B=np.ascontiguousarray(packet["B"][:, order]))
    mapped = tuple(int(np.flatnonzero(order == target)[0]) for target in targets)
    result = solve(permuted, mapped, controls)
    np.testing.assert_allclose(result["energy"], grouped["energy"], atol=1e-11, rtol=0)
    np.testing.assert_allclose(result["cotangents"]["foo"],
                               grouped["cotangents"]["foo"][np.ix_(order, order)],
                               atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(result["cotangents"]["fvv"], grouped["cotangents"]["fvv"],
                               atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(result["cotangents"]["B"], grouped["cotangents"]["B"][:, order],
                               atol=1e-10, rtol=1e-10)


def test_grouped_preparation_wrapper_uses_representative(boys_water_dimer):
    assert hasattr(prepare, "prepare_boys_group_inputs")
    mf, common, static = boys_water_dimer
    group = dlno_stc.build_domain_groups(static)[0]
    packet = prepare.prepare_boys_group_inputs(mf, common, static, group)
    assert set(packet) == set(KEYS)
    singleton = prepare.prepare_inputs(mf, common, static, group.representative_index)
    for key in KEYS:
        np.testing.assert_array_equal(packet[key], singleton[key])


def test_grouped_preparation_and_response(tmp_path):
    from pyscfad.dlno_stc.test import _boys_acceptance as accept
    accept.grouped_preparation_and_response(tmp_path)


def test_grouped_fixed_selection_coordinate_response(tmp_path):
    from pyscfad.dlno_stc.test import _boys_acceptance as accept
    accept.grouped_coordinate_response(tmp_path)
