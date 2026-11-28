"""Domain links select occupied partners before pooling virtual support over targets."""
from dataclasses import replace

import numpy as np
import pytest

from pyscfad import dlno_stc
from pyscfad.dlno_stc import _pairs, _selection, driver
from pyscfad.dlno_stc.test.test_boys_prepare import boys_water_dimer
from pyscfad.dlno_stc.test.test_primary_response import _compare


@pytest.fixture
def domain_link_dimer(boys_water_dimer, monkeypatch):
    mf, _, original = boys_water_dimer
    thresholds = replace(original.thresholds, near_pair_distance=0., pair_energy=1e-5)
    # Orbitals 0 and 2 connect only through 1. Orbital 6 shares an atom
    # group with 0 but has no graph edge of its own.
    compact = ([0, 1], [0, 1, 2, 3, 5], [2, 3, 4], [2], [1, 4], [1, 4], [0], [4])
    tight = ([0, 1], list(range(6)), [2, 3, 4], [2, 4], [1, 4], [1, 4, 5], [0], [4])
    groups = ([0, 6], [1], [2], [3], [4], [5], [7])
    bp = _selection.get_bp_domain

    def fixed_bp(*args, bp_thr, **kwargs):
        if bp_thr == thresholds.bp_occ:
            return tuple(map(np.array, compact))
        if bp_thr == thresholds.bp_ed:
            return tuple(map(np.array, tight))
        return bp(*args, bp_thr=bp_thr, **kwargs)

    monkeypatch.setattr(_selection, "get_bp_domain", fixed_bp)
    monkeypatch.setattr(_selection, "_boys_atom_groups",
                        lambda *args: tuple(map(np.array, groups)))
    scores = np.zeros((8, 8))
    scores[0, 3] = scores[4, 5] = -2e-5
    scores[0, 2] = scores[2, 3] = -2e-6
    scores[0, 1] = -thresholds.pair_energy / 100
    scores += scores.T
    monkeypatch.setattr(_pairs, "_exact_boys_pair_scores", lambda *args, **kw: scores)

    def build(mf, *, pair_cutoff=1e-5, **options):
        return dlno_stc.build_domain_selections(
            mf, frozen=2, thresholds=replace(thresholds, pair_energy=pair_cutoff),
            boys_grouping="atom",
            pair_energy_model="exact", **options)

    return mf, build


def test_domain_links_share_target_support_and_weak_partition(domain_link_dimer):
    mf, build = domain_link_dimer
    linked = build(mf, domain_link=True)
    baseline = build(mf)
    np.testing.assert_array_equal(baseline.strong_mask[0].nonzero()[0], [0, 3, 6])
    np.testing.assert_array_equal(linked.strong_mask[0].nonzero()[0], [0, 2, 3, 6])
    np.testing.assert_array_equal(linked.strong_mask[2].nonzero()[0], [0, 2])
    for i in (0, 6):
        np.testing.assert_array_equal(linked.fragments[i].pao_center_atoms, range(6))
        np.testing.assert_array_equal(linked.fragments[i].extended_atoms, range(6))
    # An external target with a linked *partner* does not inherit the component.
    np.testing.assert_array_equal(linked.fragments[3].strong_fragments, [0, 3])
    np.testing.assert_array_equal(linked.fragments[3].pao_center_atoms, [0, 1, 2])
    np.testing.assert_array_equal(linked.fragments[3].extended_atoms, [0, 1, 2, 4])

    fused, _, _, receipt = driver._select_domain_reference(
        mf, lambda ref: build(ref, domain_link=True))
    np.testing.assert_array_equal(fused.strong_mask, linked.strong_mask)
    common, _, pending = receipt.take(fused)
    expected_pairs = _pairs.weak_pair_indices(linked)
    assert pending[0] == expected_pairs
    assert (0, 2) not in expected_pairs and _pairs.weak_pair_weight(fused, 2, 6) == .5
    replay = _pairs.weak_pair_energy_and_grad(mf, common, linked)
    _compare(pending[1], replay)


def test_component_gate_uses_fixed_coarse_reference(domain_link_dimer):
    mf, build = domain_link_dimer
    linked = build(mf, pair_cutoff=1e-4, domain_link=True)
    # 4 and 5 share two centers and have a score of 2e-5. They form a
    # separate component with every internal pair strong at the reference
    # 1e-5, so the looser user cutoff activates neither kind of linking.
    np.testing.assert_array_equal(linked.fragments[4].strong_fragments, [4])
    np.testing.assert_array_equal(linked.fragments[4].extended_atoms, [1, 4])
    assert linked.strong_mask[0, 2]  # The component with a weak internal pair is active.
