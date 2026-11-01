from pyscfad.dlno import dlno_base, domain as domain_module, mp2
from pyscfad.dlno import _selection

from dataclasses import replace

import numpy as np
import pytest


from pyscfad.dlno.domain import get_bp_domain
from pyscfad.dlno.test.test_boys_targets import water_mf
from pyscfad.dlno.test.test_mp2 import separated_water_dimer_mf
from pyscfad.mp import dfmp2


def test_scalar_bp_domains_and_permuted_targets(water_mf):
    groups = [[3], [1], [0], [2]]
    top = domain_module.build_domain_topology(
        water_mf, frozen=1, lo_type="boys", frag_lolist=groups,
        pair_energy_model="all",
    )
    assert top.lo_type == "boys"
    for field, threshold in (("compact_bp_domain", top.thresholds.bp_occ),
                             ("primary_bp_domain", top.thresholds.bp_primary),
                             ("tight_bp_domain", top.thresholds.bp_ed)):
        reference = get_bp_domain(water_mf.mol, top.target_coeff,
                                  s1e=top.s1e, bp_thr=threshold)
        for position, group in enumerate(groups):
            np.testing.assert_array_equal(getattr(top, field)[position], reference[group[0]])
    for fragment, partners in enumerate(top.strong_fragments):
        np.testing.assert_array_equal(top.strong_lmo_indices[fragment], [groups[p][0] for p in partners])
        for field, source in (("pao_center_domain", "compact_bp_domain"),
                              ("extended_domain", "tight_bp_domain")):
            reference = np.unique(np.concatenate([getattr(top, source)[p] for p in partners]))
            np.testing.assert_array_equal(getattr(top, field)[fragment], reference)
    screen = domain_module._fragment_screen_space(water_mf, top, 0)
    np.testing.assert_array_equal(screen["weights"], [1.0])
    np.testing.assert_allclose(screen["occupied_coeff"][0].T @ top.s1e @ screen["occupied_coeff"][0], 1.0, atol=1e-10)


def test_boys_rejects_atom_fragments_and_grouping(water_mf):
    for kwargs in ({"frag_atmlist": [[0], [1], [2]]},
                   {"frag_lolist": [[0, 1], [2], [3]]}):
        with pytest.raises(ValueError, match="Boys|singleton"):
            domain_module.build_domain_topology(water_mf, frozen=1, lo_type="boys", **kwargs)


def test_full_domain_boys_mp2_with_frozen_core_and_virtual(water_mf):
    frozen = [0, water_mf.mo_coeff.shape[1] - 1]
    top = domain_module.build_domain_topology(
        water_mf, lo_type="boys", frozen=frozen, force_full_domains=True,
        thresholds=domain_module.DLNOThresholds(domain_pao=0.0, ed_pao=0.0, pao_norm=1e-10),
    )
    actual = mp2.evaluate_domain_mp2(water_mf, top)
    reference, _ = dfmp2.MP2(water_mf, frozen=frozen).kernel(with_t2=False)
    assert actual.e_corr == pytest.approx(reference, abs=5e-9)
    assert all(fragment.n_domain_occ == 4 and fragment.n_domain_vir == 1 for fragment in actual.fragments)


def test_mp2_gradient_rejects_wrong_target_mode_before_scf(water_mf):
    top = domain_module.build_domain_topology(water_mf, lo_type="boys", pair_energy_model="all")
    with pytest.raises(ValueError, match="localization"):
        mp2.DLNOMP2.value_and_grad(
            water_mf.mol, build_mf=lambda mol: None, topology=top, lo_type="iao"
        )


def test_truncated_ed_preserves_raw_target_and_partner_weights(water_mf):
    top = domain_module.build_domain_topology(
        water_mf, frozen=1, lo_type="boys", pair_energy_model="all",
        thresholds=domain_module.DLNOThresholds(ed_pao=0.0, domain_pao=0.0),
    )
    oxygen_domain = np.empty(4, dtype=object)
    oxygen_domain[:] = [np.array([0])] * 4
    top = replace(top, extended_domain=oxygen_domain, pao_center_domain=oxygen_domain)
    losses = []
    for target in range(4):
        domain = domain_module._build_fragment_domain_orbitals(water_mf, top, target)
        rows = domain["ao_idx"]
        sdd = top.s1e[np.ix_(rows, rows)]
        projected = np.linalg.solve(sdd, top.s1e[rows] @ top.target_coeff)
        metric = projected.T @ sdd @ projected
        eig, vec = np.linalg.eigh(metric)
        keep = eig > top.thresholds.metric_rank
        orth = projected @ (vec[:, keep] / np.sqrt(eig[keep]))
        reference_energy = np.linalg.eigvalsh(orth.T @ top.fock[np.ix_(rows, rows)] @ orth)
        np.testing.assert_allclose(domain["occupied_energy"], reference_energy, atol=1e-10)
        co = domain["occupied_coeff"]
        np.testing.assert_allclose(co.T @ sdd @ co, np.eye(co.shape[1]), atol=1e-10)
        overlap = top.target_coeff.T @ top.s1e[:, rows] @ co
        raw = overlap[target:target+1].T @ overlap[target:target+1]
        np.testing.assert_allclose(domain["target_weight"], raw, atol=1e-12)
        np.testing.assert_allclose(domain["partner_weight"], overlap.T @ overlap, atol=1e-12)
        norm = np.trace(raw)
        losses.append(1.0 - norm)
        assert not np.allclose(domain["partner_weight"], np.eye(co.shape[1]), atol=1e-5)
    assert max(losses) > 0.01


def test_separated_boys_pairs_are_single_modes_and_weak(separated_water_dimer_mf, monkeypatch):
    def no_weight_modes(*args, **kwargs):
        raise AssertionError("Boys must preserve its actual localized frame")
    monkeypatch.setattr(domain_module, '_fragment_multipole_modes', no_weight_modes)
    top = domain_module.build_domain_topology(
        separated_water_dimer_mf, frozen=2, lo_type="boys",
    )
    np.testing.assert_array_equal(top.strong_mask, top.strong_mask.T)
    assert np.all(np.diag(top.strong_mask))
    assert np.any(~top.strong_mask)
    assert np.any(top.weak_pair_energy[~top.strong_mask] != 0)
    for i in range(len(top.frag_lolist)):
        screen = domain_module._fragment_screen_space(separated_water_dimer_mf, top, i)
        np.testing.assert_array_equal(screen["weights"], [1.0])


def test_pair_cutoff_is_individual_and_strict(water_mf, monkeypatch):
    threshold = 1e-4
    def pairs(mf, topology):
        pair = np.zeros((4, 4))
        pair[0, 1] = pair[1, 0] = -0.6 * threshold
        pair[0, 2] = pair[2, 0] = -0.6 * threshold
        pair[0, 3] = pair[3, 0] = -threshold
        return pair, np.eye(4, dtype=bool)
    monkeypatch.setattr(domain_module, '_multipole_fragment_pairs', pairs)
    top = domain_module.build_domain_topology(
        water_mf, frozen=1, lo_type="boys",
        thresholds=domain_module.DLNOThresholds(pair_energy=threshold),
    )
    np.testing.assert_array_equal(top.strong_mask, np.eye(4, dtype=bool))
    for i in range(4):
        np.testing.assert_array_equal(top.extended_domain[i], top.tight_bp_domain[i])
        np.testing.assert_array_equal(top.pao_center_domain[i], top.compact_bp_domain[i])


@pytest.mark.parametrize("truncate", [False, True])
def test_concrete_and_rebuilt_boys_domains_and_screens(water_mf, truncate):
    

    top = domain_module.build_domain_topology(
        water_mf, frozen=1, lo_type="boys", frag_lolist=[[3], [1], [0], [2]],
        pair_energy_model="all",
        lo_kwargs={"conv_tol": 1e-12, "conv_tol_grad": 1e-9},
        thresholds=domain_module.DLNOThresholds(ed_pao=0.0, domain_pao=0.0),
    )
    if truncate:
        domains = np.empty(4, dtype=object)
        domains[:] = [np.array([0])] * 4
        top = replace(top, extended_domain=domains, pao_center_domain=domains)
    static = _selection.build_domain_selections(water_mf, top)
    common = dlno_base.rebuild_domain_data(water_mf, static)
    for target in range(4):
        reference = domain_module._build_fragment_domain_orbitals(water_mf, top, target)
        actual = dlno_base.build_strong_ed_domain(common, static, target)
        rows = reference["ao_idx"]
        metric = top.s1e[np.ix_(rows, rows)]
        for space in ("occupied", "virtual"):
            left = reference[space + "_coeff"]
            right = np.asarray(getattr(actual, space + "_coeff"))
            principal = np.linalg.svd(left.T @ metric @ right, compute_uv=False)
            np.testing.assert_allclose(principal, np.ones(left.shape[1]), atol=2e-8)
            np.testing.assert_allclose(reference[space + "_energy"], getattr(actual, space + "_energy"), atol=2e-8)
        rotation = reference["occupied_coeff"].T @ metric @ np.asarray(actual.occupied_coeff)
        for weight in ("target_weight", "partner_weight"):
            np.testing.assert_allclose(rotation.T @ reference[weight] @ rotation, getattr(actual, weight), atol=2e-8)
        screen_ref = domain_module._fragment_screen_space(water_mf, top, target)
        screen = dlno_base.build_weak_multipole_screen(common, static, target)
        np.testing.assert_allclose(screen.weights, screen_ref["weights"], atol=1e-12)
        np.testing.assert_allclose(screen.occupied_energy, screen_ref["occupied_energy"], atol=2e-8)
        np.testing.assert_allclose(screen.virtual_energy, screen_ref["virtual_energy"][0], atol=2e-8)


def test_ed_rejects_lost_central_boys_target(separated_water_dimer_mf):
    mf = separated_water_dimer_mf
    top = domain_module.build_domain_topology(
        mf, frozen=2, lo_type="boys", force_full_domains=True,
        thresholds=domain_module.DLNOThresholds(ed_pao=0.0, domain_pao=0.0),
    )
    domains = np.empty(len(top.frag_lolist), dtype=object)
    domains[:] = [np.array([0, 1, 2])] * len(domains)
    top = replace(top, extended_domain=domains, pao_center_domain=domains)
    rows = domain_module.tools.ao_index_by_atom(mf.mol, domains[0])
    projected = np.linalg.solve(top.s1e[np.ix_(rows, rows)], top.s1e[rows] @ top.target_coeff)
    norm = np.diag(projected.T @ top.s1e[np.ix_(rows, rows)] @ projected)
    target = int(np.argmin(norm))
    assert norm[target] < top.thresholds.metric_rank
    with pytest.raises(RuntimeError, match="central Boys target"):
        domain_module._build_fragment_domain_orbitals(mf, top, target)


def test_empty_screen_is_forced_strong_and_nonfinite_pairs_raise(separated_water_dimer_mf, monkeypatch):
    mf = separated_water_dimer_mf
    top = domain_module.build_domain_topology(mf, frozen=2, lo_type="boys")
    with monkeypatch.context() as patch:
        patch.setattr(domain_module, '_fragment_screen_space', lambda *args: None)
        pair, forced = domain_module._multipole_fragment_pairs(mf, top)
        assert np.all(forced)
        np.testing.assert_array_equal(pair, 0)
    with monkeypatch.context() as patch:
        patch.setattr(domain_module.static_multipole, "multipole_pair_energy_cross",
                      lambda *args, **kwargs: np.full((1, 1), np.nan))
        with pytest.raises(RuntimeError, match="nonfinite Boys screening"):
            domain_module._multipole_fragment_pairs(mf, top)
