import numpy as np
import pytest

from pyscfad import gto, scf
from pyscfad.dlno import iao_mp2
from pyscfad.mp import dfmp2
from pyscfad.dlno.fragment_mp2 import fragment_pair_energy_from_ovov
from pyscfad.dlno.iao_mp2 import (
    IAOFragmentMP2Thresholds,
    build_iao_fragment_topology,
    evaluate_iao_fragment_mp2,
    kernel,
)


def _molecule(atom, basis="sto-3g"):
    mol = gto.Mole()
    mol.atom = atom
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.verbose = 0
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


@pytest.fixture(scope="module")
def ethane_mf():
    mol = _molecule(
        """
        C  0.00  0.00 -0.77
        C  0.00  0.00  0.77
        H  0.00  1.02 -1.13
        H  0.88 -0.51 -1.13
        H -0.88 -0.51 -1.13
        H  0.00 -1.02  1.13
        H  0.88  0.51  1.13
        H -0.88  0.51  1.13
        """
    )
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    return mf


@pytest.fixture(scope="module")
def separated_water_dimer_mf():
    mol = _molecule(
        """
        O 0.00  0.00 0.00
        H 0.00  0.75 0.58
        H 0.00 -0.75 0.58
        O 0.00  0.00 8.00
        H 0.00  0.75 8.58
        H 0.00 -0.75 8.58
        """
    )
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    return mf


def _canonical_components(mf):
    pt = dfmp2.MP2(mf)
    eris = pt.ao2mo()
    nocc = pt.nocc
    nvir = pt.nmo - nocc
    lov = np.asarray(
        pt.loop_ao2mo(eris.mo_coeff, nocc, with_t2=False)
    ).reshape(-1, nocc, nvir)
    ovov = np.einsum("Lia,Ljb->iajb", lov, lov, optimize=True)
    identity = np.eye(nocc)
    return fragment_pair_energy_from_ovov(
        ovov,
        np.asarray(eris.mo_energy[:nocc]),
        np.asarray(eris.mo_energy[nocc:]),
        identity,
        identity,
    )


def _full_domain_thresholds():
    return IAOFragmentMP2Thresholds(
        pao_norm=1e-10,
        domain_pao=0.0,
        ed_pao=0.0,
        occupied_weight=1e-12,
    )


def test_full_domain_iao_fragment_mp2_recovers_canonical_dfmp2(ethane_mf):
    result = kernel(
        ethane_mf,
        thresholds=_full_domain_thresholds(),
        pair_energy_model="all",
        force_full_domains=True,
    )
    reference = _canonical_components(ethane_mf)

    assert abs(result.e_corr - reference.total) < 5e-9
    assert abs(result.e_strong_os - reference.opposite_spin) < 5e-9
    assert abs(result.e_strong_ss - reference.same_spin) < 5e-9
    assert result.e_weak_multipole_os == 0.0
    assert sum(fragment.target_weight_trace for fragment in result.fragments) \
        == pytest.approx(ethane_mf.mol.nelectron // 2, abs=1e-10)

    nocc = ethane_mf.mol.nelectron // 2
    nvir = ethane_mf.mol.nao - nocc
    for fragment in result.fragments:
        assert fragment.n_domain_occ == nocc
        assert fragment.n_domain_vir == nvir
        assert fragment.n_domain_ao == ethane_mf.mol.nao


def test_full_domain_identity_with_frozen_carbon_cores(ethane_mf):
    frozen = 2
    result = kernel(
        ethane_mf,
        frozen=frozen,
        thresholds=_full_domain_thresholds(),
        pair_energy_model="all",
        force_full_domains=True,
    )
    reference, _ = dfmp2.MP2(
        ethane_mf, frozen=frozen
    ).kernel(with_t2=False)
    assert abs(result.e_corr - reference) < 5e-9


def test_full_domain_identity_with_a_frozen_virtual(ethane_mf):
    frozen = [ethane_mf.mo_coeff.shape[1] - 1]
    result = kernel(
        ethane_mf,
        frozen=frozen,
        thresholds=_full_domain_thresholds(),
        pair_energy_model="all",
        force_full_domains=True,
    )
    reference_solver = dfmp2.MP2(ethane_mf, frozen=frozen)
    reference, _ = reference_solver.kernel(with_t2=False)
    assert abs(result.e_corr - reference) < 5e-9
    assert all(
        fragment.n_domain_vir == reference_solver.nmo - reference_solver.nocc
        for fragment in result.fragments
    )


def test_fragment_bp_topology_is_invariant_to_iao_block_rotation(ethane_mf):
    reference = build_iao_fragment_topology(
        ethane_mf, pair_energy_model="all"
    )
    rng = np.random.default_rng(4)
    rotated_iao = reference.iao_coeff.copy()
    for indices in reference.frag_lolist:
        rotation, _ = np.linalg.qr(rng.normal(size=(len(indices), len(indices))))
        rotated_iao[:, indices] = reference.iao_coeff[:, indices] @ rotation

    rotated = build_iao_fragment_topology(
        ethane_mf,
        iao_coeff=rotated_iao,
        frag_lolist=reference.frag_lolist,
        frag_atmlist=reference.frag_atmlist,
        pair_energy_model="all",
    )
    for field in (
        "compact_bp_domain",
        "primary_bp_domain",
        "primary_domain",
        "tight_bp_domain",
        "extended_domain",
    ):
        left = getattr(reference, field)
        right = getattr(rotated, field)
        for left_atoms, right_atoms in zip(left, right):
            np.testing.assert_array_equal(left_atoms, right_atoms)
    for left, right in zip(
        reference.fragment_occupied_data,
        rotated.fragment_occupied_data,
    ):
        np.testing.assert_allclose(
            left.occupied_weight, right.occupied_weight, atol=1e-12
        )

    reference_energy = evaluate_iao_fragment_mp2(ethane_mf, reference)
    rotated_energy = evaluate_iao_fragment_mp2(ethane_mf, rotated)
    assert rotated_energy.e_corr == pytest.approx(
        reference_energy.e_corr, abs=1e-10
    )


def test_weak_multipole_energy_is_invariant_to_iao_block_rotation(
    separated_water_dimer_mf,
):
    thresholds = IAOFragmentMP2Thresholds(pair_energy=1e-4)
    reference = build_iao_fragment_topology(
        separated_water_dimer_mf,
        thresholds=thresholds,
        pair_energy_model="multipole",
    )
    assert not reference.strong_mask[0, 1]
    assert reference.weak_pair_energy[0, 1] != 0.0

    rng = np.random.default_rng(17)
    rotated_iao = reference.iao_coeff.copy()
    for indices in reference.frag_lolist:
        rotation, _ = np.linalg.qr(rng.normal(
            size=(len(indices), len(indices))
        ))
        rotated_iao[:, indices] = reference.iao_coeff[:, indices] @ rotation

    rotated = build_iao_fragment_topology(
        separated_water_dimer_mf,
        iao_coeff=rotated_iao,
        frag_lolist=reference.frag_lolist,
        frag_atmlist=reference.frag_atmlist,
        thresholds=thresholds,
        pair_energy_model="multipole",
    )
    np.testing.assert_allclose(
        rotated.pair_energy, reference.pair_energy, atol=1e-14, rtol=1e-12
    )
    np.testing.assert_allclose(
        rotated.weak_pair_energy,
        reference.weak_pair_energy,
        atol=1e-14,
        rtol=1e-12,
    )
    np.testing.assert_array_equal(rotated.strong_mask, reference.strong_mask)

    reference_energy = evaluate_iao_fragment_mp2(
        separated_water_dimer_mf, reference
    )
    rotated_energy = evaluate_iao_fragment_mp2(
        separated_water_dimer_mf, rotated
    )
    assert rotated_energy.e_weak_multipole_os == pytest.approx(
        reference_energy.e_weak_multipole_os, abs=1e-14, rel=1e-12
    )
    assert rotated_energy.e_corr == pytest.approx(
        reference_energy.e_corr, abs=1e-12
    )


def test_adjacent_fragments_are_forced_strong_before_multipole(ethane_mf):
    topology = build_iao_fragment_topology(
        ethane_mf, pair_energy_model="multipole"
    )
    assert topology.forced_strong_mask[0, 1]
    assert topology.strong_mask[0, 1]
    assert np.all(np.isfinite(topology.pair_energy))


def test_separated_fragment_weak_pair_and_tight_limit(separated_water_dimer_mf):
    loose = kernel(
        separated_water_dimer_mf,
        thresholds=IAOFragmentMP2Thresholds(pair_energy=1e-4),
        pair_energy_model="multipole",
    )
    assert not loose.topology.strong_mask[0, 1]
    assert np.isfinite(loose.e_weak_multipole_os)
    assert loose.e_weak_multipole_os != 0.0
    assert all(
        fragment.n_domain_ao < separated_water_dimer_mf.mol.nao
        for fragment in loose.fragments
    )

    tight = kernel(
        separated_water_dimer_mf,
        thresholds=IAOFragmentMP2Thresholds(pair_energy=0.0),
        pair_energy_model="multipole",
    )
    reference = _canonical_components(separated_water_dimer_mf).total
    assert np.all(tight.topology.strong_mask)
    assert tight.e_weak_multipole_os == 0.0
    assert abs(tight.e_corr - reference) < 1e-8
    assert abs(tight.e_corr - reference) < abs(loose.e_corr - reference)


def test_local_auxiliary_domain_energy_is_global_df_storage_independent(
    separated_water_dimer_mf,
):
    original = separated_water_dimer_mf.with_df.incore
    results = []
    try:
        for incore in (True, False):
            separated_water_dimer_mf.with_df.incore = incore
            results.append(kernel(
                separated_water_dimer_mf,
                thresholds=IAOFragmentMP2Thresholds(pair_energy=1e-4),
                pair_energy_model="multipole",
            ))
    finally:
        separated_water_dimer_mf.with_df.incore = original

    assert results[0].e_strong == pytest.approx(results[1].e_strong, abs=1e-12)
    assert results[0].e_corr == pytest.approx(results[1].e_corr, abs=1e-12)


def test_fragment_lov_transform_is_integral_direct_only(
    ethane_mf, monkeypatch,
):
    thresholds = _full_domain_thresholds()
    topology = build_iao_fragment_topology(
        ethane_mf,
        thresholds=thresholds,
        pair_energy_model="all",
        force_full_domains=True,
    )
    calls = []
    real_get_local_lov = iao_mp2.lno_base.get_local_Lov

    def checked_get_local_lov(*args, **kwargs):
        calls.append(kwargs.get("integral_direct"))
        return real_get_local_lov(*args, **kwargs)

    monkeypatch.setattr(
        iao_mp2.lno_base, "get_local_Lov", checked_get_local_lov
    )
    result = evaluate_iao_fragment_mp2(ethane_mf, topology)

    assert np.isfinite(result.e_corr)
    assert calls == [True] * len(topology.frag_lolist)


def test_post_topology_timing_is_nonnegative_and_fragment_consistent(
    ethane_mf,
):
    result = kernel(
        ethane_mf,
        thresholds=_full_domain_thresholds(),
        pair_energy_model="all",
        force_full_domains=True,
    )
    fields = (
        "ed_orbital_seconds",
        "local_ri_lov_seconds",
        "weighted_ed_mp2_seconds",
        "weak_bookkeeping_seconds",
    )

    assert result.timing is not None
    assert result.timing.total_seconds > 0.0
    for fragment in result.fragments:
        assert fragment.timing is not None
        values = [getattr(fragment.timing, field) for field in fields]
        assert all(value >= 0.0 for value in values)
        assert fragment.timing.total_seconds == pytest.approx(sum(values))

    for field in fields:
        fragment_sum = sum(
            getattr(fragment.timing, field)
            for fragment in result.fragments
        )
        assert getattr(result.timing, field) == pytest.approx(fragment_sum)
    assert result.timing.total_seconds == pytest.approx(sum(
        getattr(result.timing, field) for field in fields
    ))


def test_exact_pair_model_uses_nagy_scaled_opposite_spin_increment(
    separated_water_dimer_mf,
):
    topology = build_iao_fragment_topology(
        separated_water_dimer_mf,
        thresholds=IAOFragmentMP2Thresholds(pair_energy=1e-4),
        pair_energy_model="exact",
    )
    pt = dfmp2.MP2(separated_water_dimer_mf)
    eris = pt.ao2mo()
    nocc = pt.nocc
    nvir = pt.nmo - nocc
    lov = np.asarray(
        pt.loop_ao2mo(eris.mo_coeff, nocc, with_t2=False)
    ).reshape(-1, nocc, nvir)
    ovov = np.einsum("Lia,Ljb->iajb", lov, lov, optimize=True)
    weights = np.stack([
        fragment.occupied_weight
        for fragment in topology.fragment_occupied_data
    ])
    directed = fragment_pair_energy_from_ovov(
        ovov,
        np.asarray(eris.mo_energy[:nocc]),
        np.asarray(eris.mo_energy[nocc:]),
        weights,
    )
    # Nagy's -8 pair-increment convention is four times the standard
    # unordered SCS-OS contribution carried by this fragment decomposition.
    expected = 4.0 * (
        directed.opposite_spin + directed.opposite_spin.T
    )
    np.fill_diagonal(expected, 4.0 * np.diag(directed.opposite_spin))
    np.testing.assert_allclose(topology.pair_energy, expected, atol=1e-12)


def test_default_span_cutoff_does_not_promote_tiny_iao_tails_to_occupied():
    mol = _molecule(
        """
        O 0.00  0.00 0.00
        H 0.00  0.75 0.58
        H 0.00 -0.75 0.58
        O 0.00  0.00 4.00
        H 0.00  0.75 4.58
        H 0.00 -0.75 4.58
        """
    )
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    result = kernel(mf, pair_energy_model="multipole")
    assert np.isfinite(result.e_corr)
    assert all(fragment.n_domain_vir > 0 for fragment in result.fragments)


def test_unknown_pair_model_is_rejected_even_when_all_pairs_are_forced(
    ethane_mf,
):
    thresholds = IAOFragmentMP2Thresholds(pair_energy=0.0)
    with pytest.raises(ValueError, match="unknown pair_energy_model"):
        build_iao_fragment_topology(
            ethane_mf,
            thresholds=thresholds,
            pair_energy_model="not-a-model",
        )
