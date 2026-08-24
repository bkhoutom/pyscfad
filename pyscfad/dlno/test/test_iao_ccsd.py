"""Small-system identities for the IAO-DLNO-CCSD(T) construction.

These tests deliberately use water dimers whose IAO-MP2 graphs exercise the
two important limits for the CC driver: a compact all-strong graph and a
separated graph with two strong self domains plus one unordered weak pair.
"""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad import config_update, gto, scf
from pyscfad.cc import dfccsd
from pyscfad.dlno.ccsd import DLNOCCSD
from pyscfad.dlno.iao_ccsd import (
    _assemble_iao_dlno_correlation,
    build_iao_dlno_ccsd_static_selections,
)
from pyscfad.dlno.iao_mp2 import (
    IAOFragmentMP2Thresholds,
    build_iao_fragment_topology,
    evaluate_iao_fragment_mp2,
)
from pyscfad.dlno.iao_mp2_grad import (
    build_iao_mp2_static_selections,
    build_strong_ed_domain,
    correlation_energy,
    rebuild_iao_mp2_common,
)
from pyscfad.dlno.iao_lis import (
    build_fragment_lis,
    build_iao_lis_static_selections,
    strong_domain_prescreen,
)
from pyscfad.lno.ccsd import RCCSD as ImpurityRCCSD
from pyscfad.lno.ccsd import mp2_fragment_energy
from pyscfad.mp import dfmp2


def test_correction_bookkeeping_and_cotangent_seeds_are_unique():
    """The full strong+weak IAO-MP2 scalar is added once, not per fragment."""

    e_cc = jnp.asarray([-0.31, -0.27, -0.19])
    e_t = jnp.asarray([-0.012, -0.009, -0.006])
    e_mp2_lis = jnp.asarray([-0.22, -0.18, -0.14])
    e_iao_mp2 = jnp.asarray(-0.61)

    value = _assemble_iao_dlno_correlation(
        e_cc, e_t, e_mp2_lis, e_iao_mp2
    )
    expected = (
        np.sum(np.asarray(e_cc))
        + np.sum(np.asarray(e_t))
        - np.sum(np.asarray(e_mp2_lis))
        + float(e_iao_mp2)
    )
    np.testing.assert_allclose(value, expected, atol=0.0, rtol=0.0)

    gradients = jax.grad(
        _assemble_iao_dlno_correlation,
        argnums=(0, 1, 2, 3),
    )(e_cc, e_t, e_mp2_lis, e_iao_mp2)
    np.testing.assert_array_equal(gradients[0], np.ones(3))
    np.testing.assert_array_equal(gradients[1], np.ones(3))
    np.testing.assert_array_equal(gradients[2], -np.ones(3))
    np.testing.assert_array_equal(gradients[3], np.asarray(1.0))


def test_dlno_ccsd_public_api_has_no_legacy_mp2_correction_switches():
    constructor = inspect.signature(DLNOCCSD).parameters
    parameters = inspect.signature(DLNOCCSD.value_and_grad).parameters
    for obsolete in (
        "include_mp2_correction",
        "mp2_correction_method",
        "mp2_correction_scope",
        "sos_c_os",
    ):
        assert obsolete not in parameters
        assert obsolete not in constructor
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in inspect.signature(DLNOCCSD).parameters.values()
    )
    assert constructor["thresh"].default is None
    assert constructor["thresh_occ"].default == 1e-4
    assert constructor["thresh_vir"].default == 1e-5
    assert parameters["thresh_occ"].default == 1e-4
    assert parameters["thresh_vir"].default == 1e-5


def test_empty_fragment_excitation_space_is_a_zero_cc_increment():
    """An intentionally over-truncated LIS must not enter a zero-size CC solve."""

    mol = gto.Mole(
        atom="He 0 0 0",
        unit="Bohr",
        basis="cc-pvdz",
        verbose=0,
        max_memory=1000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = scf.RHF(mol).density_fit(auxbasis="weigend")
    mf.conv_tol = 1e-12
    mf.kernel()

    default_local = DLNOCCSD(mf)
    assert default_local.thresh_occ == 1e-4
    assert default_local.thresh_vir == 1e-5

    local = DLNOCCSD(mf, thresh=1.0)
    assert local.thresh_occ == 1.0
    assert local.thresh_vir == 1.0
    energy = local.kernel(thresh_occ=1.0, thresh_vir=1.0)
    assert np.isfinite(float(energy))
    assert local.result.lis_virtual == (0,)
    np.testing.assert_allclose(local.result.e_ccsd, 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        local.result.e_mp2_lis, 0.0, atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(
        local.e_corr_pt2, local.result.e_iao_mp2, atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(
        local.e_corr_pt2_correction,
        local.result.e_iao_mp2 - local.result.e_mp2_lis,
        atol=0.0,
        rtol=0.0,
    )
    with pytest.raises(AttributeError, match="no domain-only"):
        _ = local.e_corr_pt2_domain
    with pytest.raises(AttributeError, match="already includes"):
        local.e_corr_pt2corrected(-0.1)

    def build_he_mf(mol_):
        mf_ = scf.RHF(mol_).density_fit(auxbasis="weigend")
        mf_.conv_tol = 1e-12
        mf_.conv_tol_grad = 1e-10
        mf_.kernel()
        return mf_

    total, mol_bar = DLNOCCSD.value_and_grad(
        mol,
        build_mf=build_he_mf,
        thresh_occ=1.0,
        thresh_vir=1.0,
    )
    assert np.isfinite(float(total))
    np.testing.assert_allclose(
        np.asarray(mol_bar.coords), np.zeros((1, 3)), atol=1e-12, rtol=0.0
    )


def test_triples_and_dcsd_are_rejected_together():
    mol = _water_dimer(separated=False)
    with pytest.raises(ValueError, match="not defined for DCSD"):
        DLNOCCSD.value_and_grad(
            mol,
            build_mf=_build_mf,
            ccsd_t=True,
            dcsd=True,
        )


def _water_dimer(*, separated):
    if separated:
        atom = """
        O  0.0000000000  0.0000000000  0.0000000000
        H  0.0000000000 -0.7570000000  0.5870000000
        H  0.0000000000  0.7570000000  0.5870000000
        O  0.0000000000  0.0000000000  8.0000000000
        H  0.0000000000 -0.7570000000  8.5870000000
        H  0.0000000000  0.7570000000  8.5870000000
        """
    else:
        atom = """
        O  -1.485163346097 -0.114724564047  0.000000000000
        H  -1.868415346097  0.762298435953  0.000000000000
        H  -0.533833346097  0.040507435953  0.000000000000
        O   1.416468653903  0.111264435953  0.000000000000
        H   1.746241653903 -0.373945564047 -0.758561000000
        H   1.746241653903 -0.373945564047  0.758561000000
        """
    mol = gto.Mole(
        atom=atom,
        unit="Angstrom",
        basis="sto-3g",
        verbose=0,
        max_memory=1000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def _build_mf(mol):
    mf = scf.RHF(mol).density_fit(auxbasis="weigend")
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    return mf


def _build_local_problem(*, separated, thresholds=None,
                         pair_energy_model="multipole",
                         force_full_domains=False):
    mol = _water_dimer(separated=separated)
    mf = _build_mf(mol)
    if thresholds is None:
        thresholds = IAOFragmentMP2Thresholds(pair_energy=1e-4)
    topology = build_iao_fragment_topology(
        mf,
        thresholds=thresholds,
        pair_energy_model=pair_energy_model,
        force_full_domains=force_full_domains,
    )
    static = build_iao_mp2_static_selections(mf, topology)
    common = rebuild_iao_mp2_common(mf, static)
    return mol, mf, topology, static, common


@pytest.mark.parametrize(
    "separated, expected_mask, expected_atoms, expected_domain_shape",
    [
        (
            False,
            [[True, True], [True, True]],
            [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
            (14, 10, 4),
        ),
        (
            True,
            [[True, False], [False, True]],
            [[0, 1, 2], [3, 4, 5]],
            (7, 5, 2),
        ),
    ],
)
def test_compact_and_far_water_define_expected_iao_strong_domains(
    separated,
    expected_mask,
    expected_atoms,
    expected_domain_shape,
):
    """The CC LIS must start from exactly the validated IAO-MP2 ED graph."""

    mol, mf, _, static, common = _build_local_problem(
        separated=separated
    )

    assert mol.nao == 14
    assert np.count_nonzero(np.asarray(mf.mo_occ) > 0.0) == 10
    assert common.iao_coeff.shape == (14, 14)
    assert [indices.size for indices in static.frag_lolist] == [7, 7]
    np.testing.assert_array_equal(static.strong_mask, expected_mask)

    for fragment_index, (fragment, atoms) in enumerate(
        zip(static.fragments, expected_atoms)
    ):
        np.testing.assert_array_equal(fragment.extended_atoms, atoms)
        domain = build_strong_ed_domain(common, static, fragment_index)
        nao_domain, nocc_domain, nvir_domain = expected_domain_shape
        assert domain.occupied_coeff.shape == (nao_domain, nocc_domain)
        assert domain.virtual_coeff.shape == (nao_domain, nvir_domain)
        assert domain.target_projection.shape[1] == nocc_domain

        prescreen = strong_domain_prescreen(
            common, static, fragment_index, domain=domain
        )
        np.testing.assert_array_equal(
            prescreen["extended_primary_domain"], atoms
        )
        np.testing.assert_allclose(
            prescreen["occ_prescreen_coeff"], domain.occupied_coeff,
            atol=0.0, rtol=0.0,
        )
        np.testing.assert_allclose(
            prescreen["vir_prescreen_coeff"], domain.virtual_coeff,
            atol=0.0, rtol=0.0,
        )
        np.testing.assert_allclose(
            prescreen["orbfragloc"],
            common.iao_coeff[:, static.frag_lolist[fragment_index]],
            atol=0.0, rtol=0.0,
        )
        expected_strong_iao = np.unique(np.concatenate([
            static.frag_lolist[int(partner)]
            for partner in fragment.strong_fragments
        ]))
        np.testing.assert_array_equal(
            prescreen["strong_lmo_indices"], expected_strong_iao
        )


def _full_domain_thresholds():
    return IAOFragmentMP2Thresholds(
        pao_norm=1e-10,
        domain_pao=0.0,
        ed_pao=0.0,
        occupied_weight=1e-12,
        pair_energy=0.0,
    )


def test_far_water_pt2_is_strong_energy_plus_one_weak_pair():
    """The molecular PT2 correction contains each weak pair exactly once."""

    _, mf, topology, static, _ = _build_local_problem(separated=True)
    assert np.count_nonzero(np.triu(~static.strong_mask, k=1)) == 1

    result = evaluate_iao_fragment_mp2(mf, topology)
    assert result.e_weak_multipole_os != 0.0
    assert result.e_corr == pytest.approx(
        result.e_strong + result.e_weak_multipole_os,
        abs=2e-14,
    )

    canonical, _ = dfmp2.MP2(mf).kernel(with_t2=False)
    # At 8 Angstrom the fourth-order weak multipole model is already within
    # sub-microhartree accuracy of canonical DF-MP2.
    assert abs(result.e_corr - canonical) < 1e-6

    tight_topology = build_iao_fragment_topology(
        mf,
        thresholds=_full_domain_thresholds(),
        pair_energy_model="all",
        force_full_domains=True,
    )
    tight = evaluate_iao_fragment_mp2(mf, tight_topology)
    assert tight.e_weak_multipole_os == 0.0
    np.testing.assert_allclose(tight.e_corr, canonical, atol=2e-9, rtol=0.0)


def test_far_water_lis_rejects_distant_iao_projection_tails():
    """Tiny cross-monomer IAO projections must not become internal orbitals."""

    _, mf, _, mp2_static, common = _build_local_problem(separated=True)
    lis_static = build_iao_lis_static_selections(
        mf,
        mp2_static,
        common=common,
        thresh_occ=1e-3,
        thresh_vir=1e-3,
    )
    for fragment_index in range(2):
        lis = build_fragment_lis(
            mf, common, lis_static, fragment_index
        )
        assert lis.n_internal_occ == 5
        assert lis.n_internal_vir == 2
        assert lis.active_occupied_coeff.shape[1] == 5
        assert lis.active_virtual_coeff.shape[1] == 2


def test_full_domain_iao_pt2_exactly_cancels_lis_mp2_sum():
    """At zero LIS thresholds both MP2 branches recover the same energy."""

    _, mf, _, mp2_static, common = _build_local_problem(
        separated=False,
        thresholds=_full_domain_thresholds(),
        pair_energy_model="all",
        force_full_domains=True,
    )
    lis_static = build_iao_lis_static_selections(
        mf,
        mp2_static,
        common=common,
        thresh_occ=0.0,
        thresh_vir=0.0,
    )

    fragment_mp2 = []
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 0.0))
    nvir = int(mf.mo_occ.size - nocc)
    for fragment_index in range(len(mp2_static.fragments)):
        lis = build_fragment_lis(
            mf, common, lis_static, fragment_index
        )
        assert lis.active_occupied_coeff.shape[1] == nocc
        assert lis.active_virtual_coeff.shape[1] == nvir
        assert lis.frozen.size == 0
        np.testing.assert_allclose(
            lis.occupied_projector, np.eye(nocc), atol=2e-10, rtol=0.0
        )
        np.testing.assert_allclose(
            lis.virtual_projector, np.eye(nvir), atol=2e-10, rtol=0.0
        )

        # Evaluate precisely the conventional full-spin MP2 fragment energy
        # that the CC driver subtracts in this LIS.  No CC iterations are
        # needed for this limiting-identity test.
        cc = ImpurityRCCSD(
            mf, mo_coeff=lis.mo_coeff, frozen=lis.frozen
        )
        eris = cc.ao2mo(fockao=common.fock)
        _, _, t2 = cc.init_amps(eris=eris)
        projector = (
            np.asarray(lis.fragment_iao_coeff).T
            @ np.asarray(common.s1e)
            @ np.asarray(lis.active_occupied_coeff)
        )
        fragment_mp2.append(float(
            mp2_fragment_energy(eris, t2, projector)
        ))

    e_mp2_lis = sum(fragment_mp2)
    e_iao_mp2 = float(correlation_energy(mf, mp2_static))
    # This is the cancellation in
    #   sum_F(E_CC,F - E_MP2,LIS,F) + E_IAO-MP2.
    np.testing.assert_allclose(
        e_mp2_lis, e_iao_mp2, atol=2e-9, rtol=0.0
    )

    canonical, _ = dfmp2.MP2(mf).kernel(with_t2=False)
    np.testing.assert_allclose(
        e_iao_mp2, canonical, atol=2e-9, rtol=0.0
    )


def test_compact_full_domain_matches_canonical_dfccsd_t_energy():
    """The public energy solver recovers canonical DF-CCSD(T)."""

    mol = _water_dimer(separated=False)
    mf = _build_mf(mol)
    local = DLNOCCSD(
        mf,
        thresholds=_full_domain_thresholds(),
        pair_energy_model="all",
        force_full_domains=True,
    )
    local.thresh_occ = 0.0
    local.thresh_vir = 0.0
    local.ccsd_t = True
    local_total = mf.e_tot + local.kernel()

    canonical = dfccsd.RCCSD(mf)
    canonical.kernel()
    canonical_total = canonical.e_tot + canonical.ccsd_t()
    np.testing.assert_allclose(
        local_total, canonical_total, atol=2e-7, rtol=0.0
    )

    # In this limit the local-MP2 correction and the sum of the LIS-MP2
    # subtractions are the same scalar, so only CCSD(T) remains.
    np.testing.assert_allclose(
        local.result.e_iao_mp2,
        local.result.e_mp2_lis,
        atol=2e-9,
        rtol=0.0,
    )


def _canonical_dfccsd_total(mol):
    mf = _build_mf(mol)
    cc = dfccsd.RCCSD(mf)
    cc.kernel()
    return cc.e_tot


def _canonical_dfccsd_t_total(mol):
    mf = _build_mf(mol)
    cc = dfccsd.RCCSD(mf)
    cc.kernel()
    return cc.e_tot + cc.ccsd_t()


def test_compact_full_domain_matches_canonical_dfccsd_high_cost():
    """Zero LIS thresholds recover canonical DF-CCSD energy and gradient."""

    mol = _water_dimer(separated=False)
    thresholds = _full_domain_thresholds()

    local_energy, local_bar = DLNOCCSD.value_and_grad(
        mol,
        build_mf=_build_mf,
        thresholds=thresholds,
        pair_energy_model="all",
        force_full_domains=True,
        thresh_occ=0.0,
        thresh_vir=0.0,
        ccsd_t=False,
    )
    with (
        config_update("pyscfad_scf_implicit_diff", True),
        config_update("pyscfad_scf_first_order_custom", False),
        config_update("pyscfad_ccsd_implicit_diff", True),
    ):
        canonical_energy, canonical_bar = jax.value_and_grad(
            _canonical_dfccsd_total
        )(mol)

    np.testing.assert_allclose(
        local_energy, canonical_energy, atol=2e-7, rtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(local_bar.coords),
        np.asarray(canonical_bar.coords),
        atol=5e-7,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(local_bar.coords), axis=0),
        np.zeros(3),
        atol=2e-9,
        rtol=0.0,
    )


def test_compact_full_domain_matches_canonical_dfccsd_t_gradient_high_cost():
    """The complete local triples gradient recovers canonical DF-CCSD(T)."""

    mol = _water_dimer(separated=False)
    thresholds = _full_domain_thresholds()
    local_energy, local_bar = DLNOCCSD.value_and_grad(
        mol,
        build_mf=_build_mf,
        thresholds=thresholds,
        pair_energy_model="all",
        force_full_domains=True,
        thresh_occ=0.0,
        thresh_vir=0.0,
        ccsd_t=True,
    )
    with (
        config_update("pyscfad_scf_implicit_diff", True),
        config_update("pyscfad_scf_first_order_custom", False),
        config_update("pyscfad_ccsd_implicit_diff", True),
    ):
        canonical_energy, canonical_bar = jax.value_and_grad(
            _canonical_dfccsd_t_total
        )(mol)

    np.testing.assert_allclose(
        local_energy, canonical_energy, atol=2e-7, rtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(local_bar.coords),
        np.asarray(canonical_bar.coords),
        atol=6e-7,
        rtol=3e-5,
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(local_bar.coords), axis=0),
        np.zeros(3),
        atol=2e-9,
        rtol=0.0,
    )


def test_far_water_mixed_strong_weak_gradient_smoke_high_cost():
    """The CC gradient path handles two strong EDs and one weak pair."""

    mol = _water_dimer(separated=True)
    mf = _build_mf(mol)
    thresholds = IAOFragmentMP2Thresholds(pair_energy=1e-4)
    static = build_iao_dlno_ccsd_static_selections(
        mf,
        thresholds=thresholds,
        pair_energy_model="multipole",
        thresh_occ=1e-3,
        thresh_vir=1e-3,
    )
    np.testing.assert_array_equal(
        static.mp2_static.strong_mask,
        [[True, False], [False, True]],
    )
    assert np.count_nonzero(
        np.triu(~static.mp2_static.strong_mask, k=1)
    ) == 1

    energy, mol_bar = DLNOCCSD.value_and_grad(
        mol,
        build_mf=_build_mf,
        thresholds=thresholds,
        pair_energy_model="multipole",
        thresh_occ=1e-3,
        thresh_vir=1e-3,
        ccsd_t=False,
        static_selections=static,
    )
    gradient = np.asarray(mol_bar.coords)

    assert np.isfinite(float(energy))
    assert np.all(np.isfinite(gradient))
    np.testing.assert_allclose(
        np.sum(gradient, axis=0), np.zeros(3), atol=2e-9, rtol=0.0
    )
    # Translating the second monomer along its separation coordinate must
    # have a finite, nonzero derivative in this mixed strong/weak topology.
    intermonomer_z = float(np.sum(gradient[3:, 2]))
    assert 1e-6 < abs(intermonomer_z) < 1e-3
