import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad import gto, scf
from pyscfad.dlno.iao_lis import (
    build_fragment_lis,
    build_iao_lis_static_selections,
    strong_domain_mp2_density_from_lov,
    strong_domain_prescreen,
    target_conditioned_mp2_density_from_amplitudes,
)
from pyscfad.dlno.iao_mp2 import (
    IAOFragmentMP2Thresholds,
    build_iao_fragment_topology,
)
from pyscfad.dlno.iao_mp2_grad import (
    build_iao_mp2_static_selections,
    rebuild_iao_mp2_common,
)


def _manual_ie_density(target_amplitudes):
    ntarget, nocc, nvir, _ = target_amplitudes.shape
    dmoo = np.zeros((nocc, nocc))
    dmvv = np.zeros((nvir, nvir))
    for target in range(ntarget):
        t = target_amplitudes[target].transpose(1, 0, 2)
        dmvv += t.reshape(nvir, -1) @ t.reshape(nvir, -1).T
        dmvv -= 0.5 * np.einsum("ajc,cjb->ab", t, t)
        dmvv += t.reshape(-1, nvir).T @ t.reshape(-1, nvir)
        dmvv -= 0.5 * np.einsum("cja,bjc->ab", t, t)

        dmoo += np.einsum("aib,ajb->ij", t, t)
        dmoo -= 0.5 * np.einsum("aib,bja->ij", t, t)
        dmoo += np.einsum("bia,bja->ij", t, t)
        dmoo -= 0.5 * np.einsum("bia,ajb->ij", t, t)
    return 0.5 * (dmoo + dmoo.T), 0.5 * (dmvv + dmvv.T)


def test_target_conditioned_density_matches_ie_contractions_and_rotation():
    rng = np.random.default_rng(814)
    nocc, nvir, ntarget = 4, 5, 3
    amplitudes = rng.normal(scale=0.08, size=(nocc, nocc, nvir, nvir))
    target = rng.normal(scale=0.4, size=(ntarget, nocc))

    density = target_conditioned_mp2_density_from_amplitudes(
        jnp.asarray(amplitudes), jnp.asarray(target)
    )
    target_amplitudes = np.einsum(
        "Ip,prab->Irab", target, amplitudes
    )
    dmoo_ref, dmvv_ref = _manual_ie_density(target_amplitudes)
    np.testing.assert_allclose(density.occupied, dmoo_ref, atol=2e-13)
    np.testing.assert_allclose(density.virtual, dmvv_ref, atol=2e-13)

    rotation, _ = np.linalg.qr(rng.normal(size=(ntarget, ntarget)))
    rotated = target_conditioned_mp2_density_from_amplitudes(
        jnp.asarray(amplitudes), jnp.asarray(rotation @ target)
    )
    np.testing.assert_allclose(
        rotated.occupied, density.occupied, atol=2e-13
    )
    np.testing.assert_allclose(
        rotated.virtual, density.virtual, atol=2e-13
    )


def test_lov_density_matches_dense_amplitudes_and_has_finite_adjoint():
    rng = np.random.default_rng(291)
    naux, nocc, nvir, ntarget = 8, 4, 5, 3
    lov = rng.normal(scale=0.12, size=(naux, nocc, nvir))
    e_occ = -np.linspace(1.5, 0.7, nocc)
    e_vir = np.linspace(0.2, 1.3, nvir)
    target = rng.normal(scale=0.3, size=(ntarget, nocc))

    integrals = np.einsum("Lpa,Lrb->prab", lov, lov)
    denominator = (
        e_occ[:, None, None, None]
        + e_occ[None, :, None, None]
        - e_vir[None, None, :, None]
        - e_vir[None, None, None, :]
    )
    dense = target_conditioned_mp2_density_from_amplitudes(
        jnp.asarray(integrals / denominator), jnp.asarray(target)
    )
    blocked = strong_domain_mp2_density_from_lov(
        jnp.asarray(lov),
        jnp.asarray(e_occ),
        jnp.asarray(e_vir),
        jnp.asarray(target),
    )
    np.testing.assert_allclose(blocked.occupied, dense.occupied, atol=3e-13)
    np.testing.assert_allclose(blocked.virtual, dense.virtual, atol=3e-13)

    def scalar(lov_):
        density = strong_domain_mp2_density_from_lov(
            lov_, e_occ, e_vir, target
        )
        return jnp.trace(density.occupied) + jnp.trace(density.virtual)

    gradient = jax.grad(scalar)(jnp.asarray(lov))
    assert np.all(np.isfinite(np.asarray(gradient)))

    direction = rng.normal(size=lov.shape)
    analytic = float(jnp.vdot(gradient, direction))
    step = 2e-5
    finite_difference = float(
        (scalar(lov + step * direction) - scalar(lov - step * direction))
        / (2.0 * step)
    )
    np.testing.assert_allclose(
        analytic, finite_difference, atol=2e-8, rtol=3e-7
    )


def _water_mf(frozen=None):
    del frozen
    mol = gto.Mole(
        atom="""
        O  0.0000000000  0.0000000000  0.0000000000
        H  0.0000000000 -0.7570000000  0.5870000000
        H  0.0000000000  0.7570000000  0.5870000000
        """,
        basis="sto-3g",
        verbose=0,
        max_memory=1000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    return mf


def _full_domain_thresholds():
    return IAOFragmentMP2Thresholds(
        pao_norm=1e-10,
        domain_pao=0.0,
        ed_pao=0.0,
        occupied_weight=1e-12,
    )


@pytest.mark.parametrize("frozen", [None, 1])
def test_zero_lno_threshold_recovers_full_active_hf_spaces(frozen):
    mf = _water_mf(frozen)
    topology = build_iao_fragment_topology(
        mf,
        frozen=frozen,
        thresholds=_full_domain_thresholds(),
        pair_energy_model="all",
        force_full_domains=True,
    )
    assert len(topology.frag_lolist) == 1
    mp2_static = build_iao_mp2_static_selections(mf, topology)
    common = rebuild_iao_mp2_common(mf, mp2_static)
    lis_static = build_iao_lis_static_selections(
        mf,
        mp2_static,
        common=common,
        thresh_occ=0.0,
        thresh_vir=0.0,
    )
    result = build_fragment_lis(mf, common, lis_static, 0)

    nocc_active = len(mp2_static.active_occ_indices)
    nvir_active = len(mp2_static.active_vir_indices)
    assert result.active_occupied_coeff.shape[1] == nocc_active
    assert result.active_virtual_coeff.shape[1] == nvir_active
    assert result.mo_coeff.shape == mf.mo_coeff.shape

    overlap = np.asarray(mf.get_ovlp())
    coeff = np.asarray(result.mo_coeff)
    np.testing.assert_allclose(
        coeff.T @ overlap @ coeff,
        np.eye(coeff.shape[1]),
        atol=2e-10,
    )
    np.testing.assert_allclose(
        result.occupied_projector,
        np.eye(nocc_active),
        atol=2e-12,
    )
    np.testing.assert_allclose(
        result.virtual_projector,
        np.eye(nvir_active),
        atol=2e-12,
    )

    raw_overlap = (
        result.fragment_iao_coeff.T
        @ overlap
        @ result.active_occupied_coeff
    )
    projected_overlap = (
        result.fragment_occupied_anchor.T
        @ overlap
        @ result.active_occupied_coeff
    )
    np.testing.assert_allclose(raw_overlap, projected_overlap, atol=2e-11)
    assert np.all(np.isfinite(np.asarray(result.density_occupied_ed)))
    assert np.all(np.isfinite(np.asarray(result.density_virtual_ed)))

    prescreen = strong_domain_prescreen(
        common, mp2_static, 0, domain=result.domain
    )
    np.testing.assert_array_equal(
        prescreen["strong_lmo_indices"],
        np.arange(common.iao_coeff.shape[1]),
    )
    assert prescreen["orbfragloc"] is not None
    assert prescreen["occ_prescreen_coeff"].shape[1] == nocc_active
    assert prescreen["vir_prescreen_coeff"].shape[1] == nvir_active
