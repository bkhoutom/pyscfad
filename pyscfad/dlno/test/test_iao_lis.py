import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad import gto, scf
from pyscfad.dlno.iao_lis import (
    _density_from_target_amplitude_block,
    _density_from_target_amplitudes,
    build_fragment_lis,
    build_iao_lis_fragment_static_selection,
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
    build_strong_ed_domain,
    rebuild_iao_mp2_common,
)


def test_target_amplitude_block_density_matches_dense_complex_contraction():
    rng = np.random.default_rng(1907)
    u = rng.normal(size=(2, 3, 5, 5)) + 1j * rng.normal(size=(2, 3, 5, 5))
    dense = _density_from_target_amplitudes(jnp.asarray(u))

    occupied = np.zeros((3, 3), dtype=complex)
    virtual = np.zeros((5, 5), dtype=complex)
    for start in range(0, 5, 2):
        stop = min(start + 2, 5)
        a_block = u[:, :, :, start:stop]
        b_block = u[:, :, start:stop, :].transpose(0, 1, 3, 2)
        contribution = _density_from_target_amplitude_block(
            jnp.asarray(a_block), jnp.asarray(b_block)
        )
        occupied += np.asarray(contribution.occupied)
        virtual += np.asarray(contribution.virtual)

    occupied = 0.5 * (occupied + occupied.T.conj())
    virtual = 0.5 * (virtual + virtual.T.conj())
    np.testing.assert_allclose(occupied, np.asarray(dense.occupied), atol=2e-12)
    np.testing.assert_allclose(virtual, np.asarray(dense.virtual), atol=2e-12)


def test_target_amplitude_block_density_validates_rank_and_shapes():
    with pytest.raises(ValueError, match="rank-4"):
        _density_from_target_amplitude_block(
            jnp.zeros((2, 3, 5)), jnp.zeros((2, 3, 5))
        )
    with pytest.raises(ValueError, match="identical"):
        _density_from_target_amplitude_block(
            jnp.zeros((2, 3, 5, 2)), jnp.zeros((2, 3, 4, 2))
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


def _dense_lov_density(lov, occupied_energy, virtual_energy, target):
    integrals = jnp.einsum("Lpa,Lrb->prab", lov, lov, optimize=True)
    denominator = (
        occupied_energy[:, None, None, None]
        + occupied_energy[None, :, None, None]
        - virtual_energy[None, None, :, None]
        - virtual_energy[None, None, None, :]
    )
    return target_conditioned_mp2_density_from_amplitudes(
        integrals / denominator, target
    )


@pytest.mark.parametrize("block_nvir", [1, 2, 3, 8])
def test_blocked_lov_density_matches_dense_amplitudes(block_nvir):
    rng = np.random.default_rng(291)
    naux, nocc, nvir, ntarget = 8, 4, 5, 3
    lov = jnp.asarray(rng.normal(scale=0.12, size=(naux, nocc, nvir)))
    e_occ = jnp.asarray(-np.linspace(1.5, 0.7, nocc))
    e_vir = jnp.asarray(np.linspace(0.2, 1.3, nvir))
    target = jnp.asarray(rng.normal(scale=0.3, size=(ntarget, nocc)))

    dense = _dense_lov_density(lov, e_occ, e_vir, target)
    blocked = strong_domain_mp2_density_from_lov(
        lov, e_occ, e_vir, target, block_nvir=block_nvir
    )
    np.testing.assert_allclose(blocked.occupied, dense.occupied, atol=3e-12)
    np.testing.assert_allclose(blocked.virtual, dense.virtual, atol=3e-12)


def _density_scalar(density, occupied_weight, virtual_weight):
    return (
        jnp.sum(occupied_weight * density.occupied)
        + jnp.sum(virtual_weight * density.virtual)
    )


def _symmetric_weight(rng, size):
    weight = rng.normal(size=(size, size))
    return jnp.asarray(0.5 * (weight + weight.T))


def test_blocked_lov_density_value_and_grad_matches_dense_reference():
    rng = np.random.default_rng(977)
    naux, nocc, nvir, ntarget = 6, 3, 5, 2
    arguments = tuple(map(jnp.asarray, (
        rng.normal(scale=0.12, size=(naux, nocc, nvir)),
        -np.linspace(1.4, 0.6, nocc),
        np.linspace(0.2, 1.1, nvir),
        rng.normal(scale=0.3, size=(ntarget, nocc)),
    )))
    occupied_weight = _symmetric_weight(rng, nocc)
    virtual_weight = _symmetric_weight(rng, nvir)

    def dense_scalar(*inputs):
        return _density_scalar(
            _dense_lov_density(*inputs), occupied_weight, virtual_weight
        )

    def blocked_scalar(*inputs):
        return _density_scalar(
            strong_domain_mp2_density_from_lov(
                *inputs, block_nvir=2
            ), occupied_weight, virtual_weight
        )

    dense_value, dense_gradient = jax.value_and_grad(
        dense_scalar, argnums=(0, 1, 2, 3)
    )(*arguments)
    blocked_value, blocked_gradient = jax.value_and_grad(
        blocked_scalar, argnums=(0, 1, 2, 3)
    )(*arguments)
    np.testing.assert_allclose(blocked_value, dense_value, rtol=3e-10, atol=3e-11)
    for actual, expected in zip(blocked_gradient, dense_gradient):
        np.testing.assert_allclose(actual, expected, rtol=3e-10, atol=3e-11)


def test_blocked_lov_density_combined_directional_finite_difference():
    rng = np.random.default_rng(978)
    naux, nocc, nvir, ntarget = 6, 3, 5, 2
    arguments = tuple(map(jnp.asarray, (
        rng.normal(scale=0.12, size=(naux, nocc, nvir)),
        -np.linspace(1.4, 0.6, nocc),
        np.linspace(0.2, 1.1, nvir),
        rng.normal(scale=0.3, size=(ntarget, nocc)),
    )))
    directions = tuple(map(jnp.asarray, (
        rng.normal(size=(naux, nocc, nvir)),
        rng.normal(size=nocc),
        rng.normal(size=nvir),
        rng.normal(size=(ntarget, nocc)),
    )))
    occupied_weight = _symmetric_weight(rng, nocc)
    virtual_weight = _symmetric_weight(rng, nvir)

    def scalar(*inputs):
        return _density_scalar(
            strong_domain_mp2_density_from_lov(
                *inputs, block_nvir=2
            ), occupied_weight, virtual_weight
        )

    gradient = jax.grad(scalar, argnums=(0, 1, 2, 3))(*arguments)
    analytic = sum(jnp.vdot(value, direction)
                   for value, direction in zip(gradient, directions))
    step = 2e-5
    finite_difference = (
        scalar(*(value + step * direction
                 for value, direction in zip(arguments, directions)))
        - scalar(*(value - step * direction
                   for value, direction in zip(arguments, directions)))
    ) / (2.0 * step)
    np.testing.assert_allclose(
        analytic, finite_difference, rtol=5e-7, atol=5e-9
    )


def test_blocked_lov_density_handles_zero_sizes_and_validates_blocks():
    zero_target = strong_domain_mp2_density_from_lov(
        jnp.zeros((2, 3, 4)), jnp.arange(3.0), jnp.arange(4.0),
        jnp.zeros((0, 3)), block_nvir=2,
    )
    assert zero_target.occupied.shape == (3, 3)
    assert zero_target.virtual.shape == (4, 4)
    np.testing.assert_array_equal(zero_target.occupied, 0)
    np.testing.assert_array_equal(zero_target.virtual, 0)

    zero_occupied = strong_domain_mp2_density_from_lov(
        jnp.zeros((2, 0, 4)), jnp.zeros((0,)), jnp.arange(4.0),
        jnp.zeros((2, 0)), block_nvir=2,
    )
    assert zero_occupied.occupied.shape == (0, 0)
    assert zero_occupied.virtual.shape == (4, 4)
    np.testing.assert_array_equal(zero_occupied.virtual, 0)

    zero_virtual = strong_domain_mp2_density_from_lov(
        jnp.zeros((2, 3, 0)), jnp.arange(3.0), jnp.zeros((0,)),
        jnp.zeros((2, 3)), block_nvir=2,
    )
    assert zero_virtual.occupied.shape == (3, 3)
    assert zero_virtual.virtual.shape == (0, 0)
    np.testing.assert_array_equal(zero_virtual.occupied, 0)

    inputs = (
        jnp.zeros((2, 3, 4)),
        -jnp.arange(3.0) - 1.0,
        jnp.arange(4.0) + 0.2,
        jnp.zeros((2, 3)),
    )
    for max_memory_mb in (0.0, -1.0):
        with pytest.raises(ValueError, match="max_memory_mb"):
            strong_domain_mp2_density_from_lov(
                *inputs, max_memory_mb=max_memory_mb
            )
    for block_nvir in (0, -1, 1.5):
        with pytest.raises(ValueError, match="block_nvir"):
            strong_domain_mp2_density_from_lov(*inputs, block_nvir=block_nvir)

    clamped = strong_domain_mp2_density_from_lov(*inputs, block_nvir=99)
    reference = _dense_lov_density(*inputs)
    np.testing.assert_allclose(clamped.occupied, reference.occupied)
    np.testing.assert_allclose(clamped.virtual, reference.virtual)


def test_blocked_lov_density_uses_nested_scans_without_full_amplitudes():
    rng = np.random.default_rng(979)
    ntarget, naux, nocc, nvir, block_nvir = 2, 7, 3, 7, 2
    arguments = tuple(map(jnp.asarray, (
        rng.normal(size=(naux, nocc, nvir)),
        -np.linspace(1.3, 0.7, nocc),
        np.linspace(0.2, 1.2, nvir),
        rng.normal(size=(ntarget, nocc)),
    )))
    occupied_weight = _symmetric_weight(rng, nocc)
    virtual_weight = _symmetric_weight(rng, nvir)

    def scalar(*inputs):
        return _density_scalar(
            strong_domain_mp2_density_from_lov(
                *inputs, block_nvir=block_nvir
            ), occupied_weight, virtual_weight
        )

    def nested_values(value):
        if hasattr(value, "jaxpr") and not hasattr(value, "eqns"):
            value = value.jaxpr
        if hasattr(value, "eqns"):
            yield value
            for equation in value.eqns:
                yield from nested_values(equation.params)
        elif isinstance(value, dict):
            for item in value.values():
                yield from nested_values(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from nested_values(item)

    forward_jaxpr = jax.make_jaxpr(scalar)(*arguments)

    def jaxpr_equations(value):
        if hasattr(value, "jaxpr") and not hasattr(value, "eqns"):
            value = value.jaxpr
        return value.eqns

    nblock = (nvir + block_nvir - 1) // block_nvir
    outer_scan, = [
        equation
        for equation in jaxpr_equations(forward_jaxpr)
        if equation.primitive.name == "scan"
        and equation.params["length"] == nblock
    ]
    outer_remat, = [
        equation
        for equation in jaxpr_equations(outer_scan.params["jaxpr"])
        if equation.primitive.name.startswith("remat")
    ]
    inner_scans = [
        equation
        for nested in nested_values(outer_remat.params)
        for equation in jaxpr_equations(nested)
        if equation.primitive.name == "scan"
        and equation.params["length"] == nocc
    ]
    assert len(inner_scans) == 1
    inner_scan = inner_scans[0]
    inner_body = jaxpr_equations(inner_scan.params["jaxpr"])
    assert len(inner_body) == 1
    assert inner_body[0].primitive.name.startswith("remat")

    gradient_jaxpr = jax.make_jaxpr(
        jax.grad(scalar, argnums=(0, 1, 2, 3))
    )(*arguments)
    all_array_shapes = {
        tuple(variable.aval.shape)
        for jaxpr in nested_values(gradient_jaxpr)
        for equation in jaxpr.eqns
        for variable in (*equation.invars, *equation.outvars)
        if hasattr(variable, "aval") and hasattr(variable.aval, "shape")
    }
    forbidden_amplitude_shapes = (
        (ntarget, nocc, nvir, nvir),
        (nocc, nocc, nvir, nvir),
    )
    for forbidden_shape in forbidden_amplitude_shapes:
        assert not any(
            shape == forbidden_shape
            or (len(shape) > len(forbidden_shape)
                and shape[-len(forbidden_shape):] == forbidden_shape)
            for shape in all_array_shapes
        )

    _, pullback = jax.vjp(scalar, *arguments)
    residuals = [
        value
        for name in ("args_res", "opaque_residuals")
        for value in getattr(pullback, name, ())
        if hasattr(value, "shape")
    ]
    lov_residuals = [
        value for value in residuals if tuple(value.shape) == arguments[0].shape
    ]
    assert len(lov_residuals) <= 1
    assert not any(tuple(value.shape) == (nocc, naux, nvir)
                   for value in residuals)
    assert not any(
        value.ndim >= 5
        and tuple(value.shape[-4:]) == (ntarget, nocc, nvir, block_nvir)
        for value in residuals
    )
    assert not any(
        value.ndim >= 4
        and tuple(value.shape[-4:]) == (nocc, ntarget, nvir, block_nvir)
        for value in residuals
    )
    pullback(jnp.ones(()))


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


def _water_dimer_mf():
    mol = gto.Mole(
        atom="""
        O  0.0000000000  0.0000000000  0.0000000000
        H  0.0000000000 -0.7570000000  0.5870000000
        H  0.0000000000  0.7570000000  0.5870000000
        O  0.0000000000  0.0000000000  8.0000000000
        H  0.0000000000 -0.7570000000  8.5870000000
        H  0.0000000000  0.7570000000  8.5870000000
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


def test_fragment_static_selection_matches_serial_builder(monkeypatch):
    mf = _water_dimer_mf()
    topology = build_iao_fragment_topology(
        mf,
        thresholds=_full_domain_thresholds(),
        pair_energy_model="all",
        force_full_domains=True,
    )
    assert len(topology.frag_lolist) == 2
    mp2_static = build_iao_mp2_static_selections(mf, topology)
    common = rebuild_iao_mp2_common(mf, mp2_static)
    thresholds = {
        "thresh_occ": 2e-4,
        "thresh_vir": 3e-5,
        "internal_rank_threshold": 3e-7,
    }
    serial = build_iao_lis_static_selections(
        mf,
        mp2_static,
        common=common,
        **thresholds,
    )
    domains = tuple(
        build_strong_ed_domain(common, mp2_static, fragment_index)
        for fragment_index in range(len(mp2_static.fragments))
    )

    def fail_if_domain_is_rebuilt(*args, **kwargs):
        del args, kwargs
        raise AssertionError("the supplied ED domain must be reused")

    monkeypatch.setattr(
        "pyscfad.dlno.iao_lis.build_strong_ed_domain",
        fail_if_domain_is_rebuilt,
    )
    independent = tuple(
        build_iao_lis_fragment_static_selection(
            mf,
            mp2_static,
            fragment_index,
            common=common,
            domain=domains[fragment_index],
            **thresholds,
        )
        for fragment_index in range(len(mp2_static.fragments))
    )

    assert len(independent) == len(serial.fragments)
    for actual, expected in zip(independent, serial.fragments):
        assert actual.fragment_index == expected.fragment_index
        assert actual.full_occupied_space == expected.full_occupied_space
        assert actual.full_virtual_space == expected.full_virtual_space
        np.testing.assert_array_equal(
            actual.internal_occ_keep, expected.internal_occ_keep
        )
        np.testing.assert_array_equal(
            actual.internal_vir_keep, expected.internal_vir_keep
        )
        np.testing.assert_array_equal(
            actual.occupied_lno_keep, expected.occupied_lno_keep
        )
        np.testing.assert_array_equal(
            actual.virtual_lno_keep, expected.virtual_lno_keep
        )
