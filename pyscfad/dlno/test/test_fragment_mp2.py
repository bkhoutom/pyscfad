import numpy as np
import pytest
import jax
import jax.numpy as jnp

import pyscfad.dlno.fragment_mp2 as fragment_mp2_module
from pyscfad.dlno.fragment_mp2 import (
    build_fragment_occupied_data,
    fragment_pair_energy_from_lov,
    fragment_pair_energy_from_lov_jax,
    fragment_pair_energy_from_ovov,
    partition_fragment_pair_energies,
)


def test_jax_blocked_lov_energy_and_directional_derivative():
    rng = np.random.default_rng(120)
    naux, nocc, nvir = 7, 4, 5
    lov = rng.normal(scale=0.12, size=(naux, nocc, nvir))
    e_occ = -np.linspace(1.4, 0.6, nocc)
    e_vir = np.linspace(0.15, 1.1, nvir)
    target_factor = rng.normal(scale=0.4, size=(3, nocc))
    partner_factor = rng.normal(scale=0.4, size=(5, nocc))
    partner_weight = partner_factor.T @ partner_factor

    directions = (
        rng.normal(size=lov.shape),
        rng.normal(size=e_occ.shape),
        rng.normal(size=e_vir.shape),
        rng.normal(size=target_factor.shape),
        rng.normal(size=partner_weight.shape),
    )
    directions = directions[:-1] + (
        0.5 * (directions[-1] + directions[-1].T),
    )

    def energy(lov_, e_occ_, e_vir_, target_factor_, partner_weight_):
        return fragment_pair_energy_from_lov_jax(
            lov_,
            e_occ_,
            e_vir_,
            target_factor_,
            partner_weight_,
            block_nvir=2,
        ).total

    arguments = tuple(map(jnp.asarray, (
        lov, e_occ, e_vir, target_factor, partner_weight
    )))
    value, gradients = jax.value_and_grad(
        energy, argnums=(0, 1, 2, 3, 4)
    )(*arguments)

    target_weight = target_factor.T @ target_factor
    reference = fragment_pair_energy_from_lov(
        lov,
        e_occ,
        e_vir,
        target_weight,
        partner_weight,
        target_factor=target_factor,
        block_nvir=2,
    )
    np.testing.assert_allclose(value, reference.total, atol=2e-12)

    analytic = sum(
        np.vdot(np.asarray(gradient), direction).real
        for gradient, direction in zip(gradients, directions)
    )
    step = 2e-5
    plus = tuple(
        jnp.asarray(argument + step * direction)
        for argument, direction in zip(
            (lov, e_occ, e_vir, target_factor, partner_weight), directions
        )
    )
    minus = tuple(
        jnp.asarray(argument - step * direction)
        for argument, direction in zip(
            (lov, e_occ, e_vir, target_factor, partner_weight), directions
        )
    )
    finite_difference = float((energy(*plus) - energy(*minus)) / (2 * step))
    np.testing.assert_allclose(
        analytic, finite_difference, rtol=2e-7, atol=2e-9
    )


def _iao_fragment_problem(seed=11):
    """Return a synthetic nonorthogonal-AO IAO/occupied partition."""

    rng = np.random.default_rng(seed)
    nao = 11
    niao = 8
    nocc = 5

    metric_factor = rng.normal(size=(nao, nao))
    s1e = metric_factor.T @ metric_factor + np.eye(nao)
    se, su = np.linalg.eigh(s1e)
    s_inv_sqrt = (su / np.sqrt(se)) @ su.T

    q, _ = np.linalg.qr(rng.normal(size=(nao, nao)))
    iao_coeff = s_inv_sqrt @ q[:, :niao]

    occ_in_iao, _ = np.linalg.qr(rng.normal(size=(niao, nocc)))
    occ_coeff = iao_coeff @ occ_in_iao
    frag_lolist = (
        np.arange(0, 3),
        np.arange(3, 6),
        np.arange(6, 8),
    )
    return rng, occ_coeff, iao_coeff, frag_lolist, s1e


def _dense_mp2_problem(nocc, seed=29):
    rng = np.random.default_rng(seed)
    nvir = 4
    naux = 9
    lov = rng.normal(scale=0.2, size=(naux, nocc, nvir))
    ovov = np.einsum("Lia,Ljb->iajb", lov, lov)
    e_occ = -np.linspace(1.4, 0.6, nocc)
    e_vir = np.linspace(0.2, 1.3, nvir)
    return ovov, e_occ, e_vir


def _canonical_mp2_components(ovov, e_occ, e_vir):
    eia = e_occ[:, None] - e_vir[None, :]
    denominator = eia[:, None, :, None] + eia[None, :, None, :]
    integrals = ovov.transpose(0, 2, 1, 3)
    amplitudes = integrals / denominator
    direct = np.einsum("ijab,ijab->", amplitudes, integrals)
    exchange = np.einsum("ijab,ijba->", amplitudes, integrals)
    opposite_spin = direct
    same_spin = direct - exchange
    return opposite_spin + same_spin, opposite_spin, same_spin


def test_fragment_weights_resolve_occupied_identity():
    _, occ, iao, fragments, s1e = _iao_fragment_problem()
    data = build_fragment_occupied_data(occ, iao, fragments, s1e)

    nocc = occ.shape[1]
    weight_sum = sum(fragment.occupied_weight for fragment in data)
    assert np.linalg.norm(weight_sum - np.eye(nocc)) < 1e-12

    for fragment, indices in zip(data, fragments):
        af = iao[:, indices]
        mf = af.T @ s1e @ occ
        assert fragment.iao_coeff.shape == (occ.shape[0], len(indices))
        assert fragment.iao_occ_overlap.shape == (len(indices), nocc)
        assert fragment.occupied_projection.shape == (occ.shape[0], len(indices))
        assert fragment.occupied_weight.shape == (nocc, nocc)
        assert np.allclose(fragment.iao_occ_overlap, mf)
        assert np.allclose(fragment.occupied_projection, occ @ mf.T)
        assert np.allclose(fragment.occupied_weight, mf.T @ mf)


def test_fragment_weights_are_invariant_to_internal_iao_rotations():
    rng, occ, iao, fragments, s1e = _iao_fragment_problem()
    reference = build_fragment_occupied_data(occ, iao, fragments, s1e)

    rotated_iao = iao.copy()
    for indices in fragments:
        rotation, _ = np.linalg.qr(rng.normal(size=(len(indices), len(indices))))
        rotated_iao[:, indices] = iao[:, indices] @ rotation
    rotated = build_fragment_occupied_data(occ, rotated_iao, fragments, s1e)

    for ref, new in zip(reference, rotated):
        assert np.allclose(ref.occupied_weight, new.occupied_weight, atol=1e-12)
        assert np.allclose(ref.singular_values, new.singular_values, atol=1e-12)


def test_unweighted_occupied_range_projectors_overcount():
    _, occ, iao, fragments, s1e = _iao_fragment_problem()
    data = build_fragment_occupied_data(occ, iao, fragments, s1e)

    unweighted_sum = sum(
        fragment.right_singular_vectors
        @ fragment.right_singular_vectors.conj().T
        for fragment in data
    )
    weighted_sum = sum(fragment.occupied_weight for fragment in data)

    assert np.allclose(weighted_sum, np.eye(occ.shape[1]), atol=1e-12)
    assert np.trace(unweighted_sum) > occ.shape[1] + 1.0
    assert np.linalg.norm(unweighted_sum - np.eye(occ.shape[1])) > 0.5


def test_two_sided_fragment_pair_sum_equals_canonical_dense_mp2():
    _, occ, iao, fragments, s1e = _iao_fragment_problem()
    data = build_fragment_occupied_data(occ, iao, fragments, s1e)
    weights = np.stack([fragment.occupied_weight for fragment in data])
    ovov, e_occ, e_vir = _dense_mp2_problem(occ.shape[1])

    pair = fragment_pair_energy_from_ovov(ovov, e_occ, e_vir, weights)
    reference = _canonical_mp2_components(ovov, e_occ, e_vir)

    assert np.allclose(pair.total, pair.total.T, atol=1e-12)
    assert np.allclose(pair.opposite_spin, pair.opposite_spin.T, atol=1e-12)
    assert np.allclose(pair.same_spin, pair.same_spin.T, atol=1e-12)
    assert np.allclose(pair.summed().total, reference[0], atol=1e-12)
    assert np.allclose(pair.summed().opposite_spin, reference[1], atol=1e-12)
    assert np.allclose(pair.summed().same_spin, reference[2], atol=1e-12)

    identity = np.eye(occ.shape[1])
    canonical = fragment_pair_energy_from_ovov(
        ovov, e_occ, e_vir, identity, identity
    )
    assert np.allclose(canonical.total, reference[0], atol=1e-12)


@pytest.mark.parametrize("block_nvir", [1, 2, 3, 8])
def test_blocked_lov_kernel_matches_dense_two_sided_energy(block_nvir):
    _, occ, iao, fragments, s1e = _iao_fragment_problem()
    data = build_fragment_occupied_data(occ, iao, fragments, s1e)
    weights = np.stack([fragment.occupied_weight for fragment in data])
    rng = np.random.default_rng(91)
    nocc = occ.shape[1]
    nvir = 5
    lov = rng.normal(scale=0.15, size=(11, nocc, nvir))
    ovov = np.einsum("Lia,Ljb->iajb", lov, lov)
    e_occ = -np.linspace(1.5, 0.5, nocc)
    e_vir = np.linspace(0.1, 1.2, nvir)

    dense = fragment_pair_energy_from_ovov(
        ovov, e_occ, e_vir, weights, weights[::-1]
    )
    blocked = fragment_pair_energy_from_lov(
        lov,
        e_occ,
        e_vir,
        weights,
        weights[::-1],
        block_nvir=block_nvir,
    )
    for component in ("total", "opposite_spin", "same_spin"):
        np.testing.assert_allclose(
            getattr(blocked, component),
            getattr(dense, component),
            atol=2e-12,
        )


def test_blocked_lov_kernel_single_asymmetric_weights_and_auto_block():
    _, occ, iao, fragments, s1e = _iao_fragment_problem()
    data = build_fragment_occupied_data(occ, iao, fragments, s1e)
    left = data[0].occupied_weight
    right = data[1].occupied_weight + data[2].occupied_weight
    ovov, e_occ, e_vir = _dense_mp2_problem(occ.shape[1], seed=63)

    # Factor the positive-semidefinite synthetic OVOV tensor back through its
    # original deterministic DF construction.
    rng = np.random.default_rng(63)
    lov = rng.normal(scale=0.2, size=(9, occ.shape[1], 4))
    np.testing.assert_allclose(
        ovov, np.einsum("Lia,Ljb->iajb", lov, lov), atol=1e-14
    )
    dense = fragment_pair_energy_from_ovov(
        ovov, e_occ, e_vir, left, right
    )
    blocked = fragment_pair_energy_from_lov(
        lov, e_occ, e_vir, left, right, max_memory_mb=1e-3
    )
    assert np.ndim(blocked.total) == 0
    for component in ("total", "opposite_spin", "same_spin"):
        np.testing.assert_allclose(
            getattr(blocked, component),
            getattr(dense, component),
            atol=2e-12,
        )


def test_blocked_lov_kernel_single_complex_hermitian_weights():
    rng = np.random.default_rng(116)
    naux, nocc, nvir = 7, 4, 5
    lov = (
        rng.normal(scale=0.1, size=(naux, nocc, nvir))
        + 1j * rng.normal(scale=0.1, size=(naux, nocc, nvir))
    )
    left_factor = rng.normal(size=(2, nocc)) + 1j * rng.normal(
        size=(2, nocc)
    )
    right_factor = rng.normal(size=(3, nocc)) + 1j * rng.normal(
        size=(3, nocc)
    )
    left = left_factor.conj().T @ left_factor
    right = right_factor.conj().T @ right_factor
    e_occ = -np.linspace(1.3, 0.6, nocc)
    e_vir = np.linspace(0.2, 1.1, nvir)
    ovov = np.einsum("Lia,Ljb->iajb", lov, lov)

    dense = fragment_pair_energy_from_ovov(
        ovov, e_occ, e_vir, left, right
    )
    blocked = fragment_pair_energy_from_lov(
        lov, e_occ, e_vir, left, right, block_nvir=2
    )
    factored = fragment_pair_energy_from_lov(
        lov,
        e_occ,
        e_vir,
        left,
        right,
        target_factor=left_factor,
        block_nvir=2,
    )
    for component in ("total", "opposite_spin", "same_spin"):
        np.testing.assert_allclose(
            getattr(blocked, component),
            getattr(dense, component),
            atol=2e-12,
        )
        np.testing.assert_allclose(
            getattr(factored, component),
            getattr(dense, component),
            atol=2e-12,
        )


def test_target_factor_directional_derivative_matches_dense_weight_path():
    rng = np.random.default_rng(118)
    naux, nocc, nvir = 8, 4, 5
    lov = rng.normal(scale=0.1, size=(naux, nocc, nvir))
    target_factor = rng.normal(size=(2, nocc))
    direction = rng.normal(size=target_factor.shape)
    partner_factor = rng.normal(size=(3, nocc))
    partner_weight = partner_factor.T @ partner_factor
    e_occ = -np.linspace(1.4, 0.5, nocc)
    e_vir = np.linspace(0.1, 1.2, nvir)

    def energy(factor, use_factor):
        target_weight = factor.T @ factor
        kwargs = {"target_factor": factor} if use_factor else {}
        return fragment_pair_energy_from_lov(
            lov,
            e_occ,
            e_vir,
            target_weight,
            partner_weight,
            block_nvir=2,
            **kwargs,
        ).total

    step = 2e-6
    factor_derivative = (
        energy(target_factor + step * direction, True)
        - energy(target_factor - step * direction, True)
    ) / (2.0 * step)
    dense_derivative = (
        energy(target_factor + step * direction, False)
        - energy(target_factor - step * direction, False)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        factor_derivative, dense_derivative, rtol=2e-8, atol=2e-10
    )


def test_blocked_lov_kernel_builds_only_upper_triangle_df_blocks(monkeypatch):
    rng = np.random.default_rng(117)
    naux, nocc, nvir = 8, 4, 7
    lov = rng.normal(size=(naux, nocc, nvir))
    weight_factor = rng.normal(size=(2, nocc))
    left = weight_factor.T @ weight_factor
    right = np.eye(nocc)
    e_occ = -np.linspace(1.5, 0.5, nocc)
    e_vir = np.linspace(0.1, 1.0, nvir)

    real_einsum = fragment_mp2_module.np.einsum
    auxiliary_contractions = []

    def counted_einsum(subscripts, *operands, **kwargs):
        if "L" in subscripts:
            auxiliary_contractions.append(subscripts)
        return real_einsum(subscripts, *operands, **kwargs)

    monkeypatch.setattr(fragment_mp2_module.np, "einsum", counted_einsum)
    fragment_pair_energy_from_lov(
        lov, e_occ, e_vir, left, right, block_nvir=3
    )

    # Three virtual blocks require 3 * (3 + 1) / 2 DF reconstructions.
    assert auxiliary_contractions == ["Lia,Ljb->iajb"] * 6


def test_stacked_weights_avoid_all_at_once_scaling_eight_contractions(
    monkeypatch,
):
    rng = np.random.default_rng(119)
    naux, nocc, nvir = 8, 4, 5
    lov = rng.normal(size=(naux, nocc, nvir))
    left_factor = rng.normal(size=(2, 2, nocc))
    right_factor = rng.normal(size=(3, 3, nocc))
    left = np.einsum("fxp,fxq->fpq", left_factor, left_factor)
    right = np.einsum("gxr,gxs->grs", right_factor, right_factor)
    e_occ = -np.linspace(1.5, 0.5, nocc)
    e_vir = np.linspace(0.1, 1.0, nvir)

    real_einsum = fragment_mp2_module.np.einsum
    ovov = real_einsum("Lia,Ljb->iajb", lov, lov)
    dense = fragment_pair_energy_from_ovov(
        ovov, e_occ, e_vir, left, right
    )
    dense_swapped = fragment_pair_energy_from_ovov(
        ovov, e_occ, e_vir, right, left
    )
    contractions = []

    def recorded_einsum(subscripts, *operands, **kwargs):
        contractions.append(subscripts)
        return real_einsum(subscripts, *operands, **kwargs)

    monkeypatch.setattr(fragment_mp2_module.np, "einsum", recorded_einsum)
    blocked = fragment_pair_energy_from_lov(
        lov, e_occ, e_vir, left, right, block_nvir=2
    )
    blocked_swapped = fragment_pair_energy_from_lov(
        lov, e_occ, e_vir, right, left, block_nvir=2
    )

    assert "fpq,grs,prab,qsab->fg" not in contractions
    assert "fpq,grs,prab,sqab->fg" not in contractions
    assert "pq,prab->qrab" in contractions
    assert "qrab,qsab->rs" in contractions
    assert "qrab,sqab->rs" in contractions
    assert "prab,rs->psab" in contractions
    assert "psab,qsab->pq" in contractions
    assert "psab,sqab->pq" in contractions
    for component in ("total", "opposite_spin", "same_spin"):
        np.testing.assert_allclose(
            getattr(blocked, component),
            getattr(dense, component),
            atol=2e-10,
        )
        np.testing.assert_allclose(
            getattr(blocked_swapped, component),
            getattr(dense_swapped, component),
            atol=2e-10,
        )


def test_arbitrary_exact_strong_weak_mask_closes_identically():
    rng, occ, iao, fragments, s1e = _iao_fragment_problem()
    data = build_fragment_occupied_data(occ, iao, fragments, s1e)
    weights = np.stack([fragment.occupied_weight for fragment in data])
    ovov, e_occ, e_vir = _dense_mp2_problem(occ.shape[1])
    pair = fragment_pair_energy_from_ovov(ovov, e_occ, e_vir, weights)

    upper = np.triu(rng.random(pair.total.shape) > 0.5, k=1)
    strong_mask = upper | upper.T
    np.fill_diagonal(strong_mask, True)
    partition = partition_fragment_pair_energies(pair, strong_mask)

    for component in ("total", "opposite_spin", "same_spin"):
        original = getattr(pair, component)
        strong = getattr(partition.strong, component)
        weak = getattr(partition.weak, component)
        assert np.array_equal(strong + weak, original)
        assert np.allclose(np.sum(strong) + np.sum(weak), np.sum(original))


def test_fragment_algebra_validates_incompatible_shapes_and_partitions():
    _, occ, iao, fragments, s1e = _iao_fragment_problem()
    with pytest.raises(ValueError, match="assign every IAO"):
        build_fragment_occupied_data(occ, iao, fragments[:-1], s1e)
    with pytest.raises(ValueError, match="more than one fragment"):
        bad_fragments = (fragments[0], fragments[0], np.arange(3, iao.shape[1]))
        build_fragment_occupied_data(occ, iao, bad_fragments, s1e)
    with pytest.raises(ValueError, match="s1e"):
        build_fragment_occupied_data(occ, iao, fragments, s1e[:-1, :-1])

    weights = np.stack(
        [
            fragment.occupied_weight
            for fragment in build_fragment_occupied_data(
                occ, iao, fragments, s1e
            )
        ]
    )
    ovov, e_occ, e_vir = _dense_mp2_problem(occ.shape[1])
    with pytest.raises(ValueError, match="ovov"):
        fragment_pair_energy_from_ovov(
            ovov[:, :, :-1, :], e_occ, e_vir, weights
        )
    pair = fragment_pair_energy_from_ovov(ovov, e_occ, e_vir, weights)
    with pytest.raises(ValueError, match="same shape"):
        partition_fragment_pair_energies(pair, np.ones((1,), dtype=bool))
