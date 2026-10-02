"""Finite Boys ED packets and their three-tensor cotangent replay."""

import jax
import numpy
import pytest

from pyscfad import gto, numpy as jnp, scf
from pyscfad.dlno import _selection, dlno_base, domain as dlno_domain
from pyscfad.dlno_stc.adjoint import pullback_inputs
from pyscfad.dlno_stc import domain as stc_domain, prepare as finite_prepare
from pyscfad.dlno_stc.prepare import (
    build_local_strong_ed_domain,
    prepare_finite_inputs,
)
from pyscfad.lno import df as lno_df


@pytest.fixture(scope="module")
def finite_water_dimer():
    mol = gto.Mole()
    mol.atom = (
        "O 0 0 0; H 0 0.75 0.58; H 0 -0.75 0.58; "
        "O 0 0 8; H 0 0.75 8.58; H 0 -0.75 8.58"
    )
    mol.unit = "Angstrom"
    mol.basis = "sto-3g"
    mol.verbose = 0
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-10
    mf.kernel()
    topology = dlno_domain.build_domain_topology(
        mf, frozen=2, lo_type="boys"
    )
    static = _selection.build_domain_selections(mf, topology)
    common = dlno_base.rebuild_domain_data(mf, static)
    return mf, common, static


def test_finite_packet_uses_one_orthonormal_local_frame(finite_water_dimer):
    mf, common, static = finite_water_dimer
    assert len(static.fragments) == 8
    for fragment_index, fragment in enumerate(static.fragments):
        assert len(fragment.extended_ao_indices) < mf.mol.nao
        domain = build_local_strong_ed_domain(
            common, static, fragment_index
        )
        packet = prepare_finite_inputs(mf, common, static, fragment_index)
        co = numpy.asarray(domain.occupied_coeff)
        cv = numpy.asarray(domain.virtual_coeff)
        ao = numpy.asarray(fragment.extended_ao_indices)
        fock22 = numpy.asarray(common.fock)[numpy.ix_(ao, ao)]
        nocc, nvir = co.shape[1], cv.shape[1]
        assert packet["foo"].shape == (nocc, nocc)
        assert packet["fvv"].shape == (nvir, nvir)
        assert packet["B"].shape[1:] == (nocc, nvir)
        numpy.testing.assert_allclose(packet["foo"], co.T @ fock22 @ co,
                                      atol=1e-9)
        numpy.testing.assert_allclose(packet["fvv"], cv.T @ fock22 @ cv,
                                      atol=1e-9)
        overlap = numpy.asarray(common.s1e)
        overlap22 = overlap[numpy.ix_(ao, ao)]
        numpy.testing.assert_allclose(co.T @ overlap22 @ co, numpy.eye(nocc),
                                      atol=1e-9)
        numpy.testing.assert_allclose(cv.T @ overlap22 @ cv, numpy.eye(nvir),
                                      atol=1e-9)
        numpy.testing.assert_allclose(co.T @ overlap22 @ cv,
                                      numpy.zeros((nocc, nvir)), atol=1e-9)
        target_coeff = numpy.asarray(
            common.fragment_occupied_data[fragment_index].iao_coeff
        )
        projection = target_coeff.T @ overlap[:, ao] @ co
        numpy.testing.assert_allclose(domain.target_projection, projection,
                                      atol=1e-9)
        numpy.testing.assert_allclose(domain.target_weight,
                                      projection.T @ projection, atol=1e-9)
        reference_B = lno_df.get_local_Lov(
            mf, numpy.concatenate((co, cv), axis=1), nocc,
            fragment.extended_atoms, integral_direct=True,
        ).reshape(packet["B"].shape)
        numpy.testing.assert_allclose(packet["B"], reference_B, atol=1e-9)


def test_finite_packet_retains_noncanonical_local_fock_blocks(finite_water_dimer):
    """A finite local frame must retain the Fock couplings STC needs."""
    mf, common, static = finite_water_dimer
    largest_offdiagonal = 0.0
    for fragment_index in range(len(static.fragments)):
        packet = prepare_finite_inputs(mf, common, static, fragment_index)
        for name in ("foo", "fvv"):
            block = numpy.asarray(packet[name])
            if block.shape[0] > 1:
                offdiagonal = block - numpy.diag(numpy.diag(block))
                largest_offdiagonal = max(
                    largest_offdiagonal, numpy.max(numpy.abs(offdiagonal))
                )
    assert largest_offdiagonal > 1e-5


def test_finite_virtual_anchors_are_fixed_pao_parent_columns(finite_water_dimer):
    mf, common, static = finite_water_dimer
    fragment = static.fragments[0]
    anchors = finite_prepare.select_virtual_anchor_columns(common, static, 0)
    assert len(anchors) == len(fragment.strong_virtual.metric_keep)
    assert len(set(anchors)) == len(anchors)
    assert set(anchors) <= set(fragment.strong_virtual.parent_columns)
    expected = prepare_finite_inputs(mf, common, static, 0)
    fixed = prepare_finite_inputs(
        mf, common, static, 0, virtual_anchor_columns=anchors
    )
    for name in expected:
        numpy.testing.assert_allclose(fixed[name], expected[name],
                                      atol=1e-10)


def test_export_preparation_builds_one_frame_and_replays_its_anchors(
    finite_water_dimer, monkeypatch,
):
    mf, common, static = finite_water_dimer
    original = finite_prepare._build_local_strong_ed_domain_and_anchors
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(
        finite_prepare, "_build_local_strong_ed_domain_and_anchors", counted
    )
    packet, anchors = finite_prepare.prepare_finite_export_inputs(
        mf, common, static, 0
    )
    assert calls == [0]
    replay = prepare_finite_inputs(
        mf, common, static, 0, virtual_anchor_columns=anchors,
    )
    for name in packet:
        numpy.testing.assert_allclose(packet[name], replay[name], atol=1e-10)


def test_full_retention_avoids_support_overlap_eigensolver(
    finite_water_dimer, monkeypatch,
):
    _mf, common, static = finite_water_dimer
    selection = static.fragments[0].strong_virtual
    assert len(selection.overlap_keep) == len(selection.canonical_keep)
    assert len(selection.completeness_keep) == len(selection.overlap_keep)
    original = dlno_base.scipy.linalg.eigh
    calls = []

    def counting_eigh(*args, **kwargs):
        calls.append(args[0].shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(dlno_base.scipy.linalg, "eigh", counting_eigh)
    build_local_strong_ed_domain(common, static, 0)
    assert calls == [(len(selection.parent_columns),) * 2]


def test_finite_packet_cotangents_pull_back_through_frame(finite_water_dimer):
    mf, common, static = finite_water_dimer
    ao = numpy.asarray(static.fragments[0].extended_ao_indices)
    direction = numpy.zeros_like(numpy.asarray(common.fock))
    direction[ao[0], ao[1]] = 1.0
    direction[ao[1], ao[0]] = 1.0
    direction = jnp.asarray(direction)

    def prepare(scale):
        current = common._replace(fock=common.fock + scale * direction)
        return prepare_finite_inputs(mf, current, static, 0)

    scale = jnp.asarray(0.013)
    saved = jax.tree_util.tree_map(numpy.asarray, prepare(scale))
    b_weight = jnp.asarray(
        numpy.random.default_rng(27).normal(size=saved["B"].shape)
    )

    def energy(current_scale):
        packet = prepare(current_scale)
        return (0.05 * jnp.sum(packet["foo"] ** 2)
                + 0.1 * jnp.sum(packet["fvv"] ** 2)
                + 0.01 * jnp.sum(b_weight * packet["B"]))

    bars = {
        "foo": 0.1 * saved["foo"],
        "fvv": 0.2 * saved["fvv"],
        "B": 0.01 * numpy.asarray(b_weight),
    }
    result = {
        "energy": float(energy(scale)),
        "cotangents": bars,
        "metadata": {"cotangent_seed": 1.0},
    }
    (pullback,) = pullback_inputs(prepare, (scale,), saved, result)
    # In the pre-semicanonical frame, B has no dependence on this isolated
    # Fock perturbation; orbital response is tested separately below.
    direct = jax.grad(energy)(scale)
    step = 1e-4
    finite_difference = (energy(scale + step) - energy(scale - step)) / (2 * step)
    numpy.testing.assert_allclose(pullback, direct, rtol=1e-8, atol=1e-8)
    numpy.testing.assert_allclose(pullback, finite_difference,
                                  rtol=2e-4, atol=1e-6)


def test_finite_B_bar_follows_local_orbital_response(finite_water_dimer):
    mf, common, static = finite_water_dimer
    fragment = static.fragments[0]
    direction = numpy.zeros_like(numpy.asarray(common.pao_coeff))
    direction[fragment.extended_ao_indices[0],
              fragment.strong_virtual.parent_columns[0]] = 1.0
    direction = jnp.asarray(direction)

    def prepare(scale):
        current = common._replace(
            pao_coeff=common.pao_coeff + scale * direction
        )
        return prepare_finite_inputs(mf, current, static, 0)

    scale = jnp.asarray(0.003)
    saved = jax.tree_util.tree_map(numpy.asarray, prepare(scale))
    B_bar = numpy.random.default_rng(117).normal(size=saved["B"].shape)
    result = {
        "energy": float(jnp.sum(saved["B"] * B_bar)),
        "cotangents": {
            "foo": numpy.zeros_like(saved["foo"]),
            "fvv": numpy.zeros_like(saved["fvv"]),
            "B": B_bar,
        },
        "metadata": {"cotangent_seed": 1.0},
    }
    (pullback,) = pullback_inputs(prepare, (scale,), saved, result)

    def energy(current_scale):
        return jnp.sum(prepare(current_scale)["B"] * B_bar)

    step = 1e-4
    finite_difference = (energy(scale + step) - energy(scale - step)) / (2 * step)
    assert abs(float(pullback)) > 1e-8
    numpy.testing.assert_allclose(pullback, jax.grad(energy)(scale),
                                  rtol=1e-8, atol=1e-8)
    numpy.testing.assert_allclose(pullback, finite_difference,
                                  rtol=2e-4, atol=1e-6)


def test_local_packet_is_invariant_to_boys_orbital_signs(finite_water_dimer):
    mf, common, static = finite_water_dimer
    expected = prepare_finite_inputs(mf, common, static, 0)
    data = list(common.fragment_occupied_data)
    for partner in static.fragments[0].strong_fragments:
        partner = int(partner)
        item = data[partner]
        data[partner] = item._replace(
            iao_coeff=-item.iao_coeff,
            iao_occ_overlap=-item.iao_occ_overlap,
            occupied_projection=-item.occupied_projection,
        )
    flipped = common._replace(fragment_occupied_data=tuple(data))
    actual = prepare_finite_inputs(mf, flipped, static, 0)
    for name in expected:
        numpy.testing.assert_allclose(actual[name], expected[name],
                                      atol=1e-10)


def test_local_virtual_frame_ignores_retained_eigenspace_order(
    finite_water_dimer, monkeypatch,
):
    mf, common, static = finite_water_dimer
    expected_domain = build_local_strong_ed_domain(common, static, 0)
    expected_packet = prepare_finite_inputs(mf, common, static, 0)
    original = stc_domain._rebuild_local_virtual_subspace
    calls = []

    def reversed_retained_space(*args, **kwargs):
        virtual = original(*args, **kwargs)
        calls.append(virtual.shape[1])
        signs = jnp.where(jnp.arange(virtual.shape[1]) % 2, -1.0, 1.0)
        return virtual[:, ::-1] * signs

    monkeypatch.setattr(
        stc_domain, "_rebuild_local_virtual_subspace", reversed_retained_space
    )
    actual_domain = build_local_strong_ed_domain(common, static, 0)
    actual_packet = prepare_finite_inputs(mf, common, static, 0)
    assert calls == [expected_domain.virtual_coeff.shape[1]] * 2
    numpy.testing.assert_allclose(actual_domain.virtual_coeff,
                                  expected_domain.virtual_coeff, atol=1e-10)
    for name in expected_packet:
        numpy.testing.assert_allclose(actual_packet[name], expected_packet[name],
                                      atol=1e-10)


def test_traced_replay_keeps_the_exported_frame(finite_water_dimer):
    mf, common, static = finite_water_dimer
    for fragment_index in range(len(static.fragments)):
        anchors = finite_prepare.select_virtual_anchor_columns(
            common, static, fragment_index
        )
        exported = prepare_finite_inputs(
            mf, common, static, fragment_index,
            virtual_anchor_columns=anchors,
        )
        replayed, _ = jax.vjp(
            lambda current: prepare_finite_inputs(
                mf, current, static, fragment_index,
                virtual_anchor_columns=anchors,
            ), common,
        )
        for name in exported:
            numpy.testing.assert_allclose(replayed[name], exported[name],
                                          rtol=1e-9, atol=1e-11)
