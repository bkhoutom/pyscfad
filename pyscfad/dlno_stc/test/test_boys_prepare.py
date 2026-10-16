"""Independent Boys packets preserve fixed frames and three-tensor response."""

from copy import copy

import jax
import numpy
import pytest

from pyscfad import gto, numpy as jnp, scf
from pyscfad.dlno_stc import adjoint, domain, prepare
from pyscfad.lno import df as lno_df


def _result(packet, bars):
    return {"energy": numpy.float64(0), "energy_standard_error": numpy.float64(0),
            "diagnostics": {}, "cotangents": bars,
            "metadata": {"cotangent_seed": 1.0}}


def test_selected_immediate_pullback_accepts_exact_three_tensor_packet():
    def project(scale):
        return {"foo": jnp.reshape(scale ** 2, (1, 1)),
                "fvv": jnp.reshape(3 * scale, (1, 1)),
                "B": jnp.reshape(2 * scale, (1, 1, 1))}

    packet, pullback = jax.vjp(project, jnp.asarray(0.5))
    bars = {name: numpy.ones_like(value) for name, value in packet.items()}
    actual, = adjoint.apply_selected_pullback(
        pullback, packet, _result(packet, bars), energy_bar=2.0,
    )
    numpy.testing.assert_allclose(actual, 12.0, atol=0, rtol=0)
    extra = dict(bars, target_projection=numpy.ones((1, 1)))
    with pytest.raises(ValueError, match="exactly"):
        adjoint.apply_selected_pullback(pullback, packet, _result(packet, extra))


def test_phase_and_cotangent_helpers_match_native_numerical_semantics():
    from types import SimpleNamespace
    native = pytest.importorskip("pyscfad.dlno.mp2")

    coeff = jnp.asarray([[1.0, 0.0, -2.0], [-1.0 - 1e-12, 0.0, 0.1]])
    expected = native._fix_restart_mo_phases(SimpleNamespace(mo_coeff=coeff))
    actual = adjoint.fix_restart_mo_phases(SimpleNamespace(mo_coeff=coeff))
    numpy.testing.assert_array_equal(actual.mo_coeff, expected.mo_coeff)
    leaves = [None, jnp.asarray([1.0, 2.0]),
              numpy.zeros((2,), dtype=jax.dtypes.float0)]
    for left in leaves:
        for right in leaves:
            actual = adjoint.add_cotangent(left, right)
            expected = native._add_cotangent(left, right)
            if actual is None:
                assert expected is None
            else:
                numpy.testing.assert_array_equal(actual, expected)


@pytest.fixture(scope="module")
def boys_water_dimer():
    from pyscfad.dlno_stc import build_domain_selections

    mol = gto.Mole(
        atom="O 0 0 0; H 0 0.75 0.58; H 0 -0.75 0.58; "
             "O 0 0 8; H 0 0.75 8.58; H 0 -0.75 8.58",
        basis="sto-3g", verbose=0, max_memory=3000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    static = build_domain_selections(mf, frozen=2)
    common = domain.rebuild_domain_data(mf, static)
    return mf, common, static


def test_boys_packet_uses_finite_ao_aux_support_and_fixed_frame(boys_water_dimer):
    mf, common, static = boys_water_dimer
    fragment = static.fragments[0]
    frame = domain.build_boys_ed_frame(common, static, 0)
    actual = prepare.prepare_inputs(mf, common, static, 0)
    assert set(actual) == {"foo", "fvv", "B"}
    ao = numpy.asarray(frame.extended_ao_indices)
    assert len(ao) < mf.mol.nao
    assert frame.target_index == tuple(fragment.strong_fragments).index(0)
    assert frame.virtual_coeff.shape[1] == len(fragment.strong_virtual.metric_keep)
    co, cv = numpy.asarray(frame.occupied_coeff), numpy.asarray(frame.virtual_coeff)
    nocc, nvir = co.shape[1], cv.shape[1]
    fock = numpy.asarray(common.fock)[numpy.ix_(ao, ao)]
    numpy.testing.assert_allclose(actual["foo"], co.T @ fock @ co, atol=1e-10, rtol=0)
    numpy.testing.assert_allclose(actual["fvv"], cv.T @ fock @ cv, atol=1e-10, rtol=0)
    expected_B = lno_df.get_local_Lov(
        mf, jnp.concatenate((co, cv), axis=1), nocc,
        frame.extended_atoms, integral_direct=True,
    ).reshape((-1, nocc, nvir))
    numpy.testing.assert_allclose(actual["B"], expected_B, atol=1e-10, rtol=0)
    assert actual["B"].shape[0] < mf.with_df.auxmol.nao


def test_boys_three_bar_fock_direction_and_disk_replay(boys_water_dimer):
    mf, common, static = boys_water_dimer
    ao = numpy.asarray(static.fragments[0].extended_ao_indices)
    perturbation = numpy.zeros_like(common.fock)
    perturbation[ao[0], ao[1]] = perturbation[ao[1], ao[0]] = 1.0
    perturbation = jnp.asarray(perturbation)

    def projected(scale):
        return prepare.prepare_inputs(
            mf, common._replace(fock=common.fock + scale * perturbation), static, 0,
        )

    scale = jnp.asarray(0.013)
    packet, pullback = jax.vjp(projected, scale)
    bars = {name: numpy.asarray(value) * 0.1 for name, value in packet.items()}
    result = _result(packet, bars)
    direct, = adjoint.apply_selected_pullback(pullback, packet, result)
    replay, = adjoint.pullback_inputs(projected, (scale,), packet, result)
    numpy.testing.assert_allclose(direct, replay, atol=1e-11, rtol=0)

    def functional(value):
        return sum(jnp.sum(projected(value)[name] * bar) for name, bar in bars.items())

    for step in (1e-5, 5e-6):
        finite = (functional(scale + step) - functional(scale - step)) / (2 * step)
        numpy.testing.assert_allclose(direct, finite, rtol=1e-6, atol=1e-8)


def test_boys_B_bar_follows_retained_pao_response(boys_water_dimer):
    mf, common, static = boys_water_dimer
    fragment = static.fragments[0]
    perturbation = numpy.zeros_like(common.pao_coeff)
    perturbation[fragment.extended_ao_indices[0],
                 fragment.strong_virtual.parent_columns[0]] = 1.0
    perturbation = jnp.asarray(perturbation)

    def projected(scale):
        return prepare.prepare_inputs(
            mf, common._replace(pao_coeff=common.pao_coeff + scale * perturbation),
            static, 0,
        )

    scale = jnp.asarray(0.003)
    packet, pullback = jax.vjp(projected, scale)
    bars = {name: numpy.zeros_like(value) for name, value in packet.items()}
    bars["B"] = numpy.random.default_rng(117).normal(size=packet["B"].shape)
    actual, = adjoint.apply_selected_pullback(pullback, packet, _result(packet, bars))
    assert abs(float(actual)) > 1e-8
    for step in (1e-5, 5e-6):
        finite = jnp.sum((projected(scale + step)["B"]
                          - projected(scale - step)["B"]) * bars["B"]) / (2 * step)
        numpy.testing.assert_allclose(actual, finite, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("fock_from_scf", [False, True])
def test_system_preparation_preserves_active_spectra(boys_water_dimer, fock_from_scf):
    mf, _common, static = boys_water_dimer
    occupied = numpy.asarray(static.active_occ_indices)
    virtual = numpy.asarray(static.active_vir_indices)
    frame = domain.build_system_local_frame(mf, occupied, virtual)
    packet = prepare.prepare_system_inputs(
        mf, occupied, virtual, boys_reference=frame.boys_reference,
        fock_from_scf=fock_from_scf)
    assert set(packet) == {"foo", "fvv", "B"}
    for name, indices in (("foo", occupied), ("fvv", virtual)):
        numpy.testing.assert_allclose(numpy.linalg.eigvalsh(packet[name]),
                                      numpy.sort(numpy.asarray(mf.mo_energy)[indices]),
                                      atol=1e-8, rtol=0)


def test_zero_cotangent_helpers_preserve_live_mf_and_common_trees(boys_water_dimer):
    native = pytest.importorskip("pyscfad.dlno.mp2")

    mf, common, _static = boys_water_dimer
    expected = native._zero_term_cotangents(mf, common)
    actual = adjoint.zero_term_cotangents(mf, common)
    for left, right in zip(jax.tree_util.tree_leaves(actual),
                           jax.tree_util.tree_leaves(expected)):
        numpy.testing.assert_array_equal(left, right)
        assert left.dtype == right.dtype
    mf_bar = adjoint.zero_mf_cotangent(mf, common.s1e.dtype)
    for left, right in zip(jax.tree_util.tree_leaves(mf_bar),
                           jax.tree_util.tree_leaves(actual[0])):
        numpy.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("source", ["saved", "direct"])
def test_independent_system_streamed_and_direct_three_bar_response(
    boys_water_dimer, source, tmp_path, monkeypatch,
):
    import h5py
    from pyscfad import config_update

    reference, _common, static = boys_water_dimer

    def build_mf(mol):
        current = scf.RHF(mol).density_fit()
        current.conv_tol = 1e-11
        current.kernel()
        return current

    with (config_update("pyscfad_scf_implicit_diff", True),
          config_update("pyscfad_scf_first_order_custom", False)):
        mf, scf_pullback = jax.vjp(build_mf, reference.mol)
    occupied = numpy.asarray(static.active_occ_indices)
    virtual = numpy.asarray(static.active_vir_indices)
    frame = domain.build_system_local_frame(mf, occupied, virtual)
    options = dict(boys_reference=frame.boys_reference,
                   fock_from_scf=True)
    expected, expected_pullback = jax.vjp(
        lambda value: prepare.prepare_system_inputs(value, occupied, virtual, **options), mf)
    if source == "saved":
        path = str(tmp_path / "boys-system-cderi.h5")
        with h5py.File(path, "w") as handle:
            handle.create_dataset("j3c", data=numpy.asarray(mf.with_df._cderi))

    def select_source(value):
        current = copy(value)
        current.with_df = copy(value.with_df)
        if source == "saved":
            current.with_df.attach_outcore_cderi(path)
        else:
            current.with_df._cderi = None
            current.with_df._prefer_cderi_to_save = False
        return current

    current, source_pullback = jax.vjp(select_source, mf)

    def forbidden_fock(*args, **kwargs):
        raise AssertionError("optimized preparation must use the SCF-energy projection")

    # Guard preparation only: implicit SCF response builds its own Fock map.
    with monkeypatch.context() as guard:
        guard.setattr(type(current), "get_fock", forbidden_fock)
        actual, pullback = jax.vjp(
            lambda value: prepare.prepare_system_inputs(value, occupied, virtual, **options),
            current,
        )
    for name in actual:
        numpy.testing.assert_allclose(actual[name], expected[name], atol=1e-9, rtol=0)
    rng = numpy.random.default_rng(932)
    bars = {name: rng.normal(size=value.shape) for name, value in actual.items()}
    for name in ("foo", "fvv"):
        bars[name] = (bars[name] + bars[name].T) / 2
    result = _result(actual, bars)
    actual_bar, = adjoint.apply_full_pullback(pullback, actual, result)
    expected_bar, = adjoint.apply_full_pullback(expected_pullback, expected, result)
    # Cached factors carry their own bar; streamed/direct integrals return
    # molecular bars. Close both through the same SCF/DF construction before
    # comparing physical responses, rather than unequal intermediate leaves.
    actual_mf_bar, = source_pullback(actual_bar)
    actual_mol_bar, = scf_pullback(actual_mf_bar)
    expected_mol_bar, = scf_pullback(expected_bar)
    jax.block_until_ready((actual_mol_bar, expected_mol_bar))
    assert numpy.linalg.norm(numpy.asarray(expected_mol_bar.coords)) > 1e-4
    numpy.testing.assert_allclose(actual_mol_bar.coords, expected_mol_bar.coords,
                                  atol=1e-8, rtol=1e-8)
