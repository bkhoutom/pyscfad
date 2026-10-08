"""The system solver uses one complete differentiable local orbital frame."""

import jax
import numpy as np
import pytest

from pyscfad import config_update, gto, scf
from pyscfad import dlno_stc
from pyscfad.dlno.mp2 import _fix_restart_mo_phases
from pyscfad.dlno_stc import backend, domain, prepare
from pyscfad.dlno_stc.adjoint import apply_full_pullback


OCCUPIED = np.array([1, 2, 3, 4], dtype=np.int32)
VIRTUAL = np.array([5, 6], dtype=np.int32)


def _mf(mol):
    mf = scf.RHF(mol).density_fit(auxbasis="weigend")
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    return mf


@pytest.fixture(scope="module")
def reference():
    mol = gto.Mole(
        atom="O 0 0 0; H 0.1 -0.75 0.57; H 0 0.8 0.61",
        basis="sto-3g", verbose=0, max_memory=3000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mol.incore_anyway = True
    return _mf(mol)


def _off_diagonal_norm(block):
    block = np.asarray(block)
    return np.linalg.norm(block - np.diag(np.diag(block)))


def _controls():
    nodes, weights = np.polynomial.legendre.leggauss(64)
    return {"mode": "deterministic", "laplace_roots": 20 * (nodes + 1),
            "laplace_weights": 20 * weights, "virtual_block_size": 1}


def _local_prepare(mf, frame, virtual=VIRTUAL):
    return prepare.prepare_system_inputs(
        mf, OCCUPIED, virtual, boys_reference=frame.boys_reference,
        virtual_anchor_columns=frame.virtual_anchor_columns,
    )


def test_system_preparation_uses_nondiagonal_local_fock_blocks(reference):
    # Replacing the local frame with active canonical MOs loses both local
    # Fock couplings, even though the resulting MP2 energy is invariant.
    inputs = prepare.prepare_system_inputs(reference, OCCUPIED, VIRTUAL)
    assert _off_diagonal_norm(inputs["foo"]) > 1e-3
    assert _off_diagonal_norm(inputs["fvv"]) > 1e-3


@pytest.mark.parametrize("virtual", [VIRTUAL, VIRTUAL[1:]],
                         ids=["all-virtuals", "frozen-virtual"])
def test_local_frame_spans_exactly_the_selected_active_spaces(reference, virtual):
    # Forming PAOs from the complement of occupied MOs would reintroduce a
    # frozen virtual here; retaining AO parents directly would lose S metric
    # orthogonality or produce more than the selected number of columns.
    frame = domain.build_system_local_frame(reference, OCCUPIED, virtual)
    overlap = np.asarray(reference.get_ovlp())
    canonical = np.asarray(reference.mo_coeff)
    co, cv = np.asarray(frame.occupied_coeff), np.asarray(frame.virtual_coeff)
    assert co.shape == (reference.mol.nao, len(OCCUPIED))
    assert cv.shape == (reference.mol.nao, len(virtual))
    joined = np.concatenate((co, cv), axis=1)
    np.testing.assert_allclose(joined.T @ overlap @ joined,
                               np.eye(joined.shape[1]), atol=2e-10, rtol=0)
    for local, columns in ((co, OCCUPIED), (cv, virtual)):
        selected = canonical[:, columns]
        np.testing.assert_allclose(local @ local.T, selected @ selected.T,
                                   atol=2e-10, rtol=0)
    frozen = np.setdiff1d(np.arange(canonical.shape[1]),
                          np.concatenate((OCCUPIED, virtual)))
    np.testing.assert_allclose(canonical[:, frozen].T @ overlap @ joined,
                               0, atol=2e-10, rtol=0)
    anchors = tuple(frame.virtual_anchor_columns)
    assert len(anchors) == len(set(anchors)) == len(virtual)
    assert all(0 <= column < reference.mol.nao for column in anchors)


@pytest.mark.parametrize("virtual", [VIRTUAL, VIRTUAL[1:]],
                         ids=["all-virtuals", "frozen-virtual"])
def test_local_preparation_rotates_every_tensor_in_the_same_frame(reference, virtual):
    # Independently transform the canonical tensors: using the local Focks
    # with canonical B, or localizing either retained space incompletely,
    # breaks these contractions even if individual dimensions still match.
    frame = domain.build_system_local_frame(reference, OCCUPIED, virtual)
    actual = _local_prepare(reference, frame, virtual)
    canonical = prepare.prepare_canonical_system_inputs(reference, OCCUPIED, virtual)
    overlap = np.asarray(reference.get_ovlp())
    coeff = np.asarray(reference.mo_coeff)
    uo = coeff[:, OCCUPIED].T @ overlap @ np.asarray(frame.occupied_coeff)
    uv = coeff[:, virtual].T @ overlap @ np.asarray(frame.virtual_coeff)
    expected = {
        "foo": uo.T @ np.asarray(canonical["foo"]) @ uo,
        "fvv": uv.T @ np.asarray(canonical["fvv"]) @ uv,
        "B": np.einsum("Pia,ij,ab->Pjb", canonical["B"], uo, uv),
    }
    assert set(actual) == {"foo", "fvv", "B"}
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], atol=5e-10, rtol=2e-10)


def test_virtual_frame_is_the_cholesky_orthonormalized_ao_projection(reference):
    # A further arbitrary virtual rotation preserves the active projector but
    # changes this prescribed PAO anchor gauge and its coordinate response.
    frame = domain.build_system_local_frame(reference, OCCUPIED, VIRTUAL)
    overlap = np.asarray(reference.get_ovlp())
    cv = np.asarray(reference.mo_coeff[:, VIRTUAL])
    columns = np.asarray(frame.virtual_anchor_columns)
    projected = cv @ (cv.T @ overlap[:, columns])
    gram = projected.T @ overlap @ projected
    lower = np.linalg.cholesky((gram + gram.T) / 2)
    expected = np.linalg.solve(lower, projected.T).T
    np.testing.assert_allclose(frame.virtual_coeff, expected, atol=2e-11, rtol=0)


@pytest.mark.parametrize("metadata", ["none", "boys-only", "anchors-only"])
def test_traced_system_preparation_requires_all_concrete_frame_metadata(reference, metadata):
    frame = domain.build_system_local_frame(reference, OCCUPIED, VIRTUAL)
    options = {}
    if metadata == "boys-only":
        options["boys_reference"] = frame.boys_reference
    if metadata == "anchors-only":
        options["virtual_anchor_columns"] = frame.virtual_anchor_columns
    # Discrete pivoting or Boys branch selection during an input VJP would
    # depend on tracer values instead of the fixed concrete reference.
    with pytest.raises(ValueError, match="concrete.*before tracing"):
        jax.vjp(lambda mf: prepare.prepare_system_inputs(
            mf, OCCUPIED, VIRTUAL, **options,
        ), reference)


@pytest.mark.parametrize("columns", [(5,), (5, 5), (0, 7)],
                         ids=["wrong-count", "duplicate", "out-of-range"])
def test_system_frame_rejects_invalid_fixed_virtual_anchors(reference, columns):
    boys_reference = domain.select_system_boys_reference(reference, OCCUPIED)
    with pytest.raises(ValueError, match="anchor columns"):
        domain.build_system_local_frame(
            reference, OCCUPIED, VIRTUAL, boys_reference=boys_reference,
            virtual_anchor_columns=columns,
        )


def test_system_local_frame_handles_one_selected_occupied_orbital(reference):
    occupied = OCCUPIED[:1]
    frame = domain.build_system_local_frame(reference, occupied, VIRTUAL)
    np.testing.assert_allclose(frame.occupied_coeff, reference.mo_coeff[:, occupied],
                               atol=2e-11, rtol=0)
    inputs = prepare.prepare_system_inputs(
        reference, occupied, VIRTUAL, boys_reference=frame.boys_reference,
        virtual_anchor_columns=frame.virtual_anchor_columns,
    )
    assert inputs["B"].shape[1:] == (1, 2)
    np.testing.assert_allclose(inputs["foo"], [[reference.mo_energy[occupied[0]]]],
                               atol=2e-10, rtol=0)


def test_system_boys_response_is_stable_under_large_occupied_rotations(reference):
    # Global logarithmic rotation coordinates become nearly singular when
    # two rotation planes approach pi. A valid occupied frame must still
    # give the same smooth Boys response with its reference labels fixed.
    from copy import copy
    from scipy.linalg import block_diag

    frame = domain.build_system_local_frame(reference, OCCUPIED, VIRTUAL)
    angle = np.pi - 1e-4
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    occupied = jax.numpy.asarray(
        np.asarray(frame.occupied_coeff) @ block_diag(rotation, rotation),
    )
    direction = np.outer(np.asarray(reference.mo_coeff[:, VIRTUAL[0]]),
                         np.array([0.3, -0.2, 0.4, 0.1]))
    seed = jax.numpy.asarray(np.random.default_rng(48).normal(size=occupied.shape))

    def localize(coeff):
        current = copy(reference)
        current.mo_coeff = current.mo_coeff.at[:, OCCUPIED].set(coeff)
        return domain.build_system_local_frame(
            current, OCCUPIED, VIRTUAL,
            boys_reference=frame.boys_reference,
            virtual_anchor_columns=frame.virtual_anchor_columns,
        ).occupied_coeff

    _, pullback = jax.vjp(localize, occupied)
    coeff_bar, = pullback(seed)
    actual = jax.numpy.sum(coeff_bar * direction)
    step = 1e-4
    finite = jax.numpy.sum(
        (localize(occupied + step * direction)
         - localize(occupied - step * direction)) * seed,
    ) / (2 * step)
    assert abs(float(finite)) > 1e-3
    np.testing.assert_allclose(actual, finite, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("guess", [None, "atomic", "cholesky", "random"],
                         ids=["domain-default", "atomic", "cholesky", "random"])
def test_system_boys_uses_domain_setup_without_extra_searches(guess, monkeypatch):
    # This occupied space can produce a stationary saddle. System preparation
    # should follow the shared domain scheme: one reference search, then one
    # differentiated replay, without a separate stability-restart algorithm.
    from scipy.linalg import expm
    from pyscfad.dlno import targets

    calls = []
    real_build = targets.build_targets

    def record_localization(*args, **kwargs):
        calls.append(kwargs["lo_kwargs"])
        return real_build(*args, **kwargs)

    monkeypatch.setattr(targets, "build_targets", record_localization)

    atoms = [("C", (-0.77, 0, 0)), ("C", (0.77, 0, 0))]
    for side in (-1, 1):
        angles = np.arange(3) * 2 * np.pi / 3 + (0 if side == -1 else np.pi / 3)
        for angle in angles:
            atoms.append(("H", (side * (0.77 + 1.09 / 3),
                                 1.09 * np.sqrt(8) / 3 * np.cos(angle),
                                 1.09 * np.sqrt(8) / 3 * np.sin(angle))))
    mol = gto.Mole(atom=atoms, basis="sto-3g", verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = _mf(mol)
    occupied = np.arange(2, 9, dtype=np.int32)
    perturbation = np.random.default_rng(217).normal(size=(7, 7))
    perturbation -= perturbation.T.copy()
    coeff = np.asarray(mf.mo_coeff).copy()
    coeff[:, occupied] = coeff[:, occupied] @ expm(1e-7 * perturbation)
    mf.mo_coeff = jax.numpy.asarray(coeff)

    options = None if guess is None else {"init_guess": guess}
    selected = domain.select_system_boys_reference(mf, occupied, lo_kwargs=options)
    assert len(calls) == 1
    assert calls[0]["init_guess"] == (guess or "atomic")
    assert calls[0]["conv_tol"] == 1e-10
    assert calls[0].get("conv_tol_grad") is None
    assert calls[0]["gmres_options"] == {"restart": 120, "maxiter": 100}
    virtual = np.arange(9, coeff.shape[1], dtype=np.int32)
    frame = domain.build_system_local_frame(
        mf, occupied, virtual, boys_reference=selected,
    )
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[1]["init_guess"], np.eye(len(occupied)))
    overlap = np.asarray(mf.get_ovlp())
    canonical = coeff[:, occupied]
    localized = np.asarray(frame.occupied_coeff)
    np.testing.assert_allclose(localized.T @ overlap @ localized,
                               np.eye(len(occupied)), atol=2e-10, rtol=0)
    np.testing.assert_allclose(localized @ localized.T,
                               canonical @ canonical.T, atol=2e-10, rtol=0)


def test_system_driver_passes_boys_options_to_reference_selection(reference, monkeypatch):
    from pyscfad.dlno_stc import driver

    calls = []
    real_select = driver.select_system_boys_reference
    options = {"init_guess": "cholesky", "conv_tol": 1e-11,
               "gmres_options": {"restart": 80, "maxiter": 60}}

    def record_selection(*args, **kwargs):
        calls.append(kwargs["lo_kwargs"])
        return real_select(*args, **kwargs)

    monkeypatch.setattr(driver, "select_system_boys_reference", record_selection)
    energy = dlno_stc.kernel(
        reference, scope="system", frozen=1, controls=_controls(), lo_kwargs=options,
    )
    assert np.isfinite(energy)
    assert calls == [options]


@pytest.mark.parametrize("entry", ["kernel", "value_and_grad"])
def test_domain_scope_uses_saved_localization_options(entry):
    with pytest.raises(ValueError, match="domain.*static"):
        if entry == "kernel":
            dlno_stc.kernel(None, controls=_controls(), lo_kwargs={"init_guess": "atomic"})
        else:
            dlno_stc.value_and_grad(
                None, None, controls=_controls(), lo_kwargs={"init_guess": "atomic"},
            )


@pytest.fixture(scope="module")
def coordinate_response(reference):
    # Both representations use one SCF pullback, one active selection, and
    # the same Laplace grid. The native C++ solver supplies all three bars.
    with (config_update("pyscfad_scf_implicit_diff", True),
          config_update("pyscfad_scf_first_order_custom", False)):
        mf, scf_pullback = jax.vjp(
            lambda mol: _fix_restart_mo_phases(_mf(mol)), reference.mol,
        )
    # Keep the production localization defaults in this fixture: the
    # arbitrary-seed finite difference also checks their response accuracy.
    frame = domain.build_system_local_frame(mf, OCCUPIED, VIRTUAL)
    local, local_pullback = jax.vjp(lambda mf_: _local_prepare(mf_, frame), mf)
    canonical, canonical_pullback = jax.vjp(
        lambda mf_: prepare.prepare_canonical_system_inputs(mf_, OCCUPIED, VIRTUAL), mf,
    )
    values = {}
    for name, inputs, pullback in (("local", local, local_pullback),
                                   ("canonical", canonical, canonical_pullback)):
        host = backend.host_inputs(inputs, scope="system")
        result = backend.solve(host, {}, _controls(), with_grad=True, scope="system")
        mf_bar, = apply_full_pullback(pullback, host, result)
        mol_bar, = scf_pullback(mf_bar)
        values[name] = (result["energy"], mol_bar)
    return {"mol": reference.mol, "frame": frame, "inputs": local,
            "local_pullback": local_pullback, "scf_pullback": scf_pullback,
            **values}


def test_local_energy_and_coordinate_gradient_match_canonical(coordinate_response):
    local_energy, local_bar = coordinate_response["local"]
    canonical_energy, canonical_bar = coordinate_response["canonical"]
    np.testing.assert_allclose(local_energy, canonical_energy, atol=2e-10, rtol=0)
    np.testing.assert_allclose(local_bar.coords, canonical_bar.coords,
                               atol=2e-7, rtol=2e-5)
    assert np.linalg.norm(np.asarray(local_bar.coords)) > 1e-4


def test_local_preparation_coordinate_vjp_rebuilds_continuous_frame(coordinate_response):
    # An arbitrary tensor seed is not invariant under occupied/virtual
    # rotations. This catches frozen Boys/PAO coefficients that an invariant
    # MP2 energy comparison alone cannot expose.
    state = coordinate_response
    rng = np.random.default_rng(217)
    seeds = {name: jax.numpy.asarray(rng.normal(size=value.shape))
             for name, value in state["inputs"].items()}
    mf_bar, = state["local_pullback"](seeds)
    mol_bar, = state["scf_pullback"](mf_bar)
    coords = np.asarray(state["mol"].atom_coords())
    direction = np.zeros_like(coords)
    direction[1, 0], direction[1, 2], direction[2, 1] = 0.3, 1, -0.2
    step = 1e-4

    def displaced(sign):
        moved = state["mol"].set_geom_(coords + sign * step * direction,
                                       unit="Bohr", inplace=False)
        inputs = _local_prepare(_fix_restart_mo_phases(_mf(moved)), state["frame"])
        return sum(np.sum(np.asarray(inputs[name]) * np.asarray(seed))
                   for name, seed in seeds.items())

    finite = (displaced(1) - displaced(-1)) / (2 * step)
    actual = np.sum(np.asarray(mol_bar.coords) * direction)
    assert abs(finite) > 1e-4
    np.testing.assert_allclose(actual, finite, atol=2e-5, rtol=2e-4)


def test_system_driver_passes_local_focks_and_matches_canonical_response(
        coordinate_response, monkeypatch):
    # Check the real backend boundary as well as the invariant final result:
    # returning to a canonical shortcut would otherwise pass energy tests.
    real_solve = backend.solve

    def local_solve(inputs, metadata, controls, **kwargs):
        assert _off_diagonal_norm(inputs["foo"]) > 1e-3
        assert _off_diagonal_norm(inputs["fvv"]) > 1e-3
        return real_solve(inputs, metadata, controls, **kwargs)

    monkeypatch.setattr(backend, "solve", local_solve)
    energy, bar = dlno_stc.value_and_grad(
        coordinate_response["mol"], _mf, scope="system", frozen=1,
        controls=_controls(),
    )
    canonical_energy, canonical_bar = coordinate_response["canonical"]
    np.testing.assert_allclose(energy, canonical_energy, atol=2e-10, rtol=0)
    np.testing.assert_allclose(bar.coords, canonical_bar.coords, atol=2e-7, rtol=2e-5)
