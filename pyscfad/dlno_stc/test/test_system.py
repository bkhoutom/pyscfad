"""Whole-system preparation, public options, and restricted DF response."""

import jax
import numpy as np
import pytest
from types import SimpleNamespace

from pyscfad import config_update, gto, scf
from pyscfad import dlno_stc
from pyscfad.dlno_stc import backend, driver, prepare, protocol


def _controls():
    nodes, weights = np.polynomial.legendre.leggauss(64)
    return {"laplace_roots": 20 * (nodes + 1),
            "laplace_weights": 20 * weights, "mode": "deterministic"}


def _mf(mol):
    mf = scf.RHF(mol).density_fit(auxbasis="weigend")
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    return mf


@pytest.fixture(scope="module")
def molecule():
    mol = gto.Mole(atom="O 0 0 0; H 0.1 -0.75 0.57; H 0 0.8 0.61",
                   basis="sto-3g", verbose=0, max_memory=3000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    # Native DFMP2 counts the entire process, including unrelated JAX caches.
    # This seven-AO reference always fits in core.
    mol.incore_anyway = True
    return mol


def test_public_options_are_explicit_before_reference_use():
    for options, error, match in [
        ({}, TypeError, "static"),
        ({"scope": "other"}, ValueError, "scope"),
        ({"scope": "system", "static": object()}, ValueError, "static"),
        ({"frozen": 0}, ValueError, "frozen"),
    ]:
        with pytest.raises(error, match=match):
            dlno_stc.kernel(None, controls=_controls(), **options)


def test_system_canonical_shortcut_cannot_bypass_local_sampling_frame():
    controls = dict(_controls(), canonical_fock=True)
    with pytest.raises(ValueError, match="local Fock"):
        dlno_stc.kernel(None, scope="system", frozen=0, controls=controls)


def test_full_boundary_retains_three_arrays_and_validates_bars():
    inputs = {"foo": np.array([[-1.0]]), "fvv": np.array([[1.0]]),
              "B": np.ones((2, 1, 1))}
    host = backend.host_inputs(inputs, scope="system")
    result = {"energy": np.float64(-1), "energy_standard_error": np.float64(0),
              "diagnostics": {}, "cotangents": {key: value.copy() for key, value in host.items()}}
    protocol.validate_full_result(result, host)
    result["cotangents"]["target_projection"] = np.ones((1, 1))
    with pytest.raises(ValueError, match="cotangents"):
        protocol.validate_full_result(result, host)
    with pytest.raises(ValueError, match="float64"):
        backend.host_inputs(dict(inputs, B=inputs["B"].astype(np.float32)), scope="system")


@pytest.mark.parametrize("frozen", [True, -1, 6, [True], [-1], [7], [0, 0], [0.5]])
def test_system_rejects_invalid_frozen_before_preparation(molecule, frozen):
    with pytest.raises((TypeError, ValueError), match="frozen"):
        dlno_stc.kernel(_mf(molecule), scope="system", frozen=frozen, controls=_controls())


def test_system_preparation_preserves_active_spaces_without_mutating_mf(molecule):
    mf = _mf(molecule)
    before = jax.tree_util.tree_structure(mf)
    original_coeff = np.asarray(mf.mo_coeff).copy()
    arrays = prepare.prepare_system_inputs(mf, np.array([1, 2, 3, 4]), np.array([5, 6]))
    assert set(arrays) == {"foo", "fvv", "B"}
    assert arrays["B"].shape[1:] == (4, 2)
    for key, selected in (("foo", slice(1, 5)), ("fvv", slice(5, 7))):
        block = np.asarray(arrays[key])
        assert np.linalg.norm(block-np.diag(np.diag(block))) > 1e-3
        np.testing.assert_allclose(np.linalg.eigvalsh(block),
                                   np.asarray(mf.mo_energy)[selected], atol=2e-10)
    assert jax.tree_util.tree_structure(mf) == before
    np.testing.assert_array_equal(mf.mo_coeff, original_coeff)


@pytest.mark.parametrize("energies", [np.zeros(6), np.full(7, np.nan),
                                     np.ones(7, dtype=complex)])
def test_scf_fock_projection_requires_real_finite_matching_energies(molecule, energies):
    def invalid_reference(mol):
        mf = _mf(mol)
        mf.mo_energy = jax.numpy.asarray(energies)
        return mf

    with pytest.raises(ValueError, match="orbital energies"):
        dlno_stc.value_and_grad(molecule, invalid_reference, scope="system", frozen=1,
                                controls=_controls())


def test_empty_active_spaces_add_hf_once_without_fitting_or_backend(molecule, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("zero-work system must skip fitting and backend")
    monkeypatch.setattr(driver, "prepare_system_inputs", forbidden)
    monkeypatch.setattr(driver, "run_backend", forbidden)
    for frozen in (5, [0, 1, 2, 3, 4], [5, 6]):
        assert float(dlno_stc.kernel(_mf(molecule), scope="system", frozen=frozen,
                                    controls=_controls())) == 0
    energy, bar = dlno_stc.value_and_grad(molecule, _mf, scope="system", frozen=5,
                                         controls=_controls())
    assert float(energy) == 0
    np.testing.assert_array_equal(bar.coords, np.zeros((3, 3)))
    total, total_bar = dlno_stc.value_and_grad(molecule, _mf, scope="system", frozen=5,
                                             controls=_controls(), include_hf=True)
    with (config_update("pyscfad_scf_implicit_diff", True),
          config_update("pyscfad_scf_first_order_custom", False)):
        hf_energy, hf_bar = jax.value_and_grad(lambda mol: _mf(mol).e_tot)(molecule)
    np.testing.assert_allclose(total, hf_energy, atol=1e-10)
    np.testing.assert_allclose(total_bar.coords, hf_bar.coords, atol=2e-8)


def test_system_matches_native_dfmp2_with_frozen_and_coordinate_response(molecule, monkeypatch):
    from pyscfad.dlno.mp2 import _fix_restart_mo_phases
    from pyscfad.dlno_stc import exchange
    from pyscfad.mp import dfmp2

    def forbidden(*args, **kwargs):
        raise AssertionError("system scope reached domain construction or STC disk exchange")
    monkeypatch.setattr(driver, "rebuild_domain_data", forbidden)
    monkeypatch.setattr(driver, "select_virtual_anchor_columns", forbidden)
    monkeypatch.setattr(driver, "_correlation_term_specs", forbidden)
    for name in ("read_input", "read_result", "write_input", "write_result"):
        monkeypatch.setattr(exchange, name, forbidden)

    controls = _controls()
    with (config_update("pyscfad_scf_implicit_diff", True),
          config_update("pyscfad_scf_first_order_custom", False)):
        mf, scf_pullback = jax.vjp(lambda mol: _fix_restart_mo_phases(_mf(mol)), molecule)
    energies = np.asarray(mf.mo_energy)
    gaps = energies[None, 5:] - energies[1:5, None]
    denominators = (gaps[:, None, :, None] + gaps[None, :, None, :]).ravel()
    interval = np.linspace(denominators.min(), denominators.max(), 101)
    exponentials = np.exp(-interval[:, None] * controls["laplace_roots"])
    np.testing.assert_allclose(exponentials @ controls["laplace_weights"],
                               1 / interval, atol=1e-9, rtol=0)
    np.testing.assert_allclose(exponentials @ (-controls["laplace_roots"] *
                                               controls["laplace_weights"]),
                               -1 / interval ** 2, atol=1e-9, rtol=0)
    native_energy, native_pullback = jax.vjp(
        lambda mf: dfmp2.MP2(mf, frozen=1).kernel(with_t2=False)[0], mf)
    native_mf_bar, = native_pullback(jax.numpy.ones_like(native_energy))
    native_bar, = scf_pullback(native_mf_bar)
    energy, bar = dlno_stc.value_and_grad(molecule, _mf, scope="system", frozen=1,
                                         controls=controls)
    np.testing.assert_allclose(energy, native_energy, atol=2e-10, rtol=0)
    np.testing.assert_allclose(bar.coords, native_bar.coords, atol=2e-7, rtol=2e-5)
    np.testing.assert_allclose(dlno_stc.kernel(mf, scope="system", frozen=[0],
                                              controls=controls), energy, atol=2e-10, rtol=0)
    list_energy, list_bar = dlno_stc.value_and_grad(molecule, _mf, scope="system", frozen=[0],
                                                   controls=controls)
    np.testing.assert_allclose(list_energy, energy, atol=2e-10, rtol=0)
    np.testing.assert_allclose(list_bar.coords, bar.coords, atol=2e-9, rtol=0)
    total, total_bar = dlno_stc.value_and_grad(molecule, _mf, scope="system", frozen=1,
                                             controls=controls, include_hf=True)
    hf_energy, hf_pullback = jax.vjp(lambda mf: mf.e_tot, mf)
    hf_mf_bar, = hf_pullback(jax.numpy.ones_like(hf_energy))
    hf_bar, = scf_pullback(hf_mf_bar)
    np.testing.assert_allclose(total - energy, hf_energy, atol=2e-10, rtol=0)
    np.testing.assert_allclose(total_bar.coords - bar.coords, hf_bar.coords, atol=2e-8, rtol=0)

    direction = np.zeros_like(np.asarray(molecule.atom_coords()))
    direction[1, 2] = 1
    step = 1e-4
    def displaced(sign):
        moved = molecule.set_geom_(np.asarray(molecule.atom_coords()) + sign * step * direction,
                                   unit="Bohr", inplace=False)
        return dlno_stc.kernel(_mf(moved), scope="system", frozen=1, controls=controls)
    finite_difference = (displaced(1) - displaced(-1)) / (2 * step)
    np.testing.assert_allclose(np.sum(np.asarray(bar.coords) * direction), finite_difference,
                               atol=1e-6, rtol=0)


def test_system_rejects_nonpositive_fitting_metric_before_transform(molecule):
    from pyscfad.df import addons

    mf = _mf(molecule)
    # Cached preparation must validate the auxiliary molecule it actually
    # uses, rather than a regenerated molecule from the original auxbasis.
    bad_basis = {symbol: [[0, [1.0, 1.0]], [0, [1.0, 1.0]]]
                 for symbol in ("O", "H")}
    mf.with_df.auxmol = addons.make_auxmol(mf.mol, bad_basis)
    with pytest.raises(NotImplementedError, match="positive.definite"):
        dlno_stc.kernel(mf, scope="system", frozen=1, controls=_controls())
