"""Weighted finite-domain molecular integration, including weak and HF terms."""

import jax
import numpy as np
import pytest
from types import SimpleNamespace

from pyscfad import config_update, gto, scf
from pyscfad.dlno._selection import DomainSelections, build_domain_selections
from pyscfad.dlno.domain import DLNOThresholds, build_domain_topology
from pyscfad.dlno.mp2 import (
    _correlation_term_specs, _fix_restart_mo_phases,
    correlation_value_and_grad,
)


def _mf(mol):
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    return mf


def _controls():
    nodes, weights = np.polynomial.legendre.leggauss(48)
    return {"laplace_roots": 20 * (nodes + 1),
            "laplace_weights": 20 * weights, "mode": "deterministic"}


def test_coordinate_only_driver_rejects_basis_leaves_before_scf():
    from pyscfad import dlno_stc

    driver = getattr(dlno_stc, "value_and_grad", None)
    assert callable(driver), "the weighted molecular driver is missing"
    mol = gto.Mole(atom="H 0 0 0; H 0 0 1", basis="sto-3g", verbose=0)
    mol.build()

    def forbidden_scf(mol):
        raise AssertionError("SCF must not run for unsupported basis derivatives")

    with pytest.raises(NotImplementedError, match="coordinate|nuclear"):
        driver(mol, forbidden_scf, None, controls=_controls())


def test_fractional_boys_labels_are_rejected_before_reference_use():
    from pyscfad.dlno_stc import kernel

    labels = np.array([0.9])
    static = DomainSelections(
        frozen=0, thresholds=None, active_occ_indices=np.array([0]),
        active_vir_indices=np.array([], dtype=int),
        pao_projected_out_indices=np.array([0]),
        pao_parent_ao_indices=np.array([], dtype=int), ao2pao_map=np.array([], dtype=int),
        frag_lolist=(labels,), frag_atmlist=(None,), strong_mask=np.ones((1, 1), dtype=bool),
        fragments=(SimpleNamespace(iao_indices=labels, strong_fragments=np.array([0])),),
        lo_type="boys",
    )
    with pytest.raises(ValueError, match="singleton"):
        kernel(None, static, controls=_controls())


def test_finite_weighted_energy_and_molecular_gradient_include_weak_and_hf():
    from pyscfad.dlno_stc import kernel, value_and_grad

    mol = gto.Mole(
        atom="O 0 0 0; H 0.1 -0.75 0.57; H 0 0.8 0.61; "
             "O 0 0 8; H 0.13 -0.74 8.57; H 0.02 0.81 8.61",
        basis="sto-3g", verbose=0, max_memory=3000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = _mf(mol)
    topology = build_domain_topology(
        mf, frozen=2, lo_type="boys",
        lo_kwargs={"conv_tol": 1e-12, "conv_tol_grad": 1e-9},
        force_full_domains=False,
        thresholds=DLNOThresholds(pair_energy=1e-4),
    )
    static = build_domain_selections(mf, topology)
    assert any(len(fragment.extended_ao_indices) < mol.nao
               for fragment in static.fragments)
    assert any(spec[0] == "weak" for spec in _correlation_term_specs(static))

    controls = _controls()
    energy, bar = value_and_grad(mol, _mf, static, controls=controls)
    np.testing.assert_allclose(kernel(mf, static, controls=controls), energy,
                               atol=2e-10, rtol=0)
    with (config_update("pyscfad_scf_implicit_diff", True),
          config_update("pyscfad_scf_first_order_custom", False)):
        native_mf, scf_pullback = jax.vjp(
            lambda mol_: _fix_restart_mo_phases(_mf(mol_)), mol,
        )
    native_energy, native_mf_bar = correlation_value_and_grad(native_mf, static)
    native_bar, = scf_pullback(native_mf_bar)
    np.testing.assert_allclose(energy, native_energy, atol=2e-9, rtol=0)
    np.testing.assert_allclose(bar.coords, native_bar.coords, atol=2e-7, rtol=2e-5)

    total_energy, total_bar = value_and_grad(
        mol, _mf, static, controls=controls, include_hf=True,
    )
    hf_energy, hf_pullback = jax.vjp(lambda mf_: mf_.e_tot, native_mf)
    hf_mf_bar, = hf_pullback(jax.numpy.ones_like(hf_energy))
    hf_bar, = scf_pullback(hf_mf_bar)
    np.testing.assert_allclose(total_energy - energy, hf_energy, atol=2e-10, rtol=0)
    np.testing.assert_allclose(total_bar.coords - bar.coords, hf_bar.coords,
                               atol=2e-7, rtol=2e-5)

    coords = np.asarray(mol.atom_coords())
    direction = np.zeros_like(coords)
    direction[1, 2] = 1
    step = 1e-4

    def displaced(sign):
        moved = mol.set_geom_(coords + sign * step * direction,
                              unit="Bohr", inplace=False)
        return kernel(_mf(moved), static, controls=controls)

    difference = (displaced(1) - displaced(-1)) / (2 * step)
    np.testing.assert_allclose(np.sum(np.asarray(bar.coords) * direction), difference,
                               atol=2e-6, rtol=3e-4)

    mf.mo_occ = np.asarray(mf.mo_occ).copy()
    mf.mo_occ[0] = 1
    with pytest.raises(NotImplementedError, match="closed.shell"):
        kernel(mf, static, controls=controls)
