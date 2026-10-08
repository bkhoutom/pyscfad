"""Whole-system preparation reuses global fitted factors and their response."""

from types import SimpleNamespace

import h5py
import jax
import numpy as np
import pytest

from pyscf import df as pyscf_df
from pyscfad import config_update, df, gto, scf
from pyscfad import numpy as adnp
from pyscfad.df import _cderi_vjp, incore
from pyscfad.dlno_stc import domain, prepare
from pyscfad.lno import _df_direct, _df_outcore
from pyscfad.lno import df as lno_df


OCCUPIED = np.array([1, 2, 3, 4])
VIRTUAL = np.array([5, 6])


@pytest.fixture(scope="module")
def reference():
    mol = gto.Mole(atom="O 0 0 0; H 0.1 -0.75 0.57; H 0 0.8 0.61",
                   basis="sto-3g", verbose=0, max_memory=3000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    # DFMP2's in-core guard counts unrelated JAX caches accumulated by the
    # full test suite. This seven-AO reference always fits in memory; keep
    # its tiny oracle independent of the process-wide cache footprint.
    mol.incore_anyway = True
    mf = scf.RHF(mol).density_fit(auxbasis="weigend")
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    # A packed, genuinely fitted AO factor also supplies the disk fixture.
    with config_update("pyscfad_moleintor_opt", True):
        mf.kernel()
    return mf


def _holder(mol, coeff, with_df):
    # Hold the Fock matrix fixed to isolate the fitted-factor response.
    return SimpleNamespace(mol=mol, mo_coeff=coeff, with_df=with_df,
                           get_fock=lambda: adnp.eye(mol.nao))


def _frame_selections(mf):
    # Select the discrete Boys branch and PAO anchors before either VJP.
    return {
        "boys_reference": domain.select_system_boys_reference(mf, OCCUPIED),
        "virtual_anchor_columns": domain.select_system_virtual_anchor_columns(
            mf, VIRTUAL,
        ),
    }


def _direct_B(mol, coeff, *, boys_reference, virtual_anchor_columns):
    with_df = df.DF(mol, auxbasis="weigend")
    holder = _holder(mol, coeff, with_df)
    # This oracle tests fitted-factor reuse in the tracked local frame; the
    # independent local-frame tests cover the localization itself.
    frame = domain.build_system_local_frame(
        holder, OCCUPIED, VIRTUAL, boys_reference=boys_reference,
        virtual_anchor_columns=virtual_anchor_columns,
    )
    active_coeff = adnp.concatenate(
        (frame.occupied_coeff, frame.virtual_coeff), axis=1,
    )
    return lno_df.get_local_Lov(
        holder, active_coeff, len(OCCUPIED), np.arange(mol.natm),
        integral_direct=True,
    ).reshape((-1, len(OCCUPIED), len(VIRTUAL)))


def _cotangent(B):
    # A general, nonsymmetric cotangent checks more than the squared norm.
    return adnp.asarray(np.random.default_rng(109).normal(size=B.shape))


def test_system_outcore_cache_preserves_factors_coordinate_and_coefficient_vjps(
        reference, tmp_path, monkeypatch):
    mol, coeff = reference.mol, reference.mo_coeff
    frame_selections = _frame_selections(reference)
    expected, direct_pullback = jax.vjp(
        lambda m, c: _direct_B(m, c, **frame_selections), mol, coeff,
    )
    B_bar = _cotangent(expected)
    expected_mol_bar, expected_coeff_bar = direct_pullback(B_bar)

    source = str(tmp_path / "system-cderi.h5")
    with h5py.File(source, "w") as handle:
        handle.create_dataset("j3c", data=np.asarray(reference.with_df._cderi))

    def cached_B(mol_, coeff_):
        with_df = df.DF(mol_, auxbasis="weigend", incore=False)
        with_df.attach_outcore_cderi(source)
        return prepare.prepare_system_inputs(
            _holder(mol_, coeff_, with_df), OCCUPIED, VIRTUAL,
            **frame_selections,
        )["B"]

    def forbidden_regeneration(*args, **kwargs):
        raise AssertionError("whole-system STC must reuse the existing global CDERI")

    # Once the fit exists, neither forward nor coefficient response may
    # regenerate raw integral values or reconstruct a dense fitted AO factor.
    monkeypatch.setattr(lno_df, "_local_direct_nr_e2", forbidden_regeneration)
    monkeypatch.setattr(_df_direct, "_local_direct_raw_int3c_blocks", forbidden_regeneration)
    monkeypatch.setattr(incore, "cholesky_eri", forbidden_regeneration)
    monkeypatch.setattr(pyscf_df.outcore, "cholesky_eri", forbidden_regeneration)
    real_int3c = _df_direct._int3c_cross_opt.int3c_cross

    def derivative_integrals_only(*args, **kwargs):
        intor = kwargs.get("intor", "int3c2e")
        if "ip" not in intor:
            raise AssertionError("cached STC regenerated raw three-center integral values")
        return real_int3c(*args, **kwargs)

    monkeypatch.setattr(_df_direct._int3c_cross_opt, "int3c_cross", derivative_integrals_only)
    real_array = h5py.Dataset.__array__
    real_getitem = h5py.Dataset.__getitem__
    phase = ["forward"]
    source_rows = {"forward": [], "coefficient": []}

    def reject_dense_factor(dataset, *args, **kwargs):
        if dataset.file.filename == source and dataset.name == "/j3c":
            raise AssertionError("whole-system STC converted the complete AO-factor dataset")
        return real_array(dataset, *args, **kwargs)

    def checked_factor_read(dataset, key):
        if (dataset.file.filename == source and dataset.name == "/j3c"
                and phase[0] in source_rows):
            row_key = key[0] if isinstance(key, tuple) and key else key
            rows = np.arange(dataset.shape[0])[row_key]
            if rows.size == dataset.shape[0]:
                raise AssertionError(f"{phase[0]} read the complete AO-factor dataset")
            source_rows[phase[0]].append(rows.size)
        return real_getitem(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__array__", reject_dense_factor)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", checked_factor_read)
    # Make this tiny molecule exercise the same slab paths used by large fits.
    monkeypatch.setattr(_df_outcore, "_outcore_nr_e2_block_mb", lambda: 1e-6)
    monkeypatch.setattr(_cderi_vjp, "_df_jk_style_blockdim", lambda *args: 2)
    real_coeff_source_vjp = _cderi_vjp.nr_e2_mo_coeff_vjp_from_cderi_source

    def checked_coefficient_source_vjp(*args, **kwargs):
        previous_phase, phase[0] = phase[0], "coefficient"
        try:
            return real_coeff_source_vjp(*args, **kwargs)
        finally:
            phase[0] = previous_phase

    monkeypatch.setattr(_cderi_vjp, "nr_e2_mo_coeff_vjp_from_cderi_source",
                        checked_coefficient_source_vjp)
    actual, pullback = jax.vjp(cached_B, mol, coeff)
    # The coordinate response may use one small slab under its existing
    # one-MiB minimum budget; raw derivative integrals remain allowed.
    phase[0] = "coordinate"
    actual_mol_bar, actual_coeff_bar = pullback(B_bar)

    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(actual_mol_bar.coords, expected_mol_bar.coords,
                               atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(actual_coeff_bar, expected_coeff_bar,
                               atol=2e-10, rtol=2e-10)
    np.testing.assert_array_equal(np.asarray(actual_coeff_bar)[:, 0], 0)
    assert np.max(np.abs(np.asarray(actual_mol_bar.coords))) > 1e-5
    assert np.max(np.abs(np.asarray(actual_coeff_bar))) > 1e-5
    assert len(source_rows["forward"]) > 1 and max(source_rows["forward"]) == 1
    assert len(source_rows["coefficient"]) > 1 and max(source_rows["coefficient"]) <= 2


def test_system_incore_cache_preserves_factors_and_coefficient_vjp(reference, monkeypatch):
    mol, coeff = reference.mol, reference.mo_coeff
    frame_selections = _frame_selections(reference)
    expected, direct_pullback = jax.vjp(
        lambda c: _direct_B(mol, c, **frame_selections), coeff,
    )
    B_bar = _cotangent(expected)
    expected_coeff_bar, = direct_pullback(B_bar)
    source = reference.with_df._cderi

    def forbidden_regeneration(*args, **kwargs):
        raise AssertionError("whole-system STC must reuse the existing global CDERI")

    monkeypatch.setattr(lno_df, "_local_direct_nr_e2", forbidden_regeneration)

    def cached_B(coeff_):
        return prepare.prepare_system_inputs(
            _holder(mol, coeff_, reference.with_df), OCCUPIED, VIRTUAL,
            **frame_selections,
        )["B"]

    actual, pullback = jax.vjp(cached_B, coeff)
    actual_coeff_bar, = pullback(B_bar)
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(actual_coeff_bar, expected_coeff_bar,
                               atol=2e-10, rtol=2e-10)
    assert reference.with_df._cderi is source


def test_system_without_global_cache_preserves_direct_factor_response(reference):
    mol, coeff = reference.mol, reference.mo_coeff
    frame_selections = _frame_selections(reference)
    expected, direct_pullback = jax.vjp(
        lambda m, c: _direct_B(m, c, **frame_selections), mol, coeff,
    )
    B_bar = _cotangent(expected)
    expected_mol_bar, expected_coeff_bar = direct_pullback(B_bar)

    def uncached_B(mol_, coeff_):
        with_df = df.DF(mol_, auxbasis="weigend")
        result = prepare.prepare_system_inputs(
            _holder(mol_, coeff_, with_df), OCCUPIED, VIRTUAL,
            **frame_selections,
        )["B"]
        # A trace-safe no-cache fallback must not build a new AO-factor leaf.
        assert with_df._get_cderi_source() is None
        return result

    actual, pullback = jax.vjp(uncached_B, mol, coeff)
    actual_mol_bar, actual_coeff_bar = pullback(B_bar)
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(actual_mol_bar.coords, expected_mol_bar.coords,
                               atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(actual_coeff_bar, expected_coeff_bar,
                               atol=2e-10, rtol=2e-10)


def test_system_outcore_total_energy_and_nuclear_gradient_match_dfmp2(
        reference, tmp_path, monkeypatch):
    from pyscfad import dlno_stc
    from pyscfad.mp import dfmp2

    source = str(tmp_path / "total-energy-cderi.h5")
    with h5py.File(source, "w") as handle:
        handle.create_dataset("j3c", data=np.asarray(reference.with_df._cderi))

    def build_mf(mol):
        mf = scf.RHF(mol).density_fit(auxbasis="weigend")
        mf.with_df.attach_outcore_cderi(source)
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-9
        mf.kernel()
        return mf

    def conventional_total(mol):
        mf = build_mf(mol)
        return mf.e_tot + dfmp2.MP2(mf, frozen=1).kernel(with_t2=False)[0]

    with (config_update("pyscfad_moleintor_opt", True),
          config_update("pyscfad_scf_implicit_diff", True),
          config_update("pyscfad_scf_first_order_custom", False)):
        expected_energy, expected_bar = jax.value_and_grad(conventional_total)(reference.mol)

    def forbidden_regeneration(*args, **kwargs):
        raise AssertionError("whole-system energy/gradient must reuse the global CDERI")

    monkeypatch.setattr(lno_df, "_local_direct_nr_e2", forbidden_regeneration)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    controls = {"mode": "deterministic", "laplace_roots": 20 * (nodes + 1),
                "laplace_weights": 20 * weights}
    with config_update("pyscfad_moleintor_opt", True):
        actual_energy, actual_bar = dlno_stc.value_and_grad(
            reference.mol, build_mf, scope="system", frozen=1,
            controls=controls, include_hf=True,
        )
    np.testing.assert_allclose(actual_energy, expected_energy, atol=2e-10, rtol=0)
    np.testing.assert_allclose(actual_bar.coords, expected_bar.coords,
                               atol=2e-7, rtol=2e-5)
