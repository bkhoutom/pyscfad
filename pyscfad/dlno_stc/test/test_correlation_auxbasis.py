"""Independent molecular checks for distinct SCF and correlation DF fits."""

import jax
import h5py
import hashlib
import numpy as np
import pytest
from pyscf import df as pyscf_df, lib

from pyscfad import config_update, df, gto, scf
from pyscfad import dlno_stc
from pyscfad.dlno_stc import domain, prepare
from pyscfad.dlno_stc.test.test_backend import native
from pyscfad.dlno_stc.test.test_boys_prepare import boys_water_dimer


JKFIT = "cc-pvdz-jkfit"
RIFIT = "cc-pvdz-ri"


def _controls():
    nodes, weights = np.polynomial.legendre.leggauss(64)
    return dict(mode="deterministic", threads=1,
                laplace_roots=20 * (nodes + 1), laplace_weights=20 * weights)


def _mf(mol):
    mf = scf.RHF(mol).density_fit(auxbasis=JKFIT)
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
    mf.kernel()
    return mf


@pytest.fixture(scope="module")
def molecule():
    # Include JAX executables retained by other molecular tests in the process
    # budget; the independent PySCF MP2 oracle also subtracts that resident RSS.
    mol = gto.Mole(atom="O 0 0 0; H 0.1 -0.75 0.57; H 0 0.8 0.61",
                   basis="sto-3g", verbose=0, max_memory=10000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mol.incore_anyway = True
    return mol


def _oracle_factors(mol, co, cv, auxbasis):
    # PySCF's independent packed AO Cholesky builder bypasses every LNO/STC
    # fitting implementation. The local frame is deliberately held fixed.
    packed = pyscf_df.incore.cholesky_eri(mol, auxbasis=auxbasis)
    ao = lib.unpack_tril(packed)
    return np.einsum("Pmn,mi,na->Pia", ao, np.asarray(co), np.asarray(cv))


def test_system_distinct_fit_uses_ri_factors_and_preserves_jk_fock(molecule):
    mf = _mf(molecule)
    occupied, virtual = np.arange(1, 5), np.arange(5, 7)
    frame = domain.build_system_local_frame(mf, occupied, virtual)
    before_tree = jax.tree_util.tree_structure(mf)
    before_cderi = mf.with_df._cderi
    shared = prepare.prepare_system_inputs(mf, occupied, virtual)
    actual = prepare.prepare_system_inputs(
        mf, occupied, virtual, correlation_auxbasis=RIFIT)
    assert set(actual) == {"foo", "fvv", "B"}
    expected = _oracle_factors(molecule, frame.occupied_coeff,
                               frame.virtual_coeff, RIFIT)
    assert actual["B"].shape[0] != shared["B"].shape[0]
    np.testing.assert_allclose(actual["B"], expected, atol=2e-11, rtol=0)
    for name in ("foo", "fvv"):
        np.testing.assert_array_equal(actual[name], shared[name])
    assert mf.with_df.auxbasis == JKFIT
    assert mf.with_df._cderi is before_cderi
    assert jax.tree_util.tree_structure(mf) == before_tree
    explicit_none = prepare.prepare_system_inputs(
        mf, occupied, virtual, correlation_auxbasis=None)
    for name in shared:
        np.testing.assert_array_equal(shared[name], explicit_none[name])


def _independent_mp2_total(mol):
    from pyscf import mp, scf as pyscf_scf
    # Neither replacing mf.with_df nor rebuilding the reference using RIFIT
    # is equivalent to assigning an independent fit to the MP2 object.
    mf = pyscf_scf.RHF(mol).density_fit(auxbasis=JKFIT)
    mf.conv_tol = 1e-13
    mf.kernel()
    pt = mp.dfmp2.DFMP2(mf, frozen=1)
    pt.with_df = pyscf_df.DF(mol, auxbasis=RIFIT)
    return mf.e_tot + pt.kernel()[0]


def test_system_distinct_fit_total_energy_and_coordinate_gradient(molecule):
    native()
    total, gradient = dlno_stc.value_and_grad(
        molecule, _mf, scope="system", frozen=1, controls=_controls(),
        correlation_auxbasis=RIFIT, include_hf=True)
    np.testing.assert_allclose(total, _independent_mp2_total(molecule),
                               atol=3e-10, rtol=0)
    mf = _mf(molecule)
    energy_only = mf.e_tot + dlno_stc.kernel(
        mf, scope="system", frozen=1, controls=_controls(),
        correlation_auxbasis=RIFIT)
    np.testing.assert_allclose(energy_only, total, atol=2e-11, rtol=0)
    direction = np.zeros((molecule.natm, 3))
    direction[1, 2] = 1
    direction[2, 1] = -0.4
    step = 1e-4
    def displaced(sign):
        moved = molecule.set_geom_(
            np.asarray(molecule.atom_coords()) + sign * step * direction,
            unit="Bohr", inplace=False)
        return _independent_mp2_total(moved)
    finite = (displaced(1) - displaced(-1)) / (2 * step)
    np.testing.assert_allclose(np.sum(np.asarray(gradient.coords) * direction),
                               finite, atol=1e-6, rtol=0)
    np.testing.assert_allclose(np.sum(gradient.coords, axis=0), 0, atol=2e-8, rtol=0)


def test_system_none_preserves_default_energy_and_gradient(molecule):
    native()
    options = dict(scope="system", frozen=1, controls=_controls(), include_hf=True)
    default, default_bar = dlno_stc.value_and_grad(molecule, _mf, **options)
    explicit, explicit_bar = dlno_stc.value_and_grad(
        molecule, _mf, correlation_auxbasis=None, **options)
    np.testing.assert_allclose(explicit, default, atol=2e-12, rtol=0)
    np.testing.assert_allclose(explicit_bar.coords, default_bar.coords,
                               atol=2e-10, rtol=0)


def test_system_distinct_fit_outcore_jk_source_and_geometry_response(
        molecule, tmp_path, monkeypatch):
    from pyscfad.df import _ooc_cotangent
    from pyscfad.tools.linear_solver import eager_gmres
    native()
    mol = molecule.copy()
    mol.incore_anyway = False
    mol.max_memory = 8000
    path = str(tmp_path / "jkfit.h5")
    integrals = df.DF(mol, auxbasis=JKFIT, incore=False)
    integrals._cderi_to_save = path
    with config_update("pyscfad_moleintor_opt", True):
        integrals.build()
    with h5py.File(path, "r") as handle:
        assert set(handle) == {"j3c"}
    def fingerprint():
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    original_digest = fingerprint()
    monkeypatch.setenv("PYSCFAD_FACTOR_PANEL_MEMORY_MB", "0")
    references, receipts, builds = [], [], []
    build = df.DF.build
    finish = _ooc_cotangent.GeometryScope.finish
    def jk_build(obj, *args, **kwargs):
        builds.append(obj.auxbasis)
        assert obj.auxbasis == JKFIT, "correlation must not build an AO-pair DF source"
        return build(obj, *args, **kwargs)
    def finished(scope, bar):
        result = finish(scope, bar)
        receipts.append(dict(scope.stats))
        return result
    def outcore_mf(current):
        mf = scf.RHF(current).density_fit(auxbasis=JKFIT)
        mf.with_df.attach_outcore_cderi(path)
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
        with eager_gmres():
            mf.kernel()
        references.append(mf)
        return mf
    options = dict(scope="system", frozen=1, controls=_controls(),
                   correlation_auxbasis=RIFIT, include_hf=True)
    with config_update("pyscfad_moleintor_opt", True):
        expected, expected_bar = dlno_stc.value_and_grad(mol, _mf, **options)
        monkeypatch.setattr(df.DF, "build", jk_build)
        monkeypatch.setattr(_ooc_cotangent.GeometryScope, "finish", finished)
        actual, actual_bar = dlno_stc.value_and_grad(mol, outcore_mf, **options)
    np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=0)
    np.testing.assert_allclose(actual_bar.coords, expected_bar.coords,
                               atol=2e-7, rtol=0)
    assert receipts and receipts[-1]["geometry_passes"] == 1
    assert receipts[-1]["disk_contributions"] > 0
    assert all(ref.with_df.auxbasis == JKFIT for ref in references)
    assert all(ref.with_df._get_cderi_source() == path for ref in references)
    assert fingerprint() == original_digest
    assert tuple(tmp_path.iterdir()) == (tmp_path / "jkfit.h5",)


def test_finite_domain_distinct_fit_matches_local_ri_oracle(boys_water_dimer):
    from pyscf import gto as pyscf_gto
    mf, common, static = boys_water_dimer
    frame = domain.build_boys_ed_frame(common, static, 0)
    assert len(frame.extended_ao_indices) < mf.mol.nao
    atoms = tuple(int(i) for i in frame.extended_atoms)
    local_mol = pyscf_gto.M(
        atom=[(mf.mol.atom_symbol(i), np.asarray(mf.mol.atom_coords())[i])
              for i in atoms], unit="Bohr", basis=mf.mol.basis, verbose=0)
    shared = prepare.prepare_inputs(mf, common, static, 0)
    actual = prepare.prepare_inputs(
        mf, common, static, 0, correlation_auxbasis=RIFIT)
    expected = _oracle_factors(local_mol, frame.occupied_coeff,
                               frame.virtual_coeff, RIFIT)
    assert actual["B"].shape[0] != shared["B"].shape[0]
    assert actual["B"].shape[0] < pyscf_df.addons.make_auxmol(mf.mol, RIFIT).nao
    np.testing.assert_allclose(actual["B"], expected, atol=2e-11, rtol=0)
    for name in ("foo", "fvv"):
        np.testing.assert_array_equal(actual[name], shared[name])


def _cc_plan(mf):
    from pyscfad.dlno_stc import ccsd
    domains = dlno_stc.build_domain_selections(
        mf, frozen=1, boys_grouping="molecule", force_full_domains=True,
        correlation_auxbasis=RIFIT)
    return ccsd.build_static_selections(
        mf, domains, matrix_kind="W", thresh_occ=0., thresh_vir=0.,
        baseline_controls=_controls(), matrix_controls=_controls(),
        smooth=False, single_pass=True, correlation_auxbasis=RIFIT)


def _independent_ccsd_total(mol):
    from pyscf import cc, scf as pyscf_scf
    mf = pyscf_scf.RHF(mol).density_fit(auxbasis=JKFIT)
    mf.conv_tol = 1e-13
    mf.kernel()
    solver = cc.CCSD(mf, frozen=1).density_fit(auxbasis=RIFIT)
    solver.conv_tol, solver.conv_tol_normt = 1e-12, 1e-10
    solver.kernel()
    assert solver.converged
    return solver.e_tot


def test_cc_correction_distinct_fit_half_direct_and_coordinate_response(
        molecule, monkeypatch):
    from pyscfad.dlno_stc import ccsd
    from pyscfad.lno import ccsd as lno_ccsd
    native()
    # The DF budget includes unrelated JAX executables retained by earlier
    # molecular tests, despite this solver having only six active orbitals.
    molecule = molecule.copy()
    molecule.max_memory = 10000
    # Match the tight independent reference; production amplitude tolerances
    # are intentionally looser than the fitting and finite-difference checks.
    monkeypatch.setattr(lno_ccsd.RCCSD, "conv_tol", 1e-12)
    monkeypatch.setattr(lno_ccsd.RCCSD, "conv_tol_normt", 1e-10)
    static = _cc_plan(_mf(molecule))
    options = dict(baseline_controls=_controls(), matrix_controls=_controls(),
                   correlation_auxbasis=RIFIT, include_hf=True)
    results = []
    for mode in ("direct", "half"):
        monkeypatch.setenv("PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE", mode)
        value, gradient = ccsd.value_and_grad(molecule, _mf, static, **options)
        results.append((value, np.asarray(gradient.coords)))
    (direct, direct_bar), (half, half_bar) = results
    np.testing.assert_allclose(half, direct, atol=3e-11, rtol=0)
    np.testing.assert_allclose(half_bar, direct_bar, atol=3e-8, rtol=0)
    np.testing.assert_allclose(half, _independent_ccsd_total(molecule),
                               atol=3e-9, rtol=0)
    np.testing.assert_allclose(ccsd.kernel(_mf(molecule), static, **options),
                               half, atol=3e-10, rtol=0)
    step = 1e-4
    direction = np.zeros((molecule.natm, 3))
    direction[1, 2] = 1
    def displaced(sign):
        moved = molecule.set_geom_(
            np.asarray(molecule.atom_coords()) + sign * step * direction,
            unit="Bohr", inplace=False)
        return _independent_ccsd_total(moved)
    finite = (displaced(1) - displaced(-1)) / (2 * step)
    np.testing.assert_allclose(np.sum(half_bar * direction), finite,
                               atol=2e-6, rtol=0)
