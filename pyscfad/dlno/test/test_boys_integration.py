"""Explicit numerical acceptance for the shared Boys DLNO pipeline."""
import jax.numpy as jnp
import numpy as np

from pyscfad import config_update, gto, scf
from pyscfad.dlno.mp2 import DLNOMP2, evaluate_domain_mp2
from pyscfad.dlno.domain import DLNOThresholds, build_domain_topology
from pyscfad.dlno._selection import build_domain_selections
from pyscfad.dlno.mp2 import correlation_energy


def _mf(mol):
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    return mf


def test_boys_truncated_strong_and_weak_mp2_gradient_high_cost():
    with (config_update("pyscfad_moleintor_opt", True),
          config_update("pyscfad_scf_implicit_diff", True),
          config_update("pyscfad_scf_first_order_custom", True)):
        mol = gto.Mole(
            atom="O 0 0 0; H 0.1 -0.75 0.57; H 0 0.8 0.61; "
                 "O 0 0 8; H 0.13 -0.74 8.57; H 0.02 0.81 8.61",
            basis="sto-3g", verbose=0, max_memory=3000,
        )
        mol.build(trace_exp=False, trace_ctr_coeff=False)
        options = {"conv_tol": 1e-12, "conv_tol_grad": 1e-9}
        mf = _mf(mol)
        topology = build_domain_topology(
            mf, frozen=2, lo_type="boys", lo_kwargs=options,
            thresholds=DLNOThresholds(pair_energy=1e-4),
        )
        static = build_domain_selections(mf, topology)
        assert np.count_nonzero(~static.strong_mask) > 0
        assert any(len(f.extended_atoms) < mol.natm for f in static.fragments)
        np.testing.assert_allclose(
            correlation_energy(mf, static),
            evaluate_domain_mp2(mf, topology).e_corr,
            atol=2e-9, rtol=0,
        )
        energy, bar = DLNOMP2.value_and_grad(
            mol, build_mf=_mf, frozen=2, topology=static,
            lo_type="boys", lo_kwargs=options,
        )
        direction = np.zeros((6, 3))
        direction[1, 2], direction[4, 1] = 0.8, 0.6
        coords = np.asarray(mol.atom_coords())
        h = 1e-4

        def displaced(sign):
            current = mol.set_geom_(coords + sign*h*direction, unit="Bohr", inplace=False)
            current_mf = _mf(current)
            return current_mf.e_tot + correlation_energy(current_mf, static)

        fd = (displaced(1) - displaced(-1)) / (2*h)
        np.testing.assert_allclose(jnp.sum(bar.coords * direction), fd, atol=2e-6, rtol=1e-5)
        assert np.isfinite(float(energy))


def test_boys_full_space_matches_canonical_ccsd_t_high_cost(monkeypatch):
    from pyscfad.cc import dfccsd
    from pyscfad.dlno.ccsd import DLNOCCSD
    from pyscfad.lno.ccsd import RCCSD as ImpurityRCCSD
    monkeypatch.setattr(ImpurityRCCSD, "conv_tol", 1e-11)
    monkeypatch.setattr(ImpurityRCCSD, "conv_tol_normt", 1e-9)
    mol = gto.Mole(atom="O 0 0 0; H 0.1 -0.75 0.57; H 0 0.8 0.61",
                   basis="sto-3g", verbose=0, max_memory=3000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = _mf(mol)
    local = DLNOCCSD(
        mf, frozen=1, lo_type="boys", thresh_occ=0.0, thresh_vir=0.0,
        force_full_domains=True,
        thresholds=DLNOThresholds(
            domain_pao=0.0, ed_pao=0.0, pao_norm=1e-10,
        ),
    )
    local.ccsd_t = True
    local.kernel()
    canonical = dfccsd.RCCSD(mf, frozen=1)
    canonical.conv_tol = 1e-11
    canonical.conv_tol_normt = 1e-9
    canonical.kernel()
    expected = canonical.e_tot + canonical.ccsd_t()
    np.testing.assert_allclose(local.e_tot, expected, atol=2e-8, rtol=0)
    np.testing.assert_allclose(local.result.e_iao_mp2, local.result.e_mp2_lis,
                               atol=2e-9, rtol=0)
    assert len(local.result.lis_occupied) == 4


def test_boys_truncated_lis_ccsd_t_gradient_high_cost(monkeypatch):
    from pyscfad.dlno.ccsd import DLNOCCSD
    from pyscfad.dlno.ccsd import build_static_selections
    from pyscfad.lno.ccsd import RCCSD as ImpurityRCCSD
    monkeypatch.setattr(ImpurityRCCSD, "conv_tol", 1e-11)
    monkeypatch.setattr(ImpurityRCCSD, "conv_tol_normt", 1e-9)
    with (config_update("pyscfad_moleintor_opt", True),
          config_update("pyscfad_scf_implicit_diff", True),
          config_update("pyscfad_scf_first_order_custom", True)):
        mol = gto.Mole(atom="O 0 0 0; H 0.1 -0.75 0.57; H 0 0.8 0.61",
                       basis="sto-3g", verbose=0, max_memory=3000)
        mol.build(trace_exp=False, trace_ctr_coeff=False)
        settings = dict(frozen=1, lo_type="boys",
                        lo_kwargs={"conv_tol": 1e-12, "conv_tol_grad": 1e-9},
                        thresh_occ=3e-3, thresh_vir=1e-4)
        static = build_static_selections(_mf(mol), **settings)
        assert any(len(f.occupied_lno_keep) + 1 < 4 for f in static.fragments)
        assert any(len(f.virtual_lno_keep) > 0 for f in static.fragments)
        energy, bar = DLNOCCSD.value_and_grad(
            mol, build_mf=_mf, static_selections=static, ccsd_t=True, **settings
        )
        direction = np.zeros((3, 3))
        direction[1, 2] = 1.0
        coords = np.asarray(mol.atom_coords())
        h = 1e-4

        def displaced(sign):
            current = mol.set_geom_(coords + sign*h*direction, unit="Bohr", inplace=False)
            solver = DLNOCCSD(_mf(current), **settings)
            solver.ccsd_t = True
            solver.kernel(static_selections=static)
            return solver.e_tot

        fd = (displaced(1) - displaced(-1)) / (2*h)
        np.testing.assert_allclose(jnp.sum(bar.coords * direction), fd, atol=2e-6, rtol=1e-5)
        assert np.isfinite(float(energy))
