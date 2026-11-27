"""End-to-end sequential-W CCSD(T) nuclear derivatives with fixed LIS ranks."""
import json
import jax
import numpy as np
import pytest
from pyscfad import config_update, gto, scf
from pyscfad.dlno_stc import ccsd, build_domain_selections, materialize_domain_selections
from pyscfad.dlno_stc.test._saved_selection import save_from_run


def build_mf(mol):
    mf = scf.RHF(mol).density_fit(auxbasis='weigend')
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    return mf


def _check_nuclear_finite_difference(cutoff, ed_naf, record_property, *, lazy=False, matrix_kind="W"):
    mol = gto.Mole(atom='O 0 0 0; H .1 -.75 .57; H 0 .8 .61',
                   basis='6-31g', verbose=0, max_memory=12000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    controls = dict(mode='deterministic', threads=4)
    options = dict(baseline_controls=controls, matrix_controls=controls,
                   include_hf=True, ccsd_t=True, mp2_aux_svd_tol=ed_naf,
                   lis_aux_svd_tol=None)
    with (config_update('pyscfad_scf_implicit_diff', True),
          config_update('pyscfad_scf_first_order_custom', False)):
        mf = build_mf(mol)
        domains = build_domain_selections(mf, frozen=1, boys_grouping='atom_singleton',
                    full_domain=True, lo_kwargs={'conv_tol': 1e-12, 'conv_tol_grad': 1e-9})
        domains = materialize_domain_selections(mf, domains)
        plan = ccsd.build_static_selections(mf, domains, thresh_occ=cutoff,
                    thresh_vir=1e-4, sequential_virtual=True, matrix_kind=matrix_kind,
                    baseline_controls=controls, matrix_controls=controls)
        static = save_from_run(mf, plan, **options)
        dimensions = [ccsd._fragment_dimensions(static, i) for i in range(len(static.fragments))]
        parents = [s.parent_dimensions for s in static.fragments]
        if cutoff:
            assert any(d[0] < p[0] for d, p in zip(dimensions, parents)), (dimensions, parents)
            assert any(d[0] > 1 and d[0] < p[0] for d, p in zip(dimensions, parents))
        energy, gradient = ccsd.value_and_grad(mol, build_mf, plan if lazy else static, **options)
        expected = ccsd.kernel(mf, static, **options)
        np.testing.assert_allclose(energy, expected, atol=2e-9, rtol=0.)
        assert np.isfinite(gradient.coords).all()
        direction = np.random.default_rng(39+int(lazy)).normal(size=(mol.natm, 3))
        direction -= direction.mean(axis=0)
        direction /= np.linalg.norm(direction)
        analytic = float(np.sum(gradient.coords*direction))
        finite = []
        for h in (4e-4, 2e-4):
            energies = []
            for sign in (1, -1):
                displaced = mol.set_geom_(np.asarray(mol.atom_coords()) + sign*h*direction,
                                          unit='Bohr', inplace=False)
                energies.append(float(ccsd.kernel(build_mf(displaced), static, **options)))
            finite.append((energies[0]-energies[1])/(2*h))
        np.testing.assert_allclose(finite, analytic, atol=2e-6, rtol=0.)
        np.testing.assert_allclose(np.sum(gradient.coords, axis=0), 0., atol=2e-7, rtol=0.)
        result = dict(matrix_kind=matrix_kind, cutoff=cutoff, ed_naf=ed_naf, lazy_selection=lazy, dimensions=dimensions,
                      energy=float(energy), gradient=np.asarray(gradient.coords).tolist(),
                      direction=direction.tolist(), analytic=analytic, finite_differences=finite,
                      errors=[abs(f-analytic) for f in finite])
        record_property('sequential_gradient', json.dumps(result))
        print('\nSEQUENTIAL_GRADIENT ' + json.dumps(result), flush=True)
    jax.clear_caches()


@pytest.mark.parametrize('cutoff,ed_naf', [(5e-3, 1e-3), (0., None)])
def test_sequential_ccsdt_nuclear_finite_difference(cutoff, ed_naf, record_property):
    _check_nuclear_finite_difference(cutoff, ed_naf, record_property)


def test_sequential_ccsdt_lazy_nuclear_finite_difference(record_property):
    # Production selects ranks during the gradient call and consumes a saved
    # LIS construction tape; the displaced oracles replay the reference labels.
    _check_nuclear_finite_difference(5e-3, 1e-3, record_property, lazy=True)


def test_sequential_d_ccsdt_nuclear_finite_difference(record_property):
    _check_nuclear_finite_difference(1.7e-3, 1e-3, record_property,
                                    lazy=True, matrix_kind='D')
