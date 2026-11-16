"""One fitted global MP2/W grid and independent saved fragment (T) grids."""
from dataclasses import replace

import numpy as np
import pytest

from pyscfad import dlno_stc
from pyscfad.dlno_stc import backend, ccsd, laplace, smooth_triples
from pyscfad.dlno_stc._restart import RestartManager, scientific_digest
from pyscfad.dlno_stc.test.test_system import molecule, _mf


@pytest.fixture(scope='module')
def selected(molecule):
    mf = _mf(molecule)
    domains = dlno_stc.build_domain_selections(
        mf, frozen=1, full_domain=True, mp2_laplace_points=5)
    coupled = ccsd.build_static_selections(
        mf, domains, thresh_occ=0., thresh_vir=0., triples_laplace_points=3)
    from pyscfad.dlno_stc.test._saved_selection import save_from_run
    coupled = save_from_run(mf, coupled)
    return mf, domains, coupled


def test_global_fit_uses_full_active_spectrum_and_is_saved(selected, monkeypatch, tmp_path):
    mf, domains, coupled = selected
    roots, weights, expected = laplace.make_laplace_grid(
        np.asarray(mf.mo_energy)[domains.active_occ_indices],
        np.asarray(mf.mo_energy)[domains.active_vir_indices], kind='fitted', points=5)
    np.testing.assert_array_equal(domains.mp2_grid[0], roots)
    np.testing.assert_array_equal(domains.mp2_grid[1], weights)
    assert domains.mp2_grid_diagnostics['denominator_bounds'] == expected['denominator_bounds']
    assert 'fitting_seconds' not in domains.mp2_grid_diagnostics
    assert coupled.selection_options is None
    assert len(domains.fragments) > 1
    for actual, wanted in zip(coupled.mp2_grid, domains.mp2_grid):
        np.testing.assert_array_equal(actual, wanted)
    for grid in coupled.lis_static.triples_grid:
        assert grid[2]['points'] == len(grid[0]) == 3
        assert grid[2]['denominator_bounds'][0] > 0.
    # Repeated selection at the same reference has stable checkpoint identity.
    again = dlno_stc.build_domain_selections(mf, frozen=1, full_domain=True, mp2_laplace_points=5)
    assert scientific_digest(domains) == scientific_digest(again)
    manager = RestartManager(tmp_path, method='saved-quadratures', scientific_payload={})
    manager.save_static(coupled)
    restored = manager.load_static(expected_type=type(coupled))
    manager.close()
    assert scientific_digest(restored) == scientific_digest(coupled)
    def forbidden(*args, **kwargs):
        raise AssertionError('saved quadratures must not be refitted')
    monkeypatch.setattr(laplace, 'make_laplace_grid', forbidden)
    monkeypatch.setattr(smooth_triples, 'make_grid', forbidden)
    assert np.isfinite(ccsd.kernel(mf, restored, ccsd_t=True))


@pytest.mark.parametrize('method', ['mp2', 'ccsd', 'ccsd_t'])
def test_saved_grids_reach_every_domain_and_fixed_grid_forces(selected, monkeypatch, method):
    mf, domains, coupled = selected
    def forbidden(*args, **kwargs):
        raise AssertionError('energy/force evaluation refitted a saved quadrature')
    monkeypatch.setattr(laplace, 'make_laplace_grid', forbidden)
    monkeypatch.setattr(smooth_triples, 'make_grid', forbidden)
    name = 'solve' if method == 'mp2' else 'solve_matrices'
    solve = getattr(backend, name)
    seen = []
    def checked(inputs, metadata, controls, **kwargs):
        for key, expected in zip(('laplace_roots', 'laplace_weights'), domains.mp2_grid):
            np.testing.assert_array_equal(controls[key], expected)
        seen.append(metadata['fragment_id'])
        return solve(inputs, metadata, controls, **kwargs)
    monkeypatch.setattr(backend, name, checked)
    if method == 'mp2':
        def forbidden_w(*args, **kwargs):
            raise AssertionError('standalone MP2 must not compute W or its pullback')
        monkeypatch.setattr(backend, 'solve_matrices', forbidden_w)
        monkeypatch.setattr(backend, 'pullback_matrices', forbidden_w)
        def energy(mol):
            current = _mf(mol)
            return dlno_stc.kernel(current, domains, controls={}, merge_identical=False) + current.e_tot
        value, bar = dlno_stc.value_and_grad(
            mf.mol, _mf, domains, controls={}, include_hf=True, merge_identical=False)
    else:
        options = dict(ccsd_t=method == 'ccsd_t', include_hf=True, merge_identical=False)
        def energy(mol):
            return ccsd.kernel(_mf(mol), coupled, **options)
        value, bar = ccsd.value_and_grad(mf.mol, _mf, coupled, **options)
    assert sorted(seen) == list(range(len(domains.physical_groups)))
    np.testing.assert_allclose(value, energy(mf.mol), atol=2e-9, rtol=0)
    coords = np.asarray(mf.mol.atom_coords())
    direction = np.zeros_like(coords)
    direction[1, 2] = 1.
    step = 1e-4
    values = [energy(mf.mol.set_geom_(coords + sign*step*direction, unit='Bohr', inplace=False))
              for sign in (1, -1)]
    np.testing.assert_allclose(np.sum(bar.coords*direction), (values[0]-values[1])/(2*step),
                               atol=3e-6, rtol=0)


def test_explicit_grid_is_saved_and_reused_without_repeating_override(selected):
    mf, domains, _ = selected
    grid = ([.1, .4], [.2, .7])
    override = dlno_stc.build_domain_selections(mf, frozen=1, full_domain=True, mp2_grid=grid)
    assert override.mp2_grid_diagnostics['kind'] == 'explicit'
    for actual, expected in zip(override.mp2_grid, grid):
        np.testing.assert_array_equal(actual, expected)
    coupled = ccsd.build_static_selections(
        mf, domains, thresh_occ=0., thresh_vir=0., mp2_grid=grid, triples_laplace_points=3)
    omitted = ccsd.kernel(mf, coupled)
    explicit = ccsd.kernel(mf, coupled, mp2_grid=grid)
    assert omitted == explicit
    with pytest.raises(ValueError, match='scientific controls differ'):
        ccsd.kernel(mf, coupled, mp2_grid=([.2], [.8]))
    with pytest.raises(ValueError, match='saved MP2 grid'):
        dlno_stc.kernel(mf, replace(domains, mp2_grid=None), controls={})
