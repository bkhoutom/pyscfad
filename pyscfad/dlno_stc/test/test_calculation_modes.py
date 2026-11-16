"""The four workflows share physical fragment and full-domain semantics."""
from pyscfad.dlno_stc.test._cc_options import cc_options
import jax
import numpy as np
import pytest

from pyscfad import dlno_stc
from pyscfad.dlno_stc import ccsd
from pyscfad.dlno_stc.test import _boys_acceptance as accept
from pyscfad.dlno_stc.test.test_system import molecule, _mf, _controls


@pytest.mark.parametrize('options, error', [
    ({'scope': 'other'}, ValueError),
    ({'full_domain': 'true'}, TypeError),
    ({'full_domain': 1}, TypeError),
    ({'force_full_domains': True}, TypeError),
    ({'mp2_laplace_points': 0}, ValueError),
    ({'mp2_laplace_points': True}, ValueError),
])
def test_invalid_selection_options_before_reference_use(options, error):
    with pytest.raises(error):
        dlno_stc.build_domain_selections(None, frozen=0, **options)


@pytest.mark.parametrize('grouping', ['singleton', 'atom'])
def test_full_domain_preserves_physical_fragments(grouping):
    mf = accept.build_mf(accept.molecule())
    local = dlno_stc.build_domain_selections(mf, frozen=2, boys_grouping=grouping)
    full = dlno_stc.build_domain_selections(mf, frozen=2, boys_grouping=grouping,
                                             full_domain=True)
    assert full.physical_groups == local.physical_groups
    assert len(full.physical_groups) > 1
    assert any(len(f.extended_atoms) < mf.mol.natm for f in local.fragments)
    assert full.strong_mask.all()
    for name in ('compact_bp_domain', 'primary_bp_domain', 'tight_bp_domain',
                 'primary_domain', 'pao_bp_domain'):
        for atoms in getattr(full, name):
            np.testing.assert_array_equal(atoms, np.arange(mf.mol.natm))
    for fragment in full.fragments:
        np.testing.assert_array_equal(fragment.extended_atoms, np.arange(mf.mol.natm))
        np.testing.assert_array_equal(fragment.pao_center_atoms, np.arange(mf.mol.natm))
        np.testing.assert_array_equal(fragment.strong_fragments, np.arange(8))
    np.testing.assert_array_equal(full.active_occ_indices, local.active_occ_indices)
    np.testing.assert_array_equal(full.active_vir_indices, local.active_vir_indices)


@pytest.mark.parametrize('triples', [False, True])
@pytest.mark.parametrize('df_mode', ['half', 'direct'])
def test_cc_energy_and_force_use_single_system_fragment(molecule, monkeypatch, triples, df_mode):
    monkeypatch.setenv('PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE', df_mode)
    mf = _mf(molecule)
    controls = dict(_controls(), threads=1)
    domains = dlno_stc.build_domain_selections(mf, frozen=1, scope='system')
    assert domains.physical_groups == ((0, 1, 2, 3),)
    static = ccsd.build_static_selections(
        mf, domains, thresh_occ=0., thresh_vir=0.,
        **cc_options(controls, controls))
    options = dict(**cc_options(controls, controls),
                   ccsd_t=triples, include_hf=True)

    def forbidden(*args, **kwargs):
        raise AssertionError('forward-only energy constructed a response tape')

    with monkeypatch.context() as guard:
        guard.setattr(jax, 'vjp', forbidden)
        forward = ccsd.kernel(mf, static, **options)
        mp2 = dlno_stc.kernel(mf, scope='system', frozen=1, controls=controls)
    assert np.isfinite(mp2)
    from pyscfad.lno import _df_shared
    receipts = []
    close = _df_shared.LocalDFReverse.close
    def closed(scope):
        receipts.append(dict(scope.receipt))
        close(scope)
    with monkeypatch.context() as counter:
        counter.setattr(_df_shared.LocalDFReverse, 'close', closed)
        energy, gradient = ccsd.value_and_grad(molecule, _mf, static, **options)
    assert len(receipts) == 1
    assert receipts[0]['raw_source_passes'] == (1 if df_mode == 'half' else 4)
    assert receipts[0]['geometry_passes'] == receipts[0]['metric_passes'] == 1
    np.testing.assert_allclose(energy, forward, atol=2e-9, rtol=0)
    assert np.isfinite(gradient.coords).all()
    # Fixed selections and a displaced coordinate test the full CC response,
    # including the MP2 baseline and the triples correction when requested.
    step = 1e-4
    coords = np.asarray(molecule.atom_coords())
    direction = np.zeros_like(coords)
    direction[1, 2] = 1
    def displaced(sign):
        mol = molecule.set_geom_(coords + sign * step * direction, unit='Bohr', inplace=False)
        return ccsd.kernel(_mf(mol), static, **options)
    finite = (displaced(1) - displaced(-1)) / (2 * step)
    np.testing.assert_allclose(np.sum(gradient.coords * direction), finite, atol=2e-6, rtol=0)


def test_matrix_defaults_and_removed_density_mode():
    from pyscfad.dlno_stc import backend, controls
    from pyscfad.dlno_stc.test.test_backend import native
    from pyscfad.dlno_stc.test.test_full_kernel import full_fixture
    x = full_fixture()
    assert controls.normalize_matrix_controls({})['mode'] == 'deterministic'
    assert controls.normalize_triples_controls(None, ccsd_t=True)['mode'] == 'deterministic'
    assert ccsd.driver._driver_controls({}).get('mode', 'deterministic') == 'deterministic'
    assert ccsd._cc_solver_options('deterministic', None) == {}
    default = backend.solve_matrices(x, {'target_indices': (0,)}, {})
    explicit = backend.solve_matrices(x, {'target_indices': (0,)},
                                      {'mode': 'deterministic'})
    for name in ('energy', 'occupied', 'virtual'):
        np.testing.assert_array_equal(default[name], explicit[name])
    with pytest.raises(TypeError):
        controls.normalize_matrix_controls({}, 'D')
    with pytest.raises(TypeError):
        backend.solve_matrices(x, {'target_indices': (0,)}, {}, matrix_kind='D')
    with pytest.raises(TypeError):
        native().solve_matrices(x['foo'], x['fvv'], x['B'], [0], {}, matrix_kind='D')


def test_non_smooth_selections_are_rejected(molecule):
    from dataclasses import replace
    mf = _mf(molecule)
    domains = dlno_stc.build_domain_selections(mf, frozen=1, scope='system')
    selection = dict(thresh_occ=0., thresh_vir=0., baseline_controls={}, matrix_controls={})
    with pytest.raises(TypeError):
        ccsd.build_static_selections(mf, domains, smooth=False, **selection)
    static = ccsd.build_static_selections(mf, domains, **selection)
    # Old concrete semicanonical checkpoints must not run with smooth solvers.
    obsolete = replace(static, lis_static=replace(static.lis_static, smooth_frames=None))
    with pytest.raises(ValueError, match='only smooth'):
        ccsd.kernel(mf, obsolete, baseline_controls={}, matrix_controls={})


def test_mp2_and_triples_quadratures_are_independent(molecule):
    mf = _mf(molecule)
    domains = dlno_stc.build_domain_selections(mf, frozen=1, scope='system')
    options = dict(thresh_occ=0., thresh_vir=0.)
    mp2_grid = ([.13, .71], [.21, .82])
    selected = ccsd.build_static_selections(
        mf, domains, mp2_grid=mp2_grid, triples_laplace_points=3, **options)
    changed_mp2 = ccsd.build_static_selections(
        mf, domains, mp2_grid=([.2], [.9]), triples_laplace_points=3, **options)
    changed_triples = ccsd.build_static_selections(
        mf, domains, mp2_grid=mp2_grid, triples_laplace_points=4, **options)
    from pyscfad.dlno_stc.test._saved_selection import save_from_run
    selected = save_from_run(mf, selected)
    changed_mp2 = save_from_run(mf, changed_mp2)
    changed_triples = save_from_run(mf, changed_triples)
    nodes, weights, diagnostics = selected.lis_static.triples_grid[0]
    assert len(nodes) == len(weights) == diagnostics['points'] == 3
    assert len(changed_triples.lis_static.triples_grid[0][0]) == 4
    # At zero cutoffs the LIS is unchanged: changing MP2/W quadrature must
    # not alter the fitted triple-denominator grid or its denominator bounds.
    for actual, expected in zip(changed_mp2.lis_static.triples_grid[0][:2], (nodes, weights)):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert changed_mp2.controls_identity != selected.controls_identity
    assert changed_triples.controls_identity == selected.controls_identity
    energy = ccsd.kernel(mf, selected, mp2_grid=mp2_grid, ccsd_t=True)
    assert np.isfinite(energy)
    with pytest.raises(ValueError, match='scientific controls differ'):
        ccsd.kernel(mf, selected, mp2_grid=([.2], [.9]))
