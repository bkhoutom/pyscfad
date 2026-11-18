"""Load saved selections, retaining response without repeating their searches."""
from copy import copy
from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

from pyscfad import scf
from pyscfad.dlno_stc import ccsd, domain, _selection
from pyscfad.dlno_stc import _ccsd_restart_reference as reference
from pyscfad.dlno_stc import _restart as storage
from pyscfad.dlno_stc._ccsd_checkpoint import open_checkpoint
from pyscfad.dlno_stc.test.test_boys_prepare import boys_water_dimer
from pyscfad.dlno_stc.test.test_triples import checkpoint_reference


def _save(path, mf, selected, *, common=None):
    state = dict(mf=mf)
    if common is not None:
        state['common'] = common
    manager = open_checkpoint(state, selected,
        ccsd.build_domain_groups(selected.mp2_static), checkpoint_dir=path,
        resume=False, options={})
    manager.close()


def _forbid_searches(monkeypatch):
    from pyscfad.dlno_stc import driver
    from pyscfad.lo import boys

    def forbidden(*args, **kwargs):
        pytest.fail('restart repeated a localization/domain/LIS selection search')

    for module, names in [
        (_selection, ('build_domain_selections', '_build_boys_targets', 'get_bp_domain')),
        (ccsd, ('build_static_selections',)),
        (domain, ('_build_boys_targets',)),
        (driver, ('_select_domain_reference',)),
        (boys, ('_boys', 'kernel', '_lowest_curvature')),
    ]:
        for name in names:
            monkeypatch.setattr(module, name, forbidden)


def test_load_choices_checks_configuration_without_search(tmp_path, checkpoint_reference, monkeypatch):
    mf, selected = checkpoint_reference
    _save(tmp_path, mf, selected)
    _forbid_searches(monkeypatch)
    domains, lis = selected.mp2_static, selected.lis_static
    loaded = ccsd.load_static_selections(tmp_path, mf.mol,
        domain_options=dict(frozen=domains.frozen, thresholds=domains.thresholds,
            lo_kwargs=domains.lo_kwargs, boys_grouping=domains.boys_grouping,
            pair_energy_model=domains.pair_energy_model, full_domain=True),
        selection_options=dict(thresh_occ=lis.thresh_occ, thresh_vir=lis.thresh_vir,
            internal_rank_threshold=lis.internal_rank_threshold,
            triples_laplace_points=selected.selection_options['triples_laplace_points']))
    assert storage.scientific_digest(loaded) == storage.scientific_digest(selected)


@pytest.mark.parametrize('changed', [
    'frozen', 'thresholds', 'lo_kwargs', 'boys_grouping', 'pair_energy_model',
    'frag_lolist', 'thresh_occ', 'thresh_vir', 'internal_rank_threshold',
    'triples_laplace_points',
])
def test_changed_requested_settings_are_rejected(tmp_path, checkpoint_reference, changed):
    mf, selected = checkpoint_reference
    _save(tmp_path, mf, selected)
    domains = selected.mp2_static
    choices = dict(frozen=0,
        thresholds=replace(domains.thresholds, pair_energy=1e-3),
        lo_kwargs=dict(domains.lo_kwargs, conv_tol=1e-5), boys_grouping='atom',
        pair_energy_model='exact', frag_lolist=tuple(reversed(domains.frag_lolist)),
        thresh_occ=1e-3, thresh_vir=1e-3, internal_rank_threshold=1e-3,
        triples_laplace_points=8)
    keyword = 'domain_options' if changed in {
        'frozen', 'thresholds', 'lo_kwargs', 'boys_grouping', 'pair_energy_model',
        'frag_lolist'} else 'selection_options'
    with pytest.raises(storage.RestartMismatchError, match='setting'):
        ccsd.load_static_selections(tmp_path, mf.mol, **{keyword: {changed: choices[changed]}})


@pytest.mark.parametrize('changed', ['geometry', 'basis', 'charge', 'spin', 'cart'])
def test_changed_molecule_is_rejected_before_selection(tmp_path, checkpoint_reference, changed):
    mf, selected = checkpoint_reference
    _save(tmp_path, mf, selected)
    mol = mf.mol.set_geom_(np.array(mf.mol.atom_coords()), unit='Bohr', inplace=False)
    if changed == 'geometry':
        coords = np.array(mol.atom_coords()); coords[1, 2] += 1e-5
        mol = mol.set_geom_(coords, unit='Bohr', inplace=False)
    elif changed == 'basis':
        mol._basis = dict(mol._basis, H=[[0, [1., 1.]]])
    else:
        setattr(mol, changed, {'charge': 2, 'spin': 2, 'cart': True}[changed])
    with pytest.raises(storage.RestartMismatchError, match='molecular input'):
        ccsd.load_static_selections(tmp_path, mol)


@pytest.mark.parametrize('changed', ['conv_tol', 'df_basis', 'sgx'])
def test_changed_backend_is_rejected_before_boys_response(
        tmp_path, checkpoint_reference, monkeypatch, changed):
    mf, selected = checkpoint_reference
    _save(tmp_path, mf, selected)
    current = copy(mf)
    if changed == 'conv_tol':
        current.conv_tol *= 10
    elif changed == 'df_basis':
        current.with_df = copy(mf.with_df)
        current.with_df.auxbasis = 'weigend'
    else:
        current._sgx_backend = SimpleNamespace(auxbasis=None, grid_level=0,
            fit_ovlp=True, integral_backend='generated', accuracy={})
    _forbid_searches(monkeypatch)
    with pytest.raises(storage.RestartMismatchError, match='SCF'):
        ccsd.kernel(current, checkpoint_dir=tmp_path, resume=True)


def test_static_integrity_is_checked_before_loading(tmp_path, checkpoint_reference):
    import h5py
    mf, selected = checkpoint_reference
    _save(tmp_path, mf, selected)
    with h5py.File(tmp_path / 'static.h5', 'r+') as handle:
        arrays = []
        handle.visititems(lambda name, obj: arrays.append(obj)
                          if isinstance(obj, h5py.Dataset) and obj.dtype.kind == 'f' and obj.size else None)
        assert arrays
        data = arrays[0][()]; data.flat[0] += .01; arrays[0][...] = data
    with pytest.raises(storage.RestartCorruptionError):
        ccsd.load_static_selections(tmp_path, mf.mol)


def test_full_domain_request_rejects_saved_finite_domains(tmp_path, boys_water_dimer):
    mf, _, domains = boys_water_dimer
    assert any(len(f.extended_ao_indices) < mf.mol.nao for f in domains.fragments)
    selected = ccsd.build_static_selections(mf, domains, thresh_occ=0., thresh_vir=0.)
    _save(tmp_path, mf, selected)
    with pytest.raises(storage.RestartMismatchError, match='full_domain'):
        ccsd.load_static_selections(tmp_path, mf.mol, domain_options=dict(full_domain=True))


@pytest.mark.parametrize('mapping', ['domain_options', 'selection_options'])
def test_unknown_requested_setting_is_not_silently_ignored(tmp_path, checkpoint_reference, mapping):
    mf, selected = checkpoint_reference
    _save(tmp_path, mf, selected)
    with pytest.raises(ValueError, match='unsupported restart'):
        ccsd.load_static_selections(tmp_path, mf.mol, **{mapping: dict(other_setting=1)})


@pytest.mark.parametrize('with_grad', [False, True])
def test_resume_loads_choices_and_retains_energy_force_without_search(
        tmp_path, checkpoint_reference, monkeypatch, with_grad):
    from pyscfad.dlno_stc import driver
    from pyscfad.dlno_stc.test._boys_acceptance import build_mf
    from pyscfad.tools import linear_solver

    monkeypatch.setattr(scf.hf, 'gen_gmres', partial(
        linear_solver.gen_gmres_with_default_kwargs, tol=1e-10, maxiter=100))
    mf, selected = checkpoint_reference
    controls = dict(mode='deterministic', threads=1)
    options = dict(baseline_controls=controls, matrix_controls=controls,
                   include_hf=True, ccsd_t=True, checkpoint_dir=tmp_path,
                   lis_aux_svd_tol=None)
    if with_grad:
        expected = ccsd.value_and_grad(mf.mol, build_mf, selected, **options)
    else:
        expected = ccsd.kernel(mf, selected, **options)
    saved = reference.load_scf_reference(tmp_path, mf.mol)
    before = (tmp_path / 'run.json').read_bytes()
    _forbid_searches(monkeypatch)

    def no_group(*args, **kwargs):
        pytest.fail('a completed CC group was recomputed')
    monkeypatch.setattr(ccsd, '_run_group', no_group)

    def restore(mol):
        current = scf.RHF(mol).density_fit(auxbasis=mf.with_df.auxbasis)
        current.conv_tol, current.conv_tol_grad = mf.conv_tol, mf.conv_tol_grad
        return reference.restore_scf_reference(current, saved)

    if with_grad:
        def no_scf(*args, **kwargs):
            pytest.fail('saved SCF was iterated')
        monkeypatch.setattr(scf.hf, 'kernel', no_scf)
        actual = ccsd.value_and_grad(mf.mol, restore, resume=True, **options)
        np.testing.assert_allclose(actual[0], expected[0], atol=1e-9, rtol=0.)
        np.testing.assert_allclose(actual[1].coords, expected[1].coords, atol=1e-8, rtol=0.)
    else:
        actual = ccsd.kernel(mf, resume=True, **options)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0.)
    assert (tmp_path / 'run.json').read_bytes() == before


def test_nonstationary_saved_boys_is_rejected_without_refinement(
        tmp_path, checkpoint_reference, monkeypatch):
    from pyscfad.lo.boys import BoysStationarityError
    mf, selected = checkpoint_reference
    target = np.array(selected.mp2_static.target_reference_coeff)
    angle = .03
    rotation = np.eye(target.shape[1])
    rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    domains = replace(selected.mp2_static, target_reference_coeff=target @ rotation)
    selected = replace(selected, lis_static=replace(selected.lis_static, mp2_static=domains))
    _save(tmp_path, mf, selected)
    _forbid_searches(monkeypatch)
    with pytest.raises(BoysStationarityError, match='stationary'):
        ccsd.kernel(mf, checkpoint_dir=tmp_path, resume=True)


def test_saved_refined_frame_preserves_full_response_without_search(
        tmp_path, checkpoint_reference, monkeypatch):
    import jax
    from pyscfad.dlno_stc import driver
    mf, selected = checkpoint_reference
    target = np.array(selected.mp2_static.target_reference_coeff)
    angle = .003
    rotation = np.eye(target.shape[1])
    rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    domains = replace(selected.mp2_static, target_reference_coeff=target @ rotation)
    selected = replace(selected, lis_static=replace(selected.lis_static, mp2_static=domains))
    data, data_pullback = jax.vjp(domain.build_scf_data, mf)
    expected, expected_pullback = driver._build_common_with_pullback(
        mf, domains, data, data_pullback)
    assert np.max(np.abs(np.asarray(expected.iao_coeff) - domains.target_reference_coeff)) > 1e-4
    _save(tmp_path, mf, selected, common=expected)
    _forbid_searches(monkeypatch)
    root = reference.load_boys_reference(tmp_path, mf, domains)
    np.testing.assert_array_equal(root, expected.iao_coeff)
    actual, actual_pullback = driver._build_common_with_pullback(
        mf, domains, data, data_pullback, stationary_boys=True,
        boys_reference_coeff=root)
    rng = np.random.default_rng(297)
    seed = jax.tree_util.tree_map(lambda x: rng.normal(size=x.shape), expected)
    for got, want in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_allclose(got, want, atol=2e-10, rtol=2e-10)
    for got, want in zip(jax.tree_util.tree_leaves(actual_pullback(seed)),
                         jax.tree_util.tree_leaves(expected_pullback(seed))):
        if getattr(got, 'dtype', None) != jax.dtypes.float0:
            np.testing.assert_allclose(got, want, atol=2e-8, rtol=2e-8)
