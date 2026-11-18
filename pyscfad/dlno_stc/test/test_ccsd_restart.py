"""A density-restarted SCF can reuse compatible completed CC contributions."""
from copy import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pyscfad.dlno_stc import ccsd
from pyscfad.dlno_stc._ccsd_checkpoint import open_checkpoint, restore_progress, save_progress
from pyscfad.dlno_stc._restart import RestartMismatchError, RestartCorruptionError
from pyscfad.dlno_stc.test.test_triples import checkpoint_reference


def _options():
    return dict(with_grad=False, include_hf=True, ccsd_t=False,
                triples_controls=None, vvvv_direct=False, merge_identical=True)


def _open(directory, mf, selected, *, resume=False):
    return open_checkpoint(dict(mf=mf), selected,
                           ccsd.build_domain_groups(selected.mp2_static),
                           checkpoint_dir=directory, resume=resume, options=_options())


@pytest.mark.parametrize('change', ['scf', 'boys', 'grid', 'compact_bp_domain',
                                  'primary_bp_domain', 'tight_bp_domain',
                                  'pao_bp_domain', 'primary_domain'])
def test_roundoff_reuses_saved_reference_and_completed_energy(tmp_path, checkpoint_reference, change):
    """Bitwise hashes of reconverged/fitted numbers must not reject a restart."""
    mf, selected = checkpoint_reference
    manager = _open(tmp_path, mf, selected)
    state = dict(energy=-.125, variance=0., baseline_energy=-.125, weak_energy=-.01)
    save_progress(manager, state, [], 1, True, with_grad=False)
    manager.close()

    current = copy(mf)
    rebuilt = selected
    if change == 'scf':
        current.e_tot = mf.e_tot + 2e-10
        current.mo_energy = np.asarray(mf.mo_energy) + 2e-10
        current.mo_coeff = np.asarray(mf.mo_coeff) + 2e-10
    elif change != 'grid':
        domains = selected.mp2_static
        if change == 'boys':
            domains = replace(domains, target_reference_coeff=domains.target_reference_coeff + 2e-10)
        else:
            domains = replace(domains, **{change: (np.array([0]), *getattr(domains, change)[1:])})
        rebuilt = replace(selected, lis_static=replace(selected.lis_static, mp2_static=domains))
    else:
        roots, weights = selected.mp2_grid
        grid = (np.asarray(roots) * (1. + 2e-10), np.asarray(weights) * (1. + 2e-10))
        baseline, matrix = ccsd._normalize_controls(None, None, grid)
        rebuilt = replace(selected, mp2_grid=grid,
                          controls_identity=ccsd._scientific_identity(baseline, matrix))

    manager = _open(tmp_path, current, rebuilt, resume=True)
    try:
        restored = manager.load_static(expected_type=type(selected))
        np.testing.assert_array_equal(restored.mp2_static.target_reference_coeff,
                                      selected.mp2_static.target_reference_coeff)
        np.testing.assert_array_equal(restored.mp2_grid[0], selected.mp2_grid[0])
        resumed = dict(energy=0., variance=0., baseline_energy=0., weak_energy=0.)
        assert restore_progress(manager, resumed, with_grad=False) == (1, True, [])
        assert resumed['energy'] == -.125
    finally:
        manager.close()


@pytest.mark.parametrize('change', ['geometry', 'threshold', 'occupation', 'energy',
                                  'orbital_energy', 'orbital_coeff', 'domain_labels',
                                  'grid', 'nonfinite'])
def test_incompatible_restart_is_rejected(tmp_path, checkpoint_reference, change):
    """Relaxing numeric identity must not accept another calculation or frame."""
    mf, selected = checkpoint_reference
    manager = _open(tmp_path, mf, selected)
    manager.close()
    current = copy(mf)
    rebuilt = selected
    if change == 'geometry':
        coords = np.array(mf.mol.atom_coords())
        coords[1, 0] += .01
        current.mol = mf.mol.set_geom_(coords, unit='Bohr', inplace=False)
    elif change == 'threshold':
        domains = replace(selected.mp2_static,
                          thresholds=replace(selected.mp2_static.thresholds, pair_energy=1e-3))
        rebuilt = replace(selected, lis_static=replace(selected.lis_static, mp2_static=domains))
    elif change == 'occupation':
        current.mo_occ = np.asarray(mf.mo_occ).copy()
        current.mo_occ[0], current.mo_occ[-1] = current.mo_occ[-1], current.mo_occ[0]
    elif change == 'energy':
        current.e_tot = mf.e_tot + .01
    elif change == 'orbital_energy':
        current.mo_energy = np.asarray(mf.mo_energy) + .01
    elif change in ('orbital_coeff', 'nonfinite'):
        current.mo_coeff = np.asarray(mf.mo_coeff).copy()
        current.mo_coeff[0, 0] += .01 if change == 'orbital_coeff' else np.nan
    elif change == 'domain_labels':
        domains = selected.mp2_static
        changed = replace(domains.fragments[0], primary_ao_indices=np.array([0], dtype=int))
        domains = replace(domains, fragments=(changed, *domains.fragments[1:]))
        rebuilt = replace(selected, lis_static=replace(selected.lis_static, mp2_static=domains))
    else:
        roots, weights = selected.mp2_grid
        grid = (np.asarray(roots) * 1.2, weights)
        baseline, matrix = ccsd._normalize_controls(None, None, grid)
        rebuilt = replace(selected, mp2_grid=grid,
                          controls_identity=ccsd._scientific_identity(baseline, matrix))
    with pytest.raises((RestartMismatchError, ValueError)):
        manager = _open(tmp_path, current, rebuilt, resume=True)
        manager.close()


@pytest.mark.parametrize('with_grad', [False, True])
def test_resume_initializes_response_from_saved_selection(tmp_path, checkpoint_reference,
                                                        monkeypatch, with_grad):
    """Completed bars must never be combined with a newly selected Boys frame."""
    mf, selected = checkpoint_reference
    options = dict(_options(), with_grad=with_grad)
    manager = open_checkpoint(dict(mf=mf), selected,
                              ccsd.build_domain_groups(selected.mp2_static),
                              checkpoint_dir=tmp_path, resume=False, options=options)
    manager.close()
    domains = replace(selected.mp2_static,
                      target_reference_coeff=selected.mp2_static.target_reference_coeff + 2e-10)
    rebuilt = replace(selected, lis_static=replace(selected.lis_static, mp2_static=domains))

    class SavedFrameReached(Exception):
        pass

    def check_frame(mf, static, *args, **kwargs):
        np.testing.assert_array_equal(static.target_reference_coeff,
                                      selected.mp2_static.target_reference_coeff)
        raise SavedFrameReached

    monkeypatch.setattr(ccsd.driver, '_initialize', check_frame)
    options = dict(checkpoint_dir=tmp_path, resume=True, include_hf=True,
                   lis_aux_svd_tol=None)
    with pytest.raises(SavedFrameReached):
        if with_grad:
            monkeypatch.setattr(ccsd.driver, '_initialize_scf',
                                lambda *args, **kwargs: (mf, lambda x: x))
            ccsd.value_and_grad(mf.mol, lambda mol: mf, static=rebuilt, **options)
        else:
            ccsd.kernel(mf, rebuilt, **options)


@pytest.mark.parametrize('change', ['memory', 'sgx_grid', 'sgx_accuracy', 'screening'])
def test_scf_method_settings_and_workspace(tmp_path, checkpoint_reference, change):
    mf, selected = checkpoint_reference
    current = copy(mf)
    current._sgx_backend = SimpleNamespace(
        auxbasis=None, grid_level=2, fit_ovlp=True, integral_backend='auto',
        accuracy=dict(screening='static', mixed_precision=False), df_memory_mb=4000.)
    manager = _open(tmp_path, current, selected)
    manager.close()
    if change == 'memory':
        current.max_memory = 8000.
        current._sgx_backend.df_memory_mb = 8000.
        manager = _open(tmp_path, current, selected, resume=True)
        manager.close()
        return
    if change == 'sgx_grid':
        current._sgx_backend.grid_level = 3
    elif change == 'sgx_accuracy':
        current._sgx_backend.accuracy['mixed_precision'] = True
    else:
        current.direct_scf_tol *= 10
    with pytest.raises(RestartMismatchError):
        _open(tmp_path, current, selected, resume=True)


def test_numeric_reference_keeps_integrity_checks(tmp_path, checkpoint_reference):
    import h5py
    mf, selected = checkpoint_reference
    manager = _open(tmp_path, mf, selected)
    path = manager.record_path('cc-reference')
    manager.close()
    with h5py.File(path, 'r+') as handle:
        arrays = []
        handle.visititems(lambda name, obj: arrays.append(obj)
                          if isinstance(obj, h5py.Dataset) and obj.shape == mf.mo_coeff.shape else None)
        assert len(arrays) == 1
        arrays[0][0, 0] += 1e-12
    with pytest.raises(RestartCorruptionError):
        _open(tmp_path, mf, selected, resume=True)


def test_scf_density_restart_resumes_interrupted_cc_gradient(tmp_path, monkeypatch):
    """Resume one of two real CC groups, including all common and SCF response."""
    from pyscf.lib import chkfile
    from pyscfad import scf
    from pyscfad.dlno_stc import _ccsd_checkpoint, build_domain_selections
    from pyscfad.dlno_stc.test._boys_acceptance import molecule, LO_OPTIONS

    mol = molecule()
    scf_path = tmp_path / 'scf.chk'
    checkpoint = tmp_path / 'domains'
    density_restarts = []

    def build_mf(mol):
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
        mf.chkfile = str(scf_path)
        density = None
        if scf_path.exists():
            saved = chkfile.load(str(scf_path), 'scf')
            coeff, occ = np.asarray(saved['mo_coeff']), np.asarray(saved['mo_occ'])
            density = (coeff * occ) @ coeff.T
        density_restarts.append(density is not None)
        mf.kernel(dm0=density)
        return mf

    def build_static(mf):
        domains = build_domain_selections(mf, frozen=2, boys_grouping='atom',
                                          lo_kwargs=LO_OPTIONS)
        selected = ccsd.build_static_selections(mf, domains, thresh_occ=0., thresh_vir=0.)
        assert len(selected.fragments) == 2
        assert len(ccsd.build_domain_groups(domains)) == 2
        return selected

    options = dict(build_static=build_static, include_hf=True, ccsd_t=True,
                   merge_identical=True, lis_aux_svd_tol=None)
    solved = []
    original_group = ccsd._run_group

    def count_group(*args, **kwargs):
        solved.append(args[2].representative_index)
        return original_group(*args, **kwargs)

    monkeypatch.setattr(ccsd, '_run_group', count_group)

    class Interrupted(Exception):
        pass

    original_save = _ccsd_checkpoint.save_progress

    def interrupt(manager, state, records, next_group, weak_done, *, with_grad):
        original_save(manager, state, records, next_group, weak_done, with_grad=with_grad)
        if next_group == 1:
            raise Interrupted

    with monkeypatch.context() as patch:
        patch.setattr(_ccsd_checkpoint, 'save_progress', interrupt)
        with pytest.raises(Interrupted):
            ccsd.value_and_grad(mol, build_mf, checkpoint_dir=checkpoint, **options)
    assert len(solved) == 1
    solved.clear()
    resume_options = {key: value for key, value in options.items() if key != 'build_static'}
    from pyscfad.dlno_stc import driver, _selection

    def no_selection(*args, **kwargs):
        pytest.fail('restart rebuilt domain selections')

    with monkeypatch.context() as patch:
        patch.setattr(driver, '_select_domain_reference', no_selection)
        patch.setattr(_selection, 'get_bp_domain', no_selection)
        resumed = ccsd.value_and_grad(mol, build_mf, checkpoint_dir=checkpoint,
                                     resume=True, **resume_options)
        assert len(solved) == 1
        solved.clear()
        completed = ccsd.value_and_grad(mol, build_mf, checkpoint_dir=checkpoint,
                                       resume=True, **resume_options)
        assert solved == []
    reference = ccsd.value_and_grad(mol, build_mf, **options)
    assert len(solved) == 2
    assert density_restarts == [False, True, True, True]
    for result in (resumed, completed):
        np.testing.assert_allclose(result[0], reference[0], atol=1e-9, rtol=0.)
        np.testing.assert_allclose(result[1].coords, reference[1].coords, atol=1e-8, rtol=0.)
