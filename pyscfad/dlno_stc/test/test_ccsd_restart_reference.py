"""Roundoff-tolerant restart with fixed settings and orbital-frame guards."""
from copy import copy
from dataclasses import replace

import jax
import numpy as np
import pytest

from pyscfad.dlno_stc import ccsd, domain
from pyscfad.dlno_stc import _ccsd_checkpoint as checkpoint
from pyscfad.dlno_stc import _ccsd_restart_reference as reference
from pyscfad.dlno_stc import _restart as storage
from pyscfad.dlno_stc.adjoint import zero_term_cotangents
from pyscfad.dlno_stc.test.test_triples import checkpoint_reference


def _state(mf):
    return dict(mf=mf, energy=-.2, variance=0., baseline_energy=-.1, weak_energy=0.)


def _open(path, mf, static, *, resume=False, state=None, options=None):
    return checkpoint.open_checkpoint(_state(mf) if state is None else state,
        static, ccsd.build_domain_groups(static.mp2_static), checkpoint_dir=path,
        resume=resume, options={} if options is None else options)


def test_roundoff_resumes_against_original_numerical_reference(tmp_path, checkpoint_reference):
    mf, static = checkpoint_reference
    first = _open(tmp_path, mf, static)
    original_digest = first.base_digest
    first.close()
    manifest_bytes = (tmp_path / 'run.json').read_bytes()
    changed = copy(mf)
    changed.mo_coeff = changed.mo_coeff + 2e-11
    changed.mo_energy = changed.mo_energy + 3e-11
    changed.e_tot = changed.e_tot + 2e-10
    for _ in range(2):
        resumed = _open(tmp_path, changed, static, resume=True)
        assert resumed.base_digest == original_digest
        resumed.close()
        assert (tmp_path / 'run.json').read_bytes() == manifest_bytes


@pytest.mark.parametrize('field,delta', [('mo_coeff', 1e-4), ('mo_energy', 1e-5),
                                       ('e_tot', 1e-6), ('mo_occ', .1),
                                       ('mo_coeff', 3e-6), ('mo_energy', 3e-7),
                                       ('e_tot', 3e-8)])
def test_reference_changes_are_rejected(tmp_path, checkpoint_reference, field, delta):
    mf, static = checkpoint_reference
    _open(tmp_path, mf, static).close()
    changed = copy(mf)
    setattr(changed, field, getattr(changed, field) + delta)
    with pytest.raises(storage.RestartMismatchError):
        _open(tmp_path, changed, static, resume=True)


def test_changed_scf_settings_and_unconverged_reference_are_rejected(tmp_path, checkpoint_reference):
    mf, static = checkpoint_reference
    _open(tmp_path, mf, static).close()
    changed = copy(mf)
    changed.conv_tol *= 10
    with pytest.raises(storage.RestartMismatchError, match='SCF/SGX settings'):
        _open(tmp_path, changed, static, resume=True)
    changed = copy(mf)
    changed.converged = False
    with pytest.raises(storage.RestartMismatchError, match='converged SCF'):
        _open(tmp_path, changed, static, resume=True)


def test_saved_selection_values_are_reused_but_settings_remain_exact(tmp_path, checkpoint_reference):
    mf, static = checkpoint_reference
    _open(tmp_path, mf, static).close()
    domains = static.mp2_static
    noisy = replace(domains, target_reference_coeff=domains.target_reference_coeff + 1e-12)
    current = replace(static, lis_static=replace(static.lis_static, mp2_static=noisy))
    saved = reference.resume_selections(tmp_path, current)
    assert storage.scientific_digest(saved) == storage.scientific_digest(static)
    changed_threshold = replace(domains.thresholds, pair_energy=domains.thresholds.pair_energy + 1e-12)
    current = replace(static, lis_static=replace(static.lis_static,
        mp2_static=replace(domains, thresholds=changed_threshold)))
    with pytest.raises(storage.RestartMismatchError, match='pair_energy'):
        reference.resume_selections(tmp_path, current)
    current = replace(static, lis_static=replace(static.lis_static,
        mp2_static=replace(domains, active_occ_indices=domains.active_occ_indices[::-1])))
    with pytest.raises(storage.RestartMismatchError, match='active_occ_indices'):
        reference.resume_selections(tmp_path, current)


def _legacy_case(tmp_path, mf, static, *, canonical_bar=False):
    state = _state(mf)
    state['common'] = domain.rebuild_domain_data(mf, static.mp2_static)
    state['mf_bar'], state['common_bar'] = zero_term_cotangents(mf, state['common'])
    if canonical_bar:
        state['mf_bar'].mo_coeff = state['mf_bar'].mo_coeff + 1.
    first = _open(tmp_path, mf, static, state=state)
    checkpoint.save_progress(first, state, [], 0, True, with_grad=True)
    first.close()
    (tmp_path / 'records/cc-reference/state.h5').unlink()
    return state


def test_legacy_energy_is_authenticated_and_canonical_bars_are_guarded(tmp_path, checkpoint_reference):
    mf, static = checkpoint_reference
    for canonical_bar in (False, True):
        path = tmp_path / str(canonical_bar)
        state = _legacy_case(path, mf, static, canonical_bar=canonical_bar)
        changed = copy(mf)
        changed.e_tot = changed.e_tot + 2e-10
        state['mf'] = changed
        with pytest.raises(storage.RestartMismatchError, match='no numerical SCF reference'):
            _open(path, changed, static, state=state, resume=True)
        with pytest.raises(storage.RestartMismatchError, match='original checkpoint hash'):
            reference.recover_legacy_energy(path, float(mf.e_tot) + 1e-10)
        reference.recover_legacy_energy(path, float(mf.e_tot))
        if canonical_bar:
            with pytest.raises(storage.RestartMismatchError, match='gauge-dependent'):
                _open(path, changed, static, state=state, resume=True)
        else:
            with pytest.warns(RuntimeWarning, match='Recovered legacy'):
                resumed = _open(path, changed, static, state=state, resume=True)
            assert checkpoint.restore_progress(resumed, state, with_grad=True)[:2] == (0, True)
            resumed.close()


def test_original_reference_checksum_is_enforced(tmp_path, checkpoint_reference):
    mf, static = checkpoint_reference
    manager = _open(tmp_path, mf, static)
    record = manager.load_record('cc-reference', templates={})
    scalars = record.scalars
    scalars['canonical']['e_tot'] = scalars['canonical']['e_tot'] + 1e-10
    manager.save_record('cc-reference', scalars=scalars)
    manager.close()
    with pytest.raises(storage.RestartCorruptionError, match='original identity'):
        _open(tmp_path, mf, static, resume=True)
    with pytest.raises(storage.RestartCorruptionError, match='original identity'):
        reference.scf_initial_guess(tmp_path, mf.mol)


def test_saved_scf_density_initializes_reference_without_atomic_guess(
        tmp_path, checkpoint_reference, monkeypatch):
    from pyscf import scf
    mf, static = checkpoint_reference
    state = _state(mf)
    # Loading canonical arrays must not require live templates for saved trees.
    state['common'] = (np.eye(2),)
    _open(tmp_path, mf, static, state=state).close()
    path = tmp_path / 'records/cc-reference/state.h5'
    original = path.read_bytes()
    density = reference.scf_initial_guess(tmp_path, mf.mol)
    np.testing.assert_allclose(density, mf.make_rdm1(), atol=1e-14, rtol=0.)

    def forbidden(*args, **kwargs):
        raise AssertionError('restart requested a cold atomic guess')

    resumed = scf.RHF(mf.mol.to_pyscf()).density_fit(auxbasis=mf.with_df.auxbasis)
    resumed.conv_tol, resumed.conv_tol_grad = mf.conv_tol, mf.conv_tol_grad
    monkeypatch.setattr(resumed, 'get_init_guess', forbidden)
    resumed.kernel(dm0=density)
    assert resumed.converged
    np.testing.assert_allclose(resumed.e_tot, mf.e_tot, atol=1e-10, rtol=0.)
    assert path.read_bytes() == original


def test_scf_guess_rejects_changed_geometry_before_scf(tmp_path, checkpoint_reference):
    mf, static = checkpoint_reference
    _open(tmp_path, mf, static).close()
    changed = mf.mol.to_pyscf().copy()
    changed.set_geom_(changed.atom_coords() + .001, unit='Bohr')
    with pytest.raises(storage.RestartMismatchError, match='coords_bohr'):
        reference.scf_initial_guess(tmp_path, changed)


def test_legacy_checkpoint_cannot_silently_start_scf_cold(tmp_path, checkpoint_reference):
    mf, static = checkpoint_reference
    _legacy_case(tmp_path, mf, static)
    with pytest.raises(storage.RestartMismatchError, match='no numerical SCF reference'):
        reference.scf_initial_guess(tmp_path, mf.mol)


def test_gradient_resume_omits_saved_group_with_roundoff(tmp_path, checkpoint_reference, monkeypatch):
    from pyscfad.dlno_stc.test._boys_acceptance import build_mf
    mf, static = checkpoint_reference
    controls = dict(mode='deterministic', threads=1)
    options = dict(baseline_controls=controls, matrix_controls=controls,
                   include_hf=True, checkpoint_dir=tmp_path, ccsd_t=False)
    energy, gradient = ccsd.value_and_grad(mf.mol, build_mf, static, **options)

    def rerun(current):
        new = build_mf(current)
        new.e_tot = new.e_tot + 2e-10
        return new

    def forbidden(*args, **kwargs):
        raise AssertionError('a checkpointed group was recomputed')

    monkeypatch.setattr(ccsd, '_run_group', forbidden)
    resumed_energy, resumed_gradient = ccsd.value_and_grad(
        mf.mol, rerun, static, resume=True, **options)
    np.testing.assert_allclose(resumed_energy, energy, atol=1e-8, rtol=0.)
    np.testing.assert_allclose(resumed_gradient.coords, gradient.coords, atol=1e-8, rtol=0.)


def test_gradient_resume_uses_saved_scf_without_iterations(tmp_path, checkpoint_reference, monkeypatch):
    from functools import partial
    from pyscfad import scf
    from pyscfad.tools import linear_solver
    from pyscfad.dlno_stc.test._boys_acceptance import build_mf
    # Compare equivalent response constructions below the default 1e-6
    # iterative-solver tolerance, without changing production tolerances.
    monkeypatch.setattr(scf.hf, 'gen_gmres', partial(
        linear_solver.gen_gmres_with_default_kwargs, tol=1e-10, maxiter=100))
    mf, static = checkpoint_reference
    controls = dict(mode='deterministic', threads=1)
    options = dict(baseline_controls=controls, matrix_controls=controls,
                   include_hf=True, checkpoint_dir=tmp_path, ccsd_t=False)
    energy, gradient = ccsd.value_and_grad(mf.mol, build_mf, static, **options)
    saved = reference.load_scf_reference(tmp_path, mf.mol)
    progress = (tmp_path / 'records/cc-progress/state.h5').read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError('restart repeated SCF iterations or completed fragments')

    monkeypatch.setattr(scf.hf, 'kernel', forbidden)
    monkeypatch.setattr(scf.hf, '_scf', forbidden)
    monkeypatch.setattr(scf.hf.SCF, 'get_fock', forbidden)
    monkeypatch.setattr(scf.hf.RHF, 'get_veff', forbidden)
    monkeypatch.setattr(ccsd, '_run_group', forbidden)

    def restored(current):
        obj = scf.RHF(current).density_fit()
        obj.conv_tol, obj.conv_tol_grad = mf.conv_tol, mf.conv_tol_grad
        return reference.restore_scf_reference(obj, saved)

    resumed_energy, resumed_gradient = ccsd.value_and_grad(
        mf.mol, restored, static, resume=True, **options)
    np.testing.assert_allclose(resumed_energy, energy, atol=1e-10, rtol=0.)
    np.testing.assert_allclose(resumed_gradient.coords, gradient.coords, atol=1e-8, rtol=0.)
    assert (tmp_path / 'records/cc-progress/state.h5').read_bytes() == progress


def test_saved_common_fock_is_loaded_without_building(tmp_path, checkpoint_reference, monkeypatch):
    mf, static = checkpoint_reference
    state = _state(mf)
    state['common'] = domain.rebuild_domain_data(mf, static.mp2_static)
    expected = np.asarray(state['common'].fock).copy()
    def forbidden(*args, **kwargs):
        pytest.fail('checkpoint saving/loading rebuilt a saved Fock')
    monkeypatch.setattr(type(mf), 'get_fock', forbidden)
    _open(tmp_path, mf, static, state=state).close()
    saved = reference.load_scf_reference(tmp_path, mf.mol)
    np.testing.assert_array_equal(saved['fock'], expected)


@pytest.mark.parametrize('bad_fock', [None, np.zeros((1, 1)), np.full((7, 7), np.nan)])
def test_restore_rejects_missing_or_invalid_fock(tmp_path, checkpoint_reference, bad_fock):
    mf, static = checkpoint_reference
    _open(tmp_path, mf, static).close()
    saved = reference.load_scf_reference(tmp_path, mf.mol)
    saved['fock'] = bad_fock
    with pytest.raises((storage.RestartMismatchError, storage.RestartCorruptionError), match='Fock'):
        reference.restore_scf_reference(copy(mf), saved)


def test_early_scf_reference_survives_interrupted_selection(tmp_path, checkpoint_reference, monkeypatch):
    from pyscfad import ops
    mf, static = checkpoint_reference
    captured = []
    def save_host(host):
        captured.append(np.asarray(host.mo_coeff).copy())
        reference.save_early_scf_reference(tmp_path, host)
    # Exercise the same detached save used from within the differentiable builder.
    def save(current):
        from pyscfad.dlno_stc.test._boys_acceptance import build_mf
        obj = build_mf(current)
        ops.stop_trace(save_host)(obj)
        return obj.e_tot
    jax.vjp(save, mf.mol)
    path = tmp_path / '.scf-reference.h5'
    original = path.read_bytes()
    assert reference.prepare_resume(tmp_path, True) is False
    saved = reference.load_scf_reference(tmp_path, mf.mol)
    np.testing.assert_array_equal(saved['canonical']['mo_coeff'], captured[0])
    # A partially initialized CC manifest is archived without losing the SCF.
    manager = storage.RestartManager(tmp_path, method='stc-ccsd', scientific_payload={'test': 1})
    manager.save_static(static)
    manager.close()
    with pytest.warns(RuntimeWarning, match='no fragment progress'):
        assert reference.prepare_resume(tmp_path, True) is False
    reference.load_scf_reference(tmp_path, mf.mol)
    assert path.read_bytes() == original
    changed = copy(mf)
    changed.conv_tol *= 10
    with pytest.raises(storage.RestartMismatchError, match='SCF/SGX settings'):
        reference.restore_scf_reference(changed, saved)


@pytest.mark.parametrize('complete_cc_checkpoint', [False, True])
def test_benchmark_resume_bypasses_both_scf_kernels(
        tmp_path, checkpoint_reference, monkeypatch, complete_cc_checkpoint):
    import importlib.util
    import os
    import sys
    from pathlib import Path
    from pyscf.scf import hf as pyscf_hf
    from pyscfad import scf
    from pyscfad.dlno_stc import driver
    from pyscfad.lno.ccsd import RCCSD

    mf, static = checkpoint_reference
    host = copy(mf)
    host.conv_tol, host.conv_tol_grad = 1e-9, 1e-6
    directory = tmp_path / 'checkpoint'
    if complete_cc_checkpoint:
        from pyscfad.dlno_stc import build_domain_selections
        # The runner now validates selection settings before entering CC.
        domains = build_domain_selections(host, frozen=1, boys_grouping='atom',
            lo_kwargs=dict(init_guess='atomic', conv_tol=1e-10,
                           gmres_options=dict(restart=120, maxiter=100)))
        static = ccsd.build_static_selections(host, domains, thresh_occ=1e-4,
            thresh_vir=1e-5, baseline_controls=dict(mode='deterministic', threads=1,
                                                  global_seed=666),
            matrix_controls=dict(mode='deterministic', threads=1))
        _open(directory, host, static).close()
    else:
        reference.save_early_scf_reference(directory, host)
    xyz = tmp_path / 'water.xyz'
    xyz.write_text('3\nwater\nO 0 0 0\nH .1 -.75 .57\nH 0 .8 .61\n')
    examples = Path(__file__).resolve().parents[3] / 'examples/dlno_stc'
    for name in ('PYSCFAD_DLNO_RESOURCE_PROFILE', 'PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE'):
        monkeypatch.setenv(name, os.environ.get(name, ''))
    for name in ('conv_tol', 'conv_tol_normt', 'max_cycle'):
        monkeypatch.setattr(RCCSD, name, getattr(RCCSD, name))
    monkeypatch.syspath_prepend(str(examples))
    spec = importlib.util.spec_from_file_location('restart_benchmark_test', examples / 'benchmark_lno_w_triples.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'configure_cpu_workload', lambda threads: [0])
    monkeypatch.setattr(module, 'check_worker_affinity', lambda cpus: dict(runtime_threads=1))
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--threads', '1', '--basis', 'sto-3g',
        '--xyz-file', str(xyz), '--frozen', '1', '--boys-grouping', 'atom',
        '--output-dir', str(tmp_path / 'output'), '--checkpoint-dir', str(directory), '--resume'])

    def forbidden(*args, **kwargs):
        raise AssertionError('benchmark restart ran cold or warm SCF')

    class ReachedFragments(Exception):
        pass

    def stop_before_fragments(mol, build, **kwargs):
        restored, _ = driver._initialize_scf(mol, build, share_scf_data=True)
        np.testing.assert_array_equal(restored.mo_coeff, mf.mo_coeff)
        assert kwargs['resume'] is complete_cc_checkpoint
        if complete_cc_checkpoint:
            loaded = kwargs['build_static'](restored)
            assert storage.scientific_digest(loaded) == storage.scientific_digest(static)
        raise ReachedFragments

    monkeypatch.setattr(pyscf_hf, 'kernel', forbidden)
    monkeypatch.setattr(scf.hf, 'kernel', forbidden)
    if complete_cc_checkpoint:
        from pyscfad import dlno_stc
        from pyscfad.lo import boys
        monkeypatch.setattr(dlno_stc, 'build_domain_selections', forbidden)
        monkeypatch.setattr(ccsd, 'build_static_selections', forbidden)
        monkeypatch.setattr(boys, 'kernel', forbidden)
    monkeypatch.setattr(ccsd, 'value_and_grad', stop_before_fragments)
    with pytest.raises(ReachedFragments):
        module.main()
    if complete_cc_checkpoint:
        import json
        settings = json.loads((tmp_path / 'output/settings.json').read_text())
        assert settings['selection_source'] == 'checkpoint'
        assert settings['domain_selection_seconds'] == settings['selection_seconds'] == 0.


@pytest.mark.parametrize('failure', ['unconverged', 'energy_mismatch'])
def test_early_checkpoint_requires_successful_scf(tmp_path, checkpoint_reference, failure):
    mf, _ = checkpoint_reference
    host = copy(mf)
    cold_energy = float(mf.e_tot)
    if failure == 'unconverged':
        host.converged = False
    else:
        cold_energy += 1e-3
    with pytest.raises(storage.RestartMismatchError):
        reference.save_early_scf_reference(tmp_path, host, reference_energy=cold_energy)
    assert not (tmp_path / '.scf-reference.h5').exists()
