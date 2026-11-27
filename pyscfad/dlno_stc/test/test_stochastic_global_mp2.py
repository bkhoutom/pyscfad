"""Stochastic global MP2 uses one scalar solve, independent of local V/CC."""
import numpy as np
import pytest

from pyscfad import config_update
from pyscfad.dlno_stc import backend, ccsd, _global_mp2
from pyscfad.dlno_stc.test.test_global_v_workflow import reference, build_mf, select, forbidden
from pyscfad.dlno_stc.test.test_backend import native


def sampling_controls(*, adaptive=False, cutoff=.2):
    controls = dict(mode='stochastic', global_seed=19, system_workload_cutoff=cutoff,
                    auxiliary_group_size=4, pilot_samples=32, min_production_samples=32)
    if adaptive:
        controls['energy_tolerance'] = .004
    else:
        controls['production_samples'] = 2000
    return controls


@pytest.mark.parametrize('adaptive', [False, True])
def test_stochastic_baseline_is_full_system_and_reports_one_uncertainty(reference, monkeypatch, adaptive):
    extension = native()
    mf, domains = reference
    controls = sampling_controls(adaptive=adaptive)
    selected = select(mf, domains, baseline_controls=controls)
    calls = []
    original = backend.solve

    def solve(inputs, metadata, settings, **kwargs):
        result = original(inputs, metadata, settings, **kwargs)
        calls.append((inputs, metadata, settings, result))
        return result

    monkeypatch.setattr(backend, 'solve', solve)
    monkeypatch.setattr(ccsd, '_run_weak', forbidden)
    energy, parts = ccsd.kernel(mf, selected, baseline_controls=controls,
                                return_components=True, mp2_aux_svd_tol=None)
    assert len(calls) == 1
    inputs, metadata, settings, result = calls[0]
    assert metadata['target_indices'] == tuple(range(len(domains.active_occ_indices)))
    assert inputs['B'].shape[1:] == (len(domains.active_occ_indices), len(domains.active_vir_indices))
    assert result['diagnostics']['actual_sample_count'] > 0
    assert parts.baseline_standard_error > 0.
    assert parts.e_mp2_base == result['energy']
    assert parts.baseline_standard_error == result['energy_standard_error']
    assert parts.e_mp2_weak == 0.
    assert all(row.matrix_diagnostics['mode'] == 'deterministic' for row in parts.fragments)
    np.testing.assert_allclose(energy, parts.e_corr, atol=1e-14)
    # Direct whole-system entry point, with the exact derived stream and the
    # same auxiliary partition, must reproduce the global selected-target call.
    arrays = backend.host_inputs(inputs)
    naux = arrays['B'].shape[0]
    offsets = np.array([*range(0, naux, controls['auxiliary_group_size']), naux])
    direct = extension.solve_full(arrays['foo'], arrays['fvv'], arrays['B'],
        dict(settings, global_seed=result['metadata']['effective_seed']),
        aux_offsets=offsets, with_grad=False)
    if not adaptive:
        # Adaptive allocation may use measured pilot timings, so exact stream
        # reproducibility is asserted for fixed production counts.
        assert direct['energy'] == result['energy']
        assert direct['energy_standard_error'] == result['energy_standard_error']
    deterministic = original(inputs, metadata,
        {key: settings[key] for key in ('laplace_roots', 'laplace_weights')}, with_grad=False)
    assert abs(result['energy'] - deterministic['energy']) < 7 * result['energy_standard_error']
    if adaptive:
        targets = np.asarray(result['diagnostics']['point_variance_targets'])
        np.testing.assert_allclose(np.sum(targets * settings['laplace_weights']**2),
                                   controls['energy_tolerance']**2, rtol=1e-12)
    else:
        repeated, repeated_parts = ccsd.kernel(mf, selected, baseline_controls=controls,
                                               return_components=True, mp2_aux_svd_tol=None)
        assert repeated_parts.e_mp2_base == parts.e_mp2_base
        assert repeated_parts.baseline_standard_error == parts.baseline_standard_error
        np.testing.assert_allclose(repeated, energy, atol=1e-14)


def test_interrupted_restart_keeps_stochastic_baseline_and_variance(reference, monkeypatch, tmp_path):
    extension = native()
    mf, domains = reference
    controls = sampling_controls()
    selected = select(mf, domains, baseline_controls=controls)
    options = dict(baseline_controls=controls, return_components=True,
                   checkpoint_dir=tmp_path, mp2_aux_svd_tol=None)
    saved = {}
    original = _global_mp2.accumulate

    def accumulate(state, *args, **kwargs):
        original(state, *args, **kwargs)
        saved.update(energy=state['baseline_energy'], variance=state['variance'])

    def interrupt(*args, **kwargs):
        raise RuntimeError('interrupt after committing global MP2')

    with monkeypatch.context() as patch:
        patch.setattr(_global_mp2, 'accumulate', accumulate)
        patch.setattr(ccsd, '_run_group', interrupt)
        with pytest.raises(RuntimeError, match='interrupt after'):
            ccsd.kernel(mf, selected, **options)
    assert saved['variance'] > 0.
    monkeypatch.setattr(_global_mp2, 'accumulate', forbidden)
    monkeypatch.setattr(ccsd, '_run_weak', forbidden)
    energy, parts = ccsd.kernel(mf, selected, resume=True, **options)
    assert parts.e_mp2_base == saved['energy']
    assert parts.baseline_standard_error == saved['variance']**.5
    np.testing.assert_allclose(energy, parts.e_corr, atol=1e-14)
    with pytest.raises(ValueError, match='scientific controls'):
        ccsd.kernel(mf, selected, baseline_controls=dict(controls, global_seed=20))
    with pytest.raises(ValueError, match='scientific controls'):
        ccsd.kernel(mf, selected, baseline_controls={'mode': 'deterministic'})
    # A V matrix provider version alone cannot identify a sampled MP2 baseline.
    with monkeypatch.context() as patch:
        patch.setattr(extension, 'matrix_kernel_version', 'incompatible-stochastic-provider')
        from pyscfad.dlno_stc._restart import RestartMismatchError
        with pytest.raises(RestartMismatchError, match='global_mp2_version'):
            ccsd.kernel(mf, selected, resume=True, **options)


def test_stochastic_baseline_gradient_and_all_kept_limit(reference, monkeypatch):
    native()
    mf, domains = reference
    monkeypatch.setattr(ccsd, '_run_weak', forbidden)
    options = dict(return_components=True, mp2_aux_svd_tol=None)
    with config_update('pyscfad_scf_implicit_diff', True):
        exact = ccsd.value_and_grad(mf.mol, build_mf, select(mf, domains), **options)
        kept_controls = sampling_controls(cutoff=1e-100)
        kept = ccsd.value_and_grad(mf.mol, build_mf,
            select(mf, domains, baseline_controls=kept_controls),
            baseline_controls=kept_controls, **options)
        np.testing.assert_allclose(kept[0], exact[0], atol=3e-12, rtol=0)
        np.testing.assert_allclose(kept[1].coords, exact[1].coords, atol=3e-10, rtol=0)
        assert kept[2].baseline_standard_error == 0.
        controls = sampling_controls()
        selected = select(mf, domains, baseline_controls=controls)
        sampled = ccsd.value_and_grad(mf.mol, build_mf, selected, baseline_controls=controls, **options)
    forward, parts = ccsd.kernel(mf, selected, baseline_controls=controls, **options)
    assert sampled[2].baseline_standard_error > 0.
    assert np.isfinite(sampled[1].coords).all()
    np.testing.assert_allclose(sampled[2].e_mp2_base, parts.e_mp2_base, atol=2e-13, rtol=0)
    # A freshly converged SCF/CC response run can differ from the cached forward
    # solve at the solver's convergence tolerance, independently of sampling.
    np.testing.assert_allclose(sampled[0], forward, atol=2e-11, rtol=0)
    np.testing.assert_allclose(sampled[2].baseline_standard_error, parts.baseline_standard_error, atol=2e-14)
