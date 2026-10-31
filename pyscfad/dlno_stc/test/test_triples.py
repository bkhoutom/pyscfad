"""Triples controls/checkpoints and independent deterministic LNO references."""
import importlib
import inspect
import os
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from pyscfad.dlno_stc import ccsd, controls

TRIPLES_KEYS = ('t1', 't2', 'Bvo', 'Boo', 'Bvv', 'eocc', 'evir', 'U', 'fov')


def triples_native():
    if importlib.util.find_spec('pyscfad.dlno_stc._stc_mp2') is None:
        if os.environ.get('PYSCFAD_REQUIRE_STC_EXTENSION'):
            pytest.fail('STC C++ extension has not been built')
        pytest.skip('optional STC C++ extension is not built')
    module = importlib.import_module('pyscfad.dlno_stc._stc_mp2')
    assert hasattr(module, 'triples_asymmetric_energy'), 'native triples binding is missing'
    assert hasattr(module, 'triples_asymmetric_energy_vjp'), 'native triples VJP binding is missing'
    return module


def triples_fixture(*, grouped=False):
    """Real DF factors and pair-symmetric amplitudes, with mixed occupied U."""
    rng = np.random.default_rng(1907)
    no, nv, naux = (3 if grouped else 2), 3, 4
    t1 = rng.normal(scale=.02, size=(no, nv))
    t2 = rng.normal(scale=.03, size=(no, no, nv, nv))
    t2 = .5 * (t2 + t2.transpose(1, 0, 3, 2))
    Bvo = rng.normal(scale=.15, size=(naux, no, nv))
    Boo = rng.normal(scale=.15, size=(naux, no, no))
    Boo = .5 * (Boo + Boo.swapaxes(1, 2))
    Bvv = rng.normal(scale=.15, size=(naux, nv, nv))
    Bvv = .5 * (Bvv + Bvv.swapaxes(1, 2))
    return dict(zip(TRIPLES_KEYS, (
        t1, t2, Bvo, Boo, Bvv,
        np.array([-1.1, -.8, -.6]) if grouped else np.array([-1.1, -.6]),
        np.array([.2, .5, .9]),
        np.array([[.6, 0.], [-.8, 0.], [0., 1.]]) if grouped else
        np.array([[.6], [-.8]]), np.zeros((no, nv)))))


def triples_reference(t1, t2, Bvo, Boo, Bvv, eocc, evir, U, fov):
    """Independent canonical reference, used only by scientific tests."""
    return triples_matrix_reference(t1, t2, Bvo, Boo, Bvv, eocc, evir, U @ U.T, fov)


def triples_matrix_reference(t1, t2, Bvo, Boo, Bvv, eocc, evir, M, fov):
    """Slow deterministic reference with an arbitrary occupied weight."""
    from pyscfad.lno.ccsd_t_slow import _compute_et
    no, nv = t1.shape
    ovvv = jnp.einsum('Pia,Pbc->iabc', Bvo, Bvv)
    ovoo = jnp.einsum('Pia,Pjk->iajk', Bvo, Boo)
    ovov = jnp.einsum('Pia,Pjb->iajb', Bvo, Bvo)
    # Enumerate all virtual triples, including independent tensor directions.
    # Folding b/c by symmetry would only validate derivatives projected onto
    # the physical symmetry manifold, not every native input cotangent.
    indices = [(a, b, c) for a in range(nv)
               for b in range(nv) for c in range(nv)]
    factors = [2.] * len(indices)
    eijk = eocc[:, None, None] + eocc[None, :, None] + eocc[None, None, :]
    return _compute_et(
        M, t1, t2, ovvv.transpose(1, 3, 0, 2),
        ovoo.transpose(1, 0, 3, 2), ovov.transpose(1, 3, 0, 2),
        fov.T, eijk, evir,
        jnp.asarray(indices, dtype=jnp.int32), jnp.asarray(factors))


def test_impurity_triples_hook_preserves_default_and_returns_aux():
    from pyscfad import gto, scf
    from pyscfad.lno.ccsd import _impurity_solve_core
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 1.4'
    mol.basis = 'sto-3g'
    mol.unit = 'Bohr'
    mol.verbose = 0
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = scf.RHF(mol).density_fit().run()
    args = (mf, mf.mo_coeff, mf.mo_coeff[:, :1], mf.get_fock(), mf.get_ovlp())
    baseline = _impurity_solve_core(*args, ccsd_t=True)
    assert len(baseline) == 3
    seen = []

    def replacement(mcc, eris, prjlo, *, t1, t2, verbose=0):
        seen.append((prjlo, t1, t2))
        return jnp.asarray(-.25), {'samples': jnp.asarray(13)}

    components, aux = _impurity_solve_core(
        *args, ccsd_t=True, triples_solver=replacement)
    assert len(seen) == 1
    np.testing.assert_allclose(components[:2], baseline[:2], atol=1e-13)
    assert float(components[2]) == -.25
    assert int(aux['samples']) == 13
    seen.clear()
    no_triples, aux = _impurity_solve_core(
        *args, ccsd_t=False, triples_solver=replacement)
    assert not seen
    assert float(no_triples[2]) == 0.
    assert aux == {}


# Outer controls, fragment diagnostics and serial checkpoint identity.
def test_triples_defaults_are_independent_and_copy_controls():
    normalize = getattr(controls, 'normalize_triples_controls', None)
    assert callable(normalize), 'triples controls need their own normalizer'
    assert normalize(None, ccsd_t=False) is None
    assert normalize(None, ccsd_t=True) == {'mode': 'deterministic'}
    supplied = {'mode': 'stochastic', 'nsamples': np.int64(7)}
    assert normalize(supplied, ccsd_t=True) == {'mode': 'stochastic', 'nsamples': 7, 'global_seed': 71}
    assert supplied == {'mode': 'stochastic', 'nsamples': 7}
    assert normalize({'mode': 'stochastic', 'nsamples': 2, 'global_seed': 2**64-1}, ccsd_t=True)['global_seed'] == 2**64-1


@pytest.mark.parametrize('supplied', [{}, {'mode': 'deterministic'}, {'mode': 'stochastic', 'nsamples': 2}])
def test_disabled_triples_reject_explicit_controls(supplied):
    normalize = getattr(controls, 'normalize_triples_controls', None)
    assert callable(normalize), 'triples controls need their own normalizer'
    with pytest.raises(ValueError, match='ccsd_t'):
        normalize(supplied, ccsd_t=False)


@pytest.mark.parametrize('supplied', [
    {'mode': 'unknown'}, {'mode': 'stochastic'},
    {'mode': 'stochastic', 'nsamples': 0}, {'mode': 'stochastic', 'nsamples': 1},
    {'mode': 'stochastic', 'nsamples': True}, {'mode': 'stochastic', 'nsamples': 2.0},
    {'mode': 'stochastic', 'nsamples': np.bool_(True)},
    {'mode': 'stochastic', 'nsamples': 2, 'global_seed': -1},
    {'mode': 'stochastic', 'nsamples': 2, 'global_seed': 2**64},
    {'mode': 'stochastic', 'nsamples': 2, 'global_seed': True},
    {'mode': 'stochastic', 'nsamples': 2, 'global_seed': 1.0},
    {'mode': 'deterministic', 'nsamples': 2},
    {'mode': 'deterministic', 'global_seed': 71},
    {'mode': 'stochastic', 'nsamples': 2, 'target_standard_error': 1e-4},
])
def test_invalid_triples_controls_reject_unavailable_sampling_modes(supplied):
    normalize = getattr(controls, 'normalize_triples_controls', None)
    assert callable(normalize), 'triples controls need their own normalizer'
    with pytest.raises(ValueError):
        normalize(supplied, ccsd_t=True)


@pytest.mark.parametrize('api', [ccsd.kernel, ccsd.value_and_grad])
def test_public_triples_validation_precedes_reference_work(api):
    assert 'triples_controls' in inspect.signature(api).parameters, 'public triples option missing'
    args = (None, None)
    with pytest.raises(ValueError, match='ccsd_t'):
        api(*args, baseline_controls={}, matrix_controls={}, triples_controls={})


def test_fragment_seeds_do_not_depend_on_order_and_are_distinct():
    derive = getattr(ccsd, '_triples_fragment_seed', None)
    assert callable(derive), 'fragment-owned triples seed derivation missing'
    forward = {i: derive(71, i) for i in range(100)}
    reverse = {i: derive(71, i) for i in reversed(range(100))}
    assert forward == reverse
    assert len(set(forward.values())) == 100
    assert all(0 <= seed < 2**64 for seed in forward.values())
    assert derive(72, 0) != derive(71, 0)


def test_component_triples_uncertainty_is_separate_from_baseline():
    f = ccsd.FragmentComponents(0, -1., -2., -.1, -9.)
    assert getattr(f, 'triples_method', None) == 'none', 'missing default triples diagnostics'
    assert f.triples_standard_error == 0. and f.triples_samples == 0
    assert f.triples_timings is None
    rows = (replace(f, triples_standard_error=.03), replace(f, fragment_index=1, triples_standard_error=.04))
    report = ccsd.CCSDComponents(-5., 0., rows, 'D', .8)
    assert report.triples_standard_error == pytest.approx(.05)
    assert report.baseline_standard_error == .8
    assert report.e_corr == pytest.approx(-7.2)


@pytest.fixture(scope='module')
def checkpoint_reference():
    from pyscfad.dlno_stc.test.test_backend import native
    native()
    from pyscfad import gto
    from pyscfad.dlno_stc import build_domain_selections
    from pyscfad.dlno_stc.test._boys_acceptance import build_mf
    mol = gto.Mole(atom='O 0 0 0; H .1 -.75 .57; H 0 .8 .61', basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = build_mf(mol)
    domain = build_domain_selections(mf, frozen=1, force_full_domains=True,
        lo_kwargs={'conv_tol': 1e-12, 'conv_tol_grad': 1e-9})
    settings = {'mode': 'deterministic', 'threads': 1}
    static = ccsd.build_static_selections(mf, domain, smooth=False, matrix_kind='D', thresh_occ=0., thresh_vir=0.,
        baseline_controls=settings, matrix_controls=settings)
    return mf, static


def test_checkpoint_rejects_changed_triples_controls_and_native_version(tmp_path, monkeypatch, checkpoint_reference):
    from pyscfad.dlno_stc import _stc_mp2
    from pyscfad.dlno_stc._ccsd_checkpoint import open_checkpoint, save_progress, restore_progress
    from pyscfad.dlno_stc._restart import RestartMismatchError
    mf, static = checkpoint_reference
    groups = ccsd.build_domain_groups(static.mp2_static)
    # Real checkpoint write/resume must incorporate native triples version even
    # when only completed numerical values (no native solver handles) are saved.
    monkeypatch.setattr(_stc_mp2, 'TRIPLES_VERSION', 'outer-test-v1', raising=False)
    options = dict(ccsd_t=True, triples_controls={'mode':'stochastic', 'nsamples':2, 'global_seed':71})
    state = dict(mf=mf, energy=-.2, variance=.01, baseline_energy=-.1, weak_energy=0.)
    manager = open_checkpoint(state, static, groups, checkpoint_dir=tmp_path, resume=False, options=options)
    fragment = ccsd.FragmentComponents(0, -.1, -.2, -.01, 0.,
        triples_method='stochastic', triples_standard_error=.003, triples_samples=2,
        triples_timings={'sampling_seconds': .02},
        triples_diagnostics={'timings': {'sampling_seconds': .02}, 'setup_threads': 1})
    save_progress(manager, state, [fragment], 1, False, with_grad=False)
    manager.close()
    manager = open_checkpoint(state, static, groups, checkpoint_dir=tmp_path, resume=True, options=options)
    assert restore_progress(manager, state, with_grad=False) == (1, False, [fragment])
    manager.close()
    for changed in ({**options, 'triples_controls': {**options['triples_controls'], 'nsamples':3}},
                    {**options, 'triples_controls': {**options['triples_controls'], 'global_seed':72}},
                    {**options, 'triples_controls': {'mode':'deterministic'}}):
        with pytest.raises(RestartMismatchError):
            open_checkpoint(state, static, groups, checkpoint_dir=tmp_path, resume=True, options=changed)
    monkeypatch.setattr(_stc_mp2, 'TRIPLES_VERSION', 'outer-test-v2')
    with pytest.raises(RestartMismatchError):
        open_checkpoint(state, static, groups, checkpoint_dir=tmp_path, resume=True, options=options)


@pytest.mark.parametrize('with_grad', [False, True])
def test_group_carries_triples_diagnostics_through_response(monkeypatch, with_grad):
    from types import SimpleNamespace
    import jax
    import jax.numpy as jnp
    from pyscfad.dlno_stc.test.test_full_kernel import full_fixture
    from pyscfad.dlno_stc.test.test_backend import native
    native()
    x = full_fixture()
    base = ccsd.driver._driver_controls({'mode': 'deterministic', 'threads': 1})
    group = SimpleNamespace(representative_index=0, fragment_indices=(0,), target_indices=(0,))
    static = SimpleNamespace(mp2_static=SimpleNamespace(frag_lolist=((0,),), fragments=(None,)), matrix_kind='D')
    settings = {'mode': 'stochastic', 'nsamples': 2, 'global_seed': 71}
    def prepare(mf, common):
        return {k: jnp.asarray(v) * (1 + .07 * mf + .03 * common) for k, v in x.items()}
    def prepared(state, *args):
        inputs, pullback = jax.vjp(prepare, state['mf'], state['common'])
        return inputs, ccsd.backend.host_inputs(inputs), pullback
    def fragment(mf, common, static, index, matrices, group, **options):
        assert options['triples_controls'] == settings
        assert options['with_grad'] == with_grad
        d, v = matrices['occupied'], matrices['virtual']
        values = (.02 * mf, (.7 + mf) * jnp.sum(v*v) + common * jnp.sum(d), .013*common)
        return values, {'standard_error': .003, 'samples': 2, 'sampling_seconds': .02,
                        'probability_setup_seconds': .01, 'setup_threads': 1, 'sampling_threads': 1,
                        'pilot_samples': 0, 'pilot_standard_error': 0.,
                        'amplitude_packing_seconds': .004, 'vjp_reverse_seconds': .005,
                        'timings': {'sampling_seconds': jnp.asarray(.02)}}
    monkeypatch.setattr(ccsd.driver, '_prepare', prepared)
    monkeypatch.setattr(ccsd, '_group_cc_fragments', lambda static, group: ((0, (0,)),))
    monkeypatch.setattr(ccsd, '_solve_fragment', fragment)
    state = dict(mf=jnp.asarray(.2), common=jnp.asarray(-.1), energy=jnp.zeros(()),
                 variance=0., mf_bar=jnp.zeros(()), common_bar=jnp.zeros(()))
    records = ccsd._run_group(state, static, group, base, {'mode':'deterministic', 'threads':1},
        with_grad=with_grad, verbose_imp=0, ccsd_t=True, triples_controls=settings,
        include_baseline=False, fragment_indices=(0,))
    row, = records
    assert row.triples_method == 'stochastic'
    assert row.triples_standard_error == .003 and row.triples_samples == 2
    assert row.triples_timings == {'sampling_seconds': .02, 'probability_setup_seconds': .01,
        'amplitude_packing_seconds': .004, 'vjp_reverse_seconds': .005}
    assert row.triples_diagnostics['setup_threads'] == 1
    assert row.triples_diagnostics['pilot_samples'] == 0
    import json
    assert json.loads(json.dumps(row.triples_diagnostics))['timings']['sampling_seconds'] == .02
    if with_grad:
        assert row.direct_host_bar_norm > 0.
        assert row.native_host_bar_norm is None
        assert np.isfinite(float(state['mf_bar'])) and np.isfinite(float(state['common_bar']))
        assert abs(float(state['mf_bar'])) > 0. and abs(float(state['common_bar'])) > 0.


def test_deterministic_default_resume_accepts_explicit_mode_without_new_native(tmp_path, monkeypatch, checkpoint_reference):
    from pyscfad.dlno_stc import _stc_mp2
    mf, static = checkpoint_reference
    monkeypatch.delattr(_stc_mp2, 'TRIPLES_VERSION', raising=False)
    settings = {'mode': 'deterministic', 'threads': 1}
    options = dict(baseline_controls=settings, matrix_controls=settings, ccsd_t=True,
                   fragment_indices=(0,), include_baseline=False, return_components=True,
                   checkpoint_dir=tmp_path)
    default = ccsd.kernel(mf, static, **options)
    explicit = ccsd.kernel(mf, static, triples_controls={'mode':'deterministic'}, resume=True, **options)
    assert float(default[0]) == float(explicit[0])
    assert default[1] == explicit[1]
    assert default[1].fragments[0].triples_method == 'deterministic'
    assert default[1].triples_standard_error == 0.


def test_cc_mpi_worker_supports_frozen_orbital_lists_with_jax_initial_values():
    """Static worker occupations must work with PySCF's frozen-list count."""
    from pyscfad import gto, scf
    from pyscfad.dlno_stc import _mpi_ccsd
    from pyscfad.dlno_stc._mpi import _reference_signature
    from pyscfad.lno.ccsd import RCCSD

    mol = gto.Mole(atom='H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3',
                   basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    reference = scf.RHF(mol).density_fit()
    canonical = dict(mo_coeff=np.eye(4), mo_energy=np.array([-1., -.5, .2, .8]),
                     mo_occ=np.array([2., 2., 0., 0.]), e_tot=np.array(-1.2))
    for name, value in canonical.items():
        setattr(reference, name, value)

    def initialize(initial):
        worker = scf.RHF(mol).density_fit()
        for name, value in initial.items():
            setattr(worker, name, jnp.asarray(value))
        worker.converged = True
        return worker

    build_worker = getattr(_mpi_ccsd, '_build_cc_worker', None)
    assert callable(build_worker), 'CC worker metadata boundary is missing'
    worker = build_worker(initialize, canonical, _reference_signature(reference))
    cc = RCCSD(worker, frozen=np.array([0, 3], dtype=np.int32))
    assert cc.nocc == 1 and cc.nmo == 2
    for name, expected in canonical.items():
        actual = getattr(worker, name)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype


def test_cc_mpi_worker_preserves_disk_df_factors_for_finite_domain(tmp_path):
    """A worker must retain the root's global-auxiliary finite-domain fitting."""
    from pyscf import df as conventional_df
    from pyscfad import gto, scf
    from pyscfad.dlno_stc._mpi import _reference_signature
    from pyscfad.dlno_stc._mpi_ccsd import _build_cc_worker
    from pyscfad.lno.df import transform_df_to_mo

    mol = gto.Mole(atom='O 0 0 0; H .1 -.75 .57; H 0 .8 .61; '
                        'O 0 0 8; H .13 -.74 8.57; H .02 .81 8.61',
                   basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    path = str(tmp_path / 'global-df.h5')
    reference = scf.RHF(mol).density_fit()
    reference.with_df.attach_outcore_cderi(path)
    conventional_df.outcore.cholesky_eri(
        mol.to_pyscf(), path, auxmol=reference.with_df.auxmol.to_pyscf(), verbose=0)
    nao = mol.nao
    canonical = dict(mo_coeff=np.eye(nao), mo_energy=np.linspace(-1., 1., nao),
                     mo_occ=np.array([2.] * 10 + [0.] * (nao - 10)), e_tot=np.array(-150.))
    for name, value in canonical.items():
        setattr(reference, name, value)

    def initialize(initial):
        worker = scf.RHF(mol).density_fit()
        worker.with_df.attach_outcore_cderi(path)
        for name, value in initial.items():
            setattr(worker, name, jnp.asarray(value))
        worker.converged = True
        return worker

    worker = _build_cc_worker(initialize, canonical, _reference_signature(reference))
    local = (0, 1, 2)
    coeff = jnp.asarray(canonical['mo_coeff'][:, :4])
    root_factors = transform_df_to_mo(reference, coeff, (0, 2, 2, 4), atmlst=local)
    worker_factors = transform_df_to_mo(worker, coeff, (0, 2, 2, 4), atmlst=local)
    np.testing.assert_allclose(worker_factors, root_factors, atol=1e-13, rtol=0.)
