"""Production uses selected groups and allocates its total budget once."""
import numpy as np


def test_selected_host_backend_uses_actual_ordered_targets():
    from pyscfad.dlno_stc import backend, _stc_mp2
    inputs = dict(foo=np.diag([-.8, -.5]), fvv=np.diag([.3, .7]),
                  B=np.random.default_rng(3).normal(size=(3, 2, 2)))
    controls = dict(mode='deterministic', laplace_roots=[.2], laplace_weights=[.8])
    result = backend.solve(inputs, {'fragment_id': 3, 'target_indices': (1,)}, controls)
    expected = _stc_mp2.solve_selected(*inputs.values(), (1,), controls, with_grad=True)
    np.testing.assert_allclose(result['energy'], expected['energy'])
    for key in inputs:
        np.testing.assert_allclose(result['cotangents'][key], expected['cotangents'][key])


def test_total_budget_uses_actual_group_jobs_without_mutating_controls():
    from pyscfad.dlno_stc import _workflow
    controls = {'energy_tolerance': .0003, 'production_samples': 100}
    allocate = getattr(_workflow, 'allocate_domain_controls', None)
    assert callable(allocate), 'shared strong-job budget allocator is missing'
    assert allocate(controls, 4)['energy_tolerance'] == .000075
    assert allocate(controls, 16)['energy_tolerance'] == .00001875
    assert allocate(controls, 0) == controls
    assert controls == {'energy_tolerance': .0003, 'production_samples': 100}


def test_driver_accepts_saved_partner_order():
    from pyscfad.dlno_stc.driver import _validate_static
    from pyscfad.dlno_stc.test.test_domain_grouping import grouping_fixture
    static = grouping_fixture()
    before = [fragment.strong_fragments.copy() for fragment in static.fragments]
    _validate_static(static)
    for fragment, order in zip(static.fragments, before):
        np.testing.assert_array_equal(fragment.strong_fragments, order)


def test_direct_sixteen_targets_four_groups_budget(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from pyscfad.dlno_stc import driver
    from pyscfad.dlno_stc.test.test_domain_grouping import grouping_fixture
    static = grouping_fixture()
    template = static.fragments[0]
    fragments = tuple(replace(template, fragment_index=i, iao_indices=np.array([i]),
        strong_fragments=np.arange(4*(i//4), 4*(i//4+1)),
        strong_occ_metric_keep=np.arange(4), target_index=i%4,
        extended_atoms=np.array([i//4])) for i in range(16))
    static = replace(static, active_occ_indices=np.arange(16), fragments=fragments,
        target_reference_coords=np.zeros((4, 3)),
        frag_lolist=tuple(np.array([i]) for i in range(16)), frag_atmlist=(None,)*16,
        strong_mask=np.equal.outer(np.arange(16)//4, np.arange(16)//4))
    def initialize():
        return dict(energy=np.float64(0), variance=0., weak_pairs=((0, 4),),
                    mf=None, common=None, mf_bar=None, common_bar=None)
    monkeypatch.setattr(driver, '_prepare', lambda *a: (None, {}, None))
    monkeypatch.setattr(driver, 'weak_pair_energy', lambda *a: np.float64(-.25))
    monkeypatch.setattr(driver, '_finish', lambda state, **k: state['energy'])
    captured = []
    def capture(inputs, metadata, controls, backend, **kwargs):
        captured.append((metadata, controls.copy()))
        return dict(energy=np.float64(-len(metadata['target_indices'])), energy_standard_error=0.)
    monkeypatch.setattr(driver, 'run_backend', capture)
    controls = dict(mode='stochastic', energy_tolerance=.0003, production_samples=32)
    for merge, count, tolerance in ((True, 4, 7.5e-5), (False, 16, 1.875e-5)):
        captured.clear()
        value = driver._run_domain(initialize, static, controls, None,
                                   with_grad=False, merge_identical=merge)
        assert value == -16.25
        assert len(captured) == count
        assert sum(len(md['target_indices']) for md, _ in captured) == 16
        for md, settings in captured:
            assert settings['energy_tolerance'] == tolerance
            assert settings['production_samples'] == 32
            assert md['budget']['group_count'] == count
    assert controls == dict(mode='stochastic', energy_tolerance=.0003, production_samples=32)
