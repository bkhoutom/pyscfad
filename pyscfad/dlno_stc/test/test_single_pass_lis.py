"""Compact virtual selection and a single live domain/W preparation tape."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad import config_update
from pyscfad.dlno_stc import ccsd, lis
from pyscfad.dlno_stc._retained_space import retained_projector
from pyscfad.dlno_stc.test import _boys_acceptance as accept
from pyscfad.dlno_stc.test.test_shared_domain_frame import molecular_group, counted_frames
from pyscfad.dlno_stc.test.test_shared_ccsd_pullback import group_case


@pytest.mark.parametrize('precision', [False, True])
def test_deferred_lis_reaches_stc_cc_once_per_fragment(group_case, monkeypatch, precision):
    """STC solves and replicas must consume the LIS selected from the live W."""
    case = group_case
    reference, _ = case.run(matrix_kind='W')
    run_group, solve_fragment = ccsd._run_group, ccsd._solve_fragment
    selected, solved = [], []

    def with_plan(state, static, *args, **options):
        static.deferred_selection = {'indices': (0, 1)}
        return run_group(state, static, *args, **options)

    def select(state, static, index, *args):
        selected.append(index)
        return SimpleNamespace(**(vars(static) | {'deferred_selection': None,
                                                  'selected_index': index}))

    def solve(mf, common, static, index, *args, **options):
        assert static.deferred_selection is None and static.selected_index == index
        solved.append(index)
        values = solve_fragment(mf, common, static, index, *args, **options)
        return values, {'ccsd': {'seed': options['stc_cc_controls']['seed']}}

    monkeypatch.setattr(ccsd, '_run_group', with_plan)
    monkeypatch.setattr(ccsd, '_select_current_fragment', select)
    monkeypatch.setattr(ccsd, '_solve_fragment', solve)
    controls = {'seed': 71}
    if precision:
        controls.update(energy_tolerance=1e3, precision_pilot_replicas=2,
                        precision_min_replicas=2, precision_max_replicas=2)
    actual, rows = case.run(matrix_kind='W', cc_solver='stc', stc_cc_controls=controls)
    assert selected == [0, 1]
    assert solved == ([0]*4 + [1]*4 if precision else [0, 1])
    assert all(row.ccsd_method == 'stc' for row in rows)
    for key in ('energy', 'baseline_energy', 'variance', 'mf_bar', 'common_bar'):
        np.testing.assert_allclose(actual[key], reference[key], rtol=2e-12, atol=2e-13)


@pytest.mark.parametrize('cutoff', [.15, .8, 1.7])
def test_compact_virtual_projector_preserves_metric_and_mapping_response(cutoff):
    rng = np.random.default_rng(812)
    q, _ = np.linalg.qr(rng.normal(size=(9, 4)))
    mapping = q @ np.array([[1.3, .2, 0., .1], [0., .8, .15, 0.],
                            [0., 0., 1.1, .2], [0., 0., 0., .7]])
    density = np.diag([-1.8, -.4, .7, 2.4])
    probe = rng.normal(size=(9, 9)); probe = (probe + probe.T)/2
    _, compact = lis._compact_mapped_density(jnp.asarray(mapping), jnp.asarray(density))
    global_keep = tuple(np.flatnonzero(np.abs(np.linalg.eigvalsh(mapping@density@mapping.T)) > cutoff))
    local_keep = tuple(np.flatnonzero(np.abs(np.linalg.eigvalsh(compact)) > cutoff))
    assert len(global_keep) == len(local_keep) > 0

    def loss(r, w, small):
        if small:
            basis, k = lis._compact_mapped_density(r, w)
            projector = basis @ retained_projector(local_keep)(k) @ basis.T
        else:
            projector = retained_projector(global_keep)(r @ w @ r.T)
        return jnp.sum(projector * probe)

    reference, ref_bars = jax.value_and_grad(loss, argnums=(0, 1))(mapping, density, False)
    value, bars = jax.value_and_grad(loss, argnums=(0, 1))(mapping, density, True)
    np.testing.assert_allclose(value, reference, atol=2e-12, rtol=0)
    for actual, expected in zip(bars, ref_bars):
        np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-10)
    direction = rng.normal(size=mapping.shape)
    h = 1e-5
    fd = (loss(mapping+h*direction, density, True)-loss(mapping-h*direction, density, True))/(2*h)
    np.testing.assert_allclose(jnp.sum(bars[0]*direction), fd, atol=2e-7, rtol=2e-7)
    assert np.linalg.norm(bars[0]) > .1


def occupied_case():
    counts = [3, 3, 3, 3, 4, 2, 3, 4, 3, 2, 3, 3, 3, 3, 3, 3, 4, 2, 3, 4]
    assert sum(counts) == 61
    boundaries = np.cumsum([0]+counts)
    domains = SimpleNamespace(active_occ_indices=np.arange(61),
        fragments=(None,)*61, boys_grouping='atom',
        frag_lolist=tuple(np.array([i]) for i in range(61)),
        physical_groups=tuple(tuple(range(a,b)) for a,b in zip(boundaries[:-1],boundaries[1:])))
    return SimpleNamespace(mp2_static=domains, deferred_selection={'indices': tuple(range(20))},
                           lis_static=SimpleNamespace(smooth_frames=(None,)*20))


@pytest.mark.parametrize('indices', [tuple(range(20)), (0, 4, 5), ()])
def test_occupied_count_budgets_use_whole_molecule_denominator(indices):
    static = occupied_case()
    supplied = dict(mode='stochastic', nsamples=16, energy_tolerance=.00015,
                    variance_allocation='occupied_count')
    result = ccsd._allocate_triples_controls(supplied, len(indices), static=static, indices=indices)
    assert 'fragment_standard_errors' not in supplied
    for i in indices:
        count = len(static.mp2_static.physical_groups[i])
        local = ccsd._fragment_triples_controls(result, i)
        assert local['fragment_occupied_count'] == count
        assert local['total_active_occupied'] == 61
        assert local['fragment_variance_fraction'] == count/61
        assert 'fragment_reference_energy' not in local
        np.testing.assert_allclose(local['target_standard_error']**2, .00015**2*count/61)
    expected = sum(len(static.mp2_static.physical_groups[i]) for i in indices)/61
    np.testing.assert_allclose(sum(result['fragment_variance_fractions'].values()), expected)


@pytest.mark.parametrize('settings', [dict(variance_allocation='bad'),
    dict(variance_allocation='occupied_count', fragment_energies=(-1.,))])
def test_invalid_variance_allocations_rejected(settings):
    with pytest.raises(ValueError):
        ccsd.normalize_triples_controls(dict(mode='stochastic', nsamples=16, **settings), ccsd_t=True)


def test_molecular_single_pass_matches_two_pass_energy_and_full_gradient(molecular_group, monkeypatch):
    case = molecular_group
    monkeypatch.setenv('PYSCFAD_LNO_SHARED_LOCAL_DF_REVERSE', '1')
    counts = counted_frames(monkeypatch)
    forwards, reverses = [], []
    solve, reverse = ccsd.backend.solve_matrices, ccsd.backend.pullback_matrices

    def counted_forward(*args, **kwargs):
        forwards.append(args[1]['fragment_id'])
        return solve(*args, **kwargs)

    def counted_reverse(*args, **kwargs):
        reverses.append(args[1]['metadata']['fragment_id'])
        return reverse(*args, **kwargs)

    monkeypatch.setattr(ccsd.backend, 'solve_matrices', counted_forward)
    monkeypatch.setattr(ccsd.backend, 'pullback_matrices', counted_reverse)
    plan = ccsd.build_static_selections(
        case.mf, case.static.mp2_static, matrix_kind='W', thresh_occ=1e-4, thresh_vir=3e-5,
        baseline_controls=case.controls, matrix_controls=case.controls,
        fragment_indices=case.indices, single_pass=True)
    assert counts == dict(forward=0, reverse=0)
    assert forwards == reverses == []
    assert plan.mp2_fragment_energies is None

    triples = ccsd.normalize_triples_controls(dict(mode='stochastic', nsamples=16,
        pilot_samples=16, global_seed=71, integral_transform='rotate'), ccsd_t=True)

    def run(static):
        counts.update(forward=0, reverse=0); forwards.clear(); reverses.clear()
        state = ccsd.driver._initialize(case.mf, static.mp2_static, case.controls,
            with_grad=True, scf_pullback=case.scf_pullback)
        rows = ccsd._run_group(state, static, case.group, case.controls, case.controls,
            with_grad=True, verbose_imp=0, ccsd_t=True, triples_controls=triples,
            fragment_indices=case.indices)
        value, grad = ccsd.driver._finish(state, with_grad=True, include_hf=False)
        return float(value), np.asarray(grad.coords), rows, counts.copy(), list(forwards), list(reverses)

    reference = run(case.static)
    actual = run(plan)
    assert actual[3] == dict(forward=1, reverse=1)
    assert actual[4:] == reference[4:]
    assert len(actual[4]) == len(set(actual[4]))
    np.testing.assert_allclose(actual[0], reference[0], atol=3e-11, rtol=0)
    np.testing.assert_allclose(actual[1], reference[1], atol=3e-9, rtol=3e-7)
    for a, b in zip(actual[2], reference[2]):
        assert (a.nocc_lis, a.nvir_lis) == (b.nocc_lis, b.nvir_lis)
        assert a.triples_samples == b.triples_samples
        np.testing.assert_allclose([a.e_mp2_lis, a.e_ccsd, a.e_triples],
                                   [b.e_mp2_lis, b.e_ccsd, b.e_triples], atol=3e-11, rtol=0)
    accept.record_result('single_pass_domain_lis', dict(
        energy_difference_eh=actual[0]-reference[0],
        max_gradient_difference_eh_per_bohr=float(np.max(np.abs(actual[1]-reference[1]))),
        frame_counts=actual[3], w_forward_fragments=actual[4], w_reverse_fragments=actual[5],
        corrected_fragments=list(case.indices), lis_eigenspace='domain'))


def test_public_gradient_factory_with_occupied_count_budget(molecular_group, monkeypatch):
    """Exercise detached selection, live S/F reuse and total-SE allocation together."""
    case = molecular_group
    monkeypatch.setenv('PYSCFAD_LNO_SHARED_LOCAL_DF_REVERSE', '1')
    plans = []

    def build_static(mf):
        plan = ccsd.build_static_selections(mf, case.static.mp2_static,
            matrix_kind='W', thresh_occ=1e-4, thresh_vir=3e-5,
            baseline_controls=case.controls, matrix_controls=case.controls,
            fragment_indices=case.indices, single_pass=True)
        plans.append(plan)
        return plan

    options = dict(baseline_controls=case.controls, matrix_controls=case.controls,
        ccsd_t=True, fragment_indices=case.indices, include_baseline=False,
        return_components=True, triples_controls=dict(mode='stochastic', nsamples=16,
            pilot_samples=16, global_seed=71, integral_transform='rotate',
            variance_allocation='occupied_count', energy_tolerance=1e-3))
    with (config_update('pyscfad_scf_implicit_diff', True),
          config_update('pyscfad_scf_first_order_custom', False)):
        reference = ccsd.value_and_grad(case.mf.mol, accept.build_mf, case.static, **options)
        actual = ccsd.value_and_grad(case.mf.mol, accept.build_mf,
                                    build_static=build_static, **options)
    assert len(plans) == 1
    assert plans[0].mp2_fragment_energies is None
    np.testing.assert_allclose(actual[0], reference[0], atol=3e-11, rtol=0)
    difference = float(np.max(np.abs(actual[1].coords-reference[1].coords)))
    np.testing.assert_allclose(actual[1].coords, reference[1].coords, atol=3e-9, rtol=3e-7)
    assert len(actual[2].fragments) == len(case.indices)
    for row in actual[2].fragments:
        assert row.triples_diagnostics['variance_allocation'] == 'occupied_count'
        assert row.triples_diagnostics['total_active_occupied'] == 8
        assert row.triples_diagnostics['fragment_occupied_count'] == 1
        np.testing.assert_allclose(row.triples_diagnostics['fragment_variance_fraction'], 1/8)
    accept.record_result('single_pass_public_gradient', dict(
        energy_difference_eh=float(actual[0]-reference[0]),
        max_gradient_difference_eh_per_bohr=difference,
        corrected_fragments=list(case.indices), variance_allocation='occupied_count'))
