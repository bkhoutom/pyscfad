"""ED plans become live frames only when their fragment is prepared."""
from dataclasses import replace
import gc
import weakref
import jax
import numpy as np
import pytest

from pyscfad.dlno_stc import _selection, domain, driver
from pyscfad.dlno_stc.test.test_boys_prepare import boys_water_dimer
from pyscfad.dlno_stc.test.test_shared_domain_frame import molecular_group


def _plan(mf, reference):
    return _selection.build_domain_selections(
        mf, frozen=reference.frozen, target_coeff=reference.target_reference_coeff,
        thresholds=reference.thresholds)


def test_global_selection_does_not_build_ed_spaces_or_qr(boys_water_dimer, monkeypatch):
    mf, _, reference = boys_water_dimer
    old = _selection._reference_pao_overlap_selection
    def primary_only(*args, **kwargs):
        assert kwargs['completeness_threshold'] != reference.thresholds.ed_pao
        return old(*args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError('ED QR must run in the fragment loop')
    monkeypatch.setattr(_selection, '_reference_pao_overlap_selection', primary_only)
    monkeypatch.setattr(_selection, '_select_virtual_anchor_columns', forbidden)
    plan = _plan(mf, reference)
    assert plan.ed_selection_pending
    assert all(f.strong_virtual is None and f.virtual_anchor_columns is None for f in plan.fragments)
    for got, want in zip(plan.fragments, reference.fragments):
        for name in ('strong_fragments', 'extended_atoms', 'extended_ao_indices', 'pao_center_atoms'):
            np.testing.assert_array_equal(getattr(got, name), getattr(want, name))
    assert _selection.build_domain_groups(plan) == _selection.build_domain_groups(reference)


@pytest.mark.parametrize('with_grad', [False, True])
def test_one_fragment_reuses_selected_frame_and_preserves_packet_response(
        boys_water_dimer, monkeypatch, with_grad):
    from pyscfad.dlno_stc import _ed_selection
    mf, common, reference = boys_water_dimer
    plan = _plan(mf, reference)
    group = _selection.build_domain_groups(plan)[0]
    state = dict(mf=mf, common=common)
    controls = dict(mode='deterministic', threads=1)
    calls = []
    select = _ed_selection.select_boys_ed_frame
    def counted(*args):
        calls.append(args[2])
        return select(*args)
    monkeypatch.setattr(_ed_selection, 'select_boys_ed_frame', counted)
    inputs, _, pullback, frame = driver._prepare(state, plan, group, controls, with_grad, return_frame=True)
    assert calls == [group.representative_index]
    resolved = state.pop('_prepared_domain_static')
    assert all((f.strong_virtual is not None) == (i in group.fragment_indices)
               for i, f in enumerate(resolved.fragments))
    # The caller's global plan is immutable and never holds numerical frames.
    assert all(f.strong_virtual is None for f in plan.fragments)
    expected, _, expected_pb, expected_frame = driver._prepare(
        state, resolved, group, controls, with_grad, return_frame=True)
    for name in inputs:
        np.testing.assert_allclose(inputs[name], expected[name], atol=2e-10, rtol=2e-9)
    np.testing.assert_allclose(frame.virtual_coeff, expected_frame.virtual_coeff, atol=2e-10)
    if with_grad:
        rng = np.random.default_rng(208)
        bars = {key:rng.normal(size=value.shape) for key,value in inputs.items()}
        extra = (rng.normal(size=frame.occupied_coeff.shape), rng.normal(size=frame.virtual_coeff.shape))
        actual, want = pullback((bars, extra)), expected_pb((bars, extra))
        for got, ref in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(want)):
            if got.dtype != jax.dtypes.float0:
                np.testing.assert_allclose(got, ref, atol=2e-8, rtol=2e-8)


def test_pending_plan_roundtrips_without_frames(boys_water_dimer, tmp_path):
    from pyscfad.dlno_stc._restart import RestartManager
    mf, _, reference = boys_water_dimer
    plan = _plan(mf, reference)
    manager = RestartManager(tmp_path, method='ed-plan-test', scientific_payload={'test': 1})
    try:
        manager.save_static(plan)
        restored = manager.load_static(expected_type=type(plan))
    finally:
        manager.close()
    assert restored.ed_selection_pending
    np.testing.assert_array_equal(restored.ao_atom_slices, plan.ao_atom_slices)
    assert _selection.build_domain_groups(restored) == _selection.build_domain_groups(plan)


def test_fragment_frame_and_pullback_leave_scope(boys_water_dimer):
    mf, common, reference = boys_water_dimer
    plan = _plan(mf, reference)
    state = dict(mf=mf, common=common)
    previous = []
    for group in _selection.build_domain_groups(plan):
        assert all(ref() is None for ref in previous)
        inputs, host, pullback, frame = driver._prepare(
            state, plan, group, dict(mode='deterministic', threads=1), True, return_frame=True)
        previous = [weakref.ref(frame.occupied_coeff), weakref.ref(frame.virtual_coeff),
                    weakref.ref(pullback)]
        seed = ({key:jax.numpy.zeros_like(value) for key,value in inputs.items()},
                (jax.numpy.zeros_like(frame.occupied_coeff), jax.numpy.zeros_like(frame.virtual_coeff)))
        bars = pullback(seed)
        jax.block_until_ready(bars)
        del inputs, host, pullback, frame, seed, bars
        gc.collect()
        assert all(ref() is None for ref in previous)


def test_unmerged_physical_group_receives_shared_labels(boys_water_dimer):
    from pyscfad.dlno_stc import ccsd
    mf, common, reference = boys_water_dimer
    plan = _selection.build_domain_selections(
        mf, frozen=2, target_coeff=reference.target_reference_coeff, boys_grouping='atom')
    group = _selection.build_domain_groups(plan, merge_identical=False)[0]
    state = dict(mf=mf, common=common)
    driver._prepare(state, plan, group, dict(mode='deterministic', threads=1), False)
    resolved = state.pop('_prepared_domain_static')
    assert ccsd._group_cc_fragments(resolved, group) == ccsd._group_cc_fragments(plan, group)


def test_streamed_ed_ccsdt_matches_fixed_ed_energy_and_full_gradient(molecular_group, monkeypatch):
    from pyscfad.dlno_stc import ccsd, _ed_selection
    case = molecular_group
    fixed = case.static.mp2_static
    pending = replace(fixed, ed_selection_pending=True, fragments=tuple(
        replace(f, strong_virtual=None, virtual_anchor_columns=None) for f in fixed.fragments))
    builds = []
    original = _ed_selection.select_boys_ed_frame
    def select(*args):
        builds.append(args[2])
        return original(*args)
    monkeypatch.setattr(_ed_selection, 'select_boys_ed_frame', select)
    def run(domains):
        static = ccsd.build_static_selections(
            case.mf, domains, matrix_kind='W', thresh_occ=1e-4, thresh_vir=3e-5,
            baseline_controls=case.controls, matrix_controls=case.controls,
            fragment_indices=case.indices)
        state = driver._initialize(case.mf, domains, case.controls, with_grad=True,
                                   scf_pullback=case.scf_pullback)
        rows = ccsd._run_group(
            state, static, case.group, case.controls, case.controls, with_grad=True,
            verbose_imp=0, ccsd_t=True, fragment_indices=case.indices,
            triples_controls=ccsd.normalize_triples_controls(
                dict(mode='deterministic', integral_transform='rotate'), ccsd_t=True))
        value, grad = driver._finish(state, with_grad=True, include_hf=False)
        return value, grad.coords, rows
    expected = run(fixed)
    assert builds == []
    actual = run(pending)
    assert builds == [case.group.representative_index]
    np.testing.assert_allclose(actual[0], expected[0], atol=3e-11, rtol=0)
    np.testing.assert_allclose(actual[1], expected[1], atol=3e-9, rtol=3e-7)
    for got, want in zip(actual[2], expected[2]):
        assert (got.nocc_lis, got.nvir_lis) == (want.nocc_lis, want.nvir_lis)
        np.testing.assert_allclose([got.e_mp2_lis, got.e_ccsd, got.e_triples],
                                  [want.e_mp2_lis, want.e_ccsd, want.e_triples], atol=3e-11)
