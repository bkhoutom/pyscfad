"""Individual ED PAO selection, fixed replay, and its retained reverse tape."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import _ed_selection, _selection, domain


def local_case(x=0.):
    """A finite support with an independent but sub-cutoff Gram mode."""
    eye = np.eye(7)
    parents = np.column_stack((
        eye[:, 1], eye[:, 1] + 7e-4 * eye[:, 2], eye[:, 3],
        np.sqrt(.99495) * eye[:, 4] + np.sqrt(.00505) * eye[:, 5],
        np.sqrt(.9948) * eye[:, 2] + np.sqrt(.0052) * eye[:, 6]))
    rng = np.random.default_rng(402)
    direction = rng.normal(size=(7, 7)) * .03
    overlap = jnp.eye(7) + x * (direction + direction.T)
    pao = jnp.asarray(parents) + x * rng.normal(size=parents.shape) * .02
    pao = pao / jnp.sqrt(jnp.sum(pao * (overlap @ pao), axis=0))
    occupied = jnp.asarray((eye[:, 0] + .1 * eye[:, 1])[:, None])
    occupied = occupied + x * rng.normal(size=occupied.shape) * .02
    occupied = occupied / jnp.sqrt(jnp.sum(occupied * (overlap @ occupied)))
    common = domain.DomainData(
        overlap, jnp.diag(jnp.arange(7.)), occupied, jnp.asarray(eye[:, 1:]),
        jnp.array([-1.]), jnp.arange(6.), occupied, pao,
        (domain.TargetOccupiedData(occupied),))
    one, ao, empty = np.array([0]), np.arange(5), np.array([], dtype=int)
    fragment = _selection.FragmentDomainSelection(
        0, one, one, one, one, ao, one, one, None,
        one, ao, one, empty, empty, None, 0)
    static = _selection.DomainSelections(
        0, _selection.BoysDomainThresholds(), one, np.arange(1, 7), one,
        ao, np.array([0, 1, 2, 3, 4, -1, -1]), (one,), (one,),
        np.ones((1, 1), dtype=bool), (fragment,),
        ed_selection_pending=True, ao_atom_slices=np.array([[0, 5], [5, 7]]))
    return common, static


def test_individual_completeness_and_one_local_rank_selection(monkeypatch):
    common, static = local_case()
    sizes = []
    original = domain.scipy.linalg.eigh
    def counted(matrix, *args, **kwargs):
        sizes.append(matrix.shape[0])
        return original(matrix, *args, **kwargs)
    monkeypatch.setattr(domain.scipy.linalg, 'eigh', counted)
    frame, selected = _ed_selection.select_boys_ed_frame(common, static, 0)
    assert frame.virtual_coeff.shape == (5, 3)
    # The first two parents are independent, yet their small local mode drops.
    assert np.linalg.matrix_rank(np.asarray(common.pao_coeff[:, :4])) == 4
    np.testing.assert_array_equal(selected.strong_virtual.completeness_keep, [0, 1, 2, 3])
    np.testing.assert_array_equal(selected.strong_virtual.metric_keep, [1, 2, 3])
    assert sizes == [4]  # scalar occupied normalization, one virtual Gram
    co, cv, s = map(np.asarray, (frame.occupied_coeff, frame.virtual_coeff, common.s1e[:5, :5]))
    # ED MP2 retains the individual PAO directions, including their small
    # occupied component; cross-orthogonalization is deferred to the LIS.
    assert np.linalg.norm(co.T @ s @ cv) > .09
    np.testing.assert_allclose(cv.T @ s @ cv, np.eye(3), atol=2e-13)
    # Both individually complete directions survive, including .99495 < .995.
    np.testing.assert_allclose((cv @ cv.T)[3:, 3:], np.eye(2), atol=2e-13)


def test_lis_removes_ed_occupied_overlap_with_live_frame_response():
    from pyscfad.dlno_stc import lis
    from pyscfad.dlno_stc.smooth import orthonormalize
    common, domains = local_case()
    frame, selected = _ed_selection.select_boys_ed_frame(common, domains, 0)
    domains = replace(domains, fragments=(selected,), ed_selection_pending=False)
    # Force nonzero cross overlap independently of the ED construction under
    # test so removing the LIS projection alone breaks this regression.
    contaminated = frame.virtual_coeff + .08 * frame.occupied_coeff @ jnp.ones((1, 3))
    frame = frame._replace(virtual_coeff=orthonormalize(contaminated, common.s1e[:5, :5]))
    matrices = dict(occupied=jnp.eye(1), virtual=jnp.eye(3))
    chosen = lis.build_fragment_lis_selection(None, domains, 0, common=common,
        frame=frame, matrices=matrices, matrix_kind='D', thresh_occ=0., thresh_vir=0.)
    static = lis.LISSelections(domains, 'D', 0., 0., 1e-6, (chosen,))
    seed = jnp.asarray(np.random.default_rng(19).normal(size=(7, 7)))
    direction = jnp.asarray(np.random.default_rng(20).normal(size=(5, 3))) * .03
    def build(x):
        v = orthonormalize(frame.virtual_coeff + x * direction, common.s1e[:5, :5])
        return lis.build_fragment_lis(None, common, static, 0, matrices=matrices,
                                      frame=frame._replace(virtual_coeff=v))
    value = build(0.)
    np.testing.assert_allclose(value.mo_coeff.T @ common.s1e @ value.mo_coeff,
                               np.eye(4), atol=3e-13)
    def objective(x):
        v = build(x).active_virtual_coeff
        return jnp.sum((v @ v.T) * seed)
    derivative = jax.grad(objective)(0.)
    finite = (objective(1e-5) - objective(-1e-5)) / 2e-5
    assert abs(float(derivative)) > 1e-4
    np.testing.assert_allclose(derivative, finite, atol=2e-8, rtol=2e-7)


def test_lis_rejects_annihilated_virtual_direction():
    from pyscfad.dlno_stc import lis
    occupied = jnp.eye(4)[:, :1]
    with pytest.raises(RuntimeError, match='LIS virtual.*rank'):
        lis._orthogonalize_lis_virtual(occupied, occupied, jnp.eye(4))


@pytest.mark.parametrize('pending', [False, True])
@pytest.mark.parametrize('with_grad', [False, True])
def test_ed_occupied_and_raw_paos_share_one_overlap_solve(monkeypatch, pending, with_grad):
    common, static = local_case()
    if not pending:
        _, selected = _ed_selection.select_boys_ed_frame(common, static, 0)
        static = replace(static, fragments=(selected,), ed_selection_pending=False)
    solves = []
    def counted(original):
        def solve(matrix, rhs, *args, **kwargs):
            solves.append((matrix.shape, rhs.shape))
            return original(matrix, rhs, *args, **kwargs)
        return solve
    monkeypatch.setattr(np.linalg, 'solve', counted(np.linalg.solve))
    monkeypatch.setattr(jnp.linalg, 'solve', counted(jnp.linalg.solve))
    def coefficients(current):
        frame = domain.build_boys_ed_frame(current, static, 0)
        return frame.occupied_coeff, frame.virtual_coeff
    if with_grad:
        values, pullback = jax.vjp(coefficients, common)
        bars, = pullback(jax.tree_util.tree_map(jnp.ones_like, values))
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(bars))
    else:
        values = coefficients(common)
    assert values[1].shape == (5, 3)
    # One AO factorization serves one occupied target and five raw PAOs.
    assert solves == [((5, 5), (5, 6))]


def test_materialized_replay_projector_and_vjp_match_finite_difference():
    common, static = local_case()
    selected_frame, selected = _ed_selection.select_boys_ed_frame(common, static, 0)
    assert selected_frame.virtual_coeff.shape[1] == 3
    resolved = replace(static, fragments=(selected,), ed_selection_pending=False)
    seed = jnp.asarray(np.random.default_rng(58).normal(size=(5, 5)))
    def projector(x, pending):
        current, _ = local_case(x)
        frame = domain.build_boys_ed_frame(current, static if pending else resolved, 0)
        return frame.virtual_coeff @ frame.virtual_coeff.T
    def objective(x, pending):
        return jnp.sum(projector(x, pending) * seed)
    np.testing.assert_allclose(projector(0., True), projector(0., False), atol=2e-13)
    actual, replay = (jax.grad(objective)(0., pending) for pending in (True, False))
    finite = (objective(1e-5, False) - objective(-1e-5, False)) / 2e-5
    assert abs(float(actual)) > 1e-4
    np.testing.assert_allclose(actual, replay, atol=2e-10, rtol=2e-9)
    np.testing.assert_allclose(actual, finite, atol=2e-8, rtol=2e-7)


def test_saved_anchor_frame_response_matches_finite_difference():
    common, static = local_case()
    selected_frame, selected = _ed_selection.select_boys_ed_frame(common, static, 0)
    resolved = replace(static, fragments=(selected,), ed_selection_pending=False)
    seed = jnp.asarray(np.random.default_rng(81).normal(size=(5, 3)))
    def objective(x, pending):
        current, _ = local_case(x)
        frame = domain.build_boys_ed_frame(current, static if pending else resolved, 0)
        return jnp.sum(frame.virtual_coeff * seed)
    replay = domain.build_boys_ed_frame(common, resolved, 0)
    np.testing.assert_allclose(selected_frame.virtual_coeff, replay.virtual_coeff, atol=2e-13)
    actual = jax.grad(objective)(0., True)
    np.testing.assert_allclose(actual, jax.grad(objective)(0., False), atol=2e-10)
    finite = (objective(1e-5, False) - objective(-1e-5, False)) / 2e-5
    np.testing.assert_allclose(actual, finite, atol=2e-8, rtol=2e-7)


def test_ed_local_rank_cutoff_is_independent_of_occupied_metric_rank():
    common, static = local_case()
    relaxed = replace(static, thresholds=replace(static.thresholds, ed_pao_rank=1e-8))
    frame, _ = _ed_selection.select_boys_ed_frame(common, relaxed, 0)
    assert frame.virtual_coeff.shape == (5, 4)
    np.testing.assert_allclose(frame.virtual_coeff.T @ common.s1e[:5, :5]
                               @ frame.virtual_coeff, np.eye(4), atol=2e-12)


def test_zero_completeness_cutoff_excludes_zero_norm_columns_and_keeps_finite_vjp():
    common, static = local_case()
    static = replace(static, thresholds=replace(static.thresholds, ed_pao=0.))
    common = common._replace(pao_coeff=common.pao_coeff.at[:, 4].set(jnp.eye(7)[:, 6]))
    def coefficients(current):
        frame = domain.build_boys_ed_frame(current, static, 0)
        return frame.virtual_coeff
    frame, selected = _ed_selection.select_boys_ed_frame(common, static, 0)
    np.testing.assert_array_equal(selected.strong_virtual.completeness_keep, [0, 1, 2, 3])
    values, pullback = jax.vjp(coefficients, common)
    bars, = pullback(jnp.ones_like(values))
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree_util.tree_leaves(bars))


@pytest.mark.parametrize('which', ['completeness', 'metric'])
def test_replay_rejects_changed_discrete_local_space(which):
    common, static = local_case()
    _, selected = _ed_selection.select_boys_ed_frame(common, static, 0)
    thresholds = replace(static.thresholds, **({'ed_pao': .999} if which == 'completeness'
                                             else {'ed_pao_rank': 1e-8}))
    resolved = replace(static, thresholds=thresholds, fragments=(selected,), ed_selection_pending=False)
    with pytest.raises(RuntimeError, match='changed.*rebuild domains'):
        domain.build_boys_ed_frame(common, resolved, 0)


@pytest.mark.parametrize('profile', [False, True])
def test_legacy_fixed_pao_replay_retains_values_and_response(monkeypatch, profile):
    common, static = local_case()
    common = common._replace(pao_coeff=jnp.eye(7)[:, 1:4])
    columns, ao = np.arange(3), np.arange(5)
    legacy = _selection.FixedPAOSubspaceSelection(
        columns, ao, columns, columns, columns, columns)
    fragment = replace(static.fragments[0], strong_occ_metric_keep=np.arange(1),
                       strong_virtual=legacy, virtual_anchor_columns=(0, 1, 2))
    static = replace(static, fragments=(fragment,), ed_selection_pending=False)
    monkeypatch.setenv('PYSCFAD_DLNO_RESOURCE_PROFILE', str(int(profile)))
    def objective(x):
        current = common._replace(s1e=common.s1e + x * jnp.diag(jnp.arange(7.) * .02))
        frame = domain.build_boys_ed_frame(current, static, 0)
        return jnp.sum(frame.virtual_coeff @ frame.virtual_coeff.T)
    value, actual = jax.value_and_grad(objective)(0.)
    finite = (objective(1e-5) - objective(-1e-5)) / 2e-5
    np.testing.assert_allclose(value, 2.801980198019802, atol=2e-13)
    np.testing.assert_allclose(actual, finite, atol=2e-9, rtol=2e-7)


@pytest.mark.parametrize('profile', [False, True])
@pytest.mark.parametrize('pending', [False, True])
def test_reverse_uses_forward_tape_without_domain_reconstruction(monkeypatch, profile, pending):
    common, static = local_case()
    if not pending:
        frame, selected = _ed_selection.select_boys_ed_frame(common, static, 0)
        static = replace(static, fragments=(selected,), ed_selection_pending=False)
    monkeypatch.setenv('PYSCFAD_DLNO_RESOURCE_PROFILE', str(int(profile)))
    def coefficients(current):
        frame = domain.build_boys_ed_frame(current, static, 0)
        return frame.occupied_coeff, frame.virtual_coeff
    values, pullback = jax.vjp(coefficients, common)
    assert values[1].shape == (5, 3)
    def forbidden(*args, **kwargs):
        raise AssertionError('backward repeated ED construction or selection')
    monkeypatch.setattr(_ed_selection, 'select_boys_ed_frame', forbidden)
    monkeypatch.setattr(domain, '_local_ed_pao_space', forbidden)
    monkeypatch.setattr(domain, '_project_ed_columns', forbidden)
    monkeypatch.setattr(domain, '_project_to_ao_support', forbidden)
    monkeypatch.setattr(domain, '_stable_virtual_frame', forbidden)
    monkeypatch.setattr(domain.scipy.linalg, 'eigh', forbidden)
    bars, = pullback(jax.tree_util.tree_map(jnp.ones_like, values))
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree_util.tree_leaves(bars))
    assert np.linalg.norm(np.asarray(bars.pao_coeff)) > 1e-4
