"""Physical weak response remains compatible at the public Boys defaults."""
from dataclasses import fields, is_dataclass
import json

import jax
import numpy as np

from pyscfad.dlno_stc import build_domain_selections, domain, driver
from pyscfad.dlno_stc import _selection
from pyscfad.dlno_stc._pairs import weak_pair_energy_and_grad, weak_pair_indices
from pyscfad.dlno_stc.adjoint import add_cotangent
from .test_boys_prepare import boys_water_dimer


def _compare_discrete(actual, expected, issues, path='selection'):
    """Include scalar target slots and tuple anchors as well as array labels."""
    if type(actual) is not type(expected):
        issues.append(dict(field=path, actual=repr(actual), expected=repr(expected)))
        return 1
    if is_dataclass(expected):
        return sum(_compare_discrete(getattr(actual, item.name), getattr(expected, item.name),
                                     issues, path + '.' + item.name) for item in fields(expected))
    if isinstance(expected, np.ndarray):
        if expected.dtype.kind in 'biu':
            if not np.array_equal(actual, expected):
                issues.append(dict(field=path, actual=actual.tolist(), expected=expected.tolist()))
            return 1
        return 0
    if isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            issues.append(dict(field=path, actual=repr(actual), expected=repr(expected)))
            return 1
        return sum(_compare_discrete(a, b, issues, f'{path}[{index}]')
                   for index, (a, b) in enumerate(zip(actual, expected)))
    if actual != expected:
        issues.append(dict(field=path, actual=repr(actual), expected=repr(expected)))
    return 1


def _close(common_pullback, weak_result):
    energy, (direct, common_bar) = weak_result
    common_mf_bar, = common_pullback(common_bar)
    closed = jax.tree_util.tree_map(add_cotangent, direct, common_mf_bar)
    jax.block_until_ready((energy, closed))
    return energy, closed


def test_default_selection_preserves_weak_energy_and_closed_mf_response(
        boys_water_dimer, record_property, monkeypatch):
    mf, _, _ = boys_water_dimer
    pivot_gaps = {'legacy': [], 'shared': []}
    phase = 'legacy'
    select_anchors = _selection._select_virtual_anchor_columns

    def anchors(coeff, overlap21, pao, parents):
        result = select_anchors(coeff, overlap21, pao, parents)
        cross = coeff.T.conj() @ overlap21 @ pao[:, parents]
        norms = np.sort(np.sum(np.abs(cross) ** 2, axis=0))
        pivot_gaps[phase].append(dict(top=float(norms[-1]),
                                     first_gap=float(norms[-1] - norms[-2])))
        return result

    monkeypatch.setattr(_selection, '_select_virtual_anchor_columns', anchors)
    # The former route: ordinary host selection, then an ordinary aligned
    # Boys replay and a fresh weak response. No localization options override
    # the public defaults in either route.
    legacy_static = build_domain_selections(mf, frozen=2)
    data, data_pullback = jax.vjp(domain.build_scf_data, mf)
    legacy_common, legacy_pullback = driver._build_common_with_pullback(
        mf, legacy_static, data, data_pullback)
    legacy_pairs = weak_pair_indices(legacy_static)
    assert legacy_pairs
    legacy_weak = weak_pair_energy_and_grad(mf, legacy_common, legacy_static)
    legacy_energy, legacy_closed = _close(legacy_pullback, legacy_weak)

    phase = 'shared'
    shared_static, _, _, receipt = driver._select_domain_reference(
        mf, lambda reference: build_domain_selections(reference, frozen=2))
    shared_common, shared_pullback, pending = receipt.take(shared_static)
    pairs, shared_weak = pending
    assert pairs == legacy_pairs
    label_issues = []
    checked_labels = _compare_discrete(shared_static, legacy_static, label_issues)
    anchor_issues = [issue for issue in label_issues
                     if '.virtual_anchor_columns[' in issue['field']]
    label_issues = [issue for issue in label_issues if issue not in anchor_issues]
    shared_energy, shared_closed = _close(shared_pullback, shared_weak)

    # This symmetric dimer has QR pivot squared norms tied within a few ulps
    # (observed first-pivot gaps 0--4.5e-16). Reordered PAO arithmetic and the
    # live polar alignment can choose another equally valid anchor. Preserve
    # all support/rank/target-slot labels exactly, then check the physical ED
    # subspaces instead of requiring identical indices for that QR gauge.
    if anchor_issues:
        for point in pivot_gaps['legacy'] + pivot_gaps['shared']:
            assert point['first_gap'] <= 64 * np.finfo(float).eps * point['top']
    strong_spaces = []
    seen = set()
    for index, fragment in enumerate(legacy_static.fragments):
        key = tuple(tuple(getattr(fragment, name)) for name in
                    ('strong_fragments', 'pao_center_atoms', 'extended_atoms'))
        if key in seen:
            continue
        seen.add(key)
        ordinary = domain.build_boys_ed_frame(legacy_common, legacy_static, index)
        shared = domain.build_boys_ed_frame(shared_common, shared_static, index)
        ao = fragment.extended_ao_indices
        overlap = np.asarray(legacy_common.s1e)[np.ix_(ao, ao)]
        point = dict(fragment=index)
        for space in ('occupied', 'virtual'):
            left = np.asarray(getattr(ordinary, space + '_coeff'))
            right = np.asarray(getattr(shared, space + '_coeff'))
            left_metric, right_metric = left.T @ overlap @ left, right.T @ overlap @ right
            np.testing.assert_allclose(left_metric, np.eye(left.shape[1]), atol=2e-10, rtol=0)
            np.testing.assert_allclose(right_metric, np.eye(right.shape[1]), atol=2e-10, rtol=0)
            left_projector = left @ np.linalg.solve(left_metric, left.T @ overlap)
            right_projector = right @ np.linalg.solve(right_metric, right.T @ overlap)
            singular = np.linalg.svd(left.T @ overlap @ right, compute_uv=False)
            np.testing.assert_allclose(right_projector, left_projector, atol=2e-10, rtol=0)
            np.testing.assert_allclose(singular, 1., atol=2e-10, rtol=0)
            point[space + '_projector_max_abs'] = float(np.max(np.abs(right_projector - left_projector)))
            point[space + '_principal_overlaps'] = singular.tolist()
        strong_spaces.append(point)

    errors = {}
    actual_paths, actual_tree = jax.tree_util.tree_flatten_with_path(shared_closed)
    expected_paths, expected_tree = jax.tree_util.tree_flatten_with_path(legacy_closed)
    assert actual_tree == expected_tree
    for (path, actual), (expected_path, expected) in zip(actual_paths, expected_paths):
        assert path == expected_path
        if actual.dtype != jax.dtypes.float0:
            label = jax.tree_util.keystr(path)
            errors[label] = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)), initial=0.))

    metrics = dict(
        weak_pairs=len(pairs), checked_discrete_fields=checked_labels,
        discrete_mismatches=label_issues,
        tied_anchor_mismatches=anchor_issues, anchor_pivot_gaps=pivot_gaps,
        strong_spaces=strong_spaces,
        initial_to_replay_orbital_max_abs=float(np.max(np.abs(
            np.asarray(legacy_common.iao_coeff) - legacy_static.target_reference_coeff))),
        shared_to_initial_orbital_max_abs=float(np.max(np.abs(
            np.asarray(shared_common.iao_coeff) - legacy_static.target_reference_coeff))),
        weak_energy=float(shared_energy), energy_abs_error=float(abs(shared_energy - legacy_energy)),
        closed_mf_max_abs_error=max(errors.values()), closed_mf_leaf_max_abs_errors=errors,
    )
    print('DEFAULT_REFERENCE_PARITY ' + json.dumps(metrics, sort_keys=True), flush=True)
    record_property('default_reference_parity', json.dumps(metrics, sort_keys=True))
    assert not label_issues, label_issues
    # Same energy and full-leaf tolerances as the weak benchmark comparison.
    np.testing.assert_allclose(shared_energy, legacy_energy, atol=2e-12, rtol=0)
    for (_, actual), (_, expected) in zip(actual_paths, expected_paths):
        if actual.dtype != jax.dtypes.float0:
            np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-8)
