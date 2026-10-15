"""Endpoint-factored native weak response and progressive restart checks."""

from collections import Counter
from types import SimpleNamespace
from typing import NamedTuple
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno import mp2, multipole
from pyscfad.dlno._restart import RestartManager


class _Common(NamedTuple):
    s1e: object
    parameter: object


def _static(nfragment=4, order=4, *, all_strong=False):
    mask = (
        np.ones((nfragment, nfragment), dtype=bool)
        if all_strong else np.eye(nfragment, dtype=bool)
    )
    return SimpleNamespace(
        fragments=tuple(SimpleNamespace(
            fragment_index=i,
            primary_atoms=(i,),
            strong_fragments=np.flatnonzero(mask[i]),
            extended_atoms=(i,),
            extended_ao_indices=(i,),
            strong_occ_metric_keep=(0,),
            strong_virtual=SimpleNamespace(metric_keep=(0,)),
        ) for i in range(nfragment)),
        strong_mask=mask,
        thresholds=SimpleNamespace(multipole_order=order),
    )


def _endpoint(mf, common, static, index):
    """Small unequal-mode endpoint with all differentiable physical inputs."""
    nmode = 2 + index % 2
    records = []
    for mode in range(nmode):
        x = mf + common.parameter * (index + 1) + 0.03 * mode
        center = jnp.asarray([3.0 * index, 0.2 * mode, 0.1]) + x * jnp.asarray([0.03, 0.02, 0.01])
        mu = jnp.arange(6.0).reshape(3, 2) * 0.013 + x * 0.02 + 0.01 * mode
        gap = jnp.asarray([0.8, 1.1]) + x * 0.01
        theta = omega = None
        if static.thresholds.multipole_order >= 3:
            theta = jnp.arange(18.0).reshape(3, 3, 2) * 0.002 + x * 0.007
        if static.thresholds.multipole_order >= 4:
            omega = jnp.arange(54.0).reshape(3, 3, 3, 2) * 0.0003 + x * 0.002
        records.append((center, mu, gap, theta, omega))
    weights = jnp.linspace(0.3, 0.9, nmode) + 0.01 * mf + 0.02 * common.parameter
    return weights, tuple(records)


def _pair_oracle(left, right, order):
    return jnp.real(sum(
        left[0][i] * right[0][j] * multipole._multipole_pair_energy(a, b, order)
        for i, a in enumerate(left[1]) for j, b in enumerate(right[1])
    ))


def _strong(mf, common, static, spec):
    assert spec[0] == "strong", "production driver rebuilt a weak pair"
    return (spec[1] + 1) * (0.17 * mf ** 2 + 0.09 * common.parameter ** 2)


def _oracle(mf, common, static):
    result = jnp.asarray(0.0)
    for spec in mp2._correlation_term_specs(static):
        if spec[0] == "strong":
            result += _strong(mf, common, static, spec)
        else:
            result += _pair_oracle(
                _endpoint(mf, common, static, spec[1]),
                _endpoint(mf, common, static, spec[2]),
                static.thresholds.multipole_order,
            )
    return result


def _assert_trees_close(actual, expected):
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
    assert actual_tree == expected_tree
    for a, e in zip(actual_leaves, expected_leaves):
        np.testing.assert_allclose(a, e, rtol=2e-12, atol=2e-13)


def _install_screen_model(monkeypatch, *, calls=None, moments=None):
    """Replace orbital algebra/AO integrals, retaining the real split VJPs."""
    calls = Counter() if calls is None else calls
    moments = Counter() if moments is None else moments

    def screen(common, static, index):
        calls[index] += 1
        if calls[index] > 1:
            raise AssertionError("orbital frame rebuilt during weak response")
        return (common.parameter * (index + 1),)

    def from_screen(mf, saved, static, index):
        moments[index] += 1
        common = _Common(jnp.eye(2), saved[0] / (index + 1))
        return _endpoint(mf, common, static, index)

    monkeypatch.setattr(mp2, "build_weak_multipole_screen", screen)
    monkeypatch.setattr(mp2, "_weak_endpoint_from_screen", from_screen)
    monkeypatch.setattr(mp2, "_correlation_term_energy", _strong)
    return calls, moments


@pytest.mark.parametrize("order", [2, 3, 4])
def test_weighted_multimode_pair_energy_and_cotangents(order):
    assert hasattr(mp2, "weak_endpoint_pair_energy"), (
        "compact weak endpoint contraction is missing"
    )
    static = _static(order=order)
    common = _Common(jnp.eye(2), jnp.asarray(0.23))
    left = _endpoint(jnp.asarray(0.4), common, static, 0)
    right = _endpoint(jnp.asarray(0.4), common, static, 1)
    expected = jax.value_and_grad(_pair_oracle, argnums=(0, 1))(left, right, order)
    actual = jax.value_and_grad(mp2.weak_endpoint_pair_energy, argnums=(0, 1))(left, right, order)
    _assert_trees_close(actual, expected)


def test_forward_energy_reuses_distinct_endpoint_values(monkeypatch):
    static = _static()
    mf, common = jnp.asarray(0.4), _Common(jnp.eye(2), jnp.asarray(0.23))
    calls = Counter()
    def counted(m, c, s, i):
        calls[i] += 1
        return _endpoint(m, c, s, i)
    monkeypatch.setattr(mp2, "rebuild_domain_data", lambda *args, **kwargs: common)
    monkeypatch.setattr(mp2, "build_weak_endpoint", counted)
    monkeypatch.setattr(mp2, "_correlation_term_energy", _strong)
    actual = mp2.correlation_energy(mf, static)
    np.testing.assert_allclose(actual, _oracle(mf, common, static), atol=2e-13, rtol=2e-12)
    assert calls == Counter({i: 1 for i in range(4)})


@pytest.mark.parametrize("order", [2, 3, 4])
def test_real_multimode_endpoint_energy_and_response_matches_pair_api(monkeypatch, order):
    from pyscfad import config_update, gto
    from pyscfad.dlno.dlno_base import WeakScreen
    class MF(NamedTuple):
        mol: object
    mol = gto.Mole(atom="H 0 0 0; H 0 0 2", basis="sto-3g", unit="Bohr", verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = MF(mol)
    left = WeakScreen(
        jnp.asarray([0.4, 0.9]), jnp.asarray([-0.7, -0.5]),
        jnp.asarray([[0.8, 0.6], [0.1, 0.3]]),
        jnp.asarray([0.3]), jnp.asarray([[0.2], [0.7]]),
    )
    right = WeakScreen(
        jnp.asarray([0.6, 0.8]), jnp.asarray([-0.6, -0.4]),
        jnp.asarray([[0.2, 0.1], [0.7, 0.9]]),
        jnp.asarray([0.4]), jnp.asarray([[0.8], [0.1]]),
    )
    static = _static(2, order)
    static.fragments = tuple(SimpleNamespace(primary_atoms=(0, 1)) for _ in range(2))
    monkeypatch.setattr(mp2, "build_weak_multipole_screen", lambda screens, static_, i: screens[i])
    def factored(m, screens):
        return mp2.weak_endpoint_pair_energy(
            mp2.build_weak_endpoint(m, screens, static, 0),
            mp2.build_weak_endpoint(m, screens, static, 1), order,
        )
    def original(m, screens):
        return mp2.weak_screen_pair_energy(m, screens[0], screens[1], static, 0, 1)
    with config_update("pyscfad_moleintor_opt", True):
        actual, actual_pb = jax.vjp(factored, mf, (left, right))
        expected, expected_pb = jax.vjp(original, mf, (left, right))
        actual_bars, expected_bars = actual_pb(jnp.asarray(1.0)), expected_pb(jnp.asarray(1.0))
        left_value, left_state = mp2._prepare_weak_endpoint(mf, (left, right), static, 0)
        right_value, right_state = mp2._prepare_weak_endpoint(mf, (left, right), static, 1)
        retained_value, (left_bar, right_bar) = jax.value_and_grad(
            mp2.weak_endpoint_pair_energy, argnums=(0, 1)
        )(left_value, right_value, order)
        _, left_pullback = mp2._replay_weak_endpoint(mf, left_state, static, 0)
        _, right_pullback = mp2._replay_weak_endpoint(mf, right_state, static, 1)
        retained_bars = jax.tree_util.tree_map(
            lambda a, b: a + b, left_pullback(left_bar), right_pullback(right_bar),
        )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)
    _assert_trees_close(actual_bars, expected_bars)
    np.testing.assert_allclose(retained_value, expected, rtol=2e-12, atol=2e-13)
    _assert_trees_close(retained_bars, expected_bars)


def test_endpoint_moment_check_tolerates_c20_near_zero_moment_roundoff():
    static = _static()
    common = _Common(jnp.eye(2), jnp.asarray(0.23))
    weights, records = _endpoint(jnp.asarray(0.4), common, static, 0)
    center, mu, gap, theta, omega = records[0]
    reference_omega = omega.at[0, 0, 0, 0].set(0.0)
    rebuilt_omega = reference_omega.at[0, 0, 0, 0].set(5e-10)
    reference = (
        weights, ((center, mu, gap, theta, reference_omega), *records[1:]),
    )
    rebuilt = (
        weights, ((center, mu, gap, theta, rebuilt_omega), *records[1:]),
    )
    # C20 reproduces its orbital frame but has roundoff at this scale in
    # otherwise near-zero octupole components during differentiated replay.
    mp2._check_weak_endpoint_moments(rebuilt, reference, 0, "response replay")


@pytest.mark.parametrize("family", [1, 3, 4])
def test_endpoint_moment_check_scales_near_zero_noise_by_its_moment_column(family):
    static = _static()
    common = _Common(jnp.eye(2), jnp.asarray(0.23))
    weights, records = _endpoint(jnp.asarray(0.4), common, static, 0)
    reference_record = list(records[0])
    shape = reference_record[family].shape
    moments = jnp.zeros(shape).reshape(-1, shape[-1]).at[1, 0].set(10.0)
    reference_record[family] = moments.reshape(shape)
    rebuilt_record = list(reference_record)
    rebuilt_record[family] = moments.at[0, 0].set(2.5e-9).reshape(shape)
    reference = weights, (tuple(reference_record), *records[1:])
    rebuilt = weights, (tuple(rebuilt_record), *records[1:])
    mp2._check_weak_endpoint_moments(rebuilt, reference, 0, "response replay")


@pytest.mark.parametrize("family", [1, 3, 4])
def test_endpoint_moment_check_does_not_use_another_virtual_column_scale(family):
    static = _static()
    common = _Common(jnp.eye(2), jnp.asarray(0.23))
    weights, records = _endpoint(jnp.asarray(0.4), common, static, 0)
    reference_record = list(records[0])
    shape = reference_record[family].shape
    moments = jnp.zeros(shape).reshape(-1, shape[-1]).at[1, 1].set(1e6)
    reference_record[family] = moments.reshape(shape)
    rebuilt_record = list(reference_record)
    rebuilt_record[family] = moments.at[0, 0].set(1e-6).reshape(shape)
    reference = weights, (tuple(reference_record), *records[1:])
    rebuilt = weights, (tuple(rebuilt_record), *records[1:])
    with pytest.raises(RuntimeError, match="moments.*fragment 0"):
        mp2._check_weak_endpoint_moments(rebuilt, reference, 0, "response replay")


def test_endpoint_moment_check_does_not_use_another_multipole_order_scale():
    static = _static()
    common = _Common(jnp.eye(2), jnp.asarray(0.23))
    weights, records = _endpoint(jnp.asarray(0.4), common, static, 0)
    center, mu, gap, theta, omega = records[0]
    reference_record = center, jnp.zeros_like(mu), gap, theta, omega * 1e8
    rebuilt_record = center, jnp.zeros_like(mu).at[0, 0].set(1e-6), gap, theta, omega * 1e8
    reference = weights, (reference_record, *records[1:])
    rebuilt = weights, (rebuilt_record, *records[1:])
    with pytest.raises(RuntimeError, match="moments.*fragment 0"):
        mp2._check_weak_endpoint_moments(rebuilt, reference, 0, "response replay")


@pytest.mark.parametrize("field", ["center", "gap", "weights"])
def test_endpoint_moment_check_keeps_non_moment_componentwise_tolerances(field):
    static = _static()
    common = _Common(jnp.eye(2), jnp.asarray(0.23))
    weights, records = _endpoint(jnp.asarray(0.4), common, static, 0)
    center, mu, gap, theta, omega = records[0]
    reference_record = center.at[0].set(0.0), mu, gap.at[0].set(0.0), theta, omega * 1e8
    reference_weights = weights.at[0].set(0.0)
    rebuilt_record = list(reference_record)
    rebuilt_weights = reference_weights
    if field == "weights":
        rebuilt_weights = reference_weights.at[0].set(2.5e-9)
    else:
        index = 0 if field == "center" else 2
        rebuilt_record[index] = rebuilt_record[index].at[0].set(2.5e-9)
    reference = reference_weights, (reference_record, *records[1:])
    rebuilt = rebuilt_weights, (tuple(rebuilt_record), *records[1:])
    with pytest.raises(RuntimeError, match="moments.*fragment 0"):
        mp2._check_weak_endpoint_moments(rebuilt, reference, 0, "response replay")


@pytest.mark.parametrize("value", [jnp.nan, jnp.inf])
def test_endpoint_moment_check_rejects_nonfinite_moments(value):
    static = _static()
    common = _Common(jnp.eye(2), jnp.asarray(0.23))
    weights, records = _endpoint(jnp.asarray(0.4), common, static, 0)
    center, mu, gap, theta, omega = records[0]
    rebuilt = weights, ((center, mu, gap, theta, omega.at[0, 0, 0, 0].set(value)), *records[1:])
    reference = weights, records
    with pytest.raises(RuntimeError, match="moments.*fragment 0"):
        mp2._check_weak_endpoint_moments(rebuilt, reference, 0, "response replay")


@pytest.mark.parametrize("value", [jnp.nan, jnp.inf])
def test_endpoint_moment_check_rejects_nonfinite_mode_weights(value):
    static = _static()
    common = _Common(jnp.eye(2), jnp.asarray(0.23))
    weights, records = _endpoint(jnp.asarray(0.4), common, static, 0)
    invalid = weights.at[0].set(value), records
    # Even identical infinities are invalid, despite numpy.allclose accepting
    # them. An endpoint containing nonfinite weights cannot define pair bars.
    with pytest.raises(RuntimeError, match="moments.*fragment 0"):
        mp2._check_weak_endpoint_moments(invalid, invalid, 0, "response replay")


@pytest.mark.parametrize("family", [1, 3, 4])
def test_endpoint_moment_check_keeps_absolute_component_sized_roundoff_allowance(family):
    static = _static()
    common = _Common(jnp.eye(2), jnp.asarray(0.23))
    weights, records = _endpoint(jnp.asarray(0.4), common, static, 0)
    reference_record = list(records[0])
    reference_record[family] = jnp.zeros_like(reference_record[family])
    rebuilt_record = list(reference_record)
    rebuilt_record[family] = rebuilt_record[family].at[..., 0].set(0.75e-9)
    reference = weights, (tuple(reference_record), *records[1:])
    rebuilt = weights, (tuple(rebuilt_record), *records[1:])
    # Uniform noise in a nearly zero Cartesian column was accepted by the
    # former elementwise absolute tolerance and remains within its RMS floor.
    mp2._check_weak_endpoint_moments(rebuilt, reference, 0, "response replay")


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("driver", ["serial", "mpi"])
def test_retained_screen_response_never_rebuilds_the_orbital_frame(monkeypatch, order, driver):
    static = _static(order=order)
    mf, common = jnp.asarray(0.4), _Common(jnp.eye(2), jnp.asarray(0.23))
    calls, moments = _install_screen_model(monkeypatch)
    if driver == "serial":
        expected, pullback = jax.vjp(lambda m, c: _oracle(m, c, static), mf, common)
        actual = mp2._progressive_correlation_pullback(mf, common, static)
    else:
        mpi = pytest.importorskip("mpi4py.MPI")
        from pyscfad.dlno import mp2_mpi
        weak = tuple(spec for spec in mp2._correlation_term_specs(static) if spec[0] == "weak")
        expected, pullback = jax.vjp(lambda m, c: _weak_oracle(m, c, static), mf, common)
        actual = mp2_mpi._weak_endpoint_correlation_pullback(
            mf, common, static, weak, comm=mpi.COMM_SELF, root=0,
        )
    _assert_trees_close(actual[:3], (expected, *pullback(jnp.asarray(1.0))))
    assert calls == Counter({i: 1 for i in range(4)})
    assert moments == Counter({i: 2 for i in range(4)})


def _weak_oracle(mf, common, static):
    return sum((_pair_oracle(_endpoint(mf, common, static, left),
                            _endpoint(mf, common, static, right),
                            static.thresholds.multipole_order)
                for kind, left, right in mp2._correlation_term_specs(static)
                if kind == "weak"), jnp.asarray(0.))


@pytest.mark.parametrize("driver", ["serial", "mpi"])
@pytest.mark.parametrize("all_strong", [False, True])
def test_retained_screen_and_pullbacks_are_released_before_boundary_collection(
    monkeypatch, driver, all_strong,
):
    static = _static(all_strong=all_strong)
    mf, common = jnp.asarray(0.4), _Common(jnp.eye(2), jnp.asarray(0.23))
    calls, moments = _install_screen_model(monkeypatch)
    refs, collections = [], []
    prepare = mp2._prepare_weak_endpoint

    def tracked_prepare(*args):
        value, state = prepare(*args)
        refs.extend((weakref.ref(state[0][0]), weakref.ref(state[1])))
        return value, state

    def collect():
        collections.append(moments.copy())
        if moments and all(n == 2 for n in moments.values()):
            assert all(ref() is None for ref in refs)

    if driver == "serial":
        monkeypatch.setattr(mp2, "_prepare_weak_endpoint", tracked_prepare)
        monkeypatch.setattr(mp2.gc, "collect", collect)
        actual = mp2._progressive_correlation_pullback(mf, common, static)
        expected, pb = jax.vjp(lambda m, c: _oracle(m, c, static), mf, common)
        assert len(collections) == 4 + (0 if all_strong else 1)
    else:
        mpi = pytest.importorskip("mpi4py.MPI")
        from pyscfad.dlno import mp2_mpi
        monkeypatch.setattr(mp2_mpi, "_prepare_weak_endpoint", tracked_prepare)
        monkeypatch.setattr(mp2_mpi.gc, "collect", collect)
        weak = tuple(spec for spec in mp2._correlation_term_specs(static) if spec[0] == "weak")
        actual = mp2_mpi._weak_endpoint_correlation_pullback(
            mf, common, static, weak, comm=mpi.COMM_SELF, root=0,
        )
        expected, pb = jax.vjp(lambda m, c: _weak_oracle(m, c, static), mf, common)
        assert len(collections) == (0 if all_strong else 1)
    _assert_trees_close(actual[:3], (expected, *pb(jnp.asarray(1.0))))
    assert all(ref() is None for ref in refs)
    assert calls == (Counter() if all_strong else Counter({i: 1 for i in range(4)}))


def test_sparse_weak_graph_builds_only_incident_screens(monkeypatch):
    static = _static(5, all_strong=True)
    for left, right in ((0, 2), (1, 2)):
        static.strong_mask[left, right] = static.strong_mask[right, left] = False
    mf, common = jnp.asarray(0.4), _Common(jnp.eye(2), jnp.asarray(0.23))
    calls, moments = _install_screen_model(monkeypatch)
    actual = mp2._progressive_correlation_pullback(mf, common, static)
    expected, pb = jax.vjp(lambda m, c: _oracle(m, c, static), mf, common)
    _assert_trees_close(actual, (expected, *pb(jnp.asarray(1.0))))
    assert calls == Counter({0: 1, 1: 1, 2: 1})
    assert moments == Counter({0: 2, 1: 2, 2: 2})


@pytest.mark.parametrize("stop_count", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("return_details", [False, True])
def test_restart_rebuilds_interrupted_weak_stage_without_repeating_strong_rows(
    tmp_path, monkeypatch, stop_count, return_details,
):
    from pyscfad.dlno import _restart
    static = _static()
    mf, common = jnp.asarray(0.4), _Common(jnp.eye(2), jnp.asarray(0.23))
    _install_screen_model(monkeypatch)
    expected, pb = jax.vjp(lambda m, c: _oracle(m, c, static), mf, common)
    expected_bars = pb(jnp.asarray(1.0))
    path = tmp_path / "checkpoint"
    manager = RestartManager(path, method="weak-test", scientific_payload={"test": 1})
    class Stop(RuntimeError):
        pass
    saves = 0
    saved_specs = []
    saved_closed = False
    def stop(stage, key, filename):
        nonlocal saves, saved_closed
        if stage != "correlation_progress":
            return
        saves += 1
        if saves == stop_count:
            templates = dict(zip(("mf_bar", "common_bar"), mp2._zero_term_cotangents(mf, common)))
            record = manager.load_record(stage, templates=templates)
            saved_specs.extend(tuple(row) for row in record.metadata["completed_specs"])
            saved_closed = record.metadata["weak_response_closed"]
            assert record.metadata["weak_response_version"] == 2
            assert "weak_endpoint_bars" not in record.metadata
            assert "weak_endpoint_values" not in record.metadata
            if not saved_closed:
                assert all(spec[0] == "strong" for spec in saved_specs)
            raise Stop
    monkeypatch.setattr(_restart, "_CHECKPOINT_EVENT_HOOK", stop)
    with pytest.raises(Stop):
        mp2._progressive_correlation_pullback(
            mf, common, static, restart=manager, return_details=return_details,
        )
    manager.close()
    monkeypatch.setattr(_restart, "_CHECKPOINT_EVENT_HOOK", None)
    calls, moments = _install_screen_model(monkeypatch)
    completed = []
    def strong(m, c, s, spec):
        completed.append(spec)
        assert spec not in saved_specs
        return _strong(m, c, s, spec)
    monkeypatch.setattr(mp2, "_correlation_term_energy", strong)
    resumed = RestartManager(path, resume=True, method="weak-test", scientific_payload={"test": 1})
    actual = mp2._progressive_correlation_pullback(
        mf, common, static, restart=resumed, return_details=return_details,
    )
    resumed.close()
    _assert_trees_close(actual[:3], (expected, *expected_bars))
    assert calls == (Counter() if saved_closed else Counter({i: 1 for i in range(4)}))
    assert len(completed) == 4 - sum(spec[0] == "strong" for spec in saved_specs)
    if return_details:
        assert len(actual[3].terms) == 10
        assert actual[3].n_weak_pair_terms == 6
        assert [row.kind for row in actual[3].terms] == [spec[0] for spec in mp2._correlation_term_specs(static)]


@pytest.mark.parametrize("change", ["weights", "gap", "moments", "rotation", "shape"])
@pytest.mark.parametrize("driver", ["serial", "mpi"])
def test_moment_replay_rejects_changes_in_a_retained_orbital_frame(monkeypatch, change, driver):
    static = _static()
    mf, common = jnp.asarray(0.4), _Common(jnp.eye(2), jnp.asarray(0.23))
    _, calls = _install_screen_model(monkeypatch)
    original = mp2._weak_endpoint_from_screen
    def changed(m, screen, s, index):
        weights, records = original(m, screen, s, index)
        if index == 0 and calls[index] == 2:
            center, mu, gap, theta, omega = records[0]
            if change == "weights":
                weights = weights + 1e-4
            elif change == "gap":
                gap = gap + 1e-4
            elif change == "moments":
                mu = mu.at[0, 0].add(1e-4)
            elif change == "rotation":
                rotation = jnp.asarray([[0.8, -0.6], [0.6, 0.8]])
                mu, theta, omega = (jnp.einsum("...a,ab->...b", x, rotation)
                                    for x in (mu, theta, omega))
            else:
                mu, gap = mu[:, :1], gap[:1]
                theta, omega = theta[..., :1], omega[..., :1]
            records = ((center, mu, gap, theta, omega), *records[1:])
        return weights, records
    monkeypatch.setattr(mp2, "_weak_endpoint_from_screen", changed)
    with pytest.raises(RuntimeError, match="moments.*fragment 0"):
        if driver == "serial":
            mp2._progressive_correlation_pullback(mf, common, static)
        else:
            mpi = pytest.importorskip("mpi4py.MPI")
            from pyscfad.dlno import mp2_mpi
            weak = tuple(spec for spec in mp2._correlation_term_specs(static) if spec[0] == "weak")
            mp2_mpi._weak_endpoint_correlation_pullback(
                mf, common, static, weak, comm=mpi.COMM_SELF, root=0,
            )


@pytest.mark.parametrize("order", [2, 3, 4])
def test_mpi_timed_weak_phase_has_one_screen_build_per_endpoint(monkeypatch, order):
    mpi = pytest.importorskip("mpi4py.MPI")
    from pyscfad.dlno import mp2_mpi
    static = _static(order=order)
    mf, common = jnp.asarray(0.4), _Common(jnp.eye(2), jnp.asarray(0.23))
    calls, moments = _install_screen_model(monkeypatch)
    weak = tuple(spec for spec in mp2._correlation_term_specs(static) if spec[0] == "weak")
    actual = mp2_mpi._weak_endpoint_correlation_pullback(
        mf, common, static, weak, comm=mpi.COMM_SELF, root=0, collect_timing=True,
    )
    expected, pb = jax.vjp(lambda m, c: _weak_oracle(m, c, static), mf, common)
    _assert_trees_close(actual[:3], (expected, *pb(jnp.asarray(1.0))))
    assert calls == Counter({i: 1 for i in range(4)})
    assert moments == Counter({i: 2 for i in range(4)})
    assert len(actual[3]) == len(weak)
    assert all(row.frame_build_seconds >= 0 for row in actual[3])
    assert all(row.frame_replay_seconds >= 0 for row in actual[3])


@pytest.mark.parametrize("stage", ["pair", "replay"])
@pytest.mark.parametrize("stop_index", [1, 3])
def test_interrupted_weak_work_is_recomputed_after_resuming_strong_record(
    tmp_path, monkeypatch, stage, stop_index,
):
    static = _static()
    mf, common = jnp.asarray(0.4), _Common(jnp.eye(2), jnp.asarray(0.23))
    _install_screen_model(monkeypatch)
    expected, pb = jax.vjp(lambda m, c: _oracle(m, c, static), mf, common)
    expected_bars = pb(jnp.asarray(1.0))
    path = tmp_path / 'checkpoint'
    manager = RestartManager(path, method='weak-test', scientific_payload={'test': 1})
    name = '_weak_endpoint_pair_value_and_grad' if stage == 'pair' else '_replay_weak_endpoint'
    original = getattr(mp2, name)
    class Stop(RuntimeError):
        pass
    count = 0
    def interrupted(*args, **kwargs):
        nonlocal count
        count += 1
        if count == stop_index:
            raise Stop
        return original(*args, **kwargs)
    monkeypatch.setattr(mp2, name, interrupted)
    with pytest.raises(Stop):
        mp2._progressive_correlation_pullback(mf, common, static, restart=manager)
    record = manager.load_record('correlation_progress', templates=dict(zip(
        ('mf_bar', 'common_bar'), mp2._zero_term_cotangents(mf, common))))
    saved = {tuple(spec) for spec in record.metadata['completed_specs']}
    assert not record.metadata['weak_response_closed']
    assert all(spec[0] == 'strong' for spec in saved)
    manager.close()
    monkeypatch.setattr(mp2, name, original)
    calls, moments = _install_screen_model(monkeypatch)
    def checked_strong(m, c, s, spec):
        assert spec not in saved, 'resumed calculation repeated committed strong response'
        return _strong(m, c, s, spec)
    monkeypatch.setattr(mp2, '_correlation_term_energy', checked_strong)
    resumed = RestartManager(path, resume=True, method='weak-test', scientific_payload={'test': 1})
    actual = mp2._progressive_correlation_pullback(mf, common, static, restart=resumed)
    resumed.close()
    _assert_trees_close(actual, (expected, *expected_bars))
    assert calls == Counter({i: 1 for i in range(4)})
    assert moments == Counter({i: 2 for i in range(4)})


def test_checkpoint_events_only_follow_strong_rows_or_closed_weak_response(tmp_path, monkeypatch):
    from pyscfad.dlno import _restart
    static = _static()
    mf, common = jnp.asarray(0.4), _Common(jnp.eye(2), jnp.asarray(0.23))
    _install_screen_model(monkeypatch)
    manager = RestartManager(tmp_path / 'checkpoint', method='weak-test', scientific_payload={'test': 1})
    states = []
    def saved(stage, key, filename):
        if stage != 'correlation_progress':
            return
        record = manager.load_record(stage, templates=dict(zip(
            ('mf_bar', 'common_bar'), mp2._zero_term_cotangents(mf, common))))
        states.append((record.scalars['completed_count'], record.metadata['weak_response_closed']))
    monkeypatch.setattr(_restart, '_CHECKPOINT_EVENT_HOOK', saved)
    mp2._progressive_correlation_pullback(mf, common, static, restart=manager)
    manager.close()
    assert states == [(1, False), (2, False), (3, False), (9, True), (10, True)]


@pytest.mark.parametrize('closed', [False, True])
def test_legacy_checkpoints_are_accepted_only_with_closed_weak_response(tmp_path, monkeypatch, closed):
    static = _static()
    mf, common = jnp.asarray(0.4), _Common(jnp.eye(2), jnp.asarray(0.23))
    calls, _ = _install_screen_model(monkeypatch)
    expected, pb = jax.vjp(lambda m, c: _oracle(m, c, static), mf, common)
    expected_bars = pb(jnp.asarray(1.0))
    work = mp2._correlation_term_specs(static)
    path = tmp_path / 'checkpoint'
    manager = RestartManager(path, method='weak-test', scientific_payload={'test': 1})
    zero_bars = mp2._zero_term_cotangents(mf, common)
    manager.save_record('correlation_progress',
        scalars={'energy': float(expected) if closed else 0.,
                 'completed_count': len(work) if closed else 1},
        trees=dict(zip(('mf_bar', 'common_bar'), expected_bars if closed else zero_bars)),
        metadata={'completed_specs': [list(spec) for spec in (work if closed else work[:1])],
                  'weak_response_closed': closed})
    manager.close()
    resumed = RestartManager(path, resume=True, method='weak-test', scientific_payload={'test': 1})
    if closed:
        actual = mp2._progressive_correlation_pullback(mf, common, static, restart=resumed)
        _assert_trees_close(actual, (expected, *expected_bars))
    else:
        with pytest.raises(RuntimeError, match='older partial weak response'):
            mp2._progressive_correlation_pullback(mf, common, static, restart=resumed)
    resumed.close()
    assert not calls
