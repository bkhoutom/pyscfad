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


@pytest.mark.parametrize("order", [2, 3, 4])
def test_progressive_weak_response_builds_each_endpoint_twice(monkeypatch, order):
    assert hasattr(mp2, "build_weak_endpoint"), "fragment endpoint builder is missing"
    static = _static(order=order)
    mf, common = jnp.asarray(0.4), _Common(jnp.eye(2), jnp.asarray(0.23))
    calls = Counter()
    def counted_endpoint(mf_, common_, static_, index):
        calls[index] += 1
        return _endpoint(mf_, common_, static_, index)
    monkeypatch.setattr(mp2, "build_weak_endpoint", counted_endpoint)
    monkeypatch.setattr(mp2, "_correlation_term_energy", _strong)
    value, mf_bar, common_bar = mp2._progressive_correlation_pullback(mf, common, static)
    expected, pullback = jax.vjp(lambda m, c: _oracle(m, c, static), mf, common)
    _assert_trees_close((value, mf_bar, common_bar), (expected, *pullback(jnp.asarray(1.0))))
    assert calls == Counter({i: 2 for i in range(4)})


def test_progressive_all_strong_never_builds_weak_endpoint(monkeypatch):
    assert hasattr(mp2, "build_weak_endpoint"), "fragment endpoint builder is missing"
    static = _static(all_strong=True)
    mf, common = jnp.asarray(0.4), _Common(jnp.eye(2), jnp.asarray(0.23))
    def forbidden(*args):
        raise AssertionError("all-strong topology built a weak endpoint")
    monkeypatch.setattr(mp2, "build_weak_endpoint", forbidden)
    monkeypatch.setattr(mp2, "_correlation_term_energy", _strong)
    actual = mp2._progressive_correlation_pullback(mf, common, static)
    expected, pullback = jax.vjp(lambda m, c: _oracle(m, c, static), mf, common)
    _assert_trees_close(actual, (expected, *pullback(jnp.asarray(1.0))))


@pytest.mark.parametrize('all_strong', [False, True])
def test_progressive_collects_strong_terms_and_one_complete_weak_response(
    monkeypatch, all_strong,
):
    static = _static(all_strong=all_strong)
    mf, common = jnp.asarray(0.4), _Common(jnp.eye(2), jnp.asarray(0.23))
    expected, pullback = jax.vjp(lambda m, c: _oracle(m, c, static), mf, common)
    expected_bars = pullback(jnp.asarray(1.0))
    calls, collections = Counter(), []
    compact_values = []

    def counted(m, c, s, i):
        calls[i] += 1
        result = _endpoint(m, c, s, i)
        if calls[i] == 1:
            compact_values.append(weakref.ref(result[0]))
        return result

    def collect():
        collections.append(calls.copy())
        if calls and set(calls.values()) == {2}:
            assert all(value() is None for value in compact_values)

    monkeypatch.setattr(mp2, 'build_weak_endpoint', counted)
    monkeypatch.setattr(mp2, '_correlation_term_energy', _strong)
    monkeypatch.setattr(mp2.gc, 'collect', collect)
    actual = mp2._progressive_correlation_pullback(mf, common, static)
    _assert_trees_close(actual, (expected, *expected_bars))
    assert len(collections) == 4 + (0 if all_strong else 1)
    # The large endpoint tapes are released individually, without a full
    # heap scan while any of the weak endpoint response is still pending.
    assert all(not counts or set(counts.values()) in ({1}, {2})
               for counts in collections)


@pytest.mark.parametrize("stop_count", [1, 3, 7, 10])
@pytest.mark.parametrize("return_details", [False, True])
def test_restart_prefix_keeps_pending_endpoint_cotangents(
    tmp_path, monkeypatch, stop_count, return_details,
):
    assert hasattr(mp2, "build_weak_endpoint"), "fragment endpoint builder is missing"
    from pyscfad.dlno import _restart
    static = _static()
    mf, common = jnp.asarray(0.4), _Common(jnp.eye(2), jnp.asarray(0.23))
    monkeypatch.setattr(mp2, "build_weak_endpoint", _endpoint)
    monkeypatch.setattr(mp2, "_correlation_term_energy", _strong)
    reference = mp2._progressive_correlation_pullback(
        mf, common, static, return_details=return_details,
    )
    manager = RestartManager(
        tmp_path / "checkpoint", method="weak-test",
        scientific_payload={"test": 1},
    )
    class Stop(RuntimeError):
        pass
    saves = 0
    def stop_after_prefix(stage, key, path):
        nonlocal saves
        if stage == "correlation_progress":
            saves += 1
            if saves == stop_count:
                raise Stop
    monkeypatch.setattr(_restart, "_CHECKPOINT_EVENT_HOOK", stop_after_prefix)
    with pytest.raises(Stop):
        mp2._progressive_correlation_pullback(
            mf, common, static, restart=manager, return_details=return_details,
        )
    manager.close()
    monkeypatch.setattr(_restart, "_CHECKPOINT_EVENT_HOOK", None)
    resumed = RestartManager(
        tmp_path / "checkpoint", resume=True, method="weak-test",
        scientific_payload={"test": 1},
    )
    actual = mp2._progressive_correlation_pullback(
        mf, common, static, restart=resumed, return_details=return_details,
    )
    resumed.close()
    _assert_trees_close(actual[:3], reference[:3])
    if return_details:
        assert actual[3].e_strong == reference[3].e_strong
        assert actual[3].e_weak == reference[3].e_weak
        assert len(actual[3].terms) == len(reference[3].terms)


def test_serial_progress_counts_are_unique_and_last_weak_is_closed(tmp_path, monkeypatch):
    from pyscfad.dlno import _restart
    static = _static()
    mf, common = jnp.asarray(0.4), _Common(jnp.eye(2), jnp.asarray(0.23))
    monkeypatch.setattr(mp2, "build_weak_endpoint", _endpoint)
    monkeypatch.setattr(mp2, "_correlation_term_energy", _strong)
    manager = RestartManager(
        tmp_path / "checkpoint", method="weak-test",
        scientific_payload={"test": 1},
    )
    counts, closed = [], []
    def inspect_progress(stage, key, path):
        if stage != "correlation_progress":
            return
        templates = dict(zip(("mf_bar", "common_bar"), mp2._zero_term_cotangents(mf, common)))
        record = manager.load_record(stage, templates=templates)
        counts.append(record.scalars["completed_count"])
        closed.append(record.metadata["weak_response_closed"])
    monkeypatch.setattr(_restart, "_CHECKPOINT_EVENT_HOOK", inspect_progress)
    mp2._progressive_correlation_pullback(mf, common, static, restart=manager)
    assert counts == list(range(1, 11))
    assert closed[-2:] == [True, True]
    manager.close()


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("collect_timing", [False, True])
def test_mpi_weak_phase_matches_pairwise_oracle(
    monkeypatch, order, collect_timing,
):
    mpi = pytest.importorskip("mpi4py.MPI")
    from pyscfad.dlno import mp2_mpi
    assert hasattr(mp2_mpi, "_weak_endpoint_correlation_pullback"), (
        "distributed endpoint weak phase is missing"
    )
    static = _static(order=order)
    mf, common = jnp.asarray(0.4), _Common(jnp.eye(2), jnp.asarray(0.23))
    calls = Counter()
    def counted(m, c, s, i):
        calls[i] += 1
        return _endpoint(m, c, s, i)
    monkeypatch.setattr(mp2_mpi, "build_weak_endpoint", counted)
    weak_specs = tuple(spec for spec in mp2._correlation_term_specs(static) if spec[0] == "weak")
    actual = mp2_mpi._weak_endpoint_correlation_pullback(
        mf, common, static, weak_specs, comm=mpi.COMM_SELF, root=0,
        collect_timing=collect_timing,
    )
    def oracle(m, c):
        return sum(
            _pair_oracle(
                _endpoint(m, c, static, left),
                _endpoint(m, c, static, right), order,
            ) for _, left, right in weak_specs
        )
    expected, pullback = jax.vjp(oracle, mf, common)
    _assert_trees_close(actual[:3], (expected, *pullback(jnp.asarray(1.0))))
    assert calls == Counter({i: 2 for i in range(4)})
    if collect_timing:
        assert len(actual[3]) == len(weak_specs)
        assert all(row.frame_build_seconds >= 0 for row in actual[3])
        assert all(row.frame_replay_seconds >= 0 for row in actual[3])


@pytest.mark.parametrize('all_strong', [False, True])
def test_mpi_weak_collects_once_after_complete_response(monkeypatch, all_strong):
    mpi = pytest.importorskip('mpi4py.MPI')
    from pyscfad.dlno import mp2_mpi
    static = _static(all_strong=all_strong)
    mf, common = jnp.asarray(0.4), _Common(jnp.eye(2), jnp.asarray(0.23))
    weak_specs = tuple(spec for spec in mp2._correlation_term_specs(static)
                       if spec[0] == 'weak')

    def oracle(m, c):
        return sum((_pair_oracle(_endpoint(m, c, static, left),
                    _endpoint(m, c, static, right), 4)
                    for _, left, right in weak_specs), jnp.asarray(0.))

    expected, pullback = jax.vjp(oracle, mf, common)
    expected_bars = pullback(jnp.asarray(1.0))
    calls, collections = Counter(), []
    compact_values = []

    def counted(m, c, s, i):
        calls[i] += 1
        result = _endpoint(m, c, s, i)
        if calls[i] == 1:
            compact_values.append(weakref.ref(result[0]))
        return result

    def collect():
        collections.append(calls.copy())
        assert all(value() is None for value in compact_values)

    monkeypatch.setattr(mp2_mpi, 'build_weak_endpoint', counted)
    monkeypatch.setattr(mp2_mpi.gc, 'collect', collect)
    actual = mp2_mpi._weak_endpoint_correlation_pullback(
        mf, common, static, weak_specs, comm=mpi.COMM_SELF, root=0)
    _assert_trees_close(actual[:3], (expected, *expected_bars))
    assert collections == ([] if all_strong else [Counter({i: 2 for i in range(4)})])


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
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)
    _assert_trees_close(actual_bars, expected_bars)


def test_sparse_weak_graph_builds_only_incident_endpoints(monkeypatch):
    static = _static(5, all_strong=True)
    for left, right in ((0, 2), (1, 2)):
        static.strong_mask[left, right] = static.strong_mask[right, left] = False
    mf, common = jnp.asarray(0.4), _Common(jnp.eye(2), jnp.asarray(0.23))
    calls = Counter()
    def counted(m, c, s, i):
        calls[i] += 1
        return _endpoint(m, c, s, i)
    monkeypatch.setattr(mp2, "build_weak_endpoint", counted)
    monkeypatch.setattr(mp2, "_correlation_term_energy", _strong)
    actual = mp2._progressive_correlation_pullback(mf, common, static)
    expected, pullback = jax.vjp(lambda m, c: _oracle(m, c, static), mf, common)
    _assert_trees_close(actual, (expected, *pullback(jnp.asarray(1.0))))
    assert calls == Counter({0: 2, 1: 2, 2: 2})


def _phased_endpoint(endpoint, alternate=False):
    """Flip individual virtual columns differently in each occupied mode."""
    weights, records = endpoint
    changed = []
    for mode, (center, mu, gap, theta, omega) in enumerate(records):
        signs = jnp.asarray([-1.0, 1.0])
        if (mode % 2 == 1) != alternate:
            signs = -signs
        changed.append((
            center, mu * signs, gap,
            None if theta is None else theta * signs,
            None if omega is None else omega * signs,
        ))
    return weights, tuple(changed)


def test_endpoint_gauge_tolerates_c20_near_zero_moment_roundoff():
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
    mp2._check_weak_endpoint_gauge(rebuilt, reference, 0, "response replay")


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("stop_count", [2, 7])
def test_restart_aligns_pending_bars_and_independently_phased_replay(
    tmp_path, monkeypatch, order, stop_count,
):
    from pyscfad.dlno import _restart
    static = _static(order=order)
    mf, common = jnp.asarray(0.4), _Common(jnp.eye(2), jnp.asarray(0.23))
    monkeypatch.setattr(mp2, "build_weak_endpoint", _endpoint)
    monkeypatch.setattr(mp2, "_correlation_term_energy", _strong)
    expected, pullback = jax.vjp(lambda m, c: _oracle(m, c, static), mf, common)
    expected_bars = pullback(jnp.asarray(1.0))
    path = tmp_path / "checkpoint"
    manager = RestartManager(
        path, method="weak-test", scientific_payload={"test": 1},
    )
    class Stop(RuntimeError):
        pass
    count = 0
    def stop_after_weak(stage, key, path):
        nonlocal count
        if stage == "correlation_progress":
            count += 1
            if count == stop_count:
                raise Stop
    monkeypatch.setattr(_restart, "_CHECKPOINT_EVENT_HOOK", stop_after_weak)
    with pytest.raises(Stop):
        mp2._progressive_correlation_pullback(mf, common, static, restart=manager)
    manager.close()
    monkeypatch.setattr(_restart, "_CHECKPOINT_EVENT_HOOK", None)
    calls = Counter()
    def changed_endpoint(m, c, s, index):
        calls[index] += 1
        return _phased_endpoint(
            _endpoint(m, c, s, index), alternate=calls[index] > 1,
        )
    monkeypatch.setattr(mp2, "build_weak_endpoint", changed_endpoint)
    resumed = RestartManager(
        path, resume=True, method="weak-test", scientific_payload={"test": 1},
    )
    actual = mp2._progressive_correlation_pullback(mf, common, static, restart=resumed)
    resumed.close()
    _assert_trees_close(actual, (expected, *expected_bars))
    assert calls == Counter({i: 2 for i in range(4)})


@pytest.mark.parametrize("order", [2, 3, 4])
def test_response_replay_aligns_changed_endpoint_phase_and_gradient(monkeypatch, order):
    static = _static(order=order)
    mf, common = jnp.asarray(0.4), _Common(jnp.eye(2), jnp.asarray(0.23))
    calls = Counter()
    def changed_endpoint(m, c, s, index):
        calls[index] += 1
        value = _endpoint(m, c, s, index)
        return _phased_endpoint(value) if calls[index] > 1 else value
    monkeypatch.setattr(mp2, "build_weak_endpoint", changed_endpoint)
    monkeypatch.setattr(mp2, "_correlation_term_energy", _strong)
    expected, pullback = jax.vjp(lambda m, c: _oracle(m, c, static), mf, common)
    expected_bars = pullback(jnp.asarray(1.0))
    actual = mp2._progressive_correlation_pullback(mf, common, static)
    _assert_trees_close(actual, (expected, *expected_bars))
    assert calls == Counter({i: 2 for i in range(4)})


@pytest.mark.parametrize("change", ["weights", "gap", "moments", "rotation", "shape"])
def test_response_replay_rejects_non_phase_endpoint_changes(monkeypatch, change):
    static = _static()
    mf, common = jnp.asarray(0.4), _Common(jnp.eye(2), jnp.asarray(0.23))
    calls = Counter()

    def changed_endpoint(m, c, s, index):
        calls[index] += 1
        weights, records = _endpoint(m, c, s, index)
        if index == 0 and calls[index] > 1:
            center, mu, gap, theta, omega = records[0]
            if change == "weights":
                weights = weights + 1e-4
            elif change == "gap":
                gap = gap + 1e-4
            elif change == "moments":
                mu = mu.at[0, 0].add(1e-4)
            elif change == "rotation":
                angle = 0.3
                rotation = jnp.asarray([
                    [jnp.cos(angle), -jnp.sin(angle)],
                    [jnp.sin(angle), jnp.cos(angle)],
                ])
                mu, theta, omega = (
                    jnp.einsum("...a,ab->...b", moment, rotation)
                    for moment in (mu, theta, omega)
                )
            else:
                mu, gap = mu[:, :1], gap[:1]
                theta, omega = theta[..., :1], omega[..., :1]
            records = ((center, mu, gap, theta, omega), *records[1:])
        return weights, records

    monkeypatch.setattr(mp2, "build_weak_endpoint", changed_endpoint)
    monkeypatch.setattr(mp2, "_correlation_term_energy", _strong)
    with pytest.raises(RuntimeError, match="gauge.*fragment 0"):
        mp2._progressive_correlation_pullback(mf, common, static)


def test_mpi_response_replay_tolerates_tiny_primal_roundoff(monkeypatch):
    mpi = pytest.importorskip("mpi4py.MPI")
    from pyscfad.dlno import mp2_mpi
    static = _static()
    mf, common = jnp.asarray(0.4), _Common(jnp.eye(2), jnp.asarray(0.23))
    calls = Counter()

    def rounded_endpoint(m, c, s, index):
        calls[index] += 1
        endpoint = _endpoint(m, c, s, index)
        if calls[index] > 1:
            endpoint = jax.tree_util.tree_map(lambda leaf: leaf + 1e-12, endpoint)
        return endpoint

    monkeypatch.setattr(mp2_mpi, "build_weak_endpoint", rounded_endpoint)
    weak_specs = tuple(
        spec for spec in mp2._correlation_term_specs(static) if spec[0] == "weak"
    )
    actual = mp2_mpi._weak_endpoint_correlation_pullback(
        mf, common, static, weak_specs, comm=mpi.COMM_SELF, root=0,
    )

    def oracle(m, c):
        return sum(
            _pair_oracle(
                _endpoint(m, c, static, left),
                _endpoint(m, c, static, right),
                static.thresholds.multipole_order,
            ) for _, left, right in weak_specs
        )

    expected, pullback = jax.vjp(oracle, mf, common)
    _assert_trees_close(actual[:3], (expected, *pullback(jnp.asarray(1.0))))
    assert calls == Counter({i: 2 for i in range(4)})


def test_mpi_response_replay_rejects_changed_mode_weights(monkeypatch):
    mpi = pytest.importorskip("mpi4py.MPI")
    from pyscfad.dlno import mp2_mpi
    static = _static()
    mf, common = jnp.asarray(0.4), _Common(jnp.eye(2), jnp.asarray(0.23))
    calls = Counter()

    def changed_endpoint(m, c, s, index):
        calls[index] += 1
        weights, records = _endpoint(m, c, s, index)
        if index == 0 and calls[index] > 1:
            weights = weights + 1e-4
        return weights, records

    monkeypatch.setattr(mp2_mpi, "build_weak_endpoint", changed_endpoint)
    weak_specs = tuple(
        spec for spec in mp2._correlation_term_specs(static) if spec[0] == "weak"
    )
    with pytest.raises(RuntimeError, match="gauge.*fragment 0"):
        mp2_mpi._weak_endpoint_correlation_pullback(
            mf, common, static, weak_specs, comm=mpi.COMM_SELF, root=0,
        )


@pytest.mark.parametrize("order", [2, 3, 4])
def test_mpi_response_replay_aligns_changed_endpoint_phase_and_gradient(monkeypatch, order):
    mpi = pytest.importorskip("mpi4py.MPI")
    from pyscfad.dlno import mp2_mpi
    static = _static(order=order)
    mf, common = jnp.asarray(0.4), _Common(jnp.eye(2), jnp.asarray(0.23))
    calls = Counter()

    def changed_endpoint(m, c, s, index):
        calls[index] += 1
        endpoint = _endpoint(m, c, s, index)
        if calls[index] > 1:
            endpoint = _phased_endpoint(endpoint)
        return endpoint

    monkeypatch.setattr(mp2_mpi, "build_weak_endpoint", changed_endpoint)
    weak_specs = tuple(
        spec for spec in mp2._correlation_term_specs(static) if spec[0] == "weak"
    )
    actual = mp2_mpi._weak_endpoint_correlation_pullback(
        mf, common, static, weak_specs, comm=mpi.COMM_SELF, root=0,
    )

    def oracle(m, c):
        return sum(
            _pair_oracle(_endpoint(m, c, static, left),
                         _endpoint(m, c, static, right), order)
            for _, left, right in weak_specs
        )

    expected, pullback = jax.vjp(oracle, mf, common)
    _assert_trees_close(actual[:3], (expected, *pullback(jnp.asarray(1.0))))
    assert calls == Counter({i: 2 for i in range(4)})


@pytest.mark.parametrize("family", [1, 3, 4])
def test_endpoint_gauge_scales_near_zero_noise_by_its_moment_column(family):
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
    mp2._check_weak_endpoint_gauge(rebuilt, reference, 0, "response replay")


@pytest.mark.parametrize("family", [1, 3, 4])
def test_endpoint_gauge_does_not_use_another_virtual_column_scale(family):
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
    with pytest.raises(RuntimeError, match="gauge.*fragment 0"):
        mp2._check_weak_endpoint_gauge(rebuilt, reference, 0, "response replay")


def test_endpoint_gauge_does_not_use_another_multipole_order_scale():
    static = _static()
    common = _Common(jnp.eye(2), jnp.asarray(0.23))
    weights, records = _endpoint(jnp.asarray(0.4), common, static, 0)
    center, mu, gap, theta, omega = records[0]
    reference_record = center, jnp.zeros_like(mu), gap, theta, omega * 1e8
    rebuilt_record = center, jnp.zeros_like(mu).at[0, 0].set(1e-6), gap, theta, omega * 1e8
    reference = weights, (reference_record, *records[1:])
    rebuilt = weights, (rebuilt_record, *records[1:])
    with pytest.raises(RuntimeError, match="gauge.*fragment 0"):
        mp2._check_weak_endpoint_gauge(rebuilt, reference, 0, "response replay")


@pytest.mark.parametrize("field", ["center", "gap", "weights"])
def test_endpoint_gauge_keeps_non_moment_componentwise_tolerances(field):
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
    with pytest.raises(RuntimeError, match="gauge.*fragment 0"):
        mp2._check_weak_endpoint_gauge(rebuilt, reference, 0, "response replay")


@pytest.mark.parametrize("value", [jnp.nan, jnp.inf])
def test_endpoint_gauge_rejects_nonfinite_moments(value):
    static = _static()
    common = _Common(jnp.eye(2), jnp.asarray(0.23))
    weights, records = _endpoint(jnp.asarray(0.4), common, static, 0)
    center, mu, gap, theta, omega = records[0]
    rebuilt = weights, ((center, mu, gap, theta, omega.at[0, 0, 0, 0].set(value)), *records[1:])
    reference = weights, records
    with pytest.raises(RuntimeError, match="gauge.*fragment 0"):
        mp2._check_weak_endpoint_gauge(rebuilt, reference, 0, "response replay")


@pytest.mark.parametrize("value", [jnp.nan, jnp.inf])
def test_endpoint_gauge_rejects_nonfinite_mode_weights(value):
    static = _static()
    common = _Common(jnp.eye(2), jnp.asarray(0.23))
    weights, records = _endpoint(jnp.asarray(0.4), common, static, 0)
    invalid = weights.at[0].set(value), records
    # Even identical infinities are invalid, despite numpy.allclose accepting
    # them. An endpoint containing nonfinite weights cannot define pair bars.
    with pytest.raises(RuntimeError, match="gauge.*fragment 0"):
        mp2._check_weak_endpoint_gauge(invalid, invalid, 0, "response replay")


@pytest.mark.parametrize("family", [1, 3, 4])
def test_endpoint_gauge_keeps_absolute_component_sized_roundoff_allowance(family):
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
    mp2._check_weak_endpoint_gauge(rebuilt, reference, 0, "response replay")
