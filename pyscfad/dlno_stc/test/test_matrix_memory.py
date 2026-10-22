"""Matrix memory forecasts advise; proven impossible new arrays stop."""

import warnings

import numpy as np
import pytest

from pyscfad.dlno_stc import backend
from pyscfad.dlno_stc.controls import (
    check_solver_memory, normalize_matrix_controls,
)


def test_configured_matrix_budget_warns_and_continues():
    with pytest.warns(RuntimeWarning, match="matrix"):
        assert check_solver_memory({"total_bytes": 2_000_000}, 1.) == 2.


@pytest.mark.parametrize("limit", [True, 0, -1, np.nan, np.inf])
def test_solver_memory_preserves_budget_validation(limit):
    with pytest.raises(ValueError, match="memory_limit_mb"):
        check_solver_memory({"total_bytes": 1}, limit)


def observation(**overrides):
    from pyscfad.dlno_stc._memory import MemoryObservation

    values = dict(rss_mb=2., enforced_limit_mb=10.,
                  enforced_source="test cgroup", headroom_mb=8.,
                  host_available_mb=100., allowance_mb=10.,
                  allowance_source="test cgroup", rank=0)
    values.update(overrides)
    return MemoryObservation(**values)


def matrix_inputs(*, naux=4, nocc=2, nvir=3):
    return {"foo": -np.eye(nocc), "fvv": np.eye(nvir),
            "B": np.zeros((naux, nocc, nvir))}


def test_solver_workspace_peak_adds_rss_once():
    observed = observation(rss_mb=.8, allowance_mb=2.)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_solver_memory({"total_bytes": 1_000_000}, None,
                            observation=observed, estimate_kind="workspace")
        check_solver_memory({"total_bytes": 1_000_000}, None,
                            observation=observed, estimate_kind="peak")
    assert not caught
    with pytest.warns(RuntimeWarning):
        check_solver_memory({"total_bytes": 1_100_000}, None,
                            observation=observed, estimate_kind="workspace")


def test_solver_forecast_above_limit_continues_when_new_array_fits():
    with pytest.warns(RuntimeWarning):
        check_solver_memory({"total_bytes": 20_000_000}, 1.,
                            observation=observation(), unavoidable_bytes=1_000_000)


def test_solver_proven_new_array_above_verified_limit_stops():
    with pytest.raises(MemoryError):
        check_solver_memory({"total_bytes": 20_000_000}, None,
                            observation=observation(), unavoidable_bytes=10_000_001)


def test_solver_headroom_without_verified_limit_never_proves_failure():
    observed = observation(enforced_limit_mb=None, enforced_source=None,
                           headroom_mb=.001, host_available_mb=.001,
                           allowance_mb=None, allowance_source=None)
    with pytest.warns(RuntimeWarning):
        check_solver_memory({"total_bytes": 20_000_000}, 1.,
                            observation=observed, unavoidable_bytes=10_000_001)


@pytest.mark.parametrize("matrix_kind", ["D", "W"])
def test_matrix_peak_uses_one_observation_and_keeps_existing_estimate(monkeypatch, matrix_kind):
    from pyscfad.dlno_stc import _memory

    calls = []
    def observe():
        calls.append(None)
        return observation(rss_mb=12., allowance_mb=100.)
    monkeypatch.setattr(_memory, "observe_memory", observe)
    settings = normalize_matrix_controls({"mode": "deterministic"}, matrix_kind)
    estimate, rss_mb, peak_bytes = backend._check_matrix_memory(
        matrix_inputs(), 2, settings, matrix_kind, np.array([0, 4]))
    assert len(calls) == 1
    assert rss_mb == 12.
    assert peak_bytes == estimate["total_bytes"] + 12_000_000


@pytest.mark.parametrize("matrix_kind", ["D", "W"])
def test_unknown_rss_keeps_matrix_peak_unknown(monkeypatch, matrix_kind):
    from pyscfad.dlno_stc import _memory

    monkeypatch.setattr(_memory, "observe_memory", lambda: observation(rss_mb=None))
    settings = normalize_matrix_controls({"mode": "deterministic"}, matrix_kind)
    _, rss_mb, peak_bytes = backend._check_matrix_memory(
        matrix_inputs(naux=0), 2, settings, matrix_kind, np.array([0, 0]))
    assert rss_mb is None
    assert peak_bytes is None
    forward = backend.solve_matrices(matrix_inputs(naux=0),
                                     {"target_indices": (0, 1)}, {},
                                     matrix_kind=matrix_kind)
    assert forward["diagnostics"]["current_rss_mb"] is None
    assert forward["diagnostics"]["estimated_process_peak_mb"] is None


@pytest.mark.parametrize("matrix_kind", ["D", "W"])
@pytest.mark.parametrize("stage", ["forward", "reverse"])
def test_matrix_required_new_array_stops_before_allocation(monkeypatch, matrix_kind, stage):
    from pyscfad.dlno_stc import _memory

    monkeypatch.setattr(_memory, "observe_memory", lambda: observation(
        enforced_limit_mb=.00005, allowance_mb=.00005))
    settings = normalize_matrix_controls({"mode": "deterministic"}, matrix_kind)
    # The 3 x 3 float64 output is 72 bytes, independently of the forecast.
    with pytest.raises(MemoryError):
        backend._check_matrix_memory(matrix_inputs(naux=0), 2, settings,
                                     matrix_kind, np.array([0, 0]), stage=stage)


@pytest.mark.parametrize("matrix_kind", ["D", "W"])
def test_conservative_forecast_above_limit_continues_when_each_array_fits(monkeypatch, matrix_kind):
    from pyscfad.dlno_stc import _memory

    monkeypatch.setattr(_memory, "observe_memory", lambda: observation(
        enforced_limit_mb=.0002, allowance_mb=.0002))
    settings = normalize_matrix_controls({"mode": "deterministic"}, matrix_kind)
    # The largest new native copy is B (192 bytes); the forecast is much larger.
    with pytest.warns(RuntimeWarning):
        backend._check_matrix_memory(matrix_inputs(), 2, settings, matrix_kind,
                                     np.array([0, 4]))


@pytest.mark.parametrize("matrix_kind", ["D", "W"])
@pytest.mark.parametrize("stage", ["forward", "reverse"])
def test_native_input_copy_is_a_new_array(monkeypatch, matrix_kind, stage):
    from pyscfad.dlno_stc import _memory

    monkeypatch.setattr(_memory, "observe_memory", lambda: observation(
        enforced_limit_mb=.0001, allowance_mb=.0001))
    settings = normalize_matrix_controls({"mode": "deterministic"}, matrix_kind)
    with pytest.raises(MemoryError):
        backend._check_matrix_memory(matrix_inputs(), 2, settings, matrix_kind,
                                     np.array([0, 4]), stage=stage)


@pytest.mark.parametrize("matrix_kind", ["D", "W"])
@pytest.mark.parametrize("dimensions", [{"naux": 0}, {"nvir": 0}])
def test_forward_reverse_warn_and_continue_with_advisory_budget(monkeypatch, matrix_kind, dimensions):
    from pyscfad.dlno_stc import _memory

    monkeypatch.setattr(_memory, "observe_memory", lambda: observation())
    inputs = matrix_inputs(**dimensions)
    with pytest.warns(RuntimeWarning, match=f"matrix {matrix_kind} forward"):
        forward = backend.solve_matrices(inputs, {"target_indices": (0, 1)},
                                         {"memory_limit_mb": .001},
                                         matrix_kind=matrix_kind)
    with pytest.warns(RuntimeWarning, match=f"matrix {matrix_kind} reverse"):
        bars = backend.pullback_matrices(
            inputs, forward, energy_bar=1., occupied_bar=np.zeros_like(inputs["foo"]),
            virtual_bar=np.zeros_like(inputs["fvv"]))
    for name, value in bars.items():
        np.testing.assert_array_equal(value, np.zeros_like(inputs[name]))
    # Reusing one replay is still one calculation for warning purposes.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend.pullback_matrices(
            inputs, forward, energy_bar=1., occupied_bar=np.zeros_like(inputs["foo"]),
            virtual_bar=np.zeros_like(inputs["fvv"]))
    assert not caught


@pytest.mark.parametrize("matrix_kind", ["D", "W"])
def test_native_matrices_and_bars_are_unchanged_by_advisory_warning(monkeypatch, matrix_kind):
    from pyscfad.dlno_stc import _memory
    from pyscfad.dlno_stc.test.test_backend import native

    native()
    inputs = matrix_inputs()
    inputs["B"] = np.random.default_rng(57).normal(size=(4, 2, 3)) / 5
    monkeypatch.setattr(_memory, "observe_memory", lambda: observation())
    metadata = {"target_indices": (0, 1)}
    reference = backend.solve_matrices(inputs, metadata, {}, matrix_kind=matrix_kind)
    seeds = dict(energy_bar=.7, occupied_bar=np.eye(2), virtual_bar=np.eye(3))
    reference_bars = backend.pullback_matrices(inputs, reference, **seeds)
    with pytest.warns(RuntimeWarning, match=f"matrix {matrix_kind} forward"):
        advised = backend.solve_matrices(inputs, metadata, {"memory_limit_mb": .001},
                                         matrix_kind=matrix_kind)
    with pytest.warns(RuntimeWarning, match=f"matrix {matrix_kind} reverse"):
        advised_bars = backend.pullback_matrices(inputs, advised, **seeds)
    for name in ("energy", "occupied", "virtual"):
        np.testing.assert_array_equal(advised[name], reference[name])
    for name in inputs:
        np.testing.assert_array_equal(advised_bars[name], reference_bars[name])


@pytest.mark.parametrize("matrix_kind", ["D", "W"])
def test_reverse_rechecks_limit_after_successful_forward(monkeypatch, matrix_kind):
    from pyscfad.dlno_stc import _memory

    monkeypatch.setattr(_memory, "observe_memory", lambda: observation())
    inputs = matrix_inputs(naux=0)
    forward = backend.solve_matrices(inputs, {"target_indices": (0, 1)}, {},
                                     matrix_kind=matrix_kind)
    monkeypatch.setattr(_memory, "observe_memory", lambda: observation(
        enforced_limit_mb=.00005, allowance_mb=.00005))
    with pytest.raises(MemoryError, match=f"matrix {matrix_kind} reverse"):
        backend.pullback_matrices(inputs, forward, energy_bar=1.,
                                  occupied_bar=np.zeros_like(inputs["foo"]),
                                  virtual_bar=np.zeros_like(inputs["fvv"]))


def test_outer_calculation_deduplicates_separate_matrix_replays(monkeypatch):
    from pyscfad.dlno_stc import _memory

    monkeypatch.setattr(_memory, "observe_memory", lambda: observation())
    inputs = matrix_inputs(naux=0)

    @_memory.memory_calculation
    def calculation():
        for _ in range(2):
            forward = backend.solve_matrices(inputs, {"target_indices": (0, 1)},
                                             {"memory_limit_mb": .001})
            backend.pullback_matrices(inputs, forward, energy_bar=1.,
                                      occupied_bar=np.zeros_like(inputs["foo"]),
                                      virtual_bar=np.zeros_like(inputs["fvv"]))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calculation()
        calculation()
    assert len(caught) == 4
    assert sum("matrix D forward" in str(item.message) for item in caught) == 2
    assert sum("matrix D reverse" in str(item.message) for item in caught) == 2
