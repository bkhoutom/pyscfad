"""Deterministic tiny oracles; these do not validate an STC estimator."""

import numpy as np
import pytest

from pyscfad.dlno_stc.test.reference import (
    quadratic_energy, target_mp2_energy, reference_backend,
)


def packet():
    return {
        "foo": np.array([[-0.8, 0.07], [0.07, -0.5]]),
        "fvv": np.array([[0.3, -0.02, 0.04], [-0.02, 0.6, 0.01],
                         [0.04, 0.01, 0.9]]),
        "B": np.arange(24, dtype=float).reshape(4, 2, 3) / 40,
    }


def metadata(kind="quadratic_test_v1"):
    return {"schema_version": 1, "method": "mp2", "energy_kind": kind,
            "fragment_id": 3, "target_index": 0,
            "basis_frame": "orthonormal_local", "B_axes": "Pia",
            "request_fingerprint": "a" * 64}


def test_quadratic_energy_and_full_cotangents():
    x = packet()
    result = reference_backend(x, metadata(), {})
    expected = 0.5 * sum(np.sum(v * v) for v in x.values())
    assert result["energy"] == pytest.approx(expected)
    for key in x:
        np.testing.assert_allclose(result["cotangents"][key], x[key])
    assert result["metadata"]["request_fingerprint"] == "a" * 64
    assert result["metadata"]["cotangent_seed"] == 1.0
    assert result["metadata"]["actual_sample_count"] == 0


def test_diagonal_target_mp2_matches_hand_contraction_and_whole_sum():
    x = {"foo": np.diag([-0.8, -0.5]), "fvv": np.diag([0.3, 0.7]),
         "B": np.array([[[0.2, 0.3], [0.1, 0.4]]])}
    g = np.einsum("Pia,Pjb->ijab", x["B"], x["B"])
    denom = np.diag(x["foo"])[:, None, None, None] + np.diag(x["foo"])[None, :, None, None] - np.diag(x["fvv"])[None, None, :, None] - np.diag(x["fvv"])[None, None, None, :]
    t = g / denom
    targets = [np.einsum("jab,jab->", t[f], 2 * g[f] - g[f].transpose(0, 2, 1)) for f in range(2)]
    for f in range(2):
        assert target_mp2_energy(x, f) == pytest.approx(targets[f])
    assert sum(target_mp2_energy(x, f) for f in range(2)) == pytest.approx(sum(targets))
    assert abs(targets[0] - sum(targets)) > 1e-5
    assert reference_backend(x, metadata("whole_domain_mp2_test_v1"), {})["energy"] == pytest.approx(sum(targets))


def test_target_gradient_matches_directional_differences():
    x = packet()
    result = reference_backend(x, metadata("boys_target_mp2_v1"), {})
    directions = {"foo": np.array([[0.2, -0.3], [-0.3, 0.1]]),
                  "fvv": np.array([[0.1, 0.2, -0.1], [0.2, -0.3, 0.4], [-0.1, 0.4, 0.2]]),
                  "B": np.sin(np.arange(24)).reshape(4, 2, 3) / 10}
    for key, direction in directions.items():
        analytic = np.sum(result["cotangents"][key] * direction)
        errors = []
        for h in (1e-3, 3e-4, 1e-4, 3e-5):
            plus = {k: v + h * direction if k == key else v for k, v in x.items()}
            minus = {k: v - h * direction if k == key else v for k, v in x.items()}
            finite = (target_mp2_energy(plus, 0) - target_mp2_energy(minus, 0)) / (2 * h)
            errors.append(abs(finite - analytic))
        assert min(errors) < 1e-8 + 1e-5 * abs(analytic)
    np.testing.assert_allclose(result["cotangents"]["foo"], result["cotangents"]["foo"].T)
    np.testing.assert_allclose(result["cotangents"]["fvv"], result["cotangents"]["fvv"].T)


def test_empty_virtual_and_empty_occupied_cases():
    x = {"foo": np.eye(1), "fvv": np.empty((0, 0)), "B": np.empty((2, 1, 0))}
    result = reference_backend(x, metadata("boys_target_mp2_v1"), {})
    assert result["energy"] == 0
    assert all(np.all(v == 0) for v in result["cotangents"].values())
    y = {"foo": np.empty((0, 0)), "fvv": np.eye(1), "B": np.empty((2, 0, 1))}
    assert reference_backend(y, metadata(), {})["energy"] == 0.5
    with pytest.raises(ValueError, match="active occupied"):
        reference_backend(y, metadata("boys_target_mp2_v1"), {})


def test_serial_sampled_backend_adds_deterministic_term_once():
    from pyscfad.dlno_stc.parallel import run_backend
    from pyscfad.dlno_stc.test.reference import sampled_backend
    x = packet()
    controls = {"global_seed": 19, "sample_blocks": 7}
    result = run_backend(x, metadata(), controls, sampled_backend)
    coefficients = [1.0 + 0.25 * np.sin(19 + block) for block in range(7)]
    factor = 0.125 + np.mean(coefficients)
    assert result["energy"] == pytest.approx(factor * quadratic_energy(x))
    for key in x:
        np.testing.assert_allclose(result["cotangents"][key], factor * x[key])
    assert result["metadata"]["actual_sample_count"] == 7


def test_serial_preflight_rejects_zero_total_samples():
    from pyscfad.dlno_stc.parallel import run_backend
    from pyscfad.dlno_stc.test.reference import sampled_backend
    with pytest.raises(ValueError, match="sample_blocks"):
        run_backend(packet(), metadata(), {"sample_blocks": 0}, sampled_backend)


def test_nondiagonal_target_matches_independent_linear_amplitude_solve():
    x = packet()
    foo, fvv, B = x["foo"], x["fvv"], x["B"]
    g = np.einsum("Pia,Pjb->ijab", B, B)
    shape = g.shape
    matrix = np.empty((g.size, g.size))
    for column in range(g.size):
        basis = np.zeros(shape)
        basis.flat[column] = 1
        action = (np.einsum("ik,kjab->ijab", foo, basis)
                  + np.einsum("jk,ikab->ijab", foo, basis)
                  - np.einsum("ac,ijcb->ijab", fvv, basis)
                  - np.einsum("bc,ijac->ijab", fvv, basis))
        matrix[:, column] = action.ravel()
    amplitudes = np.linalg.solve(matrix, g.ravel()).reshape(shape)
    expected = np.einsum("jab,jab->", amplitudes[0],
                         2 * g[0] - g[0].transpose(0, 2, 1))
    assert target_mp2_energy(x, 0) == pytest.approx(expected, rel=1e-10, abs=1e-12)
