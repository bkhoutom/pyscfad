"""Host-side checks for the local MP2 packet contract."""

import numpy as np
import pytest

from pyscfad.dlno_stc.protocol import (
    check_replay,
    request_fingerprint,
    validate_inputs,
    validate_result,
)


def packet():
    foo = np.array([[-0.8, 0.07], [0.07, -0.5]], dtype=np.float64)
    fvv = np.array([[0.3, -0.02, 0.04], [-0.02, 0.6, 0.01],
                    [0.04, 0.01, 0.9]], dtype=np.float64)
    B = np.arange(24, dtype=np.float64).reshape(4, 2, 3) / 40
    return {"foo": foo, "fvv": fvv, "B": B}


def metadata():
    return {
        "schema_version": 1,
        "method": "mp2",
        "energy_kind": "boys_target_laplace_mp2_v1",
        "fragment_id": 0,
        "target_indices": [1],
        "basis_frame": "orthonormal_local",
        "B_axes": "Pia",
        "units": {"energy": "Eh", "length": "bohr", "fock": "Eh", "B": "Eh**0.5"},
        "frozen": 1,
        "orbital_order": "checkpoint-orbitals-v1",
        "auxiliary_order": "checkpoint-aux-v1",
        "checkpoint_id": "checkpoint-sha256",
        "domain_options": {"force_full_domains": True},
        "code_revision": "test-revision",
        "dirty_worktree": False,
    }


def result(inputs, md, controls):
    return {
        "energy": np.float64(-0.125),
        "cotangents": {key: np.ones_like(value) for key, value in inputs.items()},
        "metadata": {
            **md,
            "request_fingerprint": request_fingerprint(inputs, md, controls),
            "cotangent_seed": 1.0,
            "backend": "tiny_reference",
            "backend_version": "1",
            "actual_sample_count": 0,
            "derivative_convention": "fixed_sample",
            "seed_replay": {
                "mode": "deterministic",
                **({"global_seed": controls["global_seed"]}
                   if "global_seed" in controls else {}),
            },
        },
        "diagnostics": {},
    }


def test_valid_full_fock_packet_and_unit_seed_result():
    inputs, md, controls = packet(), metadata(), {"global_seed": 7}
    assert validate_inputs(inputs, md) is None
    assert validate_result(result(inputs, md, controls), inputs, md,
                           controls=controls) is None


def finite_metadata():
    md = metadata()
    md.update(energy_kind="finite_boys_target_mp2_v1",
              global_target_orbitals=[5], candidate_order=[2, 5],
              frame_convention="local_before_fock_v1",
              virtual_anchor_parent_columns=[0, 1, 2],
              grouping_fingerprint="a" * 64,
              budget={"total_tolerance": .0003, "original_target_count": 2,
                      "group_count": 1, "effective_tolerance": .0003})
    return md


def test_finite_packet_declares_selected_mp2_and_rejects_old_identity():
    inputs, md = packet(), finite_metadata()
    validate_inputs(inputs, md)
    for kind in ("finite_three_tensor_v1", "boys_target_mp2_v1"):
        with pytest.raises(ValueError, match="energy_kind"):
            validate_inputs(inputs, {**md, "energy_kind": kind})


@pytest.mark.parametrize("count", [2, 4, 9])
def test_finite_packet_budget_uses_equal_group_variances(count):
    from types import SimpleNamespace
    from pyscfad.dlno_stc.mp2 import _validate_budget

    inputs, md = packet(), finite_metadata()
    total = .0003
    effective = float(total / np.sqrt(count))
    md['budget'] = dict(total_tolerance=total, original_target_count=16,
                        group_count=count, effective_tolerance=effective)
    controls = dict(energy_tolerance=effective)
    static, groups = SimpleNamespace(fragments=(None,) * 16), (None,) * count
    validate_inputs(inputs, md)
    request_fingerprint(inputs, md, controls)
    _validate_budget(md['budget'], controls, static, groups)

    # Consistent metadata and controls using the discarded rule must be rejected.
    md['budget']['effective_tolerance'] = total / count
    controls['energy_tolerance'] = total / count
    with pytest.raises(ValueError, match='budget'):
        validate_inputs(inputs, md)
    with pytest.raises(ValueError, match='budget'):
        _validate_budget(md['budget'], controls, static, groups)


@pytest.mark.parametrize("targets", [None, [], [0, 0], [-1], [2], [True], [0.5]])
def test_invalid_selected_targets(targets):
    with pytest.raises(ValueError, match="target_indices"):
        validate_inputs(packet(), {**metadata(), "target_indices": targets})


@pytest.mark.parametrize("field", ["global_target_orbitals", "candidate_order",
    "frame_convention", "virtual_anchor_parent_columns", "grouping_fingerprint", "budget"])
def test_finite_packet_requires_complete_frame_identity(field):
    md = finite_metadata()
    md.pop(field)
    with pytest.raises(ValueError, match=field):
        validate_inputs(packet(), md)


@pytest.mark.parametrize("change, match", [
    (lambda x, m: x["foo"].__setitem__((0, 1), 0.2), "symmetric"),
    (lambda x, m: x.__setitem__("B", x["B"][:, :, :2]), "shape"),
    (lambda x, m: x.__setitem__("B", x["B"].astype(np.float32)), "float64"),
    (lambda x, m: x["B"].__setitem__((0, 0, 0), np.nan), "finite"),
    (lambda x, m: m.__setitem__("B_axes", "Paj"), "B_axes"),
    (lambda x, m: m.__setitem__("method", "ccsd"), "method"),
    (lambda x, m: m.__setitem__("target_indices", 2), "target_indices"),
    (lambda x, m: m.__setitem__("energy_kind", "unknown"), "energy_kind"),
])
def test_invalid_input_is_rejected(change, match):
    inputs, md = packet(), metadata()
    change(inputs, md)
    with pytest.raises(ValueError, match=match):
        validate_inputs(inputs, md)


def test_non_array_input_has_an_informative_error():
    inputs, md = packet(), metadata()
    inputs["foo"] = [[-0.8, 0.07], [0.07, -0.5]]
    with pytest.raises(ValueError, match="foo.*float64 array"):
        validate_inputs(inputs, md)


def test_hash_and_replay_read_df_in_bounded_slices():
    class TrackedArray(np.ndarray):
        def __getitem__(self, key):
            block = super().__getitem__(key)
            if isinstance(block, np.ndarray) and block.size > 1 << 16:
                raise AssertionError("DF access exceeded the bounded block size")
            return block

    inputs, md = packet(), metadata()
    inputs["B"] = np.zeros((70, 20, 80), dtype=np.float64).view(TrackedArray)
    inputs["foo"] = np.eye(20, dtype=np.float64)
    inputs["fvv"] = np.eye(80, dtype=np.float64)
    request_fingerprint(inputs, md, {})
    check_replay(inputs, inputs)


def test_fingerprint_covers_controls_and_values_but_not_timestamp():
    inputs, md = packet(), metadata()
    controls = {"global_seed": 3, "sample_blocks": 10}
    digest = request_fingerprint(inputs, md, controls)
    assert digest == request_fingerprint(inputs,
        {**md, "timestamp": "tomorrow", "request_fingerprint": "ignored"},
        {"sample_blocks": 10, "global_seed": 3})
    assert digest != request_fingerprint(inputs, md,
                                          {**controls, "sample_blocks": 11})
    inputs["B"][0, 0, 0] += 1e-7
    assert digest != request_fingerprint(inputs, md, controls)


@pytest.mark.parametrize("change, match", [
    (lambda r: r["metadata"].__setitem__("request_fingerprint", "0" * 64), "fingerprint"),
    (lambda r: r["metadata"].__setitem__("energy_kind", "quadratic_test_v1"), "energy_kind"),
    (lambda r: r["metadata"].__setitem__("target_indices", 0), "target_indices"),
    (lambda r: r["metadata"].__setitem__("cotangent_seed", -2.0), "cotangent_seed"),
    (lambda r: r["cotangents"].pop("B"), "B"),
    (lambda r: r["cotangents"].__setitem__("foo", r["cotangents"]["foo"][:, :1]), "shape"),
    (lambda r: r["cotangents"]["B"].__setitem__((0, 0, 0), np.nan), "finite"),
    (lambda r: r["cotangents"]["foo"].__setitem__((0, 1), 7.0), "symmetric"),
])
def test_invalid_result_is_rejected(change, match):
    inputs, md, controls = packet(), metadata(), {}
    answer = result(inputs, md, controls)
    change(answer)
    with pytest.raises(ValueError, match=match):
        validate_result(answer, inputs, md, controls=controls)


def test_result_requires_nonempty_seed_replay_metadata():
    inputs, md, controls = packet(), metadata(), {}
    answer = result(inputs, md, controls)
    answer["metadata"].pop("seed_replay", None)
    with pytest.raises(ValueError, match="seed_replay"):
        validate_result(answer, inputs, md, controls=controls)
    answer["metadata"]["seed_replay"] = {}
    with pytest.raises(ValueError, match="seed_replay"):
        validate_result(answer, inputs, md, controls=controls)


def test_stochastic_request_rejects_zero_actual_samples():
    inputs, md = packet(), metadata()
    controls = {"global_seed": 7, "sample_blocks": 3}
    answer = result(inputs, md, controls)
    answer["metadata"]["seed_replay"] = {
        "mode": "logical_blocks", "global_seed": 7, "sample_blocks": 3,
    }
    with pytest.raises(ValueError, match="actual_sample_count"):
        validate_result(answer, inputs, md, controls=controls)


def test_seed_replay_matches_requested_global_seed():
    inputs, md = packet(), metadata()
    controls = {"global_seed": 7}
    answer = result(inputs, md, controls)
    answer["metadata"]["seed_replay"] = {
        "mode": "deterministic", "global_seed": 8,
    }
    with pytest.raises(ValueError, match="seed_replay.global_seed"):
        validate_result(answer, inputs, md, controls=controls)


def test_result_requires_trusted_match_context():
    inputs, md, controls = packet(), metadata(), {}
    with pytest.raises(ValueError, match="fingerprint|controls"):
        validate_result(result(inputs, md, controls), inputs, md)


def test_replay_checks_all_tensors_with_default_tolerances():
    saved = packet()
    replayed = {key: value.copy() for key, value in saved.items()}
    replayed["B"][0, 0, 0] += 5e-12
    check_replay(replayed, saved)
    replayed["B"][0, 0, 0] += 1e-6
    with pytest.raises(ValueError, match="replay mismatch"):
        check_replay(replayed, saved)
    replayed["B"] = saved["B"][::-1]
    with pytest.raises(ValueError, match="replay mismatch"):
        check_replay(replayed, saved)


def test_unknown_dirty_worktree_status_is_accepted():
    inputs, md = packet(), metadata()
    md["code_revision"] = "unknown"
    md["dirty_worktree"] = None
    validate_inputs(inputs, md)
    assert len(request_fingerprint(inputs, md, {})) == 64


@pytest.mark.parametrize("unit_name, wrong_value", [
    ("energy", "eV"),
    ("length", "angstrom"),
])
def test_unsupported_request_units_are_rejected(unit_name, wrong_value):
    inputs, md = packet(), metadata()
    md["units"][unit_name] = wrong_value
    with pytest.raises(ValueError, match="units"):
        validate_inputs(inputs, md)


def test_finite_response_frame_and_budget_are_checked():
    x, md = packet(), finite_metadata()
    controls = {"energy_tolerance": .0003}
    answer = result(x, md, controls)
    validate_result(answer, x, md, controls=controls)
    answer["metadata"]["frame_convention"] = "semicanonical"
    with pytest.raises(ValueError, match="frame_convention"):
        validate_result(answer, x, md, controls=controls)
    with pytest.raises(ValueError, match="budget"):
        request_fingerprint(x, md, {"energy_tolerance": .00015})
