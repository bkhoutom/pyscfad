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
        "energy_kind": "boys_target_mp2_v1",
        "fragment_id": 0,
        "target_index": 1,
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
            **{key: md[key] for key in (
                "schema_version", "method", "energy_kind", "fragment_id",
                "target_index", "basis_frame", "B_axes")},
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


def test_finite_packet_accepts_nondiagonal_local_fock_blocks():
    inputs, md, controls = packet(), metadata(), {}
    md.update(energy_kind="finite_three_tensor_v1", target_index=None)
    assert inputs["foo"][0, 1] != 0.0
    assert inputs["fvv"][0, 1] != 0.0
    assert validate_inputs(inputs, md) is None
    assert validate_result(result(inputs, md, controls), inputs, md,
                           controls=controls) is None


@pytest.mark.parametrize("change, match", [
    (lambda md: md.__setitem__("basis_frame", "semicanonical_local"),
     "orthonormal_local"),
    (lambda md: md.__setitem__("target_index", 0), "target_index"),
])
def test_finite_packet_rejects_wrong_frame_or_local_target(change, match):
    inputs, md = packet(), metadata()
    md.update(energy_kind="finite_three_tensor_v1", target_index=None)
    change(md)
    with pytest.raises(ValueError, match=match):
        validate_inputs(inputs, md)


@pytest.mark.parametrize("change, match", [
    (lambda x, m: x["foo"].__setitem__((0, 1), 0.2), "symmetric"),
    (lambda x, m: x.__setitem__("B", x["B"][:, :, :2]), "shape"),
    (lambda x, m: x.__setitem__("B", x["B"].astype(np.float32)), "float64"),
    (lambda x, m: x["B"].__setitem__((0, 0, 0), np.nan), "finite"),
    (lambda x, m: m.__setitem__("B_axes", "Paj"), "B_axes"),
    (lambda x, m: m.__setitem__("method", "ccsd"), "method"),
    (lambda x, m: m.__setitem__("target_index", 2), "target_index"),
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
    (lambda r: r["metadata"].__setitem__("target_index", 0), "target_index"),
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
