"""HDF5 exchange tests for complete, fingerprinted domain packets."""

import os
import subprocess
import sys

import h5py
import numpy as np
import pytest

from pyscfad.dlno_stc.exchange import (
    read_input,
    read_result,
    write_input,
    write_result,
)
from pyscfad.dlno_stc.test.test_protocol import metadata, packet, result


def test_input_and_result_round_trip(tmp_path):
    inputs, md, controls = packet(), metadata(), {"global_seed": 9, "blocks": 5}
    input_path = tmp_path / "fragment_0000" / "input.h5"
    fingerprint = write_input(input_path, inputs, md, controls)
    with h5py.File(input_path) as handle:
        assert handle.attrs["complete"]
        assert handle.attrs["request_fingerprint"] == fingerprint
        assert set(handle["inputs"]) == {"foo", "fvv", "B"}
        assert handle["inputs/B"].compression is None
    loaded, loaded_md, loaded_controls = read_input(input_path)
    for key in inputs:
        np.testing.assert_array_equal(loaded[key], inputs[key])
    assert loaded_md["request_fingerprint"] == fingerprint
    assert loaded_controls == controls
    answer = result(inputs, md, controls)
    result_path = input_path.parent / "result.h5"
    write_result(result_path, answer)
    with h5py.File(result_path) as handle:
        assert handle.attrs["complete"]
        assert set(handle["cotangents"]) == {"foo", "fvv", "B"}
    loaded_answer = read_result(result_path, loaded, loaded_md,
                                controls=loaded_controls)
    assert loaded_answer["energy"] == answer["energy"]
    for key in inputs:
        np.testing.assert_array_equal(loaded_answer["cotangents"][key],
                                      answer["cotangents"][key])


def test_default_refuses_overwrite(tmp_path):
    inputs, md, controls = packet(), metadata(), {}
    input_path = tmp_path / "input.h5"
    result_path = tmp_path / "result.h5"
    write_input(input_path, inputs, md, controls)
    write_result(result_path, result(inputs, md, controls))
    with pytest.raises(FileExistsError):
        write_input(input_path, inputs, md, controls)
    with pytest.raises(FileExistsError):
        write_result(result_path, result(inputs, md, controls))


def test_stale_or_corrupt_request_is_rejected(tmp_path):
    inputs, md, controls = packet(), metadata(), {"global_seed": 4}
    path = tmp_path / "input.h5"
    write_input(path, inputs, md, controls)
    with h5py.File(path, "r+") as handle:
        handle["inputs/B"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        read_input(path)


def test_stale_result_controls_are_rejected(tmp_path):
    inputs, md, controls = packet(), metadata(), {"global_seed": 4}
    path = tmp_path / "result.h5"
    write_result(path, result(inputs, md, controls))
    with pytest.raises(ValueError, match="fingerprint"):
        read_result(path, inputs, md, controls={"global_seed": 5})


def test_disk_result_rejects_zero_samples_for_stochastic_request(tmp_path):
    inputs, md = packet(), metadata()
    controls = {"global_seed": 7, "sample_blocks": 3}
    answer = result(inputs, md, controls)
    answer["metadata"]["seed_replay"] = {
        "mode": "logical_blocks", "global_seed": 7, "sample_blocks": 3,
    }
    path = tmp_path / "result.h5"
    write_result(path, answer)
    with pytest.raises(ValueError, match="actual_sample_count"):
        read_result(path, inputs, md, controls=controls)


def test_incomplete_final_or_temporary_result_is_not_accepted(tmp_path):
    inputs, md, controls = packet(), metadata(), {}
    path = tmp_path / "result.h5"
    temp = tmp_path / ".result.h5.interrupted.tmp"
    with h5py.File(temp, "w") as handle:
        handle.create_dataset("energy", data=-0.1)
    with pytest.raises(FileNotFoundError):
        read_result(path, inputs, md, controls=controls)
    with h5py.File(path, "w") as handle:
        handle.attrs["complete"] = False
        handle.create_dataset("energy", data=-0.1)
    with pytest.raises(ValueError, match="incomplete"):
        read_result(path, inputs, md, controls=controls)


def test_serial_import_does_not_load_mpi_or_stc():
    script = """import sys
import pyscfad.dlno_stc
assert 'mpi4py' not in sys.modules
assert not any('stc' in name.lower() and name != 'pyscfad.dlno_stc'
               and not name.startswith('pyscfad.dlno_stc.')
               for name in sys.modules)
"""
    completed = subprocess.run([sys.executable, "-c", script],
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
