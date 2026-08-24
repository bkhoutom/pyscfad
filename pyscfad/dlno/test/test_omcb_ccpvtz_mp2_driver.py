import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


# Loading mpi4py for parser-only tests must not ask Open MPI to open a network
# transport inside a sandboxed test process.
os.environ.setdefault("OMPI_MCA_btl", "self")

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples/lno/20-mpi_omcb_iao_mp2_gradient_sweep.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "mpi_omcb_iao_mp2_gradient_sweep", _SCRIPT
)
driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(driver)


def test_exact_omcb_defaults_do_not_reuse_sto_prefix(monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(_SCRIPT)])
    args = driver._parser()
    assert Path(args.geometry).name == "OMCB.xyz"
    assert driver._basis_label(args.basis) == "cc-pvtz"
    assert args.auxbasis is None
    assert args.frozen is None
    assert args.pair_thresholds == [3e-2, 1e-2, 1e-3]
    assert args.output_prefix.endswith(
        "omcb_ccpvtz_iao_mp2_mpi_pair_sweep"
    )
    assert "sto3g" not in args.output_prefix.lower()


def test_none_parsers_preserve_auto_all_electron_convention():
    assert driver._parse_auxbasis("auto") is None
    assert driver._parse_auxbasis("none") is None
    assert driver._parse_frozen("all-electron") is None
    assert driver._parse_frozen("0") == 0


def test_driver_checkpoint_triplet_is_atomic_and_round_trips(tmp_path):
    prefix = tmp_path / "exact_driver"
    reference_gradient = np.arange(108.0).reshape(36, 3)
    metadata = {
        "reference": {
            "e_corr": -2.0,
            "e_total": -464.0,
        }
    }
    row = {field: 0.0 for field in driver.CSV_FIELDS}
    row.update({
        "pair_threshold": 3e-2,
        "e_corr": -1.95,
        "e_total": -463.95,
        "strong_pair_count": 57,
        "weak_pair_count": 9,
        "total_pair_count": 66,
        "nfragment": 12,
    })
    paths = driver._write_outputs(
        prefix,
        metadata,
        [row],
        [reference_gradient + 1.0],
        reference_gradient,
    )
    assert all(path.exists() for path in paths)
    assert not list(tmp_path.glob("*.tmp"))

    payload = json.loads(prefix.with_suffix(".json").read_text())
    assert payload["rows"][0]["pair_threshold"] == 3e-2
    with np.load(prefix.with_suffix(".npz")) as arrays:
        assert arrays["gradients"].shape == (1, 36, 3)
        np.testing.assert_array_equal(
            arrays["reference_gradient"], reference_gradient
        )
