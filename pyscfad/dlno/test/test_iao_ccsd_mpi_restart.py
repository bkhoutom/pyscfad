"""Opt-in true-MPI restart regression for IAO-DLNO-CCSD(T).

Run explicitly with::

    pytest -q pyscfad/dlno/test/test_iao_ccsd_mpi_restart.py \
        -k true_mpiexec_restart_high_cost

The normal test selection excludes ``*_high_cost`` tests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


_DRIVER = Path(__file__).with_name("mpi_ccsd_restart_driver.py")


def _mpi_launcher():
    configured = os.environ.get("MPIEXEC")
    if configured:
        return configured
    return shutil.which("mpiexec") or shutil.which("mpirun")


def _launch(launcher, mode, checkpoint_dir, output, *, expect_success):
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
    command = [
        launcher,
        "-n",
        "2",
        sys.executable,
        str(_DRIVER),
        "--mode",
        mode,
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
        check=False,
    )
    if expect_success and completed.returncode != 0:
        pytest.fail(
            f"MPI driver mode {mode!r} failed with "
            f"code {completed.returncode}:\n{completed.stdout}"
        )
    if not expect_success and completed.returncode == 0:
        pytest.fail(
            f"MPI driver mode {mode!r} did not stop as requested:\n"
            f"{completed.stdout}"
        )
    return completed


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_true_mpiexec_restart_high_cost(tmp_path):
    """A fresh two-rank job resumes rank progress and then pre-SCF state."""

    launcher = _mpi_launcher()
    if launcher is None:
        pytest.skip("mpiexec/mpirun is unavailable")

    checkpoint_dir = tmp_path / "restart"
    reference_path = tmp_path / "reference.json"
    interrupted_path = tmp_path / "interrupted.json"
    resumed_path = tmp_path / "resumed.json"
    pre_scf_path = tmp_path / "pre-scf.json"

    _launch(
        launcher,
        "reference",
        checkpoint_dir,
        reference_path,
        expect_success=True,
    )
    _launch(
        launcher,
        "interrupt-progress",
        checkpoint_dir,
        interrupted_path,
        expect_success=False,
    )
    marker = interrupted_path.with_suffix(".interrupted")
    assert marker.is_file()
    assert "stage=mpi_cc_progress" in marker.read_text(encoding="utf-8")
    progress_dir = checkpoint_dir / "records" / "mpi_cc_progress"
    assert (progress_dir / "rank-000000.h5").is_file()

    _launch(
        launcher,
        "resume",
        checkpoint_dir,
        resumed_path,
        expect_success=True,
    )
    _launch(
        launcher,
        "pre-scf-forbid",
        checkpoint_dir,
        pre_scf_path,
        expect_success=True,
    )

    reference = _read(reference_path)
    resumed = _read(resumed_path)
    pre_scf = _read(pre_scf_path)
    assert reference["size"] == resumed["size"] == pre_scf["size"] == 2
    assert reference["fragment_owners"] == [[0, 0], [1, 1], [2, 0]]
    assert abs(reference["e_ccsd_t"]) > 1e-12
    np.testing.assert_allclose(
        resumed["energy"], reference["energy"], atol=2e-10, rtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(resumed["gradient"]),
        np.asarray(reference["gradient"]),
        atol=2e-8,
        rtol=0.0,
    )
    # Both ranks durably finished the first round before the injected rank-0
    # failure was reported.  Rank 0 must continue with fragment 2, while rank
    # 1 correctly skips its already committed fragment.
    assert resumed["fragment_calls"] == [1, 0]
    np.testing.assert_allclose(
        pre_scf["energy"], resumed["energy"], atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(pre_scf["gradient"]),
        np.asarray(resumed["gradient"]),
        atol=2e-12,
        rtol=0.0,
    )
    assert pre_scf["fragment_calls"] == [0, 0]
