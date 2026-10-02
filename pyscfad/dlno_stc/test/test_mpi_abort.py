"""Real-MPI regression for the example's communicator-wide failure path."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from pyscfad.dlno_stc.exchange import write_input


def test_reference_solve_aborts_after_root_backend_failure(tmp_path):
    pytest.importorskip("mpi4py")
    mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        pytest.skip("mpiexec is unavailable")

    workdir = tmp_path / "run"
    request = workdir / "fragment_0000" / "input.h5"
    inputs = {
        "foo": np.array([[-1.0]], dtype=np.float64),
        "fvv": np.array([[1.0]], dtype=np.float64),
        "B": np.array([[[0.1]]], dtype=np.float64),
    }
    metadata = {
        "schema_version": 1, "method": "mp2",
        "energy_kind": "quadratic_test_v1", "fragment_id": 0,
        "target_index": 0, "basis_frame": "orthonormal_local",
        "B_axes": "Pia", "units": {"energy": "Eh", "length": "bohr"},
        "frozen": [0],
        "orbital_order": {"occupied": [0], "virtual": [0]},
        "auxiliary_order": "test-aux", "checkpoint_id": "test-checkpoint",
        "domain_options": {"full_support": True}, "code_revision": "test",
        "dirty_worktree": False,
    }
    write_input(request, inputs, metadata, controls={})

    # Both ranks enter the backend; root then fails while the worker would
    # otherwise wait for solve_request's final result-status broadcast.
    child = """
from pathlib import Path
import sys
from examples.dlno_stc import workflow

workdir = Path(sys.argv[1])

def failing_backend(inputs, metadata, controls, *, comm=None):
    (workdir / f'backend_rank_{comm.Get_rank()}').touch()
    comm.Barrier()
    if comm.Get_rank() == 0:
        raise RuntimeError('deliberate root backend failure')
    return None

workflow.reference_backend = failing_backend
workflow._reference_solve(workdir)
"""
    root = Path(__file__).resolve().parents[3]
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
    )
    completed = subprocess.run(
        [mpiexec, "-n", "2", sys.executable, "-c", child, str(workdir)],
        cwd=root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert (workdir / "backend_rank_0").exists(), completed.stderr
    assert (workdir / "backend_rank_1").exists(), completed.stderr
    assert completed.returncode != 0, completed.stdout
    assert not (workdir / "fragment_0000" / "result.h5").exists()
