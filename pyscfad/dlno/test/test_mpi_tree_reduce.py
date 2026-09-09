"""Focused true-MPI regression for cotangent-tree reduction."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

import jax
from mpi4py import MPI
import numpy as np
import pytest

from pyscfad.dlno import iao_mp2_mpi


class _TypedCollectiveOnly:
    """Delegate to MPI while rejecting pickle-based payload gathers."""

    def __init__(self, comm):
        self._comm = comm
        self.reduce_calls = 0

    def gather(self, *args, **kwargs):
        raise AssertionError("cotangent payload used object gather")

    def Reduce(self, *args, **kwargs):
        self.reduce_calls += 1
        return self._comm.Reduce(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._comm, name)


def _run_world_driver():
    comm = _TypedCollectiveOnly(MPI.COMM_WORLD)
    rank = comm.Get_rank()
    size = comm.Get_size()
    assert size == 2

    # Force the dense and complex leaves through multiple small chunks.
    iao_mp2_mpi._TREE_REDUCE_CHUNK_BYTES = 16
    tree = {
        "dense": np.arange(12, dtype=np.float64).reshape(3, 4) + rank,
        "scalar": np.asarray(rank + 0.25, dtype=np.float64),
        "complex": np.asarray([1.0 + 2.0j, -3.0 + 0.5j]) * (rank + 1),
        "empty": np.empty((0,), dtype=np.float64),
        "none": None,
        "float0": np.zeros((3,), dtype=jax.dtypes.float0),
    }

    result = iao_mp2_mpi._tree_sum_to_root(comm, tree, root=0)
    assert comm.reduce_calls > 3
    if rank != 0:
        assert result is None
        return

    rank_sum = size * (size - 1) // 2
    np.testing.assert_array_equal(
        result["dense"],
        size * np.arange(12, dtype=np.float64).reshape(3, 4) + rank_sum,
    )
    np.testing.assert_array_equal(
        result["scalar"], np.asarray(rank_sum + 0.25 * size)
    )
    np.testing.assert_array_equal(
        result["complex"],
        np.asarray([1.0 + 2.0j, -3.0 + 0.5j])
        * (size * (size + 1) // 2),
    )
    assert result["empty"].shape == (0,)
    assert result["none"] is None
    assert result["float0"].dtype == jax.dtypes.float0


def _mpi_launcher():
    configured = os.environ.get("MPIEXEC")
    if configured:
        return configured
    return shutil.which("mpiexec") or shutil.which("mpirun")


def test_tree_sum_to_root_uses_chunked_typed_reduce_high_cost():
    launcher = _mpi_launcher()
    if launcher is None:
        pytest.skip("mpiexec/mpirun is unavailable")

    env = os.environ.copy()
    command = [
        launcher,
        "-n",
        "2",
        sys.executable,
        str(Path(__file__).resolve()),
        "--mpi-driver",
    ]
    completed = subprocess.run(
        command,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpi-driver", action="store_true", required=True)
    parser.parse_args(argv)
    _run_world_driver()


if __name__ == "__main__":
    _main()
