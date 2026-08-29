"""MPI DF-build and J/K regression for the IAO-DLNO-MP2 gradient driver.

The ordinary pytest suite excludes the true multi-rank check by its
``*_high_cost`` name.  Run it explicitly with, for example::

    pytest -q pyscfad/dlno/test/test_iao_mp2_mpi_dfjk.py \
        -k mpiexec_np1_np2_high_cost

The MPI driver collectively builds a shared out-of-core CDERI file, compares
one and two ranks, and checks both against ordinary serial IAO-DLNO-MP2 and
canonical DF-MP2. Full domains make the local correlation energy equivalent
to canonical DF-MP2 for this small water calculation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import warnings

# Keep rank-count comparisons meaningful and avoid threaded BLAS
# oversubscription in the deliberately small MPI regression.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

import jax
import h5py
from mpi4py import MPI
import numpy as np
import pytest

from pyscfad import config_update, gto, scf
from pyscfad.df.mpi_df_jk import MPIDFJKExecutor
from pyscfad.df.mpi_outcore import build_cderi
from pyscfad.dlno.iao_mp2 import (
    IAOFragmentMP2 as SerialIAOFragmentMP2,
    IAOFragmentMP2Thresholds,
)
from pyscfad.dlno.iao_mp2_mpi import IAOFragmentMP2 as MPIIAOFragmentMP2
from pyscfad.mp import dfmp2


warnings.filterwarnings(
    "ignore",
    message=r"Function mol\.dumps drops attribute .* because it is not "
            r"JSON-serializable",
)


_CDERI_ENV = "PYSCFAD_MPI_DFJK_TEST_CDERI"


def _water():
    mol = gto.Mole(
        atom="""
        O  0.0000000000  0.0000000000  0.0000000000
        H  0.0000000000 -0.7570000000  0.5870000000
        H  0.0000000000  0.7570000000  0.5870000000
        """,
        unit="Angstrom",
        basis="sto-3g",
        verbose=0,
        max_memory=1000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def _build_mf(
    mol,
    *,
    mo_coeff_init=None,
    mo_energy_init=None,
    mo_occ_init=None,
    e_tot_init=None,
):
    """Run root SCF or construct the no-SCF worker DF skeleton."""
    mf = scf.RHF(mol).density_fit(auxbasis="weigend")
    cderi_path = os.environ.get(_CDERI_ENV)
    if cderi_path:
        mf.with_df.attach_outcore_cderi(cderi_path)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    if mo_coeff_init is None:
        mf.kernel()
    else:
        mf.mo_coeff = mo_coeff_init
        mf.mo_energy = mo_energy_init
        mf.mo_occ = mo_occ_init
        mf.e_tot = e_tot_init
        mf.converged = True
    return mf


def _full_domain_thresholds():
    return IAOFragmentMP2Thresholds(
        pao_norm=1e-10,
        domain_pao=0.0,
        ed_pao=0.0,
        occupied_weight=1e-12,
        pair_energy=0.0,
    )


@contextmanager
def _gradient_options():
    with (
        config_update("pyscfad_moleintor_opt", True),
        config_update("pyscfad_scf_implicit_diff", True),
        config_update("pyscfad_scf_first_order_custom", False),
    ):
        yield


def _serial_reference():
    with _gradient_options():
        energy, mol_bar = SerialIAOFragmentMP2.value_and_grad(
            _water(),
            build_mf=_build_mf,
            thresholds=_full_domain_thresholds(),
            pair_energy_model="all",
            force_full_domains=True,
        )
    return float(energy), np.asarray(mol_bar.coords)


def _canonical_reference():
    def canonical_total(mol):
        mf = _build_mf(mol)
        correlation, _ = dfmp2.MP2(mf).kernel(with_t2=False)
        return mf.e_tot + correlation

    with _gradient_options():
        energy, mol_bar = jax.value_and_grad(canonical_total)(_water())
    return float(energy), np.asarray(mol_bar.coords)


def _run_world_driver(output: Path):
    """Build DF, run distributed J/K plus MP2, and write root results."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    cderi_path = os.environ.get(_CDERI_ENV)
    if not cderi_path:
        raise RuntimeError(f"{_CDERI_ENV} must name the shared output file")
    mol = _water()
    cderi_result = build_cderi(
        mol,
        cderi_path,
        auxbasis="weigend",
        comm=comm,
        max_memory=0.01,
        min_blocks_per_rank=2,
    )
    _, local_cderi_hash = _read_cderi(cderi_result.path)
    cderi_hashes = comm.allgather(local_cderi_hash)
    operation_counts = {
        "forward": 0,
        "density_vjp": 0,
        "lowrank_vjp": 0,
        "coordinate_vjp_blocks": 0,
    }
    operation_norms = {
        "forward": 0.0,
        "density_vjp": 0.0,
        "coordinate_vjp": 0.0,
    }
    original_forward = MPIDFJKExecutor._execute_forward
    original_density_vjp = MPIDFJKExecutor._execute_density_vjp
    original_coordinate_vjp_block = (
        MPIDFJKExecutor._execute_coordinate_vjp_block
    )

    def counted_forward(self, dfobj, payload):
        result = original_forward(self, dfobj, payload)
        operation_counts["forward"] += 1
        operation_norms["forward"] += float(np.linalg.norm(np.stack(result)))
        return result

    def counted_density_vjp(self, dfobj, payload):
        result = original_density_vjp(self, dfobj, payload)
        operation_counts["density_vjp"] += 1
        operation_counts["lowrank_vjp"] += int(
            payload["lowrank_factors"] is not None
        )
        operation_norms["density_vjp"] += float(np.linalg.norm(result))
        return result

    def counted_coordinate_vjp_block(
        self, dfobj, payload, shls_slice, ints_bar
    ):
        result = original_coordinate_vjp_block(
            self, dfobj, payload, shls_slice, ints_bar
        )
        operation_counts["coordinate_vjp_blocks"] += 1
        operation_norms["coordinate_vjp"] += float(
            np.linalg.norm(np.asarray(result[0].coords))
            + np.linalg.norm(np.asarray(result[1].coords))
        )
        return result

    MPIDFJKExecutor._execute_forward = counted_forward
    MPIDFJKExecutor._execute_density_vjp = counted_density_vjp
    MPIDFJKExecutor._execute_coordinate_vjp_block = (
        counted_coordinate_vjp_block
    )
    try:
        with _gradient_options():
            energy, mol_bar, details = MPIIAOFragmentMP2.value_and_grad(
                mol,
                build_mf=_build_mf,
                thresholds=_full_domain_thresholds(),
                pair_energy_model="all",
                force_full_domains=True,
                parallel_scf_jk=True,
                comm=comm,
                root=0,
                return_details=True,
                progress=False,
            )
    finally:
        MPIDFJKExecutor._execute_forward = original_forward
        MPIDFJKExecutor._execute_density_vjp = original_density_vjp
        MPIDFJKExecutor._execute_coordinate_vjp_block = (
            original_coordinate_vjp_block
        )
    jax.block_until_ready(energy)

    energies = comm.allgather(float(energy))
    gradient_owners = comm.allgather(mol_bar is not None)
    all_operation_counts = comm.allgather(operation_counts)
    all_operation_norms = comm.allgather(operation_norms)
    root_error = None
    if rank == 0:
        try:
            if gradient_owners != [True] + [False] * (size - 1):
                raise AssertionError(
                    "gradient ownership must be root-only; got "
                    f"{gradient_owners}"
                )
            np.testing.assert_allclose(
                energies, energies[0], atol=0.0, rtol=0.0
            )
            gradient = np.asarray(mol_bar.coords)
            if not np.all(np.isfinite(gradient)):
                raise AssertionError("root gradient contains a non-finite value")
            if details.n_fragments != 1:
                raise AssertionError(
                    "water/full-domain regression should contain one IAO "
                    f"fragment, got {details.n_fragments}"
                )
            if len(set(cderi_hashes)) != 1:
                raise AssertionError(
                    f"MPI ranks observed different CDERI files: {cderi_hashes}"
                )
            if size > 1:
                for manifest in cderi_result.manifests:
                    if manifest.pair_columns <= 0 or manifest.block_count <= 0:
                        raise AssertionError(
                            "MPI rank did not generate a nonempty CDERI shard: "
                            f"{manifest}"
                        )
                for worker_rank, counts in enumerate(all_operation_counts):
                    if counts["forward"] <= 0 or counts["density_vjp"] <= 0:
                        raise AssertionError(
                            "MPI rank did not execute both primal and reverse "
                            f"DF-J/K contractions: rank={worker_rank}, "
                            f"counts={counts}"
                        )
                    if counts["lowrank_vjp"] <= 0:
                        raise AssertionError(
                            "implicit low-rank exchange transpose was not used "
                            f"on rank {worker_rank}: counts={counts}"
                        )
                    if (
                        worker_rank != 0
                        and counts["coordinate_vjp_blocks"] <= 0
                    ):
                        raise AssertionError(
                            "MPI worker did not execute a DF coordinate-VJP "
                            f"shell block: rank={worker_rank}, counts={counts}"
                        )
                for worker_rank, norms in enumerate(all_operation_norms):
                    if norms["forward"] == 0.0 or norms["density_vjp"] == 0.0:
                        raise AssertionError(
                            "MPI rank produced only zero DF-J/K contributions: "
                            f"rank={worker_rank}, norms={norms}"
                        )
                    if (
                        worker_rank != 0
                        and norms["coordinate_vjp"] == 0.0
                    ):
                        raise AssertionError(
                            "MPI worker produced only zero DF coordinate-VJP "
                            f"contributions: rank={worker_rank}, norms={norms}"
                        )
            output.write_text(
                json.dumps(
                    {
                        "size": size,
                        "energy": energies[0],
                        "gradient": gradient.tolist(),
                        "operation_counts": all_operation_counts,
                        "cderi": {
                            "path": cderi_result.path,
                            "naux": cderi_result.naux,
                            "nao_pair": cderi_result.nao_pair,
                            "nblocks": cderi_result.nblocks,
                            "rank_hashes": cderi_hashes,
                            "manifests": [
                                asdict(item)
                                for item in cderi_result.manifests
                            ],
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception:
            root_error = traceback.format_exc()
    root_error = comm.bcast(root_error, root=0)
    if root_error is not None:
        raise AssertionError("root MPI-driver validation failed:\n" + root_error)


def _mpi_launcher():
    launcher = os.environ.get("MPIEXEC")
    if launcher:
        return launcher
    return shutil.which("mpiexec") or shutil.which("mpirun")


def _build_shared_cderi(path):
    builder = scf.RHF(_water()).density_fit(auxbasis="weigend").with_df
    builder._cderi_to_save = str(path)
    builder.build()


def _read_cderi(path):
    with h5py.File(path, "r") as handle:
        array = np.asarray(handle["j3c"])
    return array, hashlib.sha256(array.tobytes()).hexdigest()


def _launch_driver(launcher, size, output, cderi_path):
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
    env[_CDERI_ENV] = str(cderi_path)
    command = [
        launcher,
        "-n",
        str(size),
        sys.executable,
        str(Path(__file__).resolve()),
        "--mpi-driver",
        str(output),
    ]
    subprocess.run(command, env=env, check=True, timeout=900)


def test_mpiexec_np1_np2_high_cost(tmp_path):
    """Two-rank SCF-J/K forward/reverse must reproduce serial MP2 gradient."""
    launcher = _mpi_launcher()
    if launcher is None:
        pytest.skip("mpiexec/mpirun is unavailable")

    np1_path = tmp_path / "np1.json"
    np2_path = tmp_path / "np2.json"
    np1_cderi_path = tmp_path / "water_np1_cderi.h5"
    np2_cderi_path = tmp_path / "water_np2_cderi.h5"
    serial_cderi_path = tmp_path / "water_serial_cderi.h5"
    _launch_driver(launcher, 1, np1_path, np1_cderi_path)
    _launch_driver(launcher, 2, np2_path, np2_cderi_path)
    _build_shared_cderi(serial_cderi_path)

    np1 = json.loads(np1_path.read_text(encoding="utf-8"))
    np2 = json.loads(np2_path.read_text(encoding="utf-8"))
    assert np1["size"] == 1
    assert np2["size"] == 2
    assert len(np2["cderi"]["manifests"]) == 2
    assert all(
        item["pair_columns"] > 0 and item["block_count"] > 0
        for item in np2["cderi"]["manifests"]
    )
    np1_cderi, np1_hash = _read_cderi(np1_cderi_path)
    np2_cderi, np2_hash = _read_cderi(np2_cderi_path)
    serial_cderi, _ = _read_cderi(serial_cderi_path)
    assert np1["cderi"]["rank_hashes"] == [np1_hash]
    assert np2["cderi"]["rank_hashes"] == [np2_hash, np2_hash]
    np.testing.assert_allclose(
        np2_cderi, np1_cderi, atol=2e-11, rtol=2e-12
    )
    np.testing.assert_allclose(
        np2_cderi, serial_cderi, atol=2e-11, rtol=2e-12
    )
    assert not tuple(tmp_path.glob(".*cderi.h5.mpi-*"))
    np.testing.assert_allclose(
        np2["energy"], np1["energy"], atol=2e-10, rtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(np2["gradient"]),
        np.asarray(np1["gradient"]),
        atol=2e-7,
        rtol=0.0,
    )

    serial_energy, serial_gradient = _serial_reference()
    np.testing.assert_allclose(
        np2["energy"], serial_energy, atol=2e-9, rtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(np2["gradient"]),
        serial_gradient,
        atol=3e-7,
        rtol=0.0,
    )

    canonical_energy, canonical_gradient = _canonical_reference()
    np.testing.assert_allclose(
        np2["energy"], canonical_energy, atol=2e-9, rtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(np2["gradient"]),
        canonical_gradient,
        atol=3e-7,
        rtol=0.0,
    )


def _main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpi-driver", type=Path, required=True)
    args = parser.parse_args(argv)
    _run_world_driver(args.mpi_driver)


if __name__ == "__main__":
    _main()
