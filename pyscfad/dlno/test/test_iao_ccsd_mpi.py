"""Focused integration tests for fragment-parallel IAO-DLNO-CCSD.

The ordinary pytest test exercises the ``COMM_SELF`` compatibility contract.
The high-cost test launches this file as a small MPI driver with one and two
ranks, so the two-fragment water dimer actually crosses the MPI boundary.

Run the true multi-rank check explicitly with, for example::

    pytest -q pyscfad/dlno/test/test_iao_ccsd_mpi.py \
        -k mpiexec_np1_np2_high_cost

The repository-wide pytest configuration excludes ``*_high_cost`` tests by
default.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import warnings

# Establish conservative thread defaults before importing NumPy/JAX/PySCF.
# Unbounded OpenMP discovery on large login nodes can otherwise oversubscribe
# PySCF's BLAS kernels even for this deliberately small integration test.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

import jax
from mpi4py import MPI
import numpy as np
import pytest

from pyscfad import config_update, gto, scf
from pyscfad.dlno.ccsd import DLNOCCSD as SerialDLNOCCSD
from pyscfad.dlno.ccsd_mpi import DLNOCCSD as MPIDLNOCCSD
from pyscfad.dlno.iao_mp2 import IAOFragmentMP2Thresholds


warnings.filterwarnings(
    "ignore",
    message=r"Function mol\.dumps drops attribute .* because it is not "
            r"JSON-serializable",
)


def _water(*, dimer: bool, separated: bool = True):
    first = """
    O  0.0000000000  0.0000000000  0.0000000000
    H  0.0000000000 -0.7570000000  0.5870000000
    H  0.0000000000  0.7570000000  0.5870000000
    """
    if not dimer:
        atom = first
    elif separated:
        atom = first + """
        O  0.0000000000  0.0000000000  8.0000000000
        H  0.0000000000 -0.7570000000  8.5870000000
        H  0.0000000000  0.7570000000  8.5870000000
        """
    else:
        atom = """
        O  -1.485163346097 -0.114724564047  0.000000000000
        H  -1.868415346097  0.762298435953  0.000000000000
        H  -0.533833346097  0.040507435953  0.000000000000
        O   1.416468653903  0.111264435953  0.000000000000
        H   1.746241653903 -0.373945564047 -0.758561000000
        H   1.746241653903 -0.373945564047  0.758561000000
        """
    mol = gto.Mole(
        atom=atom,
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
    """Build root SCF or reconstruct the worker SCF skeleton."""
    mf = scf.RHF(mol).density_fit(auxbasis="weigend")
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    if mo_coeff_init is None:
        mf.kernel()
    else:
        # Fragment CCSD reads the three-index DF tensor in both its forward
        # and reverse passes. Build a real rank-local in-core source here;
        # an empty MPI-MP2 placeholder is insufficient for this test.
        mf.with_df.build()
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
        config_update("pyscfad_ccsd_implicit_diff", True),
    ):
        yield


def test_comm_self_matches_serial_value_and_grad():
    """The MPI wrapper must preserve the serial one-rank result."""
    mol = _water(dimer=False)
    kwargs = dict(
        build_mf=_build_mf,
        thresholds=_full_domain_thresholds(),
        pair_energy_model="all",
        force_full_domains=True,
        thresh_occ=1.0,
        thresh_vir=1.0,
        ccsd_t=False,
    )
    with _gradient_options():
        serial_energy, serial_bar = SerialDLNOCCSD.value_and_grad(
            mol, **kwargs
        )
        mpi_energy, mpi_bar, details = MPIDLNOCCSD.value_and_grad(
            mol,
            comm=MPI.COMM_SELF,
            root=0,
            return_details=True,
            progress=False,
            **kwargs,
        )

    np.testing.assert_allclose(
        np.asarray(mpi_energy), np.asarray(serial_energy), atol=1e-10, rtol=0.0
    )
    assert mpi_bar is not None
    assert details.nproc == 1
    assert details.e_total == float(mpi_energy)
    assert len(details.fragments) == 1
    assert details.fragments[0].fragment_index == 0
    assert details.fragments[0].worker_rank == 0
    assert details.fragments[0].lis_virtual == 0
    np.testing.assert_allclose(
        details.e_hf + details.e_iao_mp2 + details.e_ccsd
        + details.e_ccsd_t - details.e_mp2_lis,
        details.e_total,
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(mpi_bar.coords),
        np.asarray(serial_bar.coords),
        atol=1e-8,
        rtol=0.0,
    )


def _run_world_driver(output: Path):
    """Run one two-fragment calculation and write one root-owned JSON file."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    mol = _water(dimer=True, separated=True)
    kwargs = dict(
        build_mf=_build_mf,
        thresholds=IAOFragmentMP2Thresholds(pair_energy=1e-4),
        pair_energy_model="multipole",
        thresh_occ=1e-3,
        thresh_vir=1e-3,
        ccsd_t=False,
    )
    with _gradient_options():
        energy, mol_bar, details = MPIDLNOCCSD.value_and_grad(
            mol,
            comm=comm,
            root=0,
            return_details=True,
            progress=False,
            **kwargs,
        )
    jax.block_until_ready(energy)

    energies = comm.allgather(float(energy))
    has_gradient = comm.allgather(mol_bar is not None)
    detail_summaries = comm.allgather((
        details.nproc,
        details.e_total,
        tuple(
            (record.fragment_index, record.worker_rank)
            for record in details.fragments
        ),
    ))
    if rank == 0:
        if has_gradient != [True] + [False] * (size - 1):
            raise AssertionError(
                "gradient ownership must be root-only; got "
                f"{has_gradient}"
            )
        np.testing.assert_allclose(energies, energies[0], atol=0.0, rtol=0.0)
        assert all(summary == detail_summaries[0]
                   for summary in detail_summaries)
        assert details.nproc == size
        ownership = tuple(
            (record.fragment_index, record.worker_rank)
            for record in details.fragments
        )
        assert ownership == tuple(
            (fragment_index, fragment_index % size)
            for fragment_index in range(len(details.fragments))
        )
        np.testing.assert_allclose(
            details.e_hf + details.e_iao_mp2 + details.e_ccsd
            + details.e_ccsd_t - details.e_mp2_lis,
            details.e_total,
            atol=1e-12,
            rtol=0.0,
        )
        gradient = np.asarray(mol_bar.coords)
        if not np.all(np.isfinite(gradient)):
            raise AssertionError("root gradient contains a non-finite value")
        output.write_text(
            json.dumps(
                {
                    "size": size,
                    "energy": energies[0],
                    "gradient": gradient.tolist(),
                    "ownership": ownership,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    comm.Barrier()


def _mpi_launcher():
    launcher = os.environ.get("MPIEXEC")
    if launcher:
        return launcher
    return shutil.which("mpiexec") or shutil.which("mpirun")


def _launch_driver(launcher, size, output):
    env = os.environ.copy()
    # Keep rank-count comparisons meaningful and avoid oversubscribing a
    # developer workstation or CI worker with threaded BLAS inside each rank.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
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
    """True two-rank regression: distribute two fragments and match np=1."""
    launcher = _mpi_launcher()
    if launcher is None:
        pytest.skip("mpiexec/mpirun is unavailable")

    np1_path = tmp_path / "np1.json"
    np2_path = tmp_path / "np2.json"
    _launch_driver(launcher, 1, np1_path)
    _launch_driver(launcher, 2, np2_path)

    np1 = json.loads(np1_path.read_text(encoding="utf-8"))
    np2 = json.loads(np2_path.read_text(encoding="utf-8"))
    assert np1["size"] == 1
    assert np2["size"] == 2
    np.testing.assert_allclose(
        np2["energy"], np1["energy"], atol=1e-9, rtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(np2["gradient"]),
        np.asarray(np1["gradient"]),
        atol=1e-7,
        rtol=0.0,
    )


def _main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpi-driver", type=Path, required=True)
    args = parser.parse_args(argv)
    _run_world_driver(args.mpi_driver)


if __name__ == "__main__":
    _main()
