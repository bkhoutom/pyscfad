"""Numerical tests for collective molecular out-of-core DF construction."""

import h5py
from mpi4py import MPI
import numpy as np
from pyscf import df as pyscf_df
import pytest

from pyscfad import gto
from pyscfad.df.mpi_outcore import build_cderi


def _displaced_water():
    mol = gto.Mole(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        unit="Angstrom",
        basis="sto-3g",
        verbose=0,
    )
    # Keep the default dynamic basis leaves to exercise eager conversion.
    mol.build()
    mol.coords = mol.coords.at[1, 2].add(0.031)
    return mol


def _serial_mol_at_dynamic_geometry(mol):
    concrete = mol.to_pyscf().copy()
    concrete.set_geom_(np.asarray(mol.atom_coords()), unit="Bohr")
    return concrete


def test_comm_self_cderi_matches_pyscf_outcore(tmp_path):
    """MPI output uses PySCF's Cholesky gauge at the current AD geometry."""

    mol = _displaced_water()
    mpi_path = tmp_path / "mpi_cderi.h5"
    serial_path = tmp_path / "serial_cderi.h5"

    result = build_cderi(
        mol,
        mpi_path,
        auxbasis="weigend",
        comm=MPI.COMM_SELF,
        max_memory=0.01,
    )
    pyscf_df.outcore.cholesky_eri(
        _serial_mol_at_dynamic_geometry(mol),
        str(serial_path),
        auxbasis="weigend",
        max_memory=0.01,
    )

    with h5py.File(mpi_path, "r") as mpi_file:
        mpi_cderi = np.asarray(mpi_file["j3c"])
        assert mpi_file.attrs["decomposition"] == "CD"
        assert bool(mpi_file.attrs["pyscfad_mpi_df_complete"])
    with h5py.File(serial_path, "r") as serial_file:
        serial_cderi = np.asarray(serial_file["j3c"])

    assert mpi_cderi.shape == (result.naux, result.nao_pair)
    assert result.naux == result.naux_raw
    assert result.nproc == 1
    assert result.manifests[0].pair_columns == result.nao_pair
    assert result.manifests[0].block_count == result.nblocks
    np.testing.assert_allclose(
        mpi_cderi, serial_cderi, atol=2e-11, rtol=2e-12
    )
    assert not tuple(tmp_path.glob(".mpi_cderi.h5.mpi-*"))

    original_bytes = mpi_path.read_bytes()
    with pytest.raises(RuntimeError, match="already exists"):
        build_cderi(
            mol,
            mpi_path,
            auxbasis="weigend",
            comm=MPI.COMM_SELF,
        )
    assert mpi_path.read_bytes() == original_bytes


def test_progress_callback_failure_is_nonfatal(tmp_path):
    """A root-only reporting failure must not interrupt the collective."""

    def broken_reporter(_message):
        raise LookupError("injected progress failure")

    with pytest.warns(RuntimeWarning, match="progress callback failed"):
        result = build_cderi(
            _displaced_water(),
            tmp_path / "reported_cderi.h5",
            auxbasis="weigend",
            comm=MPI.COMM_SELF,
            max_memory=0.01,
            progress=broken_reporter,
        )
    assert result.nao_pair > 0
