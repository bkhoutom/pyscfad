"""Numerical tests for collective molecular out-of-core DF construction."""

from types import SimpleNamespace

import h5py
from mpi4py import MPI
import numpy as np
from pyscf import df as pyscf_df
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from pyscfad import gto
from pyscfad.df import mpi_outcore
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


def _pool_threads(user_api):
    return {pool["filepath"]: pool["num_threads"]
            for pool in threadpool_info()
            if pool["user_api"] == user_api
            and pool.get("threading_layer") != "disabled"}


@pytest.mark.parametrize("positive_definite", [True, False])
def test_metric_cholesky_uses_scoped_blas_budget(monkeypatch, positive_definite):
    """Dense factoring uses its budget; integral work and cleanup retain theirs."""
    monkeypatch.setenv("PYSCFAD_DENSE_BLAS_THREADS", "2")
    metric = np.array([[4., 2.], [2., 10.]])
    if not positive_definite:
        metric = np.array([[1., 2.], [2., 1.]])
    real_cholesky = mpi_outcore.scipy.linalg.cholesky
    observed = {}

    with threadpool_limits(limits=1, user_api="blas"):
        baseline_openmp = _pool_threads("openmp")

        def intor(name, hermi):
            assert name == "int2c2e" and hermi == 1
            observed["integral"] = _pool_threads("blas")
            assert observed["integral"] and set(observed["integral"].values()) == {1}
            assert _pool_threads("openmp") == baseline_openmp
            return metric

        def cholesky(*args, **kwargs):
            observed["dense"] = _pool_threads("blas")
            assert observed["dense"] and set(observed["dense"].values()) == {2}
            assert _pool_threads("openmp") == baseline_openmp
            return real_cholesky(*args, **kwargs)

        monkeypatch.setattr(mpi_outcore.scipy.linalg, "cholesky", cholesky)
        auxmol = SimpleNamespace(intor=intor)
        if positive_definite:
            low = mpi_outcore._metric_cholesky(auxmol, "int2c2e")
            np.testing.assert_allclose(low, [[2., 0.], [1., 3.]],
                                       rtol=1e-14, atol=1e-14)
            assert low.flags.c_contiguous and low.dtype == np.float64
        else:
            with pytest.raises(RuntimeError, match="not full-rank positive definite"):
                mpi_outcore._metric_cholesky(auxmol, "int2c2e")
        observed["restored"] = _pool_threads("blas")
        assert observed["restored"] and set(observed["restored"].values()) == {1}
        assert _pool_threads("openmp") == baseline_openmp
        print("metric BLAS budgets:", observed)


@pytest.mark.parametrize("fortran_three_dimensional", [False, True])
def test_transform_int3c_uses_scoped_blas_budget(monkeypatch,
                                               fortran_three_dimensional):
    """Whitening alone uses dense BLAS threads, preserving both input layouts."""
    monkeypatch.setenv("PYSCFAD_DENSE_BLAS_THREADS", "2")
    low = np.array([[2., 0.], [1., 3.]])
    raw = np.array([[1., 2.], [3., 4.], [5., 6.]])
    if fortran_three_dimensional:
        raw = np.asfortranarray(raw.reshape(1, 3, 2))
    real_solve = mpi_outcore.scipy.linalg.solve_triangular
    real_transpose = mpi_outcore.lib.transpose
    observed = {}

    with threadpool_limits(limits=1, user_api="blas"):
        baseline_openmp = _pool_threads("openmp")

        def transpose(*args, **kwargs):
            observed["transpose"] = _pool_threads("blas")
            assert observed["transpose"] and set(observed["transpose"].values()) == {1}
            return real_transpose(*args, **kwargs)

        def solve(*args, **kwargs):
            observed["dense"] = _pool_threads("blas")
            assert observed["dense"] and set(observed["dense"].values()) == {2}
            assert _pool_threads("openmp") == baseline_openmp
            return real_solve(*args, **kwargs)

        monkeypatch.setattr(mpi_outcore.lib, "transpose", transpose)
        monkeypatch.setattr(mpi_outcore.scipy.linalg, "solve_triangular", solve)
        result = mpi_outcore._transform_int3c(raw, low, 2)
        np.testing.assert_allclose(result, [[.5, 1.5, 2.5], [.5, 5/6, 7/6]],
                                   rtol=1e-14, atol=1e-14)
        assert result.flags.c_contiguous and result.dtype == np.float64
        observed["restored"] = _pool_threads("blas")
        assert observed["restored"] and set(observed["restored"].values()) == {1}
        assert _pool_threads("openmp") == baseline_openmp
        print("whitening BLAS budgets:", observed)
