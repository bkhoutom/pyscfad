"""Focused numerical tests for the molecular MPI DF-J/K contractions."""

import jax
from mpi4py import MPI
import numpy as np

from pyscfad import config_update, df, gto
from pyscfad._src import implicit_diff as implicit_diff_impl
from pyscfad.df import _df_jk_opt
from pyscfad.df.mpi_df_jk import (
    MPIDFJKExecutor,
    local_density_vjp,
    local_jk,
)


def _water_df():
    mol = gto.Mole(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
        verbose=0,
        max_memory=1000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    with config_update("pyscfad_moleintor_opt", True):
        dfobj = df.DF(mol, auxbasis="weigend")
        dfobj.build()
    return dfobj


def _symmetric_random(rng, nao):
    array = rng.normal(size=(nao, nao))
    return np.asarray(array + array.T)


def test_comm_self_forward_jk_matches_serial():
    """The MPI custom-VJP primal preserves the one-rank DF-J/K result."""
    rng = np.random.default_rng(812)
    dfobj = _water_df()
    dm = _symmetric_random(rng, dfobj.mol.nao)

    with config_update("pyscfad_moleintor_opt", True):
        vj_reference, vk_reference = _df_jk_opt.get_jk(dfobj, dm)

    executor = MPIDFJKExecutor(MPI.COMM_SELF)
    try:
        with executor.activate():
            vj_mpi, vk_mpi = dfobj.get_jk(dm)
    finally:
        executor.stop_workers()

    np.testing.assert_allclose(vj_mpi, vj_reference, atol=2e-12, rtol=2e-13)
    np.testing.assert_allclose(vk_mpi, vk_reference, atol=2e-12, rtol=2e-13)


def test_partitioned_density_transpose_satisfies_adjoint_identity():
    """Summed auxiliary-row VJPs are the transpose of summed J/K builds."""
    rng = np.random.default_rng(913)
    dfobj = _water_df()
    nao = dfobj.mol.nao
    dm = _symmetric_random(rng, nao)
    dm_direction = _symmetric_random(rng, nao)
    vj_bar = _symmetric_random(rng, nao)
    vk_bar = _symmetric_random(rng, nao)

    # Simulate more partitions than the COMM_SELF test can exercise.  These
    # helpers make no MPI calls, so the sum is exactly the quantity Reduce
    # produces in the real worker service.
    partition_count = 3
    vj_direction = np.zeros_like(dm)
    vk_direction = np.zeros_like(dm)
    dm_bar = np.zeros_like(dm)
    for rank in range(partition_count):
        local_vj, local_vk = local_jk(
            dfobj,
            dm_direction,
            rank=rank,
            size=partition_count,
        )
        vj_direction += local_vj
        vk_direction += local_vk
        dm_bar += local_density_vjp(
            dfobj,
            dm.shape,
            vj_bar,
            vk_bar,
            rank=rank,
            size=partition_count,
        )

    lhs = np.vdot(vj_bar, vj_direction) + np.vdot(vk_bar, vk_direction)
    rhs = np.vdot(dm_bar, dm_direction)
    np.testing.assert_allclose(rhs, lhs, atol=3e-10, rtol=3e-12)

    # Exercise the actual MPI custom-VJP boundary as well.  The implicit
    # response context selects the distributed density-only reverse path and
    # deliberately avoids the full CDERI-to-coordinate pullback.
    executor = MPIDFJKExecutor(MPI.COMM_SELF)
    try:
        _, pullback = jax.vjp(
            lambda dm_: executor.get_jk(dfobj, dm_), dm
        )
        with implicit_diff_impl._implicit_diff_solve_matvec():
            dm_bar_executor, = pullback((vj_bar, vk_bar))
    finally:
        executor.stop_workers()

    np.testing.assert_allclose(
        np.asarray(dm_bar_executor), dm_bar, atol=3e-10, rtol=3e-12
    )


def test_comm_self_full_coordinate_vjp_matches_serial(tmp_path):
    """The MPI full pullback preserves both DF coordinate cotangents."""

    rng = np.random.default_rng(1013)
    mol = gto.Mole(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
        verbose=0,
        max_memory=1000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    cderi_path = tmp_path / "water-cderi.h5"
    with config_update("pyscfad_moleintor_opt", True):
        dfobj = df.DF(mol, auxbasis="weigend", incore=False)
        dfobj._cderi_to_save = str(cderi_path)
        dfobj.build()

    dm = _symmetric_random(rng, mol.nao)
    vj_bar = _symmetric_random(rng, mol.nao)
    vk_bar = _symmetric_random(rng, mol.nao)
    with config_update("pyscfad_moleintor_opt", True):
        _, serial_pullback = jax.vjp(
            lambda dfobj_, dm_: _df_jk_opt.get_jk(dfobj_, dm_),
            dfobj,
            dm,
        )
        serial_df_bar, serial_dm_bar = serial_pullback((vj_bar, vk_bar))

        executor = MPIDFJKExecutor(MPI.COMM_SELF)
        try:
            _, mpi_pullback = jax.vjp(
                lambda dfobj_, dm_: executor.get_jk(dfobj_, dm_),
                dfobj,
                dm,
            )
            mpi_df_bar, mpi_dm_bar = mpi_pullback((vj_bar, vk_bar))
        finally:
            executor.stop_workers()

    np.testing.assert_allclose(
        np.asarray(mpi_dm_bar),
        np.asarray(serial_dm_bar),
        atol=3e-10,
        rtol=3e-12,
    )
    np.testing.assert_allclose(
        np.asarray(mpi_df_bar.mol.coords),
        np.asarray(serial_df_bar.mol.coords),
        atol=2e-9,
        rtol=2e-10,
    )
    np.testing.assert_allclose(
        np.asarray(mpi_df_bar.auxmol.coords),
        np.asarray(serial_df_bar.auxmol.coords),
        atol=2e-9,
        rtol=2e-10,
    )
