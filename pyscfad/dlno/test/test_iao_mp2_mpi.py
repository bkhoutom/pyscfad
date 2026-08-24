import warnings

from mpi4py import MPI
import numpy as np

from pyscfad import config_update, gto, scf
from pyscfad.dlno.iao_mp2 import (
    IAOFragmentMP2 as SerialIAOFragmentMP2,
    IAOFragmentMP2Thresholds,
)
from pyscfad.dlno.iao_mp2_mpi import IAOFragmentMP2 as MPIIAOFragmentMP2


warnings.filterwarnings(
    "ignore",
    message=r"Function mol\.dumps drops attribute .* because it is not "
            r"JSON-serializable",
)


def _water():
    mol = gto.Mole(
        atom="""
        O  0.0000000000  0.0000000000  0.0000000000
        H  0.0000000000 -0.7570000000  0.5870000000
        H  0.0000000000  0.7570000000  0.5870000000
        """,
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
    mf = scf.RHF(mol).density_fit(auxbasis="weigend")
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    if mo_coeff_init is None:
        mf.kernel()
    else:
        # COMM_SELF never enters this branch, but keeping the skeleton
        # contract here makes this builder usable by an mpiexec test driver.
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
    )


def test_comm_self_matches_serial_energy_and_gradient_exactly():
    mol = _water()
    progress_messages = []
    kwargs = dict(
        build_mf=_build_mf,
        thresholds=_full_domain_thresholds(),
        pair_energy_model="all",
        force_full_domains=True,
    )
    with (
        config_update("pyscfad_moleintor_opt", True),
        config_update("pyscfad_scf_implicit_diff", True),
        config_update("pyscfad_scf_first_order_custom", False),
    ):
        serial_energy, serial_bar, serial_details = (
            SerialIAOFragmentMP2.value_and_grad(
                mol, return_details=True, **kwargs
            )
        )
        mpi_energy, mpi_bar, mpi_details = (
            MPIIAOFragmentMP2.value_and_grad(
                mol,
                comm=MPI.COMM_SELF,
                return_details=True,
                progress=progress_messages.append,
                **kwargs,
            )
        )

    np.testing.assert_allclose(mpi_energy, serial_energy, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray(mpi_bar.coords),
        np.asarray(serial_bar.coords),
        atol=0.0,
        rtol=0.0,
    )
    assert mpi_details.e_corr == serial_details.e_corr
    assert mpi_details.e_strong == serial_details.e_strong
    assert mpi_details.e_weak == serial_details.e_weak
    assert mpi_details.n_fragments == serial_details.n_fragments
    assert mpi_details.n_strong_pairs == serial_details.n_strong_pairs
    assert mpi_details.n_weak_pairs == serial_details.n_weak_pairs
    assert [term.energy for term in mpi_details.terms] == [
        term.energy for term in serial_details.terms
    ]
    assert all(
        term.worker_rank == 0 for term in mpi_details.terms
    )
    assert mpi_details.timing.total_seconds > 0.0
    assert all(line.startswith("[IAO-MP2] ") for line in progress_messages)
    assert any("DF-RHF SCF and VJP setup: done" in line
               for line in progress_messages)
    assert any("fixed IAO fragment topology: done" in line
               for line in progress_messages)
    assert any("term 1/" in line and "E=" in line
               for line in progress_messages)
    assert any("implicit SCF response: done" in line
               for line in progress_messages)
