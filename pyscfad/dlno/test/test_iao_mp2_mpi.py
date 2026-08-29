import warnings

from mpi4py import MPI
import numpy as np
import pytest

from pyscfad import config_update, gto, scf
from pyscfad.dlno import _restart as restart_module
from pyscfad.dlno import iao_mp2_mpi as iao_mp2_mpi_module
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


def test_comm_self_matches_serial_energy_and_gradient_to_roundoff():
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

    # These are two independent SCF convergences.  BLAS reduction order can
    # move energies/gradients by a few last bits.  The discrete correlation
    # decomposition below must still be structurally identical.
    np.testing.assert_allclose(
        mpi_energy, serial_energy, atol=5e-13, rtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(mpi_bar.coords),
        np.asarray(serial_bar.coords),
        atol=5e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        [mpi_details.e_corr, mpi_details.e_strong, mpi_details.e_weak],
        [
            serial_details.e_corr,
            serial_details.e_strong,
            serial_details.e_weak,
        ],
        atol=5e-13,
        rtol=0.0,
    )
    assert mpi_details.n_fragments == serial_details.n_fragments
    assert mpi_details.n_strong_pairs == serial_details.n_strong_pairs
    assert mpi_details.n_weak_pairs == serial_details.n_weak_pairs
    np.testing.assert_allclose(
        [term.energy for term in mpi_details.terms],
        [term.energy for term in serial_details.terms],
        atol=5e-13,
        rtol=0.0,
    )
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


def test_comm_self_restart_after_correlation_and_from_pre_scf(
    tmp_path, monkeypatch
):
    """The MPI wrapper resumes both common-closed and pre-SCF records."""

    mol = _water()
    kwargs = dict(
        build_mf=_build_mf,
        thresholds=_full_domain_thresholds(),
        pair_energy_model="all",
        force_full_domains=True,
        comm=MPI.COMM_SELF,
        return_details=True,
    )
    with (
        config_update("pyscfad_moleintor_opt", True),
        config_update("pyscfad_scf_implicit_diff", True),
        config_update("pyscfad_scf_first_order_custom", False),
    ):
        reference_energy, reference_bar, reference_details = (
            MPIIAOFragmentMP2.value_and_grad(mol, **kwargs)
        )

        class InjectedStop(RuntimeError):
            pass

        def stop_after_correlation(stage, key, path):
            del key, path
            if stage == "mpi_correlation_closed":
                raise InjectedStop("durable MPI correlation reached")

        checkpoint_dir = tmp_path / "mpi-mp2-restart"
        monkeypatch.setattr(
            restart_module,
            "_CHECKPOINT_EVENT_HOOK",
            stop_after_correlation,
        )
        with pytest.raises(RuntimeError, match="durable MPI correlation"):
            MPIIAOFragmentMP2.value_and_grad(
                mol, checkpoint_dir=checkpoint_dir, **kwargs
            )

        messages = []
        monkeypatch.setattr(restart_module, "_CHECKPOINT_EVENT_HOOK", None)
        resumed_energy, resumed_bar, resumed_details = (
            MPIIAOFragmentMP2.value_and_grad(
                mol,
                checkpoint_dir=checkpoint_dir,
                resume=True,
                progress=messages.append,
                **kwargs,
            )
        )
        assert any(
            "loaded completed common-closed correlation cotangent" in line
            for line in messages
        )
        np.testing.assert_allclose(
            resumed_energy, reference_energy, atol=2e-11, rtol=0.0
        )
        np.testing.assert_allclose(
            np.asarray(resumed_bar.coords),
            np.asarray(reference_bar.coords),
            atol=2e-10,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            resumed_details.e_corr,
            reference_details.e_corr,
            atol=2e-11,
            rtol=0.0,
        )

        def forbidden_correlation(*_args, **_kwargs):
            raise AssertionError("pre-SCF restart repeated MPI correlation")

        monkeypatch.setattr(
            iao_mp2_mpi_module,
            "correlation_value_and_grad",
            forbidden_correlation,
        )
        messages.clear()
        final_energy, final_bar, final_details = (
            MPIIAOFragmentMP2.value_and_grad(
                mol,
                checkpoint_dir=checkpoint_dir,
                resume=True,
                progress=messages.append,
                **kwargs,
            )
        )
        assert any("loaded pre-SCF total cotangent" in line
                   for line in messages)
        np.testing.assert_allclose(final_energy, resumed_energy, atol=0.0)
        np.testing.assert_allclose(
            np.asarray(final_bar.coords),
            np.asarray(resumed_bar.coords),
            atol=2e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            final_details.e_corr, resumed_details.e_corr, atol=0.0
        )
