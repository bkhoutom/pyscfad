"""Resource-profile coverage for the MPI DLNO-MP2 gradient driver."""

from mpi4py import MPI

from pyscfad import config_update, gto, scf
from pyscfad.dlno import mp2_mpi
from pyscfad.dlno.domain import DLNOThresholds


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


def _build_mf(mol, **_unused):
    mf = scf.RHF(mol).density_fit(auxbasis="weigend")
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    return mf


def test_comm_self_resource_profile_covers_major_gradient_phases(monkeypatch):
    phases = []

    class ResourceRecorder:
        @staticmethod
        def start():
            return object()

        @staticmethod
        def finish(phase, before, **details):
            assert before is not None
            phases.append((phase, details))

    monkeypatch.setattr(
        mp2_mpi, "resource_profile", ResourceRecorder, raising=False
    )
    thresholds = DLNOThresholds(
        pao_norm=1e-10,
        domain_pao=0.0,
        ed_pao=0.0,
        occupied_weight=1e-12,
    )
    with (
        config_update("pyscfad_moleintor_opt", True),
        config_update("pyscfad_scf_implicit_diff", True),
        config_update("pyscfad_scf_first_order_custom", False),
    ):
        mp2_mpi.DLNOMP2.value_and_grad(
            _water(),
            build_mf=_build_mf,
            thresholds=thresholds,
            pair_energy_model="all",
            force_full_domains=True,
            comm=MPI.COMM_SELF,
        )

    phase_names = {phase for phase, _ in phases}
    assert {
        "dlno.mp2_mpi.scf_forward",
        "dlno.mp2_mpi.topology",
        "dlno.mp2_mpi.correlation.common_forward",
        "dlno.mp2_mpi.correlation.frame_build",
        "dlno.mp2_mpi.correlation.term_forward",
        "dlno.mp2_mpi.correlation.term_reverse",
        "dlno.mp2_mpi.correlation.frame_replay",
        "dlno.mp2_mpi.correlation.common_reverse",
        "dlno.mp2_mpi.correlation.total",
        "dlno.mp2_mpi.scf_response",
    } <= phase_names
