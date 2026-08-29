"""Two-rank IAO-DLNO-CCSD(T) restart integration-test driver.

This file is intentionally not collected as a pytest module.  The companion
``test_iao_ccsd_mpi_restart.py`` launches it under ``mpiexec -n 2`` several
times so a checkpoint produced by a failed MPI job is consumed by a fresh MPI
job, rather than by another call in the same Python process.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import traceback

# Establish conservative defaults before importing NumPy, JAX, or PySCF.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

import jax
from mpi4py import MPI
import numpy as np

from pyscfad import config_update, gto, scf
from pyscfad.dlno import _restart as restart_module
from pyscfad.dlno import ccsd_mpi as ccsd_mpi_module
from pyscfad.dlno.ccsd_mpi import DLNOCCSD
from pyscfad.dlno.iao_mp2 import IAOFragmentMP2Thresholds


def _water_trimer():
    """Return three separated, deliberately tiny STO-3G fragments."""

    mol = gto.Mole(
        atom="""
        O  0.0000000000  0.0000000000  0.0000000000
        H  0.0000000000 -0.7570000000  0.5870000000
        H  0.0000000000  0.7570000000  0.5870000000
        O  0.0000000000  0.0000000000  8.0000000000
        H  0.0000000000 -0.7570000000  8.5870000000
        H  0.0000000000  0.7570000000  8.5870000000
        O  0.0000000000  0.0000000000 16.0000000000
        H  0.0000000000 -0.7570000000 16.5870000000
        H  0.0000000000  0.7570000000 16.5870000000
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
    """Build root SCF or a worker skeleton with a real local DF tensor."""

    mf = scf.RHF(mol).density_fit(auxbasis="weigend")
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    if mo_coeff_init is None:
        mf.kernel()
    else:
        mf.with_df.build()
        mf.mo_coeff = mo_coeff_init
        mf.mo_energy = mo_energy_init
        mf.mo_occ = mo_occ_init
        mf.e_tot = e_tot_init
        mf.converged = True
    return mf


def _thresholds():
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


def _install_forbidden_pre_scf_work():
    """Make a pre-SCF restart fail if it enters any correlation stage."""

    def forbidden(*_args, **_kwargs):
        raise AssertionError(
            "pre-SCF restart unexpectedly entered common/fragment/MP2 work"
        )

    ccsd_mpi_module.rebuild_iao_mp2_common = forbidden
    ccsd_mpi_module._fragment_value_and_grad = forbidden
    ccsd_mpi_module._mp2_correlation_value_and_grad = forbidden


def _run(mode: str, checkpoint_dir: Path, output: Path):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 2:
        raise RuntimeError(f"restart integration driver requires 2 ranks, got {size}")

    fragment_calls = 0
    original_fragment = ccsd_mpi_module._fragment_value_and_grad

    def counted_fragment(*args, **kwargs):
        nonlocal fragment_calls
        fragment_calls += 1
        return original_fragment(*args, **kwargs)

    ccsd_mpi_module._fragment_value_and_grad = counted_fragment

    if mode == "interrupt-progress":
        fired = False

        def stop_after_rank_progress(stage, key, path):
            nonlocal fired
            del path
            if (
                stage == "mpi_cc_progress"
                and rank == 0
                and not fired
            ):
                fired = True
                output.with_suffix(".interrupted").write_text(
                    f"stage={stage} key={key} rank={rank}\n",
                    encoding="utf-8",
                )
                # save_record invokes this hook only after its atomic replace,
                # so the fresh MPI job can consume the rank-local cotangent.
                raise RuntimeError("injected stop after durable mpi_cc_progress")

        restart_module._CHECKPOINT_EVENT_HOOK = stop_after_rank_progress
    elif mode == "pre-scf-forbid":
        _install_forbidden_pre_scf_work()

    mol = _water_trimer()
    resume = mode in ("resume", "pre-scf-forbid")
    checkpoint = None if mode == "reference" else checkpoint_dir
    kwargs = dict(
        build_mf=_build_mf,
        thresholds=_thresholds(),
        pair_energy_model="all",
        force_full_domains=True,
        thresh_occ=1.0,
        thresh_vir=1.0,
        ccsd_t=True,
        comm=comm,
        root=0,
        return_details=True,
        progress=False,
        checkpoint_dir=checkpoint,
        resume=resume,
    )
    with _gradient_options():
        energy, mol_bar, details = DLNOCCSD.value_and_grad(mol, **kwargs)
    jax.block_until_ready(energy)

    all_fragment_calls = comm.gather(fragment_calls, root=0)
    all_energies = comm.gather(float(energy), root=0)
    root_error = None
    if rank == 0:
        try:
            np.testing.assert_allclose(
                all_energies, all_energies[0], atol=0.0, rtol=0.0
            )
            if mol_bar is None:
                raise AssertionError("root rank did not receive the gradient")
            gradient = np.asarray(mol_bar.coords)
            if not np.all(np.isfinite(gradient)):
                raise AssertionError("gradient contains a non-finite value")
            if len(details.fragments) != 3:
                raise AssertionError(
                    f"expected three fragments, got {len(details.fragments)}"
                )
            if abs(details.e_ccsd_t) <= 1e-12:
                raise AssertionError(
                    "the real CCSD(T) integration case has zero triples energy"
                )
            output.write_text(
                json.dumps(
                    {
                        "mode": mode,
                        "size": size,
                        "energy": all_energies[0],
                        "gradient": gradient.tolist(),
                        "e_ccsd_t": float(details.e_ccsd_t),
                        "fragment_calls": all_fragment_calls,
                        "fragment_owners": [
                            [record.fragment_index, record.worker_rank]
                            for record in details.fragments
                        ],
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
        raise AssertionError("root result validation failed:\n" + root_error)


def _main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=("reference", "interrupt-progress", "resume", "pre-scf-forbid"),
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _run(args.mode, args.checkpoint_dir, args.output)


if __name__ == "__main__":
    _main()
