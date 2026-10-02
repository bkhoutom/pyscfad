"""Export, reference-solve, and import a small exact-Boys DLNO-MP2 run.

This example fixes a water/STO-3G/full-domain problem with one frozen core.
The reference solver is a tiny deterministic oracle, not an STC sampler. An
external program can replace the middle command using the same HDF5 contract.
"""

import argparse
from pathlib import Path

import numpy as np

from pyscfad import gto, scf
from pyscfad.dlno import _selection, domain as dlno_domain
from pyscfad.dlno.mp2 import _fix_restart_mo_phases
from pyscfad.dlno_stc.mp2 import (
    export_run, imported_value_and_grad, load_disk_packet,
    load_static_from_run, solve_request,
)
from pyscfad.dlno_stc.test.reference import reference_backend


FROZEN = 1


def _molecule():
    mol = gto.Mole(
        atom="O 0 0 0; H 0.05 0.76 0.59; H -0.03 -0.70 0.63",
        basis="sto-3g", verbose=0, max_memory=3000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def _build_mf(mol):
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    return _fix_restart_mo_phases(mf)


def _export(workdir):
    mol = _molecule()
    mf = _build_mf(mol)
    topology = dlno_domain.build_domain_topology(
        mf, frozen=FROZEN, lo_type="boys", force_full_domains=True,
        thresholds=dlno_domain.DLNOThresholds(
            domain_pao=0.0, ed_pao=0.0, pao_norm=1e-10,
        ),
    )
    static = _selection.build_domain_selections(mf, topology)
    rows = export_run(mf, static, workdir, controls={}, reporter=print)
    print(f"Exported {len(rows)} target requests to {workdir}")


def _reference_solve(workdir, fragment=None):
    try:
        from mpi4py import MPI
    except ImportError:
        comm = None
    else:
        comm = MPI.COMM_WORLD
    rank = 0 if comm is None else comm.Get_rank()
    try:
        if rank == 0:
            try:
                fragments = sorted(workdir.glob("fragment_*/input.h5"))
                if not fragments:
                    raise FileNotFoundError(f"no completed input requests in {workdir}")
                names = [path.parent.name for path in fragments]
                if fragment is not None:
                    selected = f"fragment_{fragment:04d}"
                    if selected not in names:
                        raise ValueError(f"fragment {fragment} has no input request")
                    names = [selected]
                error = None
            except Exception as exc:
                names = None
                error = f"{type(exc).__name__}: {exc}"
        else:
            names = error = None
        if comm is not None:
            error, names = comm.bcast((error, names), root=0)
        if error is not None:
            raise RuntimeError(error)
        for name in names:
            folder = workdir / name
            solve_request(
                folder / "input.h5", folder / "result.h5",
                reference_backend, comm=comm,
            )
        if rank == 0:
            print(f"Solved {len(names)} target requests")
    except BaseException:
        # A failed backend may leave another rank inside its own collectives.
        # The application, not the library, owns communicator-wide termination.
        if comm is not None and comm.Get_size() > 1:
            comm.Abort(1)
        raise


def _import(workdir, fragment=None):
    mol = _molecule()
    mf = _build_mf(mol)
    static = load_static_from_run(mf, workdir, frozen=FROZEN)
    selected = (
        tuple(range(len(static.fragments))) if fragment is None
        else (fragment,)
    )
    for fragment_id in selected:
        if not 0 <= fragment_id < len(static.fragments):
            raise ValueError(f"fragment {fragment_id} is outside this run")
        result_path = workdir / f"fragment_{fragment_id:04d}" / "result.h5"
        if not result_path.is_file():
            raise RuntimeError(f"incomplete run: missing {result_path}")
    energy, gradient = imported_value_and_grad(
        mol, _build_mf, static,
        lambda fragment_id: load_disk_packet(workdir, fragment_id),
        fragment_ids=selected, include_hf=(fragment is None),
    )
    if fragment is None:
        print(f"E_total (Eh): {float(energy):.12f}")
        print("dE/dR (Eh/bohr):")
    else:
        print(f"E_fragment {fragment} (Eh): {float(energy):.12f}")
        print(f"dE_fragment {fragment}/dR (Eh/bohr):")
    for atom_index, row in enumerate(np.asarray(gradient.coords)):
        print(f"{atom_index:3d}  {row[0]: .10f}  {row[1]: .10f}  {row[2]: .10f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="action", required=True)
    for action in ("export", "reference-solve", "import"):
        command = subcommands.add_parser(action)
        command.add_argument("--workdir", type=Path, required=True)
        if action in ("reference-solve", "import"):
            command.add_argument("--fragment", type=int)
    args = parser.parse_args(argv)
    workdir = args.workdir.expanduser().resolve()
    if args.action == "export":
        _export(workdir)
    elif args.action == "reference-solve":
        _reference_solve(workdir, args.fragment)
    else:
        _import(workdir, args.fragment)


if __name__ == "__main__":
    main()
