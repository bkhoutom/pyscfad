"""Gauge-consistent MPI IAO-DLNO-MP2 energy and nuclear gradient.

Rank 0 alone runs SCF, builds the fixed IAO/domain topology, constructs the
continuous IAO/PAO representation, and owns their saved pullbacks.  Workers
adopt the broadcast canonical MO state without another diagonalization.  The
additive strong ED terms and unordered weak multipole pairs are distributed
across ranks; their cotangents are reduced before the one common-orbital and
one implicit SCF response on rank 0.

The reported correlation energy is the complete IAO-DLNO-MP2 correction
(strong ED MP2 plus the weak multipole far field).  This is the PT2 quantity
used by the IAO-DLNO-CCSD(T) driver.  The weak term is the OS-based multipole
model for the omitted total distant-pair correlation; it is not a literal
weak-pair OS+SS integral evaluation.

Run from the repository root, for example::

    mpirun -np 2 .venv/bin/python \
        examples/lno/18-mpi_iao_fragment_mp2.py

The defaults use the water trimer with cc-pVDZ/cc-pVDZ-RI.  A small weak-pair
MPI smoke test is::

    mpirun -np 2 .venv/bin/python \
        examples/lno/18-mpi_iao_fragment_mp2.py \
        --geometry examples/lno/water_dimer_far.xyz --basis sto-3g \
        --auxbasis weigend --pair-threshold 1e-4 --quiet
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

# Avoid rank-by-rank thread oversubscription.  Explicit user settings win.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy
from mpi4py import MPI

from pyscfad import config, gto, scf
from pyscfad.dlno.iao_mp2 import IAOFragmentMP2Thresholds
from pyscfad.dlno.iao_mp2_mpi import IAOFragmentMP2


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry",
        default="water_trimer.xyz",
        help="XYZ/PySCF geometry; relative paths are also resolved beside this file",
    )
    parser.add_argument("--basis", default="cc-pvdz")
    parser.add_argument("--auxbasis", default="cc-pvdz-ri")
    parser.add_argument("--frozen", type=int, default=0)
    parser.add_argument("--pair-threshold", type=float, default=1e-4)
    parser.add_argument("--max-memory", type=float, default=4000.0)
    parser.add_argument(
        "--output-npz",
        help="optional rank-0 NPZ containing energy and nuclear gradient",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def _geometry_argument(value):
    candidate = Path(value)
    if candidate.exists():
        return str(candidate.resolve())
    beside_example = Path(__file__).resolve().parent / value
    if beside_example.exists():
        return str(beside_example)
    # Preserve PySCF's inline-geometry syntax when this is not a file.
    return value


def main(argv=None):
    args = _parser().parse_args(argv)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()

    config.update("pyscfad_moleintor_opt", True)
    config.update("pyscfad_scf_implicit_diff", True)
    config.update("pyscfad_scf_first_order_custom", False)

    mol = gto.Mole(
        atom=_geometry_argument(args.geometry),
        basis=args.basis,
        max_memory=args.max_memory,
        verbose=0 if args.quiet or rank != 0 else 4,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)

    # Only rank 0 builds the global SCF CDERI.  Workers attach an otherwise
    # unused placeholder so their mf pytrees have the same differentiable DF
    # leaves; integral-direct local ED transforms build only their local raw
    # three-center integrals and never read a worker global CDERI.
    with tempfile.TemporaryDirectory(
        prefix=f"pyscfad-iao-mp2-rank{rank}-"
    ) as scratch:
        cderi_path = str(Path(scratch) / "cderi.h5")
        if rank == 0:
            df_builder = scf.RHF(mol).density_fit(
                auxbasis=args.auxbasis
            ).with_df
            df_builder.max_memory = mol.max_memory
            df_builder._cderi_to_save = cderi_path
            df_builder.build()
            del df_builder
        comm.Barrier()

        def build_mf(
            mol_,
            *,
            mo_coeff_init=None,
            mo_energy_init=None,
            mo_occ_init=None,
            e_tot_init=None,
        ):
            mf = scf.RHF(mol_).density_fit(auxbasis=args.auxbasis)
            mf.with_df.max_memory = mol_.max_memory
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

        thresholds = IAOFragmentMP2Thresholds(
            pair_energy=args.pair_threshold
        )
        energy, mol_bar = IAOFragmentMP2.value_and_grad(
            mol,
            build_mf=build_mf,
            frozen=args.frozen,
            thresholds=thresholds,
            pair_energy_model="multipole",
            include_hf=True,
            comm=comm,
        )

        if rank == 0:
            gradient = numpy.asarray(mol_bar.coords)
            print(
                "IAO-DLNO-MP2 total energy "
                f"(strong ED + weak multipole, MPI {nproc} ranks) = "
                f"{float(energy):.12f}"
            )
            print("IAO-DLNO-MP2 gradient (Hartree/Bohr):")
            print(gradient)
            if args.output_npz:
                output = Path(args.output_npz).expanduser().resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                numpy.savez(
                    output,
                    energy=float(energy),
                    gradient=gradient,
                    nproc=nproc,
                    pair_threshold=args.pair_threshold,
                    basis=args.basis,
                    auxbasis=args.auxbasis,
                )
                print(f"Wrote {output}")

        # Keep the root CDERI and every worker DF placeholder alive until all
        # reverse passes and collectives finish.
        comm.Barrier()


if __name__ == "__main__":
    main()
