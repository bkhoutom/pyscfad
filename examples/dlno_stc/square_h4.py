"""One-process square H4 DLNO-STC CCSD(T) acceptance run.

The default calculation uses cc-pVTZ, one complete molecular Boys fragment,
deterministic MP2/W, and adaptive stochastic triples.  H4 has no frozen core.
Set the numerical library thread limits before importing this module.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

for variable in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "PYSCFAD_DENSE_BLAS_THREADS"):
    os.environ[variable] = "32"
os.environ["OMP_DYNAMIC"] = "FALSE"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz-file", type=Path, default=Path(__file__).with_suffix(".xyz"))
    parser.add_argument("--basis", default="cc-pvtz")
    parser.add_argument("--auxbasis", default="auto")
    parser.add_argument("--max-memory", type=float, default=800000, help="PySCF memory cap in MB")
    parser.add_argument("--gradient", action="store_true", help="also compute the nuclear gradient")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import jax
    import numpy as np
    from pyscf import gto as reference_gto, scf as reference_scf
    from pyscfad import gto, scf, dlno_stc
    from pyscfad.dlno_stc import ccsd
    from pyscfad.dlno_stc.laplace import make_laplace_grid

    def report(stage):
        print(f"[{time.perf_counter() - started:.1f}s] {stage}", flush=True)

    started = time.perf_counter()
    auxbasis = None if args.auxbasis.lower() in ("auto", "none") else args.auxbasis
    mol = gto.Mole(atom=str(args.xyz_file.resolve()), basis=args.basis,
                   unit="Angstrom", max_memory=args.max_memory, verbose=4)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    if mol.natm != 4 or any(mol.atom_symbol(i) != "H" for i in range(4)):
        raise ValueError("square H4 input must contain exactly four hydrogen atoms")
    report("Molecule built")

    # Square H4 has multiple RHF stationary solutions.  The PySCF density
    # starts the differentiable SCF on its lower-energy branch and is held
    # fixed as an initial guess during the implicit nuclear response.
    reference_mol = reference_gto.M(
        atom=str(args.xyz_file.resolve()), basis=args.basis,
        unit="Angstrom", max_memory=args.max_memory, verbose=0)
    reference_mf = reference_scf.RHF(reference_mol).density_fit(auxbasis=auxbasis)
    reference_mf.conv_tol = 1e-9
    reference_mf.conv_tol_grad = 1e-6
    reference_mf.kernel()
    if not reference_mf.converged:
        raise RuntimeError("Reference DF-RHF did not converge")
    dm0 = np.asarray(reference_mf.make_rdm1())
    report(f"Reference DF-RHF converged: {float(reference_mf.e_tot):.12f} Eh")

    def build_mf(current):
        mf = scf.RHF(current).density_fit(auxbasis=auxbasis)
        mf.conv_tol = 1e-9
        mf.conv_tol_grad = 1e-6
        mf.chkfile = None
        mf.kernel(dm0=dm0)
        if not mf.converged:
            raise RuntimeError("DF-RHF did not converge")
        return mf

    mf = build_mf(mol)
    report(f"DF-RHF converged: {float(mf.e_tot):.12f} Eh")
    frozen = 0  # Hydrogen has no core orbital to freeze.
    static = dlno_stc.build_domain_selections(
        mf, frozen=frozen, force_full_domains=True, boys_grouping="molecule",
        lo_kwargs=dict(init_guess="atomic", conv_tol=1e-10,
                       gmres_options=dict(restart=120, maxiter=100)),
    )
    report(f"Boys selection: {len(static.fragments)} targets, "
           f"{len(static.physical_groups)} physical fragment")
    energies = np.asarray(mf.mo_energy)
    roots, weights, grid_diagnostics = make_laplace_grid(
        energies[static.active_occ_indices], energies[static.active_vir_indices],
        kind="fitted", points=6, lambda_gradient=0.1,
    )
    report("Six-point Laplace grid fitted")
    grid = dict(laplace_roots=roots, laplace_weights=weights)
    baseline = dict(mode="deterministic", threads=32, global_seed=666, **grid)
    matrix = dict(mode="deterministic", threads=32,
                  memory_limit_mb=args.max_memory, **grid)
    triples = dict(mode="stochastic", nsamples=20000, global_seed=71,
                   energy_tolerance=0.00005)
    selected = ccsd.build_static_selections(
        mf, static, matrix_kind="W", thresh_occ=1e-4, thresh_vir=1e-5,
        baseline_controls=baseline, matrix_controls=matrix, merge_identical=True,
    )
    report("W matrix and LNO spaces selected")
    options = dict(baseline_controls=baseline, matrix_controls=matrix,
                   matrix_kind="W", ccsd_t=True, triples_controls=triples,
                   include_hf=True, return_components=True, merge_identical=True)
    if args.gradient:
        energy, mol_bar, components = ccsd.value_and_grad(mol, build_mf, selected, **options)
        gradient = np.asarray(mol_bar.coords).tolist()
    else:
        energy, components = ccsd.kernel(mf, selected, **options)
        gradient = None
    jax.block_until_ready(energy)
    report(f"DLNO-STC CCSD(T) complete: {float(energy):.12f} Eh")
    result = dict(geometry=str(args.xyz_file.resolve()), basis=args.basis,
                  auxbasis=args.auxbasis, frozen=frozen, threads=32,
                  max_memory_mb=args.max_memory,
                  reference_hf_energy=float(reference_mf.e_tot),
                  hf_energy=float(mf.e_tot),
                  homo_lumo_gap_eh=float(energies[mol.nelectron // 2]
                                         - energies[mol.nelectron // 2 - 1]),
                  total_energy=float(energy), gradient_eh_per_bohr=gradient,
                  components=asdict(components), laplace_grid=dict(
                      roots=roots.tolist(), weights=weights.tolist(),
                      diagnostics=grid_diagnostics),
                  elapsed_seconds=time.perf_counter() - started)
    serialized = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized, flush=True)


if __name__ == "__main__":
    main()
