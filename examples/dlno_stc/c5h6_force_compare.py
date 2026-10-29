"""Compare deterministic and stochastic (T) forces for one molecular C5H6 LIS.

Both gradients use the same frozen Boys domains, W-selected LNO space, fitted
Laplace grid, deterministic MP2 baseline, and canonical SCF initial density.
The only scientific control changed between runs is the triples solver mode.
"""

import argparse
from dataclasses import asdict
import gc
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
    parser.add_argument("--output-dir", type=Path, default=Path("results/c5h6_force_compare"))
    parser.add_argument("--max-memory", type=float, default=800000, help="PySCF memory cap in MB")
    args = parser.parse_args()

    import jax
    import numpy as np
    from pyscf import gto as reference_gto, scf as reference_scf
    from pyscfad import gto, scf, dlno_stc
    from pyscfad.dlno_stc import ccsd
    from pyscfad.dlno_stc.laplace import make_laplace_grid

    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def report(message):
        print(f"[{time.perf_counter() - started:.1f}s] {message}", flush=True)

    def save(name, data):
        (output_dir / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

    xyz = str(args.xyz_file.resolve())
    mol = gto.Mole(atom=xyz, basis="cc-pvdz", unit="Angstrom",
                   max_memory=args.max_memory, verbose=4)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    if mol.natm != 11 or [mol.atom_symbol(i) for i in range(11)] != ["C"] * 5 + ["H"] * 6:
        raise ValueError("expected the supplied C5H6 atom order")
    report(f"C5H6 built: {mol.nao} AOs, {mol.nelectron} electrons")

    reference_mol = reference_gto.M(
        atom=xyz, basis="cc-pvdz", unit="Angstrom",
        max_memory=args.max_memory, verbose=0)
    reference_mf = reference_scf.RHF(reference_mol).density_fit()
    reference_mf.conv_tol = 1e-9
    reference_mf.conv_tol_grad = 1e-6
    reference_mf.kernel()
    if not reference_mf.converged:
        raise RuntimeError("reference DF-RHF did not converge")
    dm0 = np.asarray(reference_mf.make_rdm1())
    report(f"Reference DF-RHF: {float(reference_mf.e_tot):.12f} Eh")

    def build_mf(current):
        mf = scf.RHF(current).density_fit()
        mf.conv_tol = 1e-9
        mf.conv_tol_grad = 1e-6
        mf.chkfile = None
        mf.kernel(dm0=dm0)
        if not mf.converged:
            raise RuntimeError("differentiable DF-RHF did not converge")
        return mf

    mf = build_mf(mol)
    report(f"PySCFAD DF-RHF: {float(mf.e_tot):.12f} Eh")
    if abs(float(mf.e_tot) - float(reference_mf.e_tot)) > 1e-7:
        raise RuntimeError("reference and differentiable SCF disagree")
    frozen = 5  # One 1s core orbital per carbon.
    domains = dlno_stc.build_domain_selections(
        mf, frozen=frozen, force_full_domains=True, boys_grouping="molecule",
        lo_kwargs=dict(init_guess="atomic", conv_tol=1e-10,
                       gmres_options=dict(restart=120, maxiter=100)),
    )
    if len(domains.physical_groups) != 1:
        raise RuntimeError("expected one molecular Boys fragment")
    report(f"Boys domains: {len(domains.fragments)} occupied targets in one fragment")

    orbital_energies = np.asarray(mf.mo_energy)
    roots, weights, grid_diagnostics = make_laplace_grid(
        orbital_energies[domains.active_occ_indices],
        orbital_energies[domains.active_vir_indices],
        kind="fitted", points=6, lambda_gradient=0.1,
    )
    grid = dict(laplace_roots=roots, laplace_weights=weights)
    baseline = dict(mode="deterministic", threads=32, global_seed=666, **grid)
    matrix = dict(mode="deterministic", threads=32,
                  memory_limit_mb=args.max_memory, **grid)
    selected = ccsd.build_static_selections(
        mf, domains, matrix_kind="W", thresh_occ=1e-4, thresh_vir=1e-5,
        baseline_controls=baseline, matrix_controls=matrix, merge_identical=True,
    )
    if len(selected.fragments) != 1 or selected.fragments[0] is None:
        raise RuntimeError("expected one populated molecular LIS")
    report("W matrix and one molecular LIS selected")

    common = dict(geometry=xyz, basis="cc-pvdz", auxbasis="auto", frozen=frozen,
                  threads=32, max_memory_mb=args.max_memory,
                  reference_hf_energy=float(reference_mf.e_tot),
                  hf_energy=float(mf.e_tot),
                  occupied_targets=len(domains.fragments),
                  physical_fragments=len(domains.physical_groups),
                  laplace_grid=dict(roots=roots.tolist(), weights=weights.tolist(),
                                    diagnostics=grid_diagnostics))
    options = dict(baseline_controls=baseline, matrix_controls=matrix,
                   matrix_kind="W", ccsd_t=True, include_hf=True,
                   return_components=True, merge_identical=True)

    def run(mode, triples_controls):
        phase_start = time.perf_counter()
        report(f"Starting {mode} (T) force")
        energy, molecule_bar, components = ccsd.value_and_grad(
            mol, build_mf, selected, triples_controls=triples_controls, **options)
        jax.block_until_ready((energy, molecule_bar))
        force = -np.asarray(molecule_bar.coords)
        if not np.isfinite(force).all() or not np.isfinite(float(energy)):
            raise RuntimeError(f"{mode} (T) returned a nonfinite result")
        fragment = components.fragments[0]
        if fragment.triples_method != mode:
            raise RuntimeError(f"expected {mode} triples, got {fragment.triples_method}")
        result = dict(common, triples_mode=mode, total_energy_eh=float(energy),
                      force_eh_per_bohr=force.tolist(),
                      components=asdict(components),
                      elapsed_seconds=time.perf_counter() - phase_start)
        save(f"{mode}.json", result)
        report(f"{mode} (T) complete: {float(energy):.12f} Eh")
        del energy, molecule_bar, components
        gc.collect()
        return force, result

    deterministic_force, deterministic = run("deterministic", dict(mode="deterministic"))
    stochastic_force, stochastic = run("stochastic", dict(
        mode="stochastic", nsamples=20000, global_seed=71,
        energy_tolerance=0.00005))
    difference = stochastic_force - deterministic_force
    rms = float(np.sqrt(np.mean(difference**2)) * 1000)
    comparison = dict(
        rms_force_error_meh_per_bohr=rms,
        max_abs_force_error_meh_per_bohr=float(np.max(np.abs(difference)) * 1000),
        force_difference_meh_per_bohr=(difference * 1000).tolist(),
        deterministic_energy_eh=deterministic["total_energy_eh"],
        stochastic_energy_eh=stochastic["total_energy_eh"],
        stochastic_triples_samples=stochastic["components"]["fragments"][0]["triples_samples"],
        stochastic_triples_standard_error_meh=
            stochastic["components"]["fragments"][0]["triples_standard_error"] * 1000,
        elapsed_seconds=time.perf_counter() - started,
    )
    save("comparison.json", comparison)
    report(f"RMS stochastic force error: {rms:.6f} mEh/bohr")
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
