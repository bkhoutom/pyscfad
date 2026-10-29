"""Resumable smooth or conventional LNO(W)-CCSD(T) energy/force benchmark.

The serial queue runs one realization per process with deterministic domain/LIS
selection. Each result is written immediately after completion.
"""

import argparse
from dataclasses import asdict
import faulthandler
import gc
import json
import os
from pathlib import Path
import time

from benchmark_affinity import configure_cpu_workload, check_worker_affinity

os.environ["PYSCFAD_DLNO_RESOURCE_PROFILE"] = "1"
faulthandler.enable()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, required=True,
                        help="CPU affinity budget and numerical worker count")
    parser.add_argument("--xyz-file", type=Path, required=True)
    parser.add_argument("--basis", required=True)
    parser.add_argument("--frozen", type=int, required=True)
    parser.add_argument("--boys-grouping", choices=("molecule", "atom"), required=True)
    parser.add_argument("--targets-meh", type=float, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--job-indices", type=int, nargs="+",
                        help="run these zero-based stochastic jobs from the full target/seed grid")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-samples", type=int, default=20000)
    parser.add_argument("--max-memory-mb", type=float, default=800000)
    parser.add_argument("--cc-conv-tol", type=float, default=1e-7)
    parser.add_argument("--cc-conv-tol-normt", type=float, default=1e-5)
    parser.add_argument("--triples-method", choices=("conventional", "smooth"),
                        default="conventional")
    parser.add_argument("--smooth-gap", type=float, default=.01)
    parser.add_argument("--smooth-laplace-points", type=int, default=10)
    parser.add_argument("--smooth-trace-exponent", type=float, default=5.)
    parser.add_argument("--selection-only", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--deterministic-only", action="store_true")
    mode.add_argument("--stochastic-only", action="store_true")
    parser.add_argument("--reference", type=Path,
                        help="reuse a validated deterministic energy/force result")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    benchmark_cpus = configure_cpu_workload(args.threads)
    if not (0 < args.cc_conv_tol < float("inf")
            and 0 < args.cc_conv_tol_normt < float("inf")):
        parser.error("CCSD convergence tolerances must be finite and positive")
    if args.deterministic_only and args.reference:
        parser.error("--deterministic-only computes a new reference; omit --reference")
    if any(x <= 0 for x in args.targets_meh) or len(set(args.targets_meh)) != len(args.targets_meh):
        parser.error("targets must be positive and distinct")
    if any(x < 0 for x in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be nonnegative and distinct")
    total_jobs = len(args.targets_meh) * len(args.seeds)
    if args.job_indices is not None and (
            len(set(args.job_indices)) != len(args.job_indices)
            or any(i < 0 or i >= total_jobs for i in args.job_indices)):
        parser.error(f"job indices must be distinct integers from 0 to {total_jobs-1}")

    import jax
    import numpy as np
    from pyscf import gto as pgto, scf as pscf
    from pyscfad import gto, scf, dlno_stc
    from pyscfad.dlno_stc import ccsd
    from pyscfad.dlno_stc.laplace import make_laplace_grid
    from pyscfad.lno.ccsd import RCCSD

    RCCSD.conv_tol = args.cc_conv_tol
    RCCSD.conv_tol_normt = args.cc_conv_tol_normt
    RCCSD.max_cycle = 200
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "settings.json").exists():
        previous = json.loads((output / "settings.json").read_text())
        requested = dict(cc_conv_tol=args.cc_conv_tol,
                         cc_conv_tol_normt=args.cc_conv_tol_normt,
                         triples_mode=args.triples_method,
                         cpu_affinity=benchmark_cpus, threads=args.threads)
        for key, value in requested.items():
            if previous.get(key) != value:
                raise ValueError(f"{key} differs from existing results; use a new output directory")
    xyz = args.xyz_file.resolve()
    started = time.perf_counter()

    def report(message):
        print(f"[BENCHMARK {time.perf_counter()-started:.1f}s] {message}", flush=True)

    def save(path, value):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    basis = args.basis
    auxbasis = None
    if args.basis.lower().replace(" ", "") == "def2-sv(p)":
        # Explicit H/C definition: def2-SVP on carbon and its two s shells
        # on hydrogen. Do not let name normalization substitute def2-SVP.
        basis = {"C": pgto.basis.load("def2-svp", "C"),
                 "H": [shell for shell in pgto.basis.load("def2-svp", "H")
                       if shell[0] == 0]}
        auxbasis = "def2-svp-jkfit"
    mol = gto.Mole(atom=str(xyz), basis=basis, unit="Angstrom",
                   max_memory=args.max_memory_mb, verbose=4)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    elements = [mol.atom_symbol(i) for i in range(mol.natm)]
    if args.boys_grouping == "molecule":
        assert elements == ["C"] * 5 + ["H"] * 6
    else:
        assert elements.count("C") == 20 and elements.count("H") == 42
    refmol = pgto.M(atom=str(xyz), basis=basis, unit="Angstrom",
                     max_memory=args.max_memory_mb, verbose=0)
    ref = pscf.RHF(refmol).density_fit(auxbasis=auxbasis)
    ref.conv_tol, ref.conv_tol_grad = 1e-9, 1e-6
    ref.kernel()
    if not ref.converged:
        raise RuntimeError("reference DF-RHF failed")
    dm0 = np.asarray(ref.make_rdm1())

    def build_mf(current):
        mf = scf.RHF(current).density_fit(auxbasis=auxbasis)
        mf.conv_tol, mf.conv_tol_grad, mf.chkfile = 1e-9, 1e-6, None
        mf.kernel(dm0=dm0)
        if not mf.converged:
            raise RuntimeError("differentiable DF-RHF failed")
        return mf

    mf = build_mf(mol)
    jax.block_until_ready(mf)
    if abs(float(ref.e_tot) - float(mf.e_tot)) > 1e-7:
        raise RuntimeError("reference and differentiable SCF disagree")
    domains = dlno_stc.build_domain_selections(
        mf, frozen=args.frozen, force_full_domains=True,
        boys_grouping=args.boys_grouping,
        lo_kwargs=dict(init_guess="atomic", conv_tol=1e-10,
                       gmres_options=dict(restart=120, maxiter=100)))
    expected_groups = 1 if args.boys_grouping == "molecule" else 20
    if len(domains.physical_groups) != expected_groups:
        raise RuntimeError(f"expected {expected_groups} physical fragments")
    energies = np.asarray(mf.mo_energy)
    roots, weights, diagnostics = make_laplace_grid(
        energies[domains.active_occ_indices], energies[domains.active_vir_indices],
        kind="fitted", points=6, lambda_gradient=.1)
    grid = dict(laplace_roots=roots, laplace_weights=weights)
    baseline = dict(mode="deterministic", threads=args.threads, global_seed=666, **grid)
    matrix = dict(mode="deterministic", threads=args.threads,
                  memory_limit_mb=args.max_memory_mb, **grid)
    selection_start = time.perf_counter()
    report("Starting W-LIS selection")
    faulthandler.dump_traceback_later(120, repeat=True)
    selection_options = dict(matrix_kind="W", thresh_occ=1e-4, thresh_vir=1e-5,
                             baseline_controls=baseline, matrix_controls=matrix,
                             merge_identical=True)
    if args.triples_method == "smooth":
        selected = ccsd.build_static_selections(
            mf, domains, smooth_gap=args.smooth_gap,
            smooth_laplace_points=args.smooth_laplace_points, **selection_options)
    else:
        selected = ccsd.build_static_selections(mf, domains, **selection_options)
    selection_seconds = time.perf_counter() - selection_start
    faulthandler.cancel_dump_traceback_later()
    dimensions = [ccsd._fragment_dimensions(selected, i) for i in range(expected_groups)]
    affinity = check_worker_affinity(benchmark_cpus)
    settings = dict(geometry=str(xyz), elements=elements, basis=args.basis,
                    auxbasis=auxbasis or "auto", frozen=args.frozen, boys_grouping=args.boys_grouping,
                    force_full_domains=True, physical_fragments=expected_groups,
                    lno_thresh_occ=1e-4, lno_thresh_vir=1e-5,
                    nao=mol.nao, lis_dimensions=dimensions, threads=args.threads,
                    cpu_affinity=benchmark_cpus, affinity_check=affinity,
                    max_memory_mb=args.max_memory_mb, cc_conv_tol=args.cc_conv_tol,
                    cc_conv_tol_normt=args.cc_conv_tol_normt, cc_max_cycle=200,
                    targets_meh=args.targets_meh, seeds=args.seeds,
                    pilot_samples=args.pilot_samples, triples_mode=args.triples_method,
                    smooth_gap_eh=args.smooth_gap if args.triples_method == "smooth" else None,
                    smooth_laplace_points=(args.smooth_laplace_points
                                           if args.triples_method == "smooth" else None),
                    smooth_trace_exponent=(args.smooth_trace_exponent
                                           if args.triples_method == "smooth" else None),
                    selection_seconds=selection_seconds,
                    preselection_setup_seconds=selection_start-started,
                    mp2_laplace_grid=dict(roots=roots.tolist(), weights=weights.tolist(),
                                          diagnostics=diagnostics),
                    reference_hf_energy_eh=float(ref.e_tot),
                    reused_reference=str(args.reference.resolve()) if args.reference else None,
                    setup_seconds=time.perf_counter()-started)
    save(output / "settings.json", settings)
    report(f"CPU affinity {benchmark_cpus}; all {affinity['runtime_threads']} runtime threads verified")
    report(f"Selected {expected_groups} W-LIS fragments; dimensions {dimensions}")
    if args.selection_only:
        return
    options = dict(baseline_controls=baseline, matrix_controls=matrix,
                   matrix_kind="W", ccsd_t=True, include_hf=True,
                   return_components=True, merge_identical=True)

    deterministic_controls = dict(mode="deterministic")
    if args.triples_method == "smooth":
        deterministic_controls.update(dressing="asymmetric", dressing_backend="native")
    jobs = []
    if args.reference:
        reference = json.loads(args.reference.read_text())
        if (reference["basis"] != args.basis or reference["frozen"] != args.frozen
                or abs(reference["hf_energy"] - float(mf.e_tot)) > 1e-8
                or np.asarray(reference["force_eh_per_bohr"]).shape != (mol.natm, 3)
                or [(f["nocc_lis"], f["nvir_lis"])
                    for f in reference["components"]["fragments"]] != dimensions):
            raise ValueError("saved deterministic reference does not match this W-LIS")
        save(output / "deterministic.json", reference)
        report("Validated and reused deterministic reference")
    elif not args.stochastic_only:
        jobs.append(("deterministic", deterministic_controls))
    for target_index, target in enumerate(args.targets_meh):
        for seed_index, seed in enumerate(args.seeds):
            if args.deterministic_only:
                continue
            if (args.job_indices is not None
                    and target_index * len(args.seeds) + seed_index not in args.job_indices):
                continue
            label = f"stochastic_t{target:g}_seed{seed}"
            controls = dict(mode="stochastic", nsamples=args.pilot_samples,
                            pilot_samples=args.pilot_samples,
                            global_seed=seed, energy_tolerance=target / 1000)
            if args.triples_method == "smooth":
                controls.update(trace_exponent=args.smooth_trace_exponent,
                                adaptive=True, dressing="asymmetric",
                                dressing_backend="native")
            jobs.append((label, controls))
    for label, controls in jobs:
        destination = output / f"{label}.json"
        if destination.exists():
            report(f"Skipping completed {label}")
            continue
        report(f"BEGIN {label}")
        phase = time.perf_counter()
        energy, molecule_bar, components = ccsd.value_and_grad(
            mol, build_mf, selected, triples_controls=controls, **options)
        jax.block_until_ready((energy, molecule_bar))
        force = -np.asarray(molecule_bar.coords)
        if not np.isfinite(force).all() or not np.isfinite(float(energy)):
            raise RuntimeError(f"{label} returned nonfinite energy or force")
        fragments = asdict(components)["fragments"]
        if len(fragments) != expected_groups:
            raise RuntimeError(f"{label} returned {len(fragments)} fragments")
        result = dict(label=label, controls=controls, total_energy_eh=float(energy),
                      force_eh_per_bohr=force.tolist(), components=asdict(components),
                      geometry=str(xyz), basis=args.basis, frozen=args.frozen,
                      hf_energy=float(mf.e_tot), boys_grouping=args.boys_grouping,
                      cc_conv_tol=args.cc_conv_tol,
                      cc_conv_tol_normt=args.cc_conv_tol_normt,
                      threads=args.threads,
                      triples_mode=args.triples_method,
                      selection_seconds=selection_seconds,
                      preselection_setup_seconds=selection_start-started,
                      lno_thresh_occ=1e-4, lno_thresh_vir=1e-5,
                      elapsed_seconds=time.perf_counter()-phase,
                      completed_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        result.update(check_worker_affinity(benchmark_cpus))
        save(destination, result)
        report(f"END {label}: {result['total_energy_eh']:.12f} Eh; "
               f"{result['elapsed_seconds']:.1f} s")
        del energy, molecule_bar, components, force, fragments
        gc.collect()
    report("All requested calculations complete")


if __name__ == "__main__":
    main()
