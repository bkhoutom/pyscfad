"""Whole-system STC-MP2 energy and nuclear gradient; run with @input.par.

Use --reference to also compare against full-space PySCFAD DF-MP2.
"""

import argparse
import gc
import json
from pathlib import Path
import shlex
import time


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars="@",
                                     allow_abbrev=False)
    parser.convert_arg_line_to_args = lambda line: shlex.split(line, comments=True)
    parser.add_argument("--xyz-file", type=Path, default=Path("water_4.xyz"))
    parser.add_argument("--basis", default="cc-pvtz")
    parser.add_argument("--auxbasis", default="auto")
    parser.add_argument("--frozen", type=int, default=4)
    parser.add_argument("--max-memory", type=float, default=3000)
    parser.add_argument("--mode", choices=("deterministic", "stochastic"), default="deterministic")
    parser.add_argument("--laplace-points", type=int, default=32)
    parser.add_argument("--laplace-scale", type=float, default=5.5)
    parser.add_argument("--production-samples", type=int, default=300000)
    parser.add_argument("--keep-fraction", type=float)
    parser.add_argument("--system-workload-cutoff", type=float, default=6.5e-3)
    parser.add_argument("--seed", type=int, default=713)
    parser.add_argument("--cderi-restart", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    if args.laplace_points < 2 or args.laplace_scale <= 0 or args.production_samples < 32:
        parser.error("require points >= 2, scale > 0, and production-samples >= 32")
    if args.keep_fraction is not None and not 0 <= args.keep_fraction <= 1:
        parser.error("keep-fraction must be in [0,1]")
    if args.output_prefix is None:
        args.output_prefix = Path(f"results/stc_water4_full_{args.mode}")
    if args.auxbasis.lower() in ("auto", "none"):
        args.auxbasis = None
    return args


def run(args, comm):
    import jax
    import numpy as np
    from pyscfad import config, config_update, dlno_stc, gto, scf
    from pyscfad.df.mpi_outcore import build_cderi
    from pyscfad.mp import dfmp2

    rank = comm.Get_rank()
    started = time.perf_counter()
    config.update("pyscfad_moleintor_opt", True)
    output = args.output_prefix.resolve()
    if rank == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        Path("scratch").mkdir(exist_ok=True)
    comm.Barrier()
    mol = gto.Mole(atom=str(args.xyz_file.resolve()), basis=args.basis,
                   unit="Angstrom", max_memory=args.max_memory,
                   verbose=4 if rank == 0 else 0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    cderi = args.cderi_restart.resolve() if args.cderi_restart else Path("scratch/scf_cderi.h5").resolve()
    if args.cderi_restart is None:
        build_cderi(mol, cderi, auxbasis=args.auxbasis, comm=comm, root=0,
                    max_memory=args.max_memory, progress=True, overwrite=True)
    timings = {"df_setup_seconds": time.perf_counter() - started}
    dm0 = None

    def build_mf(current_mol):
        mf = scf.RHF(current_mol).density_fit(auxbasis=args.auxbasis)
        mf.with_df.max_memory = args.max_memory
        mf.with_df.attach_outcore_cderi(str(cderi))
        mf.conv_tol, mf.conv_tol_grad = 1e-9, 1e-8
        mf.chkfile = None
        mf.kernel(dm0=dm0)
        if not mf.converged:
            raise RuntimeError("DF-RHF did not converge")
        return mf

    scf_started = time.perf_counter()
    if rank == 0:
        mf = build_mf(mol)
        hf_energy = float(mf.e_tot)
        dm0 = np.asarray(mf.make_rdm1())
        nocc = np.count_nonzero(np.asarray(mf.mo_occ) > 0)
        eps = np.asarray(mf.mo_energy)
        eo, ev = eps[args.frozen:nocc], eps[nocc:]
        gap_range = (float(2 * (ev.min() - eo.max())), float(2 * (ev.max() - eo.min())))
        del mf
        gc.collect()
    comm.Barrier()
    timings["scf_seconds"] = time.perf_counter() - scf_started
    nodes, weights = np.polynomial.laguerre.laggauss(args.laplace_points)
    roots, laplace_weights = nodes / args.laplace_scale, weights * np.exp(nodes) / args.laplace_scale
    controls = {"mode": args.mode, "laplace_roots": roots,
                "laplace_weights": laplace_weights, "virtual_block_size": 64,
                "auxiliary_group_size": 32, "global_seed": args.seed}
    if args.mode == "stochastic":
        controls.update(production_samples=args.production_samples,
                        system_workload_cutoff=args.system_workload_cutoff)
        if args.keep_fraction is not None:
            controls["virtual_keep_fraction"] = args.keep_fraction
    solve_started = time.perf_counter()
    result = dlno_stc.value_and_grad(
        mol, build_mf, scope="system", frozen=args.frozen,
        controls=controls, comm=comm, include_hf=True,
    )
    comm.Barrier()
    timings["stc_energy_gradient_seconds"] = time.perf_counter() - solve_started
    timings["total_seconds"] = time.perf_counter() - started
    timings["post_df_seconds"] = timings["total_seconds"] - timings["df_setup_seconds"]
    if rank != 0:
        return
    energy, mol_bar = result
    energy, gradient = float(energy), np.asarray(mol_bar.coords)
    if not np.isfinite(energy) or not np.isfinite(gradient).all():
        raise RuntimeError("nonfinite energy or molecular gradient")
    denominators = np.geomspace(*gap_range, 4096)
    factors = np.exp(-denominators[:, None] * roots)
    quadrature = {"denominator_range": list(gap_range),
                  "inverse_max_error": float(np.max(np.abs(factors @ laplace_weights - 1 / denominators))),
                  "derivative_max_error": float(np.max(np.abs(factors @ (laplace_weights * roots) - 1 / denominators**2)))}
    arrays = {"energy": energy, "hf_energy": hf_energy,
              "correlation_energy": energy - hf_energy, "gradient": gradient}
    info = {"scope": "system", "mode": args.mode, "mpi_size": comm.Get_size(),
            "settings": {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()},
            "energy": energy, "hf_energy": hf_energy, "correlation_energy": energy - hf_energy,
            "gradient_unit": "Eh/bohr", "timings": timings, "quadrature": quadrature}
    if args.reference:
        def native_energy(current_mol):
            current = build_mf(current_mol)
            corr, _ = dfmp2.MP2(current, frozen=args.frozen).kernel(with_t2=False)
            return current.e_tot + corr

        reference_started = time.perf_counter()
        with (config_update("pyscfad_scf_implicit_diff", True),
              config_update("pyscfad_scf_first_order_custom", False)):
            native_value, native_bar = jax.value_and_grad(native_energy)(mol)
        jax.block_until_ready((native_value, native_bar))
        arrays.update(native_energy=float(native_value), native_gradient=np.asarray(native_bar.coords))
        info["reference"] = {"energy": float(native_value),
                             "energy_difference": energy - float(native_value),
                             "gradient_max_difference": float(np.max(np.abs(gradient - np.asarray(native_bar.coords)))),
                             "seconds": time.perf_counter() - reference_started}
    np.savez(output.with_suffix(".npz"), **arrays)
    output.with_suffix(".json").write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
    print(f"STC-MP2 ({args.mode}, system): E(HF)={hf_energy:.12f}, E(corr)={energy-hf_energy:.12f}, E(total)={energy:.12f} Eh")
    print("Nuclear gradient (Eh/bohr):\n", gradient)
    print("Timings (seconds):", timings)
    print("Laplace quadrature:", quadrature)
    if args.reference:
        print("Native DF-MP2 comparison:", info["reference"])
        if args.mode == "deterministic":
            np.testing.assert_allclose(energy, arrays["native_energy"], atol=1e-7, rtol=0)
            np.testing.assert_allclose(gradient, arrays["native_gradient"], atol=1e-6, rtol=0)


if __name__ == "__main__":
    from mpi4py import MPI
    try:
        run(arguments(), MPI.COMM_WORLD)
    except Exception:
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
