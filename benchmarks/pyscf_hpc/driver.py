#!/usr/bin/env python3
"""Compare local PySCF builds inside a SLURM allocation: python driver.py --all."""

import argparse
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
VARIANTS = ("stock", "native", "integrals")
RESULTS = HERE / "results"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require_compute_node():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this benchmark through submit.sh inside SLURM.")
    if socket.gethostname().lower().split(".")[0].startswith("login"):
        raise RuntimeError("Refusing to run scientific calculations on a login node.")


def metadata(pyscf):
    from threadpoolctl import threadpool_info

    cpu_name = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_name = line.split(":", 1)[1].strip()
                break
    variables = (
        "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES",
        "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "XLA_FLAGS",
        "JAX_ENABLE_X64", "JAX_PLATFORMS", "PYTHONPATH", "LD_LIBRARY_PATH",
        "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK",
        "SLURM_NTASKS", "SLURM_CPU_BIND", "TMPDIR",
    )
    return {
        "python": sys.executable,
        "python_version": sys.version,
        "pyscf_path": str(Path(pyscf.__file__).resolve()),
        "pyscf_version": pyscf.__version__,
        "hostname": socket.gethostname(),
        "cpu_name": cpu_name,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "environment": {key: os.environ.get(key) for key in variables},
        "threadpools": threadpool_info(),
    }


def run_variant(variant, repeat, settings):
    # All scientific imports occur after the allocation and host checks.
    import numpy as np
    import pyscf
    from pyscf import gto, scf

    expected = (HERE / variant / "pyscf" / "__init__.py").resolve()
    if Path(pyscf.__file__).resolve() != expected:
        raise RuntimeError(f"Wrong PySCF imported: {pyscf.__file__}; expected {expected}")
    if pyscf.__version__ != "2.14.0":
        raise RuntimeError(f"Expected PySCF 2.14.0, found {pyscf.__version__}")

    workloads = {}

    def configure(mf):
        mf.conv_tol = settings["conv_tol"]
        mf.conv_tol_grad = settings["conv_tol_grad"]
        mf.max_cycle = settings["max_cycle"]
        mf.chkfile = None
        return mf

    def record(label, elapsed, energy, gradient=None, **extra):
        energy = float(energy)
        if not math.isfinite(energy) or not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError(f"Invalid energy or timing in {label}")
        row = {"seconds": elapsed, "energy_eh": energy, **extra}
        if gradient is not None:
            gradient = np.asarray(gradient, dtype=float)
            if not np.isfinite(gradient).all():
                raise RuntimeError(f"Nonfinite gradient in {label}")
            row["gradient_eh_per_bohr"] = gradient.tolist()
        workloads[label] = row

    spec = settings["pyscf"]
    mol = gto.M(atom=spec["atom"], basis=spec["basis"], unit="Angstrom",
                verbose=3, max_memory=4000)
    mol.incore_anyway = False
    direct = configure(scf.RHF(mol))
    direct.direct_scf = True
    direct.max_memory = 200  # Below the four-index ERI footprint of this tetramer.
    direct._eri = None
    started = time.perf_counter()
    energy = direct.kernel()
    elapsed = time.perf_counter() - started
    if not direct.converged:
        raise RuntimeError("Direct RHF did not converge")
    if direct._eri is not None:
        raise RuntimeError("Direct RHF unexpectedly cached four-index ERIs")
    record("direct_rhf", elapsed, energy, converged=True,
           cycles=getattr(direct, "cycles", None), nao=mol.nao_nr())

    fitted = configure(scf.RHF(mol).density_fit())
    started = time.perf_counter()
    energy = fitted.kernel()
    elapsed = time.perf_counter() - started
    if not fitted.converged:
        raise RuntimeError("DF-RHF did not converge")
    record("df_rhf", elapsed, energy, converged=True,
           cycles=getattr(fitted, "cycles", None), nao=mol.nao_nr())

    gradient_method = fitted.nuc_grad_method()
    gradient_method.auxbasis_response = True
    started = time.perf_counter()
    gradient = gradient_method.kernel()
    elapsed = time.perf_counter() - started
    record("df_rhf_gradient", elapsed, energy, gradient,
           auxbasis_response=True)

    import jax
    jax.config.update("jax_enable_x64", True)
    from pyscfad import config, gto as ad_gto, scf as ad_scf

    config.update("pyscfad_moleintor_opt", True)
    config.update("pyscfad_scf_implicit_diff", True)
    config.update("pyscfad_scf_first_order_custom", False)
    spec = settings["pyscfad"]
    ad_mol = ad_gto.Mole(atom=spec["atom"], basis=spec["basis"],
                         unit="Angstrom", verbose=3, max_memory=4000)
    ad_mol.build(trace_coords=True, trace_exp=False, trace_ctr_coeff=False)

    def ad_energy(current_mol):
        mf = configure(ad_scf.RHF(current_mol).density_fit())
        energy = mf.kernel()
        return energy, mf.converged

    value_and_gradient = jax.value_and_grad(ad_energy, has_aux=True)
    for phase in ("first", "warmed"):
        started = time.perf_counter()
        result = value_and_gradient(ad_mol)
        jax.block_until_ready(result)
        elapsed = time.perf_counter() - started
        (energy, converged), gradient = result
        if not bool(converged):
            raise RuntimeError(f"PySCFAD DF-RHF did not converge ({phase})")
        record(f"pyscfad_df_rhf_value_gradient_{phase}", elapsed, energy,
               gradient.coords, converged=True, nao=ad_mol.nao_nr())

    info = metadata(pyscf)
    info["jax_version"] = jax.__version__
    info["jax_devices"] = [str(device) for device in jax.devices()]
    info["pyscfad_path"] = str((REPO / "pyscfad").resolve())
    return {"status": "ok", "variant": variant, "repeat": repeat,
            "metadata": info, "input": settings, "workloads": workloads}


def summarize(rows, settings, execution_order):
    measured = {variant: [row for row in rows if row["variant"] == variant
                          and row["repeat"] != "warmup"] for variant in VARIANTS}
    reference = measured["stock"][0]["workloads"]
    timings = {}
    comparisons = []
    valid = True
    for label in reference:
        timings[label] = {}
        stock_median = statistics.median(
            row["workloads"][label]["seconds"] for row in measured["stock"])
        for variant in VARIANTS:
            seconds = [row["workloads"][label]["seconds"] for row in measured[variant]]
            median = statistics.median(seconds)
            timings[label][variant] = {"raw_seconds": seconds,
                "median_seconds": median, "speedup_vs_stock": stock_median / median}
        # Include warmups and every repeat, including the stock repeats themselves.
        for row in rows:
            trial = row["workloads"][label]
            energy_error = abs(trial["energy_eh"] - reference[label]["energy_eh"])
            gradient_error = None
            if "gradient_eh_per_bohr" in trial:
                expected = reference[label]["gradient_eh_per_bohr"]
                actual = trial["gradient_eh_per_bohr"]
                if len(expected) != len(actual) or any(len(g) != 3 for g in actual):
                    raise RuntimeError(f"Unexpected gradient shape for {label}")
                gradient_error = max(abs(x - y) for gx, gy in zip(actual, expected)
                                     for x, y in zip(gx, gy))
            passed = energy_error <= settings["energy_tolerance_eh"]
            if gradient_error is not None:
                passed &= gradient_error <= settings["gradient_tolerance_eh_per_bohr"]
            valid &= passed
            comparisons.append({"variant": row["variant"], "repeat": row["repeat"],
                "workload": label, "energy_abs_error_eh": energy_error,
                "gradient_max_abs_error_eh_per_bohr": gradient_error, "passed": passed})
    return {"status": "ok" if valid else "numerical_mismatch", "input": settings,
            "execution_order": execution_order,
            "reference": "stock repeat 0, matching workload",
            "timings": timings, "comparisons": comparisons,
            "note": "First AD call includes compilation; warmed AD call reuses in-process caches. "
                    "Outer warmup processes are excluded from timing statistics."}


def run_all(settings):
    repeats = settings["repeats"]
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("input.json repeats must be a positive integer")
    for variant in VARIANTS:
        if not (HERE / variant / "pyscf" / "__init__.py").is_file():
            raise FileNotFoundError(f"Missing local PySCF build: {variant}/pyscf")
    rows, order = [], []
    phases = [("warmup", VARIANTS)]
    phases += [(str(repeat), VARIANTS[repeat % len(VARIANTS):]
                + VARIANTS[:repeat % len(VARIANTS)]) for repeat in range(repeats)]
    for repeat, variants in phases:
        for variant in variants:
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join((str(HERE / variant), str(REPO)))
            command = [sys.executable, str(HERE / "driver.py"),
                       "--variant", variant, "--repeat", repeat]
            print(f"Running {variant}, repeat {repeat}", flush=True)
            subprocess.run(command, cwd=HERE, env=env, check=True)
            path = RESULTS / f"{variant}-{repeat}.json"
            rows.append(json.loads(path.read_text()))
            order.append({"variant": variant, "repeat": repeat, "result": str(path.relative_to(HERE))})
    summary = summarize(rows, settings, order)
    write_json(HERE / "summary.json", summary)
    for label, variants in summary["timings"].items():
        print(label)
        for variant, timing in variants.items():
            print(f"  {variant:8s} {timing['median_seconds']:10.4f} s"
                  f"  {timing['speedup_vs_stock']:7.3f}x vs stock")
    if summary["status"] != "ok":
        raise RuntimeError("Numerical differences exceed tolerance; see summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true")
    mode.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--repeat", default="0", help="Child repeat index or warmup")
    args = parser.parse_args()
    require_compute_node()
    if args.repeat != "warmup" and not args.repeat.isdecimal():
        parser.error("--repeat must be a nonnegative integer or warmup")
    settings = json.loads((HERE / "input.json").read_text())
    RESULTS.mkdir(exist_ok=True)
    if args.all:
        run_all(settings)
    else:
        path = RESULTS / f"{args.variant}-{args.repeat}.json"
        try:
            result = run_variant(args.variant, args.repeat, settings)
        except Exception as error:
            write_json(path, {"status": "failed", "variant": args.variant,
                             "repeat": args.repeat, "error": repr(error)})
            raise
        write_json(path, result)


if __name__ == "__main__":
    main()
