#!/usr/bin/env python3
"""Validate the installed PySCF build against the local stock reference input."""

import json
import math
import os
from pathlib import Path
import socket
import sys
import time


HERE = Path(__file__).resolve().parent
EXPECTED_PYSCF = Path(
    "/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env/"
    "lib/python3.11/site-packages/pyscf/__init__.py"
)


def require_compute_node():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this validation through submit.sh inside SLURM.")
    if socket.gethostname().lower().split(".")[0].startswith("login"):
        raise RuntimeError("Refusing to run scientific calculations on a login node.")


def validate(settings, output):
    import numpy as np
    import pyscf
    from pyscf import gto, scf
    from threadpoolctl import threadpool_info

    actual_path = Path(pyscf.__file__).resolve()
    output["metadata"].update(pyscf_path=str(actual_path),
                              pyscf_version=pyscf.__version__)
    if actual_path != EXPECTED_PYSCF.resolve():
        raise RuntimeError(f"Wrong PySCF imported: {actual_path}; expected {EXPECTED_PYSCF}")
    build = json.loads(actual_path.with_name("HPC_BUILD.json").read_text())
    output["metadata"]["build"] = build
    if pyscf.__version__ != "2.14.0" or build["upstream_version"] != "2.14.0":
        raise RuntimeError("Installed package and HPC_BUILD.json must identify PySCF 2.14.0")

    def configure(mf):
        mf.conv_tol = settings["conv_tol"]
        mf.conv_tol_grad = settings["conv_tol_grad"]
        mf.max_cycle = settings["max_cycle"]
        mf.chkfile = None
        return mf

    def record(label, seconds, energy, gradient=None, reference=None):
        expected = settings["expected"][reference or label]
        energy = float(energy)
        expected_energy = float(expected["energy_eh"])
        if not all(math.isfinite(x) for x in (seconds, energy, expected_energy)):
            raise RuntimeError(f"Nonfinite timing or energy in {label}")
        energy_error = abs(energy - expected_energy)
        row = {"seconds": seconds, "energy_eh": energy,
               "energy_abs_error_eh": energy_error, "converged": True,
               "passed": energy_error <= 1e-8}
        if gradient is not None:
            actual = np.asarray(gradient, dtype=float)
            expected_gradient = np.asarray(expected["gradient_eh_per_bohr"], dtype=float)
            if (actual.shape != expected_gradient.shape or actual.ndim != 2
                    or actual.shape[1] != 3):
                raise RuntimeError(f"Unexpected gradient shape in {label}")
            if not np.isfinite(actual).all() or not np.isfinite(expected_gradient).all():
                raise RuntimeError(f"Nonfinite gradient in {label}")
            gradient_error = float(np.max(np.abs(actual - expected_gradient)))
            row.update(gradient_eh_per_bohr=actual.tolist(),
                       gradient_max_abs_error_eh_per_bohr=gradient_error)
            row["passed"] &= gradient_error <= 2e-7
        output["workloads"][label] = row

    spec = settings["pyscf"]
    mol = gto.M(atom=spec["atom"], basis=spec["basis"], unit="Angstrom",
                verbose=3, max_memory=4000)
    mf = configure(scf.RHF(mol).density_fit())
    started = time.perf_counter()
    energy = mf.kernel()
    seconds = time.perf_counter() - started
    if not mf.converged:
        raise RuntimeError("Installed PySCF DF-RHF did not converge")
    record("df_rhf", seconds, energy)

    gradient_method = mf.nuc_grad_method()
    gradient_method.auxbasis_response = True
    started = time.perf_counter()
    gradient = gradient_method.kernel()
    record("df_rhf_gradient", time.perf_counter() - started, energy, gradient)

    import jax
    jax.config.update("jax_enable_x64", True)
    import pyscfad
    from pyscfad import config, gto as ad_gto, scf as ad_scf

    output["metadata"].update(pyscfad_path=str(Path(pyscfad.__file__).resolve()),
                              jax_version=jax.__version__,
                              jax_devices=[str(device) for device in jax.devices()])
    config.update("pyscfad_moleintor_opt", True)
    config.update("pyscfad_scf_implicit_diff", True)
    config.update("pyscfad_scf_first_order_custom", False)
    spec = settings["pyscfad"]
    ad_mol = ad_gto.Mole(atom=spec["atom"], basis=spec["basis"],
                         unit="Angstrom", verbose=3, max_memory=4000)
    ad_mol.build(trace_coords=True, trace_exp=False, trace_ctr_coeff=False)

    def ad_energy(current_mol):
        ad_mf = configure(ad_scf.RHF(current_mol).density_fit())
        energy = ad_mf.kernel()
        return energy, ad_mf.converged

    value_and_gradient = jax.value_and_grad(ad_energy, has_aux=True)
    for phase in ("first", "warmed"):
        started = time.perf_counter()
        result = value_and_gradient(ad_mol)
        jax.block_until_ready(result)
        seconds = time.perf_counter() - started
        (energy, converged), gradient = result
        if not bool(converged):
            raise RuntimeError(f"Installed PySCFAD DF-RHF did not converge ({phase})")
        record(f"pyscfad_df_rhf_value_gradient_{phase}", seconds, energy,
               gradient.coords, reference="pyscfad_df_rhf_value_gradient_warmed")

    output["metadata"]["threadpools"] = threadpool_info()
    if not all(row["passed"] for row in output["workloads"].values()):
        raise RuntimeError("Installed-build results exceed the stock-reference tolerances")


def main():
    require_compute_node()
    cpu_name = "unknown"
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu_name = line.split(":", 1)[1].strip()
            break
    variables = (
        "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK",
        "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OMP_MAX_ACTIVE_LEVELS", "OMP_DYNAMIC",
        "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
        "JAX_ENABLE_X64", "JAX_PLATFORMS", "JAX_ENABLE_COMPILATION_CACHE",
        "XLA_PYTHON_CLIENT_PREALLOCATE", "PYTHONPATH", "LD_LIBRARY_PATH", "TMPDIR",
    )
    output = {"status": "running", "workloads": {}, "metadata": {
        "python": sys.executable, "python_version": sys.version,
        "hostname": socket.gethostname(), "cpu_name": cpu_name,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "environment": {key: os.environ.get(key) for key in variables}},
        "tolerances": {"energy_eh": 1e-8, "gradient_eh_per_bohr": 2e-7}}
    try:
        settings = json.loads((HERE / "input.json").read_text())
        output["input"] = settings
        validate(settings, output)
        output["status"] = "ok"
    except Exception as error:
        output.update(status="failed", error=repr(error))
        raise
    finally:
        (HERE / "results.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print("Installed PySCF and PySCFAD validation passed; see results.json", flush=True)


if __name__ == "__main__":
    main()
