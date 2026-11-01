# PySCF HPC rebuild — 2026-10-30

Installed the benchmark-selected PySCF **2.14.0** native rebuild into
`/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env`. PySCFAD continues to import
that environment's `site-packages/pyscf`. No dependencies were upgraded.

## Selected build

- GCC 11.5.0, Release (`-O3`), explicit `-march=x86-64-v3 -mtune=generic`.
- AVX2 target shared by the Intel and AMD nodes used by this project; no
  login-node-specific or `-march=native` optimization.
- LP64 pthread OpenBLAS **0.3.34** from the existing environment, replacing
  the wheel's OpenBLAS 0.3.3 for the rebuilt native code.
- OpenMP enabled; explicit environment-library RPATH. In the observed
  PySCFAD process this uses one GNU OpenMP runtime, versus two with the stock
  PySCF wheel.
- Original libcint 6.1.3, libxc, xcfun, and solvent library retained. The
  licensed solvent Fortran sources are absent from the wheel/source release.
- PySCF Python code, NumPy, SciPy, JAX, PySCFAD, PySCFADlib, and STC unchanged.

The local wheel is
`wheels/pyscf-2.14.0-1caltechavx2-py3-none-linux_x86_64.whl`.
SHA256: `37ab51101c792781d1d86f0aa03dbb61e2821666c62c126bd14d978981f59581`.
It records the configuration in installed `pyscf/HPC_BUILD.json` and is
intended for this environment and CPUs supporting x86-64-v3, not arbitrary
Linux machines. The version stays 2.14.0; wheel build tag is `1caltechavx2`.

## Benchmark

SLURM job **5736126** completed with exit code 0 in 5m22s on
`hpc-90-30`, Intel Xeon Gold 6130. One task, eight physical cores, 20 GB;
`OMP_NUM_THREADS=8`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.
All variants used the same allocation, Python dependencies, inputs and
convergence settings. Each variant had a warm-up process and three measured
processes in rotating order. This was a shared node, not a dedicated machine.

Direct and density-fitted RHF used a water tetramer/def2-TZVP. The analytic
DF gradient includes auxiliary-basis response. PySCFAD used water/def2-SVP,
coordinate-only differentiation, optimized integral derivatives and implicit
SCF differentiation. Each process measured a first AD call and a second
warm call separately, synchronizing JAX before stopping the clock.

Median wall times, seconds:

| Workload | Original wheel | Installed rebuild | Speedup |
|---|---:|---:|---:|
| Direct RHF | 10.0507 | 9.7595 | 1.03x |
| DF-RHF | 0.8818 | 0.7975 | 1.11x |
| Analytic DF-RHF gradient | 0.9008 | 0.8456 | 1.07x |
| PySCFAD DF-RHF energy + gradient, first call | 6.6759 | 6.3187 | 1.06x |
| PySCFAD DF-RHF energy + gradient, warm call | 1.4990 | 0.3231 | 4.64x |

All 12 processes and all 60 numerical comparisons passed. Across every
candidate and repetition, maximum energy difference was **5.684e-13 Eh**;
maximum gradient-component difference was **5.462e-14 Eh/bohr**. Acceptance
limits were 1e-8 Eh and 2e-7 Eh/bohr. Every SCF converged.

The third candidate additionally recompiled standard libcint 6.1.3 for AVX2.
It retained all original application symbols and exactly matched the original
headers. Its warm AD call was 0.2865 s, but its analytic DF gradient was
0.9567 s, slower than both the original and core-only build. There was no
consistent additional benefit, so the simpler core-only rebuild was selected.
Qcint was considered but not built/installed: a swap-only experiment would
require additional dependent-library rebuilding because its internal headers
differ. MKL was not introduced; the existing OpenBLAS avoids a new runtime
dependency and supports both CPU vendors used here.

These results do not establish speedups for large DLNO/STC jobs. The 3–11%
ordinary-PySCF changes overlap timing variability. The larger warm AD result
is specific to this tiny workload and combined compiler/BLAS/OpenMP change;
it cannot be attributed solely to AVX2. First-call compilation costs remain.

## Installed-build verification

SLURM job **5736718** completed with exit code 0 in 17s on `hpc-35-30`,
AMD EPYC 7543. It imported the actual installed package with `PYTHONPATH`
unset, verified `HPC_BUILD.json`, and checked DF-RHF, analytic gradients and
first/warm PySCFAD gradients against the original Intel benchmark values.
All passed: energy differences <=2.274e-13 Eh, gradient differences
<=4.797e-14 Eh/bohr. This checks cross-vendor execution/correctness; it is
not an AMD speed comparison.

Lightweight imports additionally confirmed PySCFADlib's `ompfix` version,
STC `binding-8`, and the separately installed `pyscf-properties` extension.
Packaging respected each distribution's ownership to preserve that extension.
All scientific work and compilation ran in SLURM; none ran on a login node.

## Files and reproduction

- `input.json`, `driver.py`, `submit.sh`, `build.sh`: complete build/comparison.
- `results/*.json`: per-process timing, energy, gradient and runtime metadata.
- `summary.json`: medians, speedups, raw samples and numerical comparisons.
- `build.log`, `slurm-5736126.out`: build and benchmark logs.
- `installed_check/`: its own input, driver, submit script, logs and results.
- `stock/`, `native/`, `integrals/`: independent package snapshots.
- `sources/`: pinned source archives and PyPI release metadata.
- `package_build.py`: packages only PySCF-owned files, updates WHEEL/RECORD,
  and records build provenance. Installed third-party extensions are excluded.
- `wheels/`: installed build and original-build rollback wheel.

To repeat the comparison, from this directory run `sbatch submit.sh`. Logs
retain SLURM job IDs; JSON results are overwritten on rerun, so copy those
first if you need to preserve this exact comparison. For post-install checks:

```bash
cd /resnick/groups/sharmagroup/bkhou/src/pyscfad/benchmarks/pyscf_hpc/installed_check
sbatch submit.sh
```

To restore the original installation when no jobs use the environment:

```bash
/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env/bin/python -m pip install \
  --no-index --no-deps --force-reinstall \
  /resnick/groups/sharmagroup/bkhou/src/pyscfad/benchmarks/pyscf_hpc/wheels/pyscf-2.14.0-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
```

Restart Python/Jupyter sessions after installing or restoring a native build.

## References

[PySCF build documentation](https://pyscf.org/user/install.html) documents
native CPU targeting, explicit BLAS selection and optional integral engines.
[libcint 6.1.3](https://github.com/sunqm/libcint/tree/v6.1.3) is the upstream
dependency pinned by this PySCF release.
[Caltech HPC documentation](https://www.hpc.caltech.edu/docs/index.html)
provides cluster guidance; actual account, partition, CPU types and QoS were
verified with SLURM and existing project submission scripts.
