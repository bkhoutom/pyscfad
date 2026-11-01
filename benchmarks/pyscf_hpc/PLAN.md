# PySCF 2.14.0 build and benchmark

The requested outcome is an optimized PySCF used by `pyscfad_env`, with a
small measured comparison and a recoverable original installation.

The expansion partition includes Intel Skylake/Cascade Lake/Ice Lake and AMD
EPYC. Use explicit x86-64-v3 (AVX2), not host-specific `-march=native`. Keep
PySCF's Python sources/version, NumPy, SciPy, JAX, PySCFAD and PySCFADlib fixed.

1. Preserve the installed package as `stock/pyscf`.
2. Build the same native PySCF code in `native/pyscf`, using GCC 11.5,
   Release/OpenMP, explicit LP64 OpenBLAS 0.3.34 from `pyscfad_env`, and an
   environment-library RPATH. Retain existing libcint/libxc/xcfun dependencies.
3. Copy that independent package to `integrals/pyscf`; replace its integral
   engine with a native libcint 6.1.3 build matching PySCF's pinned dependency.
   Compile explicitly for AVX2 without fast-math reassociation. Keep original
   F12/range/4c options. Preserve the original solvent library: its licensed
   Fortran sources are absent from both the wheel and the source distribution.
4. Run all builds and numerical work in a single SLURM allocation: expansion,
   sharmagroup, normal, one node/task, eight CPUs, 20 GB, one hour. Keep BLAS
   single-threaded under eight OpenMP threads. Use the same reserved cores for
   every variant, warm up, and interleave three measured repetitions.
5. Compare direct RHF, DF-RHF, analytic DF-RHF gradients, and a small PySCFAD
   DF-RHF value/gradient. Require SCF convergence and agreement within 1e-8 Eh
   for energy and 2e-7 Eh/bohr for gradient. Save raw timings and medians.
6. Inspect results before selecting a build. Package the selected tested tree
   as a local wheel, preserving source provenance and a rollback wheel.
   Install into the existing environment with no dependency upgrades; verify
   import paths and native bindings, then run a small installed-build check
   through SLURM. Do not install a numerically failing candidate.

Alternatives considered: a host-native build would limit node portability;
MKL would add a new runtime dependency; a core-only OpenBLAS rebuild is the
controlled candidate. Qcint was rejected before building: its changed
CINTEnvVars layout and CINT2e_1111_loop symbol are incompatible with periodic
code compiled against the standard libcint interface. The third candidate
instead isolates compiling standard libcint for the common CPU target.
Benchmark conclusions apply to this small workload and allocation, not all
DLNO workloads or node families.

References: https://pyscf.org/user/install.html and
https://github.com/sunqm/qcint/tree/v6.1.3 . Cluster settings were verified
through SLURM and existing user job scripts. No calculations run on login.
