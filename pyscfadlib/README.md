# pyscfadlib: support library for pyscfad

pyscfadlib is the support library for PySCFAD.
It contains most of the C/C++ optimized code for derivative calculations.

Source builds use a C/C++ compiler, its OpenMP runtime when available, and an
LP64 BLAS implementation. Binary-wheel repair may bundle these dependencies.
Installed packages may use separate OpenMP runtime copies; their libraries
are left intact. Thread-budget scopes manage the loaded runtimes. DF-J and
SGX use fixed teams during their kernels and restore the caller's settings.

The pinned libcint fork and `libcint.patch` remain part of the source build;
the resulting `libcintad` and headers are installed with this package. Do not
substitute PySCF's ordinary `libcint` for this patched dependency. MPI is not a
support-library dependency: the main package's optional `mpi` extra provides
`mpi4py`, which uses the separately installed system/cluster MPI runtime.

The CPU generalized eigensolver calls LAPACK through SciPy's Cython API. It
shares SciPy's BLAS backend and thread controls (for example,
`OPENBLAS_NUM_THREADS` or `threadpoolctl.threadpool_limits`), including for
PySCFAD's differentiable `eigh`. Use a threaded SciPy build for parallel
diagonalization. The custom eigenvector derivative and degeneracy handling
remain in PySCFAD; no Python callback is used during a compiled solve.

The packed three-center coordinate VJP and tiled AO density contraction
preserve the caller's OpenMP dynamic-team setting throughout their work.
Set `OMP_DYNAMIC=FALSE` before starting Python to prevent automatic team
shrinking, or `OMP_DYNAMIC=TRUE` to permit it. The shared Python thread-budget
scope also preserves this policy. Each kernel respects a smaller caller-supplied
thread budget and limits its team to 32. The density kernel uses serial BLAS
within each worker. Dynamic job scheduling remains the default, independently
of the dynamic-team setting.
For the coordinate VJP, `PYSCFAD_INTEGRAL_VJP_SCHEDULE=static` restores
contiguous static assignment, and `cyclic` selects cyclic static assignment.
All schedules retain private gradient buffers and the ordered final
reduction, though dynamic assignment can change floating-point summation order.
