# pyscfadlib: support library for pyscfad

pyscfadlib is the support library for PySCFAD.
It contains most of the C/C++ optimized code for derivative calculations.

The CPU generalized eigensolver calls LAPACK through SciPy's Cython API. It
shares SciPy's BLAS backend and thread controls (for example,
`OPENBLAS_NUM_THREADS` or `threadpoolctl.threadpool_limits`), including for
PySCFAD's differentiable `eigh`. Use a threaded SciPy build for parallel
diagonalization. The custom eigenvector derivative and degeneracy handling
remain in PySCFAD; no Python callback is used during a compiled solve.

The packed three-center coordinate VJP and tiled AO density contraction use
dynamic OpenMP teams and dynamic job scheduling by default. Each kernel
respects a smaller caller-supplied thread budget and limits its team to 32.
The density kernel uses serial BLAS within each worker. Both kernels restore
the caller's OpenMP dynamic-team flag on exit; other kernels are unaffected.
For the coordinate VJP, `PYSCFAD_INTEGRAL_VJP_SCHEDULE=static` restores
contiguous static assignment, and `cyclic` selects cyclic static assignment.
All schedules retain private gradient buffers and the ordered final
reduction, though dynamic assignment can change floating-point summation order.
