# pyscfadlib: support library for pyscfad

pyscfadlib is the support library for PySCFAD.
It contains most of the C/C++ optimized code for derivative calculations.

The CPU generalized eigensolver calls LAPACK through SciPy's Cython API. It
shares SciPy's BLAS backend and thread controls (for example,
`OPENBLAS_NUM_THREADS` or `threadpoolctl.threadpool_limits`), including for
PySCFAD's differentiable `eigh`. Use a threaded SciPy build for parallel
diagonalization. The custom eigenvector derivative and degeneracy handling
remain in PySCFAD; no Python callback is used during a compiled solve.
