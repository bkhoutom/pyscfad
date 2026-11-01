# Isolated MKL candidate for the fork environment

This condition prepares PySCF 2.14.0 and the fork's native PySCFADlib for
`/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env`. It does not install packages
or replace editable-source binaries. The baseline and candidate use the same
snapshot of the fork's Python code and the old environment's Python 3.11 and
nanobind 3.0.1.

The candidate uses GCC 11, Release, `-march=x86-64-v3 -mtune=generic`, and explicit
MKL 2024.2.2 LP64 sequential interface/core libraries. PySCF retains its existing
`-fno-math-errno` choice. Both native projects retain GNU OpenMP for outer loops.
The RPATH includes MKL's shared library directory and this environment's `lib`.
MKL is an existing shared cluster installation; it is not redistributed here.

Original libcint, libcintad, libxc, xcfun, and the licensed `libsolvent.so` are
preserved. Only distribution-owned PySCF files enter the wheel, so separately
installed plugins such as `pyscf-properties` are excluded. The native archive
contains only the nine top-level PySCFADlib shared objects; it does not replace
Python/C/C++ sources, version metadata, or libcintad.

Files are self-contained under this condition:

- `input.json`: geometry, thresholds, dependencies, source and baseline hashes.
- `baseline/`, `candidate/`, `python/`: immutable baseline, build candidate, and
  the common fork Python snapshot.
- `build.sh`, `submit.sh`, `driver.py`: compilation and numerical validation.
- `results/summary.json`: baseline/candidate energy and gradient comparisons.
- `wheels/`, `rollback/`: candidate and rollback artifacts after validation.
- `BUILD_METADATA.json`, `SHA256SUMS`: build provenance and artifact checksums.

Run from this directory with `sbatch submit.sh`. The job builds only the copied
sources, compares conventional and AD density-fitted RHF/MP2 for water/def2-SVP
with one frozen orbital, checks an MP2 coordinate derivative against a central
finite difference, verifies linkage, and packages only after all checks pass.
The first and warm timings are correctness-check timings, not dCMP speed claims.
Rerunning replaces this condition's generated logs/results/artifacts; retain a
copy first if those exact artifacts need to be preserved.

`prepare.py` created the current snapshot before submission. It deliberately
refuses to overwrite an existing `input.json`; reproducing the recorded build
uses the already copied source trees and does not read new live sources.

Installation remains a separate decision. The candidate wheel is
`wheels/pyscf-2.14.0-2caltechmklseq-py3-none-linux_x86_64.whl`; its counterpart in
`rollback/` has build tag `1rollbackmkl` and contains the pre-build PySCF snapshot.
Any later wheel installation should use this environment's Python with
`pip install --no-index --no-deps --force-reinstall <wheel>` after stopping
processes that use the environment. The matching native candidate is
`wheels/pyscfadlib-candidate-cp311-linux_x86_64.tar.gz`; the corresponding baseline
archive is `rollback/pyscfadlib-baseline-cp311-linux_x86_64.tar.gz`. A later native
replacement must target only the named shared objects in
`/resnick/groups/sharmagroup/bkhou/src/pyscfad/pyscfadlib/pyscfadlib`, then verify
imports and linkage in a fresh process. No installation command is run by any
script in this condition.

In the preceding common-interpreter dCMP matrix, fork MKL took 279.559 s versus
285.210 s for OpenBLAS: a 1.98% reduction from one sample each. That small change
does not establish a stable benefit or justify a fork installation by itself.
The candidate here additionally isolates the old environment's ABI and RPATH.
Phase-specific SciPy/JAX LAPACK threading is a separate change; replacing native
BLAS does not replace SciPy's bundled OpenBLAS.

Validation completed in Slurm job `5763149` on `hpc-23-10` (exit `0:0`, 2 min
2 s). Baseline and candidate total energies agree within `9.95e-14` Hartree and
full gradients within `3.11e-15` Hartree/Bohr. The candidate's RHF analytic
gradient error is `6.87e-11` Hartree/Bohr, and its MP2 central-finite-difference
error is `2.81e-8` Hartree/Bohr. All four artifact checksums pass. The build
metadata confirms that live PySCF distribution files and all nine live fork
native binaries retain their original hashes. Nothing was installed.
