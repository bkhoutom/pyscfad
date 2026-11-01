# Customized fork: local native build

Source: `/resnick/groups/sharmagroup/bkhou/src/pyscfad`, branch `dlno-stc`.
Environment: `/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env`.

The fork's Python/C/C++ sources and patched derivative integral engine are
preserved. Its nine PySCFAD support libraries now use GCC 11.5, Release,
`-march=x86-64-v3 -mtune=generic`, OpenMP, and environment-local OpenBLAS 0.3.34.
Both BLAS and LAPACK explicitly use the LP64 pthread library
`/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env/lib/libopenblas.so.0`.
The libraries' runtime paths resolve this dependency without `LD_LIBRARY_PATH`.

The components have different build histories:

| Component | Current state |
| --- | --- |
| PySCF 2.14.0 native libraries | Already rebuilt against OpenBLAS 0.3.34 before this change |
| PySCFAD support libraries (`pyscfadlib`) | Rebuilt from this fork; direct OpenBLAS 0.3.34 replaces system FlexiBLAS |
| STC native extension / Armadillo | Already uses environment OpenBLAS 0.3.34; unchanged |
| Patched `libcintad` 6.1.1 | Preserved byte-for-byte; no BLAS dependency |
| NumPy and SciPy | Existing modern bundled BLAS libraries; unchanged |

The large improvement previously seen in `pyscfad_main_env` came from replacing
its stock PySCF native build. This fork's environment already had that PySCF
optimization. The additional support-library rebuild produces a smaller gain.

## Rebuild

From this source checkout, with calculations using this installation stopped:

```bash
cd /resnick/groups/sharmagroup/bkhou/src/pyscfad
sbatch submit.sh
```

The script rebuilds the support libraries in place and checks imports/linkage
and selected native/gradient tests. It keeps the current patched `libcintad`
using `BUILD_LIBCINT=OFF`; that existing dependency must be present. It does
not reinstall Python dependencies or replace this fork with upstream main.
The AVX2 target requires x86-64-v3-capable nodes.

This script uses `build/pyscfadlib-openblas` for future builds. The initial
validated build was compiled independently in `benchmarks/pyscfad_blas/build`
and its exact binaries were activated by atomic file replacement. The older
`build/pyscfadlib` cache describes the historical FlexiBLAS build, not the
currently loaded libraries. `verify_native.py` checks the active installation.

For a calculation, use `OMP_NUM_THREADS` equal to the allocated CPUs and
`OPENBLAS_NUM_THREADS=1`, with `OMP_MAX_ACTIVE_LEVELS=1`. The full benchmark
used 16 CPUs. No BLAS CPU-family override is needed.

## Validation and measured runtime

Validation job `5760277` passed 51 selected tests covering generalized
eigenproblem derivatives, SCF/DF gradients, native AO transformations,
STC kernels, DLNO gradients and LNO triples contractions. The DLNO tests have
a 1 GB internal memory budget and were run in separate processes to avoid
memory retained by earlier JAX tests. A frozen-core water check agreed with
finite differences within 2.86e-8 Hartree/bohr.

Two full dCMP calculations ran sequentially on the same 16-core allocation:

| Build | Energy + gradient wall time |
| --- | ---: |
| Original support libraries | 321.657 s |
| Rebuilt support libraries | 308.580 s |

The rebuilt run was 1.042x faster in this single comparison. This is not an
exhaustive thread/compiler optimization or a statistical performance guarantee.
Compilation, BLAS backend and effective threading all changed: the original
OpenMP BLAS reported 16 threads, while the new pthread BLAS used one thread.
Nested OpenMP was disabled in both runs; the old setting alone does not prove
oversubscription.

The energies differed by 4.55e-13 Hartree and the complete 34-atom gradients
by at most 1.17e-11 Hartree/bohr. The fork retains its own MP2 AD implementation,
so rebuilding does not make its runtime identical to upstream main.

Build/source/library hashes are saved in `NATIVE_BUILD.json`. Actual installed
dependencies are recorded in `native-linkage.json`. Full inputs, original and
candidate binaries, validation logs, per-phase timings and comparison results
are in `benchmarks/pyscfad_blas/`. The original libraries remain available in
its `original/pyscfadlib/` directory for rollback.

## MKL comparison and density-fitting thread control

The subsequent controlled comparison in
`../pyscfad-main/build/scf-mkl-comparison/` used the same PySCF 2.14.0 sources,
GCC 11.5/AVX2 flags, Python stack and 16-core Intel allocation for both branches.
The fork took 285.21 s with OpenBLAS 0.3.34 and 279.56 s with sequential LP64
MKL 2024.2.2 when allowing 16 SciPy threads throughout. SCF forward, including
DF, took 62.50 s and 68.86 s respectively. The small single-run total difference
does not establish a meaningful MKL benefit for this fork; its active native
libraries remain linked to OpenBLAS 0.3.34.

A separately built, validated MKL candidate for this exact environment is in
`benchmarks/pyscfad_mkl/`, with wheels, native archive, rollback copies, build
metadata and checksums. It is not installed. Job 5763149 passed water energy,
full-gradient and finite-difference checks, retaining the fork's source changes.

JAX CPU LAPACK obtains BLAS/LAPACK function pointers from SciPy. Consequently,
SciPy's BLAS thread limit affects the AD density-fitting triangular solve and
Cholesky operations even though the code calls `jax.scipy.linalg`.
The updated dCMP driver selects SciPy's exact shared library with threadpoolctl:
16 threads for DF forward/reverse, one for SCF/MP2 forward/reverse. Native
OpenBLAS and NumPy stay at one thread, while native OpenMP uses the 16 allocated
CPUs. No NumPy/SciPy rebuild or package-version change was needed.

Job 5763180 tested this driver through the actual installed fork environment:
64.704 s SCF forward (including DF), 105.404 s SCF/DF reverse, and 264.019 s
for the full energy and gradient. All 34-atom gradient components agreed with
the prior driver within 6.82e-12 Hartree/bohr. This run used a different allocation
from the 285.21 s measurement; it does not isolate a phase-threading speedup.
The phase policy provides explicit DF timing and has been numerically validated,
but it is not established as globally optimal for the fork.

The production files are `reproduce_dcmp_ad.py` and `submit_pyscfad.sh` in
`/home/bkhou/DLNO_CCSDT/test_pyscfad_stc_mp2/runtime_test/dCMP`.
Original files, candidate files and validation evidence are retained under
`../pyscfad-main/build/scf-mkl-comparison/production-staging/`.

After activation, job `5760661` repeated the frozen-core energy/finite-difference
gradient check through the normal editable environment and passed. Its inputs
and results are in `benchmarks/pyscfad_blas/installed_check/`.
