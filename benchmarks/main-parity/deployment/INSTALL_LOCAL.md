# Customized fork: MKL build and DF-MP2 performance

Source: `/resnick/groups/sharmagroup/bkhou/src/pyscfad`, branch `dlno-stc`.
Environment: `/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env`.

PySCF 2.14.0 and the ten PySCFAD support libraries use Intel oneMKL 2024.2.2,
LP64, sequential BLAS/LAPACK, with GNU OpenMP in the surrounding native loops.
The build uses GCC 11.5, Release, `-march=x86-64-v3 -mtune=generic` and embedded
runtime paths. Do not load another MKL/OpenMP module or override `LD_LIBRARY_PATH`.

NumPy and SciPy retain their separate bundled OpenBLAS libraries. JAX CPU LAPACK
uses SciPy bindings. Native MKL linkage does not replace those libraries.
The custom STC/Armadillo extension retains its existing OpenBLAS 0.3.34 linkage;
the patched libcintad 6.1.1 and custom STC/DLNO functionality are preserved.

## Rebuild and verify

Stop calculations using this installation before rebuilding libraries in place:

```bash
cd /resnick/groups/sharmagroup/bkhou/src/pyscfad
sbatch submit.sh
```

The short script uses `build/pyscfadlib-mkl`, retains libcintad with
`BUILD_LIBCINT=OFF`, rebuilds ten support libraries, checks normal editable
imports and MKL linkage, then runs native and SCF-gradient tests. It does not
reinstall PySCF or change Python dependencies. Old OpenBLAS/FlexiBLAS build
caches are historical and do not describe the installed libraries.

The validated PySCF MKL wheel for this exact environment is
`benchmarks/pyscfad_mkl/wheels/pyscf-2.14.0-2caltechmklseq-py3-none-linux_x86_64.whl`.
Its rebuild inputs and metadata are in `benchmarks/pyscfad_mkl/`.
`verify_native.py` checks actual imported paths, ten support libraries, core
PySCF libraries, LP64 MKL 2024.2.2, one MKL thread, and retained libcintad hash.

## Calculation settings

The dCMP benchmark uses 16 physical cores, `OMP_NUM_THREADS=16`,
`OMP_MAX_ACTIVE_LEVELS=1`, and native MKL/NumPy BLAS at one thread. Its driver
sets SciPy BLAS to 16 only during DF construction and its reverse pass, and to
one for the other phases. This avoids changing every numerical package just
to control the relevant LAPACK work.

The fork benchmark explicitly enables its existing SCF DF preconditioner:

```python
config.update('pyscfad_scf_df_precond', True)
```

The package default remains False. Integral derivative loops retain static
scheduling by default for repeatability. The benchmark explicitly enables
load balancing:

```bash
export PYSCFAD_INTEGRAL_VJP_SCHEDULE=dynamic
```

Dynamic scheduling preserves numerical accuracy but can change floating-point
reduction roundoff. Unset, `static`, and unknown values use the original static
schedule. Original private buffers, barriers and ordered thread reduction remain.

Submit the self-contained benchmark from its calculation directory:

```bash
cd /home/bkhou/DLNO_CCSDT/test_pyscfad_stc_mp2/runtime_test/dCMP
sbatch submit_pyscfad.sh
```

The upstream comparison is in the adjacent `dCMP-main` directory and uses
`pyscfad_main_env`. Each directory retains its own input/driver and output.

## Implementation and evidence

With `pyscfad_moleintor_opt=True`, the real float64 energy-only DF-MP2 path uses
native energy/derivative kernels.
The fork's scalar API, amplitude-returning path, frozen-orbital selection,
out-of-core support, and higher-order AD fallbacks are retained. The custom JVP
moves some reverse work into linearization, so compare MP2 forward plus reverse,
not the reverse time alone.

SCF exchange cotangents use validated low-rank factors where applicable, with
dense fallback and reconstruction checks. Coulomb factor cotangents accumulate
directly through PySCF `lib.dot` to avoid large temporary arrays. The global preconditioner default and
custom response behavior remain explicit.

See [measured results](benchmarks/main-parity/RESULTS.md),
[build metadata](NATIVE_BUILD.json), and [installed linkage](native-linkage.json).
The baseline and tested source/native artifacts, full dCMP gradients, paired
phase timings, regression logs and finite-difference checks are retained under
`benchmarks/main-parity/`.

Rollback copies of changed source and support libraries are under
`benchmarks/main-parity/rollback/`, with a file hash manifest. The previous
PySCF wheel is
`benchmarks/pyscfad_mkl/rollback/pyscf-2.14.0-1rollbackmkl-py3-none-linux_x86_64.whl`.
The previous installation guide is preserved in the rollback directory; its
OpenBLAS timings describe the historical state.
