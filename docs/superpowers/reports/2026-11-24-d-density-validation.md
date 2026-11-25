# D density restoration validation

Validated in the `dlno-stc` checkout based on `a7249107`, using Python 3.11 in
`/home/bkhou/miniforge3/envs/pyscfad_env` and the rebuilt native STC extension.
W remains the default. Select D with `matrix_kind="D"` in
`ccsd.build_static_selections`, or `--matrix-kind D` in the benchmark runner.

## Water tetramer

The geometry is embedded in
`pyscfad/dlno_stc/test/test_d_water4.py`. Four core occupied orbitals are frozen.
All six cases passed: three STO-3G tests and three opt-in cc-pVDZ tests.

| Check | STO-3G | cc-pVDZ |
| --- | ---: | ---: |
| Selected MP2 scalar error versus canonical denominators, Eh | 6.94e-17 | 4.68e-15 |
| Maximum occupied D element error | 2.69e-17 | 4.03e-16 |
| Maximum virtual D element error | 2.86e-17 | 2.00e-16 |
| Maximum input directional derivative error | 3.16e-12 | 1.34e-12 |
| Untruncated D/W energy difference, Eh | 0 | 0 |
| Untruncated D/W maximum gradient difference, Eh/bohr | 3.89e-11 | 1.55e-11 |
| Truncated D nuclear directional derivative error, Eh/bohr | 9.74e-8 | 7.88e-8 |

The independent density reference uses canonical MP2 denominators and rotates
the resulting amplitudes back to the selected target frame. Native D uses an
80-point numerical Laplace grid for this matrix/reference comparison. It also
checks target additivity and separate occupied-Fock, virtual-Fock and DF-factor
directional derivatives with nonsymmetric output seeds.

The untruncated STO-3G comparison uses CCSD(T); cc-pVDZ uses CCSD to keep its
full-space test practical. Total energies are -300.03908775320366 Eh and
-304.94930259694183 Eh, respectively, for both D and W.

Both truncated tests use atom grouping, finite domains, CCSD(T), and D cutoffs
`thresh_occ=thresh_vir=1e-4`. Domain ranks, LIS labels/anchors and quadratures
are held fixed for the finite difference. The displacement is `5e-5` bohr
along atom 1's z coordinate minus 0.4 times atom 4's x coordinate, using
zero-based atom indices. For cc-pVDZ:

- Total energy: -304.9617835045908 Eh.
- LIS occupied/virtual dimensions: (8,17), (6,17), (5,17), (7,17).
- Analytic directional derivative: 0.012212854203410034 Eh/bohr.
- Central finite difference: 0.012212933029331907 Eh/bohr.

The cc-pVDZ benchmark with `--matrix-kind D --lno-thresh-vir 1e-4 --scf-jk df`
also completed: -304.9617835045732 Eh, differing by 1.76e-11 Eh from the test.

## Reproduction

From the repository root, activate the working PySCFAD environment and set:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True JAX_ENABLE_COMPILATION_CACHE=false
export PYTHONPATH=. PYSCFAD_REQUIRE_STC_EXTENSION=1
python -m pytest -q pyscfad/dlno_stc/test/test_d_water4.py
python -m pytest -q pyscfad/dlno_stc/test/test_d_water4.py -k cc-pvdz_high_cost
```

Set `PHASE2_RESULTS_DIR` to a directory to save the numerical JSON receipts.
The repository's normal pytest configuration excludes `_high_cost` cases;
the second command explicitly enables all three cc-pVDZ cases.

## Workflow compatibility

Focused regression suites cover combined energy/density pullbacks, target
batching, sampling, replay identity, controls, memory forecasts, W sequential
selection, and restart. Real interrupted W gradient checkpoints from the
preceding native version resume without recomputing completed contributions.
D/W checkpoint mismatches and unrelated native versions remain rejected.
Benchmark output directories reject switching between D and W, including
legacy settings that omit the matrix choice.

The sampled D reverse memory forecast includes the scalar energy sampler's
update records and ordering buffers, using the actual recorded allocation.
It remains advisory and is omitted for a zero scalar-energy seed. Native
pilot diagnostics include both density and scalar-energy pilots.

D uses the historical positive selection-density convention and an explicit
MP2 energy contraction, with a single combined energy/LIS pullback. Its cutoffs
are dimensionless; W cutoffs have energy units. No cutoff conversion is made.
The experimental `sequential_virtual=True` algorithm remains W-only.

## Pre-push regression check

Fresh verification of the final source passed 260 focused Python tests in four
separate processes: 111 kernel/backend/control tests, 49 workflow tests,
97 restart tests, and the three default STO-3G water4 tests. The three opt-in
cc-pVDZ water4 cases above were validated earlier and were not repeated in this
pre-push run. All three CTest checks also passed: W eigenbasis, sampling
allocation, and sampling concurrency.

The initial combined Python process reached 3.8–4.0 GB RSS and triggered nine
half-transformed DF memory-budget guards in later 3 GB fixtures. The same tests
passed in fresh-process batches without changing source, test tolerances, or
memory limits. The 260 count refers to the successful isolated batches.
