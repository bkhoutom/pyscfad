**Canonical DF-MP2 gradient performance — 2026-10-30**

The measured fork bottlenecks are in the MP2 contraction, SCF response paths, and three-centre integral scheduling. A persistent 30-second upstream/reference gap was not established: a fresh reference-shaped upstream run took 145.354 s against the supplied 143.749 s reference, while equivalent staged runs varied by 28.042 s across allocations.

**Final paired result:** job **5767581**, upstream/fork/fork/upstream on the same 16 physical cores of `hpc-53-24`, gave **152.601 s mean upstream** and **156.346 s mean fork**. The fork is **3.745 s (2.45%) slower** in this pair, now close to upstream speed; these two samples per branch do not define a confidence interval. Individual totals were 153.153/152.049 s upstream and 156.533/156.159 s fork. All complete gradients agree within **6.20e-12 Eh/bohr** and energies within **1.60e-12 Eh**. [Complete comparison](final-parity/comparison.json), [phase summary](final-parity/timing-summary.json), [input](final-parity/input.json), [tested source hashes](final-parity/source-manifest.json).

| Mean phase | Upstream, s | Optimized fork, s |
|---|---:|---:|
| DF forward | 8.219 | 8.243 |
| SCF forward, excluding DF | 51.346 | 52.677 |
| MP2 forward + reverse | 21.321 | 21.849 |
| SCF reverse, excluding DF | 56.988 | 54.706 |
| DF reverse | 14.436 | 18.608 |
| Full energy + gradient | 152.601 | 156.346 |

The earlier integrated ABBA job **5766613**, before in-place Coulomb accumulation, averaged 160.722 s upstream versus 186.935 s fork. Its [results](final-pair/comparison.json) isolated the remaining SCF cost and its source snapshot remains under `final-pair/source-snapshot/`. These two jobs used different allocations; the in-place microbenchmark below independently isolates that operation.


All full calculations use the same 34-atom dCMP geometry, cc-pVTZ orbital basis (800 AOs), cc-pVTZ-JKFIT auxiliary basis (2042 functions), 160 electrons, 24 frozen occupied orbitals, and SCF tolerances of 1e-10/1e-8. Drivers check the complete 34×3 gradient, with energy tolerance 1e-8 Eh and maximum gradient tolerance 1e-7 Eh/bohr. Inputs, drivers, submission scripts, and outputs remain in their calculation directories.

The native PySCF/PySCFAD libraries use **MKL 2024.2.2, LP64, sequential**, with GNU OpenMP. NumPy 2.4.6 and SciPy 1.17.1 retain their bundled OpenBLAS libraries; JAX CPU LAPACK uses SciPy's library. This is a mixed BLAS stack. Standard comparisons use OpenMP16, native BLAS1, NumPy1, and SciPy16 only during DF construction/reverse work, with SciPy1 elsewhere. [Native linkage and numerical validation](final-native/results.json).

The first implementation comparison, job **5765581**, held the node, affinity, native MKL configuration, staged driver, and verbosity fixed. These are single fresh-process samples; MP2 times combine forward and pullback because the new custom-JVP implementation computes input derivatives during linearization. [Full measurements](full-dcmp/comparison.json), [controls](full-dcmp/input.json).

| Implementation | Total, s | MP2 forward + pullback, s | SCF pullback excluding DF, s |
|---|---:|---:|---:|
| Fork baseline, preconditioner off | 272.261 | 96.496 | 82.548 |
| Fork with native MP2, preconditioner off | 193.709 | 22.155 | 82.832 |
| Fork with native MP2, preconditioner on | 184.312 | 22.115 | 76.736 |
| Upstream main | 154.911 | 21.521 | 57.672 |

The source changes address separate costs:

- **Native MP2:** replace the eager energy-only scan contraction with PySCF's pair-energy kernel and the upstream native input-gradient kernel. The MP2 phase fell by 74.340 s in the matched comparison. T2 requests, unsupported dtypes, missing libraries, zero auxiliary rows, and staged/higher-order AD retain appropriate fallbacks. A custom JVP preserves forward AD and arbitrary reverse cotangents.
- **Preconditioning:** explicitly enable the fork's existing Coulomb preconditioner. Its global default remains unchanged. The matched matrix reduced SCF pullback by 6.096 s; a separate [on/off comparison](../../../pyscfad-perf-scf/benchmarks/scf-parity/preconditioner-pair16/comparison.json) reduced GMRES iterations from 14 to 10.
- **Generic low-rank exchange:** extend fast density/response factor handling beyond the existing implicit-solver-only path. Pivoted Cholesky and response range recovery use full reconstruction checks; unsupported or high-rank inputs retain the dense transpose. Exploratory low-rank timings used different allocations and do not establish an isolated speedup. The final ABBA experiment measures the integrated change; [source identities](final-parity/source-manifest.json) and the regression logs below preserve its evidence.
- **Coulomb factor accumulation:** replace two `eri_blk += A @ B` operations with in-place PySCF `lib.dot(..., c=eri_blk, beta=1)`. The focused 2042×320400 factor-bar benchmark reduced the median from **6.360 to 0.378 s** (16.83×), with maximum difference **5.55e-17**. Three such calls account for approximately 18 s of the persistent SCF gap; the final full comparison includes this change. [Microbenchmark](../../../pyscfad-perf-scf/benchmarks/scf-parity/coulomb-factor16/results.json).
- **Integral scheduling:** an isolated static/dynamic/dynamic/static experiment changed only the worksharing schedule of the same native three-centre transpose. Mean wall time fell from **9.880 to 4.244 s**, with maximum gradient difference **6.22e-15**. Unequal shell/block costs explain the measured scheduling imbalance. Thread-private accumulation, barriers, and the corrected reduction remain intact. Static remains the package default; `PYSCFAD_INTEGRAL_VJP_SCHEDULE=dynamic` opts into the faster scheduling and permits tiny floating-point summation-order differences. These native-only timings use a synthetic cotangent and cannot be substituted for full-gradient timings. [Scheduling audit](df-reverse-audit/audit.json).

The upstream/reference investigation used three sequential fresh processes in job **5765368**, with installed-path/native-library checks and no JIT warm-up. [Reference comparison](../../../pyscfad-main/build/reference-comparison/comparison.json).

| Upstream condition | Total, s |
|---|---:|
| Supplied historical monolithic reference | 143.749 |
| Current staged driver, SciPy16 for DF only | 140.009 |
| Exact reference driver, SciPy1 throughout | 192.488 |
| Exact reference driver, SciPy16 throughout | 145.354 |

Staged versus monolithic rows change both AD boundaries and thread scopes. The reference's internal `scf_cycle` timer includes DF, whereas the staged SCF timer excludes the prebuilt DF; those internal numbers are not interchangeable.

A second allocation reproduced **168.051 s** for the numerically equivalent staged configuration. Matching package/native/input hashes and calculation ASTs are recorded in the [cross-allocation checks](../../../pyscfad-main/build/reference-comparison/cross-allocation.json). The [hardware audit](../../../pyscfad-main/build/reference-comparison/hardware_audit.json) identifies a socket-placement confound inferred from scheduler topology; actual NUMA causality was not proven. On that second allocation, verbosity5 took 166.146 s and NumPy16 took 163.571 s: logging does not explain the gap, and NumPy16 provided only a 4.480 s improvement. [Matched controls](../../../pyscfad-main/build/reference-comparison/logging-numpy/comparison.json).

The supplied reference's runtime metadata reports Python3.14.2/PySCF2.12.1, unlike its historical source docstring. Its actual JAX/native-BLAS revisions and compiler flags are not established. The GCC version in Python's banner is not evidence for the native-library compiler. No hardware or compiler explanation is inferred from the historical timing alone.

Validation currently records **128 passed, 2 deselected**, plus the two isolated DLNO gradient checks: [regressions](final-parity/regressions-final.log), [DLNO unfrozen](final-parity/dlno-unfrozen.log), [DLNO frozen](final-parity/dlno-frozen.log). Coverage includes native MP2 differentiation/fallbacks, SCF/DF, low-rank exchange, STC, and LNO contractions. [Independent native checks](final-native/results.json) also cover 1/16-thread three-centre gradients, scheduling/repeat behavior, and water DF-MP2 finite differences.

The two excluded failures are `pyscfad/mp/test/test_mp2.py::test_nuc_grad` and `::test_nuc_grad_n2`: conventional non-DF MP2 encounters an absent traced SCF ERI cache. Both were reproduced on the unchanged fork in [baseline job 5765640](../../../pyscfad-perf-mp2/benchmarks/native-mp2/slurm-5765640.out); they are recorded exclusions, not a claim that the entire repository suite is green.

An [environment binary audit](df-reverse-audit/environment_audit.json) found the checked NumPy/SciPy/JAX binaries identical across environments. GNU OpenMP differs only in non-runtime metadata/debug sections; all allocated ELF sections match. This provides no evidence that replacing those packages would resolve the measured SCF gap.

The [DF-only profile](../../../pyscfad-main/build/reference-comparison/df-phase-profile/RESULTS.md) measured raw three-centre integrals at 1.070/1.069 s and their native reverse calls at 3.941/3.943 s for main/fork. A synthetic DF-reverse difference occurs at a boundary that waits for pending JAX work and materializes array order; it does not establish a libcint/compiler or copying-only cause. No integral-engine rebuild was justified.

The validated changes are now active in `/resnick/groups/sharmagroup/bkhou/src/pyscfad` and `pyscfad_env`. PySCF was installed from the local MKL wheel; all 81 files owned by `pyscf-properties` were preserved. The normal-import installed check **5770098** passed source/native identity checks, MKL LP64/one-thread checks, static/dynamic/1-thread/16-thread integral derivatives, water DF-MP2 finite differences (error **1.38e-10 Eh/bohr**), and `pip check`. [Installed results](installed-check/results.json), [active linkage](../../native-linkage.json), [build/activation metadata](../../NATIVE_BUILD.json).

The two runtime directories now have updated submission scripts and local documentation. The fork driver explicitly enables its DF preconditioner and its script selects dynamic integral scheduling. Both scripts use the verified thread policy, explicit fixed-core placement, and no persistent JIT cache. [Applied file manifest](deployment/activation.json), [rebuild and rerun instructions](../../INSTALL_LOCAL.md). Static scheduling and preconditioning defaults at package level remain unchanged; custom STC and libcintad binaries match their saved original hashes. Source/native rollback copies remain in `rollback/`, with a separate PySCF rollback wheel under `../pyscfad_mkl/rollback/`.
