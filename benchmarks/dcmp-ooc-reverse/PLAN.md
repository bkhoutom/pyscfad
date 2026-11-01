# SCF and DF reverse optimization

User-authorized implementation and dCMP validation, 2026-10-31.

## Constraints

- Preserve current local source changes; baseline and candidate are complete copies of the current Python package, not a clean Git revision.
- No numerical tests on login nodes. Use static, self-contained SLURM conditions under `/home/bkhou/DLNO_CCSDT/test_pyscfad_stc_mp2/runtime_test/dCMP-ooc/test_reverse`.
- Keep disk-backed AO factors, full molecular/auxiliary derivatives, scientific inputs, 16 physical Ice Lake cores and 128 GB allocation.
- Preserve unsupported/traced/dtype fallbacks. Cache ownership must be local to the current calculation; memory-bounded fallback must remain available.
- Require energy agreement <=1e-9 Eh and all 102 dCMP gradient components <=2e-7 Eh/bohr. Small independent derivative checks use tighter tolerances.
- Do not change the live package until correctness and full dCMP performance are verified. No Git commits or unrelated edits.

## Work

1. Add meaningful regressions for streamed preconditioner reuse, retained MP2 transform, and bounded native derivative contraction. Run the new regressions against the frozen baseline on compute nodes and confirm intended failures.
2. Cache the preconditioner's fixed occupied-virtual factors within a MiB budget, use scoped dense operations, retain its streaming fallback, and profile construction/application.
3. Retain canonical MP2 forward Lov in the VJP residual and pass saved_output through the supported direct pullback; preserve fallback behavior.
4. Generalize the existing native packed-integral VJP to bounded auxiliary slices, then use it for supported coordinate-only MO/factorized DF derivative contractions. Preserve the current path for unsupported cases.
5. Run focused existing and new regression suites on compute nodes. Review implementation independently before full dCMP comparisons.
6. Benchmark frozen baseline, candidate OOC, and current in-core reference in self-contained jobs. Record full gradients, effective thread pools, source hashes, phase and new substage timings, memory and scheduler status.
7. Use profiles to address remaining repeated fixed transforms or generic AO-adjoint spills when they materially limit reverse runtime. Repeat correctness/performance checks for any further changes.
8. Install only tested changed files into the live package after verifying unchanged baseline hashes. Save exact diffs, original files, benchmark results, reproduction instructions and honest limits of measured performance.

## Rulings

- Existing user authorization covers implementation and test submissions; no additional design approval is needed.
- Use isolated package copies rather than a clean Git worktree: the user's modified and untracked implementation is the required baseline, and native libraries remain the already validated build.
- Meaningful focused regression suites and full scientific checks take precedence over an unrelated repository-wide test run. All numerical work goes through SLURM.
- The forward/reverse timing ratio is a target, not a promised numerical bound. Choose further work from measured profiles rather than assuming all excess is removable I/O.

## Progress

- Baseline and isolated candidate captured; live package remains unchanged.
- Expected red checks: 5821814 (preconditioner: 6 failed/24 passed), 5821831 (native slices/Lov reuse: 11 failed/3 passed), 5821866 (cache memory headroom: 1 failed/30 passed).
- Candidate implements bounded preconditioner cache with scoped dense BLAS, retained MP2 Lov, and sliced native integral-coordinate contractions. Independent review accepted numerical design and requested earlier release of temporary Bov; fixed.
- First integrated regression 5821893: 141 passed/48 failed, all failures caused by one accidental missing max_memory assignment outside the intended preconditioner section. Restored exactly; final source prefix verified byte-identical to baseline.
- Corrected frozen regression 5821952 (regressions_fixed): all 189 tests passed in 141.38 s; only existing mol.dumps warnings. Independent final diff audit confirmed generic prefix restored exactly. Full paired dCMP submitted after green validation.

- First full paired dCMP job 5821996 is running (baseline, first candidate OOC, first candidate in-core).
- Additional generic-geometry red job 5822122: expected 6 spill-assertion failures, 5 passes; all dense coordinate and independent density-bar comparisons passed.
- Added conservative thin-factor geometry dispatch while retaining independent full density transpose; code review found no blocking defect. Full 200-case regression submitted as 5822164.
- Second candidate all 200 tests passed on compute node (5822164), 156.94 s. Final paired dCMP job 5822210 queued with frozen baseline and final candidate packages. Live source remains unchanged.
- Queued final job 5822210 was cancelled before execution to include a newly measured dense contraction issue; its condition remains as an unused frozen snapshot.
- First candidate MP2 reverse 36.94 s vs baseline 59.56 s; profiler exposed 14.68 s in density packing. Installed PySCF einsum reaches sequential native lib.dot even within the threaded NumPy/SciPy context. Added four dense-operation tests; red job 5822279 failed all four on the expected backend guard.
- Candidate MO density now uses two bounded, flattened NumPy GEMMs. Panel microbenchmark 5822301 passed identical-value checks and measured full-construction median 0.144780 s batched vs 0.126563 s flattened; applied the corresponding single GEMM change to factorized density construction.
- Final 204-test compute suite submitted as 5822323; independent review requested for the last changes.
- First paired job 5821996 completed all three cases: baseline 811.628 s, first candidate OOC 619.103 s, first candidate in-core 199.973 s. Max first-candidate OOC gradient difference vs baseline 2.83e-12 Eh/bohr; in-core 1.47e-11; energies <=4.55e-13 Eh.
- All 204 final-GEMM tests passed (5822323, 155.14 s). Review found no blocking issue.
- Remaining factorized density construction dominated by full-array memory passes; fused prototype 5822376 passed J/K/both comparisons and exact symmetry, combined median 0.143680→0.083758 s. Incorporating the guarded real-double path before final freeze; rerun scientific regressions afterward.
- Fused implementation passed all 204 cases in job 5822404 (157.00 s). Review requested only moving native OpenMP hermi_sum outside dense BLAS scope; applied without arithmetic changes. Exact release snapshot undergoing all 204 cases again in 5822443; same snapshot submitted for final paired dCMP.
- Exact release snapshot passed all 204 cases in 5822443 (153.67 s). Final paired run 5822463 started on hpc-34-23 with normal QoS and 1-hour limit. All live changed-file hashes still match baseline; installation is pending full dCMP verification.
- Final job 5822463 completed all three cases, exit 0:0. Full energies/102 gradients and execution metadata passed comparison. Original OOC 641.376 s, optimized OOC 488.451 s, optimized in-core 169.777 s.
- Installed exactly 12 changed Python/test files after verifying all original live hashes and both tested snapshot hashes. No native rebuild, no unrelated changes, no production-input edits. Installation manifest and exact task patch saved.
