# Primary Execution Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for implementation; benchmark and review work are delegated independently.

**Goal:** Implement the two approved primary-preparation execution optimizations and measure C20 runtime without changing domains, thresholds or QR.

**Architecture:** Limit BLAS to one thread inside concrete primary metric/preprojection and Fock-finishing helpers. Discover BLAS libraries once per selection invocation. A separate NumPy/SciPy Fock helper serves concrete reference selection; differentiable selection/replay retain the JAX helper.

**Tech Stack:** Python, NumPy/SciPy, JAX, threadpoolctl, existing native deterministic LT-MP2.

**Spec:** User approval of the two proposals; `TEST_general/domain_construction/primary_preparation_audit/README.md` records the supporting investigation.

## Constraints

- Preserve singleton simplification, QR, all rank/completeness decisions, gradient response and exact domain reuse.
- Retain four OpenMP threads for integrals/native MP2; restore BLAS limits on success and failure.
- Use the existing working tree; preserve earlier uncommitted work and do not create a commit.
- C20 benchmarks compare frozen baseline/current sources, matched saved SCF/Boys, cc-pVDZ/JKFIT/RI, frozen20, LT12, pair1e-5, ED BP .9999, normal identical-domain reuse.
- Serialize benchmarks and correctness jobs; four physical cores, 10GB/no-swap for C20.

## Review focus

- NumPy helper must never enter a live JAX response path.
- Empty/singleton virtual spaces and nonfinite/nonpositive screening gaps.
- Thread restoration on exceptions and unchanged OpenMP budget.
- Basis-independent energies/projectors despite eigensolver sign conventions.
- Baseline is current working source before these two changes, not HEAD or an older diagnostic.

## Task 1: Production execution and focused verification

Files: `_selection.py`, `_pairs.py`, `test/test_primary_execution.py`, `README.md` under `pyscfad/dlno_stc`.

- [x] Add failing tests for concrete Fock parity, finite-gap guards, local BLAS limits/restoration and reference-versus-response dispatch.
- [x] Run tests and confirm expected failures before implementation.
- [x] Decorate concrete metric/preprojection helpers with `blas_threads(1)` and selection entry with `dense_blas_scope()`.
- [x] Add `_finish_weak_screen_reference(occupied, virtual, fock, fragment_index)` with NumPy/SciPy, same symmetrization/edge cases/gap validation, and local BLAS limit.
- [x] Route only `reuse_primary` concrete selection to the new helper. Keep JAX finishing intact.
- [x] Run focused primary/replay/gradient/QR/energy-reuse tests and inspect warnings.
- [x] Document execution defaults and complete independent source review.

## Task 2: Matched benchmark and molecular verification

Artifacts: `TEST_general/domain_construction/primary_execution_update`.

- [x] Freeze baseline before production edits, including singleton scalar change.
- [x] Freeze final current source and verify hashes.
- [x] Run three paired serialized fresh-process C20 calculations per source with normal domain reuse.
- [x] Verify all masks, domain dimensions and correlation energies; report medians and individual times, including preparation/kernel breakdown.
- [x] Run water4 energy/gradient validation after the timing suite, using existing validated reference and a finite-domain derivative check where appropriate.
- [x] Save report, checks and exact timing boundaries. Report only measured gains.

## Execution record

Baseline snapshot confirmed by benchmark agent before production changes. Existing scientific work remains uncommitted. Appropriate focused tests and molecular checks are used rather than the repository-wide unrelated suite.

Task 1 verification: new tests initially failed14/14 for missing reference helper/thread limits. After implementation, the seven focused files `test_primary_execution.py`, `test_singleton_metric.py`, `test_primary_response.py`, `test_primary_selection_reuse.py`, `test_virtual_subspace.py`, `test_energy_preparation_reuse.py`, and `test_energy_execution.py` passed92/92 in19.79s. Warnings were PySCF mol.dumps serialization and Boys saddle/replay notices. Tests used CPU0–3, OMP4, BLAS1, JAX64. Thread-budget tests independently set BLAS/OpenMP2 and verify only BLAS changes/restores.

Independent final review found no correctness issues: live JAX finishing is unchanged, dispatch is limited to concrete `reuse_primary`, and empty/singleton virtuals, Hermitian symmetrization, finite positive gaps, and exception restoration are covered. Benchmark snapshots differ in the two production Python files, documentation, and the new test only.

Task 2 verification: all six C20 runs completed. Median total process time decreased from79.66s to70.61s (11.36%); selection medians were26.23s and12.92s, and downstream kernel medians51.57s and57.81s. Independent stage medians do not sum to the total median. Compilation/cache work can shift between stages and was not timed separately. Domains, ranks, pair masks and actual native dimensions matched exactly; maximum correlation-energy difference was8.88e-16Eh. Source audit passed for41 production Python files and identical native binaries.

Water4 finite-domain energy and all36 analytic gradient components matched the prior validated calculation within2.27e-13Eh and9.33e-12Eh/Bohr, respectively. Two fixed-selection central differences agreed within6.98e-9Eh/Bohr. No reverse domain reconstruction occurred. Regenerated Laplace roots/weights differ below6.4e-12 relatively, with quadrature and derivative differences below8e-16 across256 denominators; the final audit records numerical equivalence and preserves the initial exact-equality diagnostic. All audit checks passed. Results and reproduction details are in `TEST_general/domain_construction/primary_execution_update/README.md`.

## Authorized integration follow-up

The user subsequently requested committing and pushing the accumulated work,
superseding the implementation-stage instruction to leave it uncommitted.
Integrated remote commit `f7b139697e9b07bb4770f11592723fd1df696cb6` by
fast-forwarding first and restoring the saved local changes. The LIS conflict
retains target-first occupied frames and applies the differentiable virtual
projection before virtual framing. Added combined full/truncated-LIS tests with
nonzero ED occupied/virtual overlap and moving-metric derivative checks. Updated
the older singleton eigensolve-count assertion and stale schema documentation.

Rebuilt the native extension. Validation covered 513 distinct Python tests and
three native CTest checks. The initial long Python process passed 233 domain
tests; one RI oracle test hit its fixture's 2 GB memory budget after accumulated
JAX compilations. Its complete six-test file passed in a fresh process without
source changes. The remaining fresh-process groups passed 215 triples/LIS tests
and 64 shared-DF tests; two high-cost tests were deselected by the repository's
normal pytest configuration. Native CTest used the existing dependency library
paths. Logs are saved in `TEST_general/domain_construction/git_integration_f7b1396`.
