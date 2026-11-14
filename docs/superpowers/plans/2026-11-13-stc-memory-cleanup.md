# STC memory cleanup implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement the tasks below.

**Goal:** Return unused host memory at completed MP2 stages while preserving
compiled kernels by default and the existing automatic integral-workspace budget.

**Architecture:** A shared cleanup helper synchronizes live results, optionally
clears JAX caches, collects garbage, and optionally calls glibc `malloc_trim(0)`.
MP2 uses it after initialization/selection, completed domains, and before SCF
response. CCSD delegates its existing opt-in cache cleanup to the helper.
Local DF workspace budgeting retains its original behavior.

**Tech Stack:** Python, JAX, ctypes/glibc, NumPy, PySCF, pytest.

**Spec:** The accepted cleanup proposal in the conversation, with the user's
clarifications that JAX compilation caches must remain intact by default and
no additional temporary integral-workspace limit is wanted.

## Global constraints

- `PYSCFAD_STC_TRIM_MEMORY=1` by default; `0` disables heap trimming.
- `PYSCFAD_STC_CLEAR_JAX_CACHES=0` by default. Serial MP2 consults it once,
  after selection/initialization. Existing CCSD opt-in boundaries remain intact.
- Preserve live numerical arrays and saved pullbacks; never free live buffers.
- Unsupported platforms/allocators skip trimming without failing the calculation.
- No additional integral-workspace cap; retain the existing automatic budget.
- No job-input changes or resubmission.
- Work in the clean existing `dlno-stc` checkout; leave changes reviewable locally.

## Review focus

- Cache clearing disabled: compiled kernels and saved pullbacks remain usable.
- Cache clearing enabled: once per serial MP2 selection, never per pair/domain.
- Synchronization and object release precede trim; unavailable libc is harmless.
- Cleanup preserves energy and full coefficient/coordinate gradients.
- Local DF implementation and its tests retain their original behavior.

### Task 1: Shared cleanup and MP2 boundaries

**Files:** Create `pyscfad/dlno_stc/_cleanup.py` and
`pyscfad/dlno_stc/test/test_memory_cleanup.py`; modify `driver.py`, `ccsd.py`.

**Interface:** `release_memory(stage, *live_results, clear_caches=False,
profile_prefix='stc.mp2', **metadata)`. Clearing additionally requires the env flag.

- [x] Add tests for retained compiled kernels/live VJPs, optional clearing,
  unsupported trimming, and completed-domain/pre-SCF cleanup boundaries.
- [x] Run new tests and confirm failures identify missing cleanup behavior.
- [x] Implement the helper and boundaries; retain existing CC profiling labels.
- [x] Run cleanup, response-lifetime, shared-reference and relevant CC tests.

### Task 2: Cleanup documentation and verification

**Files:** Modify `pyscfad/dlno_stc/README.md`.

- [x] Document cleanup defaults, controls and runtime tradeoffs.
- [x] Remove the additional integral-workspace cap and its tests at user request;
  restore the four local DF implementation/test files to their original versions.
- [x] Run the affected numerical and lifetime suites; inspect the final diff.

## Validation record

- Baseline: 3 response-lifetime/workspace tests passed before changes.
- RED: helper import and missing driver boundary tests
  failed for the intended reasons before their implementations.
- Before removing the workspace-cap changes, 170 tests passed across
  cleanup, response lifetime, production groups, shared reference, weak
  endpoints, CCSD MPI core, single-pass LIS, system gradient, and local/shared/
  half DF suites. Existing molecule-serialization and Boys warnings remained.
- Post-removal verification: 34 tests passed across cleanup, response lifetime,
  production grouping, weak-response collection and shared local DF reverse.
  The four local DF files match HEAD exactly; `git diff --check` passed.
- Fresh selection with optional clearing agrees with the saved-selection
  reference in energy (1e-11 absolute) and full molecular gradient (1e-9).
  The default retains the compiled test kernel; live saved VJPs remain usable.
- The weak endpoint lifetime test now observes collections after endpoint work
  starts, excluding the new initialization collection from its count.
- No production C50 rerun or runtime benchmark; reclaimed RSS and timing remain
  allocator/workload dependent. Job inputs and submission state are unchanged.
