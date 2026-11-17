# Native weak-endpoint moments implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the approved native endpoint implementation into `dlno-stc`, verify it, commit, and push.

**Architecture:** Keep physical tensors compact. Native BLAS performs large AO contractions; module-level JAX kernels translate fixed 128-column tiles. The weak-pair driver composes their exact pullback with the saved orbital and existing integral pullbacks, retaining general JAX fallbacks.

**Tech Stack:** Python, NumPy/BLAS, JAX, PySCF, pytest; existing native integrals.

**Spec:** `docs/superpowers/specs/2026-11-16-native-weak-moments-design.md`

## Global Constraints

- Target `origin/dlno-stc`; retain the two existing primary-bucketing commits.
- Physical orbital selections and scientific thresholds remain unchanged.
- Optimize concrete CPU order-4 packed moments in float64/complex128; preserve general JAX behavior elsewhere.
- Fixed translation tiles have 128 virtual columns, independent of physical dimensions.
- No cache clearing, map-budget policy, new dependency, or scientific approximation.

## Review Focus

- Mixed dtypes or full Cartesian operators must fall back without losing derivatives (Task 2).
- A native pullback receiving traced seeds must retain linear response (Task 2).
- Zero virtual columns and sizes crossing 128 must preserve values and cotangents (Task 1).
- Complex conjugation and nonsymmetric AO operators must retain full ambient response (Task 1).
- Shared raw adjoint buffers and endpoint replay must not drop molecular or orbital contributions (Task 2).

### Task 1: Native endpoint kernel

**Files:** Create `pyscfad/dlno_stc/_weak_moments.py` and `pyscfad/dlno_stc/test/test_weak_moment_kernels.py`; modify `pyscfad/lib/_multipole.py`.

**Interfaces:** `_weak_moments.supports(screen, raw, order)` gates supported concrete CPU arrays. `linearize(screen, raw, order)` returns `(endpoint, pullback)`; the pullback returns `(screen_bar, raw_bar)`. `endpoint(screen, raw, order)` supplies primal-only use. Factor shared Cartesian translation into `_multipole._translate_moments(...)` without changing its algebra.

- [x] Write numerical tests against the existing JAX endpoint reference: real/complex nonsymmetric operators, zero virtual columns, 127/128/129 column boundaries, all input cotangents, and a real directional finite difference.
- [x] Run the new test file and observe failure because the native implementation is absent.
- [x] Implement the native contractions, fixed-tile translation and transpose, and input guards. Keep dense BLAS thread scopes local to contractions.
- [x] Run new tests plus shared multipole/endpoint tests. Expected: all pass.
- [x] Commit the kernel and its tests with the design/plan.

### Task 2: Driver integration and fallback coverage

**Files:** Modify `pyscfad/dlno_stc/_pairs.py`, `pyscfad/dlno_stc/test/test_weak_moment_native.py`, and `pyscfad/dlno_stc/README.md`; add driver tests to `test_weak_moment_kernels.py` where appropriate.

**Interfaces:** `_linearize_weak_moments(screen, raw, order)` chooses native or reference VJP and handles traced pullback seeds. Existing `_weak_endpoint_from_screen`, `_replay_weak_endpoint`, and `_grouped_weak_pullback` retain signatures and outputs. Host cotangent cropping and accumulation must preserve the traced JAX path.

- [x] Write failing integration tests for the native replay route and stable executable compilation as physical dimensions change. Add traced seeds, unsupported orders/layouts/dtypes, and grouped/non-grouped response checks.
- [x] Run those tests before modifying the driver; verify the missing-route/compilation failure.
- [x] Route supported forward and reverse calls through Task 1, crop concrete CPU seeds on host, and accumulate raw cotangents on host with general AD fallback.
- [x] Run weak endpoint, moment, pair, primary, MPI-weak (when available), and shared multipole regressions. Compare failures against the untouched target and record baseline failures explicitly; see the validation report.
- [x] Run complete 48/96-AO MP2 energy/gradient comparisons and a fresh-process growing-shape mapping probe using integrated source. Compare against stored original numerical results; no prototype adapter in the numerical path.
- [x] Document the optimization and its scope; commit the integrated change.

### Task 3: Review and publish

**Files:** No additional product subsystem; update validation records and this plan.

**Interfaces:** Combined diff from `origin/dlno-stc` through the finished candidate, including earlier primary-bucketing commits.

- [x] Request one fresh whole-branch code review, addressing critical/important findings with reproducing tests.
- [x] Finish applicable verification, check the diff and clean status, and record exact results.
- [x] Fetch `origin`, incorporate any new target changes and retest if necessary.
- [x] Fast-forward or merge the verified result into local `dlno-stc` and push `origin dlno-stc` without force.
- [x] Verify remote/local commit agreement and report commit IDs, tests, and remaining large-system validation limits.
