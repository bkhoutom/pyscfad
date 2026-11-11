# MP2 response consistency implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task.

**Goal:** Reuse the production response lifecycle for saved MP2 packets, enable useful H reuse for domain MP2, and validate water4 correctness.

**Architecture:** Replay uses `driver._initialize_scf`, `_initialize`, `_add_bars`, and `_finish`, while retaining packet identity and budget checks. Domain MP2 and CC share cache configuration; direct system MP2 retains its existing storage policy. Each group releases H before its deferred geometry response.

**Tech Stack:** Python, JAX, PySCFAD, existing native MP2/DF libraries, pytest.

**Spec:** User request of 2026-11-10 and the preceding source review: align saved-packet replay with the main driver; enable half caching if useful; run water4 correctness checks. Preserve energies, gradients, fixed selections, fitting bases, and packet semantics.

## Constraints and review focus

- Keep unrelated working-tree changes intact; make scoped edits in the requested checkout.
- Preserve empty and partial replay, optional HF/weak contributions, and effective packet sampling budgets.
- Reuse final SCF data when captured; preserve the fallback for prebuilt SCF references.
- Honor `PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE=direct`; default domain MP2 to `half`.
- Preserve system-MP2 cached/direct factor behavior and clean active DF scopes on failure.
- Verify both molecular values and actual integral-generation counts, including local H release before geometry work.

## Task 1: Domain H caching

- [x] Extend metric-reuse tests for domain half/direct modes and unchanged system behavior; verify the half case fails before implementation.
- [x] Share `_df_inputs.configure_cache` without requiring LIS metadata for MP2; activate it only for domain scopes.
- [x] Release consumed H before geometry response, retain receipts, and run metric/DF/CC compatibility tests.

## Task 2: Saved-packet replay

- [x] Add a real packet-export/replay regression that requires final SCF data reuse and shared global geometry response; verify failure first.
- [x] Replace duplicated SCF/common/final-response plumbing with driver helpers, preserving packet validation and partial/empty replay semantics.
- [x] Run replay, response-lifetime, correlation-basis and workflow regressions.

## Task 3: Water4 validation and review

- [x] Add a reproducible example comparing direct, half, saved-packet replay and a fixed-selection finite difference.
- [x] Run water4/cc-pVDZ; save energies, gradients, timings, memory receipts and finite-difference error.
- [x] Run focused integration regressions, update documentation, and request a fresh review of the scoped diff.

## Execution record

- Baseline review: 86 focused tests passed before changes, plus water/STO-3G MP2 versus CC baseline parity below 3e-15 in gradient entries.
- Working in the existing `dlno-stc` checkout to preserve the user's concurrent changes and tested native-library environment. No commit, reset, or dependency replacement is needed.
- RED: domain half test failed with 0 rather than 1 H build; both replay tests failed on an extra final-SCF-data reconstruction.
- GREEN: domain/system metric tests (8), replay/finite workflows (5), response lifetime (2), partial replay and real one/two-rank MPI (2) passed.
- Broad affected suite: 119 passed and 15 failed in one process. RSS reached about 4.7 GB, exceeding old fixtures' 3 GB ceilings; failures were declined forward-B retention and minimum-workspace errors. Fresh-process reruns of all system-cache (6) and shared/half-cache (33) tests passed. Across these runs, all 143 distinct selected tests passed.
- Portable result packets do not require native scalar uncertainty; preserve the schema and report unavailable uncertainty as NaN.
- Water4 setup uses prebuilt factors attached inside the traced SCF builder, following existing production runners. The standard outcore builder cannot consume traced molecules directly.
- Water4/cc-pVDZ, frozen=4, JKFIT/RIFIT, 10 strong-domain groups, 75 weak pairs: direct/half/replay gradient parity below 1.1e-11 Eh/Bohr; 20/10/10 raw-integral passes; one metric, Cholesky and geometry response per group. Maximum retained half-cache size was 24.02 MB.
- Default Boys convergence gave a 1.59e-5 directional FD error despite cache/replay parity below 8e-12. Tightening only the example's Boys settings to conv_tol=1e-12, conv_tol_grad=1e-9, GMRES tol=1e-10 reduced the FD error to 1.19e-9 at step=1e-4 Bohr. Production tolerances remain unchanged.
- Results: `../TEST_general/mp2_response_consistency_water4_20261110/tight_boys/results.json` and `gradients.npz`.
- Independent read-only review found no concrete defects in the scoped implementation. No deferred code findings.
