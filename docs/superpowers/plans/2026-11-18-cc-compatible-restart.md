# Compatible serial CC restart implementation plan

> **For agentic workers:** Execute inline with the executing-plans and test-driven-development workflows. Keep numerical validation scoped to the changed restart/selection paths.

**Goal:** A converged SCF density restart must resume compatible completed CC domain contributions despite numerical differences in SCF and Boys references.

**Architecture:** Keep exact scientific-input, implementation, and discrete-domain checks. Store a numerical SCF reference alongside completed progress and compare reconverged, phase-aligned orbitals/energies with explicit tolerances. On resume validate the newly requested selection, then use the original saved Boys/LIS selections and grids before constructing common-data response. Never mix completed cotangents with newly selected frames.

**Tech stack:** Python, NumPy/JAX, HDF5, the existing serial STC RestartManager and pytest.

**Spec:** User request in this conversation: remove strict SCF/Boys equality requirements while allowing only compatible SCF/domain CC restarts.

## Constraints and review focus

- Retain exact geometry, basis, charge/spin, occupations, frozen space, thresholds, partition, solver settings and native implementation identities.
- Preserve saved discrete domain/rank/anchor labels; differences in numerical Boys references and tiny quadrature-fit noise must not reject an otherwise compatible request.
- Preserve a live differentiable SCF and rebuild its response; never install detached orbitals as a substitute for SCF response.
- Reject different SCF solutions, changed orbital dimensions, nonfinite values, corrupted files and changed scientific settings.
- Validate an SCF checkpoint restart after one completed domain group and after all groups; compare final energy/gradient with an uninterrupted calculation.
- Do not silently reinterpret legacy checkpoints that contain only hashes and no numerical SCF reference. Diagnose that limitation clearly.
- Keep commits/publishing/cluster library deployment outside this request unless separately requested.

## Task 1 — numerical compatibility and safe saved selections

Files: `pyscfad/dlno_stc/_ccsd_checkpoint.py`, new `pyscfad/dlno_stc/test/test_ccsd_restart.py`.

- [x] Add failing tests showing SCF roundoff and Boys-reference/grid roundoff can resume, while changed thresholds, occupations, geometry, SCF solution, or native options cannot.
- [x] Run those tests against the current implementation and confirm strict reference/static identity failures.
- [x] Replace canonical floating-point hashes in the run identity with a saved, checksummed numerical SCF reference and explicit reference comparison.
- [x] Validate requested selection settings/discrete structure, restore the saved selection object, and retain the current manifest/integrity checks.
- [x] Re-run the focused tests, including existing triples/ladder/auxiliary checkpoint guards.

## Task 2 — initialize response in the saved frame

Files: `pyscfad/dlno_stc/ccsd.py`, `pyscfad/dlno_stc/test/test_ccsd_restart.py`.

- [x] Add an interrupted real two-domain gradient test that reconverges from an SCF checkpoint, rebuilds selections, skips completed CC work, and compares with an uninterrupted gradient.
- [x] Reuse validated saved selections before common-frame initialization and normalize controls with the saved grid; share checkpoint option construction with the serial runner.
- [x] Check completed-progress resume and both energy-only and gradient entry points.
- [x] Check that a genuinely incompatible request is rejected before any group is consumed.

## Task 3 — documentation, integration and final review

Files: `pyscfad/dlno_stc/README.md`, restored `omol_1rdn_cc` job documentation/driver only if integration requires it.

- [x] Document reference tolerances, retained safeguards, saved-frame behavior and legacy-checkpoint limits.
- [x] Run focused restart, saved-grid, selection, triples and common-response regressions; run separate-process SGX interruption/resume using the actual job driver.
- [x] Review the complete diff and report tests, remaining limitations and deployment state.

## Validation record

- Baseline red tests: SCF/Boys/grid roundoff failed (3 failures, 9 negative guards passed).
- Both public entry points initially failed the saved-frame-before-initialization assertion; both pass after the fix.
- Actual SGX restarts exposed tied BP construction lists. These are not effective domain labels: pair graph, fragment domains, rank labels and worklist are still checked. All saved selection data retain their exact integrity digest.
- 110 focused CC/triples/auxiliary tests passed; 62 restart/quadrature/common-response tests passed. Final expanded restart suite: 25 passed, including an interrupted two-domain CCSD(T) gradient restarted from an SCF density checkpoint.
- Separate-process SGX driver test passed: one remaining CC solve after interruption, zero after completed restart. Maximum errors against uninterrupted energy/gradient: 2.34e-12 Eh and 1.55e-10 Eh/Bohr. Evidence: `/tmp/dlno_cc_compatible_restart3_20261118/verification.json`.
- Fresh-context code review found no actionable issues; `git diff --check` passed.
- User follow-ups: six active CC input sets now request LIS auxiliary tolerance 1e-3 and Dell Skylake nodes, with 32 requested CPUs and no explicit node exclusions or 32c feature restriction. All six local input-only preflights and syntax checks passed.
- Input rsync does not install the library update on Caltech. The user subsequently requested committing and pushing this restart fix to `dlno-stc`; Caltech library installation remains separate.
- Synced the 18 changed input/submission/README files to `caltech-hpc:/home/bkhou/DLNO_CCSDT/test_pyscfad_stc_mp2/dlno_ccsdt`. Post-transfer checksum dry-run was empty, and all six Caltech input-only preflights passed. No jobs were submitted.
