# DLNO-STC clean integration plan

> Execute the user's authorized integration in the isolated sibling worktree.

**Goal:** Reconcile `dlno-stc` history into `dlno-stc-clean`, preserving the
`origin/dlno-stc-fix` implementation and supported interfaces, then validate
and push only the new branch.

**Architecture:** Ordinary merge with path-by-path review of conflicts and
clean merges; retain explicit NumPy/SciPy weak response, unified fragment
workflow, saved grids, MRCC deterministic triples and DF/SGX improvements.
Carry forward useful numerical coverage without restoring obsolete helpers.

**Tech stack:** Git worktrees, Python/JAX/NumPy/SciPy/PySCF, CMake C++ extensions,
pytest and MPI when available.

**Spec:** User request of 2026-11-17, summarized by the constraints and tasks below.

## Constraints

- Original checkout stays on `dlno-stc`; preserve its files, index, builds and
  editable installation. Source branch refs must not change locally/remotely.
- Baseline: `551cd2d4c263b324d68dbfbc7fdec53fff87edf7`.
- Both original local and fetched remote `dlno-stc`:
  `c4f64d13377ebaf683659aecf695709a7db8bd8a` (zero local-only commits).
- Common ancestor: `8ade7f4b2765d0d771f2276662cf6d66b912785f`.
- Worktree: `/home/bkhou/Project/DLNO_CC/pyscfad-dlno-stc-clean`.
- No blanket ours/theirs strategy, force push, source-branch update or deletion.
- Optional moment batching remains a follow-up unless a benchmark proves value.
- Final validation follows history reconciliation; historical reports are not
  validation evidence for this tree. Separate memory-sensitive test processes.

## Tasks

- [x] Inspect instructions, clean status, refs and worktrees; fetch origin;
  record original state outside the checkout; create isolated branch from fix.
- [x] Audit all seven unique `dlno-stc` commits against the common ancestor and
  cleaned implementation, documenting every disposition and API migration in
  `docs/superpowers/reports/2026-11-17-dlno-stc-clean.md`.
- [x] Merge recorded `dlno-stc` tip with `--no-ff --no-commit`; inspect every
  staged addition/change and conflict. Resolve by function responsibility;
  omit superseded JAX helper additions and historical performance reports.
- [x] Adapt regression coverage in `pyscfad/dlno_stc/test/`; reconcile obsolete
  tests with the cleaned API, preserving valid partition/rank coverage. Repair
  the stale system-frame assertion against the test reference module.
- [x] Build changed native extensions under the clean worktree; use compatible
  installed dependencies with explicit import isolation and bounded threads.
  Print executable, Python imports and extension paths; preserve original builds.
- [x] Commit reconciliation and regression changes; assert ancestry of all
  recorded source tips before final validation.
- [x] Run focused PAO/rank/degeneracy/empty-space, weak values/all cotangents,
  grouped response, W/LIS reuse, saved grids and restart rejection tests.
- [x] Run deterministic molecular MP2 and CCSD(T) energies/coordinate gradients,
  finite differences, DF/SGX native checks and available serial/MPI comparisons.
  Keep settings explicit and report unavailable or pre-existing failures.
- [x] Inspect workstation memory/compilation behavior with a small case;
  review the whole final diff independently; rerun affected tests after fixes.
- [x] Record exact commands/results, source IDs and limitations; prepare report;
  verify unchanged original status/files/refs and remote tips, then prepare the
  authorized normal push of only `HEAD:refs/heads/dlno-stc-clean`. The push and
  remote-HEAD verification follow the evidence commit and are reported at delivery.

## Review focus

- Automatically restored JAX helpers must not reenter the host implementation.
- Empty/degenerate spaces and rank cutoffs must retain finite, correct adjoints.
- Paired tests must compare identical saved selections, quadratures and modes.
- Native modules must be rebuilt or proven source-identical and isolated.
- Existing obsolete-input tests need supported replacements, not blind deletion.

## Execution ledger

- Initial original status is clean; no repository AGENTS.md/CLAUDE.md found.
  Read root README, branch README and build instructions. Local `dlno-stc-fix`
  does not exist and must remain absent. Snapshot at
  `/tmp/dlno-stc-clean-integration/original-state.json`.
- User explicitly authorized implementation, merge, commits and the new-branch
  push. Proceed without another plan approval, honoring that authorization.

- Ordinary merge `5b319107` records both immutable source tips; no production
  delta from fix. Native rebuild/import checks and preliminary 24 adapted
  tests passed. The final 21-case host carryover suite passed after independent
  review identified and added upstream degenerate preprojection coverage.
- Final validation runs in separate processes under the isolated native env;
  detailed commands, results and resource usage will accompany the report.

- Final selected checks: 462 top-level pytest cases, 48 additional direct rank
  executions and 3 CTests passed; no final skips/failures. Three baseline Boys
  failures and ten baseline weak-MPI fixture failures per rank were reproduced
  and repaired without production edits. Full audit/evidence committed alongside
  the validation report. Original installation and 1,472 native files unchanged.
- User-selected delivery is a normal push of only `dlno-stc-clean`; preserve the
  sibling worktree and both source branches. No PR or branch cleanup is requested.
