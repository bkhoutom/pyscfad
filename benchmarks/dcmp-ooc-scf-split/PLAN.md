# Separate SCF and DF reverse timings

User-authorized new optimized OOC dCMP measurement. No new optimization or live package changes.

## Design

Use a frozen copy of the currently installed optimized package, the existing dCMP geometry/driver, 16 physical Ice Lake cores and 128 GB, existing thread settings and scratch. All calculations run through SLURM from a self-contained `dCMP-ooc/test_reverse/scf_df_split` condition.

The initial factorized J/K reverse loop computes orbital response and a compressed fitted-factor adjoint. Keep it in SCF time. The later metric reconstruction, triangular transforms, raw-integral coordinate contractions and metric pullback map that adjoint to geometry; measure the inclusive interval as the fused DF-construction pullback. Synchronize its coordinate outputs before ending the timer. Record its metric/transform, density-setup, coordinate-contraction and metric-reverse substages.

AO-adjoint assembly is fused into this DF region, unlike dense in-core materialization. Record coordinate-helper density/packing time separately. Report both the operational remainder (SCF total minus full fused DF region) and a conservative assignment that keeps density setup/assembly/packing in SCF. Do not silently present these representations as identical mathematical boundaries.

Phase-tag the timing ledger; exclude MP2 events from SCF accounting. Record generic fallback DF pullbacks with nesting protection; fail if measured OOC SCF coverage is absent or inconsistent. Never subtract nested timers twice. Retain direct molecular response, frozen orbitals and all 102 gradient components.

## Work

- [x] Inspect and independently audit the timing boundary.
- [x] Capture the current package and saved full-gradient reference.
- [x] Add only timing hooks and synchronization to the isolated factorized/coordinate-helper code.
- [x] Add driver phase tagging, response/preconditioner timing, complete result records and numerical/reference checks.
- [x] Review exact diffs, syntax and accounting; submit a static self-contained SLURM job.
- [x] Wait for completion, verify energy/all gradients and scheduler status, calculate the comparison and remaining cost breakdown.
- [x] Save a readable report; confirm live source remains unchanged.

No numerical imports/tests/calculations on login nodes. No Git operations in submission scripts, no submission manager, no commits. Existing user authorization covers the benchmark and necessary instrumentation; no additional design approval is required. The full dCMP gradient check and accounting assertions validate this measurement-only change; do not run an unrelated full package test suite.

Submitted job 5824643 from `dCMP-ooc/test_reverse/scf_df_split` after independent read-only review. All 418 live Python source hashes still match the captured baseline; recorded scientific sources match optimized OOC reference job 5822463. Only the three DF modules in the isolated snapshot differ, by instrumentation.

Completed 5824643 (0:0). Timing/coverage checks passed; all 102 gradient components agree within 4.78e-12 Eh/bohr. Final 418-file live-source hash check passed. See README.md and exact result JSON.
