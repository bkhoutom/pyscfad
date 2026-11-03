# W baseline reuse implementation plan

> Superseded by the `dlno-stc-fix` merge: W supplies the domain MP2 energy
> directly from its traces in deterministic and stochastic modes, including
> fragments without CC corrections. No separate W baseline solve remains;
> mismatched Laplace grids are rejected. The plan below records the earlier
> implementation.

> Execute the accepted optimization with regression tests and an independent review.

**Goal:** Avoid a separate deterministic MP2 baseline calculation when an existing
W calculation supplies the same selected energy and response.

**Design:** In the shared CCSD group driver, reuse W only when both modes are
deterministic and the Laplace roots and weights match exactly. Assign whole
executed fragment target sets to the baseline once; evaluate remaining baseline
targets separately. Add each reused energy once and seed its existing matrix
pullback with one. Preserve the LIS MP2 subtraction and stochastic controls.

**Files:** `pyscfad/dlno_stc/ccsd.py`, its shared-pullback tests, and
`pyscfad/dlno_stc/README.md`. No native rebuild or public API change is needed.

## Checks and implementation

- [x] Run existing shared-pullback tests, then demonstrate failure of new reuse tests.
- [x] Partition covered/uncovered targets and accumulate reused energy/response.
- [x] Verify energy and gradient parity against the independent baseline, including
  finite differences, partial/empty selections, and correction-only calculations.
- [x] Check D, stochastic, mismatched-grid, and unmerged multi-target fallbacks.
- [x] Update documentation, run relevant regressions, and obtain independent review.

The change is applied to the existing clean `dlno-stc` checkout requested by the
user. Native matrices and per-fragment replay records remain local to each group;
serial checkpoints and MPI reduction retain the same accumulated fields.

Native-only baseline sampling options retain the original solve for validation.
Regression tests cover invalid ordinary options and native integer bounds;
the independent review's signed-int64 boundary finding was fixed and verified.

Verification: 196 focused/related regression tests and all 5 molecular CC MPI
tests passed. MPI required local socket access outside the restricted sandbox.
The final independent review had no outstanding findings. No native source or
binary was changed.
