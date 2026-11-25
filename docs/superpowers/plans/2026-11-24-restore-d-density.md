# Restore D density implementation plan

**Goal:** Restore selectable D density in the current DLNO-STC workflow, keep W
as the default, and validate the matrices, energy accounting and gradients on
water tetramer.

**Design:** Restore the historical native D density and explicit MP2 scalar
contractions and their combined delayed pullback. Use the current fragment
preparation, smooth LIS construction, shared response and CC correction path.
Unlike W, D does not determine the MP2 energy by its trace. A separate driver
baseline would repeat preparation; the native D provider already computes the
needed scalar with the same amplitudes and quadrature.

**Interfaces:** `ccsd.build_static_selections(..., matrix_kind="W")` records the
choice in `LISSelections`; `backend.solve_matrices(..., matrix_kind="W")` passes
it to the native provider. Accept D and W only. Restore D stochastic energy
precision controls. Preserve old W checkpoint digests by omitting the default
field from serialization. Reject D with the experimental sequential virtual W
algorithm. Density thresholds are used as supplied; D and W thresholds have
different meanings and are not converted automatically.

## Tasks

- [x] Restore native deterministic/stochastic D output, explicit scalar energy,
  and arbitrary-seed reverse; verify against independent contractions and FD.
- [x] Restore Python controls, memory estimates, diagnostics and replay identity;
  verify D/W defaults, batching and invalid replay behavior.
- [x] Integrate density selection into CC, restart identity and the benchmark
  CLI; verify combined energy/selection response and W compatibility.
- [x] Add water4 correctness tests and run molecular energy/gradient checks,
  relevant existing regressions and native tests; obtain an independent review.

## Review focus

Explicit D energy must never come from its trace. Energy and density responses
must enter preparation once. Matrix kind must survive checkpoint round trips
and prevent mismatched replay. W defaults and existing numerical behavior must
remain unchanged. Finite-cutoff tests must exercise D eigenspace response.

Work is performed in the user-requested checkout. The pre-existing deletion of
`submit.sh` belongs to the user and is outside this change.

## Validation and review

All six water4 cases passed, including cc-pVDZ canonical D matrix comparison,
untruncated D/W CCSD gradients and truncated D CCSD(T) finite differences.
See `../reports/2026-11-24-d-density-validation.md` for numerical results and
commands. Focused Python and native regression suites passed.

Independent review found three issues, all reproduced and fixed with tests:
benchmark output reuse omitted density identity; the native version bump
invalidated unchanged W progress checkpoints; and sampled D scalar-energy
reverse buffers were missing from the memory forecast. W checkpoint identity
is preserved only for the exact compatible version transition. D reverse
forecasts now use immutable native replay allocation counts, and pilot
diagnostics include the scalar energy sampler.
