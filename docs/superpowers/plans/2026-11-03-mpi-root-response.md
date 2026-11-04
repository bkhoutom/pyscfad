# MPI root response and single-pass selection

The requested behavior is optimized root-owned SCF/DF response by default,
independent MPI fragment work, and owner-local single-pass W/LIS selection.
Distributed SCF remains available only through `parallel_scf_jk=True`.

Implementation uses the existing serial response scopes whenever no distributed
SCF executor is active. It preserves explicit distributed response, numerical
equations, memory limits, caller-owned reverse scopes, and root-only results.
MPI owners select each fragment from the W intermediates they already build;
the resulting local VJP contributes to the existing cotangent reduction.

- [x] Add regressions for default root SCF, explicit distributed opt-in, and
  numerical root-response fusion/caching under a communicator.
- [x] Enable response producer capture and the complete reverse wrapper for
  root-owned execution, including one-rank communicators and system scope.
- [x] Change public/internal STC defaults and update explicit-service tests and
  documentation.
- [x] Validate deferred MPI selections, partial fragment ownership, empty
  owners, energy/gradient parity, and one W selection per owned fragment.
- [x] Run focused unit and molecular MPI checks, review the combined diff,
  and report verification limits. No scheduler jobs or C20 production run.

Known baseline: the existing full CC MPI smoke stops in its serial reference
at the local-DF memory guard. Diagnose it without weakening production guards.

The baseline diagnosis measured 3275.95 MB process RSS against the smoke's
3000 MB budget. Its explicit test budget is now 10000 MB, matching the existing
molecular response fixtures; production memory/block rules are unchanged.

Verification: 73 distinct focused tests passed: 15 defaults/API,
36 response/reduction/LIS, 16 existing MPI acceptance, four single-pass MPI,
and two OOC root-response MPI tests. The final three-test integration run
passed in 96.95 seconds, combining single-pass with OOC response fusion and
explicit distributed SCF. Independent review found no production correctness
issues. MPI validation used one, two, three, and four local ranks; no C20
production timing or scheduler job was run.
