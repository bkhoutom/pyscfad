# MPI efficiency follow-up

Audit recent serial changes against STC, native DLNO/LNO, and optional
distributed DF/SCF execution. Preserve the preceding root-response,
single-pass, and opt-in SCF changes already in the working tree.

- [x] Check common serial-kernel reuse and option propagation for MPI owners.
- [x] Configure CC worker executable caches before worker JAX preparation and
  avoid allocating root cotangent buffers twice.
- [x] Match MPI density tagging to serial validation/low-rank exchange;
  reuse per-call response scratch and remove redundant full-source hashing
  without weakening collective source verification.
- [x] Assess discarded preparation tracing in native DLNO MP2 MPI. Retain the
  existing traced primal: eager evaluation changes frame values at roundoff
  and fails the exact replay/gauge guard. Document that requirement in code.
- [x] Assert owner-local integral/cache work counts as well as energy and
  gradient equality; run targeted tests and obtain independent review.

MPI communication, load balance, and source-file assembly remain real costs;
kernel parity alone cannot guarantee equal wall time for every workload.
Larger distributed response cache/geometry changes need explicit lifetime,
partition and memory-budget proofs before adoption.

The optional distributed SCF path still lacks occupied-panel caching across
response iterations and factorized geometry reverse. Default root-owned MPI
SCF uses the optimized serial response. Legacy DLNO CC MPI also closes CC and
MP2 common responses separately; changing that requires coordinated checkpoint
and lifetime changes. These are documented architectural differences, not
claims of equivalent end-to-end scaling.

## Verification

78 distinct targeted tests passed:

- 65 combined unit/regression cases: MPI MP2/CC scheduler ownership,
  compilation-cache controls, DF density validation and compact factors,
  response scratch reuse, and full source verification before contraction.
- 5 existing DF/service cases: water forward, density transpose and coordinate
  VJP; real two-rank source corruption and worker setup failure.
- 2 existing legacy DLNO MP2 cases: serial energy/gradient agreement and
  resource profiling after retaining the original traced preparation.
- 6 integration cases: single-pass CC energies/gradients with multiple owners
  and an idle rank, opt-in distributed SCF, invalid selection/control rejection,
  and root-owned MP2/CC out-of-core response.

The single-pass default MPI energy and gradient matched serial exactly in
these fixtures; the optional distributed-SCF energy differed by 8.33e-17.
Every tested domain built raw integrals and the metric once, reused the half
transform for its LIS, and released the cache. Root response combined three
consumers into one geometry pass and reduced occupied transforms from six to
one with five cache hits.

Two independent reviews found no production defects. A test-environment
finding was fixed by explicitly selecting half-cache mode and its allowance
in the launcher; the gradient/idle-rank case passed again with conflicting
parent settings (`direct`, zero cache allowance). `git diff --check` passed.
No C20 wall-time or multi-node scaling benchmark was run.
