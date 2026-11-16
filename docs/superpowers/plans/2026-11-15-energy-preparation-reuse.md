# Deterministic domain MP2 preparation reuse

Preserve domain selections, numerical energies, gradient response, and the QR
anchor step. Optimize the existing single-Boys deterministic LT-MP2 workflow.

1. Retain the numerical selection result and screened weak correction in a
   bounded, transient receipt. Reuse only for the exact unchanged reference and
   selections; changed references, restarts, and gradients retain live replay.
2. Project occupied and raw virtual columns through the same ED overlap solve.
   Reuse local Gram contractions without changing rank choices or derivatives.
3. Default energy-only domain DF preparation to direct transformation, honoring
   explicit cache-mode overrides. Limit BLAS to one thread inside deterministic
   energy kernels while preserving the surrounding OpenMP budget and restoring
   limits on errors. Keep gradient defaults.
4. Verify mutation invalidation, weak weights, repeated kernels, saved and pending
   ED replay, derivative response, native thread restoration, and DF overrides.
5. Review remaining redundant work. Run three cold C20 energy jobs with the saved
   reference/grid and compare with the preserved implementation and MRCC. Run
   water4 full-domain energy/gradient checks against independent RI-MP2 and finite
   differences. Record source provenance and construction counts.

The existing checkout contains approved domain changes and is retained. No source
commit, external publishing, or new domain construction in the pullback is needed.
