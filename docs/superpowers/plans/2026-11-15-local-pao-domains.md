# Population-ordered occupied domains and local ED PAOs

User-approved scope: implement the two previously tested domain changes, compare
C20 domain MP2 with the archived implementation and MRCC, and validate water-4
MP2 energy and nuclear gradients. Preserve unrelated thresholds and PAO BP growth.

1. Preserve baseline source at commit 1a223de6e8d8065c54500e0a9e6380b3e7a708ef.
2. Add focused regression tests before implementing population-ordered occupied
   BP growth. Keep distance ordering as the helper default for PAO domains.
3. Add typed ED selection metadata and select normalized PAO parents locally,
   applying completeness, occupied projection, and one final Gram rank cutoff.
   Replay saved selections and reuse the forward VJP tape; never reconstruct
   domain topology in the reverse pass. Preserve legacy saved-frame replay.
4. Integrate grouping and checkpoint serialization, invalidating stale response
   checkpoints when equations change. Test derivatives, replay, and call counts.
5. Run relevant DLNO_STC regression tests and obtain independent code review.
6. Benchmark archived/current C20 with identical fixtures, quadrature, CPU/thread
   limits and thresholds; use MRCC restart inputs that skip only SCF/localization.
7. Run water-4 cc-pVTZ/JK-fit/RI-fit, frozen oxygen cores, full-domain MP2 total
   energy and gradient, independently checked against canonical DF-MP2 and
   nuclear finite differences. Record errors, timings and resource limitations.

No implementation commits, external publication, or unrelated source changes.
