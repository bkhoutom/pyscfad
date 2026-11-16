# MRCC PAO alignment implementation plan

User-authorized scope: use initial PAO squared norm >1e-4 and omit ED occupied removal from individual ED PAOs; compare C20 deterministic domain MP2 against the previous commit and MRCC; validate water-4 MP2 energies and gradients.

- [x] Preserve existing dirty changes and committed baseline. Work in the requested source tree; independent benchmark snapshots supply isolation.
- [x] Add failing scientific tests for norm convention, retained ED occupied overlap, and finite-domain pullback. Implement the minimal ED change without constructing domains again in reverse.
- [x] Maintain orthogonal LIS input to CC after the ED change, with matching derivatives; invalidate incompatible saved response state.
- [x] Run relevant domain, LIS, replay, and gradient regressions with bounded resources.
- [x] Benchmark C20 cc-pVDZ, JKFIT/RI, singleton Boys, deterministic LT MP2 with common timing boundaries, serialized repeats, and explicit prior-commit provenance. Record PD/ED dimensions, canonical errors and total process wall time.
- [x] Run water-4 cc-pVTZ full-domain MP2 energy/gradient versus independent reference and finite-domain fixed-selection gradient checks, preserving forward tapes without reverse reconstruction.
- [x] Independently review the scientific changes and report measured results and limitations.

No commits, source resets, or external publication. Existing working changes must be retained. Heavy jobs run one at a time under the existing memory/CPU caps. Changing the parameter's numeric default to pao_norm=1e-2 preserves its public norm convention while implementing squared norm >1e-4.

Validation note: the optional aggressive ed_pao=.90 water profile was rejected for negative gaps both with and without occupied projection. Keeping the same finite-domain profile and using production ed_pao=.995 passed the gradient test; all failed diagnostics remain archived.
