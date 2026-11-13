# Joint DF rotations in retained LIS spaces

Joint DF rotations provide the default private sampling frame for STC CCSD
inside retained LIS spaces. The implementation and its physical responses are
validated on water4.

## Design

For STC CCSD, default to Boys initialization followed by the upstream DFAll
occupied/virtual/auxiliary optimization on the already SVD-compressed LIS Lov.
Compose orbital rotations and apply the auxiliary rotation to all three DF
blocks. Rotate the Fock matrix, initial amplitudes and physical fragment
projector consistently. CCSD and smooth triples share the resulting frame.

`df_rotations=True` and `df_rotation_tol=5e-3` are adapter controls;
`df_rotations=False` restores Boys-only behavior. `localize=False` disables
both private localizations. These options do not change physical domains or
retained ranks and do not change the overall deterministic CC solver default.

The numerical optimizer is outside AD. Its orthogonal matrices are fixed in
the pullback; physical factors, Fock, projector, LIS selection response and
SCF remain differentiable. No localization-response equation is necessary.
Use an evaluation-scoped preparation object to reuse rotations for replicas
of a fixed physical LIS, never a persistent dimension- or fragment-key cache.

## Implementation and validation

1. Add failing controls, joint tensor/energy/response covariance, and reuse
   tests. Include a nontrivial physical projector and SVD-reduced auxiliary
   dimension. Validate invalid controls and the zero-factor edge case.
2. Add joint numerical rotation preparation, consistent live transformations,
   replica-scoped reuse and diagnostics; preserve legacy controls.
3. Update adapter documentation. Run focused tests for controls, rotations,
   shared frames, compression and replica routing.
4. Build a compatible Python 3.12 native environment if needed; run water4
   checks with real upstream DFAll and STC, documenting basis, tolerances,
   gradient tests and stochastic limitations. Save reproducible artifacts.
5. Independently review the diff and verify final results before reporting.

Finite stochastic runs in different frames are not identical realizations.
Use deterministic tensor/functional covariance to test exact invariance and
actual STC runs to test integration, reporting sampling uncertainty honestly.

## Completed validation

- New tests first failed for missing controls, rotations and reuse routing.
- Final focused suite: 88 passed, including actual upstream forward/adjoint,
  projected CCSD/(T) covariance, auxiliary compression responses and replica
  cache lifetime. The independent code review found no blocking issue.
- Water4/STO-3G, frozen=4, four physical LIS, auxiliary SVD cutoff=1e-5:
  STC CCSD energy error -9.04e-7 Eh and max nuclear-gradient error 2.74e-5
  Eh/bohr; CCSD+(T) errors -7.73e-7 Eh and 2.52e-5 Eh/bohr versus the
  corresponding deterministic calculations. These integration checks sample
  one STC contraction. All 36 gradient components were compared.
- Real optimized compressed-LIS CCSD/(T) covariance: energy differences below
  3.5e-17 Eh and physical-input cotangent differences below 2.2e-16.
- Each complete four-fragment energy/gradient calculation performed exactly
  four sampling-frame optimizations and four local ERI preparations. Precision
  replica routing/call reuse is additionally covered by regression tests.
- Water4/cc-pVTZ, atom grouping, four CPU threads, all 23 contractions sampled:
  with a 0.4 mEh update-noise target and 0.2 mEh local stopping SE, DF rotations
  on/off took 312.46/347.14 s for the complete energy and gradient. CC forward
  times were 47.53/57.00 s and complete reverse times were 132.02/156.79 s.
  Both runs met the stopping SE in every LIS. This is one matched run per
  setting (seed 731), with first-use compilation included, not a general
  speed or force-accuracy guarantee. Boys localization remained enabled
  with DF rotations off.
- Six broader MP2 controls tests fail on a stale None-valued domain fixture;
  an independently archived, untouched HEAD reproduces all six failures.
  They were not modified as part of this feature.

Scripts, JSON results, regression and baseline logs are retained in the parent
workspace at `TEST_general/dlno_stc_test/df_rotations_water4/`. The compatible
environment is `stc-cc-jax/work/venv-py312` with PySCF 2.14 and a newly built
CPython 3.12 DLNO extension. Physical LIS/SVD preparation remains per replica;
only the new numerical sampling-frame optimization is shared across replicas.
