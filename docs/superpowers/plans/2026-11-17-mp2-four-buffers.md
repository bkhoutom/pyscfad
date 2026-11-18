# Four persistent MP2 factor buffers

Implement on the current `dlno-stc` branch, as requested. The user approved
temporary layout-conversion overlap and four persistent B-sized tensors.

## Design and constraints

Standalone deterministic domain/system MP2 will retain one host-owned fitted
factor in native `(P, i*a)` column-major storage. Its `(P,i,a)` NumPy view is
shared by preparation, the analytic DF pullback, and the native solver. The
other three full factors are dressed C, its adjoint, and the accumulated B
adjoint. Small frame/Fock/molecule transformations retain their JAX VJPs;
the integral factor itself stays outside those tapes. No caller-owned array
is deleted or modified. Existing differentiable preparation, W/CC and
stochastic APIs remain available. H caching, thread settings and integral
workspace policy are unchanged. Four factors is not a bound on total RSS.

## Tasks

1. Add failing native storage-sharing and numerical tests, plus host
   preparation/analytic-pullback equivalence tests for direct and half modes.
2. Add a validated native-layout scalar interface, borrow the original factor
   in the deterministic kernel, and remove the unused output preallocation.
   Preserve public C-layout input/output behavior and stochastic ownership.
3. Prepare a host factor with an explicit analytic pullback; route standalone
   deterministic MP2 through it for saved and deferred domains, serial/MPI.
   Share the factor with half-cache entries when that mode is selected.
4. Rebuild the extension; run native and molecular regressions, a scaled
   memory measurement, and review the complete diff. Document the scope and
   measured checks. Leave changes on the requested branch without pushing.

## Review focus

Lifetime of borrowed arrays across GIL release and exceptions; read-only
inputs; empty virtual/auxiliary dimensions; layout validation; finite-domain
and distinct auxiliary supports; direct and resident-H modes; deferred
domain selection; gradients through the metric, orbitals, and coordinates;
compatibility of public C-layout and stochastic/W callers. Distinguish live
factor storage from allocator retention and other native work arrays.

## Validation record

Baseline: `46fa0f34`. The extension was rebuilt with the existing CMake
configuration using Python 3.11 in `pyscfad_env`.

The standalone RSS probe used `(naux,nocc,nvir)=(2048,32,256)`, two selected
occupied targets, two Laplace points and two OpenMP threads. Each factor
was 128 MiB. Fresh processes compared the baseline extension with two
preparation factors against the new native layout with one shared factor:

| Measurement | Baseline | Shared factor |
| --- | ---: | ---: |
| Peak RSS, MiB | 1160.953 | 648.035 |
| Scalar solve wall time, seconds | 3.825 | 3.395 |
| MP2 energy, Eh | -0.33462119438568705 | -0.33462119438568705 |
| B-adjoint norm | 0.2996941628728438 | 0.2996941628728434 |

The 512.918 MiB RSS decrease agrees with four removed 128 MiB factors.
This is a synthetic factor/solver measurement, excluding SCF, domain
selection, and AO-integral response; it is not a production molecular peak
or a general runtime benchmark. For the previously discussed C66 domain,
four 5.83584 GiB factors correspond to 23.34335 GiB less live factor storage.

A fresh-context review checked the native ownership and Python integration.
The resulting fixes preserve the zero-response fast path for empty orbital
pairs, the unshared forward metric eigenvalue fallback, valid automatic
auxiliary offsets for zero auxiliary functions, and the correct auxiliary
support in the domain memory forecast. Each received a failing regression
before its fix. Existing lightweight test fixtures were updated with the
auxiliary-support field used by real fragment selections.

Focused verification covers independent native energy/derivative formulas,
full molecular coordinate finite differences, saved/deferred preparation,
direct/resident-H and unshared DF, separate correlation fitting bases,
empty/singleton and read-only factors, returned-adjoint ownership,
stochastic compatibility, W/CC paths, and MPI domain ownership/response.

Final result: 211 unique Python/MPI tests passed (115 DF/storage/memory,
64 scalar/molecular/metadata/cleanup, 22 calculation-mode/W, 10 MPI tests),
plus all three native CTest checks. One local DF test was deselected by the
repository default. The final MPI suite passed after the review fixes,
including serial/1/2/4-rank energy and response agreement. Native source
hashes and `git diff --check` passed. Implemented on `dlno-stc`.
The local Python 3.11 extension has been rebuilt; other Python environments
and HPC installations must rebuild their native extension when taking these
source changes.
