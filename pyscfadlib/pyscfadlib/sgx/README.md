# Portable generated SGX integrals

This optional engine evaluates Coulomb integrals between Gaussian shell
pairs and grid points, then contracts directly into SGX G arrays. It covers
s, p, d, f and g shells, including general contractions and Cartesian or real
spherical AOs. Version 2 also generates first spatial derivatives on both
shells, used by the complete geometry-response driver. Higher angular momentum
and modified Coulomb operators use libcint in `integral_backend='auto'` mode.
Version 3 adds optional density-dependent tile screening and mixed precision.
Version 4 adds independent per-pair screening with size-independent nominal cutoffs.
The original static-screened FP64 path remains the default accuracy reference.

The public precision switch is `mixed_precision`:

| Setting | Forward K | Density-response K | Geometry integrals |
| --- | --- | --- | --- |
| `False` | FP64 | FP64 | FP64 |
| `True` | Selected FP32/FP64 | Selected FP32/FP64 | Selected FP32/FP64, with faster FP64 promotions |

Screening is independent of this switch. Both precision modes can use the
same `screening='per_pair'` skip thresholds. Density-response K disables
density-dependent skipping in both modes. VRR/HRR selection is internal;
there is no separate user mode for each recurrence or an all-FP32 mode.

AO-density contractions pack the AO columns selected by the existing grid
mask into 896-point BLAS subpanels, retaining every density-output column.
Geometry contractions form only the intermediate columns needed by the AO
derivatives and fuse the atom/grid reductions. These subpanels do not change
the integral tiles, grid, screening thresholds, or outer block planner.
The native digest fuses the two symmetric value updates and all three spatial
derivatives while preserving each output's shell-pair accumulation order.
These layout changes preserve each buffer's precision and use no CPU-specific
instructions.

`SGX_contract_geometry` is an optional additive ABI-4 entry point. Python falls
back to compact NumPy reductions with older generated libraries. Floating-point
reduction order can change the last few digits; the mathematical contractions
and screening policy are unchanged.

The outer block planner counts the selected backend's actual jobs: 128 points
for generated kernels and 224 for libcint. Generated panels are multiples of
896 points, preserving both integral-tile boundaries and PySCF's screening/AO
mask alignment. Memory caps round down to this alignment; the final panel can
be shorter. The planner targets about four jobs per thread without changing
screening thresholds or the precision policy.

The 15 unordered angular-momentum pairs cover all 25 ordered pairs. Both ordinary
and same-center variants are provided: 30 vertical-recurrence kernels and 20
horizontal-recurrence kernels. Another 30 kernels combine values with all six
shell derivatives. The derivative generator uses a shared recurrence graph
within groups of at most 64 Cartesian AO pairs. Each group evaluates values
and all six derivatives together. The bound keeps the SIMD loops within
compiler dependency-analysis limits without compiler tuning flags. SymPy is
needed only to regenerate the original value kernels. Derivative generation,
compilation, and execution do not require SymPy.

The derivative sources occupy about 2.7 MiB. Splitting the largest graphs
recomputes some intermediates between groups; Boys values and primitive-pair
geometry remain shared. No additional integral tensor is retained.

## Build and use

Normal pyscfadlib builds include this library. It can also be built separately
from the repository root:

```sh
cmake -S pyscfadlib/pyscfadlib/sgx -B /tmp/pyscfad-sgx-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/pyscfad-sgx-build -j 4
```

Intel LLVM is also supported; select it in a separate build directory:

```sh
cmake -S pyscfadlib/pyscfadlib/sgx -B /tmp/pyscfad-sgx-icpx-build \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx
cmake --build /tmp/pyscfad-sgx-icpx-build -j 4
```

Intel LLVM builds explicitly use `-fp-model=precise`. No build enables
architecture-specific instructions by default. Compiler choice is independent
of the Python `integral_backend` setting.

Install `libsgx_ad.so` (or the platform equivalent) beside the other native
libraries in the active `pyscfadlib` package. Retain the accompanying
`codegen/upstream/LICENSE` when distributing the library; CMake installation
installs it as `pyscfadlib/licenses/snk_integral_codegen.LICENSE`.

```python
from pyscfad.sgx import sgx_fit
mf = sgx_fit(mf, integral_backend='generated')

# The two precision modes with identical per-pair skip settings.
# Keep these cutoffs fixed across system sizes:
settings = dict(integral_backend='generated', screening='per_pair',
                energy_tol=1e-10, potential_tol=1e-8, fp32_multiplier=1e5)
mf_fp64 = sgx_fit(mf, mixed_precision=False, **settings)
mf_mixed = sgx_fit(mf, mixed_precision=True, **settings)
```

`generated` requires the new library and supported integrals. The default `auto`
falls back to libcint when the library is unavailable, the operator/basis is
unsupported, or the primitive-pair metadata exceeds its memory allowance.
`libcint` explicitly selects the reference engine. The density-independent
shell-pair screen, grids, overlap correction and exchange operator are shared.
An older version 1 library still accelerates values and retains libcint for
integral derivatives.

### Adaptive screening and mixed precision

For `screening='adaptive'`, the `auto` energy budget is the SCF integral tolerance (normally 1e-13), and
the `auto` potential budget is its square root (approximately 3.16e-7). These
match the installed PySCF automatic screening tolerance values. Before this
adaptive implementation, PySCFAD explicitly disabled density-dependent energy
and potential screening; these were not active cutoffs in its reference path.
The adaptive driver reuses PySCF's nominal defaults with newly implemented
tile estimates and budget selection. It does not reproduce upstream PySCF's
sorted screening and AO-dependent potential-budget allocation.

For `screening='per_pair'`, auto means `energy_tol=1e-10` and
`potential_tol=1e-8`, the SNK paper's nominal K-timing parameters. Each
128-point tile tests each pair independently against thresholds
`tolerance * mean(abs(grid_weights)) * tile_points / 1024`. The mean is over
the complete retained molecular grid. There is no total-grid-size divisor;
the nominal cutoffs remain fixed as the system grows. Scaling by
`tile_points/1024` approximately accounts for the paper's larger screening
batches and is our convention, not Q-Chem's exact algorithm. Integral
estimates and the AO norm used by the local Q-Chem checkout also differ.
Equal parameters do not establish equal decisions or molecular accuracy.

For each shell and 128-point tile it computes the weighted L2 norm of the
Cartesian F entries. For multiple densities/adjoints it takes the maximum
shell norm across inputs, giving a common conservative selection. Let these
norms be f_A and f_B, let x be the weighted L2 norm of the sum of absolute AO
values, and let M_AB be the cached sampled integral estimate. The energy and
potential estimates are M_AB*f_A*f_B and M_AB*hypot(f_A,f_B)*x. Absolute grid
weights are used, including when the quadrature contains negative weights.

In cumulative (`adaptive`) mode, each tile receives a fraction of the energy/potential budgets proportional to
its number of grid points. Candidate contributions are grouped in logarithmic
magnitude buckets; the smallest are skipped first, with a partial final bucket
if needed. This linear-time selection prevents a few early pairs from consuming
the budget before much smaller pairs are considered. A pair is skipped only
when both cumulative estimates fit their budgets. This accounts for sums
of skipped contributions within a tile. The estimates are not rigorous bounds
for every basis, overlap-fit condition number, or force component.

Per-pair mode bypasses this bucket selection and cumulative accounting.
Both estimates must be below their individual thresholds to skip. FP32
selection uses these same thresholds times `fp32_multiplier`; changing the
multiplier does not change skipping. The shared density/adjoint union and
derivative guard remain active in both screening modes. Accumulated errors
can increase with system size and must be measured against a strict reference.

Density pullbacks, including implicit SCF response matvecs, use mixed exchange
when `mixed_precision=True`, with the same FP32 classification thresholds as
forward K but no density-dependent tile skipping. With mixed precision off,
they use static FP64. Coulomb density pullbacks disable the native density-dependent
prescreen while retaining integral metadata and the fitted metric. Absolute
cutoffs otherwise make the action on a tiny final solution inconsistent with
the action on normalized Krylov vectors, preventing true-residual convergence.
Forward J/K and geometry derivatives retain their selected accuracy policies.
Mixed response is approximately linear because of classification and FP32
rounding. There is no response-only precision override. For independent
reference/residual checks, construct a separate `sgx_fit` object with
`mixed_precision=False`. Enabling mixed precision does not automatically add
an extra FP64 operator application to each solve.

Pairs whose individual estimates lie below the tile budgets multiplied by
`fp32_multiplier` can use FP32. Geometry, primitive metadata and coordinate
differences are formed in FP64. Boys interpolation/recurrences, generated
integral recurrences, primitive contractions, and separate F/G/dG buffers use
FP32 on that path. The FP32 G/dG buffers are converted and added to FP64 once
per tile before spherical transformation and molecular reductions. There is
no ISA-specific code or relaxed-math requirement.
Exact power-of-two scaling (2^32 for F and integral seeds, 2^-64 on the merged
G/dG) reduces subnormal arithmetic without reducing significand precision.
Unsafe scaled inputs fall back to FP64. Boys F0 uses a rare FP64 square-root
fallback for arguments above 1e30, where the FP32 reciprocal can underflow
before F0 itself does.

For derivatives, both estimates are multiplied by
`max(1, 4*(l_A+l_B+2)*sqrt(max(alpha,beta)))` to account heuristically for
raised/lowered integrals and their cancellation. This is not a mathematical
force-error bound. The shared value/derivative traversal uses this stricter
criterion for both outputs. AO, grid-motion, quadrature-weight and overlap-fit
derivatives are still evaluated by the existing response driver.

Density-dependent selection makes K approximately linear: different trial
densities can cross a cutoff and select different integrals. Derivatives
neglect changes of the discrete selection. FP32 rounding adds another small
departure from exact linearity. Check signed-density additivity, adjoint
consistency, implicit-solver convergence and the final coordinate gradient
against `screening='static', mixed_precision=False`. A force tolerance applies
to the final tested force difference, not to each discarded integral. When
tuning precision, keep `energy_tol` and `potential_tol` fixed and vary only
`fp32_multiplier`: increasing it routes more retained contributions through
FP32 without changing the skipped pairs. A system-specific target such as
1e-12 Eh/Bohr requires a comparison with FP64 at the same geometry and density,
plus sufficiently converged SCF/response solves when comparing total forces.

The native plan exposes `screening_stats()` with counts of skipped, FP32 and
FP64 shell-pair tiles and primitive/grid points; `reset_stats()` clears them.
The planner includes both precision workspaces in its memory allowance.

## Implementation

Grid loops use ordinary C++ and OpenMP SIMD directives, with optional OpenMP
threading. There are no ISA intrinsics, fixed SIMD widths, `-march=native`,
AVX requirements or fast-math flags. The thread count is passed explicitly from
PySCF, including when the two libraries use distinct OpenMP runtimes.

Each 128-point tile transforms F to Cartesian AOs once, evaluates shell pairs,
contracts each small integral tile into G, and transforms G back once. Only
primitive-pair metadata is cached across builds; no full grid integral tensor
is retained. The live grid planner reserves per-thread native scratch.

Boys functions use a long-double-generated table, eighth-degree interpolation,
downward recurrence, and a large-argument asymptote. Table gathers are separated
from polynomial arithmetic in reusable thread-local scratch. Mixed tiles pack
large arguments before vectorized asymptotic evaluation and scatter the values
back afterward. Table lookup, lane classification and scatter can remain scalar
on baseline targets; interpolation and recurrence arithmetic use SIMD. The
native scratch estimate includes the Boys workspace. Recurrences reuse
reciprocals across grid points or orders instead of repeatedly dividing.
The VRR/HRR selector uses angular momentum, primitive-pair count and coincident
centers, with crossover measurements from both GCC and Intel LLVM. It retains
same-center VRR specializations for single primitive pairs and selects HRR
for sufficiently high-angular-momentum distinct centers even without
contraction. General contractions reuse Boys values and primitive geometry
across coefficient columns.

FP32 forward HRR additionally covers distinct-center d–d and f–p pairs with
one primitive pair. FP64 derivatives use a separate conservative crossover
table in `prefer_derivative_hrr`: same-center p–p/d–p/f–p start at three
primitive pairs; distinct-center f–f and g–d/g–f/g–g can use HRR without
contraction. Low-angular and shallow contractions otherwise retain VRR.
Only derivative scratch reserves the additional HRR base storage.

For mixed derivatives, precision classification sets the minimum accuracy.
FP32-eligible work may execute FP64 HRR when the cross-precision table in
`prefer_fp64_for_mixed_derivative` predicts a portable speed advantage.
For example, distinct-center f–f switches at 16 primitive pairs and g–g at
four; same-center crossovers differ. The table requires a measured advantage
of at least 10% in both GCC and Intel LLVM. FP64 VRR showed no portable wins
over FP32 VRR in these measurements. Remaining eligible work uses FP32 VRR.
Performance selection runs after skipping, never demotes FP64-required work,
and leaves value-only forward/response-K selection unchanged. Statistics count
the precision actually executed; promoted work contributes to FP64 counters.
The tested water cc-pVTZ shells do not trigger these precision promotions.

Derivative HRR contracts ordinary, 2-alpha-weighted and 2-beta-weighted base
integrals separately before translation; applying the exponent weights after
contraction would be wrong. Derivatives executed in FP32 retain VRR: high-order base errors
can be amplified by cancellation in HRR, even for a single primitive pair.
Promoting only the final HRR translation to FP64 does not repair those errors.
Screening and mixed-precision classification thresholds are unchanged.

For geometry response, `contract_with_grad` shares Boys evaluation and vertical
recurrence intermediates between values and both shell derivatives, contracts
them immediately, and returns G and its bra spatial derivative. It matches
libcint's `int1e_grids_ip` sign convention. The Python response driver supplies
the nuclear-center minus sign, AO derivatives, moving-grid terms, partition
weight derivatives and overlap-fit response. Those terms are all retained.

The spatial derivative identity is formed using the original primitive
exponents and normalization before specializing same-center integrals. Thus
independent derivatives remain correct when two centers coincide. A g shell
uses internal h intermediates and Boys functions through order nine. Input
h basis shells and higher derivatives are not supported by this backend.

## Regenerate and validate

The original value generator is vendored from Jesse DiCenso's MIT-licensed
[snk_integral_codegen](https://github.com/jesse-dicenso/snk_integral_codegen).
Its original files are unmodified; `generated/manifest.json` records their
SHA-256 hashes and the generated kernel inventory. The wrapper retains only
the FP64 instantiations and places them in a private namespace.
The first-derivative generator is implemented locally in
`codegen/generate_derivatives.py`; `generated/derivatives_manifest.json` records
its source hash and kernel inventory.
`codegen/generate_float.py` generates FP32 siblings in a separate namespace
from those same graphs, including float constants to avoid implicit promotion.
`codegen/generate_hrr_derivatives.py` generates the additional FP64-only
contracted derivative HRR kernels and their base-storage inventory.

With SymPy 1.13.3 available:

```sh
python pyscfadlib/pyscfadlib/sgx/codegen/generate.py --jobs 2
python pyscfadlib/pyscfadlib/sgx/codegen/generate_derivatives.py
python pyscfadlib/pyscfadlib/sgx/codegen/generate_hrr_derivatives.py
python pyscfadlib/pyscfadlib/sgx/codegen/generate_float.py
pytest tests/test_sgx_native.py tests/test_sgx.py tests/test_sgx_shared.py tests/test_sgx_primal_cache.py
pytest tests/test_sgx_mixed.py
```

Validation compares every ordered s–g pair with libcint, on shared and distinct
centers, for both recurrence methods and multiple contraction columns. It also
checks Boys-function boundaries, spherical/Cartesian fused contractions, full
K builds, linearity, self-adjointness and the existing coordinate-response tests.
Derivative tests compare both shell derivatives for every ordered s–g pair with
libcint, check fused spherical/Cartesian contractions, and verify independently
moving coincident g centers and moving grid points against finite differences.
