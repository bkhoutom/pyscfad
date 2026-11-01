# Generic SCF geometry transpose without AO-adjoint spill

Scope: isolated candidate only, generic eager get_jk_bwd dispatch and new test_reverse_generic_geometry.py. Live package and frozen dCMP snapshots untouched. No numerical work on login nodes.

## Algebra and intended dispatch

For validated real D = S S.T, the geometry derivative holding D fixed equals the geometry derivative holding S fixed. This does not imply that Sbar can recover independent AO-density Dbar. Preserve the existing generic density-only transpose for that result, including virtual-virtual and nonsymmetric-seed contributions. Use existing factorized reverse with factor_response=False solely for molecular/auxiliary coordinates. Rebuild the returned DF cotangent tree with None for inactive leaves.

Eligibility is conservative: first-order concrete coordinate-only mol and aux trees, real finite float64 single density and active output bars, hermi=1, disk placeholder, no physical CDERI cotangent, no supplied coordinate callback/MPI context, occupied-VJP switch enabled, validated positive thin rank. Require nao*rank < nao*(nao+1)/2, so thin metric solves actually use fewer columns than the generic packed solve. Unsupported cases retain the generic path. No global cache or deferred adjoint is introduced.

Tradeoff: the density and geometry branches each stream DF once, replacing the old fused pass plus full AO-adjoint file and pair-column metric solves. Measure dCMP performance rather than assume it is faster.

## Tests prepared first

New file has 11 cases:
- J/K/both with two arbitrary nonsymmetric cotangents plus repeated first cotangent; molecular, auxiliary, and full AO-density bars match independent dense Cholesky plus explicit tensor contractions. Require a nonzero density response outside occupied support. Assert no full AO-factor adjoint spill after numerical checks.
- Geometry-only mol/aux/both activity retains absent density, CDERI and inactive coordinate bars, with independent numerical references and no spill.
- Disabled switch, asymmetric or indefinite density, and supplied coordinate callback forbid factorized dispatch and retain dense-reference results.
- Stale orbital tags cannot change coordinates or density bars; validated factors may be recovered from the actual density.

Dense oracle accepts both packed and full s1 factor layouts, since higher tracer levels may select s1 integrals. AST parsing passed. Parent submitted frozen red job 5822122 before the layout robustness adjustment; pending compute evidence. No production implementation before red evidence.

## Red evidence and implementation

Parent reported normal-QoS compute job 5822122: exactly six expected no-spill failures after all independent numerical comparisons passed; five fallback/stale-tag cases passed. Parent then authorized implementation.

Added `_factorized_geometry_bwd` and one dispatch site after traced handling and before generic allocations. The helper checks the existing occupied-VJP switch, disk-placeholder status, inactive MPI hook, real finite float64 single density/output bars, coordinate-only float64 molecular leaves, and validated rank whose metric-column count is lower than packed AO pairs. Parent's custom-coordinate callback and implicit-solve matvec bypass the helper entirely. Density activity invokes the existing generic transpose with all geometry/factor activity false; geometry uses existing factorized `_backward(..., factor_response=False)`. Only active molecular/auxiliary bars are copied into a None-filled DF tree. New profiles report density, geometry and total times.

Candidate preconditioner function remains byte-identical to frozen first-candidate benchmark snapshot. Static AST passed. Existing implicit STO-3G water instrumentation remains unchanged: occupied rank five in seven AOs does not satisfy the thin-column criterion and correctly falls back. No claimed numerical or performance success until parent green compute and full dCMP results.
