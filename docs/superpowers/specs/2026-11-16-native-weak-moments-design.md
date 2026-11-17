# Native weak-endpoint moments

The user approved integrating, testing, committing, and pushing the native
BLAS/fixed-size implementation described in the preceding local experiment
report. The target is `origin/dlno-stc`; work continues in the existing isolated
`codex/omol3tdx-llvm` worktree, including its two primary-bucketing commits.

The failure is growth of LLVM executable mappings as physical endpoint sizes
vary. Keep physical orbital selections and scientific thresholds unchanged.
For concrete CPU order-4 endpoints, contract packed AO moments with native
BLAS, translate moments in fixed 128-column JAX tiles, and supply exact input
pullbacks. Keep host cropping and raw-moment adjoint accumulation outside
shape-specialized JAX operations. Share the existing translation algebra.

The experiment compared exact-size JAX, geometric buckets, maximum-size
padding, fixed tiles, and native BLAS with fixed translation tiles. The user
selected the last approach after its mapping and numerical checks, and its
lower endpoint runtime. Maximum padding wastes work; geometric buckets still
introduce compiled variants at larger sizes.

Preserve the original JAX path for tracing, higher derivatives, unsupported
moment layouts/dtypes/devices, and lower multipole orders. A concrete native
linearization must also fall back if its eventual cotangent becomes traced.
Both grouped and individual endpoint replay must preserve molecular and
orbital responses. No cache clearing, map-budget policy, new dependency, or
scientific approximation is introduced.

This change bounds shape-dependent compilation in endpoint moment operations,
not every phase of a full large-system calculation. Pair-energy and remaining
primary/strong-domain kernels retain their existing behavior. Dense
eigensystems are not split into independent tiles.

Acceptance: agreement of all endpoint leaves and input cotangents against the
existing JAX reference (including complex nonsymmetric inputs and tile edges),
working traced fallback, unchanged complete small-molecule MP2 gradients, and
no additional endpoint executable compilations as new dimensions are added
after warm-up. Run affected regressions, review the combined branch, integrate
without rewriting shared history, and push the verified `dlno-stc` branch.
