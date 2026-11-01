# Bounded native DF reverse contraction

## Scope and design

Only isolated candidate Python source under `benchmarks/dcmp-ooc-reverse/candidate` is being changed. No numerical imports or tests run on login, no native rebuild, no live edits, no commits.

Existing `GTOnr3c_ij_r0_vjp_s2ij` computes both AO-centre derivative legs directly into atomic gradients. Its cotangent layout is Fortran `(packed_pair_block, auxiliary_functions_in_shell_slice)`. AO-pair rows retain global lower-triangle indexing minus the selected first-row offset. The auxiliary index is local to the supplied shell slice. Python must intersect each auxiliary atom's shell range with the requested range, then use AO offsets relative to that requested range.

For MO cotangents, physical density is `D + D.T`, where `D[p,u,v] = C[u,k] z[p,k,l] C[v,l]`. The factorized-density reader already supplies this physical density. Native packed cotangents use its lower triangle, with diagonal halved because the C driver already counts both AO derivative legs. Native atomic AO derivatives have a leading minus sign; auxiliary derivatives follow from translation invariance independently on each auxiliary atom.

Only ordinary int3c2e coordinate-only, concrete, real float64, symmetric-density contractions are representable by this path. The existing explicit-ip1 contractions remain available for unsupported dtypes or asymmetric callback density. Metadata that includes extra traced molecular leaves must not be sent to coordinate-only native bwd.

Reader slabs remain bounded by the requested row cap, including inside oversized shells. A native shell call requires the packed cotangent for at least one whole auxiliary shell; this is an unavoidable shell-size floor with the existing driver. No `3 * nao * nao * naux_block` derivative tensor is created on the optimized path.

## Tests written before implementation

`pyscfad/df/test/test_reverse_native_slices.py` has 12 parameterized cases:

- Six auxiliary shell slices (within atom, crossing atom boundary, last atom), crossed with AO-pair blocks starting at shell zero or one and ending before final AO shell. Each compares independent explicit-ip1 contraction, full native VJP with zero padding, and sliced native VJP.
- Four real MO/density-reader combinations cover both rectangular MO contraction orders, all AO and auxiliary centres, physical-density diagonal weighting, and row cap one. Explicit numerical reference is computed before instrumentation forbids full derivative tensors.
- Complex MO and asymmetric density fallback cases compare independent explicit derivatives while rejecting entry into real packed native VJP.

AST parsing passed. Numerical red/green evidence is delegated to the parent compute-node runs and has not yet been supplied.

## Implementation and red evidence

Parent confirmed compute job 5821831: six native-slice failures at the old full-auxiliary assertion, four no-derivative-array failures at the instrumentation guard, and both fallback cases passed. Full-native versus independent explicit reference assertions passed before the sliced-native failures. Production changes began only after this evidence.

Implemented auxiliary atom/slice intersections and validated packed cotangent dimensions before C entry. Kept AO shell-pair selection unchanged. Extracted the existing MO physical-density construction and scoped dense BLAS; native selection requires concrete coordinate-only molecules, ordinary int3c2e, float64 operands, and symmetric callback density. Per-shell packed arrays contain half-weighted physical-density diagonals. Unsupported chunks retain the explicit-ip1 path and original fallback result dtype. Mixed supported/unsupported reader chunks can accumulate independently.

The conservative existing 5*AO-square per-auxiliary memory estimate remains; native allocation replaces the derivative tensor with roughly half an AO-square packed cotangent. One entire shell is the minimum native slab, while density/callback reads still obey the cap. New tests additionally instrument actual cotangent bytes to enforce the single-shell bound for cap one.

Only obsolete forward-integral instrumentation in `test_cderi_vjp.py::test_int3c_mo_deriv_coords_vjp_oversized_shell_fallback_is_disjoint` now tracks native backward shell calls. Numerical and read-coverage assertions are unchanged.

When `PYSCFAD_PROFILE_BACKWARD_PHASES` is enabled, the helper emits `int3c coordinate reverse density_pack=... native_contract=... explicit_contract=... native_blocks=... explicit_blocks=...` for performance attribution. Callback reads are included in density/packing time.

Final AST parsing of the implementation and both changed test files passed. Compute-node green verification is pending parent execution. No claim of numerical success or speedup is made yet.
