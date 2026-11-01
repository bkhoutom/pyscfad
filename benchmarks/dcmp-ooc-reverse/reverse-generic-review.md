# Second-candidate generic geometry review

Read-only review on 2026-10-31, comparing `benchmarks/dcmp-ooc-reverse/candidate/pyscfad` with the tested first candidate at `/home/bkhou/DLNO_CCSDT/test_pyscfad_stc_mp2/runtime_test/dCMP-ooc/test_reverse/paired/candidate_code/pyscfad`.

**No blocking defect identified by static review.** No imports, numerical tests, source changes, or jobs were performed by this reviewer. The reported red run 5822122 establishes intended baseline failures; candidate compute validation is still required.

## Diff boundaries

Only one production file differs from the tested first candidate: `df/_df_jk_opt.py`. Its two hunks add `_factorized_geometry_bwd` and dispatch to it from `get_jk_bwd`. No existing algebra or other production file was modified. The sole added Python file is `df/test/test_reverse_generic_geometry.py`. The restored generic `max_memory` assignment and preconditioner Bov release remain intact.

## Correctness and fallback review

- **Full independent density cotangent preserved (`_df_jk_opt.py:664-668`).** The recursive call receives original J/K seeds and the original density. Its activity is `(True, False, False, False)`, so it cannot re-enter factorized geometry. It retains the existing generic density-only transpose, including density directions outside the span of S. No projection or reconstruction from Sbar occurs.
- **Asymmetric seed semantics preserved (`:660-672`, `_df_jk_factorized.py:93-94`).** Original seeds enter the density-only transpose unchanged. Only the separate geometry helper symmetrizes them. Since D=S S.T and coordinate perturbations keep the DF J/K outputs symmetric, antisymmetric seed parts have zero geometry contribution, while their independent density contribution is retained. New tests deliberately use random nonsymmetric seeds and check a nonzero density-cotangent block outside the factor span.
- **Activity respected (`:673-677`).** The return starts with a None-valued DF tree, copies only active molecular/auxiliary bars, leaves the disk placeholder bar absent, and returns dm_bar=None when density is inactive. SymbolicZero/disabled seeds have already been converted to None by the outer routine; zero arrays are supplied only as helper inputs for disabled components. The all-inactive-output case returns before dispatch.
- **Factors describe the actual density (`:651-657`).** Existing strict reconstruction validation rejects stale tagged factors; successful pivoted-Cholesky recovery remains acceptable. Nonfinite/asymmetric/indefinite densities and zero rank cannot enter the new route. Factorization is used only for geometry, with factor_response=False.
- **Fallback boundaries preserved (`:630-650`, `:697-735`).** Real float64, one two-dimensional density, coordinate-only molecule trees, finite real float64 seeds, hermi=1, and disk-placeholder state are required. Traced execution is handled before the new dispatch. Implicit-solve matvecs and supplied coordinate callbacks bypass it. The MPI context hook is checked. Resident physical CDERI cannot satisfy the placeholder check, so its independent factor cotangent remains in the original route. The occupied-VJP opt-out is honored.
- **Cost guard is conservative (`:655-657`).** Requiring `nao*rank < nao_pair` reduces the number of exchange metric-solve columns. This does not guarantee lower total wall time or storage: the geometry helper keeps two thin panels and the independent density route causes a second CDERI read. Near the threshold, both thin panels together can exceed one original packed-adjoint array. Panel storage remains governed by the existing memory/spill policy. For J-only calls the rank guard is unnecessarily restrictive but safe.
- **Repeated use and lifetime.** No new cache/global mutable state is introduced. The factorized helper's context owns and closes its existing T/R scratch panels. Generic AO-pair-adjoint scratch is bypassed only after all eligibility checks pass.

## Nonblocking observations

**P3 profiling omission (`_df_jk_opt.py:651-659,678-682`):** the new helper starts its timer after `_density_factors` and tag validation. Its `done` line therefore excludes density factorization/reconstruction, potentially material at large nao. The outer `get_jk_bwd` returns immediately without emitting its usual total-duration line. Use a caller-level total timer when judging the optimization, or include eligibility/factorization time in the new reported total. Density/geometry substage timings themselves are correctly scoped.

The rank guard measures metric solve width, not a measured performance break-even point. Keep the first-versus-second candidate benchmark comparison; do not infer speedup solely from avoiding `pyscfad_dfjk_eri_bar_` files.

## Test assessment and remaining evidence

The new tests use an independent dense JAX expression, arbitrary seeds, separate molecular/auxiliary inputs, repeated cotangents, J-only/K-only/both cases, explicit outside-factor-span density checks, geometry-only activity, stale tags, and disabled/asymmetric/indefinite/custom-callback fallback. This is meaningful coverage rather than a comparison against the same factorized implementation.

Before deployment, require the submitted expanded suite to pass and compare full dCMP energies/gradients against the frozen baseline and in-core reference under the planned tolerances. Confirm profile output shows the new route for the intended extra-cycle geometry call, and measure total reverse time and peak memory.

Useful coverage not present in this new file, but not a static blocker: an eligible low-rank density with actual SymbolicZero seeds; an MPI hook present without a custom coordinate callback; high-rank/zero-rank cost-guard fallback; float32/complex/batched fallback; forced thin-panel spill and cleanup. Existing broader suites cover several related activity, traced, resident-factor, MPI, and panel-lifetime cases, but low-rank inputs are needed to exercise this new eligibility boundary directly.

## Final review: flattened NumPy density GEMMs

Compared the final candidate with `/home/bkhou/DLNO_CCSDT/test_pyscfad_stc_mp2/runtime_test/dCMP-ooc/test_reverse/regressions_geometry/code/pyscfad`. Only the expected density-contraction bodies in `df/_cderi_vjp.py` and `df/_df_jk_factorized.py` differ, plus the added `df/test/test_reverse_dense_density.py`. No other Python-source changes were found. No numerical work was run by this reviewer.

**No blocking issue found.** The two physical-density branches preserve the original contraction D[p,u,v]=sum(k,l) Ck[u,k] z[p,k,l] Cl[v,l]:

- For kc<=lc, the first flattened GEMM produces tmp[p,k,v]. Transposing to [p,v,k] and multiplying by Ck.T produces [p,v,u]; the final swap restores [p,u,v].
- For kc>lc, flattening z[p,l,k] and multiplying by Ck.T produces [p,l,u]. Its transpose is [p,u,l], so multiplication by Cl.T directly produces [p,u,v].
- Ordinary transpose, rather than conjugate transpose, preserves the existing complex convention. NumPy matmul supplies ordinary dtype promotion; no new real cast is introduced. Negative/nonunit z strides are supported through reshape copies when a view is impossible.
- `_df_jk_factorized.read_density` simply fuses the auxiliary and AO row dimensions of panel[p,u,r] before multiplying by S[v,r]. Restoring [p,u,v] is equivalent to the prior batched matrix product, and the subsequent symmetrization/scaling is unchanged.

The new operations stay inside the existing scoped dense-BLAS context. This makes the intended NumPy backend explicit instead of relying on PySCF einsum's backend selection.

**Memory assessment:** transposed reshapes may copy one z block and/or one thin intermediate. They remain bounded by the auxiliary range; no full-auxiliary AO-square allocation or persistent cache is added. The intermediate is released before D+D.T. The kc<=lc output initially has transposed AO strides, so subsequent packing may require a contiguous block copy. For physical orbital spaces with kc,lc<=nao, the retained conservative five-AO-square block estimate leaves room for these thin/block-local temporaries on the real route. It remains a workspace target rather than a hard process RSS bound, and complex fallback arrays have larger element size, as before. The factorized panel is normally contiguous in its existing RAM/HDF5 representation, so its flattened multiplication normally requires no additional panel copy.

The four new tests compare against an independent three-factor NumPy einsum with kc<lc/kc>lc, float64/complex128, and negative-stride z, while rejecting a return to PySCF einsum. They cover layout and conjugation errors meaningfully. Caller-level empty-MO handling remains before this helper; no new zero-rank call path is introduced.

Reviewed final source hashes:

- `_cderi_vjp.py`: `8348da3144f7547a5b108935faf0610088d4247b9bb0bbcbbf6fcdfe930fc4b4`
- `_df_jk_factorized.py`: `cf8e681a7b97a30081d76ae74c7f8c838ffac450f6021ce5c56371dc599ead10`

The reported construction benchmark improvement (0.14478 s to 0.12656 s median, job 5822301) is a microbenchmark claim supplied by the parent, not independently executed here. Final suite 5822323 and full dCMP energy/gradient/performance results remain the deployment evidence.

## Last review: fused real physical-density construction

Compared candidate with `/home/bkhou/DLNO_CCSDT/test_pyscfad_stc_mp2/runtime_test/dCMP-ooc/test_reverse/regressions_final/code/pyscfad`. The sole Python-source difference is the guarded fused path added to `_df_jk_factorized._reverse`/its density reader; no tests or other production files differ. Reviewed file SHA256: `721719dcd9679334f8d324bf5a9f8d7c78db7f29f2edaae29217988d79396d4e`.

**No numerical correctness blocker found.** Let K=panel S.T and J=a G+b D. The original density is 2(K+K.T+J). The new path constructs raw=2K+J, then returns raw+raw.T. Requiring real-double inputs and exactly symmetric G,D when J is enabled gives the same expression, including diagonal terms. K need not be symmetric. J-only and K-only reduce correctly. Other dtypes or nonexact J symmetry retain the untouched original arithmetic, avoiding a change from transpose to conjugate-transpose semantics. Fused arithmetic may change rounding order; the complete gradient checks remain necessary.

The packed J columns have shape (nao²,2), with both G and D flattened in the same C order as the density. Coefficients have shape (2,aux_block_rows). The GEMM output has shape (nao²,aux_block_rows) and adds into the transpose of C-contiguous raw rows. Using the returned `updated` array handles SciPy output-copy behavior correctly. Its transpose/reshape restores [p,u,v] in C order, suitable for the native in-place hermi_sum. The installed PySCF helper confirms that this operation sums both legs and doubles the diagonal.

**Memory:** persistent additions are two nao² J columns and one nao*rank doubled-factor matrix; they are owned by this reverse call, not cached globally. Building the Fortran J-column array can briefly hold both stacked and converted arrays. Per auxiliary block, the fused path avoids the old transposed-density copy and broadcast J products, though SciPy may still copy an output buffer. It therefore reduces large block temporaries, not necessarily total peak memory for every block size; with one-row blocks the new fixed AO-square columns can outweigh the removed temporary. All additional storage remains independent of the full auxiliary count, apart from existing thin panels and vectors.

**P3 thread-scope issue:** the inspected code returns `lib.hermi_sum(...)` while still inside `dense_blas_threads()`. `pyscfad/lib/_threading.py` explicitly documents that OpenMP kernels must not be enclosed, and the installed `pyscf/lib/np_helper/transpose.c:129` implements `NPdsymm_021_sum` with `omp parallel`. Move only this final symmetrization outside the BLAS context to follow the intended scope. This is not an observed numerical error or demonstrated slowdown; it is a clear mismatch with the local threading contract.

The parent reports prototype job 5822376 passed repeated J/K/JK numerical and exact-symmetry comparisons, with JK construction medians 0.14368→0.08376 s. These are supplied compute results, not reviewer execution. Expanded candidate suite 5822404 and the final full dCMP benchmark must establish the final source's correctness and overall benefit.
