# Fused factorized DF coordinate-density construction

Changed only the isolated candidate `pyscfad/df/_df_jk_factorized.py` within `_reverse` and its nested density reader. No live files, batch snapshots, native code, public API, global cache or memory-budget settings changed.

Parent supplied compute evidence from prototype job 5822376: J/K/both values agreed and the fused result was exactly symmetric. Combined J/K median block-construction time improved from 0.143680 s to 0.083758 s. This microbenchmark is not a full-calculation speed claim.

For real float64 inputs and exactly symmetric G,D, the reader now forms `2 K + J` using a flattened NumPy GEMM and an in-place SciPy BLAS GEMM, then uses real `lib.hermi_sum` to produce `2*(K+K.T+J)`. The J coefficient vectors are a and b. Precomputed doubled S transpose and G/D column matrices are local to the reverse call; all auxiliary data remains block bounded. BLAS's returned buffer is consumed whether or not it aliases the requested output.

Other dtypes or nonexact G/D symmetry retain the original arithmetic, preserving its no-conjugation convention. Scoped dense BLAS restores thread settings when the nested reader returns. AST parsing passed. Parent owns final 204-case compute regression, independent review and full dCMP timing/gradient validation.

Independent review requested moving the native OpenMP `lib.hermi_sum` outside the dense BLAS scope. Applied this scope-only fix to the candidate: fused GEMMs finish and restore BLAS thread limits before `hermi_sum`, while fallback arithmetic retains its original context. AST inspection confirms the intended nesting. Frozen regression job 5822404 was not edited; final compute validation remains parent-owned.
