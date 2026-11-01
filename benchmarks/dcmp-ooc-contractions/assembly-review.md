# Packed-adjoint assembly review

The final completed-job evidence and current adoption status are recorded in [evidence-review.md](evidence-review.md); pending-job statements below preserve the earlier review checkpoints.

Static review of the assembly-only candidate against `PLAN.md`, 2026-10-31, with follow-up reviews of the alias fix, paired timing instrumentation, and final packed-J thread scope. No numerical imports, tests, calculations, or batch submissions were performed. Source text, directory differences, hashes, Python ASTs, and saved compute-job results were inspected.

**Final static status: no remaining blocking findings.** The output-aliasing defect is resolved, and the final one-thread packed-J scope restores prior thread limits correctly. Saved controlled results support adopting this local improvement if the final exact-code full-gradient and regression checks pass. At the latest review, full job 5828194 and final regression job 5828195 were still running according to the coordinator. Earlier validation checkpoints below describe their respective revisions.

## Final packed-J threading and performance review

The final change imports `threadpool_limits` and replaces only the packed rank-two J update's dense context with `threadpool_limits(limits=1, user_api='blas')`. AST inspection confirms that the scope contains only the `dgemm` call. K expansion remains in `dense_blas_threads()`, and both native symmetric packing and the subsequent native coordinate contraction run outside either local BLAS context. Dense fallback arithmetic and its threading are unchanged.

Inspection of the installed `threadpoolctl.py` confirms that the context records the loaded pools' prior limits and restores them in `__exit__`, including exceptions. Its setter selects only BLAS libraries for `user_api='blas'`; it does not lower the independent OpenMP pool. Nested callers regain their previous BLAS settings rather than an assumed default. As with the existing dense-thread helper, limits are process-wide and competing concurrent Python thread policies remain unsupported. No new restoration or scope defect was found.

The actual saved `jobs/assembly_native/results.5828088.json` contains five alternating-order samples over three real dCMP auxiliary shell blocks (12, 14, and 16 rows), with 800 AOs, 2,042 auxiliary functions, and random rank-80 factors. Each measurement includes assembly immediately followed by the full native molecular/auxiliary coordinate contraction for the block.

| Variant | Median combined seconds |
| --- | ---: |
| Original dense assembly | 0.714894 |
| Packed J, 16 BLAS threads | 0.659504 |
| Packed J, 1 BLAS thread | 0.482990 |
| Full-space J plus native packing | 0.604680 |

Packed J with one BLAS thread reduces the combined median by 32.4% relative to the original and by 26.8% relative to packed J with sixteen threads. It wins all five same-index comparisons against the original; every J1 total (0.445–0.501 s) is below every original total (0.662–0.744 s). The recorded median of paired speedup ratios is 1.5005, distinct from the 1.4801 ratio of the two medians. The native-only median is still higher than the original (0.314174 versus 0.238154 s), so the relevant evidence is the net combined time, not assembly alone.

Independent explicit-density/derivative-integral checks cover packing and every molecular/auxiliary coordinate for each block and variant. The largest recorded packing error is `1.11e-16`; the largest coordinate error is `6.57e-14`. Saved pool snapshots show both BLAS libraries at one thread outside dense scopes, sixteen inside, and restored to one after all samples, while OpenMP remains at sixteen.

This is strong local performance evidence, with two limits. First, the microdriver's J1 variant uses the already-one-thread ambient state and does not enter the final production code's explicit `threadpool_limits` context per block. That context's discovery/restoration overhead is therefore absent from the microtiming and must be included by the final full run. Second, the experiment uses random factors and three representative blocks, and excludes one-time J-column setup; it is not a full gradient-run speedup estimate. The original and candidate full timing hooks include setup and the final context overhead.

Saved v1 paired job 5826995 validates all 102 gradient components: maximum deviations from the reference are `3.92e-12` and `4.56e-12` Eh/bohr, with energy deviations at most `2.27e-13` Eh. It does **not** justify attributing its total 710.010→523.584 s change to the assembly code. Unchanged SCF forward changes 290.795→145.806 s. In the changed reverse region, assembly plus setup improves 63.252→39.743 s, but native coordinate contraction worsens 32.829→51.501 s; combined factorized DF reverse improves only 131.708→126.906 s. The final J1 revision should be described using controlled local evidence and final exact-code results, with full-run variability stated explicitly.

Final scoped-review hashes:

```text
candidate/pyscfad/df/_df_jk_factorized.py          48f9b5cd68f75d5d224b40f30da7b5f053bb0cb24020d6d8edbabb65893da66f
jobs/assembly_native/driver.py                   c40cd6100c71fc89db28dfbf60c7401ddd29c23e58b1cdd7f5d3ca2167abe8f1
jobs/assembly_native/results.5828088.json         54e150979e9ceefaed0e6c12f7d3d1e56f5e723bffad1a3d0824f8ff9a132b05
```

## Finding

**Resolved P2 — Reject overlapping input/output storage in `_pack_tril_symmetric_sum`.**

The follow-up revision calls `numpy.shares_memory(density, out)` after the contiguous-input conversion and raises before entering C. This rejects aliases of the buffer actually read by C. If a strided original input required a separate contiguous copy, overlap with only the original storage is safe. Two new tests cover output offsets 0 and 3 and require the input to remain unchanged after rejection. The defect below describes the original reviewed revision.

Location: `candidate/pyscfad/df/_cderi_vjp.py:1613–1624`.

The new helper accepts any writable, C-contiguous float64 `out` of the expected shape, including a view of its input. `numpy.ascontiguousarray` does not copy an already contiguous input. The native implementation first writes the lower triangle into `out`, then reads the upper triangle from the original input (`native/pyscfadlib/vjp/np_helper/pack_tril_vjp.c:28–37`). An overlapping output therefore overwrites values that the second pass still needs. Multiple overlapping batch rows can also race in the OpenMP loop.

Concrete input, derived directly from the native loops (not executed here):

```python
density = numpy.arange(1., 10.).reshape(1, 3, 3)
out = density.reshape(-1)[:6].reshape(1, 6)
_pack_tril_symmetric_sum(density, out=out)
```

The original guards accepted this call. The native loops produce `[1, 8, 5, 12, 17, 9]` instead of `[1, 6, 5, 10, 14, 9]`. The factorized producer uses separate allocations, so the finding did not establish a failure of the dCMP execution path. The new rejection closes the accepted `out=` contract hole.

## Checks that support the change

- **Normalization:** The native unpack VJP with `hermi=3` writes `P(A)[i,j] = A[i,j] + A[j,i]` for `i > j` and `P(A)[i,i] = A[i,i]`. This is exactly `pack(A + A.T)` with its diagonal halved. It uses no conjugation.
- **K assembly:** For `X = 2 R S.T`, packing `X` directly matches the previous physical density `2(R S.T + S R.T)`, including the native diagonal half weighting.
- **J assembly:** For symmetric `G,D`, the packed columns are `P(G),P(D)`. Their combination `a P(G) + b P(D)` matches the previous physical J density `2(aG + bD)` after native normalization.
- **Flags and layouts:** Packed columns and coefficients are Fortran-contiguous for `dgemm`, and each supplied output row slice is C-contiguous, making `out.T` suitable as its output. J-only uses `beta=0`, so uninitialized output is not accumulated; J+K uses `beta=1`; K-only fully fills the buffer. Both flags false explicitly zero the callback buffer. If SciPy copies its output, the returned array is copied back.
- **Type and fallback contracts:** The producer enables packing only for float64 `S,G,H,D`, exact J symmetry, and an available native pack symbol. The helper validates input/output float64 and square batches, and rejects strided/read-only output. Its noncontiguous input is copied. The coordinate helper retains the original native-eligibility gate, including coordinate-only molecule leaves, tracer exclusion, supported integral names, and an explicit non-double `z_dtype` veto. A filler requires a dense fallback reader. Symmetry and complete filling are producer promises; the coordinate consumer deliberately does not reread a physical density to verify them.
- **Existing fallback arithmetic:** Complex/non-double or nonsymmetric inputs retain the previous dense producer. Missing `_symmetric_pack` disables the new producer path. The new code introduces no additional tracer conversion into the native path; higher-order/JIT support is not demonstrated by this static review.
- **Memory bounds:** The coordinate callback writes into one shell-block-sized packed buffer, with the existing one-shell minimum and subrange cap. K constructs only a bounded auxiliary slab of AO-square density plus a thin panel. J retains two AO-pair vectors, independent of `naux`; the former dense J columns are now lazy. There is no new all-auxiliary AO-pair allocation or cross-VJP cache. Packing is outside the dense BLAS thread scope.

## Test assessment and remaining evidence

The literal packed-value test independently checks diagonal weighting, triangle order, and transposed input. The coordinate test independently contracts explicit `int3c2e_ip1` integrals and checks one-row reads, shell-boundary storage, molecular/auxiliary atom offsets, and translation balance. The six J/K and factor-response combinations use an independent dense tensor expression and force disk panels and small solve slabs. Their integral derivative backend shares the existing native coordinate driver, but not the new factorized assembly; the separate explicit-integral test supplies the independent check of coordinate normalization.

The fallback tests cover complex, float32, and asymmetric physical densities. They do not currently exercise an unavailable pack symbol, a filler present with coordinate-native eligibility disabled by molecule metadata/tracers, or both J/K flags false. The first two would strengthen the new interface's fallback coverage, but inspection found no defect in those gates. Overlapping buffers are now covered by the two added cases. The factorized integration test records that a callback was supplied, rather than recording callback invocation; wrapping the filler would make the test directly prove that production packed assembly executed.

No mathematical or bounded-memory blocker was found in the intended float64 coordinate-only producer. Adoption still requires the compute-node green regressions, paired all-component gradient tolerances, and measured improvement specified in `PLAN.md`.

## Paired timing instrumentation follow-up

`instrument.py` changes only frozen benchmark copies. It uses unique-match assertions and AST parsing to check its text substitutions. A full directory comparison (excluding `__pycache__`) shows each frozen copy differs from its respective scientific source in exactly the three instrumented modules. Comparing the two frozen copies shows only the two assembly implementation files and the added packed-density test differ. `_df_jk_opt.py` is byte-identical across the pair. The initial factor contractions have not changed; the unsuccessful contraction microexperiments were not adopted.

The timing boundaries are consistent across baseline and candidate:

- `factor_response_prefix_seconds` spans entry through setup and the initial complete DF loop. `df_seconds` begins after that loop and ends after metric reverse and an explicit readiness barrier on both coordinate cotangents.
- Metric transform, density setup, coordinate, and metric reverse boundaries partition `df_seconds`. Density setup includes the candidate's one-time J packing, so its cost is included. Per-block density timing includes the whole candidate filler, including native packing and BLAS thread-scope overhead.
- Read timing measures generator advancement before yielded blocks. Unpack timing includes the small row-index update. Dense timing includes the scoped contractions, thin-panel writes, and buffer deletion. These are diagnostic subsets of the total prefix, not an exhaustive partition or pure BLAS/I/O timings: prefix setup, final generator exhaustion/cleanup, and timer bookkeeping remain outside their sum.
- Coordinate helper timing is nested within factorized coordinate timing. The ledger reports these as breakdowns rather than adding nested coordinate time again to `df_seconds`. It filters the SCF split by phase and has explicit expected-call-count and partition checks. The new prefix diagnostics are saved in raw `timing_events`; `timing_ledger.summarize` does not yet aggregate them.
- The implicit-matvec hook measures the existing synchronous NumPy/native path. The factorized end barrier makes pending metric reverse work part of its measured interval. Per-block timer calls and that barrier add benchmark overhead, identically in both copies; the pair therefore measures instrumented executions.

No instrumentation change to scientific algebra or asymmetric timer placement was found. Comparing combined density setup plus per-block assembly is necessary because the candidate deliberately moves J work into setup. A zero-width coordinate-helper early return has no event, as before; the fixed nonempty dCMP calculation does not use that route.

Follow-up SHA-256 values:

```text
instrument.py                                    2e7edc92ae40d19b292c16a6b1232fbca1a4fee21eafd4ad38af60e2a412e8e1
candidate/pyscfad/df/_cderi_vjp.py                c4321e1ead3308dffe63e52ba04876ed683e3a4cde1b3a73ef3d8caaa178cbdc
candidate/pyscfad/df/_df_jk_factorized.py          0058fd805506de574c35fa1e1494d7b85d55f8a3a7720bc13d3a024df364bbc3
candidate/pyscfad/df/test/test_reverse_packed_density.py
                                                 4f7440b7d57211002de270a249f9557258c6134886a885754c8a0634feaf31dd
paired baseline _cderi_vjp.py                    d0cd0d6beaeafdbb3cd424c4cd58d8f5ceb1fa206a7486c4dfcd6180312d94e6
paired candidate _cderi_vjp.py                   fb4cb14a3b61a2dca7ac293be55bfe572dceb39414d95303a4ddf40c6478699f
paired baseline _df_jk_factorized.py             1fac6ee57b68800427e4f526c4e2aa0cf1f375bbd294c3cf8622aa6f4351c6fe
paired candidate _df_jk_factorized.py            60731690d2077fcd4428fb8d3105e28fac7f2cab8ca93def60c297fd83b2b397
paired both _df_jk_opt.py                        7c440ab22ee78e5a2ac70a8d8c5623003bec3bf45e8700fbad683464d2dd07d1
```

Original-review SHA-256 values (before the alias fix):

```text
candidate.patch                                  84ffd839ced2c53b61950a7194173e2555145d0ae7b78f592903911acf6fd3a3
candidate/pyscfad/df/_cderi_vjp.py                22294d2ea21c96b0db16ced17853ece817a0fa5162317e687ba4c34fb2ffbdc6
candidate/pyscfad/df/_df_jk_factorized.py          0058fd805506de574c35fa1e1494d7b85d55f8a3a7720bc13d3a024df364bbc3
candidate/pyscfad/df/test/test_reverse_packed_density.py
                                                 23e512591b36c6b16954653483834e3c7c72ec8c5b155720f15748faef0d0eb5
```
