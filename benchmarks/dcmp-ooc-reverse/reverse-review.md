# Independent reverse optimization review

Reviewed 2026-10-31 against the user-modified snapshot in `benchmarks/dcmp-ooc-reverse/baseline/pyscfad`. Candidate source only; no source edits, jobs, imports, or numerical tests performed by this reviewer. Read PLAN.md and the native C contraction implementation. Compute validation remains required.

## Finding

**P2: preconditioner setup unnecessarily retains its previous/final Bov block.** Candidate `df/_df_jk_opt.py:869-881` removed the baseline's `bov = None` after each Gram accumulation. Python allocates the next `numpy.empty` before releasing the old binding, so consecutive occupied blocks coexist transiently. The final block also survives through Cholesky factorization and inversion, where multiple naux-square arrays coexist. This occurs even with cache budget zero, weakening the low-memory fallback. Restore the explicit release immediately after the Gram update. Separately, `kmat += bov @ bov.T` creates an additional naux-square temporary versus baseline's in-place `lib.dot(..., c=kmat, beta=1)`; account for this in peak-memory measurements or use scoped SciPy BLAS with in-place accumulation if it matters. The temporary is not a numerical defect.

The initially known missing cache headroom check was repaired during this review: current lines 859-862 limit cache size to both the environment budget and 30% of available `dfobj.max_memory`. Static review accepts this guard, pending its red/green compute test. This bounds the retained cache; it is not a hard bound on total preconditioner workspace.

## Numerical and ownership assessment

- `df/_df_jk_opt.py:875-899`: cached A=B/sqrt(e) gives K=1/4+A A.T, and application sqrt(e) A.T K^-1 A (s/sqrt(e)) equals B.T K^-1 B e^-1 s. Flattening remains virtual-major/occupied-minor. The explicit dense Hessian regression is an independent and useful oracle. Cache lifetime is the returned closure, with no module/global sharing between calculations.
- `df/_int3c_cross_opt.py:79-138`: auxiliary shell intersections and cotangent offsets are now correctly relative to the requested slice. AO packed-row offsets agree with `pyscfadlib/pyscfadlib/vjp/gto/fill_nr_3c_vjp.c:117-159`. Slices can start/end inside an atom, and auxiliary/molecular atom counts need not agree. Auxiliary gradients use translation invariance separately per auxiliary atom.
- `df/_cderi_vjp.py:1708-1712`: physical density already contains both AO derivative legs. Off-diagonal packed weights remain D_ij+D_ji, while diagonal half-weight compensates the C driver's explicit factor of two. This is algebraically correct. Tests compare with explicit s1 ip1 values and independently formed physical densities, including random diagonal and off-diagonal contributions.
- `df/_cderi_vjp.py:1625-1746`: packed workspace spans one balanced auxiliary-shell range. Logical density/z reads obey the row cap; one indivisible shell can exceed the target. Per-shell packed buffers are released. C worker scratch and optional explicit-fallback derivative arrays remain additional allocations. Existing conservative block sizing remains adequate for the normal real native route; it should not be described as a hard process-memory cap.
- `df/_cderi_vjp.py:1647-1700`: native dispatch excludes unsupported integral names, extra molecular derivative leaves, molecular tracers, non-real-double MO factors/bars, and asymmetric density-reader outputs. Unsupported blocks retain the explicit contraction. Exact symmetry checking may select fallback for a density with roundoff asymmetry; numerical behavior is conservative, but benchmark native/explicit block counters must establish whether the intended optimization was exercised. This does not establish new higher-order/JIT support: pre-existing host conversions remain.
- `mp/dfmp2.py:374-428`: the VJP residual retains the actual forward result. `saved_output` is read in the existing metric cotangent calculation (`df/_cderi_vjp.py:1869-1880`), without mutation. C_vjp remains separately computed from the original disk-backed factors. Arbitrary MO slices, including unused columns, retain their full coefficient cotangent. Frozen selection occurs upstream and was not changed.
- Repeated-pullback tests use differing cotangents and then repeat the first, checking both bars and output immutability. They also require exactly one forward streamed transform while requiring each reverse MO-coefficient pullback. This meaningfully tests the intended reuse.

## Compute evidence required before deployment

1. Green focused suite after the preconditioner headroom and Bov-release fixes, including all three new reverse test files, existing `df/test/test_cderi_vjp.py`, factorized DF tests, and `mp/test/test_dfmp2_outcore.py` (notably generic float32 fallback).
2. Full frozen-selection/out-of-core energy and gradient checks. The new saved-Lov test has a nonzero MO slice and full C_vjp comparison; existing native-MP2 frozen tests alone may use the in-core transform and are not direct proof of this changed OOC residual.
3. The planned full dCMP comparison: energy difference <=1e-9 Eh and every one of 102 gradient components <=2e-7 Eh/bohr, with all molecular and auxiliary responses retained.
4. Effective BLAS/OpenMP pools, peak RSS, setup/application counts, cached-versus-streamed preconditioner mode, and native-versus-explicit coordinate block counts. New wall timings are reasonable synchronous substage measurements but do not themselves prove reduced I/O, bounded total memory, or overall speedup.

Optional follow-up coverage, not independently identified blockers: Cartesian auxiliary shells, a mixed symmetric/asymmetric density reader within one shell range, and positive-but-insufficient available-memory cache thresholds. The current new tests already cover spherical partial-shell/atom offsets, both MO contraction orders, complex bars, asymmetric-density fallback, reader caps, and zero cache headroom.

No additional numerical correctness defect was identified by static review. This is not a numerical sign-off; pending compute results determine deployability.

## Follow-up: restored diff boundaries and remaining generic SCF geometry work

After compute job 5821893 exposed the accidental removal of `max_memory` from generic `get_jk_bwd`, the assignment was restored. A full Python source comparison now shows:

- Every byte of `_df_jk_opt.py` before `def gen_coulomb_precond` matches baseline. The NameError edit is no longer present.
- The Bov lifetime finding above is repaired by `bov = mo = eri1 = None` after each Gram update.
- Exactly four production Python files differ: `_df_jk_opt.py`, `_int3c_cross_opt.py`, `_cderi_vjp.py`, and `mp/dfmp2.py`; all hunks are inside the reviewed functions.
- Existing `test_cderi_vjp.py` changes its integral-range observer from explicit int3c calls to native bwd calls; `test_dfmp2_outcore.py` changes the expected forward transform count to exactly one. Three new reverse test files are present. No other Python-source differences were found.
- `_df_jk_opt.py` SHA256 at this audit: `9a2a084b5521f9553319173e4136799bd9e4c1bb9fe70d70e62141d1e81f1fe0`.

The proposed next hotspot is real in the code path, though its runtime significance must come from the pending profile. `scf/hf.py:665` calls generic `get_veff` with the independently differentiated converged density to recover orbital derivatives. Its disk-backed geometry reverse (`df/_df_jk_opt.py:721-827`) writes `(nao_pair,naux)` AO-factor bars to HDF5, then processes them through generic metric and integral pullbacks. The later occupied-factor energy call and the external implicit-root VJP already avoid that route.

**Recommended mathematical split:** preserve generic density differentiation and replace only geometry work for eligible inputs. For fixed real symmetric D=S S.T, the coordinate derivative of J(D),K(D) is exactly the coordinate derivative of J(S S.T),K(S S.T) holding S fixed. The density derivative must remain the original independent AO-density derivative. A factor cotangent Sbar=2 Dbar S cannot reconstruct all components of Dbar; therefore changing the SCF call itself to a factor-only primitive is not equivalent.

A minimal first implementation could dispatch inside eager `get_jk_bwd`, before allocating AO-factor bars:

1. Obtain S from existing `_density_factors(_tag_dm_for_fast_exchange(dfobj, dm), hermi)`, whose reconstruction check rejects stale tags and indefinite/unsupported densities. Reuse this validation rather than trusting the process-global orbital cache. Start with one density and positive thin rank.
2. Call the existing generic backward with activity `(dm_active, False, False, False)` to retain its full density-only result, including non-symmetric seed components. This recursive call cannot re-enter the geometry fast path because geometry activity is false. Skip this call if density is inactive.
3. Reuse `_df_jk_factorized._backward(..., factor_response=False, residual=(dfobj,S), cotangents=(G,H))` for geometry, or expose an equivalent internal helper. Supply ordinary zero arrays for inactive J/K seeds and disable their respective work flags. This uses thin T/R panels, their bounded metric solves, and the now-native coordinate contractions; it never needs the generic AO-factor-adjoint file.
4. Build the returned DF tree with `None` for every inactive field, then copy only active mol/aux coordinate leaves. Do not return `_df_jk_factorized`'s complete zero-filled DF tree unchanged: generic symbolic activity semantics require inactive bars absent. Leave the factor cotangent unused.

Keep this initial dispatch conservative: coordinate-only mol/aux trees, real float64 density and active cotangents, hermi=1, one density, validated rank>0, out-of-core placeholder with no physical CDERI bar requested, non-traced first-order execution, not an implicit-solve matvec, and `coordinate_vjp is None`. Honor the existing occupied-VJP enable switch. A supplied coordinate callback is used by MPI code and must retain the current route unless explicitly supported. Batched, zero-rank, indefinite, complex, traced, higher-order, resident-factor, extra-derivative-leaf, and custom-callback cases should initially retain the generic route.

This simple split reads CDERI twice (density and geometry) instead of the original fused reverse block loop. It saves AO-pair bar writes/reads and replaces metric solves over nao_pair columns with solves over nao*rank columns. That is likely advantageous for occupied rank much smaller than nao, but the extra read and unpacking must be timed. A shared block loop is a later optimization only if profiles justify its complexity.

Necessary independent compute tests for this additional change:

- Compare full `dm_bar`, molecular coordinates, and auxiliary coordinates against the unchanged generic path and a small dense reference, using random positive-semidefinite D and arbitrary J/K seeds, including non-symmetric seeds. Explicitly test that dm_bar is unchanged outside occupied-factor support.
- Cover geometry-only and density-plus-geometry activity; individually inactive mol/aux and symbolic J/K zeros; J-only, K-only, and both. Retain absent bars for inactive leaves.
- Forbid the generic `pyscfad_dfjk_eri_bar_` temporary on eligible geometry calls, and verify that required factor-panel spills still clean up on success/failure.
- Verify fallback for indefinite/non-Hermitian densities, unsupported dtype/tracers, batched inputs, zero rank, resident physical factors, stale tags, and supplied coordinate callbacks.
- Re-run complete implicit-SCF/MP2 energy and gradients, frozen-selection checks, and full dCMP tolerances. Update instrumentation-based tests that currently require one generic AO-adjoint file per geometry call; preserve their numerical oracle and implicit-matvec exclusion assertions.

No implementation of this optional next optimization was performed by the reviewer.
