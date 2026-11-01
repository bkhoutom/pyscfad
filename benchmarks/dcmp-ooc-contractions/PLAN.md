# OOC reverse contraction and adjoint assembly optimization

User requested trying the proposed optimizations and adopting successful changes. That authorization covers implementation, compute-node tests/benchmarks, and installation of validated winners. No further design confirmation or commits are needed. Scientific calculations and tests must never run on login nodes.

## Design

Preserve the current dirty source as a baseline and work in isolated package snapshots. Keep bounded-memory OOC semantics, real/complex fallback behavior, and all molecular/auxiliary derivatives. Prefer existing native kernels and local algebraic improvements over global caches across AD invocations.

Approaches:
1. Reuse the existing native unpack-triangle VJP to form packed symmetric adjoints directly from the unsymmetrized density. This can eliminate full-array symmetrization, symmetry checks, packing temporaries and diagonal normalization passes without recompilation. Supply an optional packed-block callback to the coordinate helper and retain its current dense fallback.
2. Consolidate repeated small matrix multiplications into larger GEMMs in each initial factor loop. In particular flatten the auxiliary/AO dimensions for B@S, batch H@t using a small thin-panel transpose, and reduce the real-symmetric B@r sum with a single transposed GEMM. Benchmark copy costs and preserve unsupported dtype paths.
3. Cross-VJP caching and delayed adjoint aggregation have broader lifetime/memory/correctness risks. Consider only if local improvements do not suffice and exact input identity can be established. Do not reuse unequal orbital factors.

## Validation and adoption

- [x] Capture baseline source/native hashes and isolated current-code copies.
- [x] Write independent dense-algebra and coordinate-pullback regression tests before implementing the new packed interface; demonstrate missing-interface failures on compute nodes.
- [x] Benchmark old/new contraction and packing kernels using dCMP shapes and existing 16-core thread settings, including all copies and packing costs.
- [x] Implement only supported local winners in the candidate; review normalization, symmetry, noncontiguous inputs, J-only/K-only paths, unsupported-native fallback and memory bounds.
- [x] Run focused new tests and relevant existing DF/SCF/MP2 OOC regressions on compute nodes.
- [x] Run paired full dCMP baseline/candidate calculations in the same allocation. Repeat in reversed order if needed to separate performance from run-order variability. Save all gradients and exact timing boundaries.
- [x] Adopt only improvements with validated full gradients (energy within 1e-9 Eh, all 102 gradient components within 2e-7 Eh/bohr), no memory regression that defeats OOC, and measured speed improvement. Preserve prior dirty edits and rollback artifacts.
- [x] Verify the installed file hashes match tested artifacts and report source changes, paired timings and limitations.

Jobs live in self-contained condition directories under `dCMP-ooc/test_reverse/contractions`: local driver/input/submit/code and outputs. Use the prior working expansion/sharmagroup/normal Ice Lake configuration, 16 physical cores, 128 GB, scratch under `/resnick/scratch/$USER`. Static SLURM scripts only; submit from each condition directory. No Git in batch scripts, no submission framework or automatic retries. Existing trusted environment and libraries are retained unless evidence requires a rebuild.

## Task interfaces / ownership

Assembly implementer owns candidate `df/_cderi_vjp.py`, the density-reader section of `df/_df_jk_factorized.py`, and new `df/test/test_reverse_packed_density.py`. Root coordinates shared-file edits, integration and benchmarks. Contraction exploration initially produces a standalone benchmark only; edits to the initial factor loop occur after the assembly implementer finishes that file. Independent reviewer audits each completed change and the final diff.

Native packed adjoint must represent off-diagonal A_ij+A_ji and diagonal A_ii, exactly matching `pack(A+A.T)` with its diagonal halved. No conjugation may be introduced. Packed callback is used only when the native coordinate path is valid; the old density callback remains usable for fallbacks. Keep tests sensitive to diagonal weighting, shell/subrange boundaries and complex/nonsymmetric behavior.


## Execution ledger

- 5826633: expected missing-helper failures, 2 selected tests failed as intended; compute node only.
- 5826749: first kernel comparison passed output checks. Complete J/K density median .17806 -> .08251 s with packed J/native packing. Blanket GEMM flattening showed no consistent gain.
- 5826824: initial 18 packed-adjoint derivative/fallback cases passed. Subsequent static review found overlapping output buffers; fixed with rejection and two extra regressions (20 new cases now).
- 5826921: seven-sample packed half-transform comparison passed numerical checks but did not consistently beat original factor contractions; these alternatives remain benchmark-only and are not adopted.
- Independent review: assembly-review.md, no remaining static blockers. No native rebuild is required; existing NPdunpack_tril_2d_vjp symbol supplies fused packing.
- 5826994: final broader regression job completed: 224 passed, 0 failures/errors/skips, 183.34 s; condition `contractions/regressions`. Tested source hashes match current candidate.
- 5826995: paired baseline/candidate full dCMP running on hpc-20-35, condition `contractions/paired`. Both source snapshots have identical timing hooks and additional initial-loop read/unpack/dense diagnostics. Candidate production delta is packed assembly only; initial contractions remain unchanged.

Remaining: wait for both jobs, analyze full precision JSON and gradient/accounting checks, determine if performance evidence warrants adoption or reversed-order confirmation, install only validated production files/tests with rollback/hashes, report. Current live package remains untouched. External base: `/home/bkhou/DLNO_CCSDT/test_pyscfad_stc_mp2/runtime_test/dCMP-ooc/test_reverse/contractions`.

Paired baseline complete: total710.009792s, SCF reverse332.681601s, mixed assembly63.251922s, conservative SCF264.225159s. Full gradient and timing checks passed. Initial factor prefix77.233657s = read16.885288 + unpack7.752372 + dense52.559846 + small overhead. Candidate is running in the same allocation.

Paired5826995 completed0:0. Candidate total523.583723s, SCF reverse305.381145s, assembly39.742767s vsbaseline710.009792/332.681601/63.251922. Max pairedgradientdifference5.36754e-12, energy2.27374e-13; RSS5.61261->5.65564GiB. Unchangedforward290.80->145.81s proves substantialenvironmentvariation; do notattributealltotalspeeduptocode. Assemblyimproved23.51s butunchangednativecontraction32.83->51.50s offsetsmostofthat. Followupwholeassembly+nativekernelcomparison5828088 running in conditionassembly_native tocompare currentpackedJ16, packedJ1, fullJnativepacking, andoriginal onsamerealgeometryblocks. No liveinstallationyet; selectbestnetvariant thenverifyanynewchangesbeforeadopting.

Whole-geometry-block check5828088 completed with valid packing/coordinates. Combined three-block medianseconds: original.714894, packedJ16.659504, packedJ1.482990, nativefullJ.604680. Adopt packedJ1 as candidate: restrict only rank2 Coulombupdate toBLAS1; retainKexpansion16/nativeOMP16. Preservev1purecode in candidate_packed_j16. Finalfulljob5828194 and224-regressionjob5828195 submitted; awaitbothbeforeliveinstallation. Finalproductiondelta stillonly _cderi_vjp.py, _df_jk_factorized.py, andnewtest_reverse_packed_density.py. Existingnativepackingkernelunchanged; no rebuild.


## Completion

Final full dCMP job 5828194 and regression job 5828195 completed with exit code 0:0. All 224 targeted tests passed. All 102 final gradient components agree with the saved reference within 3.93e-12 Eh/bohr. Final full runtime is 479.14 s, conservative SCF reverse without DF is 187.91 s, and SCF forward without DF is 142.35 s. Peak RSS is 5.53 GiB.

Adopted the packed-adjoint implementation with scoped BLAS1 for its rank-two Coulomb update. Installed the two production modules and new regression test, after checking that their hashes equal the tested candidate. All other baseline source/native files were preserved. No native rebuild or production submission edits were needed. Read README.md for measured improvements, rejected alternatives, and the limitation on interpreting full-run speed differences. Installation and rollback provenance are recorded in installation.json.
