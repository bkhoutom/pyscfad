# DLNO-STC clean integration audit (2026-11-17)

Audited immutable inputs:

- Common base: `8ade7f4b2765d0d771f2276662cf6d66b912785f`.
- Source `dlno-stc`: `c4f64d13377ebaf683659aecf695709a7db8bd8a`.
- Authoritative fix baseline: `551cd2d4c263b324d68dbfbc7fdec53fff87edf7`.
- Local `dlno-stc` equals its fetched remote tip; `git rev-list --left-right --count origin/dlno-stc...dlno-stc` returned `0 0`. No unpushed local commits exist.

The audit used commit objects, commit diffs, source inspection, and read-only `git merge-tree --trivial-merge`. Early filesystem reads preceded the integration; once its merge began, baseline conclusions were checked against `git show 551cd2d4:path`. Numerical results in historical source reports are not fresh clean-branch validation.

## Commit-by-commit disposition

| Commit | Classification | Evidence and integration recommendation |
| --- | --- | --- |
| `11c00a34c68e3f08a99e989de234cb825999b3e9` Interpret STC PAO cutoff as a squared overlap norm | Already incorporated | Fix tip `551cd2d4` implements the same `BoysDomainThresholds.pao_norm=1e-4`, strict `norm2 > pao_norm` in `_selection.build_domain_selections` and `domain._build_selection_reference`, retained square-root normalization, `individual-partners-pair4-pcd-aux-pao-norm2-v3` guards, and restart ABI 6. `test_ed_metadata.py` is identical between these commits. `test_pao_norm_threshold.py` differs only by migration from `pair_energy_model="all", force_full_domains=True` to `full_domain=True`. Keep baseline implementation, docs, tests and migrated API. |
| `63d458e2241380bb1d7edf430963ade40bc2811e` Bucket primary-response workspaces with original-frame fallbacks | Production superseded; useful numerical regression adaptation | `_primary_kernels.py` and old `_primary_response._select_preprojection/_select_target_bucketed` use padded JAX algebra and special original-frame fallbacks. Fix `_primary_response.prepare_primary` instead calls `_weak_numpy.preprojection` and `.target`; those use unpadded host factorizations and explicit VJPs. Omit old production helpers and restore neither old selector. Adapt supported real numerical cases from `test_primary_kernel_buckets.py` to host-vs-test-oracle comparisons. |
| `441da7d8692aead722be92843423a89edc20ce80` Scale primary workspace bucket widths with domain size | Bucket policy superseded; response assertion useful | `_primary_kernels._size` changes 32-element buckets to relative spacing. Fix host weak algebra has no JAX workspace buckets, so retain none of this policy. Its `test_primary_execution` enhancement checks actual retained `s1e`, `fock`, `pao_coeff` cotangents rather than relying solely on a particular finishing function; that observable response check is useful in the current test. Historical LLVM plan is not current guidance. |
| `8c55ea5d3f748dfe4a39772214762d54b390a369` Add native weak-moment contractions with fixed translation tiles | Production superseded; real endpoint regressions useful | `_weak_moments.linearize` contracts physical AO arrays on the host but translates in 128-column JAX tiles. Fix `lib/_multipole_numpy.endpoint` performs contractions, translation and reverse entirely in NumPy, handles orders 2/3/4 and both packed/full Cartesian inputs. Its real nonsymmetric AO-operator cotangents, empty virtuals, repeated seeds and directional finite difference remain valuable; adapt them to the host API. Omit `_weak_moments.py` and the unneeded `_multipole._translate_moments` extraction, keeping the baseline JAX oracle unchanged. Complex and traced-input fallback tests are incompatible with the current explicit real API. |
| `9ee9a6659135524c7fd9738b9a70fbd885d77f13` Use native weak moments in endpoint replay and grouped gradients | Production superseded; grouped molecular regression useful | Old `_pairs._linearize_weak_moments`, crop and mutable accumulation helpers dispatch to native or JAX fallbacks. Fix `_pairs._weak_moment_vjp` calls `_multipole_numpy.endpoint(..., with_grad=True)` directly; `_prepare_weak_endpoint`, `_replay_weak_endpoint`, `_grouped_weak_pullback` use host primary tapes and existing native coordinate moment VJP. Keep fix production. Adapt shared-support grouped/nongrouped molecular replay regression from `test_weak_moment_native.py` to host endpoint/VJP interception while using the independent JAX test oracle. Keep growth-in-physical-size/no-compilation as a host-path regression if desired, without restoring padding/crop contracts. |
| `300f326eb49ff3180f0ecdf3fe40ac58e2c982a8` Record reviewed weak-moment integration and final validation | Intentionally omitted historical completion records | Only old plan/report files change. They describe tests, pushes, compile behavior and production dispatch on the prior branch. Preserve attribution in this integration report; do not install them as current evidence or current implementation documentation. |
| `c4f64d13377ebaf683659aecf695709a7db8bd8a` Remove obsolete DCSD and domain-selection tests | Useful cleanup, with selective adaptation instead of wholesale deletion | Two native-domain parity tests and their unused fixture compare incompatible PAO conventions and are obsolete. Two Boys tests contain valid ordering, partition and rank checks but assume eagerly populated ED labels; preserve those checks after materializing selection. DCSD remains a class and has an explicit reference route, so retain its independent numerical gradient regression with custom DF-CCSD response disabled. Details below. |

## Fix-only commits retained in full

Both sides were compared to `8ade7f4b`. These four fix-only commits form the
implementation baseline and are retained without production changes:

| Commit | Retained changes |
| --- | --- |
| `c141a4f7e74719ebf3924f8d2e45d97cea58b6ba` | Unified fragment workflow; W/LIS reuse; saved MP2 and triples grids; removal of obsolete workflows; migrated examples/tests. |
| `37f8b39b0746e4077875aadb75de82a66f026112` | Aligned SGX defaults and bulk-water domain documentation/examples. |
| `d2a410fa02cbbe3b62b9744370922247d40b96ce` | MRCC deterministic triples; explicit NumPy/SciPy weak forward/reverse; DF contraction/native-team improvements and their tests. |
| `551cd2d4c263b324d68dbfbc7fdec53fff87edf7` | Squared PAO norm cutoff and restart incompatibility safeguards, adapted to cleaned APIs. |

## Numerical cases worth carrying forward

Use `_weak_numpy.target/preprojection` as production and `test/_weak_reference.py` as the JAX oracle. Both numerical values and all independent input cotangents must be compared; raw overlap/product perturbations should not be symmetrized accidentally.

- Real primary targets across small/large physical sizes, singleton/full/rank-deficient spans and permuted AO support. Compare physical labels, occupied values, virtual energies and gauge-invariant virtual projectors/energy-weighted projectors, with the ambient overlap/Fock/candidate/RHS response.
- Zero occupied or zero virtual rank must produce an absent screen and matching labels; padding must not create a physical mode (now checked without padding).
- Real preprojection with identical and different support/representation, rank-deficient PAO Gram, reordered columns, and independent Gram/overlap-product response.
- Degenerate projected spectrum whose eigenvector-wise completeness cutoff can split a subspace. Verify current host completeness labels and selected subspace against the original unpadded oracle; never reintroduce padded eigensolver routes to make this test fit.
- Degenerate virtual Fock denominators, including the upstream preprojection frame, and correct regularized response in that same frame.
- Real nonsymmetric packed moment operators, zero virtuals and varied virtual widths, repeated pullback seeds without modifying seeds/tape, and an independent directional finite difference.
- Shared molecular AO supports with both grouped and nongrouped replay, including reordered support `(2, 0)`, different endpoint widths, and nonzero nuclear plus orbital response.
- Growing physical sizes should not invoke `jax.vjp` or compile weak numerical kernels. Baseline `test_weak_numpy.test_molecular_host_weak_forward_reverse_never_calls_jax_vjp` already checks all multipole orders; retain it.

Intentional non-ports: complex, float32, mixed-dtype, traced-primal and traced-cotangent fallback support. `_weak_numpy.require_supported` explicitly rejects complex/non-float64 inputs and JAX tracers; README documents the explicit coordinate VJP contract. Order-2/3 and full Cartesian cases are supported by the new host route and already covered in `test_weak_numpy.test_endpoint_host_reverse`; they should not be called unsupported.

## Cleanup without losing valid tests

`c4f64d13` deletes exactly five test functions, not entire Boys/parity coverage:

1. `cc/test/test_dcsd.py::test_df_nuc_grad`: preserve and adapt. `dfdcsd.RDCSD` still subclasses `dfccsd.RCCSD` and returns `dcsd=True`. Default custom response fails by design: `_dfccsd_kernel_custom_bwd` requests saved update intermediates; `ccsd.update_amps` rejects DCSD there, and `_dfccsd_update_amps_eris_cotangent` explicitly rejects DCSD. `_dfccsd_native.enabled` also excludes it. However `dfccsd.RCCSD.ccsd` retains the superclass/reference path when `pyscfad_dfccsd_custom_response=False`. The test's fixed nuclear-gradient expectation remains a useful independent reference. The integration wraps the full `jax.grad(energy)(mol)` evaluation in that context, without changing production code or tolerance.
2. `test_boys_construction.py::test_public_selection_preserves_permuted_labels_and_concrete_anchors`: preserve valid permuted-label/ordered occupied-frame and normalization checks, but materialize domain selections before inspecting `virtual_anchor_columns`. Current ED MP2 allows occupied–virtual overlap, so check each subspace's normalization rather than assuming their concatenation is orthogonal.
3. `test_boys_construction.py::test_selection_rejects_invalid_target_partition_and_rank_loss`: preserve invalid target partition/span assertions and materialize first before injecting bad `strong_occ_metric_keep`; a pending ED plan correctly recomputes labels and therefore cannot exercise saved-label rejection.
4. `test_domain_parity.py::test_boys_localization_and_common_arrays_match_native`: retire the cross-implementation expectation because native DLNO uses a different parent PAO norm convention (historical evidence showed 12 versus 14 PAOs).
5. `test_domain_parity.py::test_fixture_supports_weak_spaces_and_target_mapping_match_native`: retire for the same reason, removing only its unused `native_case` fixture and `dataclasses.fields` import.

Preserve all remaining Boys tests: standalone system-frame response; empty/explicit frozen indices; complex occupation/target/Fock rejection; DFT rejection; integer-overflow label checks. Preserve all remaining parity tests: ordered projected-Cholesky occupied frame, selected Fock-direction finite difference (both steps), and full coordinate-response finite difference. Current `test_ed_streaming.py` already verifies pending labels, frame/tape reuse and unchanged value/response; `test_domain_grouping.py::test_grouping_preserves_order_and_rejects_invalid_saved_labels` covers saved rank rejection; `test_atom_grouping.py` checks permuted label membership. Do not drop these files wholesale.

## API migrations that must survive integration

- `pao_norm` now means squared overlap norm, strict `>`, default `1e-4`; square old explicit norm cutoffs. Restart ABI 6 and domain convention v3 reject old saved selections.
- `force_full_domains` -> `full_domain`; `scope="system"` -> `boys_grouping="system"`; `molecule` remains a grouping alias. Full domains expand supports without collapsing fragment membership.
- One fragment pipeline: no `single_pass`, W prepass, exported/imported solver replay, per-domain MPI replica averaging, or public fragment-subset/correction-only mode. Keep standalone MP2 and CC `kernel`/`value_and_grad` APIs.
- Removed fixed arguments `matrix_kind`, `smooth`, `lis_space`; `smooth_laplace_points` -> `triples_laplace_points`. MP2/W share saved `mp2_grid`; triples use independent saved per-fragment grids. CC schema 18 and checkpoints remain.
- Weak functions require concrete real float64 and explicit `weak_pair_energy_and_grad`; JAX remains a test oracle, not a production fallback. Coordinate-only basis leaves; actual endpoint dimensions without buckets.
- Fresh global domain selection is an ED plan. Use `materialize_domain_selections` for inspecting saved ranks/anchors; production selects live per-fragment frames and consumes the same pullback.
- Keep MRCC triples, SGX defaults and updated example workflows from the fix branch; source-only commits provide no reason to restore earlier triples/workflow paths.

## Exact normal-merge risks

An ordinary three-way merge does NOT automatically restore files deleted only by the fix branch and untouched on the source since the common base. The intersection of fix deletions and source-touched paths is empty. In particular it preserves deletion of these 14 paths:

- `examples/dlno_stc/workflow.py`
- `pyscfad/dlno_stc/exchange.py`
- `pyscfad/dlno_stc/finite.py`
- `pyscfad/dlno_stc/mp2.py`
- `pyscfad/dlno_stc/parallel.py`
- `pyscfad/dlno_stc/test/mpi_smoke.py`
- `pyscfad/dlno_stc/test/test_exchange.py`
- `pyscfad/dlno_stc/test/test_export_auxiliary_reuse.py`
- `pyscfad/dlno_stc/test/test_finite_workflow.py`
- `pyscfad/dlno_stc/test/test_mp2_replay_response.py`
- `pyscfad/dlno_stc/test/test_mpi_abort.py`
- `pyscfad/dlno_stc/test/test_single_rank.py`
- `pyscfad/dlno_stc/test/test_triples_quadrature_reuse.py`
- `pyscfad/dlno_stc/test/test_workflow.py`

The merge DOES automatically add obsolete source-only files unless explicitly omitted:

- `pyscfad/dlno_stc/_primary_kernels.py`
- `pyscfad/dlno_stc/_weak_moments.py`
- `pyscfad/dlno_stc/test/test_primary_kernel_buckets.py`
- `pyscfad/dlno_stc/test/test_weak_moment_kernels.py`
- `docs/superpowers/plans/2026-11-16-native-weak-moments.md`
- `docs/superpowers/plans/2026-11-16-omol3tdx-llvm.md`
- `docs/superpowers/reports/2026-11-16-native-weak-moments-validation.md`
- `docs/superpowers/specs/2026-11-16-native-weak-moments-design.md`

`test_pao_norm_threshold.py` is added on both branches; keep the fix version with migrated `full_domain`. `_multipole.py` would auto-merge the translation-helper extraction because fix left it unchanged; retaining baseline avoids an unused oracle refactor. `_pairs.py`, `_primary_response.py`, `test_primary_execution.py`, and `test_weak_moment_native.py` can reintroduce obsolete dispatch/test API assumptions through conflict resolution. README resolution must not restore old scope/export/JAX fallback instructions. `test_domain_parity.py` accepts the source cleanup automatically; `test_dcsd.py` is automatically removed, hence the explicit preservation. Boys construction deletion conflicts with fix API edits and needs the selective adaptations above.

## Exhaustive source-delta path disposition (24 paths)

These are all paths changed between the common base and the source tip; status describes the intended clean integration, not an assertion that final verification has run.

| Path | Disposition |
| --- | --- |
| `doc/source/user_guide/smooth_lis_triples.md` | Keep fix version: squared-norm documentation already incorporated; preserve new workflow/triples API. |
| `docs/superpowers/plans/2026-11-16-native-weak-moments.md` | Omit superseded historical plan. |
| `docs/superpowers/plans/2026-11-16-omol3tdx-llvm.md` | Omit superseded bucket plan. |
| `docs/superpowers/reports/2026-11-16-native-weak-moments-validation.md` | Omit old validation as current artifact; cite history in integration report. |
| `docs/superpowers/specs/2026-11-16-native-weak-moments-design.md` | Omit superseded native/JAX-tile design. |
| `pyscfad/cc/test/test_dcsd.py` | Preserve baseline numerical expectation; adapt to reference response configuration. |
| `pyscfad/dlno_stc/README.md` | Keep fix version: explicit NumPy weak API, unified pipeline and PAO migration. |
| `pyscfad/dlno_stc/_pairs.py` | Keep fix production unchanged. |
| `pyscfad/dlno_stc/_primary_kernels.py` | Omit padded JAX production helper. |
| `pyscfad/dlno_stc/_primary_response.py` | Keep fix host dispatch unchanged. |
| `pyscfad/dlno_stc/_restart.py` | Keep fix ABI 6 and migrated selection conventions unchanged. |
| `pyscfad/dlno_stc/_selection.py` | Keep fix squared-norm selection and unified fragment API unchanged. |
| `pyscfad/dlno_stc/_weak_moments.py` | Omit fixed-JAX-tile helper; host multipoles supersede it. |
| `pyscfad/dlno_stc/domain.py` | Keep fix squared-norm response and deferred ED semantics unchanged. |
| `pyscfad/dlno_stc/driver.py` | Keep fix domain convention guards and unified driver unchanged. |
| `pyscfad/dlno_stc/test/test_boys_construction.py` | Adapt two eager-label tests to materialization, retain remaining tests. |
| `pyscfad/dlno_stc/test/test_domain_parity.py` | Remove exactly the two incompatible native parity comparisons plus unused fixture/import; retain numerical frame/response tests. |
| `pyscfad/dlno_stc/test/test_ed_metadata.py` | Keep fix unchanged; same PAO convention/ABI regression already present. |
| `pyscfad/dlno_stc/test/test_pao_norm_threshold.py` | Keep fix version with migrated full_domain API. |
| `pyscfad/dlno_stc/test/test_primary_execution.py` | Adapt observable retained-cotangent check from source; preserve current host/concrete dispatch checks. |
| `pyscfad/dlno_stc/test/test_primary_kernel_buckets.py` | Omit obsolete test file; adapt supported numerical cases in host-focused regression tests. |
| `pyscfad/dlno_stc/test/test_weak_moment_kernels.py` | Omit obsolete test file; adapt supported real numerical cases in host-focused regression tests. |
| `pyscfad/dlno_stc/test/test_weak_moment_native.py` | Preserve baseline native-coordinate tests and adapt shared molecular replay case to host endpoint pullback. |
| `pyscfad/lib/_multipole.py` | Keep fix/baseline oracle unchanged; extraction is unnecessary without tile helper. |

## Final implementation decisions

Production Python, C/C++, examples and API documentation remain byte-for-byte
identical to `551cd2d4`. Only tests and integration records intentionally differ:

- `test_domain_parity.py`: retire two comparisons to incompatible native DLNO
  conventions; retain occupied-frame and finite-difference response tests.
- `test_boys_construction.py`: materialize deferred ED labels, test each space's
  normalization, and point the stale system-frame assertion at `system_reference`.
- `test_primary_execution.py`: retain host routing checks and assert finite,
  nonzero saved overlap, Fock and PAO cotangents.
- `cc/test/test_dcsd.py`: retain DCSD gradient coverage with its reference response.
- `test_host_carryover.py`: adapt useful source cases to concrete real float64
  host APIs, including shared molecular endpoint response.
- This audit, the execution plan and final validation report record provenance.

No optional production optimization is included. Batched NumPy moment contractions
remain a follow-up: benchmark representative AO/virtual sizes, memory and reverse
cost against `_multipole_numpy.endpoint`, preserve all ambient cotangents, and
commit separately only if repeatable end-to-end benefit is demonstrated. No such
benefit is claimed here.

Ordinary merge `5b319107` reconciles both recorded source tips (the original local
`dlno-stc` tip equals its remote). The audit was completed before that merge and
is committed with its numerical test adaptations. Source branches stay untouched.
Fresh verification appears in `2026-11-17-dlno-stc-clean-validation.md`.

Carryover size choice: bucket-specific 193/211/259-row primary cases were reduced
to at most 67 target rows and 19 preprojection rows because the host path has no
bucket boundary. Varied endpoint widths retain 127/128/129 and reach 259 for the
compilation-growth check. These are bounded workstation regressions, not a claim
about HPC-scale memory or end-to-end performance.

## Additional baseline fixture repair found during final validation

`test_mpi_weak.py` was unchanged between the source tips, but its synthetic
fixtures still expected the pre-NumPy response boundaries. A fresh four-rank
run failed 10 of 12 cases on every rank: missing fragment metadata, JAX AD
through a host-only pair function, and a guard on a removed allocation helper.
The real molecular domain/pipeline MPI checks passed independently.

The test-only repair supplies explicit screen/raw/moment tapes at current host
boundaries, uses the independent JAX pair oracle, and guards the current host
zero allocator. All original tests and assertions remain, including asymmetric
pair weights, idle owners, collective failures, variable buffers and tape
release. Production endpoint exchange, replay and pair derivatives remain active.
After review, all 12 cases passed on each of four ranks and the bounded subprocess
wrapper passed. This is an additional intentional test difference from the fix
baseline, not a production modification.
