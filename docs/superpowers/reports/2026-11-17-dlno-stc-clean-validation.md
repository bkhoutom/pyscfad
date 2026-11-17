# DLNO-STC clean validation — 2026-11-17

**Result:** 462 top-level pytest checks passed, including six MPI launch checks;
48 additional direct weak-MPI case executions passed (12 on each of four ranks);
3 CTests passed. No selected final failures, errors or skips remain.

The tested code/test commit is `7f07f47e09813b9dbf3e186c441b44013c059f86`. The final delivery commit adds only
this evidence and completes the execution record. All production code, native
sources, examples and API documentation are byte-identical to the fix baseline.
No production changes followed validation. History reconciliation was complete
before the final runs; later test-only repairs were independently reviewed and
rerun. Earlier historical branch reports were not used as pass evidence.

## Sources and preservation

| Reference | Recorded SHA |
| --- | --- |
| Common ancestor | `8ade7f4b2765d0d771f2276662cf6d66b912785f` |
| Fetched `origin/dlno-stc-fix` | `551cd2d4c263b324d68dbfbc7fdec53fff87edf7` |
| Fetched `origin/dlno-stc` and original local `dlno-stc` | `c4f64d13377ebaf683659aecf695709a7db8bd8a` |
| Ordinary merge | `5b319107fdfa8a193551ce797837b219971e7fe5` |
| Adapted regression cases and audit | `0570c146` |
| Additional weak-MPI fixture repair | `7f07f47e09813b9dbf3e186c441b44013c059f86` |

There were no unpushed local source commits. The merge parents are the two
recorded remote tips, so both and the original local tip are ancestors of HEAD.
The original checkout remains clean on `dlno-stc`; its index and source ref are
unchanged. No local `dlno-stc-fix` ref existed, and none was created. Remote source
refs were rechecked unchanged before delivery. No source branch was updated,
renamed, archived or deleted. No force push is used.

Worktree: `/home/bkhou/Project/DLNO_CC/pyscfad-dlno-stc-clean`.
Branch: https://github.com/bkhoutom/pyscfad/tree/dlno-stc-clean.
The [audit](2026-11-17-dlno-stc-clean-audit.md) accounts for all seven old-only
commits, four fix-only commits and all 24 old-source delta paths.

## Environment and native provenance

Printed and checked during validation:

```text
Python: /home/bkhou/miniforge3/envs/pyscfad_env/bin/python
pyscfad: /home/bkhou/Project/DLNO_CC/pyscfad-dlno-stc-clean/pyscfad/__init__.py
pyscfadlib: /tmp/dlno-stc-clean-integration/support-source/pyscfadlib/__init__.py
STC: /home/bkhou/Project/DLNO_CC/pyscfad-dlno-stc-clean/pyscfad/dlno_stc/_stc_mp2.cpython-311-x86_64-linux-gnu.so
CCSD: /home/bkhou/Project/DLNO_CC/pyscfad-dlno-stc-clean/pyscfad/cc/_ccsd_kernels.cpython-311-x86_64-linux-gnu.so
libcgto_vjp.so: /tmp/dlno-stc-clean-integration/support-source/pyscfadlib/libcgto_vjp.so
libao2mo_vjp.so: /tmp/dlno-stc-clean-integration/support-source/pyscfadlib/libao2mo_vjp.so
```

Python 3.11, JAX/JAXlib 0.10.2, NumPy 2.4.6, SciPy 1.17.1, PySCF 2.14.0,
pytest 9.1.1, mpi4py 4.1.2 and Open MPI 5.0.10. The existing compatible environment
was used without installs or changes to its editable source. A plain import from
`/tmp` without the validation PYTHONPATH still resolves to the original checkout
and installed support package.

Rebuilt 10 support-library targets (including both changed VJP sources and the
SciPy LAPACK bridge), STC and optional CCSD kernels. CMake used at most two compile
jobs. The unchanged SGX artifact was copied only after matching 317 source files
to its recorded build source and verifying binary SHA256
`9f541ef8307f0927485f4d1b7b25fdbfe4088126ae5427fe89122c424716966c`.
The external libcint dependency was copied unchanged. Armadillo/pybind11 and
associated native dependencies were read from the original build dependency
cache, never rebuilt there. All 1,472 audited original native/environment files
have identical before/after hashes. The JSON evidence retains native paths,
hashes, versions, reuse provenance and exact build/validation recipes.

Current-workstation activation (applies only to this shell):

```bash
cd /home/bkhou/Project/DLNO_CC/pyscfad-dlno-stc-clean
source /tmp/dlno-stc-clean-integration/native-env.sh
"$DLNO_PYTHON" -c 'import sys,pyscfad; print(sys.executable, pyscfad.__file__)'
```

The support staging area and detailed logs are under `/tmp`; retain or rebuild
them before reusing this activation after temporary-file cleanup. The committed
JSON includes the exact environment, captured build recipes and their helper
scripts. Rebuilding also requires the recorded compatible dependencies/cache;
run the included audit helper to capture a fresh before-state before rebuilding. Do not redirect the
original editable installation merely to validate this branch.

## Focused results

Every row used a separate Python process. All rows passed; commands, JUnit hashes
and exact timings are in [the evidence manifest](2026-11-17-dlno-stc-clean-validation.json).
RSS is per-process maximum in MiB, not aggregate system memory.

| Group | Passed | Wall seconds | Peak MiB |
| --- | ---: | ---: | ---: |
| `cc_checkpoints` | 3 | 9.1 | 879 |
| `dcsd` | 1 | 20.9 | 1863 |
| `domain_parity` | 4 | 19.7 | 1673 |
| `fd_ccsd` | 1 | 27.9 | 2079 |
| `fd_ccsd_t` | 1 | 26.9 | 2125 |
| `fd_mp2` | 1 | 21.3 | 1680 |
| `fragment_pipeline` | 15 | 177.9 | 3050 |
| `independent_mp2` | 1 | 24.7 | 1591 |
| `lis_reuse` | 9 | 36.8 | 2559 |
| `modes` | 16 | 46.3 | 2290 |
| `mrcc` | 17 | 22.8 | 1520 |
| `native_ccsd_t` | 1 | 67.8 | 3383 |
| `pao_reuse` | 46 | 14.6 | 1109 |
| `pao_selection` | 60 | 25.1 | 2073 |
| `primary_response` | 7 | 15.3 | 1242 |
| `restart` | 13 | 1.1 | 283 |
| `saved_grids` | 2 | 10.8 | 938 |
| `w_baseline` | 4 | 2.8 | 479 |
| `w_native` | 39 | 4.3 | 594 |
| `weak_endpoints` | 50 | 10.8 | 861 |
| `weak_native` | 20 | 1.8 | 443 |
| `weak_numpy` | 31 | 14.7 | 1386 |
| Adapted host carryovers | 21 | 28.6 | 672 |
| Native DF, half-transform and SGX | 93 | 6.1 | 325 |
| MPI domain/pipeline/subprocess launch checks | 6 | 244.4 | See aggregate below |

Coverage includes PAO squared norms, retained-rank changes, degeneracies and empty
virtuals; all weak-pair/endpoint/common cotangents; grouped versus individual
molecular replay; one-time W/LIS construction and saved grids; checkpoint round
trips and rejection of changed equations, conventions, native versions, ladder
settings and scientific controls. MRCC tests compare energies and all nine
prepared-input bars with independent expressions. DCSD retains its reference
response regression. Native tests cover DF slice/packing adjoints, thread-team
restoration and SGX contractions, coordinate derivatives and defaults.

## Matching settings and measured numerical checks

- Saved-grid MP2, CCSD and CCSD(T): asymmetric water, STO-3G, Weigend DF, frozen=1,
  singleton physical fragments with `full_domain=True`, LIS cutoffs zero,
  `merge_identical=False`, HF included, SCF tolerances 1e-12/1e-9. The same saved
  five-point MP2/W grid and three-point per-fragment triples grids are used for
  values, gradients and displaced geometries; refitting is forbidden in the
  tests. Deterministic CCSD/MRCC triples, SGX off. Central differences use
  h=1e-4 Bohr and pass 3e-6 Eh/Bohr absolute tolerance. This tests the fixed-grid
  functional, not infinite-quadrature convergence.
- Independent system MP2: the same frozen space, Weigend fit and explicit
  64-point quadrature for both value/response; inverse-denominator and derivative
  approximation checked before comparison. Energy agrees with DFMP2 within
  2e-10 Eh, coordinate acceptance uses atol=2e-7 Eh/Bohr and rtol=2e-5; central FD
  passes 1e-6 Eh/Bohr.
- Finite-domain water dimer: STO-3G, frozen=2, pair cutoff 5e-5, fixed permuted
  singleton labels, identical saved supports and explicit 64-point quadrature,
  default matching density-fit basis and SGX off. Correlation energy
  **-0.07406737501085058 Eh**. Selected H-z derivative
  **-0.016822148415105716 Eh/Bohr**; FD errors are **2.35e-10** and **1.12e-9**
  Eh/Bohr at h=1e-5 and 5e-6. Fock-direction FD errors are 1.17e-11 and 3.14e-12.
- Native versus reference CCSD(T): identical saved selections, controls and
  deterministic triples with `integral_transform='rotate'`, direct local DF.
  Energy difference is **0 Eh** and maximum gradient difference
  **5.58e-10 Eh/Bohr**. The fragment pipeline separately agrees with saved-label
  execution to 6.15e-14 Eh and 2.76e-12 Eh/Bohr.
- MPI uses the same saved/scientific settings in serial and parallel runs.
  Domain MP2 at 1/2/4 ranks: total-energy differences zero, maximum total-gradient
  difference **1.96e-14 Eh/Bohr**. CC pipeline at 2 ranks (both SCF J/K modes)
  and 3 ranks (gradient with empty owner): maximum energy difference
  **2.28e-14 Eh**, gradient **4.57e-16 Eh/Bohr**. Weak owner tests cover all
  three multipole orders, asymmetric weights, idle rank, tape release and
  synchronized build/pair/replay/allocation failures.

## Failures, exclusions and resource limits

**Pre-existing and repaired:** a byte-identical copy of the fix baseline's Boys
tests reproduced three failures: inspecting pending anchors, corrupting pending
rather than saved rank labels, and asserting a moved production symbol.
The adapted final tests pass. The unchanged baseline weak-MPI fixture suite
initially failed 10 of 12 cases per rank (40 failures across four ranks); its
old callback/AD/allocation assumptions were migrated, preserving every assertion.
The final run passes 12/12 on every rank, and its bounded launcher also passes.

**Introduced failures remaining:** none in the selected checks. New carryover
oracle setup issues encountered while writing tests were corrected without
production edits. Independent review has no remaining critical/important findings.

**Unavailable/skipped:** no selected final check skipped. Sandbox PMIx socket
restrictions were resolved by authorized local MPI execution outside the sandbox;
MPI was not marked unavailable. The three saved-grid FD cases were excluded from
one grouping only to run each separately. The full unrelated repository suite and
HPC-scale benchmarks were not run. Weak complex/float32/traced AD and higher-order
response remain outside the cleaned API's supported contract.

The workstation has 16 CPUs and about 30.5 GiB RAM. Serial test peak was **3.30 GiB**;
maximum sampled MPI process-tree peak was **4.30 GiB** (10 GiB guard never reached).
Rank threads were capped at one; ordinary tests used one or two OpenMP threads
and one BLAS thread. Native CTest concurrency cases intentionally create up to
32 worker threads but ran on two pinned CPUs. No HPC-sized calculation was launched.
The host endpoint growth regression observes JAX compilation logging, then confirms
zero new compilations through 133 AO rows/259 virtual columns; the full carryover
process peaked at 672 MiB. This is a bounded regression, not a performance claim.
Batched NumPy moments remain an unbenchmarked follow-up, with no optimization commit.

## Existing-script migration

| Old input/use | Clean input/use |
| --- | --- |
| `force_full_domains=True` | `full_domain=True` |
| `scope='system'` | `boys_grouping='system'` (or supported molecule alias) |
| `scope='domain'` | Omit `scope`; pass the domain plan/static selection |
| `smooth_laplace_points` | `triples_laplace_points` |
| Fixed `matrix_kind`, `smooth`, `lis_space`, `single_pass` switches | Remove them; use the unified workflow |
| Inspect anchors/ranks immediately after domain selection | Call `materialize_domain_selections` first |
| Different defaults/old checkpoints | Set matching grouping, thresholds, fit, grids and triples mode; rebuild incompatible selections/checkpoints |

`pair_energy_model='all'`/`'all-strong'` remain supported and need no renaming.
Specify `thresh_occ` and `thresh_vir` explicitly for CC plans. Deterministic triples
use MRCC; old asymmetric deterministic controls and removed export/replay/subset
workflows need migration to supported entry points. Weak response uses concrete
real float64 plus explicit VJPs. Current `dlno-stc` already has the squared-PAO
cutoff correction: **do not square an already converted cutoff again**. Only
scripts predating that correction must square an old norm-valued threshold.
Keep SCF checkpoints when compatible, but rebuild old domain/LIS selections and
correlation checkpoints rejected by ABI 6/domain convention v3/CC schema 18.
