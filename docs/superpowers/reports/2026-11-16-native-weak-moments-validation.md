# Native weak-moment integration validation

Target: `dlno-stc`, based on `11c00a34`, including primary workspace bucketing
(`63d458e2`, `441da7d8`) and native moment contractions (`8c55ea5d`).

Local environment: Python 3.11, JAX 0.10.2, NumPy 2.4.6, CPU float64,
AMD Ryzen 7 7800X3D. Numerical checks used CPU affinity 0–3,
`OMP_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`PYSCFAD_DENSE_BLAS_THREADS=4`, `JAX_PLATFORMS=cpu`, and
`JAX_ENABLE_X64=True`. The integration probes disabled the persistent JAX
compilation cache and did not clear in-process caches.

## Numerical and compilation checks

- Initial affected baseline: 81 tests passed.
- Kernel plus shared multipole/endpoint tests: 59 passed. Includes real and
  complex nonsymmetric AO operators, all input cotangents, zero virtuals,
  127/128/129-column boundaries, repeated pullback seeds, and a directional
  finite difference.
- Integrated weak endpoint/moment/product tests: 128 passed. Includes grouped
  and individual replay, all molecular/shared-orbital contributions, traced
  seeds and primals, unsupported input fallback, and no new compilation when
  physical dimensions change after warm-up.
- Broader primary/selection/domain/shared multipole/threading tests: 166
  passed across the initial batch, remaining batch, and isolated RI check.
  Four domain-selection failures also occur on the untouched target (below).
  The MPI subprocess wrapper passed with four ranks and checks for 12 passed
  collective tests per rank; 11 MPI-only cases skip in the serial parent.

These are separate test invocations with some overlap, not additive totals.

Complete deterministic MP2 calculations used the integrated production code,
without the prototype adapter. Each molecule ran once cold and twice warm.
Native forward and reverse calls were observed in every calculation. Domain
selections and weak-pair sets matched the saved original implementation.

| Case | AO | Weak pairs | Maximum energy difference (hartree) | Maximum gradient difference (hartree/bohr) |
| --- | ---: | ---: | ---: | ---: |
| Water dimer / cc-pVDZ | 48 | 16 | 1.71e-13 | 1.57e-13 |
| Four waters / cc-pVDZ | 96 | 96 | 2.28e-13 | 3.56e-12 |

The same-process endpoint probe introduced physical shapes `17x5`, `143x129`,
`282x79`, `564x158`, `2256x435`, and `3072x768`, exercising production
linearization, cotangent cropping, and raw-adjoint accumulation. The first
shape added 86 executable mappings; every later shape added **zero**.
Cumulative total mapping growth remained between 280 and 285. The largest
synthetic case peaked at 6,123 MiB process RSS. This is an endpoint test, not
a complete 3,072-AO molecule or a full-job mapping bound.

Production source hashes were recorded before the molecular/mapping probes
and checked unchanged afterward. Raw results, source hashes, reproduction
drivers, and original comparison data are retained under the project root:

`TEST_general/dlno_stc_test/runtime_tests/LLVM_scaling_20261116/size_standardization_local/`

The integrated run is in `integrated/`; prior original molecular results are
in `molecules/summary.json`. See `findings.txt` for the earlier comparative
runtime study. Small complete calculations establish numerical preservation;
they do not establish a large-system whole-job speedup.

## Repository-wide test limitation

`python -m pytest -q --maxfail=1` stopped after 11 passing tests, 2 skips,
and 18 deselections at `pyscfad/cc/test/test_dcsd.py::test_df_nuc_grad`:
`NotImplementedError: Saved update intermediates require ordinary DF-CCSD`
from `pyscfad/cc/ccsd.py:237`. The identical test fails on the untouched target
commit `11c00a34`; neither file is changed by this branch. This is an existing
full-suite failure, not a passing full-suite result.

The broader primary/domain checks reached 101 passes before three failures.
The remaining checks added 64 passes, one more baseline failure, and a
process-memory-related failure. All four domain-selection failures reproduce
on unchanged `11c00a34` with the same errors:

- `test_boys_construction.py::test_public_selection_preserves_permuted_labels_and_concrete_anchors`: virtual anchor columns are `None`.
- `test_boys_construction.py::test_selection_rejects_invalid_target_partition_and_rank_loss`: the expected exception is absent.
- `test_domain_parity.py::test_boys_localization_and_common_arrays_match_native`: the PAO arrays have 12 versus 14 columns.
- `test_domain_parity.py::test_fixture_supports_weak_spaces_and_target_mapping_match_native`: PAO parent indices have 12 versus 14 entries.

`test_correlation_df_selection.py::test_exact_pair_screening_matches_independent_pyscf_ri_oracle`
hit its configured memory allowance late in the combined process (3,411 MB
needed). It passed on both the unchanged target and the candidate in fresh
processes. No source change or relaxed numerical tolerance was needed.

The kernel gate ran `test_weak_moment_kernels.py`, `lib/test/test_multipole.py`,
and `lib/test/test_weak_endpoint.py`. The driver gate ran
`test_weak_moment_kernels.py`, `test_weak_moment_native.py`,
`test_weak_endpoints.py`, and `test_weak_local_products.py`. The broader gates
covered primary response/execution/selection reuse/buckets, selection response,
Boys pairs/preparation/construction/alignment, domain parity/grouping, shared
frames, correlation DF selection, MPI weak response, MP2 normalization,
multipole replay, and threading. All used `python -m pytest ... -q` under
the environment above. Exact outputs are retained with the local artifacts.

The endpoint optimization removes size-dependent compilation from its covered
CPU moment contractions. Pair-energy, primary, and later strong-domain work
still need complete large-system validation on the HPC environment.
