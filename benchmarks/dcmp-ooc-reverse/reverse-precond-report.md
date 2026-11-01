# Reverse preconditioner cache

Candidate-only tests: `benchmarks/dcmp-ooc-reverse/candidate/pyscfad/df/test/test_reverse_precond_cache.py`. Real disk-backed HDF5 packed DF factors are used with instrumented `DF.loop` reads. Results are compared against an independently assembled orbital Hessian, including every occupied orbital. Tests require repeat applications to avoid DF reads when a 1 MiB budget fits; zero and insufficient budgets must stream and remain correct. Empty, malformed, negative, NaN, and infinite budgets use the safe default budget.

Production edits: pending evidence from compute-node red tests. Validation so far: test file passes `python -m py_compile`; numerical validation is pending SLURM as required by the plan.

Red evidence: compute job 5821814 showed six cache-read failures as intended; 24 other and fallback checks passed, including existing `test_scf_precond` (reported by coordinator).

Implementation: closure-local scaled occupied-virtual cache with 1024 MiB default finite nonnegative budget; invalid budgets use default; zero or insufficient budgets retain streamed J-build fallback. Setup and application emit timing and application-count profiling when requested; dense operations use scoped BLAS threading. Candidate implementation and test file pass `py_compile`. Compute-node green tests and dCMP validation pending.

Headroom review: added test requiring streaming with `dfobj.max_memory = 0` despite a positive cache budget; compute job 5821866 failed that assertion as intended and passed 30 other checks (coordinator report). Cache now additionally respects 30% of available DF memory; profiling reports the effective limit. Updated files pass `py_compile`; green compute validation pending.

Independent review identified temporary panel retention. Candidate now releases `bov`, `mo`, and the last `eri1` immediately after each setup Gram update, preventing overlap with later allocations and closure lifetime; syntax check passed. Green job 5821893 used an earlier snapshot and does not validate this final adjustment.

Regression job 5821893 had 48 failures from a `NameError` in `get_jk_bwd` and 141 passes. An unrestricted edit accidentally removed its `max_memory` assignment; the exact baseline line is restored. Byte comparison confirms all source before `gen_coulomb_precond` matches baseline; syntax check passes. A fresh combined compute run is required.
