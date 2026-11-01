# dCMP: separate optimized OOC SCF and DF reverse times

Job **5824643** completed with exit code **0:0** on `hpc-19-28`, using 16 physical Ice Lake cores and 128 GB. All calculations ran on the compute node. Total energy/gradient wall time was **562.895950 s**; peak process RSS was **5.565 GiB**.

The self-contained calculation is in `/home/bkhou/DLNO_CCSDT/test_pyscfad_stc_mp2/runtime_test/dCMP-ooc/test_reverse/scf_df_split`. Its `submit.sh`, `driver.py`, `timing_ledger.py`, `input.xyz`, `code/pyscfad`, reference, manifests, original logs and full result JSON preserve what was run. To rerun, enter that directory and use `sbatch submit.sh`.

## Answer and comparison

SCF reverse remains slower than SCF forward after excluding DF construction reverse. A conservative assignment, retaining AO-adjoint assembly/packing in SCF, gives **268.55 s reverse versus 144.93 s forward (1.853x)**. Subtracting the entire fused DF region instead gives **203.58 s (1.405x)**. The difference between those two boundaries is measured assembly/packing work, not unexplained time.

| Wall time, seconds | Original dCMP in-core | Updated in-core reference | New optimized OOC profiling run |
|---|---:|---:|---:|
| DF build | 7.83 | 7.92 | 14.89 |
| SCF forward without DF build | 45.82 | 51.57 | 144.93 |
| SCF reverse without DF construction reverse, retaining adjoint assembly | 52.10 | 63.05 | 268.55 |
| DF construction reverse plus final attachment | 18.50 | 24.81 | 77.45 |
| SCF reverse including DF and attachment | 70.60 | 87.87 | 346.00 |
| Total energy + gradient | 145.29 | 169.78 | 562.90 |

Original in-core: `dCMP/reproduce_dcmp_ad.out.16.3.5777503`. Updated in-core: `dCMP-ooc/test_reverse/final/results.5822463.optimized_incore.json`. These are historical runs, on different nodes; the updated reference also uses the later optimized source. Only the new OOC forward/reverse ratio is a comparison within this new run and allocation. The original in-core column is preserved as requested.

The previous optimized OOC run (5822463, `hpc-34-23`) was faster overall: 488.45 s total and 254.10 s SCF reverse including embedded DF. It did not record the present split, so these new substage times must not be assigned to that older run. This measurement run is not a replacement claim for the previous best runtime. It uses the same recorded scientific source before instrumentation, but a different node and CPU affinity; this experiment does not isolate the reason for timing variability.

## Exact accounting

The main driver SCF reverse timer is **345.960712 s**. Three factorized geometry calls contain **142.381997 s** of fused DF construction reverse. Subtraction leaves **203.578715 s** of SCF work.

The fused region includes **64.968186 s** of AO-adjoint setup/assembly/packing. Retaining that work in SCF gives **268.546901 s** SCF and **77.413811 s** DF reverse. Final attachment costs **0.041020 s**, producing **77.454831 s** for the DF-plus-attachment table row.

Thus `268.546901 + 77.413811 = 345.960712`, and adding attachment gives **346.001732 s**. MP2 pullback events are excluded from every SCF/DF figure above.

The representation boundary differs from in-core: OOC transforms thin adjoint factors before expanding AO-pair blocks, whereas in-core constructs a dense fitted-integral cotangent before the DF pullback. The conservative assignment intentionally retains all measured mixed assembly/packing in SCF; it is not a claim of identical intermediate representations.

## Remaining SCF costs

These are disjoint and sum to the conservative SCF reverse time:

| Work | Seconds |
|---|---:|
| Three initial factor-response/compressed-adjoint passes | 93.61 |
| AO-adjoint setup, expansion and packing | 64.97 |
| 11 implicit-response DF J/K transpose operations | 66.72 |
| Coulomb-preconditioner setup plus 12 cached applications | 14.78 |
| Other SCF work, including independent density transpose and orbital/JAX work | 28.47 |
| Total | 268.55 |

The 93.61 s comprises streamed fitted-integral reads/unpacking and contractions such as `B @ S`, `H @ t`, plus factor-response work where needed. It is not DF construction reverse and cannot be subtracted as such. The 66.72 s timer covers the DF J/K part of response matrix-vector applications; surrounding orbital transforms remain in the last row. The preconditioner timer similarly covers its Coulomb core.

The three geometry paths are final HF energy, final-output Fock/eigendecomposition, and the external-geometry pullback of the implicit SCF fixed point. Their factor-response flags are `True, False, False`; all have rank 80. Each traverses the fitted integrals and constructs its own geometry adjoint. The forward SCF took 22 iterations plus its extra cycle; reverse uses a response solve and multiple output adjoints, so the workloads are not symmetric.

The highest measured SCF targets are the repeated factor contractions and AO-adjoint assembly. Reusing valid shared contractions, improving batching/packing, and measuring thread placement are plausible next experiments. These timings combine data access and arithmetic; they do not establish that physical disk I/O alone dominates. Both dense BLAS and OpenMP were configured for 16 threads within their existing scopes.

The remaining DF region consists mainly of **40.91 s metric/triangular transforms**, **34.13 s native derivative-integral contractions**, **2.14 s metric reverse**, and small overhead. All 393 SCF coordinate blocks used the native path; no explicit fallback was taken. Reusing metric factorizations or accumulating geometry cotangents before one integral contraction may help, but requires preserving the separate orbital/density responses and bounded memory. Simply adding the three J/K seeds is not valid because their density factors can differ.

## Validation and changes

- All **102 gradient components** compared against optimized OOC reference job 5822463; maximum difference **4.7766e-12 Eh/bohr**. Energy difference **9.0949e-13 Eh**.
- All timing coverage, nesting, positivity and substage-sum checks passed. Three SCF factorized/coordinate events were distinguished from the MP2 event.
- Only the isolated snapshot's `df/_df_jk_factorized.py`, `df/_cderi_vjp.py` and `df/_df_jk_opt.py` were instrumented. `instrumentation.patch` records all source changes. AST comparison after stripping instrumentation preserves scientific expressions.
- `driver.py` adds phase tagging, source hashes and reference checks; `timing_ledger.py` records exact nested timers. DF coordinate outputs are synchronized before stopping the inclusive DF timer.
- All **418 live package Python files** still match the captured baseline hashes. Submitted source, driver and ledger hashes match the result record. No production source was changed by this measurement.

Full precision data: `job/results.5824643.optimized_ooc_split.json`; original output: `job/slurm-5824643.out` and `.err`. The two stderr warnings concern pre-existing non-JSON-serializable molecule attributes during checkpoint serialization; the job and gradient validation completed successfully.
