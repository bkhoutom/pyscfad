# Adopted OOC reverse adjoint optimization

The validated changes are installed in `/resnick/groups/sharmagroup/bkhou/src/pyscfad`. New jobs in the existing editable environment use them automatically. **No recompilation, dependency installation, or submission-script change is required.**

## What changed

- `pyscfad/df/_cderi_vjp.py`: optional packed-adjoint callback and a checked wrapper around the existing native `NPdunpack_tril_2d_vjp` kernel. It sums the two off-diagonal contributions while keeping each diagonal once. This removes full-block symmetrization, repeated symmetry scans, a temporary packing buffer and a diagonal-scaling pass. Dense, complex and unsupported-native fallbacks remain available. Overlapping input/output buffers are rejected before native writes.
- `pyscfad/df/_df_jk_factorized.py`: construct the adjoint directly into its final packed buffer; prepack the fixed Coulomb matrices once per pullback and perform their rank-two update in packed space. **Only this small Coulomb update uses scoped BLAS1.** Large exchange GEMMs retain the configured dense BLAS threads; native integral kernels retain OpenMP threads. All previous thread limits are restored afterward.
- `pyscfad/df/test/test_reverse_packed_density.py`: 20 new cases for triangle/diagonal normalization, noncontiguous inputs, unsafe output buffers, bounded auxiliary subranges, dense fallbacks, and independent molecular/auxiliary/orbital gradients for J-only, K-only and combined paths, including disk panels.

The initial factor-contraction code was retained. Two alternatives—flattened GEMMs and native packed half-transforms—passed numerical checks but did not consistently outperform it. No cross-pullback cache was introduced: orbital factors at the three reverse graph sites cannot generally be treated as identical.

## Controlled performance evidence

Job **5828088**, `hpc-19-28`, compared five alternating samples on three representative auxiliary-shell blocks of the actual dCMP geometry (800 AOs, 2042 auxiliary functions, rank 80). Each sample included both assembly and the following native derivative-integral contraction. All packed entries and all molecular/auxiliary coordinates were checked against an independent explicit derivative contraction.

| Three-block median, seconds | Original | Packed J with BLAS16 | Adopted packed J with BLAS1 |
|---|---:|---:|---:|
| Assembly + native contraction | 0.71489 | 0.65950 | **0.48299** |

The adopted variant won all five samples and reduced combined time by **32.4%**. This is a kernel-pipeline improvement, not a claim that the entire energy/gradient calculation is 32.4% faster. The first packed version made assembly faster but the following native contraction slower; measuring the complete pipeline exposed this tradeoff. The final local thread setting improved the combined result. The benchmark does not uniquely identify the underlying scheduler/cache/spin-wait mechanism.

## Final full dCMP run

Job **5828194** completed with exit code **0:0** on `hpc-19-28`, using 16 physical Ice Lake cores and 128 GB. Peak process RSS was **5.53 GiB**, comparable to the previous 5.61 GiB baseline. Input, basis, frozen orbitals and convergence tolerances match the previous runs.

| Phase | Wall seconds |
|---|---:|
| DF build | 14.84 |
| SCF forward without DF build | **142.35** |
| MP2 forward | 20.03 |
| MP2 reverse | 34.90 |
| SCF reverse including embedded DF pullbacks | **266.81** |
| Final attachment | 0.05 |
| Total energy + gradient | **479.14** |

The conservative SCF/DF split retains adjoint assembly/packing in SCF:

| SCF reverse work | Seconds |
|---|---:|
| Initial factor-response and compressed-adjoint passes | 70.44 |
| Eleven response-solver DF J/K operations | 62.99 |
| Coulomb preconditioner setup/applications | 6.25 |
| Other SCF work | 24.71 |
| Adjoint setup/assembly/packing | **23.52** |
| SCF reverse excluding DF construction reverse | **187.91** |
| DF construction reverse, excluding that assembly | 78.90 |

Thus reverse without DF is **1.32x forward** in this run. Subtracting the entire fused DF region, including its assembly, instead gives **164.39 s**, or **1.15x forward**. These are two accounting conventions, not statistical bounds. They match the definitions in the preceding `scf_df_split` report.

For context, the earlier profiled OOC run 5824643 on the same named node recorded 144.93 s forward, 268.55 s conservative reverse, 64.97 s assembly and 562.90 s total. Those were different allocations/times. The earlier best recorded OOC total was 488.45 s on another node. Neither comparison is a controlled estimate of overall speedup.

The new paired full run **5826995** also passed: original/first-candidate totals were 710.01/523.58 s, and reverse totals 332.68/305.38 s. However, unchanged forward SCF changed from 290.80 to 145.81 s within the same allocation. **Most of that total-runtime difference cannot be attributed to the optimization.** The controlled pipeline experiment and final full-gradient validation are the adoption evidence.

## Validation and provenance

- Final targeted regression job **5828195**: **224 passed**, no failures/errors/skips, 110.67 s. This includes all 20 new cases and the existing DF/SCF/MP2 OOC checks.
- Final dCMP energy differs from reference job 5822463 by **4.55e-13 Eh**. All **102 gradient components** agree within **3.93e-12 Eh/bohr**. Timing coverage and accounting checks passed; all 393 SCF coordinate blocks used the native path.
- Independent review found no remaining blockers. Final source hashes match the regression-tested snapshot. Benchmark timing hooks are confined to frozen job copies; production contains only the optimization.
- The pre-existing dirty source and all native artifacts were preserved. Only the two production modules and the new test file above changed. `installation.json` records the installed hashes; `candidate.patch` records this task's exact delta; `rollback/` preserves the prior two modules.

All numerical work ran on compute nodes. The self-contained conditions are under `/home/bkhou/DLNO_CCSDT/test_pyscfad_stc_mp2/runtime_test/dCMP-ooc/test_reverse/contractions`: `kernel`, `kernel_packed`, `assembly_native`, `packed_red`, `packed_green`, `regressions`, `paired`, `regressions_final`, and `final`. Each contains its driver, local source/input where required, static `submit.sh` and outputs. To rerun the final full calculation, enter `final` and run `sbatch submit.sh`.

Full precision data: `results-summary.json` and `final/results.5828194.packed_ooc_final.json` in the external testing tree; the workspace mirrors job outputs under `jobs/`. Scratch files remain under the existing `/resnick/scratch/$USER` convention. Production `dCMP-ooc/submit_pyscfad.sh` already has the tested global thread/panel settings.
