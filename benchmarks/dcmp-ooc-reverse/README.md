# dCMP SCF/DF reverse optimization

Work requested on 2026-10-31. All molecular calculations and pytest runs use SLURM compute nodes. Login-node work is limited to file inspection, syntax checks, and analysis of saved results.

Installed and verified in the live package. All 204 focused regressions passed in job 5822443, and final same-node baseline/OOC/in-core comparison 5822463 completed successfully. Installed hashes match both tested snapshots. `installation.json` records the exact files and verification jobs.

## Source changes

Paths below are relative to `/resnick/groups/sharmagroup/bkhou/src/pyscfad`.

| File | Change |
| --- | --- |
| [pyscfad/df/_df_jk_opt.py](/resnick/groups/sharmagroup/bkhou/src/pyscfad/pyscfad/df/_df_jk_opt.py) | Retain bounded occupied–virtual preconditioner factors; use scoped threaded dense operations; preserve streaming fallback. Use validated thin factors for eligible generic geometry pullbacks while retaining the independent full AO-density cotangent. |
| [pyscfad/df/_int3c_cross_opt.py](/resnick/groups/sharmagroup/bkhou/src/pyscfad/pyscfad/df/_int3c_cross_opt.py) | Support bounded auxiliary-shell slices in the existing native packed-integral coordinate VJP, including partial atom ranges and AO-pair offsets. |
| [pyscfad/df/_cderi_vjp.py](/resnick/groups/sharmagroup/bkhou/src/pyscfad/pyscfad/df/_cderi_vjp.py) | Contract real coordinate derivatives directly with the native VJP instead of materializing derivative arrays; preserve unsupported-case fallback. Use flattened NumPy GEMMs for MO-density construction. |
| [pyscfad/df/_df_jk_factorized.py](/resnick/groups/sharmagroup/bkhou/src/pyscfad/pyscfad/df/_df_jk_factorized.py) | Combine density-construction operations, reuse small fixed coefficient arrays, and perform real symmetrization with the existing OpenMP routine outside the dense BLAS scope. |
| [pyscfad/mp/dfmp2.py](/resnick/groups/sharmagroup/bkhou/src/pyscfad/pyscfad/mp/dfmp2.py) | Retain the actual forward Lov result in the VJP residual, removing a repeated forward AO-to-MO transformation. |

There are no C/C++ changes or native-library rebuilds in this task. Five new test files cover cache reuse/memory fallback, native slices and packing weights, repeated MP2 pullbacks, independent generic density/geometry bars, and dense-contraction ordering/complex conventions. Two existing tests update obsolete call-count instrumentation while retaining their numerical assertions.

`candidate.patch` is the exact task diff against the captured user-modified baseline, not a clean Git checkout. `candidate-changes.json` records old and new hashes. Existing unrelated local changes are preserved. Independent code-review reports are included here.

## Runtime settings and limits

The AO CDERI remains disk-backed. Thin occupied panels and the bounded preconditioner cache can remain in RAM; this does not convert the AO DF calculation to in-core mode.

- 16 physical Ice Lake cores, 128 GB allocation, 120000 MB PySCF advisory memory.
- Global OpenBLAS/MKL/BLIS thread counts remain 1 inside native OpenMP work. Scoped dense BLAS and the eager SciPy DF build use 16 threads.
- `PYSCFAD_FACTOR_PANEL_MEMORY_MB=4096`, `PYSCFAD_SOLVE_MEMORY_MB=64` retain the previously validated production settings.
- New `PYSCFAD_PRECOND_CACHE_MEMORY_MB` defaults to 1024 MiB. The cache also must fit 30% of currently available advisory memory. Zero disables retention and preserves streaming applications. dCMP retains 897.36 MiB for this cache.
- Scratch uses `/resnick/scratch/$USER/dcmp-reverse/$SLURM_JOB_ID/$BENCH_CASE`.

Optimized geometry dispatch is conservative: concrete coordinate-only real double inputs, supported packed DF, exact validated low-rank density, and no custom/MPI coordinate callback. Unsupported cases keep the original route. Block-size limits have a minimum of one indivisible auxiliary shell; they are not a hard bound on whole-process memory.

## Reproduction

Every condition under `/home/bkhou/DLNO_CCSDT/test_pyscfad_stc_mp2/runtime_test/dCMP-ooc/test_reverse` has its own driver, static submit script, frozen code where relevant, and outputs. The final full benchmark also contains `input.xyz` and complete baseline/candidate packages.

```bash
cd /home/bkhou/DLNO_CCSDT/test_pyscfad_stc_mp2/runtime_test/dCMP-ooc/test_reverse/final
sbatch submit.sh
# After completion, use the job ID returned by sbatch:
python compare.py JOBID
```

`compare.py` reads saved JSON only. It checks matching node, CPU affinity, input, driver, memory and thread settings; energy agreement within 1e-9 Eh; and every gradient component within 2e-7 Eh/bohr. The driver records all 102 components, source hashes, effective thread pools and peak process RSS.

The final regression condition is `regressions_release`, job 5822443: 204 passed in 153.67 s. Warnings are the existing PySCF `mol.dumps` coordinate-serialization warnings. Earlier `red_*`, `generic_red`, `dense_red`, and `precond_headroom` conditions deliberately demonstrate the missing optimization before implementation; their expected failures are historical evidence, not failures of the final candidate. The first combined regression's missing-variable error was corrected before subsequent validation. `paired_final` corresponds to job 5822210, cancelled before execution to include further measured improvements.

## Timing interpretation

The same scientific input is used throughout: 34-atom dCMP, cc-pVTZ, cc-pVTZ-JKFIT, 24 frozen orbitals, converged RHF, and full DF-MP2 energy plus geometry gradient. All timed outputs are synchronized.

OOC integral-coordinate work occurs inside the MP2 and SCF pullbacks. Its final DF attachment pullback is almost empty. In-core integral-coordinate work occurs in the final DF pullback and serves both upstream stages. Compare the sum of all reverse phases when comparing storage modes; individual SCF reverse columns have different work boundaries.

The first candidate comparison (5821996, hpc-34-29) passed all energy/gradient checks: total 811.628 s baseline, 619.103 s first-candidate OOC, 199.973 s first-candidate in-core. Maximum first-candidate OOC gradient difference was 2.83e-12 Eh/bohr; in-core 1.47e-11. This candidate predates the final generic-geometry and dense-construction improvements. The final measurements below supersede it for performance claims.

## Final measured results

Job **5822463**, **hpc-34-23**, identical 16-core affinity and settings for all cases. Wall times in seconds. The in-core reference uses the optimized code.

| Phase | Original OOC | Optimized OOC | In-core |
| --- | ---: | ---: | ---: |
| DF forward | 15.88 | 14.57 | 7.92 |
| SCF forward, excluding DF build | 164.49 | 167.57 | 51.57 |
| MP2 forward | 22.61 | 24.07 | 18.03 |
| MP2 reverse | 45.02 | 27.91 | 3.96 |
| SCF reverse | 393.09 | 254.10 | 63.05 |
| Final DF / attachment reverse | 0.04 | 0.04 | 24.81 |
| All reverse phases | 438.16 | 282.06 | 91.82 |
| Total energy + gradient | 641.38 | 488.45 | 169.78 |
| Peak process RSS (GiB) | 6.00 | 5.70 | 36.89 |

Total OOC wall time fell **23.84%** (641.38 → 488.45 s). SCF reverse fell **35.36%** (393.09 → 254.10 s); MP2 reverse fell **38.00%** (45.02 → 27.91 s). Forward SCF times differ by only 1.9% in this pair. These are individual same-node observations, not a statistical estimate of cluster-wide runtime.

Combined reverse OOC/in-core ratio is **3.07×**, compared with **3.25×** for SCF forward. SCF reverse is still **1.52×** its own forward time; MP2 reverse is **1.16×** its forward time. The scaling is close to the proposed forward-ratio heuristic, but does not establish an I/O-only lower bound.

Saved energies are identical at the reported precision. Maximum absolute gradient differences against original OOC are **4.30e-12 Eh/bohr** for optimized OOC and **1.56e-11 Eh/bohr** for in-core, across all 102 components. Both jobs completed with scheduler exit code 0:0.

The optimized SCF reverse comprises about 99.78 s in two top-level factorized J/K reverses, 62.61 s in the generic geometry helper (including its third factorized reverse), 67.18 s in 11 response matvecs, 5.58 s in preconditioner setup, 0.38 s in 12 cached applications, and 18.57 s in remaining work. The three integral-density construction stages total about 52.97 s; native integral contractions total 26.70 s. Those are nested timings and must not be added again to the top-level totals.

The remaining large costs are the separate DF geometry contributions, their metric/transformation work, and streamed response matvecs. Further gains would require sharing/fusing these computations or retaining additional fixed transforms. No such unverified changes are installed.

## Running production jobs

Your existing `dCMP-ooc/submit_pyscfad.sh` already supplies the validated 16-core threading and panel settings. The environment is an editable installation of this repository. New jobs pick up these Python changes; **no recompilation or reinstall is needed**. The new preconditioner cache budget defaults to 1024 MiB, so the existing submission script needs no additional export.

```bash
cd /home/bkhou/DLNO_CCSDT/test_pyscfad_stc_mp2/runtime_test/dCMP-ooc
sbatch submit_pyscfad.sh
```
