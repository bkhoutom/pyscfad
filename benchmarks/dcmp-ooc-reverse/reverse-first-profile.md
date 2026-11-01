# First paired dCMP SCF reverse profile

Source: `/home/bkhou/DLNO_CCSDT/test_pyscfad_stc_mp2/runtime_test/dCMP-ooc/test_reverse/paired/slurm-5821996.out`, job 5821996. Times in seconds. Single paired observation; first candidate predates later changes. In-core run was still in progress at extraction.

| SCF reverse top-level component | Baseline | First candidate |
| --- | ---: | ---: |
| SCF pullback, excluding DF attachment | 504.963260 | 345.889240 |
| Two factorized J/K reverses | 175.88 (92.84, 83.04) | 157.09 (78.04, 79.05) |
| Generic J/K reverse total | 94.87 | 78.37 |
| 11 density-only implicit matvecs | 81.45 | 80.93 |
| Preconditioner setup | not instrumented | 6.61 |
| Preconditioner applications | not instrumented | 0.89 (12 calls) |
| Unassigned remainder | 152.763 | 21.999 |

The unassigned remainder is the SCF total minus the listed measured top-level components, using rounded log values; baseline preconditioner setup and applications are included there. It is not a measurement of preconditioner cost alone.

Nested within generic J/K: DF block loop 24.40 vs 21.66, geometry 69.51 vs 55.64. The candidate geometry includes int3c pullback 17.48, low-bar GEMM 13.73, reads 9.41, and triangular solves 7.87; these are children of the geometry stage, not additional costs. Matvec DF block loops sum to 76.34 vs 75.40, included in the matvec totals.

Within the two candidate factorized calls, density packing took 35.769 and 35.150; native contraction took 11.318 and 12.128. Baseline did not emit these substage timings. Packing is a measured target for further investigation, not evidence of an achievable speedup. Other large measured costs are the two factorized reverses, the generic geometry pass, and the 11 streamed matvecs. The candidate preconditioner cache used 897.36 MiB under a 1024 MiB effective limit.
