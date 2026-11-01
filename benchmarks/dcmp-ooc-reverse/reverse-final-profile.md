# Final paired dCMP OOC profile

Source `/home/bkhou/DLNO_CCSDT/test_pyscfad_stc_mp2/runtime_test/dCMP-ooc/test_reverse/final/slurm-5822463.out`, job 5822463. One paired observation for each case. All times are seconds.

| Component | Baseline | Optimized OOC |
| --- | ---: | ---: |
| Total energy and gradient | 641.375542 | 488.451036 |
| SCF pullback excluding DF attachment | 393.093131 | 254.104394 |
| Two top-level factorized J/K calls | 121.62 | 99.78 |
| Generic J/K reverse / optimized helper | 64.61 | 62.61 |
| Eleven density-only implicit matvecs | 73.16 | 67.18 |
| Preconditioner setup | not instrumented | 5.58 |
| Preconditioner applications | not instrumented | 0.38 (12 calls) |
| Unassigned remainder | 133.703 | 18.574 |

The optimized helper's 62.61-second total excludes validation; that validation is included in the unassigned remainder. Its geometry stage (51.24) includes the third factorized call (50.83), which is not added again to the top-level sum. The helper's density stage (11.37) includes its DF loop (11.05). Optimized matvec DF loops total 62.40, nested in their parent matvec totals. Baseline preconditioner work is uninstrumented and included in baseline unassigned; neither remainder is a direct preconditioner measurement. Parent and child timings are rounded in the log.

The optimized three factorized calls each contain density packing of 17.757, 17.665, and 17.547, and native contractions of 9.205, 9.159, and 8.339. These are nested in the calls. Remaining dominant measured work is the two top-level factorized calls, the helper including its third factorized geometry call, and the 11 matvecs; further speedup is not implied by timing alone.

Stored energies differ by 0 Eh and the maximum absolute difference over all 102 gradient components is 4.29818e-12 Eh/bohr. These comparisons use the two saved full-precision JSON result files.

The completed optimized in-core reference took 169.776913 seconds total: DF forward 7.922794, SCF forward excluding DF 51.573514, MP2 forward 18.025877, MP2 pullback 3.958949, SCF pullback excluding DF 63.050631, and DF pullback 24.814902. Its energy difference versus baseline was zero; the maximum gradient difference was 1.55445e-11 Eh/bohr (coordinator report).

For a like-for-like reverse comparison, MP2 pullback + SCF pullback excluding DF + DF pullback is 282.063326 seconds OOC versus 91.824482 seconds in-core, a 3.07177 ratio. SCF forward excluding DF is 167.573184 versus 51.573514 seconds, a 3.249 ratio. The release was installed after source hash checks and 204 passing regressions, as reported by the coordinator.
