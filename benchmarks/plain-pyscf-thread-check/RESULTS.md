# Plain PySCF dCMP runtime audit — 2026-10-31

The slow plain-PySCF results were numerically correct. The plain driver had not
received the scoped SciPy DF thread control already used by the AD driver and
`dCMP-main`. It kept SciPy at one thread throughout. Its submission also lacked
an Ice Lake constraint and retained an unnecessary MPI-module/loader setup.

| Original output/job | CPU generation | SCF incl. DF | MP2 energy | Total |
|---|---|---:|---:|---:|
| 5777502 | Skylake | 92.482 s | 6.270 s | 98.766 s |
| 5786298 | Ice Lake | 82.789 s | 5.387 s | 88.193 s |
| AD 5777503 | Ice Lake | 53.653 s | includes AD preparation | 145.287 s, energy + gradient |

All three converged in 22 SCF cycles to HF -1377.51494675713 Eh and ordinary
MP2 -1381.31940656203 Eh. The ten printed AD gradient rows match the validated
full-gradient reference within 4.98e-9 Eh/bohr, consistent with eight-decimal
printing. The other 24 rows are not available in this output for comparison.

Job **5786642** ran the original and corrected plain drivers sequentially on
`hpc-19-30` with identical 16-core affinity, installed MKL libraries, Python,
scientific input and clean loader settings:

| Phase | Original | Corrected |
|---|---:|---:|
| DF construction | 35.789 s | 5.491 s |
| SCF including DF | 84.125 s | 57.363 s |
| MP2 energy | 5.891 s | 5.900 s |
| Full energy | 90.029 s | 63.300 s |

The measured energy difference is 2.27e-13 Eh. SciPy's actual pool was verified
at one thread during the original DF build and 16 during the corrected build;
NumPy/native BLAS remained at one, and SciPy returned to one afterward. The
MPI-module change was common to both diagnostic cases; its independent timing
effect was not measured. This is one matched pair, not a confidence interval or
a universal 60-second performance guarantee.

The approximately 60 s SCF column in the previous AD comparison includes DF. The plain file's final total also includes roughly six seconds of
MP2 energy. CPU generation, placement, and run variation explain why exact
absolute times should not be compared across independent allocations, but the
large avoidable gap here was the missing DF thread control.

The production `pyscf_scf.py`, `submit_pyscf.sh`, and README in `dCMP` are updated.
The driver prebuilds DF inside the SciPy-only thread context and prints DF and
SCF timing separately. The script selects the same Ice Lake generation and
core-binding policy as the AD script, using the existing MKL environment without
MPI modules. No packages, native libraries, or AD algorithms changed.

The AD output's SCS and same/opposite-spin entries are pre-existing placeholders:
`pyscfad/mp/dfmp2.py` sets both spin components to zero. The standard MP2 total
and its gradient are valid; those SCS diagnostic lines are not computed SCS
results. The plain PySCF output does compute the spin decomposition.

Evidence: [observed logs audit](observed-output-audit.json),
[matched comparison](comparison.json), [original runtime](original.json),
[corrected runtime](optimized.json), and [applied files](activation.json).
Original production files are preserved in `original/`.

The corrected production submission was also run from `dCMP` as job
**5786841**, completing with exit code 0:0 on `hpc-35-15`. It took **5.426 s DF**,
**49.985 s SCF including DF**, **4.773 s MP2**, and **54.788 s total**. The energy
remained -1381.31940656203 Eh and stderr was empty. This separate allocation
is faster than the corrected 63.300 s matched test; it demonstrates residual
allocation-dependent timing variation, not an additional code change.
[Production verification](production-verification.json) records the result.
