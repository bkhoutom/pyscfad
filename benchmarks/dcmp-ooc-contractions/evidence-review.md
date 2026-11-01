# Final evidence review

**No remaining review blocker.** The final packed-assembly candidate has passing full-gradient and regression evidence, consistent timing accounting, and controlled evidence of a local speed improvement. This review independently read saved results and logs using standard-library parsing; no scientific imports, new tests, or jobs were run. Installation and preservation of prior workspace changes are handled separately by the coordinator.

## Correctness and provenance

- Final full result: `jobs/final/results.5828194.packed_ooc_final.json`, job **5828194**. Recomputed comparisons from all **102 finite gradient components**, rather than relying on saved pass flags. Relative to reference job 5822463, energy differs by **4.54747e-13 Eh** and the largest gradient difference is **3.93325e-12 Eh/bohr**. Relative to the preserved baseline in job 5826995, the differences are **2.27374e-13 Eh** and **5.51359e-12 Eh/bohr**. These satisfy the required `1e-9` and `2e-7` tolerances.
- The final XYZ file hash agrees with the reference, baseline, and final result. All recorded source, driver, and timing-ledger hashes match the frozen final-job files inspected.
- `jobs/regressions_final/results.5828195.xml` confirms **224 tests, zero failures, errors, or skips**. The job log reports **224 passed in 110.67 s**, with 30 coordinate-serialization warnings and empty stderr.
- Thread snapshots preserve OpenMP at 16; both BLAS pools are 1 at the end of the run, and the dense scopes report 16. The final packed-J scope and its exception-safe restoration were reviewed in `assembly-review.md`.

## Timing accounting

Every final split field was independently reproduced from raw SCF-phase events. There are three factorized geometry calls, three coordinate helpers, **393 native blocks**, no coordinate fallback, 11 implicit matvecs, one preconditioner setup, and 12 applications. MP2 events are excluded from the SCF split.

The final SCF pullback is **266.811310 s**, including embedded DF reverse. The separate DF attachment pullback is **0.045072 s**; it is not included in that number. Embedded factorized DF reverse is **102.424485 s**, leaving **164.386826 s** outside it. Assigning the **23.524854 s** assembly/setup work to SCF gives the conservative accounting split:

```text
SCF reverse excluding DF, including mixed assembly: 187.911680 s
DF reverse excluding mixed assembly:                78.899631 s
Sum:                                               266.811310 s
```

These are allocations of one observed execution, not timings from separately run SCF-only/DF-only calculations. The four factorized DF substage timings sum to their parent interval. Coordinate density plus native contraction leaves 0.109580 s of helper overhead inside coordinate time; it is not lost or double-counted. The total run is **479.137454 s**, while the six phase timers sum to **478.977572 s**; the remaining **0.159882 s** is surrounding driver overhead. Prefix read/unpack/dense diagnostics are subsets, not an exhaustive partition.

## Performance evidence and limits

| Observed seconds | Baseline, job 5826995 | Packed J16, job 5826995 | Final packed J1, job 5828194 |
| --- | ---: | ---: | ---: |
| Total energy + gradient | 710.010 | 523.584 | 479.137 |
| SCF forward, excluding DF build | 290.795 | 145.806 | 142.350 |
| SCF reverse, including embedded DF | 332.682 | 305.381 | 266.811 |
| Assembly including setup | 63.252 | 39.743 | 23.525 |
| Native coordinate contraction | 32.829 | 51.501 | 43.712 |
| Embedded factorized DF reverse | 131.708 | 126.906 | 102.424 |

The earlier pair ran on `hpc-20-35`; the final run used `hpc-19-28`. Unchanged forward work varies substantially. **Do not attribute the full 710→479 s reduction to this code change or claim a statistically established new best runtime.** The previous verified reference also took 488.451 s. The final run confirms correctness and includes the actual one-thread context overhead, but is one full execution on a different allocation.

The stronger local comparison is job **5828088**: five alternating-order measurements over three real dCMP shell blocks, with complete assembly followed by native coordinate contraction. Combined medians are **0.714894 s original**, **0.659504 s packed J16**, and **0.482990 s packed J1**: **32.4% lower combined time** for J1 versus original. J1 wins all five comparisons; independent packing and all molecular/auxiliary coordinate checks pass. This supports the adopted local optimization, not a 32.4% full-program speedup. The microbenchmark uses random rank-80 factors and omits one-time J setup and the final explicit J1 context-entry cost; the final full timing includes those costs.

Final peak process RSS is **5.533936 GiB**, versus **5.612610 GiB** baseline and **5.655636 GiB** v1. No process-RSS regression is observed; this measurement is not total node memory or a general scaling guarantee.

Final-result SHA-256: `ac8cc959a3d716096f9b6312269b88ef2a976c3b210f7a87331da9d6c10c7976`.
