# D/W single-fragment runtime comparison

The restored D implementation and the default W implementation were compared
on water tetramer, n-icosane (C20H42), and three NWH geometries. Each timing is
the median of three interleaved calls per method after one warm-up per method.
Different molecule jobs ran sequentially on an AMD Ryzen 7 7800X3D.

All calculations use cc-pVDZ / cc-pVDZ-JKFIT, frozen first-row cores,
atom-grouped Boys fragments, deterministic density and CCSD(T), and LIS
auxiliary compression tolerance 1e-5. Occupied/virtual cutoffs are
**1e-4 / 1e-5 for D** and **3e-4 / 3e-5 for W**. No automatic cutoff conversion
is implemented. **W remains the API and command-line default.**

The four-thread setting is the density/PySCF CPU budget, pinned to CPUs 0–3.
BLAS defaults are one thread, and this native build's MRCC triples kernel
reports one worker. W forward retains its dynamic OpenMP team policy.

Paired results are D / W; LIS dimensions are occupied × virtual.

| Molecule | LIS D / W | Density + pullback (s) | Fragment total (s) |
| --- | --- | ---: | ---: |
| Water4 | 10 × 36 / 10 × 36 | 0.122 / 0.084 | 2.913 / 2.913 |
| C20H42, central carbon fragment | 14 × 64 / 14 × 62 | 1.783 / 1.569 | 39.140 / 32.596 |
| NWH 215, C3H6N2 | 13 × 54 / 13 × 56 | 0.102 / 0.090 | 17.248 / 19.163 |
| NWH 932, C5H8O2 | 15 × 61 / 16 × 62 | 0.288 / 0.267 | 38.090 / 42.702 |
| NWH 912, C7H8 | 18 × 72 / 18 × 71 | 0.239 / 0.223 | 69.425 / 72.081 |

Density timing includes the selected MP2 scalar and its response: an explicit
contraction for D and a trace-derived value for W. Fragment totals include
domain preparation, density, smooth LIS selection, CCSD(T), and the local
reverse pass. SCF, initial domain construction, common initialization, other
fragments, weak-pair work, and final common-data/SCF response are excluded.
These are not full molecular gradient or energy-only timings.

The 3× heuristic gives approximately comparable sizes, but only water4 has
identical occupied and virtual ranks. Equal ranks would still select different
orbitals. Total times therefore include differences in the CCSD(T) workload
and should not be interpreted as an equal-accuracy comparison. W's density
stage is faster in these medians, while the total-time ordering varies.

Native profiles identify an implementation difference beyond the explicit D
energy contraction: W rotates factors into the virtual Fock eigenbasis once
and scales columns at each quadrature point; D currently repeats dense virtual
propagation and its reverse. These measurements describe the current code,
not an intrinsic cost ratio between the density definitions.

NWH 215 has a slower first measured pair. Its density ranges are
0.097–0.306 s for D and 0.090–0.167 s for W; all three measurements are retained
in the medians. Its total ranges are 17.031–17.943 s and 19.050–22.185 s.
NWH 932 and 912 have substantially tighter total-time ranges. NWH 217 was
stopped during its first warm-up after selecting a 27 × 97 D LIS; it contributes
no measurements to this table.

The same physical fragment and domain are reused for D and W within each run.
The NWH targets are atoms 3 (215), 5 (932), and 0 (912), using zero-based indices.
Target selection chooses the group with maximal population on that atom; the
912 group has its largest total population on atom 5. Local target slots can
differ when domains are independently rebuilt at another thread count.

Every completed run passed checks for stable same-method energies and ranks,
finite local responses, exact target-slot reuse, zero sampling error, matching
D/W selected MP2 baselines, and
`E_fragment = E_ED_MP2 + E_CCSD + E_triples - E_LIS_MP2`.
Independent water4 matrix and gradient correctness tests are documented in
[the D validation report](2026-11-24-d-density-validation.md).

The workspace retains the benchmark harness, full configurations, geometries,
source/native hashes, raw JSON/log receipts, timing ranges, and validation
scripts under `runtime_validation/20261124_d_vs_w` and
`runtime_validation/20261124_d_vs_w_nwh`. Those local benchmark artifacts are
outside this repository. The compact water4 runtime geometry differs from the
geometry embedded in the correctness tests.
