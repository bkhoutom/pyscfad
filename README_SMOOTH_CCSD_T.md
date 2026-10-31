# Smooth DLNO-CCSD(T): modes, defaults, and forces

This branch supports multiple physical occupied fragments **S**, each with its
own possibly truncated local interacting space (**LIS**). The energy combines
an independent MP2 baseline with fragment `CCSD + (T) - MP2-LIS` corrections.
The gradient includes the retained D/W subspace, AO overlap, integrals, CCSD
amplitudes, fragment weights, and full occupied/virtual Fock response.

See the [complete API and command-line reference](doc/source/user_guide/smooth_lis_triples.md)
for every smooth CCSD(T) driver option, units, allocation formulas, and diagnostics.
The [frame theory note](NOTE_SMOOTH_LIS_FRAME.md) explains the orbital-gap problem.
The [STC guide](pyscfad/dlno_stc/README.md) covers MP2, D/W matrices, and native builds.

## Supported modes

| Selection / triples setting | Behavior |
| --- | --- |
| `smooth=True` (default), `ccsd_t=False` (default) | Anchored full-LIS CCSD, without triples. |
| `smooth=True`, `ccsd_t=True`, `triples_controls=None` or `{"mode": "deterministic"}` | Deterministic enumeration of native asymmetric Laplace triples on the fitted grid. Useful as a small-system reference; enumeration is expensive. |
| `smooth=True`, `ccsd_t=True`, `{"mode": "stochastic", ...}` | Native asymmetric Laplace triples with selected-space sampling; energy and gradient use the same production draws. |
| `smooth=False`, deterministic triples | Conventional semicanonical reference with direct denominators. Stochastic triples are rejected. |

The smooth path always uses one complete occupied frame and one complete
virtual frame **within the retained LIS**. This does not disable LIS truncation.
Native dressing and asymmetric sampling are fixed choices in the current code.
Canonical stochastic triples, equal dressing, JAX production dressing,
clustered Fock frames, and explicit-S frames are no longer supported modes.

## Main Python defaults

These are library defaults, not the overrides in the run scripts below.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `boys_grouping` | `"singleton"` | One S per active Boys orbital; `"atom"` groups by heavy atom with associated H atoms, and `"molecule"` makes one S. |
| `force_full_domains` | `False` | Full extended domains when enabled; `"molecule"` enables this automatically. This is separate from LIS eigenvalue cutoffs. |
| `matrix_kind`, `thresh_occ`, `thresh_vir` | Required | Choose D or W and its occupied/virtual absolute-eigenvalue cutoffs; zero keeps the full corresponding space. |
| `internal_rank_threshold` | `1e-6` | Numerical rank threshold when completing the LIS to the full MO layout; all original S orbitals remain retained. |
| `smooth` / `smooth_laplace_points` | `True` / `10` | Anchored frames and a positive fitted triples grid per LIS. |
| `ccsd_t` / `triples_controls` | `False` / `None` | Enable triples explicitly; omitted controls then mean deterministic. |
| `triples_controls.mode` | `"deterministic"` | `"stochastic"` enables sampling. |
| `nsamples` | Required for stochastic mode, integer ≥ 2 | Nominal production budget **per S** in fixed-count mode; still required but does not cap tolerance-driven sampling. |
| `energy_tolerance` | Omitted (equivalent to `0`) | Positive value is the total requested `(T)` energy SE in **Eh**; `5e-5` Eh = **0.05 mEh**. |
| `pilot_samples` | `10000` | Independent discarded pilot draws **per Laplace point per S**. |
| `global_seed` | `71` | Separate fragment, point, pilot, and production streams are derived from this seed. |
| `trace_exponent` | `5.0` | Triples Laplace-point variance allocation exponent. The MP2 trace model uses `1.4`. |
| `fragment_energies` | Saved MP2 fragment proxies | Optional full vector of reference energies, e.g. CCSD fragment energies; variance shares use absolute values. |
| `include_hf` / `include_baseline` | `False` / `True` | Return correlation energy by default; enable HF for total molecular energy and forces. |
| `merge_identical` | `True` | Share preparation for identical domains; preserve physical S corrections. |
| `return_components` | `False` | Include per-fragment energies, sample counts, and SE diagnostics when enabled. |

Sampling-only controls are invalid with `mode="deterministic"`. The triples
dictionary has no thread, batch, dressing, or separate backward-sampling option.
MP2 baseline and D/W matrix controls are independent dictionaries; changing their
mode does not change the triples mode.

## Are these only forward options?

**They are supplied before evaluation, but affect both energy and gradient.**
LIS ranks, reference anchors, quadrature nodes/weights, proposal probabilities,
and sample-allocation decisions are held fixed during differentiation. Production
draws contribute both the energy and its tensor derivatives; reverse mode does
not draw a new sample set. Continuous physical responses remain differentiated.

CCSD convergence controls also matter in reverse. In the default DF-CCSD custom
response path, `conv_tol_normt` is reused as the response-lambda convergence
tolerance, and `max_cycle` limits both solves. Requesting forces does not
automatically tighten the forward CCSD criteria.

| CCSD control | Library default | `smooth_force_compare.py` | `benchmark_lno_w_triples.py` |
| --- | --- | --- | --- |
| `conv_tol` / `--cc-conv-tol` (energy change, Eh) | `1e-7` | `1e-10` | `1e-7` |
| `conv_tol_normt` / `--cc-conv-tol-normt` (amplitude-update norm) | `1e-5` | `1e-8` | `1e-5` |
| `max_cycle` | `50` | `200` (fixed) | `200` (fixed) |

Both forward CCSD convergence conditions must be satisfied. The update uses
diagonal-Fock occupied–virtual denominators as its preconditioner, while retaining
off-diagonal Fock terms in the equations. At the reference geometry, anchors are
semicanonical; the force differentiates all Fock entries in their transported frame.

## Total triples SE and point allocation

Positive `energy_tolerance` allocates total **variance** among S in proportion to
absolute fragment energy proxies, then among its Laplace points using the trace
model. Every point has an independent pilot, at least 10000 production draws,
and adaptive extension until its reported SE meets its allocation. Prepared
tensors and probability tables are reused for that point's pilot and production.
The code checks the measured SE target; the target is not a guarantee on the
unknown exact energy error or the force RMS error.

With omitted/zero tolerance, pilots allocate `nsamples` across points using their
measured standard deviations. Rounding and a two-draw minimum per point can make
the actual production count larger than `nsamples`. Pilots are additional draws.
Quadrature error is reported separately and is not included in sampling SE.

## Running with 32 threads

Install this checkout and build the native extension using the
[build instructions](pyscfad/dlno_stc/README.md#build-and-checks). The runners set
CPU affinity and numerical thread limits before importing JAX. Always pass
`--threads 32` for our runs. Native sample blocks are processed in groups of up to
the actual OpenMP thread count; there is no separate user batch-count setting.

From the repository root, compare stochastic forces with conventional
deterministic forces for C5H6/cc-pVDZ, one untruncated LIS and five frozen cores:

```bash
python examples/dlno_stc/smooth_force_compare.py \
  --xyz-file examples/dlno_stc/c5h6_force_compare.xyz \
  --basis cc-pvdz --frozen 5 --boys-grouping molecule \
  --lno-thresh-occ 0 --lno-thresh-vir 0 \
  --threads 32 --triples-laplace-points 10 \
  --triples-pilot-samples 10000 --triples-energy-tolerance-mha 0.05 \
  --triples-trace-exponent 5.0 --seeds 71 \
  --output-dir results/c5h6_smooth_current
```

For multiple S and truncated LISs on C6H14/def2-SV(P):

```bash
python examples/dlno_stc/smooth_force_compare.py \
  --xyz-file examples/dlno_stc/c6h14.xyz \
  --basis 'def2-SV(P)' --frozen 6 --boys-grouping atom \
  --lno-thresh-occ 1e-4 --lno-thresh-vir 1e-5 \
  --threads 32 --triples-laplace-points 10 \
  --triples-pilot-samples 10000 --triples-energy-tolerance-mha 0.05 \
  --triples-trace-exponent 5.0 --seeds 71 \
  --output-dir results/c6h14_smooth_current
```

These commands use full extended domains; add `--finite-domains` for finite
domains with atom grouping. `smooth_force_compare.py` sets memory to **800000 MB**
for the approximately 900 GB host. `benchmark_lno_w_triples.py` exposes
`--max-memory-mb` with the same default. STC planning budgets are advisory, not
reservations of host memory.

The API and both runners default to triples exponent **5.0**; MP2 uses **1.4**.
The comparison runner defaults to pilots **10000**, three seeds **71, 72, 73**,
and total SE **0.05 mEh**. The benchmark runner defaults to pilots **20000** and
requires targets/seeds explicitly. The examples above spell out exponent 5.0
and 10000 pilots.
The benchmark runner currently restricts molecule grouping to C5H6 and atom
grouping to C20H42; use the comparison runner for C6H14 or H4.

Results include `comparison.json`, with Cartesian-component RMS force error in
**mEh/bohr**, individual energy/force records, and `settings.json`.
The [full reference](doc/source/user_guide/smooth_lis_triples.md#command-line-runners)
lists optional reference reuse, fragment-only comparisons, finite differences,
and all other CLI defaults.
