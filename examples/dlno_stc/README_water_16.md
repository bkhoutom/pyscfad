# Water-cluster DLNO-CCSD(T) force example

This guide describes the water runner and launcher overrides. Use the
[DLNO-STC user guide](../../pyscfad/dlno_stc/README.md) for library defaults,
the MP2 versus CC pipelines, quadratures, and integral storage choices.

## Bulk water16 from the completed SGX benchmark

The completed November 10 SGX energy-and-force run is recorded in the local
`results/water16_water24_sgx_dlno_ccsdt_frozen64_recovery_20261110/` archive:
`recovery_report.md` and `water16/attempt_1/settings.json`. These archived
outputs are excluded from this repository.
It used cc-pVTZ, 16 frozen oxygen cores, 16 atom-grouped Boys fragments,
LIS cutoffs `1e-4`/`3e-5` Eh, and **`pair_energy=1.5e-5` Eh**.
The saved result was `-1221.4137050689542` Eh with a `(T)` SE of
`0.099105` mEh, finite forces and 16/16 converged local responses. Its
2541.76 s wall time and 45.22 GiB peak RSS describe that historical run only.

The new [launcher](run_water_16_bulk_sgx.sh) uses the exact archived geometry
in [water_16_bulk.xyz](water_16_bulk.xyz), whose SHA256 is
`6696d451e230da0b4cdeed720ab0944a7ec2081011886420291f3bb989e3926f`.
This bulk-carved cluster differs from the minimized `water_16.xyz` below.
The [domain input](water_16_bulk_thresholds.json) preserves every recorded
domain threshold except the requested change to `pair_energy=5e-6` Eh:

| Domain parameter | Archived run | New example | Current library default |
| --- | --- | --- | --- |
| `pair_energy` (Eh) | `1.5e-5` | `5e-6` | `5e-6` |
| `pao_norm` | `1e-4` | `1e-4` | `1e-2` |
| `bp_occ`, `bp_primary`, `bp_ed` | `0.985`, `0.999`, `0.9998` | Same | Same |
| `bp_pao`, `domain_pao`, `ed_pao` | `0.98`, `1e-4`, `0.995` | Same | Same |
| `metric_rank`, `near_pair_distance`, `multipole_order` | `1e-10`, `3.5` bohr, `4` | Same | Same |

The current `ed_pao_rank=1e-6` is also explicit in the input; the old run
predates that independent option. The public CC builder has **no implicit
occupied/virtual LIS cutoffs**: both arguments are required. The general
water runner defaults to `1e-4`/`1e-5` Eh; this launcher explicitly sets
`1e-4`/`3e-5` Eh.

From the repository root, with PySCFAD and its STC, generated SGX and optional
C++ CCSD extensions built in the selected Python environment:

```bash
PYTHON_BIN=.venv/bin/python THREADS=32 \
  examples/dlno_stc/run_water_16_bulk_sgx.sh \
  --output-dir output/water_16_bulk_deterministic
```

This runs cold and warm SCF followed by the full CCSD(T) energy and analytic
forces. The launcher explicitly selects deterministic W, CCSD and (T), with
**three fitted Laplace points per LIS for (T)**. The archived run used
stochastic (T); its sampling controls are not part of this example.

The command uses generated mixed-precision SGX, per-pair screening,
grid level 2, overlap fitting, `1e-10`/`1e-8` energy/potential tolerances,
resident half-transformed correlation integrals, and the packed CC virtual
ladder (`vvvv_direct=False`). The C++ ladder is preferred when its extension
and sufficient memory are available. SCF does not build global DF factors.
See the [SGX settings and build guide](../../pyscfadlib/pyscfadlib/sgx/README.md#build-and-use)
for backend, screening and precision options. Ordinary Boys remains selected.
Numerical settings are saved in `settings.json`; energy, forces and fragment
diagnostics go to `deterministic.json`.

`MAX_MEMORY_MB=900000` and `DF_MEMORY_MB=500000` preserve the old workspace
budgets; set these environment variables for your machine. They are not
predictions or hard limits on process memory. The launcher uses the current
pipeline and separate fitted grids: six global MP2 points and three (T) points
per LIS, replacing the old ten-point smooth-triples setting. Changed code,
pair screening and quadratures mean the historical energy is not a regression
target for these new commands. This example has been checked for input/CLI
consistency; the updated full water16 calculation has not been rerun.

## Minimized water-cluster runner

The geometry in `water_16.xyz` is a copy of the existing
`stc_qc_paper/run/xyz/water/water16_min.xyz` geometry (Angstrom). Its XYZ
comment is metadata from that file, not the energy calculated here. Supply a
different 16-water geometry with `--xyz` if needed. The shared driver also accepts
`--waters N` and freezes one oxygen core orbital per water molecule.

For the 10-water cluster, `water_10.xyz` is copied from the same collection's
`stc_qc_paper/run/xyz/water/water10_min.xyz`. Run it with the same method settings:

```bash
THREADS=8 PYTHON_BIN=.venv/bin/python \
  examples/dlno_stc/run_water_10_ccsd_t.sh \
  --max-memory-mb 20000 --df-memory-mb 1000 --vvvv-direct \
  --output-dir output/water_10_ccpvtz_stc
```

This uses 30 atoms and 10 frozen oxygen cores. Fresh DF-factor construction is
included in forward SCF timing and also recorded separately as `water.df_factors`.

From the repository root, in an environment with PySCFAD, mpi4py and the current
STC native extension installed:

```bash
THREADS=8 PYTHON_BIN=.venv/bin/python \
  examples/dlno_stc/run_water_16_ccsd_t.sh \
  --max-memory-mb 10000 --df-memory-mb 1000 --vvvv-direct \
  --output-dir output/water_16_ccpvtz_stc
```

Adjust `PYTHON_BIN`, `THREADS`, memory, and output directory for the machine.
The launcher defaults to this repository's `.venv/bin/python` and uses Python
isolated mode so an inherited `PYTHONPATH` cannot substitute another PySCF
checkout. Startup prints the actual interpreter, PySCF version and source path,
and rejects the old Boys interface before doing expensive work.
The launcher starts exactly one MPI process, sets numerical thread limits before
imports, and disables Open MPI's default process binding so its threads can use
the allocated CPUs. Scheduler or container CPU restrictions still apply. The
memory setting is a PySCF/W workspace budget in MB, not a hard overall RSS cap.
The lower-memory CCSD virtual ladder is enabled by default (`--vvvv-direct`);
`--no-vvvv-direct` opts out. Other intermediates
can still be large. The 928-AO cc-pVTZ calculation is substantial and needs
adequate memory, scratch space, and runtime. This input uses PySCF's default DF
auxiliary basis (`--auxbasis` selects another). It now defaults to prebuilding
packed SCF factors on disk in `scf_cderi.h5` with a separate 1000 MB workspace;
the traced SCF and its gradient stream these factors. The same explicit J/K
workspace keeps SCF panels bounded even when other calculation state already
occupies more than 1000 MB; unrelated resident memory no longer collapses these
panels to the minimum size. This allowance does not impose a process RSS limit.
Native integral derivatives are enabled. W processes four targets together by
default, sharing MP2 propagation and its reverse across each water fragment.
An eight-thread benchmark at the largest domain size reduced combined W/MP2
forward and reverse time from 38.4 to 21.8 seconds, with peak resident memory
3.45 versus 3.50 GB in the isolated benchmark. `--matrix-target-batch-size 1`
reduces amplitude storage if needed. These storage/blocking choices preserve
the method and domain/LNO thresholds; the benchmark compared every element of
the resulting W matrices and input derivatives.

The original in-memory configuration needs about 15.32 GB for one full global
DF tensor (928 AOs and 2224 auxiliary functions). Without native integral
derivatives, a three-component derivative tensor is about 45.97 GB. Setting
`max_memory` alone does not prevent those allocations. Disk storage removes this
startup problem, but later local CCSD and gradient intermediates can still exceed
10 GB. The packed disk factors themselves occupy about 7.67 GB; building them
also needs temporary disk space. `--df-storage incore` restores RAM storage.
Add `--df-builder direct` to construct the final packed dataset without a second
full tensor on disk. Generation, whitening, prefetch buffers and HDF5 staging
are sized together under `--df-memory-mb`; an allowance too small for the metric
and one shell panel raises an error before generation. This optional builder
preserves the existing factor layout and derivative path. `standard` remains
the default. See [serial DF construction](../../README_SERIAL_DF_SCF.md#optional-direct-to-final-layout-df-construction)
for the memory policy and interrupted-file cleanup.
Completed disk factors are reused only when their saved geometry/basis/version
metadata and packed dataset dimensions match the current input.

For lower resident memory, `PYSCFAD_STC_CLEAR_JAX_CACHES=1` releases unused JAX
compilation caches at eager calculation boundaries. It preserves live derivative
tapes and scientific settings, but increases compilation time and does not
guarantee a fixed RAM ceiling. `PYSCFAD_DLNO_RESOURCE_PROFILE=1` logs the cleanup
stages and their measured memory changes.
The launcher defaults to this cache cleanup, direct local DF inputs with shared
disk-backed adjoints, synchronous JAX CPU dispatch, and disabled Eigen multithreading;
explicit environment values override those defaults. `PJRT_NPROC` still follows
`THREADS`, so the JAX CPU worker pools use the requested parallelism.

This example explicitly keeps the separate occupied-factor cache for the SCF response off.
For this water16 input it retains about 1260 MiB and saved only 3.6 seconds in
the measured response after the J/K workspace fix (136.5 to 132.9 seconds).
That comparison reused a warm recorded SCF pullback; it included the occupied
cache's allocation and first fill. To opt in, set both
`PYSCFAD_SCF_OCCUPIED_CACHE=1` and `PYSCFAD_SCF_RESPONSE_FACTOR_CACHE_MB=2048`
(the latter is a MiB cap). The cache is released after each implicit root solve.

Local DF integral blocks use the smaller of the configured DF workspace and
half the available process-memory headroom, after output reservations. There
is no separate fixed stripe cap. The shared reverse forms packed AO-pair bars
directly. A representative water16 domain benchmark
with synthetic bars took 17.12 seconds with dense bars and the former 128 MB target,
versus 9.78 seconds with packed bars and the 512 MB target; coordinate derivatives
agreed within `9e-16` for each matched tile size. This is a component benchmark,
not a new full calculation; the quoted 128/512 MB targets describe that historical
comparison rather than the current automatic workspace policy.

The STC extension is required in the selected environment; ordinary PySCFAD
imports alone do not establish that it is installed. Build it for this branch
and Python environment with the CMake instructions in
[`pyscfad/dlno_stc/README.md`](../../pyscfad/dlno_stc/README.md#build-and-verification).
This requires pybind11, Armadillo, and OpenMP. Rebuild after native source changes.

Use a PySCF installation with OpenMP enabled to parallelize integral construction
as well as the STC extension. Configure Armadillo/BLAS library discovery for the
selected environment as described in the build instructions.

The current branch also supports resident half-transformed DF inputs. Set
`PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE=half` to enable them for this example;
the core driver defaults to `half`, while these launchers retain the measured
lower-memory `direct` policy. Both modes share the local integral adjoints.
Resident inputs consume additional memory; the main guide explains their
lifetime and the advisory process-memory policy.

The component timings above are historical measurements. Checkpoint
compatibility is checked against the current selections, settings and code;
see the [checkpoint guidance](../../pyscfad/dlno_stc/README.md#mpi-and-checkpoints).

The runner's defaults and explicit overrides are:

- Restricted DF-HF, cc-pVTZ, one frozen oxygen core orbital per water.
- Boys atom-grouped fragments and finite domains, with `pair_energy=5e-6` Eh.
- Deterministic W, supplying the strong-domain MP2 baseline and selecting the
  smooth CCSD LIS; occupied/virtual W cutoffs are `1e-4`/`1e-5`. Weak-pair MP2
  completes the baseline. This CC calculation does not run a separate scalar
  strong-domain MP2 solver.
- Explicit stochastic (T), with 20000 pilot samples, seed 71, and total triples
  energy SE target `0.1` mEh, allocated by original occupied target counts.
  This runner override is separate from the deterministic library default.
- One automatically fitted global six-point MP2/W grid and a separate fitted
  four-point (T) grid for each nonempty LIS. Each grid is fixed during its
  associated response. The energy SE excludes quadrature, domain, LNO and
  force errors.

The total energy, gradient in Eh/bohr, fragment components, sampling diagnostics
and settings are written to `result.json` in the output directory. Forces are
the negative gradient. Fragment progress is checkpointed in `checkpoint/`;
resume an interrupted run using the same input, settings and output directory
with `--resume`. Changed settings or code may invalidate the checkpoint.
The checkpoint requires an exact canonical reference; small numerical changes
when rerunning multithreaded SCF can also reject a resume. Preserve that checkpoint
and start a fresh calculation rather than bypassing the reference check. A
completed result is never overwritten; use a new output directory for another
seed or calculation. Run the Python driver with `--help` for all options.

Every run also writes `timing_events.jsonl` as stages finish, plus `timings.csv`
and `timings.json`. The final table orders forward SCF, domain construction,
MP2, W density matrix, LIS, CCSD, and incremental stochastic (T), followed by
their reverse stages in the opposite order. JSON also contains attributable
per-fragment totals. Shared/global preparation appears in the overall totals.
Reports made before completion have `complete: false`; their residual time
includes any unfinished stage. Previous timing files are archived on restart.

Times are wall seconds, including compilation when it occurs. Nested timers
are partitioned rather than added twice. The fused strong-domain MP2/W pass is
split using native amplitude/propagation and density-contraction timers. Most
stochastic (T) derivatives are evaluated during sampling; this cost is reported
under reverse (T), although it executes earlier. Weak-pair MP2 energy and its
gradient share one compiled kernel and therefore have a separate combined row.
LIS includes fixed rank/frame/grid selection and differentiable LIS construction;
the fixed selection steps have no reverse step. Domain includes integral
transformations and their derivatives, plus remaining correction/ERI propagation
in both directions. Earlier reports assigned the forward integral transformation
and correction remainder to LIS, which made the comparisons misleading.
The JSON `phase_timings` table gives inclusive and exclusive times for each
instrumented operation; only the exclusive times can be added. `native_timings`
provides the native subcomponents inside these Python timers. MP2 reverse includes
amplitude/propagator reconstruction to save memory. SCF reverse solves an implicit
response problem and is not expected to take a fixed multiple of forward SCF.
An overhead row covers imports, I/O, checkpointing, cache
cleanup, and work outside the named stages so the rows add up to elapsed time.
The timing-enabled native extension must be rebuilt to split MP2 from W;
`native_split_unavailable` explicitly lists unsplit phases with an older binary.
