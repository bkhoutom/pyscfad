# Domain and whole-system STC-MP2

## In-memory whole-system MP2

`scope="system"` runs unweighted MP2 over all active occupied and virtual
orbitals, with full molecular AO and auxiliary support. After frozen-space
selection, it Boys-localizes only the active occupied orbitals. The active
virtual space is represented by a complete PAO-derived frame: pivoted QR
selects exactly one independent projected AO column per active virtual, then
metric Cholesky orthonormalization retains the entire selected space. No
domains, support restrictions, virtual-space truncation, pair screening or
weak-pair corrections are constructed. Existing calls default to
`scope="domain"` and retain their five-input weighted calculation.

```python
from pyscfad import dlno_stc

e_domain = dlno_stc.kernel(mf, static, controls=controls)
e_system = dlno_stc.kernel(mf, scope="system", frozen=0, controls=controls)
energy, mol_bar = dlno_stc.value_and_grad(
    mol, build_mf, scope="system", frozen=0,
    controls=controls, include_hf=True,
)
gradient = mol_bar.coords  # Eh/bohr
```

`scope` chooses the physical calculation; `controls["mode"]` chooses
deterministic or stochastic evaluation. System calls require `static=None`.
System `frozen=None` means zero; an integer freezes that many leading occupied
MOs, and a list freezes the specified occupied or virtual MO indices. Domain
calls require `static` and take the frozen setting from `static.frozen`;
their separate `frozen` keyword must remain `None`.

System preparation passes exactly `foo`, `fvv`, and fitted `B` in memory to
`_stc_mp2.solve_full`. All three use the same active occupied Boys coefficients
and complete local virtual coefficients. With selected AO labels `J`, the
virtual construction is `A = Cv @ (Cv.T @ S[:, J])`, followed by
`Cloc = A @ inv(L.T)` for `A.T @ S @ A = L @ L.T`. The full projected
`foo = Co.T @ F @ Co` and `fvv = Cloc.T @ F @ Cloc` are generally
non-diagonal and passed directly to the solver. Preparation does not rotate
them into Fock eigenvectors: that would change the sampling frame.

The Boys reference orbitals/coordinates and virtual AO labels are fixed
metadata selected before constructing the preparation VJP. Current Boys
orbitals, virtual projections, metric Cholesky, Fock blocks and fitted factors
remain differentiated. System localization uses the shared Boys-domain
setup: `init_guess="atomic"`, `conv_tol=1e-10`, and localization-response
GMRES settings `restart=120`, `maxiter=100`. The Boys wrapper derives its
gradient tolerance from `conv_tol` (about `3.16e-6` for this default).
Pass `lo_kwargs` to system `kernel` or `value_and_grad` to change these
options, including an explicit `conv_tol_grad` if needed. Domain scope uses
the options already saved in `static.lo_kwargs`. Low-level traced
calls to `prepare_system_inputs` must supply `boys_reference` and
`virtual_anchor_columns`, selected from concrete data using the helpers in
`domain.py`. Only these choices are fixed, not the coefficient arrays.

Boys replay starts from the reference-aligned occupied coefficients and solves
for the remaining small rotation. This avoids poorly conditioned response
coordinates for large canonical-to-local rotations while retaining orbital
and coordinate derivatives. CIAH inner cutoffs tighten with the requested
localization gradient tolerance so near-converged replay can reach it.
As in the Boys-domain workflow, preparation performs one reference
localization followed by one short differentiable replay. It adds no separate
Hessian stability search or localization restart loop.

The original unweighted algorithm uses half-exponent
Laplace dressing. The domain solver retains full-exponent dressing and its
two projection inputs. The shared matrix-exponential VJP returns full symmetric
Fock cotangents, including off-diagonal response at a diagonal canonical
primal. The three system bars are applied immediately to a saved preparation
VJP, then one implicit SCF response closes the molecular derivative. Optional
HF energy and response are added once. No tensor exchange or replay files are
used by either production scope.

Whole-system `value_and_grad` forms each local Fock block from the converged
canonical SCF energies: `U = C_active.T @ S @ C_local` and
`F_local = sym(U.T @ (eps_active[:, None] * U))`. Its `build_mf` must return
converged canonical orbitals with their matching orbital energies; do not
rotate or otherwise replace these outputs after SCF. The local Boys/PAO
frames, full off-diagonal Fock blocks, and fitted factors are retained.
Energies, canonical coefficients, overlap, and local-frame rotations all
remain differentiated, and the saved SCF pullback completes their response.
This avoids an additional AO Fock/J/K construction and its reverse during
input preparation. The Fock response inside SCF itself is still required.
If two active occupied energies or two active virtual energies differ by
`1e-9` Hartree or less, the driver automatically retains AO-Fock preparation.
SCF's eigensolver suppresses rotations within such degenerate blocks, so
reconstructing the Fock from its outputs would lose off-diagonal response.

Standalone `kernel(mf)` and low-level `prepare_system_inputs` retain the
general AO-Fock projection. The latter accepts `fock_from_scf=True` to opt
into the same canonical-SCF specialization. Its partial mean-field VJP is
different from independent Fock/coefficient differentiation; equivalence
holds after composing with canonical SCF response for nondegenerate active
blocks. Low-level callers must check this prerequisite themselves. At finite SCF tolerance,
the two local Fock representations can differ by the SCF residual.

`prepare_canonical_system_inputs` is retained as a low-level deterministic
reference for checking energies and coordinate gradients with the same active
spaces and Laplace grid. High-level `kernel` and `value_and_grad` always use
the local frame and reject `controls["canonical_fock"] = True`; remove that
setting from earlier system inputs. Canonical sampling results use a different
frame and must be assessed separately from local sampling variance and timing.

The low-level native `canonical_fock` option requires exactly diagonal `foo` and `fvv`.
Its Fock cotangents represent derivatives of the diagonal energies; their
off-diagonal entries are zero. Independent full-matrix derivatives require
the default native path, including at a diagonal primal. Both paths reuse one
kept-pair Gram matrix for direct and exchange contributions, and combine the
owner's cotangent updates in one matrix multiplication when the bounded
workspace fits.

Both scopes support converged real restricted closed-shell DF references and
coordinate-only nuclear derivatives. System preparation reuses the reference's
global fitted AO factors and auxiliary molecule. An attached out-of-core CDERI
cache is streamed in bounded blocks during preparation and the orbital-coefficient
pullback; the full AO factor and its cotangent are not loaded into memory on the
supported real, coordinate-only Cholesky path. Nuclear integral derivatives are
still evaluated. Without cached factors, preparation retains the integral-direct
fallback. Cached factors must match the reference geometry, basis and Cholesky
auxiliary metric, which must be positive definite. System
calls with no active occupied or virtual orbitals return zero correlation
energy and derivative, and still support `include_hf=True`.

The supported real, coordinate-only out-of-core DF J/K reverse uses the exact
occupied density factors, including frozen-core occupations. Coordinate and
orbital pullbacks contract thin occupied panels instead of storing a full packed
AO-integral cotangent; large panels spill to temporary HDF5 scratch. The external
SCF coordinate reverse omits density bars when they are unused. SCF response
matvecs still use the generic density derivative. Unsupported inputs and
forward-mode tracers retain generic fallbacks; SCF and response tolerances are unchanged. This
reduces reverse storage but does not remove the resident STC tensors, dense
auxiliary metric, proposal tables, or scratch-disk requirements.

Serial dense molecular reverse operations use scoped BLAS limits through
`PYSCFAD_DENSE_BLAS_THREADS`, a positive integer defaulting to `1`. Each scope
restores the previous BLAS limits on exit, including exceptions, and leaves
OpenMP limits unchanged. Set this variable to the allocated physical-core count
when testing parallel dense work, while keeping `OPENBLAS_NUM_THREADS=1` and
`MKL_NUM_THREADS=1` for BLAS calls inside OpenMP kernels. One MPI rank with
single-thread dense BLAS is a useful starting configuration; measure memory,
scratch use, and timings before adding ranks or dense threads.

On Linux, `_stc_mp2.parallel_runtime_info()` reports the actual capped native
OpenMP team, each worker's current CPU, and its allowed affinity mask. Passing
an already loaded GNU OpenMP library filepath probes that runtime's own team;
it never loads a new runtime or resets affinity. The probe uses at most 32
workers, matching the sampling ceiling; other native parallel kernels can use
larger allocated teams. Compare worker mask coverage with the inherited
physical-core allocation before expensive work. Instantaneous CPU IDs need not
be distinct, because the scheduler can temporarily place workers on the same
CPU. The clean water16 driver performs this check before allocating DF data.

In stochastic system mode, `system_workload_cutoff` defaults to the original
absolute threshold `6.5e-3`: keep virtual column `a` for occupied `i` when
`abs(weight)**0.25 * norm(T[:,i,a])` exceeds this threshold. This is distinct
from the domain's relative `workload_cutoff`. `virtual_keep_fraction` overrides
the norm rule. Retained sets partition exact and sampled work; they do not
truncate the physical MP2 target or construct orbital domains. Pilot and
production streams are independent. Energy and reverse use the same production
draws with frozen proposals, including zero-energy draws with nonzero bars.
Energy standard error does not bound gradient uncertainty.

System scope removes domain approximations but retains separate Laplace
quadrature and stochastic errors. Its reference is full-space DF-MP2 with
matching frozen space and auxiliary basis. Hold quadrature fixed for a gradient
and check both inverse-denominator and derivative errors over the relevant
denominator interval.

The system-frame regressions can be run from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_ENABLE_X64=True \
    python -m pytest -q pyscfad/dlno_stc/test/test_system_local_frame.py \
    pyscfad/dlno_stc/test/test_system.py pyscfad/dlno_stc/test/test_system_cache.py
```

They compare deterministic local and canonical energies and coordinate
gradients using identical active spaces and Laplace nodes/weights, verify
frozen virtual exclusion and the complete PAO gauge, and check a general
three-tensor cotangent against coordinate finite differences. The last check
detects accidentally fixed continuous orbitals even when invariant MP2
energies still agree. Sampling variance and timing are separate comparisons.

The full fitted occupied-virtual tensor, numerical bars, root AD residuals, dense auxiliary
metric, and proposal tables reside in memory. The working-set guard includes
pair-conditioned virtual tables scaling as `no**2 * nv` and auxiliary-group
tables scaling as `ngroups * no * nv`; it is a conservative estimate rather
than a measured peak guarantee. MPI ranks each hold a complete independent
replica and duplicate exact work. Energy and all three bars are averaged;
the combined standard error is `sqrt(sum(sigma_r**2))/nranks`. Only root owns
molecular preparation/AD and returns the public result; workers return `None`.
All ranks must call the same public entry point; root selects the scope.

`examples/dlno_stc/water_tetramer_full.py` is a standalone driver accepting
`@input.par`. It runs total STC-MP2 energy and gradient, writes JSON/NPZ results,
and accepts `--reference` to additionally compute a native full-space DF-MP2
energy and gradient. `--mode stochastic` selects sampling. Its timings separate
DF setup, initial SCF, and the complete STC energy/gradient call; optional native
reference time is reported separately.

## In-memory weighted finite-domain MP2

`driver.kernel` computes finite Boys-DLNO correlation energy from weighted
Laplace strong-domain terms and the existing weak-pair correction.
`driver.value_and_grad` additionally propagates all five numerical cotangents
through domain preparation, Boys localization, and one implicit SCF response.
Set `include_hf=True` to add the HF energy and its molecular derivative once.
Each function processes one domain at a time; the numerical solver receives
arrays in memory and writes no tensor exchange files.

The five inputs are `foo (no,no)`, `fvv (nv,nv)`, `B (naux,no,nv)`,
`target_projection (1,no)`, and `partner_weight (no,no)`. They use the same
orthonormal local frame before Fock diagonalization. Both Fock blocks remain
full matrices. The target is the raw projection of the original Boys orbital;
the partner weight is formed from the raw projections of its strong partners
into that host frame. The fitted `B` must not be fitted again by the solver.

### Optional extension build

Use one activated Python environment supplying PySCFAD and pybind11, with
Armadillo (including its BLAS/LAPACK dependencies), CMake, and an OpenMP C++
compiler available through the environment or cluster modules. From the
repository root:

```bash
cmake -S pyscfad/dlno_stc/cpp -B build/dlno_stc \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$(command -v python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$PWD/pyscfad/dlno_stc"
cmake --build build/dlno_stc -j 4
python -c 'from pyscfad.dlno_stc import _stc_mp2; print(_stc_mp2.__file__); assert hasattr(_stc_mp2, "solve_full")'
```

This builds an optional module for an editable checkout. Ordinary package
imports work without it; requesting the native backend without a usable module
raises a build instruction. No compilation or downloads occur during import.
The target uses Armadillo's normal linkage and needs no Q-Chem or MPI libraries.
Ensure the dynamic loader can find Armadillo's shared dependencies when using
a nonstandard library prefix (`LD_LIBRARY_PATH` on Linux). Build products are
ignored by git; this recipe does not add wheel packaging.

For nested threading, start with `OMP_NUM_THREADS=2`,
`OMP_MAX_ACTIVE_LEVELS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`.
Increase threads only after checking the workload and allocated CPU resources.

The native kernels parallelize exact contractions and sampling with exclusive
output-column ownership. Full-system pair derivatives use occupied-pair
symmetry; projected-domain contractions reduce private one-target adjoints.
Integral dressing uses the packed occupied matrix and contiguous virtual
blocks, and its reverse reuses the point's integral-adjoint buffer. With these
outer OpenMP loops, use one BLAS thread per worker to avoid nested pools.

Kept-set matrix products use up to 64 MiB of workspace per occupied worker;
larger pairs fall back to tiles bounded by `virtual_block_size`. There are no
complete integral-gradient replicas per worker. Sampled updates use fixed
1024-draw batches, with at most `min(OMP_NUM_THREADS, OMP_THREAD_LIMIT, 32)`
batches resident, independent of the production sample count. Dispatch uses
only useful tasks, and scatter workers exclusively own columns through a
stable update partition. At 32 workers, four-role records and partition pointers
occupy 7 MiB plus small metadata, below the 8 MiB scatter allowance. Logical
batch seeds and per-column update order are independent of team size; sampled
moments and bars pass bitwise equality tests across teams. Other kernel
reductions can still differ slightly in floating-point order. Memory guards
include worker scratch.

### Calling the driver

The caller supplies a converged real restricted closed-shell DF reference and
fixed `DomainSelections` from the existing `pyscfad.dlno` construction:

```python
from pyscfad.dlno_stc.driver import kernel, value_and_grad
from pyscfad.dlno_stc.controls import default_laplace_grid, DEFAULT_ENERGY_TOLERANCE

roots, weights = default_laplace_grid()
controls = {
    "mode": "stochastic",
    "energy_tolerance": DEFAULT_ENERGY_TOLERANCE,  # Eh; 0.3 mHa total
    "laplace_roots": roots,       # explicit nonnegative float64 quadrature
    "laplace_weights": weights,   # same length, finite
    "virtual_block_size": 64,
    "auxiliary_group_size": 32,
    "global_seed": 713,
}
e_corr = kernel(mf, static, controls=controls)
energy, mol_bar = value_and_grad(
    mol, build_mf, static, controls=controls, include_hf=True,
)
gradient = mol_bar.coords  # Eh/bohr
```

`static` must contain a complete singleton Boys partition, an explicit frozen
setting, and compatible fixed atom, partner, and rank selections. Select PAO
anchors from concrete common data before tracing each preparation. The driver
does this automatically; the selected indices are fixed while their continuous
orbital coefficients remain differentiated.

Nuclear gradients require `mol.build(trace_exp=False, trace_ctr_coeff=False)`
and no traced `r0` leaf. The local integral-direct reverse currently requires
a positive-definite auxiliary Coulomb metric. The driver requires the supported
direct transformation helper and rejects the compatibility fallback that would
cache local AO-pair CDERI. An empty retained virtual space contributes zero
energy and bars.

The clean energy/gradient drivers use the original six-point fixed Laplace
roots and weights returned by `default_laplace_grid()`, scaled by 2.6. The
high-level API also supplies this grid when both grid fields are omitted;
explicit custom grids remain supported for validation and low-level calls.

For sampling, set `mode="stochastic"`. High-level `kernel` and `value_and_grad`
default to adaptive sampling with `energy_tolerance=3e-4` Eh (0.3 mHa) when
neither a tolerance nor `production_samples` is supplied. An explicit
`production_samples` retains fixed-count sampling. The whole-system solve
receives the total tolerance unchanged. For domain scope, each strong domain
solve receives `energy_tolerance / n_d`, where `n_d` counts strong domain
specifications only; native weak terms do not consume this budget. The driver
logs the total and per-domain budgets and preserves the caller's controls.

The adaptive tolerance must be positive and finite. It is a statistical energy
standard-error target estimated from pilot samples, not an accuracy guarantee.
Production sampling has no default cap: the pilot's allocation is rounded up,
with at least 10,000 draws per active adaptive residual. Unrepresentable
allocations raise an overflow error instead of silently truncating the budget.
`max_production_samples` is an optional explicit limit (`0`, or omitting it,
means uncapped); an explicit limit can leave uncertainty above the target.
Pilots default to 100,000 draws. Whole-system adaptive sampling follows the
collaborator's trace-based Laplace variance budgets and refines difficult pilots
to 1,000,000 total draws before allocation. Domain sampling retains equal
Laplace-point budgets and a single pilot. Explicit `pilot_samples` and
`min_production_samples` override their defaults; explicit `production_samples`
retains fixed-count behavior. Sampling and cotangent updates remain bounded
in at most 32 batches of 1,024 draws, limited by the allocated OpenMP team and
thread limit, independent of the total production count.
Native diagnostics include weighted point budgets, actual production counts,
the explicit limit, and pilot refinement seeds when used. Timing uses the
current batched sampler, and proposals and random streams remain those of this
implementation, so allocation policies agree without identical realizations.
Quadrature bias and gradient uncertainty are separate. `virtual_keep_fraction`
chooses the fraction of virtual columns evaluated exactly; alternatively use
the norm-based `workload_cutoff`. Residuals exhaust the remaining index space,
so this choice changes computational work rather than the physical energy.
`auxiliary_group_size` partitions the actual fitted `B` rows into contiguous
computational groups. Both direct and exchange sampling variances contribute
to the logged energy standard error. Production counts are positive in both
energy-only and gradient calls, including zero-energy cases with nonzero bars.
Forward and reverse use the same production realization and frozen proposals.

With a supplied communicator, every rank estimates a complete independent
replica of each domain. Energy and all five bars are averaged; standard errors
combine as `sqrt(sum(sigma_r**2))/nranks`. Root owns molecular preparation and
pullbacks and returns the public result; workers return `None`. All ranks must
call the same driver entry point. Per-domain and per-replica energy errors
combine into a logged total stochastic energy error, which does not bound
gradient uncertainty. Laplace quadrature error is a separate systematic error
and must be checked for the actual denominator range.

### Water-tetramer validation

`examples/dlno_stc/water_tetramer.py` reuses the supplied water-4 geometry,
cc-pVTZ settings, SCF checkpoint, and DF data. It checks all 16 finite-domain
dimensions and selections, includes all 24 weak pairs, and compares correlation
energy, total energy, and all 36 gradient components with the native reference.
The runner generates that reference with the current native DLNO code. The
supplied September 30 saved run predates the October 2 weak-multipole sign and
prefactor correction, so its weak energy and gradient are historical values.
The original fixture and selections are read without modifying them.
Its explicit 40-point Gauss-Laguerre grid uses a fixed scale of 5.5; the runner
reports inverse-denominator and derivative quadrature errors over the actual
local spectral range. This grid is specific to the validation fixture.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u examples/dlno_stc/water_tetramer.py \
  --reference-dir /path/to/dlno_stc_test/water_tetramer \
  --selections-dir /path/to/dlno_stc_test/forward/run_water4_boys \
  --output-dir output/dlno_stc_water_tetramer
```

Omit `--selections-dir` to rebuild the fixed selections using the saved
reference parameters. Add `--energy-only` for a forward check. The same command
can run under `mpiexec -n 2`; root performs molecular AD. Outputs include
`validation.json`, `results.npz`, and `native_reference.npz` with a companion
provenance JSON. Reuse the freshly generated native baseline with
`--native-reference output/dlno_stc_water_tetramer/native_reference.npz`;
the runner checks fixed selections, geometry, native Python source fingerprints,
and the current HF energy before accepting it. `--mode stochastic` reports deviations and
the driver's sampling diagnostics without applying deterministic tolerances.

With the supplied finite cc-pVTZ fixture (`frozen=4`), deterministic validation
against current native code gives correlation energy `-1.0663636080910655` Eh
and total energy `-305.28029303807926` Eh. The total-energy difference is
`1.14e-13` Eh and the largest gradient difference is `9.46e-12` Eh/bohr.

The previous three-tensor packet APIs below remain available for their live
diagnostic/export callers. Their finite packet scalar omits the two projection
inputs and therefore has a different derivative contract.

`dlno_stc` exchanges one local `foo`, `fvv`, `B` problem at a time,
accepts an external scalar energy and three unit-seed tensor cotangents, and
reconstructs its molecular derivative. Full-support Boys packets can replace
target-MP2 terms; finite packets represent an external three-tensor scalar.
The existing `pyscfad.dlno` drivers and defaults are untouched. The example
backend in `test/reference.py` is a tiny deterministic oracle; it is not an
STC implementation.

## Supported scientific branch

The first energy replacement supports real, restricted, closed-shell,
float64, orthonormal **Boys targets with full AO, occupied, virtual, and
strong-pair support** and an explicit frozen-space setting. Every target is a
single occupied Boys orbital. The occupied packet frame remains in Boys order;
its full `foo` matrix can be non-diagonal. The virtual frame uses the complete
active virtual space. An arbitrary occupied rotation mixing the target with
its partners changes the meaning of a fixed `target_index`.

For target `f = target_index`, the backend computes

```text
E_f = sum_{j,a,b} t[f,j,a,b] * (2 (fa|jb) - (fb|ja)),
(ia|jb) = sum_P B[P,i,a] B[P,j,b].
```

The amplitudes solve the noncanonical MP2 problem from the **full** `foo` and
`fvv` blocks. The result returns `energy` and `cotangents` for `foo`, `fvv`,
and `B`. These are full-array real cotangents for a unit energy seed. Molecular
gradients are in Eh/bohr; a force has the opposite sign.

For the `boys_target_mp2_v1` energy-replacement path, finite projected
EDs retain a geometry-dependent target projection and partner weight in
the current DLNO method. A fixed target index and the three-tensor packet cannot represent
those terms, so `build_stc_domain` rejects finite support. The separate
`finite_three_tensor_v1` export and pullback path below accepts finite EDs.
Test energy identifiers `quadratic_test_v1` and `whole_domain_mp2_test_v1`
are never accepted as Boys fragment replacements. CCSD(T), LIS densities,
weak-pair estimates, and stochastic STC accuracy are outside this first
energy-replacement path.

## Finite PySCFAD domains: three-tensor exchange

`pyscfad.dlno_stc.finite` provides a separate, opt-in path for the
existing PySCFAD finite strong EDs. `export_finite_run` saves one
`fragment_####/input.h5` per Boys target with exactly `foo`, `fvv`,
and `B`. The packet uses a **pre-semicanonical orthonormal local
basis**. Its full occupied and virtual Fock blocks can be non-diagonal.
The occupied candidates follow the saved strong-partner Boys order
and are metric-orthonormalized. Pivoted QR selects fixed PAO parent columns
for the virtual frame; their indices are saved as
`virtual_anchor_parent_columns`. Those parents are projected into the
retained PAO subspace and metric-Cholesky orthonormalized during export and
replay. Neither frame is Fock diagonalized. Each orbital column is
sign-fixed by its largest AO component. `B` uses those same local
coefficients and the selected auxiliary atoms. Packet metadata
records the global target label, selected AO/atom lists, strong partners,
and dimensions. `target_index` is `null`, since the global Boys target
is generally a projection across local occupied orbitals.

For an external scalar `E(foo, fvv, B)`, a result file contains its
energy and unit-seed bars for those three tensors.
`imported_finite_value_and_grad` checks the saved request and result,
replays the finite preparation in the same orbital frame, and pulls
the bars through the shared domain construction and SCF response.
`load_finite_static_from_run` and `load_finite_disk_packet` restore
the selections and validated packets for a separate import process.
The external backend determines the scalar; the finite packet does
not assign a physical MP2 target formula.

The existing PySCFAD finite-DLNO strong MP2 energy additionally uses
the geometry-dependent target projection and partner weight, while
the complete reference also contains weak-pair terms. Its full
energy/gradient therefore cannot be reproduced by only three imported
bars. Full support remains the supported branch for the earlier
`boys_target_mp2_v1` energy replacement; it is not a requirement
for finite tensor export or pullback.

### Full-support limit of the finite construction

`dlno_stc.domain` owns both domain frames. `build_stc_domain` keeps the
original complete Boys occupied order and active virtual frame for
`boys_target_mp2_v1`, with an integer `target_index`.
`build_local_strong_ed_domain` constructs the finite orthonormal frame and
its target projection and partner weight. Both consume the same fixed DLNO
selections and continuously rebuilt orbital data.

If every atom/AO and strong partner is retained **and** the occupied and
virtual selections retain their complete active ranks, the finite occupied
and virtual *spaces* become the full-support spaces. The coefficient columns
can still differ by orthogonal rotations and signs. With overlaps
`O = C_occ_full.T @ S @ C_occ_local` and
`V = C_vir_full.T @ S @ C_vir_local`, the corresponding packets obey
`foo_local = O.T @ foo_full @ O`,
`fvv_local = V.T @ fvv_full @ V`, and
`B_local[P,j,b] = sum_{i,a} B_full[P,i,a] O[i,j] V[a,b]`.
The target projection becomes a signed basis selector in that frame and
the partner weight becomes the identity. `force_full_domains=True` supplies
full atom/AO and pair support; PAO and metric-rank selections must also
retain all active orbitals. Thus equal AO extent alone is not a full-limit
test. The two packet formats and their metadata remain distinct so saved
full-support requests and finite requests keep their original frames.

## Implementation: selections, frames, and tensors

The existing `pyscfad.dlno` code makes the discrete domain choices.
`pyscfad.dlno._selection.DomainSelections` stores fragment labels, strong
partners, AO and PAO support, retained ranks, the frozen setting, and Boys
reference data. `pyscfad.dlno._restart.RestartManager` saves these selections
in `run.json` and `static.h5`. Given a current density-fitted SCF object
`mf`, `pyscfad.dlno.dlno_base.rebuild_domain_data(mf, static)` constructs the
continuous overlap, Fock, occupied orbitals, and PAOs in `common`. The flow is:

```text
static selections (dlno/) + current mf
    -> common continuous state (dlno/dlno_base.py)
    -> occupied/virtual frame (dlno_stc/domain.py)
    -> foo, fvv, B packet (dlno_stc/prepare.py)
    -> HDF5 request / external backend / HDF5 result
    -> packet replay and molecular pullback (dlno_stc/adjoint.py,
       dlno_stc/mp2.py or dlno_stc/finite.py)
```

For one domain, let `A` be its **ordered** extended-domain (ED) AO index
list, `S` the full AO overlap matrix, and `F` the current AO Fock matrix.
Subscripts select rows and columns; thus `S_AA` and `F_AA` are local square
blocks, while `S_A:` contains all columns. The metric projection of a
full-AO coefficient matrix `C` onto this AO support solves

```text
S_AA P_A(C) = S_A: C,           P_A(C) = S_AA^(-1) S_A: C.
```

`domain.py` constructs occupied and virtual coefficient matrices `Co` and
`Cv` with rows on `A` and respectively `o` and `v` columns. They share one
orthonormal local frame:

```text
Co^T S_AA Co = I_o,     Cv^T S_AA Cv = I_v,     Co^T S_AA Cv = 0.
```

For the full-support Boys path, `A` contains every molecular AO,
`Co = common.iao_coeff` retains the saved Boys target order without occupied
mixing, and `Cv = common.virtual_coeff` spans the complete active virtual
space. `build_stc_domain` checks full atom/AO/PAO, occupied/virtual rank, and
strong-pair support before exposing the integer `target_index`. These checks
make that local index a stable Boys target label.

For the finite path, `domain.py` projects the saved strong-partner Boys
occupied candidates onto `A`, applies fixed-rank `S_AA` metric
orthonormalization, and fixes each column's sign by its largest-magnitude
AO component. It builds a retained virtual PAO subspace `Q` using the saved
PAO and rank selections, projects away the local occupied space, and chooses
well-conditioned PAO **parent columns** by pivoted QR. If `G` denotes those
parent columns projected onto `A`, the virtual gauge is fixed schematically
by

```text
G = P_A(C_PAO[:, anchors]),
Cv = phase_fix(metric_cholesky_orth(Q (Q^T S_AA G), S_AA)).
```

The selected `anchors` are saved as `virtual_anchor_parent_columns`. Their
QR pivot **indices** are fixed discrete labels during traced replay, while
the PAO parent columns selected by those labels are rebuilt from the current
continuous state and remain differentiable. Likewise saved AO, atom, partner,
and rank choices stay fixed while overlap, Fock, orbitals, and DF integrals
respond to the current geometry. The occupied and virtual frames are
**pre-semicanonical**: neither Fock block is diagonalized. The finite domain
also computes the global target and strong-partner projections

```text
X_f       = C_f^T       S_:A Co,       W_f       = X_f^T X_f,
X_partner = C_partner^T S_:A Co,       W_partner = X_partner^T X_partner,
```

where `C_f` is the global Boys target column and `C_partner` concatenates
the selected strong-partner Boys columns. `LocalStrongDomain` exposes these
as `target_projection`, `target_weight`, and `partner_weight`. In a truncated
ED, `X_f` is generally a dense row over local occupied columns. In the
complete-support, complete-rank limit it becomes a signed selector. These
projections and weights are **not** part of the three-tensor packet.

`prepare.py` uses the same `Co` and `Cv` for both Fock and density-fitting
(DF) quantities. With `i,j` occupied, `a,b` virtual, and `P` fitted auxiliary
indices, the exchanged arrays are

```text
foo[i,j] = (Co^T F_AA Co)[i,j]                 shape (o,o),
fvv[a,b] = (Cv^T F_AA Cv)[a,b]                 shape (v,v),
B[P,i,a]                                      shape (naux,o,v).
```

For a local auxiliary basis indexed by `Q,R`, when its Coulomb metric
is positive definite, one equivalent DF construction is

```text
T[Q,i,a] = sum_{mu,nu in A} Co[mu,i] (Q|mu nu) Cv[nu,a],
J[Q,R]   = (Q|R) = (L L^T)[Q,R],
B[P,i,a] = sum_Q (L^(-1))[P,Q] T[Q,i,a],
(ia|jb)_DF ~= sum_P B[P,i,a] B[P,j,b].
```

Here `J` is the local auxiliary Coulomb metric and `L` is its Cholesky
factor. The code obtains `B` through differentiable
`pyscfad.lno.df.get_local_Lov(..., integral_direct=True)`, which transforms
raw three-center integrals into the occupied-virtual basis before applying
the auxiliary fit; a compatible packed-CDERI fallback is available. The
packet keeps the **full**, potentially non-diagonal, symmetric `foo` and
`fvv` matrices. Their diagonal entries alone are not MP2 denominators.

## What an external result means

For `boys_target_mp2_v1`, define `g[i,j,a,b] = (ia|jb)_DF`. The noncanonical
amplitudes `t` satisfy the full Fock-block equation

```text
sum_k foo[i,k] t[k,j,a,b] + sum_k foo[j,k] t[i,k,a,b]
  - sum_c fvv[a,c] t[i,j,c,b] - sum_c fvv[b,c] t[i,j,a,c]
  = g[i,j,a,b].
E_f = sum_{j,a,b} t[f,j,a,b] (2 g[f,j,a,b] - g[f,j,b,a]),
      f = target_index.
```

The tiny `test/reference.py` oracle diagonalizes both complete Fock blocks,
rotates `B`, divides by semicanonical denominators, and rotates `t` back
before evaluating `E_f` in the original Boys target frame. It is a
deterministic test oracle, **not** an STC estimator or a finite-domain
reference-energy solver. The full-support importer accepts only this Boys
target energy kind as an MP2 replacement; test-only `quadratic_test_v1` and
`whole_domain_mp2_test_v1` cannot replace a Boys fragment term.

For `finite_three_tensor_v1`, the external backend supplies an arbitrary
scalar `E_ext(foo, fvv, B)` and its three unit-seed tensor cotangents. Its
packet records `target_index=null` and a separate `global_target_id`, since
the global Boys target has no fixed local occupied index. The existing
finite DLNO-MP2 reference strong term also uses the geometry-dependent
target projection `X_f` and partner weight `W_partner`; its total additionally
contains weak-pair terms. No total finite-DLNO reference energy or gradient
follows from only `foo`, `fvv`, and `B` bars.

Let `q_f(mf, common; static, anchors)` denote preparation of the three
tensors for fragment `f`, and let `(foo_bar, fvv_bar, B_bar)` be the imported
derivatives of its scalar with respect to those tensors for a unit energy
seed. `adjoint.pullback_inputs` first reconstructs and compares `q_f` with
the saved packet, then applies the VJP

```text
(mf_bar_f, common_bar_f) = (D_(mf,common) q_f)^T
                           (foo_bar, fvv_bar, B_bar).
mf_bar = sum_f mf_bar_f
         + (D_mf common)^T (sum_f common_bar_f)
         + [optional (D_mf E_HF)^T 1],
mol_bar = (D_mol mf)^T mf_bar,
E = sum_f E_ext,f + [optional E_HF].
```

The bracketed Hartree-Fock terms are included by default in
`mp2.imported_value_and_grad` for a complete full-support import and are
off by default in `finite.imported_finite_value_and_grad`; a partial
full-support import requires `include_hf=False`. Fixed `static` selections,
fragment labels, and finite anchor indices are held constant in this
derivative; the selected parent PAO coefficients are continuous and traced.
The common-state and SCF pullbacks close once after all fragments rather
than once per packet.

## Source and test file map

The immediate Python files under `pyscfad/dlno_stc/` have these roles:

| File | Responsibility and main API |
| --- | --- |
| [`__init__.py`](__init__.py) | Re-exports the serial protocol and HDF5 APIs. Importing the package does not load MPI or an external solver. |
| [`protocol.py`](protocol.py) | Defines schema v1 and validates `foo`, `fvv`, `B`, metadata, results, fingerprints, and replay. `validate_inputs`, `request_fingerprint`, `validate_result`, and `check_replay` are the public checks; large `B` arrays are hashed and compared in bounded slices. |
| [`exchange.py`](exchange.py) | Implements `write_input`, `read_input`, `write_result`, and `read_result` for complete, fingerprinted HDF5 request/result pairs. Writes publish atomically and refuse overwrite by default. |
| [`parallel.py`](parallel.py) | `run_backend` checks the root packet and calls a backend serially or on a supplied MPI communicator. MPI is imported lazily; all ranks call the backend and only root receives its combined result. |
| [`_workflow.py`](_workflow.py) | Shared host helpers for restart identity (`run_payload`), Git revision status, per-fragment dimension reporting, and a conservative DF packet memory guard. |
| [`domain.py`](domain.py) | `build_system_local_frame` rebuilds complete active Boys/PAO spaces with fixed reference and AO-anchor choices. `build_stc_domain` enforces the full Boys target frame; `build_local_strong_ed_domain` constructs finite frames and target/partner weights. |
| [`prepare.py`](prepare.py) | Projects Fock and local DF factors into one frame. `prepare_system_inputs` streams fitted factors into the complete local system frame; `prepare_inputs`, `prepare_finite_export_inputs` and `prepare_finite_inputs` prepare domain packets. |
| [`adjoint.py`](adjoint.py) | `pullback_inputs` verifies the reconstructed packet and applies imported unit-seed tensor bars through JAX's VJP. |
| [`mp2.py`](mp2.py) | Full-support Boys workflow: `export_run`, `solve_request`, `load_static_from_run`, `load_disk_packet`, and `imported_value_and_grad`. It sums validated target energies and closes the molecular response, with HF included by default. |
| [`finite.py`](finite.py) | Finite three-tensor workflow: `export_finite_run`, `load_finite_static_from_run`, `load_finite_disk_packet`, and `imported_finite_value_and_grad`. It validates finite selections and anchor metadata, with HF omitted by default. |
| [`README.md`](README.md) | This implementation guide, packet contract, and runnable example. |

`test/` is part of the source tree and supplies fixtures, oracles, and
regressions; it does not contain a production STC backend:

| File | What it checks or supplies |
| --- | --- |
| [`test/reference.py`](test/reference.py) | Tiny deterministic quadratic and noncanonical target-MP2 oracles with JAX cotangents, plus a logical-block sampled backend with analytic cotangents. These are test backends only. |
| [`test/mpi_smoke.py`](test/mpi_smoke.py) | Executable real-MPI smoke test for collective and serial agreement, subcommunicators, preflight failures, and root-only disk I/O. |
| [`test/test_mpi_abort.py`](test/test_mpi_abort.py) | Launches the example with a deliberately failing root backend and verifies communicator-wide nonzero exit without publishing a result. |
| [`test/test_exchange.py`](test/test_exchange.py) | Covers HDF5 round trips, overwrite refusal, tampered requests, stale controls, invalid sample counts, incomplete files, and lazy package import. |
| [`test/test_protocol.py`](test/test_protocol.py) | Provides packet/metadata/result fixtures and exercises tensor shapes, dtype, symmetry, units, identity, fingerprint, seed/sample metadata, and bounded-slice replay validation. |
| [`test/test_reference.py`](test/test_reference.py) | Checks the deterministic oracles against hand contractions, an independent amplitude solve and finite differences; also covers empty spaces and serial sampling. |
| [`test/test_adjoint.py`](test/test_adjoint.py) | Compares imported tensor VJPs with direct JAX differentiation and tests frame mismatch rejection and the explicit finite replay tolerance. |
| [`test/test_prepare.py`](test/test_prepare.py) | Full-support water tests for the Boys frame, Fock/DF rotation, support restrictions, molecular finite differences, and deterministic target energy/gradient equivalence. |
| [`test/test_system_local_frame.py`](test/test_system_local_frame.py) | Complete system Boys/PAO frames, frozen virtual exclusion, local/canonical tensor and energy/gradient equivalence, fixed-metadata validation, and arbitrary-cotangent coordinate finite differences. |
| [`test/test_finite_prepare.py`](test/test_finite_prepare.py) | Finite water-dimer tests for orthonormal local frames, non-diagonal Fock blocks, fixed PAO anchors and orbital gauge, traced replay, and Fock/DF cotangent response. |
| [`test/test_finite_workflow.py`](test/test_finite_workflow.py) | Truncated water-dimer export/import test for finite metadata and anchor validation, packet replay, and a molecular gradient against displacement. |
| [`test/test_workflow.py`](test/test_workflow.py) | Tests dimension reporting, stable checkpoint fingerprints, disk solving and full Boys import, and the fresh-process example including partial-import labeling. |

No existing `pyscfad/dlno/` source file needs an edit for this opt-in path.

## Example

From the repository root, in a fresh work directory:

```bash
python examples/dlno_stc/workflow.py export --workdir ./stc_demo
mpiexec -n 2 python examples/dlno_stc/workflow.py reference-solve --workdir ./stc_demo
python examples/dlno_stc/workflow.py import --workdir ./stc_demo
```

`export` prints a per-domain table with `Fragment`, `ED AO`, `Occ`, and
`Vir` sizes. The example fixes an asymmetric water/STO-3G geometry, one frozen
core, and complete Boys domains. `reference-solve` is a separate process and
may run with one or several ranks. The final command reconstructs the
SCF/localization pullbacks in another process and prints energy and `dE/dR`.
Repeating `import` is read-only and reproduces the same aggregate.

For a single target, run the same exported directory with `--fragment N` on
both commands:

```bash
mpiexec -n 2 python examples/dlno_stc/workflow.py reference-solve --workdir ./stc_demo --fragment 0
python examples/dlno_stc/workflow.py import --workdir ./stc_demo --fragment 0
```

This prints the fragment MP2 energy and derivative, without the HF term. A
full `import` requires results for every exported target and reports an
incomplete run if any is missing. Result files are not overwritten; use a
fresh work directory to repeat `export` or `reference-solve` from scratch.

`examples/dlno_stc/run_mpi.sh [workdir]` runs all three commands. Set `RANKS`
and `THREADS` before launching; it exports `OMP_NUM_THREADS`, CPU binding
hints, and single-thread BLAS limits before Python starts. Check that
MPI ranks × OpenMP threads fit available cores and that each rank can hold its
replicated `B` input and temporary sample state.

## Request and result files

The existing restart store owns the global fixed selection checkpoint. Each
`fragment_####/input.h5` has `/inputs/foo`, `/inputs/fvv`, `/inputs/B`,
`/metadata`, and `/controls`; its result has `/energy`,
`/cotangents/{foo,fvv,B}`, `/metadata`, and `/diagnostics`. The small metadata
and controls datasets are JSON strings. For example:

```python
from pyscfad.dlno_stc.exchange import read_input, read_result
inputs, metadata, controls = read_input("stc_demo/fragment_0000/input.h5")
result = read_result("stc_demo/fragment_0000/result.h5",
                     inputs, metadata, controls=controls)
print(inputs["B"].shape, result["energy"])
```

For the in-process reverse path, `imported_value_and_grad` takes a
`packet_loader(fragment_id)` callback returning exactly
`(inputs, request_metadata, controls, result)`; `load_disk_packet(workdir,
fragment_id)` supplies that tuple from these files. In-memory loaders must
preserve the original controls and request metadata as well. To import a
subset, pass `fragment_ids` and `include_hf=False`.

The SHA-256 request fingerprint covers canonical problem metadata, sampling
controls, and the three arrays. The result echoes it. A completed result is
published only after its HDF5 file closes. Full-support replay checks the
rebuilt packet with `rtol=1e-9`, `atol=1e-11`; finite local-frame replay
uses `rtol=1e-9`, `atol=1e-9`. The external backend callable is
`backend(inputs, metadata, controls, *, comm=None)`. All ranks of a supplied
communicator call it; only rank zero returns a combined result and writes the
file. Its result metadata records a unit cotangent seed, backend/version,
actual sample count, and `fixed_sample` or `expected_energy_estimate`
derivative convention. It also records a nonempty `seed_replay` mapping: the
deterministic oracle uses `{"mode": "deterministic"}`, while sampled runs
record their logical block layout and global seed. A requested `global_seed`
must match the result replay metadata, and a request with positive
`sample_blocks` must report a positive actual sample count. The backend owns
all sampling reductions. Library functions accept a communicator and never
initialize/finalize MPI. The example application aborts its communicator if
an unrecoverable backend failure leaves ranks in different collectives.

The in-core prototype rejects a packet whose estimated input, bar, replay,
and AD temporary footprint exceeds the configured `max_memory`. The estimate
uses eight times the raw `B` size; it is a guard against obvious over-allocation,
not a measured peak bound. Large-domain streaming/distributed reverse paths
remain separate work. Actual STC estimator accuracy and production OpenMP/MPI
performance require validation by the external solver authors.
