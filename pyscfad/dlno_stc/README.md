# STC selected-target MP2

STC owns Boys selection, exact-frame grouping, weak-pair correction, restart
serialization, three-tensor preparation and molecular response. Production
modules and examples do not import native DLNO. Public `frag_lolist` remains
a complete singleton partition. Internal `DomainGroup` entries merge only
identical saved supports, candidate ordering, retained virtual ranks and PAO anchors.

```python
from pyscfad import dlno_stc
static = dlno_stc.build_domain_selections(mf, frozen=4)
energy = dlno_stc.kernel(mf, static, scope="domain", controls={"mode": "deterministic"})
energy, gradient = dlno_stc.value_and_grad(
    mol, build_mf, build_static=lambda mf: dlno_stc.build_domain_selections(mf, frozen=4),
    scope="domain", controls={"mode": "deterministic"},
    include_hf=True,
)
```

For gradients, ``build_static`` selects domains from the result of one SCF
performed under ``jax.vjp``. The driver retains that SCF pullback for the final
nuclear response, so no preliminary SCF is needed. The callback runs outside
differentiation on a detached reference, protecting the live SCF object's
structure from selection-only DF caches. Alternatively, pass existing
``DomainSelections`` as ``static``; do not supply both. System gradients need
only ``mol`` and ``build_mf`` with ``scope="system"`` and a frozen-space setting.

Nuclear gradients require coordinate-only molecule leaves
(`mol.build(trace_exp=False, trace_ctr_coeff=False)`). Selections, domains,
Laplace grid, retained ranks and PAO anchors remain fixed through response;
orbitals, Fock blocks, RI factors, Boys localization and SCF are differentiated.
Finite domains and system scope project saved anchors into their retained
virtual spaces and apply metric Cholesky orthonormalization. Pivoted QR selects
the discrete anchors once during concrete setup; preparation reuses them.
Weak corrections
retain the original singleton topology. `scope="system"` retains the complete
active-space energy and system sampling policy. Its independent controls and
canonical specialization remain available through the low-level full solver.

The integral-direct local-DF buffer target defaults to `512`. Set
`PYSCFAD_LNO_LOCAL_DIRECT_INT3C_BLOCK_MB` to override it. The forward allowance
uses decimal MB, while backward block targets use MiB; this is a workspace
target rather than a process-memory limit.

Domain selection and the domain driver lazily enable persistent JAX compilation
caching in `$XDG_CACHE_HOME/pyscfad/jax`, falling back to `~/.cache/pyscfad/jax`.
The automatic policy retains small kernels for reuse in subsequent processes.
MPI launcher ranks use separate `mpi-rank-N` subdirectories.
Numerical endpoint values and response tapes are rebuilt for each calculation.
This JAX setting is process-wide. Set `JAX_ENABLE_COMPILATION_CACHE=false` to opt
out. An explicit JAX cache directory retains its existing threshold policy;
a custom cache needs `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0` and
`JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1` to retain all small kernels.
An unavailable automatic cache warns and allows the calculation to continue.

Weak response builds each orbital screen once under `jax.vjp`, retaining the
returned screen and its orbital-algebra pullback. Compact transition moments
are shared by all incident pairs. Response replays only AO moments using that
exact screen, then applies the retained orbital pullback. Orbital and energy
response remain differentiated; there is no eigensolver reconstruction or
orbital-sign matching during replay.

Only one AO-moment derivative tape lives at a time. Orbital pullbacks retain
local algebra residuals until consumed; their memory can exceed the saved
orbital coefficients alone. Endpoint values, cotangents and both pullbacks
are released progressively, with one explicit garbage collection after the
weak stage. Automatic collection and strong-group cleanup remain active.
The AO-moment consistency check compares each moment order's Cartesian RMS
per virtual column with `atol=rtol=1e-9`; centers and energy gaps retain
componentwise checks.

## MPI across domains

For `scope="domain"` and a communicator with more than one rank, complete
group jobs are distributed round-robin. Each owner prepares its three tensors,
runs the local STC solver with OpenMP, and applies its saved preparation VJP.
Weak endpoints are prepared and replayed once globally; pair work is separately
partitioned. Cotangents are summed, and communicator rank zero drives the common
and SCF response once, including HF once when requested. Only rank zero returns
the public energy or `(energy, mol_bar)`; other ranks return `None`.

Multi-rank domain calls enable `parallel_scf_jk=True` by default. The existing
MPI DF executor distributes SCF J/K contractions, their density response, and
the independent three-center coordinate derivative blocks. Root retains the
SCF iteration and differentiation tape, localization, and coupled metric
response. Common Fock preparation and its pullback use the same executor.
Workers serve these collective phases, then resume their independent domain
work. Set `parallel_scf_jk=False` to retain root-only SCF/common response.

Every rank calls the driver with the same molecule and configured MF type.
Only root supplies controls and either `static` or `build_static`. The gradient
builder must support a worker branch that adopts root's canonical reference
without SCF:

```python
def build_mf(mol, *, mo_coeff_init=None, mo_energy_init=None,
             mo_occ_init=None, e_tot_init=None):
    mf = scf.RHF(mol).density_fit(auxbasis=auxbasis)
    mf.chkfile = None
    if mo_coeff_init is None:
        mf.kernel()
    else:
        mf.mo_coeff, mf.mo_energy = mo_coeff_init, mo_energy_init
        mf.mo_occ, mf.e_tot = mo_occ_init, e_tot_init
        mf.converged = True
    return mf
```

Workers use root's signed canonical/common arrays and matching basis definitions.
With MPI SCF enabled, the worker builder first receives placeholder canonical
arrays to create an unrun RHF/DF reference. The driver builds its global DF
factors for the service phase; it later calls the builder with root's converged
canonical arrays for domain work. Configure identical auxiliary bases and
real float64 DF factors on every rank. The service packs flattened full
in-core AO-pair factors automatically and verifies factor contents and
molecular definitions before its first contraction. Its private source view distributes coordinate
derivatives for both in-core and out-of-core factors. If attaching an
out-of-core CDERI file, attach it before the worker branch returns and make it
accessible to every rank. With `parallel_scf_jk=False`, workers need no global
CDERI. For energy-only `kernel`, workers supply matching MF skeletons and the
driver installs root's canonical state; its MPI DF service covers common
preparation, while any SCF performed before calling `kernel` is caller-owned.
`comm=None` and communicator size one retain serial domain execution. System
scope and low-level independent-replica MPI retain their existing behavior.
The total sampling tolerance is divided by `sqrt(global_group_count)` once,
and group seeds are independent of the owner rank.

Set numerical thread limits before importing JAX/NumPy. For example, with eight
allocated physical cores:

```bash
OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYSCFAD_DENSE_BLAS_THREADS=2 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
mpiexec -n 4 python examples/dlno_stc/water_tetramer.py \
  --mpi --xyz-file water_4.xyz --basis cc-pvtz --auxbasis cc-pvtz-jkfit \
  --output results/water4.json
```

Memory limits apply per process. Round-robin balances group counts rather than
runtime, so large domains can accumulate on one rank; the slowest owner limits
each phase. Rank-specific JAX caches remain persistent, and cold compilation
should be separated from warm scaling measurements.

## Array and native contract

All numerical requests contain exactly `foo(oo)`, `fvv(vv)`, and `B(Pia)` as
real float64 arrays. Fock matrices are full symmetric blocks in the same local
orthonormal frame as `B`, before Fock diagonalization. Selected local occupied
indices are distinct, nonnegative integer labels and travel as metadata.
Results provide a scalar and exactly three unit-seed bars in the input frame.

`_stc_mp2.solve_selected(foo, fvv, B, target_indices, controls, with_grad=True)`
returns the sum of selected target energies with identity
`boys_target_laplace_mp2_v1`. Its sampling error estimates that sum, including
within-group covariance. `_stc_mp2.solve_full` retains the complete-system
algorithm. The old orbital-weight solver and five-array boundary are removed.
Laplace quadrature weights and mathematical sampling probabilities are retained.

## Native D/W matrices and delayed reverse

`backend.solve_matrices` and `backend.pullback_matrices` are explicit host calls.
They use the same three prepared tensors and local occupied/virtual frames as
the energy solver. D is the native MP2 selection density, with its native
normalization; W is a separate signed energy-weighted approximation. Neither
call constructs a relaxed physical one-particle RDM.

Native v2 W energy is `0.5 * (trace(Woo) + trace(Wvv))`. Its reported energy
SE includes covariance between the occupied and virtual trace samples; it is
not obtained by combining their marginal errors independently. Consequently,
an energy-only W seed is equivalent to half-identity seeds on both matrices,
using the same saved production plan. Deterministic W has both traces equal
to its energy within numerical tolerance.

```python
import jax
import numpy as np
from pyscfad.dlno_stc import adjoint, backend, prepare

# mf, common, static and fragment_index are the existing preparation objects.
inputs, preparation_vjp = jax.vjp(
    lambda current_mf, current_common: prepare.prepare_inputs(
        current_mf, current_common, static, fragment_index), mf, common)
metadata = {"fragment_id": fragment_index, "target_indices": (0,)}
forward = backend.solve_matrices(
    inputs, metadata, {"mode": "deterministic", "target_batch_size": 1},
    matrix_kind="D")
# Supply downstream output bars later, in the same input frame.
input_bars = backend.pullback_matrices(
    inputs, forward, energy_bar=0.,
    occupied_bar=np.eye(inputs["foo"].shape[0]),
    virtual_bar=np.eye(inputs["fvv"].shape[0]))
mf_bar, common_bar = adjoint.apply_input_pullback(
    preparation_vjp, inputs, input_bars)
```

Use the actual saved local target slot(s) in `metadata["target_indices"]`;
`(0,)` above is only an example. Group outputs are sums, not a per-target stack.
The reverse differentiates `energy_bar*E + sum(occupied_bar*Xoo) +
sum(virtual_bar*Xvv)` with the full Frobenius convention, including asymmetric
seeds. Its three input bars are already seeded: do not multiply them by another
energy bar. Numerical inputs, target/frame metadata, grid, resolved native
options, native version and auxiliary offsets are fingerprinted. Reusing a
forward with changed inputs, controls or native version raises an error. Replay
records are owned in memory and are not serializable; v1 records cannot be used
with v2.

Matrix controls are separate from the established stochastic energy controls:

| Control | Matrix behavior |
| --- | --- |
| `mode`, `laplace_roots`, `laplace_weights` | Deterministic or stochastic; the existing six-point grid is the default. Stochastic D/W requires strictly positive weights. |
| `global_seed`, `threads` | Explicit seed and native thread count; matrix default threads is 1. Seeds are derived per fragment and batch. |
| `target_batch_size` | Explicit target batch, default 1; full orbital spaces are retained. |
| `memory_limit_mb` | Optional decimal-MB cap on current RSS plus conservatively estimated additional matrix workspace, checked before new host/native tensor allocation. Diagnostics include the breakdown and estimated process peak. |
| `auxiliary_group_size` | Default 32 rows per group when `aux_offsets` is omitted; an explicit partition takes precedence and is included in memory/replay accounting. |
| `matrix_tolerance` | Stochastic D/W Frobenius SE target for each summed matrix, default `1e-4`. Inspect measured SEs and `target_met`; allocation is not an attainment guarantee. |
| `domain_tau`, `virtual_domain_tau` | Stochastic occupied/virtual exact-core thresholds, defaults `.003` and `.0003`. |
| `pair_pilot_samples`, `min_samples` | Stochastic matrix pilot and production minimum, defaults 100000 and 64. |
| `resident_points` | Stochastic D only: 2 or 3, default 3. W retains its fixed and dressed factors and rejects this ineffective setting. |
| `energy_tolerance`, `system_workload_cutoff` | Stochastic D only, defaults `3e-4` and `.0065`. W's half-trace energy and covariance-aware energy SE come from matrix samples; W has no separate energy allocation target. |

Deterministic matrices reject sampling controls. Matrix calls reject fixed-count
energy options (`production_samples`, `pilot_samples`, production limits) and
unknown settings. Those existing controls continue to work on the separate
stochastic energy solver. New deterministic energy calls accept
`target_batch_size` and optional `threads`; stochastic energy uses the existing
OpenMP settings and rejects these new deterministic-only controls.

The W adapter explicitly enables virtual `w_virtual_reform`, matching the
supplied v2 driver. Other sampler options retain the supplied defaults,
including occupied stratum mixtures and virtual splitting. Effective options
are reported per batch and included in replay identity; experimental switches
are not exposed as new host controls.

Each stochastic batch receives independent streams and `1/sqrt(batch_count)`
of the requested SE budget. The returned SE is the square root of the sum of
batch variances, and the combined target flags use that measured uncertainty.
Each W batch contributes its full native energy variance, including its
occupied/virtual covariance, before independent batch variances are summed.
No matrix SE or energy SE is a nuclear-gradient uncertainty estimate.

Memory diagnostics include atom-weight arrays, simultaneous B/C proposal
caches, striped locks and bounded reverse update/radix buffers, as well as
factors, compact caches and matrix accumulators. Delayed reverse rechecks
current RSS because downstream arrays may have been allocated after forward.

Deterministic energy uses the supplied Laplace energy-only/fused kernel; it
does not build D/W. Explicit target batches bound its selected pair-Gram cache.
Current QR-anchor/Cholesky orbital construction, weak completion, MPI ownership,
unit-energy scalar bars and the gradient sign convention remain as before.
Adapter/source differences and original hashes are listed in
`cpp/native/SOURCE_MANIFEST.txt`. Existing restart identities are retained:
the scalar Laplace equations are unchanged, while their grids and scientific
controls remain covered by request fingerprints.

## Energy sampling budget

Public `energy_tolerance` is the total requested sampling standard error in Eh.
It is divided exactly once by the square root of the number of actual grouped
strong solver jobs: `group_tolerance = energy_tolerance / sqrt(group_count)`.
This assigns equal variance budgets, whose sum is `energy_tolerance**2` for
independent group streams. Sixteen targets in four groups receive
`0.0003 / sqrt(4) = 0.00015 Eh` per job. With merging disabled, they receive
`0.0003 / sqrt(16) = 0.000075 Eh`. Caller dictionaries are copied. Weak pairs
and MPI replicas do not enter the divisor; fixed-count overrides and replica
reduction retain their previous meaning. A group standard error describes
energy, never force uncertainty. Reused or unverified sampling streams do not
justify summing group variances as an independent combined uncertainty.

## Disk exchange and restart

`mp2.export_run` and `finite.export_finite_run` use one implementation. Export
writes one `fragment_NNNN/input.h5` per group, even for full-support selections.
Packets declare `finite_boys_target_mp2_v1`; the name identifies the selected
finite-frame functional even when its chosen support spans the molecule.
`mp2.solve_request` consumes the saved effective controls and writes `result.h5`.
`mp2.imported_value_and_grad` rebuilds all three tensors, checks their values,
applies imported bars and closes weak, common and SCF response.

```python
from pyscfad.dlno_stc import mp2, backend
mp2.export_run(mf, static, "run", controls=controls)
# One execution per exported group:
mp2.solve_request("run/fragment_0000/input.h5", "run/fragment_0000/result.h5", backend.solve)
static = mp2.load_static_from_run(mf, "run", frozen=4, controls=controls)
energy, gradient = mp2.imported_value_and_grad(
    mol, build_mf, static, lambda i: mp2.load_disk_packet("run", i),
    include_hf=True,
)
```

Use the packet adapter exposed by `backend` for your installed backend API;
`examples/dlno_stc/workflow.py` demonstrates the complete workflow using a tiny
independent deterministic reference. `finite.py` retains thin compatibility
entry points; its replay wrapper defaults to correlation energy (`include_hf=False`).
Partial job replay requires both `include_hf=False` and `include_weak=False`.

Fingerprints cover tensors, controls, molecular preparation, frozen space,
original/global target identities, actual local `target_indices`, ordered Boys
candidates, AO/auxiliary supports, retained PAO ranks and anchors, `local_before_fock_v3`,
group membership and sampling budget. Budget metadata stores `total_tolerance`,
`original_target_count`, `group_count`, and `effective_tolerance`. Replay does
not allocate again. Responses must echo the selected identities and frame.
Saved multi-group packets using the previous linear allocation must be
regenerated before replay; budget validation requires the square-root rule.

The existing STC `RestartManager` stores concrete selections and typed grouping
metadata; its registry accepts only required STC-owned classes. The model tag is
`boys_domain_model="local_selector_v1"`. Native DLNO checkpoints and old
weighted or target-free finite packets are rejected; there is no migration.
Use `merge_identical=False` consistently for diagnostic singleton export/replay.

## Build and checks

From the repository root, with PySCFAD, pybind11, Armadillo and OpenMP available:

```bash
cmake -S pyscfad/dlno_stc/cpp -B build/dlno_stc \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$(command -v python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$PWD/pyscfad/dlno_stc"
cmake --build build/dlno_stc -j 4
ctest --test-dir build/dlno_stc --output-on-failure
python -m pytest pyscfad/dlno_stc/test/test_selected_kernel.py \
  pyscfad/dlno_stc/test/test_full_kernel.py \
  pyscfad/dlno_stc/test/test_protocol.py \
  pyscfad/dlno_stc/test/test_finite_workflow.py
```

The optional extension links Armadillo and OpenMP; it requires no Q-Chem or MPI
library. Imports do not compile or download anything. Examples include the
small disk workflow, finite `water_tetramer.py`, and unchanged system-scope
`water_tetramer_full.py`.

Restart algorithm ABI 3 and grouping convention `boys_exact_domain_groups_v3`
require fresh checkpoints after restoring the anchored virtual frame. Older
finite packets using `local_before_fock_v1` or `local_before_fock_v2` must be
re-exported; the outer tensor exchange schema and checkpoint container version
remain 1.

## D/W-selected LIS and CCSD(T)

`pyscfad.dlno_stc.ccsd` supplies `build_static_selections`, `kernel` and
`value_and_grad`. Each original Boys target has its own native D or W pair
and LIS. Identical ED frames share preparation, while CC retains the global
active-HF reference, the original Boys target overlaps, and the established
stopped rotation of discarded orbitals. Retained orbitals, target overlaps,
embedding maps and numerical ED frames respond to differentiation.

```python
from pyscfad import gto, scf
from pyscfad.dlno_stc import build_domain_selections, ccsd

mol = gto.Mole(atom='O 0 0 0; H .1 -.75 .57; H 0 .8 .61',
               basis='sto-3g', verbose=0)
mol.build(trace_exp=False, trace_ctr_coeff=False)

def build_mf(current):
    mf = scf.RHF(current).density_fit()
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    return mf

mf = build_mf(mol)
domains = build_domain_selections(
    mf, frozen=1, force_full_domains=True,
    lo_kwargs={'conv_tol': 1e-12, 'conv_tol_grad': 1e-9})
baseline_controls = {'mode': 'deterministic', 'threads': 1}
matrix_controls = {'mode': 'deterministic', 'threads': 1}
selected = ccsd.build_static_selections(
    mf, domains, matrix_kind='D', thresh_occ=1e-4, thresh_vir=1e-5,
    baseline_controls=baseline_controls, matrix_controls=matrix_controls)
energy, gradient, components = ccsd.value_and_grad(
    mol, build_mf, selected, baseline_controls=baseline_controls,
    matrix_controls=matrix_controls, include_hf=True, return_components=True)
force = -gradient.coords
```

For W, select `matrix_kind='W'` and supply its occupied/virtual cutoffs
explicitly, for example `thresh_occ=3e-4, thresh_vir=1e-4`. These example
cutoffs are independent of the D cutoffs; inspect the reported LIS dimensions.
The strict rule is `abs(eigenvalue) > cutoff`, including signed W eigenvalues.
A zero cutoff retains the whole corresponding global active-HF manifold.
Saved labels remain fixed during derivatives and finite differences; numerical
orbitals and eigenvectors are rebuilt at the current geometry.
The implicit Boys response requires a stationary localization. Use sufficiently
tight saved `lo_kwargs` and check convergence before a force comparison; loose
localization can leave fragment-specific derivative errors even with a full LIS.

The correlation accounting is
`MP2 baseline + sum(CCSD + (T) - MP2-LIS)`. `e_mp2_base` includes weak-pair
completion; `e_mp2_weak` reports its included weak part. The MP2-LIS subtraction
uses the shared solver's exact denominators. The native matrix energy is a
diagnostic and receives zero energy seed in the correction reverse. The
six-point baseline can therefore leave the expected MP2 quadrature residual
relative to conventional full-space CC. `ccsd_t=True` retains the shared
solver's existing triples option; its convergence defaults are unchanged.

Baseline and matrix precision controls stay separate. Stochastic D/W requires
positive Laplace weights. W allocation targets matrix accuracy, so its matrix
controls reject an energy tolerance. Matrix standard errors do not bound the
nonlinear CC energy or nuclear gradient error. Components retain compact native
sampling diagnostics, without per-element error arrays or replay handles.
The component `baseline_standard_error` and the CC console SE refer to the
separately evaluated MP2 baseline. Inspect native target flags for sampling
allocation misses. Budget LIS CC tensors and shared orbital/SCF response in
addition to the native workspace memory guard.

`fragment_indices=(i,)` restricts CC corrections. The baseline remains complete
unless `include_baseline=False`, which returns only the selected correction
and forbids `include_hf=True`. This is the representative-fragment C20 API;
such a result is not a complete molecular CC energy or force.

For MPI, pass `comm`; complete groups run on round-robin owners and root closes
common/SCF response once. Worker `build_mf` must accept `mo_coeff_init`,
`mo_energy_init`, `mo_occ_init`, and `e_tot_init`, adopt those arrays without
SCF, and return a matching density-fitted skeleton. Only root needs controls
and selections; only root returns a result. Shared SCF/J/K work runs on root
in this initial CC path. Native replay and pullback closures remain local.

Optional serial `checkpoint_dir` and `resume=True` reuse the existing safe STC
checkpoint machinery. Fixed selections and completed scalar/host-cotangent
contributions are saved atomically; SCF/common response closes on the current
live reference after restoration. Identity includes matrix kind, cutoffs,
scientific controls, signed canonical state, group worklist, native version
and CC convergence settings. MPI progress checkpoints are explicitly rejected;
there is no distributed D/W replay serialization.
