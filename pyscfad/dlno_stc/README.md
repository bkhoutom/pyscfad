# STC selected-target MP2

STC owns Boys selection, exact-frame grouping, weak-pair correction, restart
serialization, three-tensor preparation and molecular response. Production
modules and examples do not import native DLNO. Public `frag_lolist` remains
a complete singleton partition. Internal `DomainGroup` entries merge only
identical saved supports, candidate ordering, metric ranks and virtual anchors.

```python
from pyscfad import dlno_stc
static = dlno_stc.build_domain_selections(mf, frozen=4)
energy = dlno_stc.kernel(mf, static, scope="domain", controls={"mode": "deterministic"})
energy, gradient = dlno_stc.value_and_grad(
    mol, build_mf, static, scope="domain", controls={"mode": "deterministic"},
    include_hf=True,
)
```

Nuclear gradients require coordinate-only molecule leaves
(`mol.build(trace_exp=False, trace_ctr_coeff=False)`). Selections, domains,
Laplace grid and anchors remain fixed through response; orbitals, Fock blocks,
RI factors, Boys localization and SCF are differentiated. Weak corrections
retain the original singleton topology. `scope="system"` retains the complete
active-space energy and system sampling policy. Its independent controls and
canonical specialization remain available through the low-level full solver.

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

Weak response releases endpoint values, cotangents and reverse tapes as they
are consumed and explicitly collects garbage once after the weak stage.
Automatic Python collection and strong-group cleanup remain active.

Endpoint replay aligns real virtual-orbital signs to the saved forward moments
inside the differentiated function, so JAX applies the same phase factors in
the reverse pass. The consistency check compares each moment order's Cartesian
RMS per virtual column with `atol=rtol=1e-9`; canceled components use their
column's scale. Centers and energy gaps retain componentwise checks. Changes
beyond signs and these roundoff bounds are rejected.

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

## Sampling budget

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
candidates, AO/auxiliary supports, saved PAO anchors, `local_before_fock_v1`,
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
