# DLNO to external MP2 tensor exchange: implementation guide

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
| [`domain.py`](domain.py) | `build_stc_domain` enforces the full Boys frame; `build_local_strong_ed_domain` constructs the finite frame and internal target/partner weights. `select_virtual_anchor_columns` and the internal builder fix the finite PAO gauge for replay. |
| [`prepare.py`](prepare.py) | Projects Fock and local DF factors into one frame. `prepare_inputs` makes full Boys packets; `prepare_finite_export_inputs` returns a finite packet plus anchors; `prepare_finite_inputs` rebuilds it using saved anchors. |
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
