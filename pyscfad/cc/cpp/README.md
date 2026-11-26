# Resident RCCSD residual kernels

`kernels.cpp` provides the packed virtual ladder and strided tensor helpers.
`residual.cpp` owns the remaining real restricted CCSD equations, derived from
PySCF's `ccsd.update_amps`. There are no NumPy contractions or Python callbacks
inside the native residual. Numerical work releases the GIL.

After optional resident-workspace preparation, the Python adapter in
`_native_ccsd_forward.py` calls `residual_from_integrals` once per update:

1. The virtual ladder forms occupied-pair triangular tau, contracts it with
   packed `vvvv`, and expands its half-scaled result directly into `r2`.
   Packed tau, packed output and per-worker integral panels occupy the same
   scratch arena later used by ovvv/ovov. Their values are overwritten every
   update; only the allocation and virtual-pair index map persist. There is
   no full seed tensor, residual zero-fill or seed-add pass on this path.
2. The remaining residual builds the Fock intermediates, processes packed `ovvv` in
   virtual tiles, computes the blocked `ovov` and `oooo` terms, and returns
   denominator-preconditioned amplitudes. Its GEMMs use an OpenMP team with serial BLAS. Long
   reductions with small outputs split the reduction dimension into private
   buffers and sum in worker order; other shapes split an output dimension.
   Private reduction storage is bounded by the larger input matrix. Packing
   and tensor updates also use OpenMP. The caller BLAS policy is restored
   after the update. No entry changes OpenMP or MKL dynamic-team policy.

The standalone `ladder_residual` and `residual` interfaces remain available
for compatibility and differential checks. The Python adapter also retains
the two-call path for older extensions. Both ladder paths use the same
allocation-free `ladder_into` numerical core, including direct GEMM output
accumulation with beta=0 for the first reduction panel and beta=1 thereafter.

Binary contractions use the same SciPy Cython BLAS API as the ladder. Native
views carry strides and share ownership of temporary buffers. The planner
uses row-major or transposed matrix views where possible, including padded
leading dimensions. Each flattened row/column group must have consistent
strides; negative or non-unit inner strides still require packing. Tau virtual
tiles therefore use their original row spacing without a copy. The planner
compares the two operands' reduction orders by packing bytes and otherwise
packs in parallel.
In particular, `icjb,cjba->ia` preserves the large contiguous `ovvv` operand.
This is a bounded CCSD contraction planner, not a general einsum implementation:
it does not implement diagonals, batch indices, broadcasting, or outer products.
Tau uses a separate fused outer-product kernel.

Within each `ovov` virtual block, only the borrowed `voov` matrix layout reused
by `bjkc,ic->jbki` and `cjia,cjib->ab` is cached. It is released immediately after
the second contraction. Single-use packed layouts use scratch or an ordinary
temporary. The cache key includes data address, shape and strides; mutable
intermediates are excluded. `ovov_layout_cache` reports peak retained `bytes`,
total `packed_bytes`, `hits`, and `end_bytes` (zero after the section).

Full tau and exchange-weighted doubles intermediates are shared across virtual
blocks and the final occupied contraction within one update. Half-tau and
response-tau are formed at their first consumers and released after their last
outer virtual block. Each is still built only once per update; they do not
overlap when the full virtual space fits in one block. Partial blocks retain
them across blocks to avoid recomputation. `ovov_tau_temporaries` reports their
peak live bytes and construction counts. Fock theta and exchanged doubles
updates use fused loops, and strided additions follow the destination's storage
order. The identical singles/occupied-virtual-Fock contraction is evaluated
once and reused. All amplitude-dependent caches are local to one call; subsequent
iterations always read the current amplitudes. The Python memory plan reserves
space for these additional layouts before choosing the virtual block size.

When the configured memory budget permits, a `ResidualWorkspace` lives with the
host ERIs for one solve. It retains unpacked `ovvv` in `(virtual,occupied,virtual,
virtual)` order and an `ovoo` matrix layout `(k,j,b,i)` used by the final occupied
singles contraction. Unpacking and preparing these immutable integral layouts
happen once. The same workspace also owns full tau/theta buffers; their values
are recomputed from the current amplitudes on every update. A single theta is
shared by the `ovvv`, `ovov`, and `oooo` sections. Retaining its storage avoids
repeated allocation/free work at the end of the occupied section.

The workspace holds its input owners, validates their views, and serializes
concurrent uses of its writable buffers. Replacing an ERI array or changing its
view rebuilds the cache. As with the virtual-ladder cache, host ERIs are assumed
immutable during a solve; callers editing them in place must clear
`eris._native_residual_workspace`. If the memory budget cannot accommodate the
extra layouts, the adapter retains the bounded, per-tile unpacking path. The
planner is refreshed after cache allocation. Profiles report retained integral
and amplitude-workspace bytes under `resident_reuse`.

The `vvvv`, `ovvv`, `ovov`, `oooo` and final Fock contractions share one scratch allocation for operand
packing and temporary results. Each contraction reserves the simultaneous
storage it needs and carves disjoint views from it; those views expire before
the next contraction reuses the allocation. A resident workspace retains its
high-water capacity across updates, while the nonresident path releases its
local scratch after the update. The Python memory plan reserves the maximum
phase requirement; the existing ovov bound also covers oooo and final Fock
packing. The ladder allowance follows the native thread
ceiling and panel plan. Cached input layouts own separate allocations, so
they remain valid when temporary storage is reused. Results returned by ordinary
`Engine::E` calls never borrow scratch. The ovov exchange result is consumed
inside `with_result`, which finishes both permutations before scratch reuse.

When a destination admits a BLAS matrix view and does not overlap an input,
`ovvv`, `ovov` and `oooo` GEMMs accumulate into it with alpha/beta, including padded and transposed
destinations. Other destinations use a scratch result followed by the strided
addition. The reduction-partitioned GEMM applies alpha/beta after combining
worker results. Profiles expose `direct_calls` and `result_bytes` alongside
the existing packed-byte counts. Under `resident_reuse`, `vvvv_scratch_bytes`,
`ovvv_scratch_bytes`, `ovov_scratch_bytes` and `oooo_scratch_bytes` report each phase's required high-water storage;
`shared_scratch_bytes` reports the retained arena capacity, not their sum.
`vvvv_pair_index_bytes` reports the separately retained index map. The integrated
path reports a `vvvv` section including tau packing, panel packing/GEMM and
direct expansion. Each output owns independent storage after the call.

Final Fock contractions also use direct GEMM accumulation into the residuals
and share the scratch arena if a strided input needs packing. The two large
doubles contractions therefore need no full result temporaries. Profiles
report `finish_scratch_bytes` and split `finish_detail` into `contractions`
and `symmetry_and_denominators`; these nested times are not additional sections.

The final symmetry/denominator update distributes occupied pairs and 32-by-32
virtual tiles. Each task owns both symmetry-related outputs. For diagonal
occupied pairs, only the lower virtual triangle is processed, including
partial diagonal tiles. This replaces the uneven triangular outer-i work
distribution. Both ladder interfaces pack tau over flattened occupied-pair/
virtual rows, and expand over flattened rectangular rows. The resident
workspace retains a small occupied-pair row map (`occupied_pair_index_bytes`)
alongside the virtual index map. DIIS, norms and energy already use rectangular
row partitions.

The explicit OpenMP work-sharing loops in these forward kernels and the native
`dfccsd_ovvv_vjp` backward kernel use static scheduling. The backward kernel's
virtual-index iterations own disjoint output rows and perform equally shaped
contractions. It remains C in `pyscfadlib` to preserve its existing ctypes/C ABI;
the forward extension uses C++/pybind11. Loop scheduling is independent of
`OMP_DYNAMIC`, which controls team sizing and is left to the caller.

Inputs are borrowed and read-only; outputs own new storage. The supported path
is real float64 resident RCCSD with cached packed `vvvv`. PySCF reference
equations remain available for uncached/direct integrals, insufficient memory,
unsupported solver modes, or an unavailable extension. The public update
returns amplitudes (not the unpreconditioned equation residual).

`solver.cpp` supplies direct convergence norms, weighted history dot products,
dense amplitude accumulation, and the fused correlation-energy reduction. For
symmetric RCCSD doubles, the packed-vector metric is one half of the full
doubles dot product plus one half of its composite-index diagonal (`i=j,a=b`);
singles retain unit weight. No triangular packing is needed for forward DIIS.
`_native_ccsd_solver.py` retains the convergence policy and small DIIS coefficient
solve. Its dense amplitude/error history uses the configured `diis_space` and
the same implicit error (trial minus previous extrapolated output) as PySCF.
It falls back to the host solver for unsupported modes, custom/disk DIIS, or
insufficient configured memory. There is no fixed RSS cap.

Memory admission uses `_native_ccsd_memory.py` to bound simultaneous live
buffers in each residual/solver phase. The shared arena contributes its maximum
capacity at that point, not a separate allocation for every contraction. Ordinary
contraction results and packing outside the arena are included with their
overlapping intermediates. Input arrays and populated DIIS slots already count
toward process RSS. The solver reserves only unfilled history slots and initial
amplitude growth while planning later iterations.

`ResidualWorkspace.memory_info()` distinguishes allocated capacity from populated
storage. Integral layouts/index maps are populated during construction; tau and
theta receive RSS credit only after an update fills them. Scratch credit tracks
completed writes and excludes unused panel capacity and alignment padding. The
planner adds only the peak beyond that populated storage to current RSS, including
any required contiguous copy of `vvvv`. It chooses the largest fitting virtual
block with the normal ladder panels, then tries a single panel if needed. A
fallback drops an unused resident workspace. These conservative buffer bounds do
not include a fixed runtime margin or claim that logical allocations equal RSS.

The energy kernel fuses tau construction with the direct/exchange contraction,
avoiding a full tau temporary. It evaluates both the initial correlation energy
and the energy after each amplitude/DIIS update. Fixed row reductions keep the
summation order independent of the OpenMP team size. The existing backward and
adjoint equations, including their packed metric, are unchanged.

For diagnostics, `_native_update(cc, t1, t2, eris, plan, profile=True)` returns
`(t1new, t2new, report)`, where `plan = _native_plan(cc, t1, t2, eris)` must be
non-None. The report includes section wall/CPU seconds and per-expression
packing, GEMM, direct result-accumulation time, calls, and packed bytes.
Packing time includes output allocation. Standalone intermediate construction
and transposed additions appear in section totals; they are not attributed to
individual expressions. The integrated ladder has its own `vvvv` section;
the compatibility two-call path times the ladder separately in its caller.
Normal solver calls disable this instrumentation. Average core-equivalents
are CPU seconds divided by wall seconds and can include runtime spinning.

Tests compare the complete update and ladder to PySCF, including partial tiles,
strided/read-only input, noncanonical Fock matrices and level shifts. Build
using the repository's CCSD extension instructions, then run:

```bash
python -m pytest pyscfad/cc/test/test_native_residual.py \
  pyscfad/cc/test/test_native_kernels.py \
  pyscfad/cc/test/test_native_ccsd_forward.py \
  pyscfad/cc/test/test_native_tensor_ops.py \
  pyscfad/cc/test/test_native_solver.py
```

## Lambda and generalized response

`lambda.cpp` implements the real RCCSD lambda intermediate builder and update.
It shares `tensor.h` with the forward residual: strided matrix views with leading
 dimensions, threaded packing, direct GEMM assignment/accumulation, bounded
reduction parallelism, and one reusable contraction arena. OpenMP loops use
static scheduling and the existing ladder and final symmetry/denominator kernels.
The Python caller scopes BLAS to one thread while native teams own parallelism.

A `LambdaWorkspace` belongs to one set of immutable converged amplitudes and
integrals. It retains theta, the occupied intermediates, `wvOOv`, and the combined
`2*wVOov + wvOOv` layout. Optional unpacked `ovvv` and `Wvvov` caches are selected
from the available memory budget; smaller virtual tiles and streamed W are
available. W construction reuses each composite ovov tile, and contracts into
its destination slice. Cached W is consumed directly as a matrix during updates.
The full tau tensor is released after cached-W construction; the update contracts
its outer-product contribution in factored form. One result buffer
is reused across ovvv, ovov and oooo, including the exchange-weighted lambda
intermediate. This buffer and the inverse-metric work buffers retain capacity
across iterations. No single-use packed layout is retained.

`_native_lambda_cpp.py` selects resource policy and preserves the conventional
reference/fallback in `_native_ccsd_lambda.py`. The generalized response solver
keeps covectors dense, applies the spin metric and denominators in native loops,
and uses the same weighted reductions/dense DIIS as the forward solver. The
source subtracts the affine zero-lambda term analytically. Only the final
boundary to the integral adjoint packs a vector. This preserves the original
packed-coordinate norm and DIIS coefficients, including composite diagonals.

Native profile entry points report wall/CPU time by section and packing, GEMM,
and accumulation by contraction. Workspace memory metadata separates resident
buffers, arena capacity, and populated storage. The planner reserves future
DIIS growth and compares overlapping phase peaks with the caller's budget;
there is no fixed system-size or RSS cap. Complex, unsupported, direct-DF ladder,
and insufficient-memory cases retain the existing reference path.

## DF integral pullbacks

`integral_pullback.cpp` implements the real `ovvv` and `vvvv` adjoints directly
to the DF factors. The Python adapter `_native_integral_pullback.py` supplies
aligned C-contiguous float64 inputs and scopes BLAS to one thread, restoring
the caller's settings on return. Both kernels use the same SciPy Cython BLAS
function pointer and OpenMP runtime as lambda. The factor contractions use
the shared `parallel_gemm` helper; GEMMs inside the `ovvv` slab loop stay serial.
The `vvvv` kernel reuses one cubic virtual slab across iterations, rather than
allocating one per worker. Older extensions retain the NumPy fallback.

The deterministic DLNO route no longer calls `libcc_vjp`. Its canonical and
legacy LNO triples entry points remain available through a lazy library loader;
ordinary DLNO imports do not load that compatibility library.

Directional derivative, reference contraction, layout and threading checks are
in `test_native_integral_pullback.py`. The full DLNO energy/gradient regression
also forbids legacy CC library calls and checks that both C++ pullbacks ran.

## Deterministic W

`deterministic_w.cpp` implements selected MP2 W and its delayed input adjoint
without Armadillo. Column-major matrix views carry leading dimensions and
retain the existing virtual-eigenbasis propagation and occupied-partner tiles.
GEMMs use the shared SciPy pointer, with `parallel_gemm` outside worker regions
and serial GEMM within them. The adapter `dlno_stc/_native_w.py` obtains the
symmetric spectra through SciPy LAPACK, scopes thread counts, and preserves
the existing matrix result/replay contract. It keeps no factors or amplitudes
in ordinary delayed replays. Sequential W explicitly retains its call-local
amplitudes and Grams until the occupied projection is consumed.

## Deterministic MP2/D and MRCC triples

The W propagator also supplies scalar MP2 without constructing W outputs and
D matrices with both amplitude legs differentiated. Scalar target batching
bounds amplitude storage; matrix-exponential pullbacks include degenerate
and off-diagonal Fock response.

`triples.cpp` and `triples_dressing.cpp` retain the bounded occupied-block MRCC
algorithm, shared dressing graph and grouped adjoints. `dense_matrix.h` owns
column-major buffers and strict borrowed views; transpose descriptors go
directly to the same SciPy LP64 GEMM pointer used by CCSD. Symmetric eigensolves
use SciPy LAPACK. The Python adapter scopes serial BLAS around native calls,
while C++ owns OpenMP parallelism and preserves the caller's dynamic policy.
There is no Armadillo header, link dependency or runtime loader on this path.

The stochastic legacy extension remains separate. Deterministic checkpoint
identities preserve mathematically compatible completed contributions; replay
objects remain local to their provider and are never checkpointed.
