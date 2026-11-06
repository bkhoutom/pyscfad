# Root-owned DF, SCF and pullback performance

## Optional direct-to-final-layout DF construction

The serial benchmark and water drivers accept
`--df-storage outcore --df-builder direct --df-memory-mb 500000`.
`standard` remains the default builder. This setting is independent of
`--fragment-df-mode`: it changes construction of the global SCF factors only.

The direct builder generates and Cholesky-whitens AO-pair panels, writing them
straight into one contiguous `j3c` dataset. Existing row readers and DF/SCF
pullbacks use the same layout and Cholesky gauge. It eliminates the full
temporary block tensor and subsequent assembly pass. For 64 waters with
cc-pVTZ/cc-pVTZ-JKFIT, the new tensor occupies about 490.4 GB instead of the
standard builder's approximately 980.9 GB peak for two tensors. This counts DF
construction only; other calculation scratch and an existing old output file
still take additional space.

`--df-memory-mb` is a decimal-MB workspace allowance. The direct planner budgets
the metric, two prefetched integral buffers, whitening/layout copies and a
bounded HDF5 conversion tile together, leaving 20% for native-library overhead.
The write tile is at most 64 MiB or 8% of the allowance, whichever is smaller.
A budget too small for the metric and an indivisible shell panel raises
`MemoryError` before generating integrals. This does not impose an OS RSS limit
or count unrelated SCF/correlation state as builder workspace.

For Python callers, build factors eagerly and attach them before tracing:

```python
from pyscfad.df import direct_outcore

direct_outcore.cholesky_eri(
    reference_mol, "scf_cderi.h5", auxmol=reference_auxmol,
    max_memory=500000,
)
# Inside the differentiable SCF builder, at that same geometry:
mf.with_df.attach_outcore_cderi("scf_cderi.h5")
```

Alternatively, an eager PySCFAD `DF` object supports
`df.incore = False; df.outcore_builder = "direct"; df.build()`.
That object's `max_memory` retains its existing PySCF convention: current RSS
is subtracted before passing the remaining allowance to the builder. The
explicit low-level call and driver `--df-memory-mb` pass the workspace directly.
As with other disk-backed factors, prebuild before JAX tracing and reuse only
at the matching geometry/basis. The direct path requires a full-rank positive
definite auxiliary metric; rank-truncated eigenvalue factors are unsupported.

Construction uses a hidden `.<filename>.direct-*.h5` file in the output
directory, already in the final layout. Completion atomically renames it over
the destination, with no data copy. The destination is a standalone DF file;
unrelated datasets in an old destination are not retained. Ordinary exceptions
remove the partial file and preserve the old destination. An uncatchable kill
can leave the hidden partial file for later cleanup.

The `dlno-stc` branch already contains the shared resident DF cotangent buffer,
compact/fused out-of-core geometry contraction, low-rank SCF exchange, bounded
response-factor caching and dense-BLAS thread helpers (commit `11ebb34b`).
The small integration described here uses those implementations without changing
the SCF response equation or adding the experimental OV solver.

## Priority changes retained

1. **Shared resident cotangent buffer.** Compatible SCF consumers accumulate
   into one owned CDERI cotangent instead of repeatedly allocating and adding
   full tensors. The DF producer consumes it once, including any incoming
   explicit cotangent. In the earlier water12 benchmark, this reduced peak RSS
   from about 18.9 to 11.5 GiB. Buffers belong to one reverse invocation; repeated
   seeds require separate invocations.
2. **Existing OOC streaming and compact/fused geometry.** Keep factorized J/K
   response, bounded disk panels and on-the-fly derivative contraction. The
   explicit staged SCF→DF benchmark already combines compatible contributions
   before the coordinate contraction. Different auxiliary models stay separate.
3. **Low-rank exchange, bounded occupied-factor cache and thread scopes.**
   Low-rank exchange is already enabled for eligible SCF responses. The optional
   occupied-factor cache is owned by one root solve. A dense-BLAS controller
   scope avoids repeated library discovery; it does not change thread limits.
4. **Cheap preconditioning where measured.** The existing
   `pyscfad_scf_df_precond=False` setting skips Coulomb/Woodbury setup and
   applications. The fixed-point equation already contains inverse orbital
   gaps, so another gap division is unnecessary. Water12 SCF+DF time improved
   about 5% in the study. Keep Coulomb available for systems where it improves
   convergence or total time; fewer iterations alone are insufficient.

## Serial and root-owned MPI STC gradients

`dlno_stc.value_and_grad` scopes its final complete SCF pullback with the
existing resident-buffer, occupied-factor-cache and BLAS-controller helpers
when SCF runs on root. Both system and domain calls use this boundary. The
separate `pyscfad.dlno_stc.ccsd.value_and_grad` entry point also uses it for
CCSD and CCSD(T) response. This includes serial calls and MPI calls with the
explicit `parallel_scf_jk=False`: fragments remain distributed, their completed
cotangents are reduced to root, and root closes shared response once.
An existing caller-owned reverse
scope is reused, including a deliberately disabled one. Work is synchronized
before an owned scope is closed. The default `parallel_scf_jk=True` uses the
distributed DF J/K service and its existing response ownership.

Eligible out-of-core DF sources built or attached inside the traced SCF
builder also combine global AO-Fock and implicit SCF geometry seeds at their
shared producer. Root performs one compatible geometry/metric reverse after
the fragment cotangents arrive. Set `PYSCFAD_STC_COMBINE_SCF_DF=0` before
tracing to use separate geometry responses.

Enable buffer registration **before tracing** `build_mf`. For this workstation,
the water8/water12 runs used eight physical cores and these thread settings;
water2/water4 used four native/dense threads instead:

```bash
export OMP_NUM_THREADS=8 OMP_THREAD_LIMIT=8 OMP_MAX_ACTIVE_LEVELS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export PYSCFAD_DENSE_BLAS_THREADS=8
export PYSCFAD_DF_SHARED_COTANGENT=1
export PYSCFAD_SCF_OCCUPIED_CACHE=1
export PYSCFAD_SCF_RESPONSE_FACTOR_CACHE_MB=1024
```

Use the native molecular-integral path and keep the preconditioner choice in
scope throughout the reverse, not only inside `build_mf`:

```python
from pyscfad import config_update, dlno_stc

with (config_update("pyscfad_moleintor_opt", True),
      config_update("pyscfad_scf_df_precond", False)):
    energy, mol_bar = dlno_stc.value_and_grad(
        mol, build_mf, static, controls=controls, include_hf=True,
    )
```

`False` is already the library preconditioner default; `True` restores Coulomb.
The cache budget covers optional factors, not mandatory metrics or scratch.
Thread counts and memory budgets remain caller/workstation choices.

**Ownership requirement:** a shared-buffer scope must reach the DF producer
pullback. A `build_mf` callback that builds DF from its molecule naturally does
so. Unregistered prebuilt factors retain explicit cotangents. If registered
factors were produced on a separate retained tape, either enclose the complete
consumer→SCF→DF chain in the existing
`pyscfad.lib._cotangent_buffer.reverse_scope()` or disable sharing with
`PYSCFAD_DF_SHARED_COTANGENT=0`. An SCF-only partial VJP must not consume a
shared buffer intended for a later DF reverse. Standalone public VJPs outside
an explicit scope still return explicit cotangents.

## Staged OOC benchmark and workstation choice

Externally staged OOC geometry fusion needs an explicit DF-input cotangent
boundary: enter `pyscfad.df._ooc_cotangent.geometry_scope(df_input)`, execute the
SCF reverse, then pass `scope.finish(df_input_bar)` to the DF producer reverse.
This is already wired in `TEST_general/test_scf/benchmark_common/driver.py`.
Keep `PYSCFAD_DF_OOC_CAPTURE_FACTORS=1` and `PYSCFAD_DF_OOC_COMPACT=1` for that
staged path. Setting `OOC_COMPACT` alone does not create a finalization boundary
inside an arbitrary public `build_mf` callback.

The water benchmark launcher selects `BENCHMARK_SCF_DF_PRECOND=0`; set it to
`1` to restore its earlier Coulomb setting. The driver records the resolved
choice in result metadata. This benchmark switch does not change library-wide
preconditioner defaults.

## Changes deliberately excluded

The exploratory OV solver and its implicit-differentiation hooks showed no
reliable benefit. The experimental retained metric-VJP plumbing saved only a
small fraction of the large OOC DF contraction and remains excluded. The
supported shared-producer geometry path above is separate from those solver
experiments.
No claim is made that OOC DF pullback has fallen from about 20 s to 9 s.

The quoted measurements are from the preceding kernel investigation. They are
not new timing measurements of the public-driver integration.
