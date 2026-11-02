# Serial DF, SCF and pullback performance

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

## Ordinary serial STC gradients

`dlno_stc.value_and_grad(..., comm=None)` now scopes its final complete SCF
pullback with the existing resident-buffer and BLAS-controller helpers. Both
system and domain calls use this boundary. The separate
`pyscfad.dlno_stc.ccsd.value_and_grad` entry point also uses it for serial CCSD
and CCSD(T) response. An existing caller-owned reverse
scope is reused, including a deliberately disabled one. Work is synchronized
before an owned scope is closed. MPI paths are unchanged.

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

OOC geometry fusion still needs the existing explicit DF-input cotangent
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
reliable benefit. The retained metric-VJP plumbing saved only a small fraction
of the large OOC DF contraction. Combining the global STC input geometry with
SCF geometry saved more total host reverse time, but requires an additional
producer/finalization contract inside the currently opaque SCF tape. These
changes remain in the experimental study rather than this small integration.
No claim is made that OOC DF pullback has fallen from about 20 s to 9 s.

The quoted measurements are from the preceding kernel investigation. They are
not new timing measurements of the public-driver integration.
