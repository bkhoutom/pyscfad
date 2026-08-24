# Final-closeout CDERI memory cache

## Summary

The MPI DLNO-CCSD gradient path now has an opt-in CDERI memory cache for the
rank-0 SCF reverse closeout:

```python
energy, gradient = DLNOCCSD.value_and_grad(
    mol,
    build_mf=build_mf,
    final_cderi_cache='memory',
    # other DLNO options ...
)
```

`final_cderi_cache='disk'` is the default and preserves the previous
out-of-core behavior. `final_cderi_cache='memory'` performs one sequential read
of rank 0's file-backed `j3c` dataset immediately before `scf_vjp`, reuses that
same NumPy array for every CDERI reverse call in the closeout, and releases the
cache when `scf_vjp` exits.

The cache is not stored in the differentiable DF/JAX pytree. It is a
context-scoped, process-local mapping for one canonical file path, so the final
closeout holds one full B tensor rather than one copy per reverse call or rank.

## Changed functions

### `pyscfad/df/addons.py`

- `_cderi_cache_key`: creates the canonical `(file, dataset)` cache key.
- `cderi_memory_cache`: allocates one NumPy array and fills it with
  `h5py.Dataset.read_direct`; the entry is removed on normal or exceptional
  context exit.
- `load.__enter__`: returns the resident array for the matching source and
  otherwise retains the existing disk/array loader behavior.

### `pyscfad/df/_cderi_vjp.py`

- `cholesky_eri_vjp_from_cderi_source`
- `cholesky_eri_vjp_from_cderi_block_fn`

Both pair-block reads now request C-contiguous arrays. An HDF5 read is already
dense; a resident auxiliary-major tensor produces only a bounded pair-block
copy, not a second full-tensor copy.

### `pyscfad/dlno/ccsd_mpi.py`

- `_final_cderi_cache_context`: validates `disk`/`memory`, obtains the DF source,
  and returns the appropriate context.
- `DLNOCCSD.value_and_grad`: adds `final_cderi_cache='disk'`, validates it before
  the expensive calculation, and activates memory mode only on rank 0 around
  the final `scf_vjp`. The preload time, tensor size, and shape are included in
  the existing progress/resource diagnostics.

## Tests

- `pyscfad/df/test/test_cderi_cache.py`: one-time preload, same-array reuse,
  canonical path matching, disk restoration, exception cleanup, and invalid
  source handling.
- `pyscfad/lno/test/test_final_cderi_cache.py`: disk/memory option selection,
  invalid-mode rejection, and density-fitting requirement.
- `pyscfad/df/test/test_cderi_vjp.py`: numerical equivalence between disk-backed
  and memory-cached Cholesky CDERI reverse paths.

Validation completed in `pyscfad_env`:

```text
focused cache tests:                  6 passed
full CDERI VJP + MPI safety helpers:  50 passed, 9 skipped
```

The skipped cases require optional native kernels unavailable in the test
environment. Memory mode is intentionally explicit: the caller must request a
node allocation large enough for one complete rank-0 CDERI tensor. Smaller
allocations should continue using the default `disk` mode.
