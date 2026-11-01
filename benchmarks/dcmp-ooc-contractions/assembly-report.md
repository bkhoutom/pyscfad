# Packed adjoint assembly candidate

Implemented only in the isolated `candidate/pyscfad` snapshot. The live
package and native sources/libraries were not changed by this task.

The candidate reuses `libnp_helper_vjp.NPdunpack_tril_2d_vjp` to write
normalized symmetric packed bars directly into the final auxiliary-shell
buffer. With `X = 2 Rtilde[p] @ S.T`, its off-diagonal values are
`Xij + Xji` and its diagonal values are `Xii`. The Coulomb contribution is
added in packed space from once-packed G/D, whose off-diagonal values are
doubled and diagonal values unchanged. This matches the existing native
coordinate driver's diagonal normalization without a full symmetrized
density, repeated symmetry scan, temporary packed array or diagonal pass.
J-only blocks need no AO-square density allocation.

`_pack_tril_symmetric_sum` validates float64 inputs, square batches and
writable contiguous float64 output before ctypes. Noncontiguous inputs
are copied safely. Native packing runs outside dense BLAS thread scopes.
The optional coordinate filler always has a companion dense reader;
unsupported native inputs retain the old explicit contraction. The
factorized producer retains its existing dtype/exact-symmetry checks and
disables the optional filler if the native packing symbol is unavailable.
The old dense arithmetic remains available, with its dense Coulomb-column
setup allocated lazily only when used.

Memory remains bounded by the existing shell partitions and reader caps.
The new path preserves one packed shell buffer and at most one bounded
K density block, without any full auxiliary-by-AO-pair allocation. Its
floating-point addition order differs for packed Coulomb updates, so the
acceptance criterion is numerical agreement rather than bitwise identity.

Tests were written before implementation. Root confirmed SLURM job
5826633 failed the two selected normalization tests solely because the
new helper did not yet exist (`2 failed, 16 deselected`, 4.18 s), then
authorized implementation. The 18 test cases cover literal diagonal and
off-diagonal values, transposed input, dtype and output-buffer rejection,
bounded one-row callback reads against explicit ip1 integrals, unequal
auxiliary/molecular atom mappings, complex/float32/asymmetric dense
fallback, and all J/K combinations with factor response enabled/disabled.
The full-gradient reference independently differentiates dense fitted
integrals and tensor contractions; candidate panels and solves are forced
to disk/small slabs in those tests.

Static checks completed: all three modified/added Python files parse with
`ast.parse`; diffs are localized to the new helper/interface and density
assembly section; `nm -D` confirms the existing shared library exports
`NPdunpack_tril_2d_vjp`. No numerical imports, tests, calculations or builds
were run on the login node. Numerical green evidence, microbenchmarks and
full dCMP gradient/performance acceptance remain root-coordinated work.

DSYR2K was considered but is not part of this candidate. Its symmetric
two-product triangle costs roughly the same arithmetic as the existing
single full GEMM and may lose the latter's batching efficiency. If later
measurements justify it, the existing cvhf_vjp target already links BLAS
and OpenMP and `vhf/fblas.h` already declares dsyr2k_; no new dependency
is required. Further design detail is in `/tmp/dcmp-assembly-design.md`.

Review revision: the packing helper now rejects output buffers overlapping
the contiguous native input. The native kernel reads both triangles while
writing packed elements, so overlapping storage could corrupt later reads.
The guard occurs after input contiguity conversion, allowing a safe copied
input while rejecting actual native-pointer overlap before any write. Two
new regression cases cover overlapping slices at different offsets and
assert the original input remains unchanged. There are now 20 test cases.
Root's running `packed_green` job 5826824 uses the preceding frozen
18-case snapshot; the final suite must validate this revision. AST parsing
of the revised helper and tests passed; no numerical login execution occurred.
