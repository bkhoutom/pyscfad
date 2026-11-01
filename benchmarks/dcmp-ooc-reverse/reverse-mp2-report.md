# Retained canonical MP2 Lov

Scope: isolated candidate package only. Production ownership is limited to
`pyscfad/mp/dfmp2.py` functions `_outcore_nr_e2_fwd` and
`_outcore_nr_e2_bwd_impl`; tests are in
`pyscfad/mp/test/test_reverse_saved_lov.py`.

## Tests prepared before implementation

- `test_saved_lov_repeated_pullbacks_match_dense`: real water/STO-3G with
  Weigend auxiliary basis, arbitrary full MO coefficients and nontrivial orbital
  slices. Compare both molecular and auxiliary coordinate bars and every MO bar
  against independently differentiated dense factors. Apply two random
  cotangents, then repeat the first; verify output remains unchanged.
- `test_backward_reuses_lov_without_streamed_ao2mo`: wrap the real streaming
  AO2MO function and real MO-coefficient adjoint. Require exactly one forward
  transform for repeated pullbacks while preserving the necessary streamed MO
  adjoint reads. Frozen baseline should fail with two transforms after the
  first pullback.

Static AST parse passed. No numerical imports or tests ran on the login node.

Existing `test_dfmp2_outcore.py` test
`test_outcore_nr_e2_vjp_matches_dense_without_full_cderi_bar` explicitly expects
`forward_stream >= 2`; root authorized changing only that expectation to `== 1`
alongside implementation, and that update is now included.

## Red evidence and implementation

Root reported compute-node job 5821831: the reuse regression failed with
`ao2mo == 2` instead of `1`; the independent dense repeated-pullback comparison
passed. Root authorized implementation after that evidence.

The production diff consists of three changes: append the forward `out`
reference to the VJP residual, unpack it in `_outcore_nr_e2_bwd_impl`, and pass
`saved_output=out` to the direct three-centre pullback. There is no Lov copy.
The forward dtype handling, direct-path eligibility and all fallback branches
remain unchanged. Existing higher-order handling is unchanged apart from the
additional retained residual leaf.

Static AST parsing passed for all three changed Python files after the edit.
Compute-node green verification is pending and remains root-owned.

## Follow-up: physical density BLAS backend

Root assigned only `_int3c_mo_physical_density` in candidate
`pyscfad/df/_cderi_vjp.py` plus a new regression after profiling found the
PySCF einsum contractions select its sequential BLAS backend. Added
`pyscfad/df/test/test_reverse_dense_density.py`: four cases cover both
contraction orders, real/complex values, a strided auxiliary panel, and an
independent NumPy three-factor einsum reference. A guard rejects PySCF einsum
inside the helper.

Root confirmed red job 5822279: all four failed at that guard in 2.48 s.
After this evidence, replaced the four PySCF einsum calls with flattened 2-D
NumPy GEMMs within the existing dense-BLAS thread scope. Each branch uses two
GEMMs aggregating the auxiliary panels, avoiding one BLAS launch per auxiliary
row. Transpose/reshape ordering preserves `D[p,u,v]` and the final `D + D.T`;
no conjugation is introduced. Panel copies remain bounded by the current
auxiliary block. Static AST parsing passed for both changed files. Numerical
green checks remain root-owned; no numerical work ran on the login node.
