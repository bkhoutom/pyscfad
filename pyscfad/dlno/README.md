# DLNO: domains on top of LNO

The complete developer guide is in
[documentation/DLNO.md](../../../documentation/DLNO.md), with separate chapters
for [theory](../../../documentation/DLNO_THEORY.md) and
[implementation](../../../documentation/DLNO_IMPLEMENTATION.md). The theory
defines the quantities and equations; the implementation chapter maps those
equations to functions and explains shapes, AD, MPI, storage and restart.

```python
from pyscfad.dlno import DLNOMP2, DLNOCCSD, DLNOThresholds
from pyscfad.dlno import mp2

# mf is a converged density-fitted RHF reference.
cc = DLNOCCSD(mf, frozen=4, lo_type="iao")  # or lo_type="boys"
cc.ccsd_t = True
cc.kernel()
energy = cc.e_tot

mp2_result = mp2.kernel(mf, frozen=4, lo_type="iao")
mp2_total = mf.e_tot + mp2_result.e_corr
```

`DLNOMP2.value_and_grad` and `DLNOCCSD.value_and_grad` return the total energy
and molecular cotangent through progressive differentiation. MPI classes are
imported explicitly from `mp2_mpi` and `ccsd_mpi`. Boys targets use singleton
occupied-orbital groups and the existing `pyscfad.lo.boys.boys` localizer.

Start reading with `ccsd.kernel`, then follow:

1. `domain.build_domain_topology`: concrete target/domain/pair selection.
2. `_selection.build_domain_selections`: retain discrete indices and ranks.
3. `dlno_base.rebuild_domain_data`: rebuild current overlap, Fock, targets,
   PAOs and weights.
4. `dlno_base.build_strong_ed_domain`: current extended-domain orbitals.
5. `mp2_rdm` and `lis.build_fragment_lis`: selection densities and LIS.
6. `ccsd._solve_fragment`: the existing LNO impurity solver.
7. `ccsd._assemble_correlation_energy`: subtract LIS MP2 and add the complete
   strong-plus-weak local MP2 baseline exactly once.

For derivatives, continue with `ccsd.value_and_grad` and
`mp2._progressive_correlation_pullback`. Fixed choices are discrete indices,
not frozen current orbitals. Raw target/partner energy weights are distinct
from occupied-span projectors. Keep custom reverse rules, HDF5 blocking and
MPI frame/reduction order intact when editing numerical code.

The older dictionary-based LNO-prescreen adapter and its comparison examples
have been retired. Use the public DLNO drivers above for IAO/Boys calculations
and `pyscfad.lno` for plain LNO. Historical record fields such as `iao_coeff`
and `e_iao_mp2` also carry Boys data; see the guide's migration and checkpoint
sections before changing serialized names.
