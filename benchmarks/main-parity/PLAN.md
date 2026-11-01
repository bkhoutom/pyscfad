# MKL verification and PySCFAD runtime parity

Goal: confirm upstream uses MKL, explain its remaining reference gap, and bring the customized fork's canonical DF-MP2 gradient runtime close to upstream without removing custom STC/DLNO behavior.

Architecture: independent investigations of benchmark boundaries, MP2 kernels and SCF response. Native builds and code experiments remain isolated until numerical validation and matched full dCMP timing pass. Use existing Python environments; preserve package versions and pinned libcintad.

Constraints: simple self-contained SLURM conditions; one MPI task, 16 Intel cores for full timings, matched affinity; MKL LP64 sequential with GNU OpenMP; DF-only SciPy threading unless explicitly testing another policy. Complete coordinate gradients must agree within 1e-7 Hartree/bohr, energies within 1e-8 Hartree. No silent change to frozen orbitals, basis, convergence criteria or derivative order support.

- [x] Verify active upstream PySCF and PySCFAD native hashes, imports and MKL dependencies; preserve NumPy/SciPy as separate numerical backends.
- [x] Reconstruct the reference driver's timing boundary and compare monolithic/staged AD under the same installed MKL environment/allocation; distinguish first compilation from repeated execution.
- [x] In `pyscfad-perf-mp2`, audit upstream native DF-MP2 against fork JAX kernels. Establish numerical/dispatch regressions before porting a compatible fast path; preserve t2, frozen-index, nonoptimized and higher-order behavior using explicit fallbacks.
- [x] In `pyscfad-perf-scf`, measure existing preconditioner enablement and source differences. Change only the response bottleneck demonstrated by profiles; verify full molecular/DF cotangents and convergence.
- [x] Build an integrated fork MKL candidate with the fork environment's own Python/nanobind and embedded runtime paths. Run MP2, SCF/DF, STC, DLNO and LNO regressions plus independent finite differences.
- [x] Compare upstream, original fork and optimized fork sequentially on one allocation, recording all phase times, native identities, energies and complete gradients. Repeat comparable timing where variance affects a decision.
- [x] Apply successful changes and matching tested native artifacts to the user's fork, keeping rollback copies and custom features. Verify installed imports/linkage and update benchmark scripts/documentation.

Review focus: unsupported Hessians/higher AD orders; arbitrary frozen index lists; with_t2 semantics; fallback when optimized integrals are disabled; out-of-core/memory behavior and custom STC/DLNO callers.
