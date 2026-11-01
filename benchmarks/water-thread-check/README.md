# Water benchmark preparation and verification

Requested jobs: `water_2`, `water_4`, `water_8`, `water_12`, and `water_16` under
`/home/bkhou/DLNO_CCSDT/test_pyscfad_stc_mp2/runtime_test/water_tests`.

Each condition contains its own original XYZ input, `compare_mp2_LT.py`,
and static `submit.sh`; submission is from inside the condition. Every job
requests one task, 32 Ice Lake cores without SMT, 128 GB, and two hours using
the working expansion/sharmagroup/normal settings.

The supplied driver uses the fork-only `dlno_stc` stochastic full-system
LT-MP2 method, `ccpvtz`, no frozen core, energy tolerance `3e-4`, and seed 71.
Preserve those scientific settings, automatic auxiliary basis, SCF defaults,
and 120000 MB advisory memory budget. Use the installed `pyscfad_env` and
its normal editable imports. The original source files in the nested
`water_tests/` directory remain untouched.

Remove the driver's hard-coded eight-thread OpenMP/BLAS override. Use
OpenMP 32 and BLAS 1, with only SciPy set to 32 during DF forward/reverse.
A local custom VJP delegates to the original DF operation and its exact
JAX pullback; it controls execution scope without dropping DF derivatives.
Enable the fork's DF response preconditioner as in the validated dCMP driver.
Keep its resource profiling and synchronize the energy/gradient timer.
Save the complete gradient and timing in a job-numbered JSON beside the log.

Validation before production submission:

- Parse every Python file, check shell syntax and every XYZ atom count.
- Compare original and adapted STC energy/full gradient on the water dimer
  with STO-3G and identical deterministic quadrature, inside the requested
  32-core Ice Lake setup. This avoids stochastic noise in the equivalence
  test. Verify both DF forward and reverse execute and BLAS limits restore.
- Run the requested water jobs only after this numerical comparison passes.
- Confirm SLURM accepted the exact resources and each condition's workdir.

The settings are a consistent benchmark starting point. They do not claim
that 32 threads is individually optimal for every cluster size. The requested
stochastic energy tolerance does not specify a gradient error bound.

Validation job **5788235** completed successfully on `hpc-19-22` in 25 seconds.
The energy difference was **5.6843e-14 Eh** and the maximum difference across
the complete nuclear gradient was **8.9113e-12 Eh/bohr**. Both DF forward and
reverse ran with SciPy 32 and restored the other-phase BLAS policy. See
`validation.json` and `slurm-5788235.out`. The existing PySCF serialization
warning about `coords` appeared in both paths and did not affect this check.

The five production jobs were submitted from their respective directories:
water_2 **5788370**, water_4 **5788371**, water_8 **5788372**, water_12 **5788373**,
and water_16 **5788374**. SLURM's accepted resources and workdirs were checked.
`preparation.json` contains the deployed file hashes; the water-tests parent
directory contains `submission.json` and rerun instructions in `README.md`.
