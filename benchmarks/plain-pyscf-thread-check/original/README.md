# dCMP runtime benchmark: custom fork

Submit from this directory:

```bash
sbatch submit_pyscfad.sh
```

`reproduce_dcmp_ad.py` contains the input and computes the full energy and
coordinate gradient. `submit_pyscfad.sh` uses
`/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env/bin/python`, one MPI task,
16 physical Intel cores, 128 GB RAM and a two-hour limit. No MPI module is
needed. Program output and SLURM errors remain in this directory; scratch uses
the explicit location in the script.

The molecule has 34 atoms, cc-pVTZ / cc-pVTZ-JKFIT, 24 frozen occupied orbitals,
SCF tolerances `1e-10` / `1e-8`, and a 60000 MB advisory memory budget.
Expected HF energy is -1377.51494675713 Eh and MP2 total is -1381.31940656203 Eh.
The full gradient is computed; the driver prints its first ten rows.

PySCF 2.14.0 and BLAS-using PySCFAD native libraries use LP64 sequential
MKL 2024.2.2. Native OpenMP uses the 16 allocated cores. NumPy and SciPy retain
their separate OpenBLAS libraries: NumPy/native BLAS stay at one thread, while
the driver selects SciPy16 only for DF forward/reverse and SciPy1 elsewhere.

Timing output separates DF, SCF and MP2 forward and reverse stages. SCF forward
with DF adds the two forward phases; SCF reverse with DF adds both pullbacks.
Compare MP2 forward plus pullback across branches: the fork's custom JVP
performs part of the derivative work during its forward stage. Forward timing
inside a VJP is not a standalone energy-only measurement. The final total
includes all six phases, with synchronization, but excludes initial imports
and molecule construction. JIT compilation during these stages is included.

The driver explicitly enables `pyscfad_scf_df_precond`. The submission script
sets `PYSCFAD_INTEGRAL_VJP_SCHEDULE=dynamic` for load balancing. The package's
global preconditioner and static scheduling defaults remain unchanged; dynamic
scheduling permits insignificant summation-order roundoff.

Paired timings, full-gradient checks and build details are recorded in
`/resnick/groups/sharmagroup/bkhou/src/pyscfad/benchmarks/main-parity/RESULTS.md`.
For the separate plain-PySCF energy comparison, use `submit_pyscf.sh`.
