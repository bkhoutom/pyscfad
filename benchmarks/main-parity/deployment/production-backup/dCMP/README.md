# Xing dCMP runs on Caltech HPC

Copy this directory to the cluster, then submit **from inside it**:

```bash
sbatch submit_pyscf.sh
sbatch submit_pyscfad.sh
```

Each job requests one node, one task, **16 physical CPU cores**, **128 GB RAM**,
and two hours. The RAM request provides headroom; the archived allocation and
peak memory were not recorded. Both drivers retain `max_memory=60000` MB,
which is an advisory application budget, not a hard limit.

The Python path and OpenMPI module come from `submit_ref.sh`. The module is
retained for native-library compatibility; these are single-process jobs.
To select a different installed Python stack:

```bash
PYTHON_BIN=/path/to/env/bin/python sbatch submit_pyscfad.sh
```

The drivers retain the reference calculation settings and add explicit stage
timers. Each embeds the original 34-atom
geometry: cc-pVTZ / cc-pVTZ-JKFIT, 24 frozen orbitals, SCF tolerances
`1e-10` / `1e-8`, and 250 maximum cycles. PySCF prints energy and total seconds;
PySCF-AD computes the total energy gradient and prints its first ten rows plus
total seconds. Expected SCF energy: `-1377.51494675713` Eh; MP2 total energy:
`-1381.31940656203` Eh.

Outputs are `pyscf_scf.out.16.JOBID` and
`reproduce_dcmp_ad.out.16.3.JOBID`; stderr is in the corresponding `.err` files.
Scratch is retained under `/resnick/scratch/$USER/xing/JOBID`.

Both drivers print `SCF forward (including DF build)` and
`MP2 forward (excluding SCF)`. The AD driver also prints `MP2 pullback` and
`SCF pullback (including DF pullback)`, with synchronization at each boundary.
SCF pullback includes the complete HF and MP2 contributions through the shared
DF construction. Add the forward times for energy time, and the pullback times
for gradient time. The initial molecule build is excluded; the final bare number
remains total calculation wall seconds. The SCF forward in AD includes VJP
preparation. These timers do not depend on the installed library's logging.

The archives reported PySCF 2.12.1; their actual PySCF-AD/JAX versions were not
recorded, so matching scripts and resources does not guarantee identical
diagnostic lines or timings. The submission layout follows the supplied example
and [Caltech's OpenMP job guidance](https://www.hpc.caltech.edu/docs/documentation/example-jobs.html#multi-threaded-openmp).
