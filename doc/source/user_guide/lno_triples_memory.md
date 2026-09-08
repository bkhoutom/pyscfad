# Memory for LNO triples

The factor-direct LNO/DLNO perturbative triples `(T)` calculation selects its
virtual block width separately for each fragment and forward/backward pass.
The default cache-budget mode is automatic. It can also be selected explicitly:

```bash
export PYSCFAD_LNO_CCSD_T_FACTOR_BLOCK_MB=auto
```

Set this before starting Python/MPI, after activating the calculation environment.
A positive numeric value requests a cache budget in MiB (1024**2 bytes):

```bash
export PYSCFAD_LNO_CCSD_T_FACTOR_BLOCK_MB=8192
```

The numeric request still obeys the process and allocation limits below. Invalid,
nonfinite or nonpositive values raise an error. Older versions used 128 MiB when
the variable was unset and did not recognize `auto`.

## How automatic sizing works

The selector reads the number of local MPI ranks from
`OMPI_COMM_WORLD_LOCAL_SIZE`, then falls back to Slurm's per-node task counts.
Slurm lists such as `4(x2),2` are resolved using `SLURM_NODEID`. If local counts
are unavailable, the total job rank count is a conservative fallback. The
selector does not communicate with other ranks or initialize MPI.

Node memory comes from `SLURM_MEM_PER_NODE`, or from `SLURM_MEM_PER_CPU` multiplied
by a conservative usable CPU count on the node: local ranks times
`SLURM_CPUS_PER_TASK` (one if unspecified), capped by allocated CPUs. This avoids
counting unusable hyperthreads toward memory, but can underuse an allocation
that contains extra CPUs. An explicit per-node memory request gives a clearer
budget for whole-node jobs. Slurm's explicit `--mem=0` request uses the
node's physical memory. A node allocation is divided among local ranks; memory
from different nodes is never pooled.

For known allocations the sizing rule, in bytes, is:

```text
rank_ceiling = min(configured_max_memory, 0.8 * allocated_node_memory / local_ranks)
headroom = max(0, rank_ceiling - current_process_RSS)
cache_budget = min(0.5 * headroom, estimated_full_width_cache)
```

The 20% allocation reserve and half-headroom rule leave room for other arrays,
native workspaces and runtime overhead. The chosen width accounts for fragment
occupied/virtual dimensions and thread-private backward buffers. For real
float64 backward calculations the existing conservative estimate is:

```text
cache_bytes(width) = 4 * (threads + 1) * 8 * width * nvir * nocc * (nocc + nvir)
```

Forward uses four copies, without the backward thread multiplier. The selector
caps the width at `nvir`, and retains the minimum width of one when headroom is
insufficient. These estimates size the factor caches; they are not a hard bound
on total process RSS.

Fewer ranks per node can therefore receive larger caches. Increasing threads
per rank also increases the backward buffer estimate. The configured PySCF
`max_memory` remains a per-process ceiling for this calculation: leaving it at
26000 MB will still constrain a rank even if its node has more memory available.
PySCF uses decimal MB (1000**2 bytes), unlike the cache option's MiB.

If allocation metadata is missing or unusable, the selector uses half the
remaining configured per-process `max_memory`. It does not assume that all host
RAM belongs to the job. This fallback does not discover other schedulers or
additional container/cgroup limits; configure `max_memory` within the actual
allocation in those environments.

## Checking the decision

Enable either existing profiler:

```bash
export PYSCFAD_DLNO_RESOURCE_PROFILE=1
# Or, for triples-specific output:
export PYSCFAD_LNO_CCSD_T_PROFILE=1
```

The selection record reports automatic/manual mode, ranks per node, memory
source, allocated memory per rank, remaining memory, cache budget and virtual
width. Resource profiling labels it `phase=triples.factor_block`. Check these
fields in a new allocation before interpreting performance differences.

The numerical triples kernels are unchanged. Blocking can change floating-point
summation order, so comparisons should use appropriate numerical tolerances.

Environment semantics: [Open MPI local ranks](https://docs.open-mpi.org/en/v5.0.8/tuning-apps/environment-var.html)
and [Slurm allocation variables](https://slurm.schedmd.com/sbatch.html).
