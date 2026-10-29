"""Set the benchmark CPU budget before importing numerical runtimes."""

import os
from pathlib import Path


def choose_cpus(allowed, count):
    """Prefer distinct physical cores on one socket within the inherited mask."""
    allowed = sorted(allowed)
    if len(allowed) < count:
        raise RuntimeError(f"benchmark requires {count} CPUs, only {len(allowed)} allowed")
    sockets = {}
    seen = set()
    for cpu in allowed:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            socket = int((topology / "physical_package_id").read_text())
            core = int((topology / "core_id").read_text())
        except (OSError, ValueError):
            socket, core = 0, cpu
        if (socket, core) not in seen:
            seen.add((socket, core))
            sockets.setdefault(socket, []).append(cpu)
    for cores in sockets.values():
        if len(cores) >= count:
            return cores[:count]
    physical = [cpu for cores in sockets.values() for cpu in cores]
    return (physical + [cpu for cpu in allowed if cpu not in physical])[:count]


def configure_cpu_workload(count):
    """All subsequently created threads/children inherit this CPU affinity."""
    if count < 1:
        raise ValueError("threads must be positive")
    allowed = set(os.sched_getaffinity(0))
    requested = os.environ.get("PYSCFAD_BENCHMARK_CPU_LIST")
    cpus = ([int(value) for value in requested.split(",")]
            if requested else choose_cpus(allowed, count))
    if len(set(cpus)) != count or not set(cpus) <= allowed:
        raise RuntimeError("benchmark CPU list must contain exactly the requested allowed CPUs")
    os.sched_setaffinity(0, cpus)
    os.environ["PYSCFAD_BENCHMARK_CPU_LIST"] = ",".join(map(str, cpus))
    for name in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "PYSCFAD_DENSE_BLAS_THREADS",
                 "TF_NUM_INTRAOP_THREADS"):
        os.environ[name] = str(count)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
    # Retain the inherited mask; do not let an OpenMP placement policy widen it
    # or bind the Python main thread to a single CPU before JAX initializes.
    os.environ["OMP_PROC_BIND"] = "FALSE"
    os.environ["KMP_AFFINITY"] = "disabled"
    for name in ("GOMP_CPU_AFFINITY", "OMP_PLACES", "KMP_HW_SUBSET"):
        os.environ.pop(name, None)
    return sorted(cpus)


def check_worker_affinity(cpus):
    """Verify every live thread stays inside the budget, including JAX/LLVM."""
    allowed = set(cpus)
    masks = []
    for task in Path("/proc/self/task").iterdir():
        try:
            mask = set(os.sched_getaffinity(int(task.name)))
        except ProcessLookupError:
            continue
        if not mask <= allowed:
            raise RuntimeError(f"thread {task.name} escaped benchmark CPUs: {sorted(mask - allowed)}")
        masks.append(mask)
    return dict(cpu_affinity=sorted(allowed), runtime_threads=len(masks),
                worker_affinity_verified=True,
                memory_mappings=len(Path("/proc/self/maps").read_text().splitlines()),
                max_map_count=int(Path("/proc/sys/vm/max_map_count").read_text()))
