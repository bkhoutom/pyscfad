"""Host memory observations and advisory STC forecasts (decimal MB).

Only a verified cgroup memory-plus-swap capacity can prove that a required new
array cannot fit. RAM-only limits, usage/free space and scheduler allocations
are planning signals.
No MPI calls, native allocator settings or numerical controls live here.
"""
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import os
from pathlib import Path
import re
import warnings


@dataclass(frozen=True)
class MemoryObservation:
    rss_mb: float | None = None
    enforced_limit_mb: float | None = None
    enforced_source: str = 'unknown'
    headroom_mb: float | None = None
    headroom_source: str = 'unknown'
    host_available_mb: float | None = None
    allowance_mb: float | None = None
    allowance_source: str = 'unknown'
    rank: int = 0


_calculation = ContextVar('stc_memory_calculation', default=None)


def memory_calculation(function):
    """Give nested checks one warning set, reset on every public calculation."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        if _calculation.get() is not None:
            return function(*args, **kwargs)
        comm = kwargs.get('comm')
        rank = None if comm is None else comm.Get_rank()
        token = _calculation.set((set(), rank))
        try:
            return function(*args, **kwargs)
        finally:
            _calculation.reset(token)
    return wrapped


def _read(path):
    try:
        return path.read_text()
    except (OSError, UnicodeError):
        return ''


def _count(value):
    try:
        result = int(value)
        return result if result >= 0 else None
    except (TypeError, ValueError):
        return None


def _kb_field(text, name):
    match = re.search(rf'^{name}:\s*(\d+)\s+kB\s*$', text, re.MULTILINE)
    return int(match[1]) * 1024 / 1e6 if match else None


def _unescape_mount(value):
    return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), value)


def _cgroup_observations(proc_root):
    """Yield RAM limits, memory+swap capacity, headroom and source paths.

    mountinfo's root handles sub-tree mounts; paths are never guessed from a
    Slurm job ID. Missing files, unlimited sentinels and inaccessible namespace
    paths cannot supply a verified limit.
    """
    groups = []
    for row in _read(proc_root / 'self/cgroup').splitlines():
        fields = row.split(':', 2)
        if len(fields) != 3:
            continue
        _, controllers, group = fields
        if not controllers or 'memory' in controllers.split(','):
            groups.append((not controllers, group))
    for row in _read(proc_root / 'self/mountinfo').splitlines():
        before, sep, after = row.partition(' - ')
        left, right = before.split(), after.split()
        if not sep or len(left) < 5 or len(right) < 3:
            continue
        v2 = right[0] == 'cgroup2'
        if not v2 and (right[0] != 'cgroup' or 'memory' not in right[2].split(',')):
            continue
        root, mount = Path(_unescape_mount(left[3])), Path(_unescape_mount(left[4]))
        for group_v2, raw_group in groups:
            group = Path(raw_group)
            if v2 != group_v2 or not group.is_absolute() or '..' in group.parts:
                continue
            try:
                relative = group.relative_to(root)
            except ValueError:
                # A namespace-relative / cannot prove membership in an
                # arbitrary bind-mounted subtree. An ambiguous map is unknown.
                continue
            current = mount / relative
            leaf = current
            limit_file = 'memory.max' if v2 else 'memory.limit_in_bytes'
            usage_file = 'memory.current' if v2 else 'memory.usage_in_bytes'
            while True:
                hierarchical = (v2 or current == leaf or
                                _read(current / 'memory.use_hierarchy').strip() == '1')
                limit = _count(_read(current / limit_file).strip()) if hierarchical else None
                # v1 reports PAGE_COUNTER_MAX rounded to a page when unlimited.
                if limit is not None and (v2 or limit < 2**60):
                    usage = _count(_read(current / usage_file).strip())
                    headroom = None if usage is None else max(0, limit - usage) / 1e6
                    if v2:
                        swap = _count(_read(current / 'memory.swap.max').strip())
                        capacity = None if swap is None else limit + swap
                    else:
                        capacity = _count(_read(current / 'memory.memsw.limit_in_bytes').strip())
                        if capacity is not None and capacity >= 2**60:
                            capacity = None
                    yield (limit / 1e6, None if capacity is None else capacity / 1e6,
                           headroom, f'cgroup v{2 if v2 else 1} {current}')
                if current == mount:
                    break
                current = current.parent


def _local_ranks(env):
    sizes = []
    for name in ('OMPI_COMM_WORLD_LOCAL_SIZE', 'MPI_LOCALNRANKS', 'MV2_COMM_WORLD_LOCAL_SIZE'):
        value = _count(env.get(name))
        if value:
            sizes.append(value)
    # Slurm expands homogeneous or heterogeneous node counts, e.g. 4(x2),2.
    tasks = env.get('SLURM_STEP_TASKS_PER_NODE', env.get('SLURM_TASKS_PER_NODE',
                    env.get('SLURM_NTASKS_PER_NODE', '')))
    node = _count(env.get('SLURM_NODEID'))
    entries = []
    for part in tasks.split(','):
        match = re.fullmatch(r'(\d+)(?:\(x(\d+)\))?', part.strip())
        if not match or int(match[1]) == 0:
            entries = []
            break
        entries.append((int(match[1]), int(match[2] or 1)))
    if entries:
        if len({value for value, _ in entries}) == 1:
            sizes.append(entries[0][0])
        elif node is not None:
            for value, count in entries:
                if node < count:
                    sizes.append(value)
                    break
                node -= count
    return sizes[0] if sizes and len(set(sizes)) == 1 else None


def observe_memory(*, proc_root=Path('/proc'), environ=None):
    """Observe the current process; unreadable values remain unknown."""
    proc_root = Path(proc_root)
    env = os.environ if environ is None else environ
    rss = _kb_field(_read(proc_root / 'self/status'), 'VmRSS')
    meminfo = _read(proc_root / 'meminfo')
    limits = list(_cgroup_observations(proc_root))
    capacities = [(capacity, source + ' memory+swap') for _, capacity, _, source in limits
                  if capacity is not None]
    enforced = min(capacities) if capacities else (None, 'unknown')
    headrooms = [(value, source) for _, _, value, source in limits if value is not None]
    headroom = min(headrooms) if headrooms else (None, 'unknown')
    local = _local_ranks(env)
    planning = [(limit, source) for limit, _, _, source in limits]
    node_mb = _count(env.get('SLURM_MEM_PER_NODE'))
    if not node_mb:
        per_cpu, cpus = _count(env.get('SLURM_MEM_PER_CPU')), _count(env.get('SLURM_CPUS_ON_NODE'))
        node_mb = per_cpu * cpus if per_cpu and cpus else None
    if node_mb and local:
        # Slurm's M units are MiB; convert to the decimal MB used by PySCF.
        planning.append((node_mb * 1024**2 / 1e6 / local,
                         f'Slurm allocation / {local} local ranks'))
    # Host RAM is advisory; in an uncertain shared-node mapping it is not a
    # per-rank allowance. Never use global communicator size as a divisor.
    total = _kb_field(meminfo, 'MemTotal')
    parallel = any(name in env for name in ('SLURM_JOB_ID', 'SLURM_MEM_PER_NODE',
                   'SLURM_MEM_PER_CPU', 'OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'PMIX_RANK'))
    if total is not None and (local or not parallel):
        planning.append((total / (local or 1), f'host MemTotal / {local or 1} local ranks'))
    allowance = min(planning, key=lambda item: item[0]) if planning else (None, 'unknown')
    rank = next((_count(env[name]) for name in ('OMPI_COMM_WORLD_RANK', 'PMI_RANK',
                 'PMIX_RANK', 'MV2_COMM_WORLD_RANK', 'SLURM_PROCID')
                 if _count(env.get(name)) is not None), 0)
    return MemoryObservation(rss_mb=rss, enforced_limit_mb=enforced[0], enforced_source=enforced[1],
        headroom_mb=headroom[0], headroom_source=headroom[1],
        host_available_mb=_kb_field(meminfo, 'MemAvailable'),
        allowance_mb=allowance[0], allowance_source=allowance[1], rank=rank)


def assess_memory(stage, estimate_mb, *, configured_mb=None, estimate_kind='workspace',
                  unavoidable_bytes=0, warned=None, observation=None,
                  budget_estimate_mb=None):
    """Warn on forecasts; raise only for a proven impossible new working set.

    ``workspace`` excludes existing residency; ``peak`` already includes it.
    The caller must justify ``unavoidable_bytes`` from live, non-overlapping
    array storage, never from a safety multiplier or a cache upper bound.
    ``budget_estimate_mb`` preserves an estimator's configured-budget comparison
    if its existing-residency component was removed from the workspace forecast.
    """
    info = observe_memory() if observation is None else observation
    if estimate_kind not in ('workspace', 'peak'):
        raise ValueError('estimate_kind must be workspace or peak')
    context = _calculation.get()
    rank = context[1] if context is not None and context[1] is not None else info.rank
    def mb(value):
        return 'unknown' if value is None else f'{value:.1f} MB'
    if info.enforced_limit_mb is not None and unavoidable_bytes / 1e6 > info.enforced_limit_mb:
        raise MemoryError(f'STC memory stage={stage} rank={rank}: unavoidable new storage '
                          f'{unavoidable_bytes} bytes exceeds verified enforced '
                          f'limit={mb(info.enforced_limit_mb)} ({info.enforced_source})')
    peak = (estimate_mb if estimate_kind == 'peak' else
            None if info.rss_mb is None else estimate_mb + info.rss_mb)
    allowance = info.allowance_mb
    budget_estimate = estimate_mb if budget_estimate_mb is None else budget_estimate_mb
    over_budget = configured_mb is not None and budget_estimate > configured_mb
    # Known new workspace alone can exceed an advisory allowance even when
    # missing RSS prevents reporting a total process peak.
    over_allowance = allowance is not None and (estimate_mb if peak is None else peak) > .9 * allowance
    if over_budget or over_allowance:
        seen = context[0] if context is not None else warned
        key = (stage, rank)
        if seen is None or key not in seen:
            if seen is not None:
                seen.add(key)
            warnings.warn(f'STC memory stage={stage} rank={rank}: '
                f'estimate={mb(estimate_mb)} ({estimate_kind}), peak={mb(peak)}, '
                f'RSS={mb(info.rss_mb)}, configured={mb(configured_mb)}, '
                f'budget forecast={mb(budget_estimate)}, '
                f'limit={mb(info.enforced_limit_mb)} ({info.enforced_source}), '
                f'headroom={mb(info.headroom_mb)} ({info.headroom_source}), '
                f'MemAvailable={mb(info.host_available_mb)}, '
                f'planning allowance={mb(allowance)} ({info.allowance_source}); continuing',
                RuntimeWarning, stacklevel=2)
    return info
