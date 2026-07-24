"""Lightweight process resource diagnostics for DLNO profiling.

The profiler is deliberately calculation-neutral: it only samples wall time,
process CPU time, and operating-system memory counters.  It never converts JAX
arrays, synchronizes devices, or changes solver state.

Enable it with ``PYSCFAD_DLNO_RESOURCE_PROFILE=1``.
"""

from contextlib import contextmanager
import ctypes
import os
import resource
import sys
import threading
import time
from typing import NamedTuple, Optional


_ENV_NAME = 'PYSCFAD_DLNO_RESOURCE_PROFILE'
_PREFIX = '[DLNO-RESOURCE]'
_HEADER_PRINTED = False
_DARWIN_LIBC = None


class Snapshot(NamedTuple):
    wall_s: float
    user_s: float
    system_s: float
    rss_mib: Optional[float]
    peak_rss_mib: Optional[float]


class ProfileStart(NamedTuple):
    wall_s: float
    user_s: float
    system_s: float
    rss_mib: Optional[float]
    peak_rss_mib: Optional[float]
    sampler: Optional[object]


def enabled():
    value = os.environ.get(_ENV_NAME)
    return (
        value is not None
        and value.strip().lower() in ('1', 'true', 'yes', 'on')
    )


def _mpi_rank():
    for name in (
        'OMPI_COMM_WORLD_RANK',
        'PMI_RANK',
        'PMIX_RANK',
        'MV2_COMM_WORLD_RANK',
        'SLURM_PROCID',
    ):
        value = os.environ.get(name)
        if value is not None:
            return value
    return '0'


def _linux_rss_mib():
    try:
        with open('/proc/self/status', encoding='ascii') as status:
            for line in status:
                if line.startswith('VmRSS:'):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


class _TimeValue(ctypes.Structure):
    _fields_ = [
        ('seconds', ctypes.c_int),
        ('microseconds', ctypes.c_int),
    ]


class _MachTaskBasicInfo(ctypes.Structure):
    _fields_ = [
        ('virtual_size', ctypes.c_uint64),
        ('resident_size', ctypes.c_uint64),
        ('resident_size_max', ctypes.c_uint64),
        ('user_time', _TimeValue),
        ('system_time', _TimeValue),
        ('policy', ctypes.c_int),
        ('suspend_count', ctypes.c_int),
    ]


def _darwin_memory_mib():
    global _DARWIN_LIBC
    try:
        if _DARWIN_LIBC is None:
            libc = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
            libc.mach_task_self.restype = ctypes.c_uint32
            libc.task_info.argtypes = [
                ctypes.c_uint32,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint32),
            ]
            libc.task_info.restype = ctypes.c_int
            _DARWIN_LIBC = libc
        info = _MachTaskBasicInfo()
        count = ctypes.c_uint32(
            ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint32)
        )
        # MACH_TASK_BASIC_INFO = 20.
        result = _DARWIN_LIBC.task_info(
            _DARWIN_LIBC.mach_task_self(),
            20,
            ctypes.byref(info),
            ctypes.byref(count),
        )
        if result == 0:
            scale = 1024.0**2
            return info.resident_size / scale, info.resident_size_max / scale
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    return None, None


def _memory_mib():
    if sys.platform == 'darwin':
        return _darwin_memory_mib()

    rss_mib = _linux_rss_mib() if sys.platform.startswith('linux') else None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak = float(usage.ru_maxrss)
    if sys.platform.startswith('linux'):
        peak /= 1024.0
    else:
        peak /= 1024.0**2
    return rss_mib, peak


def snapshot():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_mib, peak_rss_mib = _memory_mib()
    return Snapshot(
        wall_s=time.perf_counter(),
        user_s=float(usage.ru_utime),
        system_s=float(usage.ru_stime),
        rss_mib=rss_mib,
        peak_rss_mib=peak_rss_mib,
    )


def _sample_interval_s():
    value = os.environ.get('PYSCFAD_DLNO_RESOURCE_SAMPLE_MS')
    if value is None:
        return 0.0
    try:
        return max(float(value), 0.0) / 1000.0
    except ValueError:
        return 0.0


class _RSSSampler:
    """Sample current RSS in the background without touching calculation data."""

    def __init__(self, interval_s, initial_rss_mib):
        self.interval_s = interval_s
        self.peak_rss_mib = initial_rss_mib
        self.sample_count = int(initial_rss_mib is not None)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name='dlno-resource-rss',
            daemon=True,
        )

    def start(self):
        self._thread.start()

    def _run(self):
        while not self._stop.wait(self.interval_s):
            rss_mib, _ = _memory_mib()
            if rss_mib is not None:
                self.sample_count += 1
                if self.peak_rss_mib is None or rss_mib > self.peak_rss_mib:
                    self.peak_rss_mib = rss_mib

    def stop(self, final_rss_mib):
        self._stop.set()
        self._thread.join()
        if final_rss_mib is not None:
            self.sample_count += 1
            if self.peak_rss_mib is None or final_rss_mib > self.peak_rss_mib:
                self.peak_rss_mib = final_rss_mib


def start():
    if not enabled():
        return None
    before = snapshot()
    sampler = None
    interval_s = _sample_interval_s()
    if interval_s > 0.0:
        sampler = _RSSSampler(interval_s, before.rss_mib)
        sampler.start()
    return ProfileStart(*before, sampler)


def estimated_array_mib(*arrays):
    """Return array storage implied by shapes/dtypes without materialization."""
    total = 0
    for array in arrays:
        if array is None:
            continue
        shape = getattr(array, 'shape', ())
        dtype = getattr(array, 'dtype', None)
        if dtype is None:
            continue
        size = 1
        for dimension in shape:
            size *= int(dimension)
        total += size * int(getattr(dtype, 'itemsize', 0))
    return total / 1024.0**2


def _format_detail(value):
    if isinstance(value, float):
        return f'{value:.3f}'
    if isinstance(value, (tuple, list)):
        return 'x'.join(str(item) for item in value)
    return str(value).replace(' ', '_')


def _print_header():
    global _HEADER_PRINTED
    if _HEADER_PRINTED:
        return
    _HEADER_PRINTED = True
    print(
        f'{_PREFIX} columns: rank phase wall_s cpu_s cpu_pct '
        '(100%=one fully busy core) host_cpu_pct rss_mib '
        'rss_delta_mib phase_peak_rss_mib phase_peak_delta_mib '
        'sample_count peak_rss_mib load1 details',
        flush=True,
    )


def finish(phase, before, **details):
    if before is None or not enabled():
        return
    after = snapshot()
    sampler = getattr(before, 'sampler', None)
    if sampler is not None:
        sampler.stop(after.rss_mib)
    wall_s = max(after.wall_s - before.wall_s, 0.0)
    cpu_s = max(
        (after.user_s - before.user_s)
        + (after.system_s - before.system_s),
        0.0,
    )
    cpu_pct = 100.0 * cpu_s / wall_s if wall_s > 0.0 else 0.0
    ncpu = max(os.cpu_count() or 1, 1)
    host_cpu_pct = cpu_pct / ncpu
    rss_delta = (
        after.rss_mib - before.rss_mib
        if after.rss_mib is not None and before.rss_mib is not None
        else None
    )
    phase_peak_rss = (
        sampler.peak_rss_mib if sampler is not None else None
    )
    phase_peak_delta = (
        phase_peak_rss - before.rss_mib
        if phase_peak_rss is not None and before.rss_mib is not None
        else None
    )
    try:
        load1 = os.getloadavg()[0]
    except (AttributeError, OSError):
        load1 = None

    _print_header()
    fields = [
        f'rank={_mpi_rank()}',
        f'phase={str(phase).replace(" ", "_")}',
        f'wall_s={wall_s:.3f}',
        f'cpu_s={cpu_s:.3f}',
        f'cpu_pct={cpu_pct:.1f}',
        f'host_cpu_pct={host_cpu_pct:.1f}',
        (
            f'rss_mib={after.rss_mib:.1f}'
            if after.rss_mib is not None else 'rss_mib=NA'
        ),
        (
            f'rss_delta_mib={rss_delta:+.1f}'
            if rss_delta is not None else 'rss_delta_mib=NA'
        ),
        (
            f'phase_peak_rss_mib={phase_peak_rss:.1f}'
            if phase_peak_rss is not None else 'phase_peak_rss_mib=NA'
        ),
        (
            f'phase_peak_delta_mib={phase_peak_delta:+.1f}'
            if phase_peak_delta is not None else 'phase_peak_delta_mib=NA'
        ),
        (
            f'sample_count={sampler.sample_count}'
            if sampler is not None else 'sample_count=NA'
        ),
        (
            f'peak_rss_mib={after.peak_rss_mib:.1f}'
            if after.peak_rss_mib is not None else 'peak_rss_mib=NA'
        ),
        f'load1={load1:.2f}' if load1 is not None else 'load1=NA',
    ]
    fields.extend(
        f'{key}={_format_detail(value)}'
        for key, value in details.items()
        if value is not None
    )
    print(f'{_PREFIX} ' + ' '.join(fields), flush=True)


def checkpoint(phase, **details):
    if not enabled():
        return
    now = snapshot()
    try:
        load1 = os.getloadavg()[0]
    except (AttributeError, OSError):
        load1 = None
    _print_header()
    fields = [
        f'rank={_mpi_rank()}',
        f'phase={str(phase).replace(" ", "_")}',
        'wall_s=NA',
        'cpu_s=NA',
        'cpu_pct=NA',
        'host_cpu_pct=NA',
        (
            f'rss_mib={now.rss_mib:.1f}'
            if now.rss_mib is not None else 'rss_mib=NA'
        ),
        'rss_delta_mib=NA',
        'phase_peak_rss_mib=NA',
        'phase_peak_delta_mib=NA',
        'sample_count=NA',
        (
            f'peak_rss_mib={now.peak_rss_mib:.1f}'
            if now.peak_rss_mib is not None else 'peak_rss_mib=NA'
        ),
        f'load1={load1:.2f}' if load1 is not None else 'load1=NA',
    ]
    fields.extend(
        f'{key}={_format_detail(value)}'
        for key, value in details.items()
        if value is not None
    )
    print(f'{_PREFIX} ' + ' '.join(fields), flush=True)


@contextmanager
def section(phase, **details):
    before = start()
    try:
        yield
    finally:
        finish(phase, before, **details)
