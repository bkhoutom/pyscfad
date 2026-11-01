# Copyright 2023-2026 The PySCFAD Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LNO fragment diagnostics and optional forward/reverse profiling."""

import time
from contextlib import contextmanager
from contextvars import ContextVar
import numpy
import jax
from pyscfad.tools import resource_profile

_FRAGMENT_PROFILE_MPI_KEYS = (
    'label',
    'pass',
    'fragment',
    'nfrag',
    'n_lo',
    'active_occ',
    'active_vir',
    'lov_mb',
    't2_mb',
    'est_work_mb',
    'make_fpno1_s',
    'impurity_solve_s',
    'wall_s',
    'replay_wall_s',
    'pullback_wall_s',
)

_FRAGMENT_PROFILE_ROWS = []
_FRAGMENT_TABLE_HEADERS = set()
_VJP_PROGRESS_PREFIX = ContextVar('pyscfad_lno_vjp_progress_prefix', default=None)


@contextmanager
def vjp_progress(prefix):
    token = _VJP_PROGRESS_PREFIX.set(prefix)
    try:
        yield
    finally:
        _VJP_PROGRESS_PREFIX.reset(token)


def _vjp_progress(msg):
    prefix = _VJP_PROGRESS_PREFIX.get()
    if prefix is not None:
        print(f'{prefix} {msg}', flush=True)


@contextmanager
def _vjp_progress_section(name):
    prefix = _VJP_PROGRESS_PREFIX.get()
    profile_start = resource_profile.start()
    if prefix is None:
        try:
            yield
        finally:
            resource_profile.finish(
                f'backward.{name}',
                profile_start,
            )
        return
    t0 = time.perf_counter()
    _vjp_progress(f'{name}: start')
    try:
        yield
    finally:
        _vjp_progress(f'{name}: {time.perf_counter() - t0:.2f} s')
        resource_profile.finish(
            f'backward.{name}',
            profile_start,
        )


def clear_fragment_profile():
    _FRAGMENT_PROFILE_ROWS.clear()
    _FRAGMENT_TABLE_HEADERS.clear()


def get_fragment_profile(label=None):
    rows = list(_FRAGMENT_PROFILE_ROWS)
    if label is not None:
        rows = [row for row in rows if row.get('label') == label]
    return rows


def _profile_sort_key(row):
    pass_order = {'forward': 0, 'backward replay': 1}
    return (
        str(row.get('label', '')),
        pass_order.get(str(row.get('pass', '')), 99),
        int(row.get('fragment', -1)),
    )


def remap_fragment_profile_row(row, indices, nfrag):
    row = dict(row)
    if 'fragment' in row:
        ifrag = int(row['fragment'])
        if 0 <= ifrag < len(indices):
            row['fragment'] = int(indices[ifrag])
    row['nfrag'] = int(nfrag)
    phase_times = row.get('phase_times')
    if phase_times is not None:
        row['phase_times'] = dict(phase_times)
    return row


def _as_profile_scalar(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (int, numpy.integer)):
        return int(value)
    if isinstance(value, (float, numpy.floating)):
        return float(value)
    try:
        arr = numpy.asarray(jax.device_get(value))
    except Exception:
        return None
    if arr.shape == ():
        if numpy.issubdtype(arr.dtype, numpy.integer):
            return int(arr)
        if numpy.issubdtype(arr.dtype, numpy.floating):
            return float(arr)
    return None


def sanitize_fragment_profile_row_for_mpi(row):
    out = {}
    for key in _FRAGMENT_PROFILE_MPI_KEYS:
        if key not in row:
            continue
        value = _as_profile_scalar(row[key])
        if value is not None:
            out[key] = value
    phase_times = row.get('phase_times') or {}
    phase_out = {}
    for key, value in phase_times.items():
        value = _as_profile_scalar(value)
        if value is not None:
            phase_out[key] = value
    if phase_out:
        out['phase_times'] = phase_out
    return out


def print_fragment_profile_rows(rows):
    for row in rows:
        _print_fragment_report(row)


def _append_fragment_profile(row):
    _FRAGMENT_PROFILE_ROWS.append(row)


def _verbose_at_least(obj, level=2):
    try:
        return int(getattr(obj, 'verbose', 0)) >= level
    except (TypeError, ValueError):
        return False


def _fragment_report_enabled(mfcc):
    return bool(getattr(mfcc, 'profile_fragments', False))


def _fragment_profile_label(mfcc):
    return str(getattr(mfcc, 'profile_label', mfcc.__class__.__name__))


def _fragment_profile_pass(mfcc):
    return str(getattr(mfcc, 'profile_pass', 'forward'))


def _phase_time(row, name):
    value = row.get(name, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _print_fragment_table_header(row):
    has_ad_times = (
        row.get('replay_wall_s') is not None
        or row.get('pullback_wall_s') is not None
    )
    key = (
        row.get('label', 'LNO'),
        row.get('pass', 'forward'),
        'ad' if has_ad_times else 'forward',
    )
    if key in _FRAGMENT_TABLE_HEADERS:
        return
    _FRAGMENT_TABLE_HEADERS.add(key)
    extra_header = (
        f" {'replay':>8} {'pullback':>8}" if has_ad_times else ""
    )
    extra_units = (
        f" {'sec':>8} {'sec':>8}" if has_ad_times else ""
    )
    print(
        "  "
        f"{'frag':>4} {'LNO':>10} "
        f"{'setup':>8} {'solve':>8} {'total':>8} {'memory':>8}"
        f"{extra_header}",
        flush=True,
    )
    print(
        "  "
        f"{'':>4} {'occ/vir':>10} "
        f"{'sec':>8} {'sec':>8} {'sec':>8} {'MB':>8}"
        f"{extra_units}",
        flush=True,
    )


def _print_fragment_report(row):
    _print_fragment_table_header(row)
    idx = int(row.get('fragment', -1)) + 1
    setup_s = _phase_time(row, 'make_fpno1_s')
    impurity_s = _phase_time(row, 'impurity_solve_s')
    total_s = _phase_time(row, 'wall_s')

    phase_times = row.get('phase_times') or {}
    if phase_times:
        solver_s = _phase_time(phase_times, 'total_s')
    else:
        solver_s = impurity_s

    active = f"{row.get('active_occ', 0)}/{row.get('active_vir', 0)}"
    print(
        "  "
        f"{idx:4d} {active:>10} "
        f"{setup_s:8.3f} {solver_s:8.3f} {total_s:8.3f} "
        f"{row.get('est_work_mb', 0.0):8.1f}"
        + (
            f" {_phase_time(row, 'replay_wall_s'):8.3f}"
            f" {_phase_time(row, 'pullback_wall_s'):8.3f}"
            if (
                row.get('replay_wall_s') is not None
                or row.get('pullback_wall_s') is not None
            )
            else ""
        ),
        flush=True,
    )


def _append_and_maybe_print_fragment_profile(mfcc, row):
    if _fragment_report_enabled(mfcc):
        _append_fragment_profile(row)
        if (
            _verbose_at_least(mfcc, 2)
            and bool(getattr(mfcc, 'profile_print', True))
        ):
            if row.get('pass') == 'backward replay':
                return
            _print_fragment_report(row)


def _print_fragment_start(mfcc, orbfragloc):
    if not (_fragment_report_enabled(mfcc) and _verbose_at_least(mfcc, 3)):
        return
    idx = int(getattr(mfcc, '_current_ifrag', -1)) + 1
    nfrag = getattr(mfcc, '_nfrag', None)
    frag_label = f'{idx}/{int(nfrag)}' if nfrag is not None else str(idx)
    print(
        f"  {_fragment_profile_label(mfcc)} {_fragment_profile_pass(mfcc)} "
        f"fragment {frag_label}: starting (LOs={orbfragloc.shape[1]})",
        flush=True,
    )


def _fragment_diagnostic_info(mfcc, eris, orbfragloc, frzfrag, orbfrag):
    from .lno_base import THRESH_OCC

    mf = mfcc._scf
    mo_occ = mf.mo_occ
    nocc = int(numpy.count_nonzero(mo_occ > THRESH_OCC))
    nmo = int(mo_occ.size)
    nvir = nmo - nocc

    if frzfrag is None or orbfrag is None:
        active_occ = 0
        active_vir = 0
    else:
        frzfrag_arr = numpy.asarray(frzfrag, dtype=numpy.int64).ravel()
        frozen_occ = int(numpy.count_nonzero(frzfrag_arr < nocc))
        frozen_vir = int(numpy.count_nonzero(frzfrag_arr >= nocc))
        active_occ = nocc - frozen_occ
        active_vir = nvir - frozen_vir

    lov = getattr(eris, 'Lov', None)
    lov_mb = 0.0
    if lov is not None:
        lov_size = int(numpy.prod(tuple(int(x) for x in lov.shape)))
        lov_mb = lov_size * numpy.dtype(numpy.float64).itemsize / 1e6

    t2_size = active_occ * active_occ * active_vir * active_vir
    t2_mb = t2_size * numpy.dtype(numpy.float64).itemsize / 1e6

    return {
        'fragment': int(getattr(mfcc, '_current_ifrag', -1)),
        'n_lo': int(orbfragloc.shape[1]),
        'active_occ': active_occ,
        'active_vir': active_vir,
        'lov_mb': lov_mb,
        't2_mb': t2_mb,
        'est_work_mb': lov_mb + 4.0 * t2_mb,
    }


class FragmentProfile:
    """Collect one LNO fragment's dimensions and stage times outside its science."""

    def __init__(self, mfcc, eris, orbfragloc):
        self.mfcc = mfcc
        self.eris = eris
        self.orbfragloc = orbfragloc
        self.started = time.perf_counter()
        self.info = {'impurity_solve_s': 0.0}
        _print_fragment_start(mfcc, orbfragloc)

    @contextmanager
    def stage(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.info[name] = time.perf_counter() - started

    def describe(self, frozen, orbitals):
        self.info.update(_fragment_diagnostic_info(
            self.mfcc, self.eris, self.orbfragloc, frozen, orbitals,
        ))
        self.info.update(label=_fragment_profile_label(self.mfcc),
                         **{'pass': _fragment_profile_pass(self.mfcc)},
                         nfrag=getattr(self.mfcc, '_nfrag', None))

    def finish(self):
        self.info['wall_s'] = time.perf_counter() - self.started
        _append_and_maybe_print_fragment_profile(self.mfcc, self.info)


def report_mp2_density(started, Lia, Ljb, eia, ejb, dmvv, dmoo):
    naux, nocc_frag, nvir = Lia.shape
    nocc_domain = Ljb.shape[1]
    itemsize = int(Lia.dtype.itemsize)
    resource_profile.finish(
        'pno.mp2_density_matrices', started,
        lia_shape=tuple(Lia.shape), ljb_shape=tuple(Ljb.shape),
        dmvv_shape=tuple(dmvv.shape), dmoo_shape=tuple(dmoo.shape),
        inputs_mib=resource_profile.estimated_array_mib(Lia, Ljb, eia, ejb),
        outputs_mib=resource_profile.estimated_array_mib(dmvv, dmoo),
        scan_t2_temp_mib=nvir * nocc_domain * nvir * itemsize / 1024.0**2,
        uncheckpointed_t2_mib=(
            nocc_frag * nvir * nocc_domain * nvir * itemsize / 1024.0**2
        ),
        naux=naux,
    )


class NaturalOrbitalProfile:
    """Report density storage and retained dimensions without retaining arrays."""

    def __init__(self, occupied_density, virtual_density):
        self.started = resource_profile.start()
        self.occupied_shape = tuple(occupied_density.shape)
        self.virtual_shape = tuple(virtual_density.shape)
        self.density_mib = resource_profile.estimated_array_mib(
            occupied_density, virtual_density,
        )

    def finish(self, nocc, nvir):
        resource_profile.finish(
            'pno.natural_orbital_compression', self.started,
            dmoo_shape=self.occupied_shape, dmvv_shape=self.virtual_shape,
            retained_occ=int(nocc), retained_vir=int(nvir),
            density_matrices_mib=self.density_mib,
        )
