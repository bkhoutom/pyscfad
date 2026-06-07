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

import warnings
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import partial, reduce
import numpy
import jax
import jax.scipy.linalg as jsp_linalg
from pyscf import __config__
from pyscf.mp.mp2 import get_frozen_mask, get_nmo, get_nocc
from pyscf import gto as pyscf_gto

from pyscfad import numpy as np
from pyscfad import pytree
from pyscfad.ops import stop_trace, stop_grad
from pyscfad.ops import is_array
from pyscfad import scipy
from pyscfad import df as df_mod
from pyscfad.tools import timer
from pyscfad.ao2mo import _ao2mo
from pyscfad.df import addons as df_addons
from pyscfad.df import incore as df_incore
from pyscfad.df import _cderi_vjp
from pyscfad.dlno import util as dlno_util
from pyscfad.lno import _checkpointed
from pyscfad.lno.mp2_rdm import make_rdm1_vo, make_rdm1_vo_frag
from pyscfad.lno.tools import autofrag, map_lo_to_frag
from pyscfad.gto._mole_helper import setup_exp, setup_ctr_coeff

USE_CHECKPOINT = True
THRESH_INTERNAL = 1e-10
THRESH_OCC = 1e-6
# Weak DLNO virtual-overlap directions can be numerically unstable when mapped
# into the current external-virtual MO space.  Prune only these nearly-null
# directions; the physically relevant subspace is unaffected.
DLNO_VIR_MAP_THRESH = 1e-6
# Tuned for benzene;
# must be bigger than the energy difference
# between degenerate semicanonical orbitals
SEMICANONICAL_DEG_THRESH = 1e-8
# Anything not bigger than the NO occupation number gap should work
COMPRESS_DEG_THRESH = 1e-12
_FRAGMENT_PROFILE_ROWS = []
_FRAGMENT_TABLE_HEADERS = set()
_VJP_PROGRESS_PREFIX = ContextVar('pyscfad_lno_vjp_progress_prefix', default=None)
DOMAIN_MP2_USE_LT = getattr(__config__, 'lno_domain_mp2_use_lt', True)
DOMAIN_MP2_LT_NLAP = getattr(__config__, 'lno_domain_mp2_lt_nlap', 9)
DOMAIN_MP2_LT_QUADRATURE = getattr(__config__, 'lno_domain_mp2_lt_quadrature', 'fit')
DOMAIN_MP2_LT_FIT_RATIO = getattr(__config__, 'lno_domain_mp2_lt_fit_ratio', 64.0)


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
    if prefix is None:
        yield
        return
    t0 = time.perf_counter()
    _vjp_progress(f'{name}: start')
    try:
        yield
    finally:
        _vjp_progress(f'{name}: {time.perf_counter() - t0:.2f} s')


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


_FRAGMENT_PROFILE_MPI_KEYS = (
    'label',
    'pass',
    'fragment',
    'nfrag',
    'n_lo',
    'domain_atoms',
    'domain_aos',
    'strong_lmo',
    'prescreen_occ',
    'prescreen_vir',
    'active_occ',
    'active_vir',
    'lov_mb',
    't2_mb',
    'est_work_mb',
    'make_fragment_eris_s',
    'make_fpno1_s',
    'impurity_solve_s',
    'wall_s',
    'replay_wall_s',
    'pullback_wall_s',
)


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
    return (
        bool(getattr(mfcc, 'profile_fragments', False))
        or (
            _verbose_at_least(mfcc, 2)
            and bool(getattr(mfcc, 'use_dlno_prescreen', False))
        )
    )


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
        f"{'frag':>4} {'domain':>10} {'prescreen':>12} {'LNO':>10} "
        f"{'setup':>8} {'solve':>8} {'total':>8} {'memory':>8}"
        f"{extra_header}",
        flush=True,
    )
    print(
        "  "
        f"{'':>4} {'atoms/AO':>10} {'occ/vir':>12} {'occ/vir':>10} "
        f"{'sec':>8} {'sec':>8} {'sec':>8} {'MB':>8}"
        f"{extra_units}",
        flush=True,
    )


def _print_fragment_report(row):
    _print_fragment_table_header(row)
    idx = int(row.get('fragment', -1)) + 1
    make_eris_s = _phase_time(row, 'make_fragment_eris_s')
    fpno_s = _phase_time(row, 'make_fpno1_s')
    impurity_s = _phase_time(row, 'impurity_solve_s')
    setup_s = make_eris_s + fpno_s
    total_s = _phase_time(row, 'wall_s')

    phase_times = row.get('phase_times') or {}
    if phase_times:
        solver_s = _phase_time(phase_times, 'total_s')
    else:
        solver_s = impurity_s

    domain = f"{row.get('domain_atoms', 0)}/{row.get('domain_aos', 0)}"
    prescreen = f"{row.get('prescreen_occ', 0)}/{row.get('prescreen_vir', 0)}"
    active = f"{row.get('active_occ', 0)}/{row.get('active_vir', 0)}"
    print(
        "  "
        f"{idx:4d} {domain:>10} {prescreen:>12} {active:>10} "
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


def _print_fragment_start(mfcc, frag_prescreen, orbfragloc):
    if not (_fragment_report_enabled(mfcc) and _verbose_at_least(mfcc, 3)):
        return
    idx = int(getattr(mfcc, '_current_ifrag', -1)) + 1
    nfrag = getattr(mfcc, '_nfrag', None)
    frag_label = f'{idx}/{int(nfrag)}' if nfrag is not None else str(idx)
    domain_atoms = 0
    domain_aos = 0
    prescreen_occ = 0
    prescreen_vir = 0
    if frag_prescreen is not None:
        atmlst = frag_prescreen.get('extended_primary_domain', ())
        atmlst = numpy.asarray(atmlst, dtype=numpy.int32).ravel()
        domain_atoms = int(atmlst.size)
        if atmlst.size > 0:
            domain_aos = int(dlno_util.ao_index_by_atom(mfcc.mol, atmlst).size)
        occ_coeff = frag_prescreen.get('occ_prescreen_coeff')
        vir_coeff = frag_prescreen.get('vir_prescreen_coeff')
        prescreen_occ = 0 if occ_coeff is None else int(occ_coeff.shape[1])
        prescreen_vir = 0 if vir_coeff is None else int(vir_coeff.shape[1])
    print(
        f"  {_fragment_profile_label(mfcc)} {_fragment_profile_pass(mfcc)} "
        f"fragment {frag_label}: starting "
        f"(LOs={orbfragloc.shape[1]}, domain atoms/AOs={domain_atoms}/"
        f"{domain_aos}, prescreen occ/vir={prescreen_occ}/{prescreen_vir})",
        flush=True,
    )


def kernel(mfcc, orbloc, frag_lolist,
           no_type='ie', eris=None, frag_nonvlist=None):
    use_dlno_fragment_eris = _use_dlno_fragment_eris(mfcc)
    if eris is None:
        if use_dlno_fragment_eris:
            eris = _make_lno_reference_eris(mfcc)
        else:
            eris = mfcc.ao2mo()
    elif use_dlno_fragment_eris and getattr(eris, 'Lov', None) is not None:
        raise RuntimeError(
            'DLNO fragment-local mode was given ERIs with a prebuilt Lov. '
            'Pass eris=None so each fragment builds only its own domain integrals.'
        )

    if use_dlno_fragment_eris and mfcc.dm_corr is True:
        raise NotImplementedError(
            'Global dm_corr=True requires a full-molecule Lov and is not '
            'compatible with DLNO fragment-local integral construction.'
        )

    if mfcc.dm_corr is True:
        mfcc.dm_corr = make_rdm1_vo(mfcc, eris=eris, ao_repr=True)
    elif mfcc.dm_corr is False:
        mfcc.dm_corr = None

    nfrag = len(frag_lolist)
    if frag_nonvlist is None:
        frag_nonvlist = [[None,None]] * nfrag

    frag_res = [None] * nfrag
    for ifrag in range(nfrag):
        fraglo = numpy.asarray(frag_lolist[ifrag]).ravel()
        orbfragloc = orbloc[:,fraglo]
        frag_target_nocc, frag_target_nvir = frag_nonvlist[ifrag]
        frag_prescreen = mfcc.get_dlno_prescreen_fragment(ifrag)
        if use_dlno_fragment_eris:
            _require_dlno_fragment_domain(mfcc, frag_prescreen, ifrag)
        mfcc._current_ifrag = ifrag
        mfcc._nfrag = nfrag
        mfcc.profile_pass = getattr(mfcc, 'profile_pass', 'forward')
        frag_res[ifrag] = kernel_1frag(mfcc, eris, orbfragloc, no_type,
                                       frag_prescreen=frag_prescreen,
                                       frag_target_nocc=frag_target_nocc,
                                       frag_target_nvir=frag_target_nvir)
    return frag_res


def _use_dlno_fragment_eris(mfcc):
    return bool(mfcc.use_dlno_prescreen and mfcc.dlno_prescreen_data is not None)


def _make_lno_reference_eris(mfcc):
    eris = _LNOERIS(fock=mfcc.fock, s1e=mfcc.s1e)
    eris._common_init_(mfcc)
    return eris


def _require_dlno_fragment_domain(mfcc, frag_prescreen, ifrag):
    if frag_prescreen is None:
        raise RuntimeError(
            f'DLNO fragment-local integral construction requires prescreen data '
            f'for fragment {ifrag}; otherwise it would fall back to a full Lov.'
        )
    atmlst = frag_prescreen.get('extended_primary_domain')
    if atmlst is None or numpy.asarray(atmlst).size == 0:
        raise RuntimeError(
            f'DLNO fragment {ifrag} has no extended primary domain; otherwise '
            f'it would fall back to a full Lov.'
        )


@dataclass(frozen=True)
class _FragmentMFSettings:
    max_memory: float
    verbose: int
    conv_tol: float
    conv_tol_grad: object
    direct_scf: bool
    direct_scf_tol: float


@dataclass(frozen=True)
class _FragmentSolverSettings:
    solver_cls: type
    frozen: object
    thresh_occ: float
    thresh_vir: float
    lo_type: str
    no_type: str
    verbose: int
    verbose_imp: int
    use_local_virt: bool
    natorb_occdeg_thresh: float
    dm_corr_frag: object
    ccsd_t: bool
    dcsd: bool
    compute_domain_pt2: bool
    profile_fragments: bool
    profile_print: bool
    profile_mpi_indices: object
    profile_mpi_nfrag: object
    profile_mpi_print: bool
    profile_label: str
    profile_pass: str


class _FragmentSCFState(pytree.PytreeNode):
    _dynamic_attr = (
        'mol',
        'with_df',
        'mo_coeff',
        'mo_energy',
        'e_tot',
        'fock',
        's1e',
        'dm_corr',
    )
    _static_attr = ('mo_occ', 'mf_settings')

    def __init__(self, mol, with_df, mo_coeff, mo_energy, e_tot, fock, s1e,
                 dm_corr, mo_occ, mf_settings):
        self.mol = mol
        self.with_df = with_df
        self.mo_coeff = mo_coeff
        self.mo_energy = mo_energy
        self.e_tot = e_tot
        self.fock = fock
        self.s1e = s1e
        self.dm_corr = dm_corr
        self.mo_occ = mo_occ
        self.mf_settings = mf_settings


def _static_frozen(frozen):
    if frozen is None or numpy.isscalar(frozen):
        return frozen
    return tuple(map(int, numpy.asarray(frozen).ravel()))


def kernel_1frag(mfcc, eris, orbfragloc, no_type,
                 frag_prescreen=None,
                 frag_target_nocc=None, frag_target_nvir=None,
                 return_info=False):
    mf = mfcc._scf
    frozen_mask = mfcc.get_frozen_mask()
    thresh_pno = (mfcc.thresh_occ, mfcc.thresh_vir)
    _print_fragment_start(mfcc, frag_prescreen, orbfragloc)
    profile_start = time.perf_counter()
    eris_start = time.perf_counter()
    eris_fpno = make_fragment_eris(mfcc, eris, frag_prescreen)
    make_fragment_eris_s = time.perf_counter() - eris_start
    fpno_start = time.perf_counter()
    frzfrag, orbfrag, domain_pt2 = make_fpno1(
        mfcc, eris_fpno, orbfragloc, no_type,
        THRESH_INTERNAL, thresh_pno,
        frag_prescreen=frag_prescreen,
        frozen_mask=frozen_mask,
        frag_target_nocc=frag_target_nocc,
        frag_target_nvir=frag_target_nvir,
    )
    make_fpno1_s = time.perf_counter() - fpno_start
    info = _fragment_diagnostic_info(mfcc, eris_fpno, frag_prescreen, orbfragloc,
                                     frzfrag, orbfrag)
    info['label'] = _fragment_profile_label(mfcc)
    info['pass'] = _fragment_profile_pass(mfcc)
    info['nfrag'] = getattr(mfcc, '_nfrag', None)
    info['make_fragment_eris_s'] = make_fragment_eris_s
    info['make_fpno1_s'] = make_fpno1_s
    # Note: domain_pt2 is intentionally NOT stored on ``info`` here.  The
    # info dict is passed through ``impurity_solve`` as the
    # ``profile_info`` argument, which is a ``nondiff_argnum`` of the
    # ``_impurity_solve_jax`` custom_vjp; under tracing ``domain_pt2`` is
    # a JAX tracer and putting it into a nondiff argument raises
    # ``UnexpectedTracerError``.  The value is preserved via ``frag_res``
    # below (line ~485).
    eris_fpno = None
    if orbfrag is None:
        frag_res = (0., 0., 0.)
        if getattr(mfcc, 'compute_domain_pt2', False):
            frag_res = frag_res + (domain_pt2,)
        info['impurity_solve_s'] = 0.0
        info['wall_s'] = time.perf_counter() - profile_start
        _append_and_maybe_print_fragment_profile(mfcc, info)
        return (frag_res, info) if return_info else frag_res
    impurity_start = time.perf_counter()
    frag_res = mfcc.impurity_solve(mf, orbfrag, orbfragloc,
                                   frozen=frzfrag, eris=eris,
                                   frag_prescreen=frag_prescreen,
                                   profile_info=info)
    if getattr(mfcc, 'compute_domain_pt2', False):
        frag_res = frag_res + (domain_pt2,)
    info['impurity_solve_s'] = time.perf_counter() - impurity_start
    info['wall_s'] = time.perf_counter() - profile_start
    _append_and_maybe_print_fragment_profile(mfcc, info)
    return (frag_res, info) if return_info else frag_res


def _fragment_diagnostic_info(mfcc, eris, frag_prescreen, orbfragloc,
                              frzfrag, orbfrag):
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

    domain_atoms = ()
    domain_aos = 0
    n_strong_lmo = 0
    n_prescreen_occ = 0
    n_prescreen_vir = 0
    if frag_prescreen is not None:
        domain_atoms = tuple(
            map(int, numpy.asarray(
                frag_prescreen.get('extended_primary_domain', ()),
                dtype=numpy.int32,
            ).ravel())
        )
        if domain_atoms:
            domain_aos = int(
                dlno_util.ao_index_by_atom(
                    mf.mol, numpy.asarray(domain_atoms, dtype=numpy.int32)
                ).size
            )
        n_strong_lmo = int(
            numpy.asarray(
                frag_prescreen.get('strong_lmo_indices', ()),
                dtype=numpy.int32,
            ).size
        )
        occ_coeff = frag_prescreen.get('occ_prescreen_coeff')
        vir_coeff = frag_prescreen.get('vir_prescreen_coeff')
        n_prescreen_occ = 0 if occ_coeff is None else int(occ_coeff.shape[1])
        n_prescreen_vir = 0 if vir_coeff is None else int(vir_coeff.shape[1])

    return {
        'fragment': int(getattr(mfcc, '_current_ifrag', -1)),
        'n_lo': int(orbfragloc.shape[1]),
        'domain_atoms': len(domain_atoms),
        'domain_aos': domain_aos,
        'strong_lmo': n_strong_lmo,
        'prescreen_occ': n_prescreen_occ,
        'prescreen_vir': n_prescreen_vir,
        'active_occ': active_occ,
        'active_vir': active_vir,
        'lov_mb': lov_mb,
        't2_mb': t2_mb,
        'est_work_mb': lov_mb + 4.0 * t2_mb,
    }


def _dlno_outside_space(full_space, selected_space, thresh):
    """Build the frozen complement used only to preserve fragment MO layout.

    The complement can become numerically null when DLNO screening is very tight.
    Its SVD backward pass is then ill-conditioned, even though the selected
    prescreen space itself is well behaved. Treat this bookkeeping complement
    as frozen metadata.
    """
    thresh = max(float(thresh), 1e-8)
    if full_space is None or full_space.shape[1] == 0:
        return np.zeros((0, 0))
    if selected_space is None or selected_space.shape[1] == 0:
        return stop_grad(orthonormalize_colspace(full_space, thresh=thresh))
    if selected_space.shape[1] >= full_space.shape[1]:
        return np.zeros((full_space.shape[0], 0), dtype=full_space.dtype)

    residual = full_space - np.dot(
        stop_grad(selected_space),
        np.dot(stop_grad(selected_space.T.conj()), stop_grad(full_space)),
    )
    return stop_grad(orthonormalize_colspace(residual, thresh=thresh))


def _spans_full_molecule(mfcc, frag_prescreen):
    if frag_prescreen is None:
        return False
    atmlst = frag_prescreen.get('extended_primary_domain')
    if atmlst is None:
        return False
    return len(numpy.asarray(atmlst).ravel()) >= mfcc.mol.natm


def _subspace_equivalent(full_space, trial_space, thresh, overlap_tol=1e-6):
    if full_space is None or trial_space is None:
        return False
    if full_space.shape[1] == 0:
        return True
    if trial_space.shape[1] < full_space.shape[1]:
        return False

    q_full = orthonormalize_colspace(full_space, thresh=thresh)
    q_trial = orthonormalize_colspace(trial_space, thresh=thresh)
    if q_full.shape[1] < full_space.shape[1] or q_trial.shape[1] < full_space.shape[1]:
        return False

    ovlp = np.dot(q_full.T.conj(), q_trial)
    sigma = scipy.linalg.svd(ovlp, compute_uv=False)
    if sigma.size < full_space.shape[1]:
        return False
    return bool(numpy.all(numpy.abs(numpy.asarray(sigma[:full_space.shape[1]]) - 1.0) < overlap_tol))


def _semicanonicalized_coeff(base_orb, coeff, fock, s1e, semicanonicalize_fn):
    coeff = np.asarray(coeff)
    if coeff.shape[-1] == 0:
        return (
            np.zeros((0,), dtype=fock.dtype),
            np.zeros((base_orb.shape[1], 0), dtype=base_orb.dtype),
        )
    moe, orb = semicanonicalize_fn(fock, np.dot(base_orb, coeff))
    coeff_sc = reduce(np.dot, (base_orb.T.conj(), s1e, orb))
    return moe, coeff_sc


def _identity_coeff(n, dtype):
    return np.eye(n, dtype=dtype)


def _hstack_or_empty(parts, nrow, dtype):
    parts = [part for part in parts if part is not None and part.shape[-1] > 0]
    if parts:
        return np.hstack(parts)
    return np.zeros((nrow, 0), dtype=dtype)


def _concat_or_empty(parts, dtype):
    parts = [part for part in parts if part is not None and part.shape[0] > 0]
    if parts:
        return np.concatenate(parts)
    return np.zeros((0,), dtype=dtype)


def _mp2_fragment_energy_from_lov(Lov, moe_occ, moe_vir, prjlo):
    if Lov.shape[1] == 0 or Lov.shape[2] == 0:
        return np.zeros((), dtype=Lov.dtype)
    eov = moe_occ[:, None] - moe_vir
    ovov = np.einsum('Lia,Ljb->iajb', Lov, Lov)
    eiajb = eov[:, None, :, None] + eov[None, :, None, :]
    t2 = ovov.transpose(0, 2, 1, 3) / eiajb
    m = np.dot(prjlo.T, prjlo)
    eij = 2 * np.einsum('pjab,qajb->pq', t2, ovov)
    eij -= np.einsum('pjab,qbja->pq', t2, ovov)
    return np.einsum('ij,ij', eij, m)


def _domain_mp2_block_nvir(naux, nocc, nvir):
    try:
        block_mb = float(os.environ.get('PYSCFAD_LNO_DOMAIN_MP2_BLOCK_MB', 128.0))
    except ValueError:
        block_mb = 128.0
    target_bytes = max(block_mb, 1.0) * 1024.0**2
    bytes_per_b2 = 3 * nocc * nocc * numpy.dtype(numpy.float64).itemsize
    bytes_per_b = 2 * naux * nocc * numpy.dtype(numpy.float64).itemsize
    block = int(numpy.sqrt(target_bytes / max(bytes_per_b2, 1)))
    block = max(1, min(nvir, block))
    while block > 1 and bytes_per_b2 * block**2 + bytes_per_b * block > target_bytes:
        block -= 1
    return block


def _mp2_fragment_energy_from_df_lov_block(Lov, occ_coeff, vir_coeff,
                                           moe_occ, moe_vir, prjlo,
                                           a0, a1, b0, b1):
    vir_a = vir_coeff[:, a0:a1]
    vir_b = vir_coeff[:, b0:b1]
    Lov_a = np.einsum('ip,Lia,aq->Lpq', occ_coeff, Lov, vir_a)
    if b0 == a0 and b1 == a1:
        Lov_b = Lov_a
    else:
        Lov_b = np.einsum('ip,Lia,aq->Lpq', occ_coeff, Lov, vir_b)
    eov = moe_occ[:, None] - moe_vir
    eov_a = eov[:, a0:a1]
    eov_b = eov[:, b0:b1]
    m = np.dot(prjlo.T, prjlo)
    ovov = np.einsum('Lia,Ljb->iajb', Lov_a, Lov_b)
    eiajb = eov_a[:, None, :, None] + eov_b[None, :, None, :]
    t2 = ovov.transpose(0, 2, 1, 3) / eiajb
    e2 = 2 * np.einsum('pq,pjab,qajb->', m, t2, ovov)
    e2 -= np.einsum('pq,pjab,jaqb->', m, t2, ovov)
    return e2


def _mp2_fragment_energy_from_df_lov_impl(Lov, occ_coeff, vir_coeff,
                                          moe_occ, moe_vir, prjlo):
    naux = Lov.shape[0]
    nocc = occ_coeff.shape[1]
    nvir = vir_coeff.shape[1]
    if nocc == 0 or nvir == 0:
        return np.zeros((), dtype=Lov.dtype)

    e2 = np.zeros((), dtype=Lov.dtype)
    block_nvir = _domain_mp2_block_nvir(naux, nocc, nvir)

    for a0 in range(0, nvir, block_nvir):
        a1 = min(a0 + block_nvir, nvir)
        for b0 in range(0, nvir, block_nvir):
            b1 = min(b0 + block_nvir, nvir)
            e2 += _mp2_fragment_energy_from_df_lov_block(
                Lov, occ_coeff, vir_coeff, moe_occ, moe_vir, prjlo,
                a0, a1, b0, b1,
            )
    return e2


@jax.custom_vjp
def _mp2_fragment_energy_from_df_lov(Lov, occ_coeff, vir_coeff,
                                     moe_occ, moe_vir, prjlo):
    return _mp2_fragment_energy_from_df_lov_impl(
        Lov, occ_coeff, vir_coeff, moe_occ, moe_vir, prjlo
    )


def _mp2_fragment_energy_from_df_lov_fwd(Lov, occ_coeff, vir_coeff,
                                         moe_occ, moe_vir, prjlo):
    e2 = _mp2_fragment_energy_from_df_lov_impl(
        Lov, occ_coeff, vir_coeff, moe_occ, moe_vir, prjlo
    )
    return e2, (Lov, occ_coeff, vir_coeff, moe_occ, moe_vir, prjlo)


def _mp2_fragment_energy_from_df_lov_block_bwd(
        Lov, occ_coeff, vir_coeff, moe_occ, moe_vir, prjlo,
        a0, a1, b0, b1, e2_bar):
    vir_a = vir_coeff[:, a0:a1]
    vir_b = vir_coeff[:, b0:b1]
    Lov_a = np.einsum('ip,Lia,aq->Lpq', occ_coeff, Lov, vir_a)
    same_vir_block = b0 == a0 and b1 == a1
    if same_vir_block:
        Lov_b = Lov_a
    else:
        Lov_b = np.einsum('ip,Lia,aq->Lpq', occ_coeff, Lov, vir_b)

    eov = moe_occ[:, None] - moe_vir
    eov_a = eov[:, a0:a1]
    eov_b = eov[:, b0:b1]
    denom = eov_a[:, None, :, None] + eov_b[None, :, None, :]
    m = np.dot(prjlo.T, prjlo)
    ovov = np.einsum('Lia,Ljb->iajb', Lov_a, Lov_b)
    t2 = ovov.transpose(0, 2, 1, 3) / denom
    scale = np.asarray(e2_bar)

    m_bar = scale * (
        2 * np.einsum('pjab,qajb->pq', t2, ovov)
        - np.einsum('pjab,jaqb->pq', t2, ovov)
    )
    t2_bar = scale * (
        2 * np.einsum('pq,qajb->pjab', m, ovov)
        - np.einsum('pq,jaqb->pjab', m, ovov)
    )
    ovov_bar = scale * (
        2 * np.einsum('pq,pjab->qajb', m, t2)
        - np.einsum('pq,pjab->jaqb', m, t2)
    )

    ovov_t = ovov.transpose(0, 2, 1, 3)
    denom_bar = -t2_bar * ovov_t / (denom * denom)
    ovov_bar += (t2_bar / denom).transpose(0, 2, 1, 3)

    moe_occ_bar = np.zeros_like(moe_occ)
    moe_vir_bar = np.zeros_like(moe_vir)
    moe_occ_bar += np.sum(denom_bar, axis=(1, 2, 3))
    moe_occ_bar += np.sum(denom_bar, axis=(0, 2, 3))
    moe_vir_bar = moe_vir_bar.at[a0:a1].add(
        -np.sum(denom_bar, axis=(0, 1, 3))
    )
    moe_vir_bar = moe_vir_bar.at[b0:b1].add(
        -np.sum(denom_bar, axis=(0, 1, 2))
    )

    Lov_a_bar = np.einsum('iajb,Ljb->Lia', ovov_bar, Lov_b)
    Lov_b_bar = np.einsum('iajb,Lia->Ljb', ovov_bar, Lov_a)
    if same_vir_block:
        Lov_a_bar += Lov_b_bar

    Lov_bar = np.einsum('ip,Lpq,aq->Lia', occ_coeff, Lov_a_bar, vir_a)
    occ_coeff_bar = np.einsum('Lia,aq,Lpq->ip', Lov, vir_a, Lov_a_bar)
    vir_coeff_bar = np.zeros_like(vir_coeff)
    vir_coeff_bar = vir_coeff_bar.at[:, a0:a1].add(
        np.einsum('ip,Lia,Lpq->aq', occ_coeff, Lov, Lov_a_bar)
    )
    if not same_vir_block:
        Lov_bar += np.einsum('ip,Lpq,aq->Lia', occ_coeff, Lov_b_bar, vir_b)
        occ_coeff_bar += np.einsum('Lia,aq,Lpq->ip', Lov, vir_b, Lov_b_bar)
        vir_coeff_bar = vir_coeff_bar.at[:, b0:b1].add(
            np.einsum('ip,Lia,Lpq->aq', occ_coeff, Lov, Lov_b_bar)
        )

    prjlo_bar = np.dot(prjlo, m_bar + m_bar.T)
    return Lov_bar, occ_coeff_bar, vir_coeff_bar, moe_occ_bar, moe_vir_bar, prjlo_bar


def _mp2_fragment_energy_from_df_lov_bwd(res, e2_bar):
    Lov, occ_coeff, vir_coeff, moe_occ, moe_vir, prjlo = res
    naux = Lov.shape[0]
    nocc = occ_coeff.shape[1]
    nvir = vir_coeff.shape[1]
    if nocc == 0 or nvir == 0:
        return tuple(np.zeros_like(x) for x in res)

    Lov_bar = np.zeros_like(Lov)
    occ_coeff_bar = np.zeros_like(occ_coeff)
    vir_coeff_bar = np.zeros_like(vir_coeff)
    moe_occ_bar = np.zeros_like(moe_occ)
    moe_vir_bar = np.zeros_like(moe_vir)
    prjlo_bar = np.zeros_like(prjlo)
    block_nvir = _domain_mp2_block_nvir(naux, nocc, nvir)

    with _vjp_progress_section('domain MP2 correction backward'):
        for a0 in range(0, nvir, block_nvir):
            a1 = min(a0 + block_nvir, nvir)
            for b0 in range(0, nvir, block_nvir):
                b1 = min(b0 + block_nvir, nvir)
                block_bars = _mp2_fragment_energy_from_df_lov_block_bwd(
                    Lov, occ_coeff, vir_coeff, moe_occ, moe_vir, prjlo,
                    a0, a1, b0, b1, e2_bar,
                )
                Lov_bar += block_bars[0]
                occ_coeff_bar += block_bars[1]
                vir_coeff_bar += block_bars[2]
                moe_occ_bar += block_bars[3]
                moe_vir_bar += block_bars[4]
                prjlo_bar += block_bars[5]

    return (Lov_bar, occ_coeff_bar, vir_coeff_bar,
            moe_occ_bar, moe_vir_bar, prjlo_bar)


_mp2_fragment_energy_from_df_lov.defvjp(
    _mp2_fragment_energy_from_df_lov_fwd,
    _mp2_fragment_energy_from_df_lov_bwd,
)


def _mp2_fragment_energy_from_lt_projected_lov(
        Lov, occ_coeff, vir_coeff, moe_occ, moe_vir, prjlo,
        nlap=DOMAIN_MP2_LT_NLAP, quadrature=DOMAIN_MP2_LT_QUADRATURE,
        fit_ratio=DOMAIN_MP2_LT_FIT_RATIO):
    from pyscfad.mp import ltdfmp2

    @jax.checkpoint
    def _impl(Lov, occ_coeff, vir_coeff, moe_occ, moe_vir, prjlo):
        nocc = occ_coeff.shape[1]
        nvir = vir_coeff.shape[1]
        if nocc == 0 or nvir == 0 or prjlo.shape[0] == 0:
            return np.zeros((), dtype=Lov.dtype)

        Lov_domain = np.einsum('ip,Lia,aq->Lpq', occ_coeff, Lov, vir_coeff)
        mo_energy = np.concatenate((moe_occ, moe_vir))
        emp2, _, _ = ltdfmp2._contract_laplace_projected_occ(
            Lov_domain,
            mo_energy,
            prjlo.T,
            nocc,
            nvir,
            nlap=nlap,
            quadrature=quadrature,
            fit_ratio=fit_ratio,
        )
        return np.sum(emp2)

    return _impl(Lov, occ_coeff, vir_coeff, moe_occ, moe_vir, prjlo)


def _domain_mp2_fragment_energy(mfcc, eris, orbfragloc,
                                orbfragocc1, moefragocc1,
                                orbfragvir1, moefragvir1,
                                uocc2, uvir2, lovir,
                                semicanonicalize_fn):
    """Per-fragment LO-projected MP2 energy in the in-domain subspace.

    The in-domain occupied subspace is the joint span of ``uocc2`` (the
    external active occupied component selected for this fragment's
    domain) and ``orbfragocc1`` (the LO span itself), expressed in the
    active occupied MO basis ``orbocc1``.  Out-of-domain occupied
    orbitals (i.e., the orthogonal complement of this joint subspace
    within ``orbocc1``) are frozen.

    The crucial detail is *how* the orbital energies for this subspace
    are obtained.  An earlier version of this routine semi-canonicalized
    ``uocc2`` and ``orbfragocc1`` separately and concatenated the two
    blocks; that ignored the off-diagonal Fock coupling between the LO
    subspace and the external subspace, biasing the MP2 denominators.
    For a *full* domain (i.e., the joint subspace = all of ``orbocc1``)
    that block-wise scheme failed to recover the canonical MP2 even in
    principle.

    Here we instead diagonalize the Fock matrix in the joint in-domain
    subspace once, giving true canonical-within-domain orbital energies.
    For full domain this recovers the canonical full-system MP2 exactly;
    for partial domain it gives the "canonical MP2 inside the domain
    with out-of-domain occupied frozen" the LNO PNO-truncation correction
    is supposed to be.  Virtuals are handled the same way when ``lovir``
    is in use; for the common occupied-LO case ``vir_coeff`` is the full
    active virtual MO basis with its canonical eigenvalues
    (``moevir1``), which already gives a canonical evaluation.
    """
    mf = mfcc._scf
    s1e = eris.s1e
    fock = eris.fock
    orbocc1 = mfcc.split_mo()[1]
    orbvir1 = mfcc.split_mo()[2]
    moeocc1 = mfcc.split_moe()[1]
    moevir1 = mfcc.split_moe()[2]

    # ---- Build the joint in-domain occupied subspace in orbocc1 basis. ----
    occ_basis_parts = []
    if uocc2 is not None and uocc2.shape[-1] > 0:
        occ_basis_parts.append(uocc2)
    coeff_lo_in_orbocc1 = reduce(np.dot, (orbocc1.T.conj(), s1e, orbfragocc1))
    occ_basis_parts.append(coeff_lo_in_orbocc1)
    joint_occ = _hstack_or_empty(occ_basis_parts, orbocc1.shape[1], orbocc1.dtype)
    if joint_occ.shape[-1] == 0:
        return np.zeros((), dtype=eris.Lov.dtype)

    # ---- Diagonalize Fock in this joint subspace ----
    # In the canonical active-occ MO basis orbocc1, Fock = diag(moeocc1).
    # The projected Fock in the joint subspace is joint_occ.T @ diag(moe) @ joint_occ.
    fock_in_joint_occ = np.dot(
        joint_occ.T.conj(), moeocc1[:, None] * joint_occ
    )
    moe_occ, U_occ = np.linalg.eigh(fock_in_joint_occ)
    occ_coeff = np.dot(joint_occ, U_occ)   # still in orbocc1 basis

    # ---- Virtual subspace ----
    if lovir and uvir2 is not None and uvir2.shape[-1] > 0:
        # Joint-canonicalize the in-domain active virtual subspace too.
        vir_basis_parts = [uvir2]
        coeff_lov_in_orbvir1 = reduce(np.dot,
                                      (orbvir1.T.conj(), s1e, orbfragvir1))
        vir_basis_parts.append(coeff_lov_in_orbvir1)
        joint_vir = _hstack_or_empty(vir_basis_parts,
                                     orbvir1.shape[1], orbvir1.dtype)
        if joint_vir.shape[-1] == 0:
            return np.zeros((), dtype=eris.Lov.dtype)
        fock_in_joint_vir = np.dot(
            joint_vir.T.conj(), moevir1[:, None] * joint_vir
        )
        moe_vir, U_vir = np.linalg.eigh(fock_in_joint_vir)
        vir_coeff = np.dot(joint_vir, U_vir)
    elif lovir:
        # lovir but uvir2 missing/empty — use only the LO virtual span.
        coeff_lov_in_orbvir1 = reduce(np.dot,
                                      (orbvir1.T.conj(), s1e, orbfragvir1))
        if coeff_lov_in_orbvir1.shape[-1] == 0:
            return np.zeros((), dtype=eris.Lov.dtype)
        fock_in_joint_vir = np.dot(
            coeff_lov_in_orbvir1.T.conj(),
            moevir1[:, None] * coeff_lov_in_orbvir1,
        )
        moe_vir, U_vir = np.linalg.eigh(fock_in_joint_vir)
        vir_coeff = np.dot(coeff_lov_in_orbvir1, U_vir)
    else:
        # Occupied-LO common case: use the full active virtual MO basis
        # (which is already canonical with eigenvalues ``moevir1``).
        vir_coeff = _identity_coeff(orbvir1.shape[1], orbvir1.dtype)
        moe_vir = moevir1

    if vir_coeff.shape[-1] == 0:
        return np.zeros((), dtype=eris.Lov.dtype)

    # ---- Fragment-LO projector in the new canonical in-domain basis. ----
    occ_domain = np.dot(orbocc1, occ_coeff)
    prjlo = reduce(np.dot, (orbfragloc.T, s1e, occ_domain))
    if getattr(mfcc, 'domain_mp2_use_lt', DOMAIN_MP2_USE_LT):
        return _mp2_fragment_energy_from_lt_projected_lov(
            eris.Lov, occ_coeff, vir_coeff, moe_occ, moe_vir, prjlo,
            nlap=getattr(mfcc, 'domain_mp2_lt_nlap', DOMAIN_MP2_LT_NLAP),
            quadrature=getattr(mfcc, 'domain_mp2_lt_quadrature',
                               DOMAIN_MP2_LT_QUADRATURE),
            fit_ratio=getattr(mfcc, 'domain_mp2_lt_fit_ratio',
                              DOMAIN_MP2_LT_FIT_RATIO),
        )
    return _mp2_fragment_energy_from_df_lov(
        eris.Lov, occ_coeff, vir_coeff, moe_occ, moe_vir, prjlo
    )


def make_fpno1(mfcc, eris, orbfragloc, no_type, thresh_internal, thresh_external,
               frag_prescreen=None,
               frozen_mask=None, frag_target_nocc=None, frag_target_nvir=None):
    mytimer = timer.Timer()
    use_checkpoint = (
        USE_CHECKPOINT
        and not bool(getattr(mfcc, '_disable_checkpointed_mp2_rdm1', False))
    )
    domain_pt2 = np.zeros((), dtype=eris.fock.dtype)

    mf = mfcc._scf
    mo_occ = mf.mo_occ
    nocc = numpy.count_nonzero(mo_occ > THRESH_OCC)
    nmo = mo_occ.size

    orbocc0, orbocc1, orbvir1, orbvir0 = mfcc.split_mo()
    moeocc0, moeocc1, moevir1, moevir0 = mfcc.split_moe()
    nocc0, nocc1, nvir1, nvir0 = [m.size for m in [moeocc0,moeocc1,
                                                   moevir1,moevir0]]
    nlo = orbfragloc.shape[1]
    s1e = eris.s1e
    fock = eris.fock
    Lov = eris.Lov
    semicanonicalize_fn = semicanonicalize

    lovir = False
    if mfcc.use_local_virt:
        lovir = abs(reduce(numpy.dot,
                    (stop_grad(orbfragloc.T),
                     stop_grad(s1e),
                     stop_grad(orbvir1)))).max() > thresh_internal

    if isinstance(thresh_external, float):
        thresh_ext_occ = thresh_ext_vir = thresh_external
    else:
        thresh_ext_occ, thresh_ext_vir  = thresh_external

    # sanity check for no_type:
    if not lovir and no_type[0] != 'i':
        raise ValueError('Input LOs span only occ but input no_type[0] is not "i".')
    if not lovir and no_type[1] == 'i':
        raise ValueError('Input LOs span only occ but input no_type[1] is "i".')

    # split active occ/vir into internal(1) and external(2)
    m = reduce(np.dot, (orbfragloc.T, s1e, orbocc1))
    uocc1, uocc2 = projection_construction(m, thresh_internal)
    moefragocc1, orbfragocc1 = semicanonicalize_fn(fock, np.dot(orbocc1, uocc1))
    uocc2_outside = np.zeros((uocc2.shape[0], 0), dtype=uocc2.dtype)

    uvir2 = None
    uvir2_outside = np.zeros((orbvir1.shape[1], 0), dtype=orbvir1.dtype)
    if lovir:
        m = reduce(np.dot, (orbfragloc.T, s1e, orbvir1))
        uvir1, uvir2 = projection_construction(m, thresh_internal)
        moefragvir1, orbfragvir1 = semicanonicalize_fn(fock, np.dot(orbvir1, uvir1))

    if mfcc.use_dlno_prescreen:
        dlno_thresh_internal = max(float(thresh_internal), 1e-8)
        if frag_prescreen is not None:
            if not _spans_full_molecule(mfcc, frag_prescreen):
                uocc2_full = uocc2
                uocc2_dlno = mfcc.get_dlno_prescreen_space(
                    orbocc1,
                    frag_prescreen,
                    'occ_prescreen_coeff',
                    anchor_spaces=(uocc1,),
                    s1e=s1e,
                    thresh=dlno_thresh_internal,
                )
                if uocc2_dlno is not None and uocc2_dlno.shape[1] > 0:
                    if uocc2_dlno.shape[1] >= uocc2_full.shape[1]:
                        uocc2 = uocc2_full
                        uocc2_outside = np.zeros((uocc2_full.shape[0], 0), dtype=uocc2_full.dtype)
                    else:
                        uocc2_dlno = np.dot(uocc2_full, np.dot(uocc2_full.T.conj(), uocc2_dlno))
                        uocc2_dlno = orthonormalize_colspace_fixed_gauge(
                            uocc2_dlno,
                            thresh=dlno_thresh_internal,
                        )
                        if uocc2_dlno.shape[1] > 0:
                            if _subspace_equivalent(uocc2_full, uocc2_dlno, dlno_thresh_internal):
                                uocc2 = uocc2_full
                                uocc2_outside = np.zeros((uocc2_full.shape[0], 0), dtype=uocc2_full.dtype)
                            else:
                                uocc2 = uocc2_dlno
                                uocc2_outside = _dlno_outside_space(
                                    uocc2_full,
                                    uocc2,
                                    dlno_thresh_internal,
                                )

                vir_anchor = (uvir1,) if lovir else ()
                uvir2_full = uvir2
                uvir2_dlno = mfcc.get_dlno_prescreen_space(
                    orbvir1,
                    frag_prescreen,
                    'vir_prescreen_coeff',
                    anchor_spaces=vir_anchor,
                    s1e=s1e,
                    thresh=dlno_thresh_internal,
                )
                if uvir2_dlno is not None and uvir2_dlno.shape[1] > 0:
                    if lovir and uvir2_full is not None and uvir2_dlno.shape[1] >= uvir2_full.shape[1]:
                        uvir2 = uvir2_full
                        uvir2_outside = np.zeros((uvir2_full.shape[0], 0), dtype=uvir2_full.dtype)
                    else:
                        if lovir and uvir2_full is not None and uvir2_full.shape[1] > 0:
                            uvir2_dlno = np.dot(uvir2_full, np.dot(uvir2_full.T.conj(), uvir2_dlno))
                        uvir2_dlno = orthonormalize_colspace_grassmann(
                            uvir2_dlno,
                            thresh=dlno_thresh_internal,
                        )
                        if uvir2_dlno.shape[1] > 0:
                            uvir2 = uvir2_dlno
                            if lovir and uvir2_full is not None and uvir2_full.shape[1] > 0:
                                uvir2_outside = _dlno_outside_space(
                                    uvir2_full,
                                    uvir2,
                                    dlno_thresh_internal,
                                )
                            else:
                                ident = np.eye(orbvir1.shape[1], dtype=orbvir1.dtype)
                                uvir2_outside = _dlno_outside_space(
                                    ident,
                                    uvir2,
                                    dlno_thresh_internal,
                                )

    # augment virtual space
    uuocc2_corr = uuvir2_corr = None
    if mfcc.dm_corr is not None:
        uuvir2_corr = augment_virt(mfcc.dm_corr, orbfragocc1, orbvir1,
                                   min(thresh_ext_occ, thresh_ext_vir),
                                   s1e, uvir2)
        if uuvir2_corr.shape[-1] == 0:
            uuvir2_corr = None

    def moe_Ov(moefragocc):
        return (moefragocc[:,None] - moevir1)
    def moe_oV(moefragvir):
        return (moeocc1[:,None] - moefragvir)
    eov = moe_Ov(moeocc1)

    # Construct PT2 dm_vv
    if no_type == 'osv':
        u = reduce(np.dot, (orbocc1.T, s1e, orbfragocc1))
        Lia = eris.get_Ov(u)
        ovov = np.einsum('lIa, lIb->Iab', Lia, Lia)
        eia = moe_Ov(moefragocc1)
        eiajb = eia[:,:,None] + eia[:,None,:]
        dmvv = ovov / eiajb
        if lovir:
            dmvv = np.einsum('ip,Ipq,qj->Iij', uvir2.T, dmvv, uvir2)
        eia = Lia = ovov = eiajb = None
    elif no_type == 'ie' and use_checkpoint:
        u = reduce(np.dot, (orbocc1.T, s1e, orbfragocc1))
        eia = moe_Ov(moefragocc1)
        ejb = eov
        Lia = eris.get_Ov(u)
        Ljb = Lov
        dmvv, dmoo = _checkpointed.make_mp2_rdm1_ie(Lia, Ljb, eia, ejb)
        if mfcc.dm_corr_frag is True:
            _dmov = make_rdm1_vo_frag(mfcc, dmoo, dmvv,
                                      Lia, Ljb, eia, ejb,
                                      eris=eris, ao_repr=False).T
            uuocc2_corr, uuvir2_corr = augment_ov(_dmov,
                                                  min(thresh_ext_occ, thresh_ext_vir),
                                                  uocc2, uvir2)
            if uuocc2_corr.shape[-1] == 0:
                uuocc2_corr = None
            if uuvir2_corr.shape[-1] == 0:
                uuvir2_corr = None
        eia = ejb = Lia = Ljb = None
    elif no_type[1] == 'r':   # OvOv: IaJc,IbJc->ab
        u = reduce(np.dot, (orbocc1.T, s1e, orbfragocc1))
        ovov = eris.get_OvOv(u)
        eia = ejb = moe_Ov(moefragocc1)
        e1_or_e2 = 'e1'
        swapidx = 'ab'
    elif no_type[1] == 'e': # Ovov: Iajc,Ibjc->ab
        u = reduce(np.dot, (orbocc1.T, s1e, orbfragocc1))
        ovov = eris.get_Ovov(u)
        eia = moe_Ov(moefragocc1)
        ejb = eov
        e1_or_e2 = 'e1'
        swapidx = 'ab'
    else:                   # oVov: iCja,iCjb->ab
        u = reduce(np.dot, (orbvir1.T, s1e, orbfragvir1))
        ovov = eris.get_oVov(u)
        eia = moe_oV(moefragvir1)
        ejb = eov
        e1_or_e2 = 'e2'
        swapidx = 'ij'

    if no_type != 'osv':
        if no_type != 'ie' or not use_checkpoint:
            eiajb = (eia.ravel()[:,None] + ejb.ravel()).reshape(ovov.shape)
            t2 = ovov / eiajb
            dmvv = make_rdm1_mp2(t2, 'vv', e1_or_e2, swapidx)
            ovov = eiajb = None
        if uvir2 is not None:
            dmvv = reduce(np.dot, (uvir2.T, dmvv, uvir2))

    # Construct PT2 dm_oo
    if no_type == 'osv':
        u = reduce(np.dot, (orbvir1.T, s1e, orbfragvir1))
        Lia = eris.get_oV(u)
        ovov = np.einsum('liA, ljA->ijA', Lia, Lia)
        eia = moe_oV(moefragvir1)
        eiajb = eia[:,None,:] + eia[None,:,:]
        dmoo = ovov / eiajb
        dmoo = np.einsum('ip,pqA,qj->Aij', uocc2.T, dmoo, uocc2)
        eia = Lia = ovov = eiajb = None
    elif no_type in ['ie','ei']: # ie/ei share same t2
        if no_type[0] == 'e':   # oVov: iAkb,jAkb->ij
            e1_or_e2 = 'e1'
            swapidx = 'ij'
        else:                   # Ovov: Kaib,Kajb->ij
            e1_or_e2 = 'e2'
            swapidx = 'ab'
    else:
        t2 = None
        if no_type[0] == 'r':   # oVoV: iAkB,jAkB->ij
            u = reduce(np.dot, (orbvir1.T, s1e, orbfragvir1))
            ovov = eris.get_oVoV(u)
            eia = ejb = moe_oV(moefragvir1)
            e1_or_e2 = 'e1'
            swapidx = 'ab'
        elif no_type[0] == 'e': # oVov: iAkb,jAkb->ij
            u = reduce(np.dot, (orbvir1.T, s1e, orbfragvir1))
            ovov = eris.get_oVov(u)
            eia = moe_oV(moefragvir1)
            ejb = eov
            e1_or_e2 = 'e1'
            swapidx = 'ij'
        else:                   # Ovov: Kaib,Kajb->ij
            u = reduce(np.dot, (orbocc1.T, s1e, orbfragocc1))
            ovov = eris.get_Ovov(u)
            eia = moe_Ov(moefragocc1)
            ejb = eov
            e1_or_e2 = 'e2'
            swapidx = 'ab'

        eiajb = (eia.ravel()[:,None] + ejb.ravel()).reshape(ovov.shape)
        t2 = ovov / eiajb
        ovov = eiajb = None

    if no_type != 'osv':
        if no_type != 'ie' or not use_checkpoint:
            dmoo = make_rdm1_mp2(t2, 'oo', e1_or_e2, swapidx)
            t2 = None
        dmoo = reduce(np.dot, (uocc2.T, dmoo, uocc2))

    if getattr(mfcc, 'compute_domain_pt2', False):
        domain_pt2 = _domain_mp2_fragment_energy(
            mfcc, eris, orbfragloc,
            orbfragocc1, moefragocc1,
            orbfragvir1 if lovir else None,
            moefragvir1 if lovir else None,
            uocc2, uvir2, lovir,
            semicanonicalize_fn,
        )

    # Compress external space by PNO
    if frag_target_nocc is not None:
        frag_target_nocc -= orbfragocc1.shape[1]
    if no_type == 'osv':
        orbfragocc2, orbfragocc0 = osv_compression(dmoo, orbocc1, thresh_ext_occ,
                                                   uocc2, frag_target_nocc)
    else:
        if uocc2.shape[-1] == 0:
            orbfragocc12 = orbfragocc1
            orbfragocc0 = np.zeros((orbfragocc12.shape[0],0))
        else:
            orbfragocc2, orbfragocc0 = natorb_compression(dmoo, orbocc1, thresh_ext_occ,
                                                          uocc2, frag_target_nocc,
                                                          uuocc2_corr, mfcc.natorb_occdeg_thresh)
            orbfragocc12 = semicanonicalize_fn(fock, np.hstack([orbfragocc2, orbfragocc1]))[1]
    if lovir:
        if frag_target_nvir is not None:
            frag_target_nvir -= orbfragvir1.shape[1]
        if no_type == 'osv':
            orbfragvir2, orbfragvir0 = osv_compression(dmvv, orbvir1, thresh_ext_vir,
                                                       uvir2, frag_target_nvir)
        else:
            orbfragvir2, orbfragvir0 = natorb_compression(dmvv, orbvir1, thresh_ext_vir,
                                                          uvir2, frag_target_nvir,
                                                          uuvir2_corr, mfcc.natorb_occdeg_thresh)
        orbfragvir12 = semicanonicalize_fn(fock, np.hstack([orbfragvir2, orbfragvir1]))[1]
    else:
        orbfragvir2, orbfragvir0 = natorb_compression(dmvv, orbvir1, thresh_ext_vir,
                                                      uvir2, frag_target_nvir,
                                                      uuvir2_corr, mfcc.natorb_occdeg_thresh)
        if orbfragvir2.shape[-1] == 0:
            warnings.warn('No virtual orbital is included for this fragment, '
                          'setting correlation energy to zero.')
            return None, None, domain_pt2
        else:
            orbfragvir12 = semicanonicalize_fn(fock, orbfragvir2)[1]

    orbocc_outside = np.dot(orbocc1, uocc2_outside)
    orbvir_outside = np.dot(orbvir1, uvir2_outside)

    orbfrag = np.hstack([orbocc0, orbocc_outside, orbfragocc0, orbfragocc12,
                         orbfragvir12, orbfragvir0, orbvir_outside, orbvir0])
    nfrzocc = orbocc0.shape[1] + orbocc_outside.shape[1] + orbfragocc0.shape[1]
    frzfrag = numpy.hstack([numpy.arange(nfrzocc),
                            numpy.arange(nocc+orbfragvir12.shape[1],nmo)])

    if _verbose_at_least(mfcc, 5):
        mytimer.timer('make_fpno1:')
    return frzfrag, orbfrag, domain_pt2

def make_rdm1_mp2(t2, kind, e1_or_e2, swapidx):
    r''' Calculate MP2 rdm1 from T2.

    Args:
        t2 (np.ndarray):
            In 'ovov' order.
        kind (str):
            'oo' for oo-block; 'vv' for vv-block
        e1_or_e2 (str):
            Which electron are the free indices on?
            'e1': iakb,jakb -> ij; iajc,ibjc -> ab
            'e2': kaib,kajb -> ij; icja,icjb -> ab
        swapidx (str):
            How is the exchange term handled in einsum?
            'ij': iajb --> jaib
            'ab': iajb --> ibja
    '''
    if kind not in ['oo','vv']:
        raise KeyError('kind must be "oo" or "vv".')
    if e1_or_e2 not in ['e1','e2']:
        raise KeyError('e1_or_e2 must be "e1" or "e2".')
    if swapidx not in ['ij','ab']:
        raise KeyError('swapidx must be "ij" or "ab".')

    def swapped(s, swapidx):
        assert len(s) == 4
        order = [2,1,0,3] if swapidx == 'ij' else [0,3,2,1]
        return ''.join([s[i] for i in order])

    if kind == 'oo':
        if e1_or_e2 == 'e1':
            ids0 = 'iakb'
            ids1 = 'jakb'
        else:
            ids0 = 'kaib'
            ids1 = 'kajb'
        ids2 = 'ij'
    else:
        if e1_or_e2 == 'e1':
            ids0 = 'iajc'
            ids1 = 'ibjc'
        else:
            ids0 = 'icja'
            ids1 = 'icjb'
        ids2 = 'ab'
    ids0x = swapped(ids0, swapidx)
    ids1x = swapped(ids1, swapidx)

    merge_ids = lambda s0,s1,s2: '->'.join([','.join([s0,s1]),s2])
    dm = (np.einsum(merge_ids(ids0 , ids1 , ids2), t2, t2)*2 -
          np.einsum(merge_ids(ids0 , ids1x, ids2), t2, t2)   -
          np.einsum(merge_ids(ids0x, ids1 , ids2), t2, t2)   +
          np.einsum(merge_ids(ids0x, ids1x, ids2), t2, t2)*2) * 0.5
    return dm


def augment_ov(dmov, thresh, prj_occ=None, prj_vir=None):
    if prj_occ is not None:
        dmov = np.dot(prj_occ.T, dmov)
    if prj_vir is not None:
        dmov = np.dot(dmov, prj_vir)
    u, sigma, vt = scipy.linalg.svd(dmov)
    idx = numpy.where(abs(sigma) > thresh)[0]
    v = vt.conj().T
    return u[:,idx], v[:,idx]

def augment_virt(dm_corr, orbo, orbv, thresh, s1e=None, prj=None):
    nocc = orbo.shape[-1]
    dm_corr_ov = transform_rdm1(dm_corr, orbo, orbv, s1e)
    if prj is not None:
        dm_corr_ov = np.dot(dm_corr_ov, prj)
    _, sigma, vt = scipy.linalg.svd(dm_corr_ov)
    idx = numpy.where(abs(sigma) > thresh)[0]
    v = vt.conj().T
    return v[:,idx]

def transform_rdm1(dm0, orb1, orb2, s1e=None):
    if s1e is None:
        dm1 = reduce(np.dot, (orb1.conj().T, dm0, orb2))
    else:
        dm1 = reduce(np.dot, (orb1.conj().T, s1e, dm0, s1e, orb2))
    return dm1

def projection_construction(M, thresh):
    r''' Given M_{mu,i} = <mu | i> the ovlp between two orthonormal basis, find
    the unitary rotation |j'> = u_ij |i> so that {|j'>} significantly ovlp with
    {|mu>}.
    '''
    #e, u = scipy.linalg.eigh(np.dot(M.T, M))
    #mask = abs(e) > thresh
    #return u[:,mask], u[:,~mask]
    if M.shape[0] > M.shape[1]:
        v, e, _ = scipy.linalg.svd(M.conj().T)
    else:
        _, e, vt = scipy.linalg.svd(M)
        v = vt.conj().T
    norb = np.count_nonzero(e > thresh)
    return v[:,:norb], v[:,norb:]

def semicanonicalize(fock, orb):
    f = reduce(np.dot, (orb.T, fock, orb))
    if orb.shape[1] == 1:
        moe = f.ravel()
    else:
        moe, u = scipy.linalg.eigh(f, deg_thresh=SEMICANONICAL_DEG_THRESH)
        orb = np.dot(orb, u)
    return moe, orb

def canonical_orth_(S, thr=1e-8):
    '''Löwdin's canonical orthogonalization'''
    # Ensure the basis functions are normalized (symmetry-adapted ones are not!)
    normlz = np.power(np.diag(S), -0.5)
    Snorm = np.dot(np.diag(normlz), np.dot(S, np.diag(normlz)))
    # Form vectors for normalized overlap matrix
    Sval, Svec = scipy.linalg.eigh(Snorm)
    X = Svec[:,Sval>=thr] / np.sqrt(Sval[Sval>=thr])
    # Plug normalization back in
    X = np.dot(np.diag(normlz), X)
    return X

def orthonormalize_colspace(A, thresh=1e-10):
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError('Input space must be a rank-2 array.')
    if A.shape[1] == 0:
        return np.zeros((A.shape[0], 0), dtype=A.dtype)
    u, sigma, _ = scipy.linalg.svd(A, full_matrices=False)
    idx = numpy.where(abs(sigma) > thresh)[0]
    if len(idx) == 0:
        return np.zeros((A.shape[0], 0), dtype=A.dtype)
    return u[:, idx]

def _regularized_hermitian_invsqrt(gram, thresh=1e-10, deg_thresh=None):
    gram = np.asarray(gram)
    if gram.ndim != 2:
        raise ValueError('Gram matrix must be rank-2.')
    if gram.shape[0] == 0:
        return np.zeros_like(gram)
    if deg_thresh is None:
        deg_thresh = max(float(thresh), 1e-9)
    e, v = scipy.linalg.eigh(gram, deg_thresh=deg_thresh)
    e = np.real(e)
    floor = max(float(thresh), 1e-14)
    scale = np.where(e > floor, 1.0 / np.sqrt(e), 1.0 / np.sqrt(floor))
    return np.dot(v * scale[None, :], v.T.conj())

def orthonormalize_colspace_smooth(A, thresh=1e-10):
    """Differentiably orthonormalize a selected subspace with mild regularization."""
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError('Input space must be a rank-2 array.')
    if A.shape[1] == 0:
        return np.zeros((A.shape[0], 0), dtype=A.dtype)
    floor = max(float(thresh), 1e-10)
    gram = np.dot(A.T.conj(), A)
    eye = np.eye(gram.shape[0], dtype=gram.dtype)
    chol = np.linalg.cholesky(gram + floor * eye)
    x = jsp_linalg.solve_triangular(chol.T.conj(), eye, lower=False)
    return np.dot(A, x)

def orthonormalize_metric_colspace_smooth(A, s, thresh=1e-10):
    """Differentiably metric-orthonormalize a subspace with mild regularization.

    Unlike the fixed-gauge helper, this uses a matrix inverse square root of the
    Gram matrix.  The result depends only on the projector-valued function of the
    Gram matrix and therefore avoids the arbitrary eigenvector gauge inside
    exactly or nearly degenerate subspaces.

    Dispatches to a pure numpy/scipy path on concrete (non-tracer) inputs to
    avoid XLA compile-cache growth in per-LMO/per-fragment eager prescreen
    builds.  Under ``jax.value_and_grad`` the JAX path is taken so the gradient
    propagates.
    """
    if isinstance(A, jax.core.Tracer) or isinstance(s, jax.core.Tracer):
        A = np.asarray(A)
        s = np.asarray(s)
        if A.ndim != 2:
            raise ValueError('Input space must be a rank-2 array.')
        if A.shape[1] == 0:
            return np.zeros((A.shape[0], 0), dtype=A.dtype)
        floor = max(float(thresh), 1e-10)
        gram = np.dot(A.T.conj(), np.dot(s, A))
        eye = np.eye(gram.shape[0], dtype=gram.dtype)
        chol = np.linalg.cholesky(gram + floor * eye)
        x = jsp_linalg.solve_triangular(chol.T.conj(), eye, lower=False)
        return np.dot(A, x)
    A_np = numpy.asarray(A)
    s_np = numpy.asarray(s)
    if A_np.ndim != 2:
        raise ValueError('Input space must be a rank-2 array.')
    if A_np.shape[1] == 0:
        return numpy.zeros((A_np.shape[0], 0), dtype=A_np.dtype)
    floor = max(float(thresh), 1e-10)
    gram = A_np.T.conj() @ s_np @ A_np
    eye = numpy.eye(gram.shape[0], dtype=gram.dtype)
    chol = numpy.linalg.cholesky(gram + floor * eye)
    from scipy.linalg import solve_triangular
    x = solve_triangular(chol.T.conj(), eye, lower=False)
    return A_np @ x


def orthonormalize_colspace_fixed_gauge(A, thresh=1e-10):
    """Orthonormalize a selected subspace while freezing gauge rotations.

    For DLNO-selected fragment spaces we want geometry response of the span, but
    the internal orthonormalization gauge is not physically meaningful and its
    derivative is often singular. Keep the span differentiable while treating
    the orthonormalizing rotation as fixed metadata.
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError('Input space must be a rank-2 array.')
    if A.shape[1] == 0:
        return np.zeros((A.shape[0], 0), dtype=A.dtype)
    if A.shape[1] == 1:
        nrm = stop_grad(np.sqrt(np.dot(A[:, 0].conj(), A[:, 0])))
        nrm = numpy.asarray(jax.device_get(nrm)).reshape(())
        if abs(nrm) <= thresh:
            return np.zeros((A.shape[0], 0), dtype=A.dtype)
        return A / stop_grad(np.asarray(nrm))
    s = stop_grad(np.dot(A.T.conj(), A))
    s = numpy.asarray(jax.device_get(s), dtype=numpy.float64)
    if s.ndim == 0:
        s = s.reshape(1, 1)
    e, v = numpy.linalg.eigh(s)
    e = numpy.real_if_close(e)
    idx = numpy.where(e > thresh)[0]
    if len(idx) == 0:
        return np.zeros((A.shape[0], 0), dtype=A.dtype)
    x = v[:, idx] / numpy.sqrt(e[idx])[None, :]
    x = stop_grad(np.asarray(x))
    return np.dot(A, x)


def orthonormalize_colspace_grassmann(A, thresh=1e-10):
    """Orthonormalize a subspace with a gauge-invariant reverse pass.

    The returned columns are an ordinary orthonormal basis, but the backward
    rule removes cotangent components that only rotate/rescale that basis
    inside its own span.  The remaining cotangent corresponds to changes of the
    selected subspace itself, which is the physically meaningful DLNO object.
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError('Input space must be a rank-2 array.')
    thresh = float(thresh)

    @jax.custom_vjp
    def _orth_grassmann(A):
        q, _ = _orthonormalize_colspace_grassmann_primal(A, thresh)
        return q

    def _orth_grassmann_fwd(A):
        q, x = _orthonormalize_colspace_grassmann_primal(A, thresh)
        return q, (q, x)

    def _orth_grassmann_bwd(res, g):
        q, x = res
        if q.shape[1] == 0:
            return (np.zeros((q.shape[0], x.shape[0]), dtype=g.dtype),)
        # Remove the vertical/internal-basis cotangent before propagating back
        # through A -> A @ x.  With q orthonormal this is (I - q q^H) g.
        g_h = g - np.dot(q, np.dot(q.T.conj(), g))
        return (np.dot(g_h, x.T.conj()),)

    _orth_grassmann.defvjp(_orth_grassmann_fwd, _orth_grassmann_bwd)
    return _orth_grassmann(A)


def _orthonormalize_colspace_grassmann_primal(A, thresh=1e-10):
    A = np.asarray(A)
    if A.shape[1] == 0:
        x = np.zeros((0, 0), dtype=A.dtype)
        return np.zeros((A.shape[0], 0), dtype=A.dtype), x
    if A.shape[1] == 1:
        nrm = stop_grad(np.sqrt(np.dot(A[:, 0].conj(), A[:, 0])))
        nrm = numpy.asarray(jax.device_get(nrm)).reshape(())
        if abs(nrm) <= thresh:
            x = np.zeros((1, 0), dtype=A.dtype)
            return np.zeros((A.shape[0], 0), dtype=A.dtype), x
        x = stop_grad(np.asarray([[1.0 / nrm]], dtype=A.dtype))
        return np.dot(A, x), x

    s = stop_grad(np.dot(A.T.conj(), A))
    s = numpy.asarray(jax.device_get(s), dtype=numpy.float64)
    if s.ndim == 0:
        s = s.reshape(1, 1)
    e, v = numpy.linalg.eigh(s)
    e = numpy.real_if_close(e)
    idx = numpy.where(e > thresh)[0]
    if len(idx) == 0:
        x = np.zeros((A.shape[1], 0), dtype=A.dtype)
        return np.zeros((A.shape[0], 0), dtype=A.dtype), x
    x = v[:, idx] / numpy.sqrt(e[idx])[None, :]
    x = stop_grad(np.asarray(x, dtype=A.dtype))
    return np.dot(A, x), x


def orthonormalize_metric_colspace_fixed_gauge(A, s, thresh=1e-10):
    """Metric-orthonormalize a subspace while freezing internal gauge rotations."""
    A = np.asarray(A)
    s = np.asarray(s)
    if A.ndim != 2:
        raise ValueError('Input space must be a rank-2 array.')
    if A.shape[1] == 0:
        return np.zeros((A.shape[0], 0), dtype=A.dtype)
    if A.shape[1] == 1:
        nrm = stop_grad(np.sqrt(np.dot(A[:, 0].conj(), np.dot(s, A[:, 0]))))
        nrm = numpy.asarray(jax.device_get(nrm)).reshape(())
        if abs(nrm) <= thresh:
            return np.zeros((A.shape[0], 0), dtype=A.dtype)
        return A / stop_grad(np.asarray(nrm))

    gram = stop_grad(np.dot(A.T.conj(), np.dot(s, A)))
    gram = numpy.asarray(jax.device_get(gram), dtype=numpy.float64)
    if gram.ndim == 0:
        gram = gram.reshape(1, 1)
    e, v = numpy.linalg.eigh(gram)
    e = numpy.real_if_close(e)
    idx = numpy.where(e > thresh)[0]
    if len(idx) == 0:
        return np.zeros((A.shape[0], 0), dtype=A.dtype)
    x = v[:, idx] / numpy.sqrt(e[idx])[None, :]
    x = stop_grad(np.asarray(x))
    return np.dot(A, x)

def compress_colspace_numerical_rank(A, thresh=1e-10):
    """Drop numerically null column-space directions while freezing only that choice.

    This is used when a residual subspace is formed by subtracting a large anchor
    space from a mapped DLNO space.  The residual often contains exact or nearly
    exact null directions that should not be carried into later differentiable
    orthonormalization steps.
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError('Input space must be a rank-2 array.')
    if A.shape[1] == 0:
        return np.zeros((A.shape[0], 0), dtype=A.dtype)
    gram = stop_grad(np.dot(A.T.conj(), A))
    gram = numpy.asarray(jax.device_get(gram), dtype=numpy.float64)
    if gram.ndim == 0:
        gram = gram.reshape(1, 1)
    e, v = numpy.linalg.eigh(gram)
    e = numpy.real_if_close(e)
    idx = numpy.where(e > thresh)[0]
    if len(idx) == 0:
        return np.zeros((A.shape[0], 0), dtype=A.dtype)
    if len(idx) == A.shape[1]:
        return A
    keep = stop_grad(np.asarray(v[:, idx], dtype=A.dtype))
    return np.dot(A, keep)

def stack_colspaces(*spaces):
    cols = []
    nrow = None
    dtype = None
    for space in spaces:
        if space is None:
            continue
        space = np.asarray(space)
        if space.ndim != 2:
            raise ValueError('Each space must be a rank-2 array.')
        if nrow is None:
            nrow = space.shape[0]
            dtype = space.dtype
        elif space.shape[0] != nrow:
            raise ValueError('All spaces must have the same row dimension.')
        if space.shape[1] > 0:
            cols.append(space)
    if cols:
        return np.hstack(cols)
    if nrow is None:
        return np.zeros((0, 0))
    return np.zeros((nrow, 0), dtype=dtype)

def collocate_unitary(us):
    '''Collocate a few unitary matrices

    Args:
        us : list, tuple
            list of unitary matrices

    Returns:
        us0 : array
            Collocated unitary matrix
        x1 : array
            Unitary matrix which transforms
            to the orthogonal vector space
    '''
    us = np.concatenate(us, axis=1)
    x = canonical_orth_(np.dot(us.T, us))
    us0 = np.dot(us, x)

    us1 = np.eye(us0.shape[0]) - np.dot(us0, us0.T)
    e1, x1 = scipy.linalg.eigh(us1)
    x1 = x1[:, e1 > 0.99]
    assert us0.shape[-1] + x1.shape[-1] == us0.shape[0]
    return us0, x1

def osv_compression(dms, orb, thresh, prj=None, norb_target=None):
    us = []
    for dm in dms:
        e, u = scipy.linalg.eigh(dm, deg_thresh=COMPRESS_DEG_THRESH)
        us.append(u[:, abs(e) > thresh])
    us0, x1 = collocate_unitary(us)

    if prj is not None:
        orb = np.dot(orb, prj)
    orb1x = np.dot(orb, us0)
    orb0x = np.dot(orb, x1)
    return orb1x, orb0x

def natorb_compression(dm, orb, thresh, prj=None, norb_target=None,
                       uuvir2_corr=None, natorb_occdeg_thresh=0):
    e, u = scipy.linalg.eigh(dm, deg_thresh=COMPRESS_DEG_THRESH)
    if norb_target is None:
        idx = numpy.where(abs(e) > thresh)[0]
        if len(idx) > 0 and natorb_occdeg_thresh > 0:
            # NOTE include near degenerate states
            idx_deg = numpy.where(abs(e - e[idx[0]]) < natorb_occdeg_thresh)[0]
            idx = numpy.union1d(idx, idx_deg)
    elif isinstance(norb_target, (int, numpy.integer)):
        if norb_target < 0:
            raise ValueError(f'Target norb is negative: {norb_target}.')
        elif norb_target > e.size:
            raise ValueError(f'Target norb exceeds total number of orbs: {norb_target} > {e.size}')
        order = e.argsort()[::-1]
        idx = order[:norb_target]
    else:
        raise TypeError('Input "norb_target" should be integer type.')
    idxc = numpy.array([i for i in range(e.size) if i not in idx])

    if prj is not None:
        orb = np.dot(orb, prj)

    if uuvir2_corr is not None:
        us = (u[:,idx], uuvir2_corr)
        us0, x1 = collocate_unitary(us)
        orb1x = np.dot(orb, us0)
        orb0x = np.dot(orb, x1)
    else:
        orbx = np.dot(orb, u)
        orb1x = sub_colspace(orbx, idx)
        orb0x = sub_colspace(orbx, idxc)
    return orb1x, orb0x

def sub_colspace(A, idx):
    if idx.size == 0:
        return np.zeros([A.shape[0],0])
    else:
        return A[:,idx]

def get_cholesky_mos(mo_coeff):
    from pyscfad.lo.cholesky import cholesky_mos
    return cholesky_mos(mo_coeff)

def get_iao(mol, mo_coeff, minao='minao', orth=True):
    from pyscfad.lo.orth import vec_lowdin
    from pyscfad.lo.iao import iao as iao_kernel
    c = iao_kernel(mol, mo_coeff, minao=minao)

    if orth:
        s = mol.intor_symmetric('int1e_ovlp')
        c = vec_lowdin(c, s)
    return c

def get_ibo(mol, mo_coeff, init_guess=None,
            conv_tol=1e-10, symmetry=False, options=None):
    return get_pm(mol, mo_coeff, pop_method='ibo',
                  init_guess=init_guess, conv_tol=conv_tol,
                  symmetry=symmetry, options=options)

def get_boys(mol, mo_coeff, init_guess=None,
             conv_tol=1e-10, symmetry=False, options=None):
    from pyscfad.lo.boys import boys
    return boys(mol, mo_coeff, init_guess=init_guess,
                conv_tol=conv_tol, symmetry=symmetry,
                gmres_options=options)

def get_pm(mol, mo_coeff, pop_method='mulliken',
           init_guess=None, conv_tol=1e-10,
           symmetry=False, options=None):
    from pyscfad.lo.pipek import pm
    return pm(mol, mo_coeff, pop_method=pop_method,
              init_guess=init_guess, conv_tol=conv_tol,
              symmetry=symmetry, gmres_options=options)

def mo_splitter(maskact, maskocc, kind='mask'):
    ''' Split MO indices into
        - frozen occupieds
        - active occupieds
        - active virtuals
        - frozen virtuals

    Args:
        maskact (array-like, bool type):
            An array of length nmo with bool elements. True means an MO is active.
        maskocc (array-like, bool type):
            An array of length nmo with bool elements. True means an MO is occupied.
        kind (str):
            Determine the return type.
            'mask'  : return masks each of length nmo
            'index' : return index arrays
            'idx'   : same as 'index'

    Return:
        See the description for input arg 'kind' above.
    '''
    maskfrzocc = ~maskact &  maskocc
    maskactocc =  maskact &  maskocc
    maskactvir =  maskact & ~maskocc
    maskfrzvir = ~maskact & ~maskocc
    if kind == 'mask':
        return maskfrzocc, maskactocc, maskactvir, maskfrzvir
    elif kind in ['index','idx']:
        return [numpy.where(m)[0] for m in [maskfrzocc, maskactocc,
                                            maskactvir, maskfrzvir]]
    else:
        raise ValueError("'kind' must be 'mask' or 'index'(='idx').")


class LNO(pytree.PytreeNode):
    _dynamic_attr = {'_scf', 'mol', 'with_df'}

    def __init__(self, mf, thresh=1e-4, frozen=None, fock=None, s1e=None, **kwargs):
        self._scf = mf
        self.mol = mf.mol
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            raise KeyError('The mean-field object has no density fitting.')

        self.frozen = frozen
        self.fock = fock
        self.s1e = s1e

        self.thresh_occ = thresh
        self.thresh_vir = thresh
        self.lo_type = 'iao'
        self.no_type = 'ie'
        self.verbose = self.verbose_imp = mf.mol.verbose

        # Whether to use local virtual orbitals
        self.use_local_virt = True
        # Natural orbitals with occupation number difference smaller than
        # natorb_occdeg_thresh will be added to the correlation space
        self.natorb_occdeg_thresh = 0
        # MP2 (relaxed dm - unrelaxed dm) in AO basis for augmenting virtual space
        self.dm_corr = None
        self.dm_corr_frag = None
        self.use_dlno_prescreen = False
        self.dlno_prescreen_data = None
        self.compute_domain_pt2 = False
        self.domain_mp2_use_lt = DOMAIN_MP2_USE_LT
        self.domain_mp2_lt_nlap = DOMAIN_MP2_LT_NLAP
        self.domain_mp2_lt_quadrature = DOMAIN_MP2_LT_QUADRATURE
        self.domain_mp2_lt_fit_ratio = DOMAIN_MP2_LT_FIT_RATIO
        self.profile_print = True
        self.profile_mpi_indices = None
        self.profile_mpi_nfrag = None
        self.profile_mpi_print = False

        self._nmo = None
        self._nocc = None
        self.mo_occ = mf.mo_occ
        self._current_ifrag = None

    get_nocc = get_nocc
    get_nmo = get_nmo

    @property
    def nocc(self):
        return self.get_nocc()

    @property
    def nmo(self):
        return self.get_nmo()

    def mo_splitter(self, mo_occ, kind='mask'):
        r''' Return index arrays that split MOs into
            - frozen occupieds
            - active occupieds
            - active virtuals
            - frozen virtuals

        Args:
            kind (str):
                'mask'  : return masks each of length nmo
                'index' : return index arrays
                'idx'   : same as 'index'
        '''
        maskact = self.get_frozen_mask()
        maskocc = mo_occ > THRESH_OCC
        return mo_splitter(maskact, maskocc, kind=kind)

    def split_mo(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None:
            mo_coeff = self._scf.mo_coeff
        if mo_occ is None:
            mo_occ = self._scf.mo_occ
        masks = self.mo_splitter(mo_occ)
        return [mo_coeff[:,m] for m in masks]

    def split_moe(self, mo_energy=None, mo_occ=None):
        if mo_energy is None:
            mo_energy = self._scf.mo_energy
        if mo_occ is None:
            mo_occ = self._scf.mo_occ
        masks = self.mo_splitter(mo_occ)
        return [mo_energy[m] for m in masks]

    def get_lo(self, mol=None, mo_coeff=None, mo_occ=None,
               lo_type='iao', init_guess=None, symmetry=False,
               options=None):
        if mol is None:
            mol = self._scf.mol
        if mo_coeff is None:
            mo_coeff = self._scf.mo_coeff
        if mo_occ is None:
            mo_occ = self._scf.mo_occ

        orbocc = self.split_mo(mo_coeff, mo_occ)[1]

        if lo_type.lower() == 'iao':
            orbloc = get_iao(mol, orbocc)
        elif lo_type.lower() == 'ibo':
            orbloc = get_ibo(mol, orbocc, init_guess=init_guess,
                             symmetry=symmetry, options=options)
        elif lo_type.lower() == 'boys':
            orbloc = get_boys(mol, orbocc, init_guess=init_guess,
                              symmetry=symmetry, options=options)
        elif lo_type.lower() == 'pm':
            orbloc = get_pm(mol, orbocc, init_guess=init_guess,
                            symmetry=symmetry, options=options)
        elif lo_type.lower() == 'cholesky':
            orbloc = get_cholesky_mos(orbocc)
        else:
            raise KeyError(f'Unrecognized orbital localization method: {lo_type}.')
        return orbloc

    def ao2mo(self, fock=None, s1e=None):
        if fock is None:
            fock = self.fock
        if s1e is None:
            s1e = self.s1e

        if self.with_df is not None:
            eris = _make_df_eris_incore(self, fock, s1e)
        else:
            raise NotImplementedError
        return eris

    def kernel(self,
               frag_lolist=None,
               frag_wghtlist=None,
               frag_atmlist=None,
               lo_type=None,
               no_type=None,
               frag_nonvlist=None,
               orbloc=None,
               lo_init_guess=None,
               lo_symmetry=False,
               lo_options=None):
        if lo_type is None:
            lo_type = self.lo_type
        if no_type is None:
            no_type = self.no_type
        if orbloc is None:
            orbloc = self.get_lo(lo_type=lo_type, init_guess=lo_init_guess,
                                 symmetry=lo_symmetry, options=lo_options)

        # LO assignment to fragments
        if frag_lolist is None:
            if frag_atmlist is None:
                #log.info('Grouping LOs by single-atom fragments')
                frag_atmlist = stop_trace(autofrag)(self.mol)
            else:
                #log.info('Grouping LOs by user input atom-based fragments')
                pass
            frag_lolist = stop_trace(map_lo_to_frag)(self.mol, orbloc, frag_atmlist,
                                         verbose=self.verbose)
        elif frag_lolist == '1o':
            #log.info('Using single-LO fragment')
            frag_lolist = np.arange(orbloc.shape[1]).reshape(-1,1)
        else:
            #log.info('Using user input LO-fragment assignment')
            pass

        nfrag = len(frag_lolist)
        if frag_wghtlist is None:
            frag_wghtlist = np.ones(nfrag)

        frag_res = kernel(self, orbloc, frag_lolist, no_type=no_type,
                          frag_nonvlist=frag_nonvlist)
        self._post_proc(frag_res, frag_wghtlist)

    def get_dlno_prescreen_fragment(self, ifrag=None):
        if not self.use_dlno_prescreen:
            return None
        data = self.dlno_prescreen_data
        if data is None:
            return None
        if ifrag is None:
            ifrag = self._current_ifrag
        if ifrag is None:
            return None
        frag_data = data.get('fragment_data')
        if frag_data is None or ifrag >= len(frag_data):
            return None
        return frag_data[ifrag]

    def get_dlno_prescreen_space(self, orb, frag_data, key, anchor_spaces=(),
                                 s1e=None, thresh=THRESH_INTERNAL):
        if frag_data is None or key not in frag_data:
            return None

        # The DLNO prescreen coefficient matrices define the selected
        # prescreen subspace inside a fixed domain.  Keep their span
        # differentiable here, but later prune only nearly-null virtual
        # overlap directions when embedding that span in the current external
        # virtual MO space.  This preserves the physical subspace response
        # without letting numerically meaningless weak directions dominate.
        coeff = np.asarray(frag_data[key])
        if coeff.ndim != 2:
            return None
        if coeff.shape[1] == 0:
            return np.zeros((orb.shape[1], 0), dtype=orb.dtype)

        atmlst = frag_data.get('extended_primary_domain')
        if atmlst is None:
            return None
        atmlst = numpy.asarray(atmlst, dtype=numpy.int32)
        if atmlst.size == 0:
            return np.zeros((orb.shape[1], 0), dtype=orb.dtype)

        aoslices = self.mol.aoslice_by_atom()[:, 2:]
        ao_idx_lst = [numpy.arange(*x) for x in aoslices[atmlst].reshape(-1, 2)]
        if not ao_idx_lst:
            return np.zeros((orb.shape[1], 0), dtype=orb.dtype)
        ao_idx = reduce(numpy.union1d, ao_idx_lst)
        if coeff.shape[0] != len(ao_idx):
            raise ValueError('DLNO prescreen data has incompatible AO dimension for '
                             f'fragment {self._current_ifrag}: expected '
                             f'{len(ao_idx)}, got {coeff.shape[0]}')

        if s1e is None:
            if self.s1e is not None:
                s1e = self.s1e
            else:
                s1e = self._scf.get_ovlp()

        coeff_full = np.zeros((self.mol.nao_nr(), coeff.shape[1]), dtype=coeff.dtype)
        coeff_full = coeff_full.at[ao_idx].set(coeff)
        u = reduce(np.dot, (orb.T.conj(), s1e, coeff_full))

        eff_thresh = thresh
        if key == 'vir_prescreen_coeff':
            eff_thresh = max(float(thresh), DLNO_VIR_MAP_THRESH)
        anchor = stack_colspaces(*anchor_spaces)
        if anchor.shape[1] > 0:
            u = u - np.dot(anchor, np.dot(anchor.T.conj(), u))
            u = compress_colspace_numerical_rank(u, thresh=eff_thresh)
        u = orthonormalize_colspace_smooth(u, thresh=eff_thresh)
        return u

    def _post_proc(self, frag_res, frag_wghtlist):
        raise NotImplementedError

    get_frozen_mask = get_frozen_mask

def _make_df_eris_incore(mycc, fock=None, s1e=None):
    if fock is None:
        fock = mycc.fock
    if s1e is None:
        s1e = mycc.s1e
    eris = _LNODFINCOREERIS(fock=fock, s1e=s1e)
    eris._common_init_(mycc)
    return eris

class _LNOERIS():
    def __init__(self, fock=None, s1e=None):
        #self.mo_coeff = None
        #self.nocc = None
        #self.h1e = None
        self.s1e = s1e
        #self.vhf = None
        self.fock = fock
        self.Lov = None

    def _common_init_(self, mcc):
        mf = mcc._scf
        if self.s1e is None:
            self.s1e = mf.get_ovlp()
        if self.fock is None:
            h1e = mf.get_hcore()
            vhf = mf.get_veff()
            self.fock = mf.get_fock(h1e=h1e, s1e=self.s1e, vhf=vhf)
            del h1e, vhf

class _LNODFINCOREERIS(_LNOERIS):
    def _common_init_(self, mcc):
        super()._common_init_(mcc)
        orbo, orbv = mcc.split_mo()[1:3]
        nocc = orbo.shape[-1]
        mo_coeff = np.concatenate((orbo, orbv), axis=-1)
        self.Lov = get_Lov(mcc._scf, mo_coeff, nocc)

    def get_Ov(self, u):
        return np.einsum('iI,Lia->LIa', u, self.Lov)

    def get_oV(self, u):
        return np.einsum('aA,Lia->LiA', u, self.Lov)

    @staticmethod
    def _get_eris(Lia, Ljb):
        return np.einsum('Lia,Ljb->iajb', Lia, Ljb)

    def get_Ovov(self, u):
        LOv = self.get_Ov(u)
        return self._get_eris(LOv, self.Lov)

    def get_OvOv(self, u):
        LOv = self.get_Ov(u)
        return self._get_eris(LOv, LOv)

    def get_oVov(self, u):
        LoV = self.get_oV(u)
        return self._get_eris(LoV, self.Lov)

    def get_oVoV(self, u):
        LoV = self.get_oV(u)
        return self._get_eris(LoV, LoV)

def _local_domain_atmlst(mf, atmlst):
    if atmlst is None or not hasattr(mf, 'with_df') or mf.with_df is None:
        return None
    atmlst = numpy.asarray(atmlst, dtype=numpy.int32).ravel()
    if atmlst.size == 0:
        return None
    return atmlst

def make_local_mol(mol, atmlst):
    fake_mol = dlno_util.fake_mol_by_atom(mol, atmlst)
    if getattr(mol, 'coords', None) is not None:
        fake_mol.coords = np.asarray(mol.atom_coords()[numpy.asarray(atmlst, dtype=numpy.int32)])
    if getattr(mol, 'exp', None) is not None:
        fake_mol.exp = np.asarray(setup_exp(fake_mol)[0])
    else:
        fake_mol.exp = None
    if getattr(mol, 'ctr_coeff', None) is not None:
        fake_mol.ctr_coeff = np.asarray(setup_ctr_coeff(fake_mol)[0])
    else:
        fake_mol.ctr_coeff = None
    return fake_mol

def get_local_df(mf, atmlst):
    atmlst = tuple(map(int, numpy.asarray(atmlst).ravel()))
    cache = getattr(mf.with_df, '_lno_local_df_cache', None)
    if cache is None:
        cache = {}
        mf.with_df._lno_local_df_cache = cache
    if atmlst in cache:
        return cache[atmlst]

    fake_mol = make_local_mol(mf.mol, atmlst)
    local_df = df_mod.DF(fake_mol, auxbasis=mf.with_df.auxbasis, incore=True)
    local_df.max_memory = mf.with_df.max_memory
    local_df.build()
    ao_idx = dlno_util.ao_index_by_atom(mf.mol, numpy.asarray(atmlst, dtype=numpy.int32))
    cache[atmlst] = (fake_mol, local_df, ao_idx)
    return cache[atmlst]


def _global_pair_indices_for_local_ao(ao_idx, nao):
    ao_idx = numpy.asarray(ao_idx, dtype=numpy.int64).ravel()
    rows, cols = numpy.tril_indices(ao_idx.size)
    grows = ao_idx[rows]
    gcols = ao_idx[cols]
    if numpy.any(grows < gcols):
        raise RuntimeError('Local AO indices must be sorted for packed s2 mapping.')
    idx = grows * (grows + 1) // 2 + gcols
    npair = nao * (nao + 1) // 2
    if idx.size and (idx.min() < 0 or idx.max() >= npair):
        raise RuntimeError('Local AO pair index exceeds global packed CDERI size.')
    return idx


def _outcore_nr_e2_block_mb():
    try:
        return max(float(os.environ.get('PYSCFAD_LNO_OUTCORE_NR_E2_BLOCK_MB', 256.0)), 1.0)
    except ValueError:
        return 256.0


def _outcore_nr_e2_from_source_blocked(cderi_source, mo_coeff, orbs_slice,
                                       aosym='s2', mosym='s1', pair_idx=None):
    """Transform out-of-core CDERI without materializing the full CDERI block."""
    out = None
    with df_addons.load(cderi_source, 'j3c') as eri1:
        if not hasattr(eri1, 'shape'):
            raise NotImplementedError('Unsupported CDERI source for blocked nr_e2.')

        naux = int(eri1.shape[0])
        if pair_idx is None:
            npair = int(eri1.shape[1])
        else:
            pair_idx = numpy.asarray(pair_idx, dtype=numpy.int64)
            if (
                pair_idx.size == int(eri1.shape[1])
                and (pair_idx.size == 0 or (
                    pair_idx[0] == 0
                    and pair_idx[-1] == pair_idx.size - 1
                    and numpy.all(numpy.diff(pair_idx) == 1)
                ))
            ):
                pair_idx = None
                npair = int(eri1.shape[1])
            else:
                npair = int(pair_idx.size)

        target_bytes = _outcore_nr_e2_block_mb() * 1024.0**2
        row_bytes = max(npair, 1) * numpy.dtype(numpy.float64).itemsize
        blksize = max(1, min(naux, int(target_bytes // row_bytes)))

        for p0 in range(0, naux, blksize):
            p1 = min(p0 + blksize, naux)
            if pair_idx is None:
                cderi = numpy.asarray(eri1[p0:p1])
            else:
                cderi = numpy.asarray(eri1[p0:p1, pair_idx])
            block = numpy.asarray(
                _ao2mo.nr_e2(
                    cderi, mo_coeff, orbs_slice, aosym=aosym, mosym=mosym
                )
            )
            if out is None:
                out = numpy.empty((naux,) + block.shape[1:], dtype=block.dtype)
            out[p0:p1] = block
            cderi = block = None
    if out is None:
        return np.empty((0,), dtype=mo_coeff.dtype)
    return np.asarray(out)


def _select_sorted_pair_positions(pair_idx, p0, p1):
    pair_idx = numpy.asarray(pair_idx, dtype=numpy.int64)
    if pair_idx.size == 0:
        return numpy.zeros(0, dtype=numpy.int64)
    if pair_idx.size == 1 or numpy.all(numpy.diff(pair_idx) >= 0):
        i0 = numpy.searchsorted(pair_idx, p0, side='left')
        i1 = numpy.searchsorted(pair_idx, p1, side='left')
        return numpy.arange(i0, i1, dtype=numpy.int64)
    return numpy.nonzero((pair_idx >= p0) & (pair_idx < p1))[0]


def _nr_e2_global_cderi_bar_block(mo_coeff, ybar, orbs_slice, pair_positions,
                                  p0, p1):
    block = numpy.zeros((ybar.shape[0], p1 - p0), dtype=numpy.asarray(ybar).dtype)
    pair_positions = numpy.asarray(pair_positions, dtype=numpy.int64)
    if pair_positions.size == 0:
        return block
    cderi_bar = _cderi_vjp.nr_e2_cderi_bar_packed_block(
        mo_coeff, ybar, orbs_slice, pair_positions
    )
    block[:, pair_positions - p0] = cderi_bar
    return block


def _nr_e2_local_cderi_bar_block(mo_coeff, ybar, orbs_slice, pair_idx, p0, p1):
    block = numpy.zeros((ybar.shape[0], p1 - p0), dtype=numpy.asarray(ybar).dtype)
    local_positions = _select_sorted_pair_positions(pair_idx, p0, p1)
    if local_positions.size == 0:
        return block
    cderi_bar = _cderi_vjp.nr_e2_cderi_bar_packed_block(
        mo_coeff, ybar, orbs_slice, local_positions
    )
    global_positions = numpy.asarray(pair_idx, dtype=numpy.int64)[local_positions]
    block[:, global_positions - p0] = cderi_bar
    return block


def _is_full_global_pair_idx(pair_idx, nao):
    pair_idx = numpy.asarray(pair_idx, dtype=numpy.int64).ravel()
    npair = nao * (nao + 1) // 2
    return (
        pair_idx.size == npair
        and (npair == 0 or (
            pair_idx[0] == 0
            and pair_idx[-1] == npair - 1
            and numpy.all(numpy.diff(pair_idx) == 1)
        ))
    )


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7))
def _outcore_local_nr_e2_from_global_cderi(mol, auxmol, mo_coeff, cderi_source,
                                           max_memory, orbs_slice, aosym,
                                           pair_idx):
    del mol, auxmol, max_memory
    if aosym not in ('s2', 's2ij'):
        raise NotImplementedError
    pair_idx = numpy.asarray(pair_idx, dtype=numpy.int64)
    return _outcore_nr_e2_from_source_blocked(
        cderi_source, mo_coeff, orbs_slice, aosym='s2', pair_idx=pair_idx
    )


def _outcore_local_nr_e2_from_global_cderi_fwd(mol, auxmol, mo_coeff,
                                               cderi_source, max_memory,
                                               orbs_slice, aosym, pair_idx):
    out = _outcore_local_nr_e2_from_global_cderi(
        mol, auxmol, mo_coeff, cderi_source, max_memory, orbs_slice, aosym,
        pair_idx,
    )
    return out, (mol, auxmol, mo_coeff)


def _outcore_local_nr_e2_from_global_cderi_bwd(cderi_source, max_memory,
                                               orbs_slice, aosym, pair_idx,
                                               res, ybar):
    mol, auxmol, mo_coeff = res
    pair_idx = numpy.asarray(pair_idx, dtype=numpy.int64)
    try:
        with _vjp_progress_section('fragment DF AO2MO MO-coeff backward'):
            mo_coeff_bar = _cderi_vjp.nr_e2_mo_coeff_vjp_from_cderi_source(
                cderi_source, mo_coeff, ybar, orbs_slice, aosym='s2',
                pair_idx=pair_idx,
            )

        ybar_np = numpy.asarray(jax.device_get(ybar))
        if _is_full_global_pair_idx(pair_idx, mol.nao):
            try:
                with _vjp_progress_section('fragment DF integral derivative backward'):
                    mol_bar, auxmol_bar = _cderi_vjp.cholesky_eri_vjp_from_mo_coeff_ybar(
                        mol,
                        auxmol,
                        cderi_source,
                        mo_coeff,
                        ybar_np,
                        orbs_slice,
                        max(max_memory, 4096),
                        int3c=mol._add_suffix('int3c2e'),
                        int2c=mol._add_suffix('int2c2e'),
                        aosym='s2ij',
                    )
            except NotImplementedError:
                mol_bar = auxmol_bar = None
            if mol_bar is not None:
                return mol_bar, auxmol_bar, mo_coeff_bar
        with _vjp_progress_section('fragment DF integral derivative backward'):
            mol_bar, auxmol_bar = _cderi_vjp.cholesky_eri_vjp_from_cderi_block_fn(
                mol,
                auxmol,
                cderi_source,
                lambda p0, p1: _nr_e2_local_cderi_bar_block(
                    mo_coeff, ybar_np, orbs_slice, pair_idx, p0, p1
                ),
                max(max_memory, 4096),
                int3c=mol._add_suffix('int3c2e'),
                int2c=mol._add_suffix('int2c2e'),
                aosym='s2ij',
            )
        return mol_bar, auxmol_bar, mo_coeff_bar
    except NotImplementedError:
        pass

    def full_fn(mol_, auxmol_, mo_coeff_):
        cderi = df_incore.cholesky_eri(
            mol_,
            auxmol=auxmol_,
            int3c=mol_._add_suffix('int3c2e'),
            int2c=mol_._add_suffix('int2c2e'),
            max_memory=max(max_memory, 4096),
            verbose=0,
        )
        cderi = cderi[:, pair_idx]
        return _ao2mo.nr_e2(cderi, mo_coeff_, orbs_slice, aosym='s2')

    _, pullback = jax.vjp(full_fn, mol, auxmol, mo_coeff)
    return pullback(ybar)


_outcore_local_nr_e2_from_global_cderi.defvjp(
    _outcore_local_nr_e2_from_global_cderi_fwd,
    _outcore_local_nr_e2_from_global_cderi_bwd,
)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def _outcore_nr_e2(mol, auxmol, mo_coeff, cderi_source, max_memory,
                   orbs_slice, aosym):
    del mol, auxmol, max_memory
    return _outcore_nr_e2_from_source_blocked(
        cderi_source, mo_coeff, orbs_slice, aosym=aosym
    )


def _outcore_nr_e2_fwd(mol, auxmol, mo_coeff, cderi_source, max_memory,
                       orbs_slice, aosym):
    out = _outcore_nr_e2(mol, auxmol, mo_coeff, cderi_source, max_memory,
                         orbs_slice, aosym)
    return out, (mol, auxmol, mo_coeff)


def _outcore_nr_e2_bwd(cderi_source, max_memory, orbs_slice, aosym, res, ybar):
    mol, auxmol, mo_coeff = res
    try:
        with _vjp_progress_section('global DF AO2MO MO-coeff backward'):
            mo_coeff_bar = _cderi_vjp.nr_e2_mo_coeff_vjp_from_cderi_source(
                cderi_source, mo_coeff, ybar, orbs_slice, aosym=aosym
            )

        ybar_np = numpy.asarray(jax.device_get(ybar))
        try:
            with _vjp_progress_section('global DF integral derivative backward'):
                mol_bar, auxmol_bar = _cderi_vjp.cholesky_eri_vjp_from_mo_coeff_ybar(
                    mol,
                    auxmol,
                    cderi_source,
                    mo_coeff,
                    ybar_np,
                    orbs_slice,
                    max(max_memory, 4096),
                    int3c=mol._add_suffix('int3c2e'),
                    int2c=mol._add_suffix('int2c2e'),
                    aosym='s2ij',
                )
        except NotImplementedError:
            with _vjp_progress_section('global DF integral derivative backward'):
                mol_bar, auxmol_bar = _cderi_vjp.cholesky_eri_vjp_from_cderi_block_fn(
                    mol,
                    auxmol,
                    cderi_source,
                    lambda p0, p1: _nr_e2_global_cderi_bar_block(
                        mo_coeff, ybar_np, orbs_slice,
                        numpy.arange(p0, p1, dtype=numpy.int64), p0, p1
                    ),
                    max(max_memory, 4096),
                    int3c=mol._add_suffix('int3c2e'),
                    int2c=mol._add_suffix('int2c2e'),
                    aosym='s2ij',
                )
        return mol_bar, auxmol_bar, mo_coeff_bar
    except NotImplementedError:
        pass

    def fn(mol_, auxmol_, mo_coeff_):
        cderi = df_incore.cholesky_eri(
            mol_,
            auxmol=auxmol_,
            int3c=mol_._add_suffix('int3c2e'),
            int2c=mol_._add_suffix('int2c2e'),
            max_memory=max(max_memory, 4096),
            verbose=0,
        )
        return _ao2mo.nr_e2(cderi, mo_coeff_, orbs_slice, aosym=aosym)

    _, pullback = jax.vjp(fn, mol, auxmol, mo_coeff)
    return pullback(ybar)


_outcore_nr_e2.defvjp(_outcore_nr_e2_fwd, _outcore_nr_e2_bwd)


def transform_df_to_mo(mf, mo_coeff, orbs_slice, aosym='s2', mosym='s1', atmlst=None):
    atmlst = _local_domain_atmlst(mf, atmlst)
    if atmlst is not None:
        ao_idx = dlno_util.ao_index_by_atom(mf.mol, atmlst)
        s1e = mf.get_ovlp()
        s21 = s1e[ao_idx]
        s22 = s1e[np.ix_(ao_idx, ao_idx)]
        mo_coeff = dlno_util.project_mo(mo_coeff, s21, s22)
        get_cderi = getattr(mf.with_df, '_get_cderi_source', None)
        cderi = get_cderi() if get_cderi is not None else mf.with_df._cderi
        has_outcore_cderi = (
            hasattr(mf.with_df, '_has_outcore_cderi_placeholder')
            and mf.with_df._has_outcore_cderi_placeholder()
        )
        if has_outcore_cderi:
            if mf.with_df.auxmol is None:
                mf.with_df.auxmol = df_mod.addons.make_auxmol(
                    mf.with_df.mol, mf.with_df.auxbasis
            )
            pair_idx = tuple(
                _global_pair_indices_for_local_ao(ao_idx, mf.mol.nao).tolist()
            )
            return _outcore_local_nr_e2_from_global_cderi(
                mf.with_df.mol, mf.with_df.auxmol, mo_coeff, cderi,
                mf.with_df.max_memory, orbs_slice, aosym, pair_idx
            )

        fake_mol, local_df, _ = get_local_df(mf, atmlst)
        get_cderi = getattr(local_df, '_get_cderi_source', None)
        cderi = get_cderi() if get_cderi is not None else local_df._cderi
    else:
        get_cderi = getattr(mf.with_df, '_get_cderi_source', None)
        cderi = get_cderi() if get_cderi is not None else mf.with_df._cderi
        has_outcore_cderi = (
            hasattr(mf.with_df, '_has_outcore_cderi_placeholder')
            and mf.with_df._has_outcore_cderi_placeholder()
        )
        if has_outcore_cderi:
            if mf.with_df.auxmol is None:
                mf.with_df.auxmol = df_mod.addons.make_auxmol(
                    mf.with_df.mol, mf.with_df.auxbasis
                )
            return _outcore_nr_e2(
                mf.with_df.mol, mf.with_df.auxmol, mo_coeff, cderi,
                mf.with_df.max_memory, orbs_slice, aosym
            )

    with df_addons.load(cderi, 'j3c') as eri1:
        if not is_array(eri1):
            eri1 = numpy.asarray(eri1)
        return _ao2mo.nr_e2(eri1, mo_coeff, orbs_slice, aosym=aosym, mosym=mosym)

def make_fragment_eris(mfcc, eris, frag_prescreen):
    atmlst = None if frag_prescreen is None else frag_prescreen.get('extended_primary_domain')
    atmlst = _local_domain_atmlst(mfcc._scf, atmlst)
    if atmlst is None:
        return eris

    orbo, orbv = mfcc.split_mo()[1:3]
    nocc = orbo.shape[-1]
    mo_coeff = np.concatenate((orbo, orbv), axis=-1)

    frag_eris = _LNODFINCOREERIS(fock=eris.fock, s1e=eris.s1e)
    frag_eris.Lov = get_Lov(mfcc._scf, mo_coeff, nocc, atmlst=atmlst)
    return frag_eris


def get_Lov(mf, mo_coeff, nocc, atmlst=None):
    assert hasattr(mf, 'with_df')
    nmo = mo_coeff.shape[-1]
    nvir = nmo - nocc
    ijslice = (0, nocc, nocc, nmo)
    Lov = transform_df_to_mo(mf, mo_coeff, ijslice, aosym='s2', mosym='s1', atmlst=atmlst)
    naux = Lov.shape[0]
    Lov = Lov.reshape((naux, nocc, nvir))
    return Lov
