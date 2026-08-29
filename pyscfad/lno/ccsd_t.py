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

'''Impurity (T) correction.
'''

import ctypes
import os
import time
from functools import partial
from types import SimpleNamespace
import numpy
from jax import custom_vjp

from pyscf.lib import (
    logger,
    prange,
    unpack_tril,
    num_threads,
    current_memory,
)

from pyscfad import numpy as np
from pyscfad.cc import ccsd_t as canonical_ccsd_t
from pyscfad.tools import resource_profile
from pyscfadlib import libcc_vjp as libcc


def _profile_enabled():
    value = os.environ.get('PYSCFAD_LNO_CCSD_T_PROFILE', '')
    return value.lower() not in ('', '0', 'false', 'no', 'off')


def _profile_start(label, nocc, nvir, enabled=False):
    if not (enabled or _profile_enabled()):
        return None
    fields = [f'nocc={nocc}', f'nvir={nvir}']
    print(f'    {label}: start ({", ".join(fields)})',
          flush=True)
    return time.perf_counter()


def _profile_done(label, t0):
    if t0 is None:
        return
    print(f'    {label}: done in {time.perf_counter() - t0:.2f} s',
          flush=True)


def _can_use_df_factor_triples(eris, t1, t2):
    """Whether the LNO triples caches can be generated from DF factors."""
    if (eris.ovvv is not None
            or getattr(eris, 'Lov', None) is None
            or getattr(eris, 'Lvv', None) is None):
        return False
    arrays = (t1, t2, eris.fock, eris.mo_energy, eris.ovoo, eris.ovov,
              eris.Lov, eris.Lvv)
    return all(numpy.dtype(x.dtype) == numpy.dtype(numpy.float64)
               for x in arrays)


def _factor_eris_view(Lov, Lvv, ovov):
    """Minimal object consumed by the shared DF ``vvop`` cache helpers."""
    return SimpleNamespace(
        Lov=numpy.asarray(Lov),
        Lvv=numpy.asarray(Lvv),
        ovov=numpy.asarray(ovov),
    )


def _build_factor_cache_pair(factor_eris, p0, p1, nvir):
    """Build ``vvop[p0:p1, :]`` and ``vvop[:, p0:p1]`` from Lov/Lvv."""
    row = canonical_ccsd_t._build_df_vvop_cache(
        factor_eris, p0, p1, 0, nvir)
    if (p0, p1) == (0, nvir):
        col = row
    else:
        col = canonical_ccsd_t._build_df_vvop_cache(
            factor_eris, 0, nvir, p0, p1)
    return row, col


def _accumulate_factor_cache_pair_bar(
        factor_eris, row_bar, col_bar, p0, p1, nvir,
        ovov_bar, Lov_bar, Lvv_bar):
    """Pull a row/column cache pair back to its DF factors."""
    canonical_ccsd_t._accumulate_df_vvop_cache_bar(
        factor_eris, row_bar, p0, p1, 0, nvir,
        ovov_bar, Lov_bar, Lvv_bar)
    if (p0, p1) != (0, nvir):
        canonical_ccsd_t._accumulate_df_vvop_cache_bar(
            factor_eris, col_bar, 0, nvir, p0, p1,
            ovov_bar, Lov_bar, Lvv_bar)


def _ccsd_t_factor_block_nvir(nocc, nvir, max_memory, backward=False):
    """Choose a bounded virtual width for factor-direct LNO triples.

    One rectangular cache of width ``w`` contains
    ``w*nvir*nocc*(nocc+nvir)`` doubles.  The reverse C kernel additionally
    creates cache-bar copies for its OpenMP workers, so its estimate includes
    those thread-private buffers.  This controls the large virtual cache;
    amplitude/bar workspaces inside the triples kernel are separate.
    """
    try:
        block_mb = float(os.environ.get(
            'PYSCFAD_LNO_CCSD_T_FACTOR_BLOCK_MB', 128.0))
    except ValueError:
        block_mb = 128.0

    nmo = nocc + nvir
    one_width_bytes = max(
        nvir * nocc * nmo * numpy.dtype(numpy.float64).itemsize, 1)
    if max_memory <= 0:
        return 1
    # At a disjoint pair of blocks the forward owns four input caches.  The
    # reverse owns those inputs plus four cache bars, with the bars replicated
    # by the C kernel for additional OpenMP workers.
    cache_multiplicity = 4 * (num_threads() + 1) if backward else 4
    target_bytes = max(block_mb, 1.0) * 1024.0**2
    target_bytes = min(target_bytes, max_memory * 1e6 * 0.5)
    width = int(target_bytes // (cache_multiplicity * one_width_bytes))
    return max(1, min(nvir, width))


def kernel(mycc, eris, ulo, t1=None, t2=None, verbose=logger.NOTE):
    log = logger.new_logger(mycc, verbose)
    if t1 is None:
        t1 = mycc.t1
    if t2 is None:
        t2 = mycc.t2
    profile_timing = bool(getattr(mycc, 'lno_ccsd_t_timing', False))

    mat = np.dot(ulo.T, ulo)
    assert mat.shape[0] == t1.shape[0]

    nocc, nvir = t1.shape
    t1T = t1.T
    t2T = t2.transpose(2,3,1,0)
    mo_energy = eris.mo_energy
    fvo = eris.fock[nocc:,:nocc]
    ovoo = eris.ovoo
    ovov = eris.ovov
    use_df_factors = _can_use_df_factor_triples(eris, t1, t2)

    if use_df_factors:
        energy_fn = (
            _ccsd_t_energy_df_lazy
            if getattr(mycc, 'profile_pass', None) == 'backward replay'
            else _ccsd_t_energy_df
        )
        et = energy_fn(
            mat, t1T, t2T, mo_energy, fvo, ovoo, ovov,
            eris.Lov, eris.Lvv, mycc.max_memory, profile_timing)
    else:
        # Dense/unsupported fallback.  Its custom VJP needs a concrete packed
        # ovvv leaf on which to return the cotangent.
        ovvv = (
            eris.get_ovvv_packed() if eris.ovvv is None else eris.ovvv
        )
        energy_fn = (
            _ccsd_t_energy_lazy
            if getattr(mycc, 'profile_pass', None) == 'backward replay'
            else _ccsd_t_energy
        )
        et = energy_fn(
            mat, t1T, t2T, mo_energy, fvo, ovoo, ovov, ovvv,
            mycc.max_memory, profile_timing)

    log.timer('CCSD(T)')
    log.note('CCSD(T) correction = %.15g', et)
    del log
    return et

def get_ovvv(ovvv, *slices):
    ovw = numpy.asarray(ovvv[slices])
    nocc, nvir, nvir_pair = ovw.shape
    ovvv = unpack_tril(ovw.reshape(nocc*nvir,nvir_pair))
    nvir1 = ovvv.shape[2]
    # pylint: disable=too-many-function-args
    return ovvv.reshape(nocc,nvir,nvir1,nvir1)

def _ovvv_unpack_block_nocc(nvir):
    try:
        block_mb = float(os.environ.get('PYSCFAD_LNO_CCSD_T_OVVV_BLOCK_MB', 128.0))
    except ValueError:
        block_mb = 128.0
    bytes_per_occ = max(nvir**3 * numpy.dtype(numpy.float64).itemsize, 1)
    return max(1, int(block_mb * 1024.0**2 // bytes_per_occ))

def _fill_vvop(vvop, ovov, ovvv, nocc, nvir):
    vvop[:,:,:,:nocc] = numpy.asarray(ovov).conj().transpose(1,3,0,2)

    block_nocc = _ovvv_unpack_block_nocc(nvir)
    for i0 in range(0, nocc, block_nocc):
        i1 = min(i0 + block_nocc, nocc)
        ovw = numpy.asarray(ovvv[i0:i1])
        nblk, _, nvir_pair = ovw.shape
        ovvv_block = unpack_tril(ovw.reshape(nblk*nvir, nvir_pair))
        ovvv_block = ovvv_block.reshape(nblk, nvir, nvir, nvir)
        vvop[:,:,i0:i1,nocc:] = ovvv_block.conj().transpose(1,3,0,2)
        ovw = ovvv_block = None

def _ccsd_t_forward_block_nvir(nocc, nvir, max_memory):
    try:
        block_mb = float(os.environ.get('PYSCFAD_LNO_CCSD_T_FWD_BLOCK_MB', 128.0))
    except ValueError:
        block_mb = 128.0

    thread_scratch = (nocc**3*3 + nocc*nvir*2 + 2) * num_threads()
    shared_scratch = nocc**3 * 3
    job_doubles_per_a = nvir * (nvir + 1) / 2 * 7
    base_doubles = thread_scratch + shared_scratch
    avail_doubles = max_memory * 1e6 / 8

    block_from_memory = int((avail_doubles - base_doubles) // job_doubles_per_a)
    block_from_target = int(block_mb * 1024.0**2 / 8 // job_doubles_per_a)
    return max(1, min(nvir, block_from_memory, max(1, block_from_target)))


@partial(custom_vjp, nondiff_argnums=(9, 10))
def _ccsd_t_energy_df(mat, t1T, t2T, mo_energy, fvo,
                      ovoo, ovov, Lov, Lvv, max_memory,
                      profile_timing=False):
    """Factor-direct impurity (T) energy.

    Rectangular ``vvop`` caches are generated from ``Lov`` and packed
    ``Lvv`` for one virtual-pair block at a time.  No packed ``ovvv`` or
    persistent global ``vvop`` tensor is constructed.  For a small fragment,
    one cache may cover its entire virtual range when that fits the cap.
    """
    del profile_timing
    nvir, nocc = t1T.shape
    resource_start = resource_profile.start()
    profile_t0 = _profile_start('forward factor-direct (T)', nocc, nvir)

    mat = numpy.asarray(mat, order='C')
    t1T = numpy.asarray(t1T, order='C')
    t2T = numpy.asarray(t2T, order='C')
    mo_energy = numpy.asarray(mo_energy, order='C')
    fvo = numpy.asarray(fvo, order='C')
    vooo = numpy.asarray(ovoo).conj().transpose(1, 0, 3, 2)
    vooo = numpy.asarray(vooo, order='C')
    factor_eris = _factor_eris_view(Lov, Lvv, ovov)

    drv = libcc.lnoccsdt_contract_df
    et_sum = numpy.zeros(1, dtype=float)

    def contract(b0, b1, c0, c1, cache):
        cache_row_b, cache_col_b, cache_row_c, cache_col_c = cache
        drv(et_sum.ctypes.data_as(ctypes.c_void_p),
            mat.ctypes.data_as(ctypes.c_void_p),
            mo_energy.ctypes.data_as(ctypes.c_void_p),
            t1T.ctypes.data_as(ctypes.c_void_p),
            t2T.ctypes.data_as(ctypes.c_void_p),
            vooo.ctypes.data_as(ctypes.c_void_p),
            fvo.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nocc), ctypes.c_int(nvir),
            ctypes.c_int(b0), ctypes.c_int(b1),
            ctypes.c_int(c0), ctypes.c_int(c1),
            cache_row_b.ctypes.data_as(ctypes.c_void_p),
            cache_col_b.ctypes.data_as(ctypes.c_void_p),
            cache_row_c.ctypes.data_as(ctypes.c_void_p),
            cache_col_c.ctypes.data_as(ctypes.c_void_p))

    mem_now = current_memory()[0]
    available_memory = max(0, max_memory - mem_now)
    bufsize = _ccsd_t_factor_block_nvir(
        nocc, nvir, available_memory, backward=False)
    peak_cache_bytes = 0

    for b0, b1 in reversed(list(prange(0, nvir, bufsize))):
        cache_row_b, cache_col_b = _build_factor_cache_pair(
            factor_eris, b0, b1, nvir)
        peak_cache_bytes = max(
            peak_cache_bytes,
            cache_row_b.nbytes
            + (0 if cache_col_b is cache_row_b else cache_col_b.nbytes),
        )
        contract(
            b0, b1, b0, b1,
            (cache_row_b, cache_col_b, cache_row_b, cache_col_b),
        )

        inner_bufsize = max(1, bufsize // 4)
        for c0, c1 in prange(0, b0, inner_bufsize):
            cache_row_c, cache_col_c = _build_factor_cache_pair(
                factor_eris, c0, c1, nvir)
            peak_cache_bytes = max(
                peak_cache_bytes,
                cache_row_b.nbytes + cache_col_b.nbytes
                + cache_row_c.nbytes + cache_col_c.nbytes,
            )
            contract(
                b0, b1, c0, c1,
                (cache_row_b, cache_col_b, cache_row_c, cache_col_c),
            )
            cache_row_c = cache_col_c = None
        cache_row_b = cache_col_b = None

    et_sum *= 2. / 3.
    et = et_sum[0].real
    _profile_done('forward factor-direct (T)', profile_t0)
    resource_profile.finish(
        'triples.forward_factor_direct',
        resource_start,
        nocc=nocc,
        nvir=nvir,
        threads=num_threads(),
        configured_memory_mib=max_memory,
        pyscf_current_memory_mib=mem_now,
        virtual_block=bufsize,
        peak_python_cache_mib=peak_cache_bytes / 1024.0**2,
        persistent_global_ovvv=False,
        persistent_global_vvop=False,
    )
    return et


def _ccsd_t_energy_df_fwd(mat, t1T, t2T, mo_energy, fvo,
                          ovoo, ovov, Lov, Lvv, max_memory,
                          profile_timing=False):
    et = _ccsd_t_energy_df(
        mat, t1T, t2T, mo_energy, fvo, ovoo, ovov, Lov, Lvv,
        max_memory, profile_timing)
    return et, (mat, t1T, t2T, mo_energy, fvo, ovoo, ovov, Lov, Lvv)


def _ccsd_t_energy_df_bwd(max_memory, profile_timing, res, et_bar):
    mat, t1T, t2T, mo_energy, fvo, ovoo, ovov, Lov, Lvv = res
    nvir, nocc = t1T.shape
    nmo = nocc + nvir
    resource_start = resource_profile.start()
    profile_t0 = _profile_start(
        'backward factor-direct (T)', nocc, nvir,
        enabled=profile_timing)

    et_bar *= 2. / 3.

    mat = numpy.asarray(mat, order='C')
    mat_bar = numpy.zeros_like(mat)
    t1T = numpy.asarray(t1T, order='C')
    t1T_bar = numpy.zeros_like(t1T)
    t2T = numpy.asarray(t2T, order='C')
    t2T_bar = numpy.zeros_like(t2T)
    mo_energy = numpy.asarray(mo_energy, order='C')
    mo_energy_bar = numpy.zeros_like(mo_energy)
    fvo = numpy.asarray(fvo, order='C')
    fvo_bar = numpy.zeros_like(fvo)
    vooo = numpy.asarray(ovoo).conj().transpose(1, 0, 3, 2)
    vooo = numpy.asarray(vooo, order='C')
    vooo_bar = numpy.zeros_like(vooo)

    factor_eris = _factor_eris_view(Lov, Lvv, ovov)
    ovov_bar = numpy.zeros_like(factor_eris.ovov)
    Lov_bar = numpy.zeros_like(factor_eris.Lov)
    Lvv_bar = numpy.zeros_like(factor_eris.Lvv)

    drv = libcc.lnoccsdt_energy_vjp

    def contract(b0, b1, c0, c1, cache, cache_bar):
        cache_row_b, cache_col_b, cache_row_c, cache_col_c = cache
        (cache_row_b_bar, cache_col_b_bar,
         cache_row_c_bar, cache_col_c_bar) = cache_bar
        drv(mat.ctypes.data_as(ctypes.c_void_p),
            mo_energy.ctypes.data_as(ctypes.c_void_p),
            t1T.ctypes.data_as(ctypes.c_void_p),
            t2T.ctypes.data_as(ctypes.c_void_p),
            vooo.ctypes.data_as(ctypes.c_void_p),
            fvo.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(et_bar),
            ctypes.c_int(nocc), ctypes.c_int(nvir),
            ctypes.c_int(b0), ctypes.c_int(b1),
            ctypes.c_int(c0), ctypes.c_int(c1),
            cache_row_b.ctypes.data_as(ctypes.c_void_p),
            cache_col_b.ctypes.data_as(ctypes.c_void_p),
            cache_row_c.ctypes.data_as(ctypes.c_void_p),
            cache_col_c.ctypes.data_as(ctypes.c_void_p),
            mat_bar.ctypes.data_as(ctypes.c_void_p),
            mo_energy_bar.ctypes.data_as(ctypes.c_void_p),
            t1T_bar.ctypes.data_as(ctypes.c_void_p),
            t2T_bar.ctypes.data_as(ctypes.c_void_p),
            vooo_bar.ctypes.data_as(ctypes.c_void_p),
            fvo_bar.ctypes.data_as(ctypes.c_void_p),
            cache_row_b_bar.ctypes.data_as(ctypes.c_void_p),
            cache_col_b_bar.ctypes.data_as(ctypes.c_void_p),
            cache_row_c_bar.ctypes.data_as(ctypes.c_void_p),
            cache_col_c_bar.ctypes.data_as(ctypes.c_void_p))

    mem_now = current_memory()[0]
    available_memory = max(0, max_memory - mem_now)
    bufsize = _ccsd_t_factor_block_nvir(
        nocc, nvir, available_memory, backward=True)
    peak_cache_bytes = 0

    for b0, b1 in reversed(list(prange(0, nvir, bufsize))):
        cache_row_b, cache_col_b = _build_factor_cache_pair(
            factor_eris, b0, b1, nvir)
        cache_row_b_bar = numpy.zeros_like(cache_row_b)
        cache_col_b_bar = (
            cache_row_b_bar if cache_col_b is cache_row_b
            else numpy.zeros_like(cache_col_b)
        )
        peak_cache_bytes = max(
            peak_cache_bytes,
            2 * cache_row_b.nbytes
            + (0 if cache_col_b is cache_row_b else 2 * cache_col_b.nbytes),
        )
        contract(
            b0, b1, b0, b1,
            (cache_row_b, cache_col_b, cache_row_b, cache_col_b),
            (cache_row_b_bar, cache_col_b_bar,
             cache_row_b_bar, cache_col_b_bar),
        )

        inner_bufsize = max(1, bufsize // 4)
        for c0, c1 in prange(0, b0, inner_bufsize):
            cache_row_c, cache_col_c = _build_factor_cache_pair(
                factor_eris, c0, c1, nvir)
            cache_row_c_bar = numpy.zeros_like(cache_row_c)
            cache_col_c_bar = numpy.zeros_like(cache_col_c)
            peak_cache_bytes = max(
                peak_cache_bytes,
                2 * (cache_row_b.nbytes + cache_col_b.nbytes
                     + cache_row_c.nbytes + cache_col_c.nbytes),
            )
            contract(
                b0, b1, c0, c1,
                (cache_row_b, cache_col_b, cache_row_c, cache_col_c),
                (cache_row_b_bar, cache_col_b_bar,
                 cache_row_c_bar, cache_col_c_bar),
            )
            _accumulate_factor_cache_pair_bar(
                factor_eris, cache_row_c_bar, cache_col_c_bar,
                c0, c1, nvir, ovov_bar, Lov_bar, Lvv_bar)
            cache_row_c = cache_col_c = None
            cache_row_c_bar = cache_col_c_bar = None

        _accumulate_factor_cache_pair_bar(
            factor_eris, cache_row_b_bar, cache_col_b_bar,
            b0, b1, nvir, ovov_bar, Lov_bar, Lvv_bar)
        cache_row_b = cache_col_b = None
        cache_row_b_bar = cache_col_b_bar = None

    ovoo_bar = numpy.asarray(vooo_bar.transpose(1, 0, 3, 2))
    _profile_done('backward factor-direct (T)', profile_t0)
    one_cache_bytes = (
        bufsize * nvir * nocc * nmo
        * numpy.dtype(numpy.float64).itemsize)
    resource_profile.finish(
        'triples.backward_factor_direct',
        resource_start,
        nocc=nocc,
        nvir=nvir,
        threads=num_threads(),
        configured_memory_mib=max_memory,
        pyscf_current_memory_mib=mem_now,
        virtual_block=bufsize,
        peak_python_cache_and_bar_mib=peak_cache_bytes / 1024.0**2,
        estimated_thread_private_cache_bar_mib=(
            4 * max(0, num_threads() - 1) * one_cache_bytes / 1024.0**2
        ),
        persistent_global_ovvv=False,
        persistent_global_vvop_bar=False,
    )
    return (mat_bar, t1T_bar, t2T_bar, mo_energy_bar, fvo_bar,
            ovoo_bar, ovov_bar, Lov_bar, Lvv_bar)


_ccsd_t_energy_df.defvjp(
    _ccsd_t_energy_df_fwd,
    _ccsd_t_energy_df_bwd,
)


@partial(custom_vjp, nondiff_argnums=(9, 10))
def _ccsd_t_energy_df_lazy(mat, t1T, t2T, mo_energy, fvo,
                           ovoo, ovov, Lov, Lvv, max_memory,
                           profile_timing=False):
    """Backward-replay variant of factor-direct impurity (T)."""
    del profile_timing
    return numpy.zeros((), dtype=t1T.dtype)


def _ccsd_t_energy_df_lazy_fwd(mat, t1T, t2T, mo_energy, fvo,
                               ovoo, ovov, Lov, Lvv, max_memory,
                               profile_timing=False):
    del profile_timing
    et = numpy.zeros((), dtype=t1T.dtype)
    return et, (mat, t1T, t2T, mo_energy, fvo, ovoo, ovov, Lov, Lvv)


_ccsd_t_energy_df_lazy.defvjp(
    _ccsd_t_energy_df_lazy_fwd,
    _ccsd_t_energy_df_bwd,
)

@partial(custom_vjp, nondiff_argnums=(8, 9))
def _ccsd_t_energy(mat, t1T, t2T, mo_energy, fvo,
                   ovoo, ovov, ovvv, max_memory, profile_timing=False):
    del profile_timing
    nvir, nocc = t1T.shape
    nmo = nocc + nvir
    resource_start = resource_profile.start()
    profile_t0 = _profile_start('forward (T)', nocc, nvir)

    mat = numpy.asarray(mat, order='C')
    t1T = numpy.asarray(t1T, order='C')
    t2T = numpy.asarray(t2T, order='C')
    mo_energy = numpy.asarray(mo_energy, order='C')
    fvo = numpy.asarray(fvo, order='C')

    vooo = numpy.asarray(ovoo).conj().transpose(1,0,3,2)
    vooo = numpy.asarray(vooo, order='C')

    vvop = numpy.empty((nvir,nvir,nocc,nmo))
    _fill_vvop(vvop, ovov, ovvv, nocc, nvir)
    vvop = numpy.asarray(vvop, order='C')

    drv = libcc.lnoccsdt_contract
    et_sum = numpy.zeros(1, dtype=float)
    def contract(a0, a1, cache):
        drv(et_sum.ctypes.data_as(ctypes.c_void_p),
            mat.ctypes.data_as(ctypes.c_void_p),
            mo_energy.ctypes.data_as(ctypes.c_void_p),
            t1T.ctypes.data_as(ctypes.c_void_p),
            t2T.ctypes.data_as(ctypes.c_void_p),
            vooo.ctypes.data_as(ctypes.c_void_p),
            fvo.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nocc), ctypes.c_int(nvir),
            ctypes.c_int(a0), ctypes.c_int(a1),
            cache.ctypes.data_as(ctypes.c_void_p))

    mem_now = current_memory()[0]
    max_memory = max(0, max_memory - mem_now)
    bufsize = _ccsd_t_forward_block_nvir(nocc, nvir, max_memory)
    for a0, a1 in prange(0, nvir, bufsize):
        contract(a0, a1, vvop)

    et_sum *= 2. / 3.
    et = et_sum[0].real
    _profile_done('forward (T)', profile_t0)
    resource_profile.finish(
        'triples.forward_contraction',
        resource_start,
        nocc=nocc,
        nvir=nvir,
        threads=num_threads(),
        configured_memory_mib=max_memory + mem_now,
        pyscf_current_memory_mib=mem_now,
        virtual_block=bufsize,
        vvop_mib=resource_profile.estimated_array_mib(vvop),
    )
    return et


def _ccsd_t_energy_fwd(mat, t1T, t2T, mo_energy, fvo,
                       ovoo, ovov, ovvv, max_memory, profile_timing=False):
    et = _ccsd_t_energy(mat, t1T, t2T, mo_energy, fvo,
                        ovoo, ovov, ovvv, max_memory, profile_timing)
    return et, (mat, t1T, t2T, mo_energy, fvo, ovoo, ovov, ovvv)

def _ccsd_t_energy_bwd(max_memory, profile_timing, res, et_bar):
    mat, t1T, t2T, mo_energy, fvo, ovoo, ovov, ovvv = res

    nvir, nocc = t1T.shape
    nmo = nocc + nvir
    resource_start = resource_profile.start()
    profile_t0 = _profile_start('backward (T)', nocc, nvir,
                                enabled=profile_timing)

    et_bar *= 2. / 3.

    mat = numpy.asarray(mat, order='C')
    mat_bar = numpy.zeros_like(mat)

    t1T = numpy.asarray(t1T, order='C')
    t1T_bar = numpy.zeros_like(t1T)
    t2T = numpy.asarray(t2T, order='C')
    t2T_bar = numpy.zeros_like(t2T)

    mo_energy = numpy.asarray(mo_energy, order='C')
    mo_energy_bar = numpy.zeros_like(mo_energy)
    fvo = numpy.asarray(fvo, order='C')
    fvo_bar = numpy.zeros_like(fvo)

    vooo = numpy.asarray(ovoo).conj().transpose(1,0,3,2)
    vooo = numpy.asarray(vooo, order='C')
    vooo_bar = numpy.zeros_like(vooo)

    vvop = numpy.empty((nvir,nvir,nocc,nmo))
    _fill_vvop(vvop, ovov, ovvv, nocc, nvir)
    vvop = numpy.asarray(vvop, order='C')
    vvop_bar = numpy.zeros_like(vvop)

    drv = libcc.lnoccsdt_energy_vjp
    def contract(a0, a1, b0, b1, cache, cache_bar):
        cache_row_a, cache_col_a, cache_row_b, cache_col_b = cache
        cache_row_a_bar, cache_col_a_bar, cache_row_b_bar, cache_col_b_bar = cache_bar
        drv(mat.ctypes.data_as(ctypes.c_void_p),
            mo_energy.ctypes.data_as(ctypes.c_void_p),
            t1T.ctypes.data_as(ctypes.c_void_p),
            t2T.ctypes.data_as(ctypes.c_void_p),
            vooo.ctypes.data_as(ctypes.c_void_p),
            fvo.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(et_bar),
            ctypes.c_int(nocc), ctypes.c_int(nvir),
            ctypes.c_int(a0), ctypes.c_int(a1),
            ctypes.c_int(b0), ctypes.c_int(b1),
            cache_row_a.ctypes.data_as(ctypes.c_void_p),
            cache_col_a.ctypes.data_as(ctypes.c_void_p),
            cache_row_b.ctypes.data_as(ctypes.c_void_p),
            cache_col_b.ctypes.data_as(ctypes.c_void_p),
            mat_bar.ctypes.data_as(ctypes.c_void_p),
            mo_energy_bar.ctypes.data_as(ctypes.c_void_p),
            t1T_bar.ctypes.data_as(ctypes.c_void_p),
            t2T_bar.ctypes.data_as(ctypes.c_void_p),
            vooo_bar.ctypes.data_as(ctypes.c_void_p),
            fvo_bar.ctypes.data_as(ctypes.c_void_p),
            cache_row_a_bar.ctypes.data_as(ctypes.c_void_p),
            cache_col_a_bar.ctypes.data_as(ctypes.c_void_p),
            cache_row_b_bar.ctypes.data_as(ctypes.c_void_p),
            cache_col_b_bar.ctypes.data_as(ctypes.c_void_p))

    mem_now = current_memory()[0]
    max_memory = max(0, max_memory - mem_now)

    min_memory = nocc**3*3+nvir*nocc*2
    min_memory+= (nmo*(nocc+1) + nvir*nocc*(nvir*nocc+1) + (nvir+6)*nocc**3 + 2) * (num_threads()-1)
    min_memory*= 8./1e6
    bufsize = (max_memory - min_memory)*1e6/8/num_threads()/(nocc*nmo*nvir+nvir)/2
    bufsize *= .8
    bufsize = int(max(8, bufsize))

    for a0, a1 in reversed(list(prange(0, nvir, bufsize))):
        full_vvop_block = (a0, a1) == (0, nvir)
        cache_row_a = numpy.asarray(vvop[a0:a1,:], order='C')
        if full_vvop_block:
            cache_row_a_bar = vvop_bar
            cache_col_a = cache_row_a
            cache_col_a_bar = cache_row_a_bar
        else:
            cache_row_a_bar = numpy.zeros_like(cache_row_a)
            cache_col_a = numpy.asarray(vvop[:,a0:a1], order='C')
            cache_col_a_bar = numpy.zeros_like(cache_col_a)
        contract(a0, a1, a0, a1,
                (cache_row_a, cache_col_a, cache_row_a, cache_col_a),
                (cache_row_a_bar, cache_col_a_bar, cache_row_a_bar, cache_col_a_bar))

        for b0, b1 in prange(0, a0, bufsize//4):
            cache_row_b = numpy.asarray(vvop[b0:b1,:], order='C')
            cache_row_b_bar = numpy.zeros_like(cache_row_b)
            cache_col_b = numpy.asarray(vvop[:,b0:b1], order='C')
            cache_col_b_bar = numpy.zeros_like(cache_col_b)
            contract(a0, a1, b0, b1,
                    (cache_row_a, cache_col_a, cache_row_b, cache_col_b),
                    (cache_row_a_bar, cache_col_a_bar, cache_row_b_bar, cache_col_b_bar))

            vvop_bar[b0:b1,:] += cache_row_b_bar
            vvop_bar[:,b0:b1] += cache_col_b_bar

        if not full_vvop_block:
            vvop_bar[a0:a1,:] += cache_row_a_bar
            vvop_bar[:,a0:a1] += cache_col_a_bar

    ovoo_bar = numpy.asarray(vooo_bar.transpose(1,0,3,2))
    ovov_bar = numpy.asarray(vvop_bar[:,:,:,:nocc].transpose(2,0,3,1))
    ovvv_bar = vvop_bar[:,:,:,nocc:].transpose(2,0,3,1)
    ovvv_bar += ovvv_bar.transpose(0,1,3,2)
    idx, idy = numpy.diag_indices(nvir)
    ovvv_bar[:,:,idx,idy] *= .5
    idx, idy = numpy.tril_indices(nvir)
    ovvv_tril_bar = numpy.asarray(ovvv_bar[:,:,idx,idy])

    _profile_done('backward (T)', profile_t0)
    resource_profile.finish(
        'triples.backward_contraction',
        resource_start,
        nocc=nocc,
        nvir=nvir,
        threads=num_threads(),
        configured_memory_mib=max_memory + mem_now,
        pyscf_current_memory_mib=mem_now,
        virtual_block=bufsize,
        vvop_and_bar_mib=resource_profile.estimated_array_mib(
            vvop, vvop_bar
        ),
    )
    return mat_bar, t1T_bar, t2T_bar, mo_energy_bar, fvo_bar, ovoo_bar, ovov_bar, ovvv_tril_bar

_ccsd_t_energy.defvjp(_ccsd_t_energy_fwd, _ccsd_t_energy_bwd)


@partial(custom_vjp, nondiff_argnums=(8, 9))
def _ccsd_t_energy_lazy(mat, t1T, t2T, mo_energy, fvo,
                        ovoo, ovov, ovvv, max_memory,
                        profile_timing=False):
    """Backward-replay variant of :func:`_ccsd_t_energy`.

    The (T) forward contraction is skipped and a zero primal is returned;
    the saved residuals are identical to those of the real forward so the
    shared backward kernel produces correct cotangents.  Only safe to call
    when the caller will discard the primal output (e.g. inside
    ``jax.vjp(frag_fn, ...)`` during DLNO replay).
    """
    del profile_timing
    return numpy.zeros((), dtype=t1T.dtype)


def _ccsd_t_energy_lazy_fwd(mat, t1T, t2T, mo_energy, fvo,
                            ovoo, ovov, ovvv, max_memory,
                            profile_timing=False):
    del profile_timing
    et = numpy.zeros((), dtype=t1T.dtype)
    return et, (mat, t1T, t2T, mo_energy, fvo, ovoo, ovov, ovvv)


_ccsd_t_energy_lazy.defvjp(_ccsd_t_energy_lazy_fwd, _ccsd_t_energy_bwd)
