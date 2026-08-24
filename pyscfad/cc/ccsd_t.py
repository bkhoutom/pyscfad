# Copyright 2021-2025 Xing Zhang
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

import ctypes
import time
import numpy
from jax import custom_vjp
from jax.tree_util import tree_flatten_with_path, tree_unflatten
from pyscf.lib import (
    prange_tril,
    num_threads,
    current_memory,
)
from pyscf.cc import ccsd_t as pyscf_ccsd_t
from pyscf.cc import _ccsd as pyscf_ccsd_lib
from pyscfad.lib import logger
from pyscfadlib import libcc_vjp as libcc


def _can_use_df_factor_triples(mycc, eris, t1, t2):
    """Whether the real, symmetry-free DF cache path can be used.

    The PySCF and pyscfad C triples kernels consume rectangular ``vvop``
    caches.  For DF ERIs those caches can be generated directly from
    ``Lov`` and packed ``Lvv``.  Symmetry sorting and complex-valued forward
    energies are deliberately left on the established dense path.  The
    pyscfad triples pullback itself is a real-double C kernel, so reverse mode
    rejects other dtypes explicitly instead of reinterpreting their buffers.
    """
    if (eris.ovvv is not None
            or getattr(eris, 'Lov', None) is None
            or getattr(eris, 'Lvv', None) is None
            or mycc.mol.symmetry):
        return False
    arrays = (t1, t2, eris.fock, eris.mo_energy, eris.ovoo, eris.ovov,
              eris.Lov, eris.Lvv)
    return all(numpy.dtype(x.dtype) == numpy.dtype(numpy.float64)
               for x in arrays)


def _require_supported_triples_vjp(mycc, eris, t1, t2):
    """Reject buffers unsupported by the native C1/double pullback."""
    if getattr(mycc.mol, 'symmetry', False):
        raise NotImplementedError(
            "the CCSD(T) custom VJP supports C1 (symmetry-disabled) "
            "calculations only"
        )
    arrays = [t1, t2, eris.fock, eris.mo_energy, eris.ovoo, eris.ovov]
    arrays.extend(
        value for value in (
            getattr(eris, 'ovvv', None), getattr(eris, 'Lov', None),
            getattr(eris, 'Lvv', None),
        ) if value is not None
    )
    if not all(numpy.dtype(x.dtype) == numpy.dtype(numpy.float64)
               for x in arrays):
        raise NotImplementedError(
            "the CCSD(T) custom VJP supports real float64 tensors only"
        )


def _tril_pair_indices(first, second):
    """Packed-tril indices for all pairs in two one-dimensional arrays."""
    first = numpy.asarray(first, dtype=numpy.intp)[:, None]
    second = numpy.asarray(second, dtype=numpy.intp)[None, :]
    hi = numpy.maximum(first, second)
    lo = numpy.minimum(first, second)
    return hi * (hi + 1) // 2 + lo


def _df_virtual_blocksize(naux, ncol, nvir, itemsize=8):
    """Limit the temporary ``Lvv[:, c, b]`` gather to about 64 MiB."""
    denom = max(1, naux * max(1, ncol) * itemsize)
    return max(1, min(nvir, (64 << 20) // denom))


def _build_df_vvop_cache(eris, row0, row1, col0, col1):
    """Build one C-kernel ``vvop`` cache directly from DF factors.

    The returned block has shape ``(row, col, occ, mo)`` and is identical to
    ``vvop[row0:row1, col0:col1]`` for

    ``vvop[a,b,i,j] = (ia|jb)`` and
    ``vvop[a,b,i,nocc+c] = (ia|cb)``.

    Only the returned cache and a bounded gather from packed ``Lvv`` are
    materialized; neither global ``ovvv`` nor global ``vvop`` is formed.
    """
    Lov = numpy.asarray(eris.Lov)
    Lvv = numpy.asarray(eris.Lvv)
    ovov = numpy.asarray(eris.ovov)
    naux, nocc, nvir = Lov.shape
    nmo = nocc + nvir
    nrow = row1 - row0
    ncol = col1 - col0
    dtype = numpy.result_type(Lov, Lvv, ovov)
    cache = numpy.empty((nrow, ncol, nocc, nmo), dtype=dtype)

    # vvop[a,b,i,j] = ovov[i,a,j,b]
    ovov_block = ovov[:, row0:row1, :, col0:col1]
    cache[:, :, :, :nocc] = ovov_block.conj().transpose(1, 3, 0, 2)

    # vvop[a,b,i,nocc+c] = sum_x Lov[x,i,a] Lvv[x,c,b].
    # Gather c in bounded tiles because unpacking all of Lvv would replace
    # the tensor we are trying to avoid with an naux*nvir*nvir temporary.
    rows = numpy.arange(row0, row1)
    cols = numpy.arange(col0, col1)
    Lov_rows = Lov[:, :, rows]
    cblksize = _df_virtual_blocksize(
        naux, ncol, nvir, numpy.dtype(dtype).itemsize)
    for c0 in range(0, nvir, cblksize):
        c1 = min(c0 + cblksize, nvir)
        pair = _tril_pair_indices(numpy.arange(c0, c1), cols)
        Lcb = Lvv[:, pair]
        cache[:, :, :, nocc+c0:nocc+c1] = numpy.einsum(
            'xia,xcb->abic', Lov_rows, Lcb, optimize=True).conj()
    return numpy.asarray(cache, order='C')


def _accumulate_df_vvop_cache_bar(eris, cache_bar,
                                   row0, row1, col0, col1,
                                   ovov_bar, Lov_bar, Lvv_bar):
    """Pull one ``vvop`` cache cotangent back to ``ovov/Lov/Lvv``."""
    Lov = numpy.asarray(eris.Lov)
    Lvv = numpy.asarray(eris.Lvv)
    naux, nocc, nvir = Lov.shape

    occ_bar = cache_bar[:, :, :, :nocc].transpose(2, 0, 3, 1)
    ovov_bar[:, row0:row1, :, col0:col1] += occ_bar

    rows = numpy.arange(row0, row1)
    cols = numpy.arange(col0, col1)
    Lov_rows = Lov[:, :, rows]
    cblksize = _df_virtual_blocksize(
        naux, col1-col0, nvir, numpy.dtype(Lvv.dtype).itemsize)
    for c0 in range(0, nvir, cblksize):
        c1 = min(c0 + cblksize, nvir)
        pair = _tril_pair_indices(numpy.arange(c0, c1), cols)
        Lcb = Lvv[:, pair]
        block_bar = cache_bar[:, :, :, nocc+c0:nocc+c1]
        Lov_bar[:, :, row0:row1] += numpy.einsum(
            'abic,xcb->xia', block_bar, Lcb, optimize=True)
        Lcb_bar = numpy.einsum(
            'abic,xia->xcb', block_bar, Lov_rows, optimize=True)
        # ``pair`` can contain repeated packed indices (c,b) and (b,c), so
        # ordinary advanced-index += would lose contributions.
        numpy.add.at(Lvv_bar, (slice(None), pair.ravel()),
                     Lcb_bar.reshape(naux, -1))


def _ccsd_t_energy_df(mycc, eris, t1, t2, max_memory):
    """Symmetry-free real DF (T) energy without global ``ovvv``/``vvop``."""
    t1 = numpy.asarray(t1)
    t2 = numpy.asarray(t2)
    nocc, nvir = t1.shape
    nmo = nocc + nvir

    mo_energy = numpy.asarray(eris.mo_energy, dtype=numpy.double, order='C')
    t1T = numpy.asarray(t1.T, dtype=numpy.double, order='C')
    t2T = numpy.asarray(t2.transpose(2, 3, 1, 0),
                       dtype=numpy.double, order='C')
    vooo = numpy.asarray(numpy.asarray(eris.ovoo).conj().transpose(1, 0, 3, 2),
                         dtype=numpy.double, order='C')
    fvo = numpy.asarray(eris.fock[nocc:, :nocc],
                        dtype=numpy.double, order='C')

    # No molecular symmetry: every orbital belongs to irrep zero.  These
    # arrays reproduce the metadata passed by pyscf.cc.ccsd_t.kernel.
    orbsym = numpy.zeros(nmo, dtype=numpy.int32)
    o_ir_loc = numpy.array([0] + [nocc] * 8, dtype=numpy.int32)
    v_ir_loc = numpy.array([0] + [nvir] * 8, dtype=numpy.int32)
    oo_ir_loc = numpy.array([0] + [nocc*nocc] * 8, dtype=numpy.int32)
    nirrep = 1

    et_sum = numpy.zeros(1, dtype=numpy.double)
    drv = pyscf_ccsd_lib.libcc.CCsd_t_contract

    def contract(a0, a1, b0, b1, cache):
        cache_row_a, cache_col_a, cache_row_b, cache_col_b = cache
        drv(et_sum.ctypes.data_as(ctypes.c_void_p),
            mo_energy.ctypes.data_as(ctypes.c_void_p),
            t1T.ctypes.data_as(ctypes.c_void_p),
            t2T.ctypes.data_as(ctypes.c_void_p),
            vooo.ctypes.data_as(ctypes.c_void_p),
            fvo.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nocc), ctypes.c_int(nvir),
            ctypes.c_int(a0), ctypes.c_int(a1),
            ctypes.c_int(b0), ctypes.c_int(b1),
            ctypes.c_int(nirrep),
            o_ir_loc.ctypes.data_as(ctypes.c_void_p),
            v_ir_loc.ctypes.data_as(ctypes.c_void_p),
            oo_ir_loc.ctypes.data_as(ctypes.c_void_p),
            orbsym.ctypes.data_as(ctypes.c_void_p),
            cache_row_a.ctypes.data_as(ctypes.c_void_p),
            cache_col_a.ctypes.data_as(ctypes.c_void_p),
            cache_row_b.ctypes.data_as(ctypes.c_void_p),
            cache_col_b.ctypes.data_as(ctypes.c_void_p))

    mem_now = current_memory()[0]
    avail_memory = max(0, max_memory - mem_now)
    bufsize = (avail_memory*.5e6/8 - nocc**3*3*num_threads()) / (nocc*nmo)
    bufsize *= .5
    bufsize *= .8
    bufsize = max(8, bufsize)

    for a0, a1 in reversed(list(prange_tril(0, nvir, bufsize))):
        cache_row_a = _build_df_vvop_cache(eris, a0, a1, 0, a1)
        cache_col_a = (cache_row_a if a0 == 0 else
                       _build_df_vvop_cache(eris, 0, a0, a0, a1))
        contract(a0, a1, a0, a1,
                 (cache_row_a, cache_col_a, cache_row_a, cache_col_a))

        for b0, b1 in prange_tril(0, a0, bufsize/8):
            cache_row_b = _build_df_vvop_cache(eris, b0, b1, 0, b1)
            cache_col_b = (cache_row_b if b0 == 0 else
                           _build_df_vvop_cache(eris, 0, b0, b0, b1))
            contract(a0, a1, b0, b1,
                     (cache_row_a, cache_col_a,
                      cache_row_b, cache_col_b))

    return float(et_sum[0] * 2)


def kernel(mycc, eris, t1=None, t2=None, verbose=logger.NOTE):
    if t1 is None:
        t1 = mycc.t1
    if t2 is None:
        t2 = mycc.t2

    max_memory = mycc.max_memory
    use_df_factors = _can_use_df_factor_triples(mycc, eris, t1, t2)

    # Keep the established dense forward/symmetry path.  It still requires an
    # ovvv pytree leaf so that the custom pullback has somewhere to attach its
    # cotangent.  Reverse mode rejects symmetry above until its sorting
    # pullback is implemented.  The common real, C1 DF path leaves ovvv lazy.
    if (not use_df_factors and eris.ovvv is None
            and hasattr(eris, 'get_ovvv_packed')):
        eris.get_ovvv_packed()

    @custom_vjp
    def _ccsd_t_kernel(eris, t1, t2):
        t1 = numpy.asarray(t1)
        t2 = numpy.asarray(t2, order='C')
        eris.fock = numpy.asarray(eris.fock, order='C')
        if use_df_factors:
            et = _ccsd_t_energy_df(mycc, eris, t1, t2, max_memory)
        else:
            et = pyscf_ccsd_t.kernel(mycc, eris, t1, t2, verbose)
        return et

    def _ccsd_t_kernel_fwd(eris, t1, t2):
        et = _ccsd_t_kernel(eris, t1, t2)
        return et, (eris, t1, t2)

    def _ccsd_t_kernel_bwd(res, ybar):
        log = logger.new_logger(mycc, verbose)
        eris, t1, t2 = res
        _require_supported_triples_vjp(mycc, eris, t1, t2)

        nocc, nvir = t1.shape
        log.info('CCSD(T) pullback: nocc=%d nvir=%d  et_bar=%.6g  max_memory=%.0f MB',
                 nocc, nvir, float(ybar), max_memory)
        wall_total = time.perf_counter()

        # TODO clean up tree unflatten
        path_vals, treedef = tree_flatten_with_path(eris)
        keys = [item[0][0].name for item in path_vals]
        shapes = [item[1].shape for item in path_vals]
        path_vals = None

        wall_vjp = time.perf_counter()
        t1_bar, t2_bar, fock_bar, mo_energy_bar,\
            ovoo_bar, ovov_bar, ovvv_bar, Lov_bar, Lvv_bar = \
            _ccsd_t_energy_vjp(
                eris, t1, t2, ybar, max_memory,
                use_df_factors=use_df_factors)
        log.info('CCSD(T) pullback  _ccsd_t_energy_vjp  wall=%.2f s',
                 time.perf_counter() - wall_vjp)

        leaves = [None] * len(keys)
        key_to_bar = {
            'fock': fock_bar,
            'mo_energy': mo_energy_bar,
            'ovoo': ovoo_bar,
            'ovov': ovov_bar,
        }
        if use_df_factors:
            key_to_bar.update({'Lov': Lov_bar, 'Lvv': Lvv_bar})
        else:
            key_to_bar['ovvv'] = ovvv_bar

        for k, val in key_to_bar.items():
            leaves[keys.index(k)] = val

        for i, leaf in enumerate(leaves):
            if leaf is None:
                leaves[i] = numpy.zeros(shapes[i])

        eris_bar = tree_unflatten(treedef, leaves)
        log.info('CCSD(T) pullback total wall=%.2f s  '
                 '|t1_bar|max=%.3e |t2_bar|max=%.3e |virtual_bar|max=%.3e',
                 time.perf_counter() - wall_total,
                 float(numpy.abs(t1_bar).max()),
                 float(numpy.abs(t2_bar).max()),
                 float(numpy.abs(Lvv_bar if use_df_factors else ovvv_bar).max()))
        return eris_bar, t1_bar, t2_bar

    _ccsd_t_kernel.defvjp(_ccsd_t_kernel_fwd, _ccsd_t_kernel_bwd)
    return _ccsd_t_kernel(eris, t1, t2)

def _ccsd_t_energy_vjp(eris, t1, t2, et_bar, max_memory,
                       use_df_factors=False):
    # JAX may hand us its host-side TypedNdArray wrapper here; coerce to
    # plain numpy so the .transpose / arithmetic API below works.
    t1 = numpy.asarray(t1)
    t2 = numpy.asarray(t2)
    nocc, nvir = t1.shape
    nmo = nocc + nvir

    et_bar *= 2

    t1T = numpy.asarray(t1.T, order='C')
    t1T_bar = numpy.zeros_like(t1T)
    t2T = numpy.asarray(t2.transpose(2,3,1,0), order='C')
    t2T_bar = numpy.zeros_like(t2T)

    mo_energy = numpy.asarray(eris.mo_energy, order='C')
    mo_energy_bar = numpy.zeros_like(mo_energy)
    fvo = numpy.asarray(eris.fock[nocc:,:nocc], order='C')
    fvo_bar = numpy.zeros_like(fvo)

    vooo = numpy.asarray(eris.ovoo).conj().transpose(1,0,3,2)
    vooo = numpy.asarray(vooo, order='C')
    vooo_bar = numpy.zeros_like(vooo)

    if use_df_factors:
        vvop = vvop_bar = None
        ovov_bar = numpy.zeros_like(numpy.asarray(eris.ovov))
        Lov_bar = numpy.zeros_like(numpy.asarray(eris.Lov))
        Lvv_bar = numpy.zeros_like(numpy.asarray(eris.Lvv))
    else:
        vvop = numpy.empty((nvir,nvir,nocc,nmo))
        vvop[:,:,:,:nocc] = numpy.asarray(eris.ovov).conj().transpose(1,3,0,2)
        vvop[:,:,:,nocc:] = eris.get_ovvv().conj().transpose(1,3,0,2)
        vvop = numpy.asarray(vvop, order='C')
        vvop_bar = numpy.zeros_like(vvop)
        Lov_bar = Lvv_bar = None

    def get_cache(row0, row1, col0, col1):
        if use_df_factors:
            return _build_df_vvop_cache(eris, row0, row1, col0, col1)
        return numpy.asarray(vvop[row0:row1, col0:col1], order='C')

    def accumulate_cache_bar(cache_bar, row0, row1, col0, col1):
        if use_df_factors:
            _accumulate_df_vvop_cache_bar(
                eris, cache_bar, row0, row1, col0, col1,
                ovov_bar, Lov_bar, Lvv_bar)
        else:
            vvop_bar[row0:row1, col0:col1] += cache_bar

    drv = libcc.ccsd_t_energy_vjp
    def contract(a0, a1, b0, b1, cache, cache_bar):
        cache_row_a, cache_col_a, cache_row_b, cache_col_b = cache
        cache_row_a_bar, cache_col_a_bar, cache_row_b_bar, cache_col_b_bar = cache_bar
        drv(mo_energy.ctypes.data_as(ctypes.c_void_p),
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
    min_memory = (nvir**2*nocc**2+nvir*nocc**3+nocc**3*6+2*nvir*nocc+nmo)*num_threads()*8/1e6
    bufsize = (max_memory - min_memory)*1e6/8/num_threads()/(nocc*nmo)
    bufsize *= .5
    bufsize *= .8
    bufsize = max(8, bufsize)

    for a0, a1 in reversed(list(prange_tril(0, nvir, bufsize))):
        cache_row_a = get_cache(a0, a1, 0, a1)
        cache_row_a_bar = numpy.zeros_like(cache_row_a)
        if a0 == 0:
            cache_col_a = cache_row_a
            cache_col_a_bar = cache_row_a_bar
        else:
            cache_col_a = get_cache(0, a0, a0, a1)
            cache_col_a_bar = numpy.zeros_like(cache_col_a)
        contract(a0, a1, a0, a1,
                (cache_row_a, cache_col_a, cache_row_a, cache_col_a),
                (cache_row_a_bar, cache_col_a_bar, cache_row_a_bar, cache_col_a_bar))

        for b0, b1 in prange_tril(0, a0, bufsize/4):
            cache_row_b = get_cache(b0, b1, 0, b1)
            cache_row_b_bar = numpy.zeros_like(cache_row_b)
            if b0 == 0:
                cache_col_b = cache_row_b
                cache_col_b_bar = cache_row_b_bar
            else:
                cache_col_b = get_cache(0, b0, b0, b1)
                cache_col_b_bar = numpy.zeros_like(cache_col_b)
            contract(a0, a1, b0, b1,
                    (cache_row_a, cache_col_a, cache_row_b, cache_col_b),
                    (cache_row_a_bar, cache_col_a_bar, cache_row_b_bar, cache_col_b_bar))

            accumulate_cache_bar(cache_row_b_bar, b0, b1, 0, b1)
            if b0 != 0:
                accumulate_cache_bar(cache_col_b_bar, 0, b0, b0, b1)

        accumulate_cache_bar(cache_row_a_bar, a0, a1, 0, a1)
        if a0 != 0:
            accumulate_cache_bar(cache_col_a_bar, 0, a0, a0, a1)

    t1_bar = numpy.asarray(t1T_bar.T)
    t2_bar = numpy.asarray(t2T_bar.transpose(3,2,0,1))
    fock_bar = numpy.zeros((nmo,nmo))
    fock_bar[nocc:,:nocc] = fvo_bar

    ovoo_bar = numpy.asarray(vooo_bar.transpose(1,0,3,2))
    if use_df_factors:
        ovvv_out_bar = None
    else:
        ovov_bar = numpy.asarray(vvop_bar[:,:,:,:nocc].transpose(2,0,3,1))
        if eris.ovvv.ndim == 4:
            ovvv_out_bar = numpy.asarray(
                vvop_bar[:,:,:,nocc:].transpose(2,0,3,1))
        else:
            ovvv_bar = vvop_bar[:,:,:,nocc:].transpose(2,0,3,1)
            ovvv_bar += ovvv_bar.transpose(0,1,3,2)
            idx, idy = numpy.diag_indices(nvir)
            ovvv_bar[:,:,idx,idy] *= .5
            idx, idy = numpy.tril_indices(nvir)
            ovvv_out_bar = numpy.asarray(ovvv_bar[:,:,idx,idy])
    return (t1_bar, t2_bar, fock_bar, mo_energy_bar, ovoo_bar, ovov_bar,
            ovvv_out_bar, Lov_bar, Lvv_bar)
