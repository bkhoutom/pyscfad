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

from functools import partial
import ctypes
import numpy
import jax
from jax import custom_vjp
from jax.tree_util import tree_flatten, tree_unflatten
from pyscf import lib
from pyscf.lib import logger
from pyscf.df import df_jk as pyscf_df_jk
from pyscfadlib import libcvhf_vjp as libvhf
from pyscfad.df import incore as df_incore

libao2mo = lib.load_library('libao2mo')
_FAST_EXCHANGE_DM_DATA = {}


def _fast_exchange_key(dfobj):
    cderi = getattr(dfobj, '_cderi', None)
    if cderi is not None:
        return ('cderi', id(cderi), tuple(cderi.shape))
    return ('dfobj', id(dfobj))


def set_fast_exchange_dm_data(dfobj, mo_coeff, mo_occ):
    _FAST_EXCHANGE_DM_DATA[_fast_exchange_key(dfobj)] = (mo_coeff, mo_occ)


def _tag_dm_for_fast_exchange(dfobj, dm):
    if getattr(dm, 'mo_coeff', None) is not None:
        return dm

    tagged = _FAST_EXCHANGE_DM_DATA.get(_fast_exchange_key(dfobj))
    if tagged is None:
        return dm
    mo_coeff, mo_occ = tagged

    dm = numpy.asarray(dm)
    return lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)


def _cderi_mol_aux_vjp(dfobj, eri_bar):
    auxmol = dfobj.auxmol
    if auxmol is None:
        return None, None

    def build_cderi(mol, auxmol):
        max_memory = max(dfobj.max_memory, 4096)
        return df_incore.cholesky_eri(
            mol,
            auxmol=auxmol,
            int3c=mol._add_suffix('int3c2e'),
            int2c=mol._add_suffix('int2c2e'),
            max_memory=max_memory,
            verbose=0,
        )

    _, pullback = jax.vjp(build_cderi, dfobj.mol, auxmol)
    mol_bar, auxmol_bar = pullback(eri_bar)
    mol_bar = tree_flatten(mol_bar)[0][0]
    auxmol_bar = tree_flatten(auxmol_bar)[0][0]
    return mol_bar, auxmol_bar


@partial(custom_vjp, nondiff_argnums=(2,3,4,5))
def get_jk(dfobj, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-13):
    dm_forward = _tag_dm_for_fast_exchange(dfobj, dm)
    return pyscf_df_jk.get_jk(dfobj, dm_forward, hermi=hermi,
                              with_j=with_j, with_k=with_k,
                              direct_scf_tol=direct_scf_tol)


def get_jk_fwd(dfobj, dm, hermi, with_j, with_k, direct_scf_tol):
    vj, vk = get_jk(dfobj, dm, hermi, with_j, with_k, direct_scf_tol)
    return (vj, vk), (dfobj, dm)


def get_jk_bwd(hermi, with_j, with_k, direct_scf_tol,
               res, ybar):
    dfobj, dm = res
    vj_bar, vk_bar = ybar

    log = logger.new_logger(dfobj)
    fmmm = libao2mo.AO2MOmmm_bra_nr_s2
    fdrv = libao2mo.AO2MOnr_e2_drv
    ftrans = libao2mo.AO2MOtranse2_nr_s2
    vjpdrv = libvhf.df_vk_vjp
    null = lib.c_null_ptr()

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]

    vj_bar = numpy.asarray(vj_bar).reshape(-1,nao,nao)
    if with_j:
        idx = numpy.arange(nao)
        dmtril = lib.pack_tril(dms + dms.conj().transpose(0,2,1))
        dmtril[:,idx*(idx+1)//2+idx] *= .5

        vj_bar_tril = lib.pack_tril(vj_bar + vj_bar.conj().transpose(0,2,1))
        vj_bar_tril[:,idx*(idx+1)//2+idx] *= .5

    #TODO save eri_bar on disk
    eri_bar = numpy.zeros((dfobj.get_naoaux(), nao*(nao+1)//2))
    dms_bar = [numpy.zeros((nao,nao), order='F'),] * nset

    vk_bar = numpy.asarray(vk_bar).reshape(-1,nao,nao)
    vk_bar = [numpy.asarray(x, order='F') for x in vk_bar]
    dms = [numpy.asarray(x, order='F') for x in dms]

    rargs = (ctypes.c_int(nao), (ctypes.c_int*4)(0, nao, 0, nao),
             null, ctypes.c_int(0))
    max_memory = dfobj.max_memory - lib.current_memory()[0]
    blksize = max(4, int(min(dfobj.blockdim, max_memory*.3e6/8/nao**2)))
    buf = numpy.empty((blksize,nao,nao))
    p1 = 0
    for eri1 in dfobj.loop(blksize):
        naux, nao_pair = eri1.shape
        p0, p1 = p1, p1 + naux
        if with_j:
            rho_bar = vj_bar_tril @ eri1.T
            dmtril_bar = rho_bar @ eri1
            dms_bar += lib.unpack_tril(dmtril_bar)

            rho = dmtril @ eri1.T
            eri_bar[p0:p1] += rho.T @ vj_bar_tril
            eri_bar[p0:p1] += rho_bar.T @ dmtril

        for k in range(nset):
            #TODO save buf1 on disk to avoid recomputation
            buf1 = buf[:naux]
            fdrv(ftrans, fmmm,
                 buf1.ctypes.data_as(ctypes.c_void_p),
                 eri1.ctypes.data_as(ctypes.c_void_p),
                 dms[k].ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux), *rargs)

            vjpdrv(eri_bar[p0:p1].ctypes.data_as(ctypes.c_void_p),
                   dms_bar[k].ctypes.data_as(ctypes.c_void_p),
                   vk_bar[k].ctypes.data_as(ctypes.c_void_p),
                   buf1.ctypes.data_as(ctypes.c_void_p),
                   eri1.ctypes.data_as(ctypes.c_void_p),
                   dms[k].ctypes.data_as(ctypes.c_void_p),
                   ctypes.c_int(naux), ctypes.c_int(nao))

    dm_bar = numpy.asarray(dms_bar).reshape(dm_shape)
    #TODO need a better way to add vjps for objects
    leaves, tree = tree_flatten(dfobj)
    bar_leaves = []
    mol_bar = auxmol_bar = None
    if len(leaves) >= 3 and hasattr(leaves[-1], 'shape') and tuple(leaves[-1].shape) != tuple(eri_bar.shape):
        mol_bar, auxmol_bar = _cderi_mol_aux_vjp(dfobj, eri_bar)
    for i, leaf in enumerate(leaves):
        if i == 0:
            bar_leaves.append(mol_bar)
        elif i == 1:
            bar_leaves.append(auxmol_bar)
        elif i == len(leaves) - 1:
            if hasattr(leaf, 'shape') and tuple(leaf.shape) == tuple(eri_bar.shape):
                bar_leaves.append(eri_bar)
            elif hasattr(leaf, 'shape'):
                bar_leaves.append(numpy.zeros(leaf.shape))
            else:
                bar_leaves.append(None)
        elif hasattr(leaf, 'shape'):
            bar_leaves.append(numpy.zeros(leaf.shape))
        else:
            bar_leaves.append(None)
    dfobj_bar = tree_unflatten(tree, bar_leaves)
    log.timer('get_jk_bwd')
    del log
    return (dfobj_bar, dm_bar)


get_jk.defvjp(get_jk_fwd, get_jk_bwd)
