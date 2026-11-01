# Copyright 2021-2025 Xing Zhang
# Portions copyright 2021-2026 The PySCFAD Authors
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
from jax import custom_vjp
from jax.tree_util import tree_flatten, tree_unflatten
from pyscf.lib import logger
from pyscf.gto.moleintor import (
    ascint3,
    make_loc,
    make_cintopt,
    getints3c,
)

#libcgto = pyscf_lib.load_library('libcgto')
from pyscfadlib import libcgto_vjp as libcgto

@partial(custom_vjp, nondiff_argnums=(2,3,4,5,6))
def int3c_cross(mol, auxmol, intor='int3c2e', comp=1, aosym='s2ij',
                shls_slice=None, out=None):
    pmol = mol + auxmol
    atm = pmol._atm
    bas = pmol._bas
    env = pmol._env
    int3c = ascint3(mol._add_suffix(intor))
    ao_loc = make_loc(bas, int3c)
    cintopt = make_cintopt(atm, bas, env, int3c)
    nbas = mol.nbas
    nauxbas = auxmol.nbas
    if shls_slice is None:
        shls_slice = (0, nbas, 0, nbas, nbas, nbas+nauxbas)
    ints = getints3c(int3c, atm, bas, env,
                     shls_slice=shls_slice, comp=comp,
                     aosym=aosym, ao_loc=ao_loc, cintopt=cintopt, out=out)
    return ints

def int3c_cross_fwd(mol, auxmol, intor, comp, aosym, shls_slice, out):
    primal_out = int3c_cross(mol, auxmol, intor=intor, comp=comp,
                             aosym=aosym, shls_slice=shls_slice, out=out)
    return primal_out, (mol, auxmol)

def int3c_cross_bwd(intor, comp, aosym, shls_slice, out,
                    res, ybar):
    mol, auxmol = res
    log = logger.new_logger(mol)

    if mol.exp is not None:
        raise NotImplementedError
    if mol.ctr_coeff is not None:
        raise NotImplementedError
    if mol.r0 is not None:
        raise NotImplementedError
    if auxmol.exp is not None:
        raise NotImplementedError
    if auxmol.ctr_coeff is not None:
        raise NotImplementedError
    if auxmol.r0 is not None:
        raise NotImplementedError
    assert intor.startswith('int3c2e') and not 'spinor' in intor
    assert aosym == 's2ij'
    assert comp == 1
    if shls_slice is None:
        shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas,
                      mol.nbas + auxmol.nbas)
    aux_sh0 = shls_slice[4] - mol.nbas
    aux_sh1 = shls_slice[5] - mol.nbas
    if not 0 <= aux_sh0 <= aux_sh1 <= auxmol.nbas:
        raise ValueError('Auxiliary shell slice is outside the auxiliary basis.')

    natm = mol.natm
    comp = 3
    vjp = numpy.zeros((natm, comp), order='C', dtype=numpy.double)
    ybar = numpy.asarray(ybar, order='F', dtype=numpy.double)

    pmol = mol + auxmol
    atm = numpy.asarray(pmol._atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(pmol._bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(pmol._env, dtype=numpy.double, order='C')
    c_atm = atm.ctypes.data_as(ctypes.c_void_p)
    c_bas = bas.ctypes.data_as(ctypes.c_void_p)
    c_env = env.ctypes.data_as(ctypes.c_void_p)

    intor1 = 'int3c2e_ip1'
    intor1 = ascint3(mol._add_suffix(intor1))
    ao_loc = make_loc(bas, intor1)
    ao_loc = numpy.asarray(ao_loc, order='C', dtype=numpy.int32)
    i0, i1 = ao_loc[shls_slice[0]], ao_loc[shls_slice[1]]
    npair = int(i1) * (int(i1) + 1) // 2 - int(i0) * (int(i0) + 1) // 2
    naux_slice = ao_loc[shls_slice[5]] - ao_loc[shls_slice[4]]
    if ybar.shape != (npair, naux_slice):
        raise ValueError(
            f'Integral cotangent has shape {ybar.shape}, '
            f'expected {(npair, naux_slice)} for the shell slice.')
    cintopt = make_cintopt(atm, bas, env, intor1)

    drv = libcgto.GTOnr3c_ij_r0_vjp
    fill = getattr(libcgto, 'GTOnr3c_ij_r0_vjp_'+aosym)

    # Adapted from upstream PySCFAD db0acafb (Apache-2.0): translation
    # invariance gives the auxiliary derivative from the two AO derivatives.
    # Keep the public (pair, aux) layout and pass contiguous auxiliary slices
    # to the existing driver; it handles each slice's AO-pair row offset.
    # Both the atom intersections and cotangent offsets must be relative to
    # the requested auxiliary slice, which may start/end inside an atom.
    vjp_aux = numpy.zeros((auxmol.natm, comp))
    aux_loc = ao_loc[mol.nbas:] - ao_loc[mol.nbas]
    aux_start = aux_loc[aux_sh0]
    for ia, (b0, b1, _, _) in enumerate(auxmol.aoslice_by_atom()):
        b0, b1 = max(b0, aux_sh0), min(b1, aux_sh1)
        if b0 >= b1:
            continue
        p0, p1 = aux_loc[b0] - aux_start, aux_loc[b1] - aux_start
        vjp_ia = numpy.zeros_like(vjp)
        ybar_ia = numpy.asarray(ybar[:, p0:p1], order='F')
        drv(getattr(libcgto, intor1), fill,
            vjp_ia.ctypes.data_as(ctypes.c_void_p),
            ybar_ia.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(comp),
            (ctypes.c_int*6)(*shls_slice[:4], mol.nbas+b0, mol.nbas+b1),
            ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
            c_atm, ctypes.c_int(len(atm)), ctypes.c_int(natm),
            c_bas, ctypes.c_int(len(bas)), c_env)
        vjp += vjp_ia
        vjp_aux[ia] = -vjp_ia.sum(axis=0)

    _, mol_tree = tree_flatten(mol)
    mol_bar = tree_unflatten(mol_tree, [-vjp,])
    _, auxmol_tree = tree_flatten(auxmol)
    auxmol_bar = tree_unflatten(auxmol_tree, [-vjp_aux,])
    log.timer('int3c_cross_bwd')
    del log
    return (mol_bar, auxmol_bar)

int3c_cross.defvjp(int3c_cross_fwd, int3c_cross_bwd)
