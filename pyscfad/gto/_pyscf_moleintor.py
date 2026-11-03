# Copyright 2021-2025 The PySCFAD Authors
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

#pylint: disable=unused-import
from functools import wraps
import warnings
import ctypes
import numpy
from pyscf import lib
from pyscf.gto import moleintor as molint

from pyscfadlib import libcgto_vjp
from pyscfad.experimental.util import replace_source_code

ANG_OF     = molint.ANG_OF
NPRIM_OF   = molint.NPRIM_OF
NCTR_OF    = molint.NCTR_OF
KAPPA_OF   = molint.KAPPA_OF
PTR_EXP    = molint.PTR_EXP
PTR_COEFF  = molint.PTR_COEFF
BAS_SLOTS  = molint.BAS_SLOTS
NGRIDS     = molint.NGRIDS
PTR_GRIDS  = molint.PTR_GRIDS


_cintoptHandler = molint._cintoptHandler
make_loc = molint.make_loc
_stand_sym_code = molint._stand_sym_code
ascint3 = molint.ascint3

_INTOR_FUNCTIONS = molint._INTOR_FUNCTIONS
_INTOR_FUNCTIONS.update({
    'int1e_ovlp_dr10'           : (3, 3),
    'int1e_ovlp_dr01'           : (3, 3),
    'int1e_kin_dr10'            : (3, 3),
    'int1e_kin_dr01'            : (3, 3),
    'int1e_nuc_dr10'            : (3, 3),
    'int1e_nuc_dr01'            : (3, 3),
    'int1e_rinv_dr10'           : (3, 3),
    'int1e_rinv_dr01'           : (3, 3),
    'int2c2e_dr10'              : (3, 3),
    'int2c2e_dr01'              : (3, 3),
    'int1e_r_dr10'              : (9, 9),
    'int1e_r_dr01'              : (9, 9),
    'int1e_rr_dr10'             : (27, 27),
    'int1e_rr_dr01'             : (27, 27),
    'int1e_rrr_dr10'            : (81, 81),
    'int1e_rrr_dr01'            : (81, 81),
    'int1e_r2_dr10'             : (3, 3),
    'int1e_r2_dr01'             : (3, 3),
    'int2e_dr1000'              : (3, 3),
    'int2e_dr0010'              : (3, 3),
    'int1e_ovlp_dr20'           : (9, 9),
    'int1e_ovlp_dr11'           : (9, 9),
    'int1e_ovlp_dr02'           : (9, 9),
    'int1e_kin_dr20'            : (9, 9),
    'int1e_kin_dr11'            : (9, 9),
    'int1e_kin_dr02'            : (9, 9),
    'int1e_nuc_dr20'            : (9, 9),
    'int1e_nuc_dr11'            : (9, 9),
    'int1e_nuc_dr02'            : (9, 9),
    'int1e_rinv_dr20'           : (9, 9),
    'int1e_rinv_dr11'           : (9, 9),
    'int1e_rinv_dr02'           : (9, 9),
    'int2c2e_dr20'              : (9, 9),
    'int2c2e_dr11'              : (9, 9),
    'int2c2e_dr02'              : (9, 9),
    'int1e_r_dr20'              : (27, 27),
    'int1e_r_dr11'              : (27, 27),
    'int1e_r_dr02'              : (27, 27),
    'int1e_rr_dr20'             : (81, 81),
    'int1e_rr_dr11'             : (81, 81),
    'int1e_rr_dr02'             : (81, 81),
    'int1e_rrr_dr20'            : (243, 243),
    'int1e_rrr_dr11'            : (243, 243),
    'int1e_rrr_dr02'            : (243, 243),
    'int1e_r2_dr20'             : (9, 9),
    'int1e_r2_dr11'             : (9, 9),
    'int1e_r2_dr02'             : (9, 9),
    'int2e_dr2000'              : (9, 9),
    'int2e_dr1100'              : (9, 9),
    'int2e_dr1010'              : (9, 9),
    'int2e_dr0020'              : (9, 9),
    'int2e_dr0011'              : (9, 9),
    'int1e_ovlp_dr30'           : (27, 27),
    'int1e_ovlp_dr21'           : (27, 27),
    'int1e_ovlp_dr12'           : (27, 27),
    'int1e_ovlp_dr03'           : (27, 27),
    'int1e_kin_dr30'            : (27, 27),
    'int1e_kin_dr21'            : (27, 27),
    'int1e_kin_dr12'            : (27, 27),
    'int1e_kin_dr03'            : (27, 27),
    'int1e_nuc_dr30'            : (27, 27),
    'int1e_nuc_dr21'            : (27, 27),
    'int1e_nuc_dr12'            : (27, 27),
    'int1e_nuc_dr03'            : (27, 27),
    'int1e_rinv_dr30'           : (27, 27),
    'int1e_rinv_dr21'           : (27, 27),
    'int1e_rinv_dr12'           : (27, 27),
    'int1e_rinv_dr03'           : (27, 27),
    'int2c2e_dr30'              : (27, 27),
    'int2c2e_dr21'              : (27, 27),
    'int2c2e_dr12'              : (27, 27),
    'int2c2e_dr03'              : (27, 27),
    'int1e_r_dr30'              : (81, 81),
    'int1e_r_dr21'              : (81, 81),
    'int1e_r_dr12'              : (81, 81),
    'int1e_r_dr03'              : (81, 81),
    'int1e_rr_dr30'             : (243, 243),
    'int1e_rr_dr21'             : (243, 243),
    'int1e_rr_dr12'             : (243, 243),
    'int1e_rr_dr03'             : (243, 243),
    'int1e_rrr_dr30'            : (729, 729),
    'int1e_rrr_dr21'            : (729, 729),
    'int1e_rrr_dr12'            : (729, 729),
    'int1e_rrr_dr03'            : (729, 729),
    'int1e_r2_dr30'             : (27, 27),
    'int1e_r2_dr21'             : (27, 27),
    'int1e_r2_dr12'             : (27, 27),
    'int1e_r2_dr03'             : (27, 27),
    'int2e_dr3000'              : (27, 27),
    'int2e_dr2100'              : (27, 27),
    'int2e_dr2010'              : (27, 27),
    'int2e_dr1200'              : (27, 27),
    'int2e_dr1110'              : (27, 27),
    'int2e_dr1020'              : (27, 27),
    'int2e_dr1011'              : (27, 27),
    'int2e_dr0030'              : (27, 27),
    'int2e_dr0021'              : (27, 27),
    'int2e_dr0012'              : (27, 27),
    'int1e_ovlp_dr40'           : (81, 81),
    'int1e_ovlp_dr31'           : (81, 81),
    'int1e_ovlp_dr22'           : (81, 81),
    'int1e_ovlp_dr13'           : (81, 81),
    'int1e_ovlp_dr04'           : (81, 81),
    'int1e_kin_dr40'            : (81, 81),
    'int1e_kin_dr31'            : (81, 81),
    'int1e_kin_dr22'            : (81, 81),
    'int1e_kin_dr13'            : (81, 81),
    'int1e_kin_dr04'            : (81, 81),
    'int1e_nuc_dr40'            : (81, 81),
    'int1e_nuc_dr31'            : (81, 81),
    'int1e_nuc_dr22'            : (81, 81),
    'int1e_nuc_dr13'            : (81, 81),
    'int1e_nuc_dr04'            : (81, 81),
    'int1e_rinv_dr40'           : (81, 81),
    'int1e_rinv_dr31'           : (81, 81),
    'int1e_rinv_dr22'           : (81, 81),
    'int1e_rinv_dr13'           : (81, 81),
    'int1e_rinv_dr04'           : (81, 81),
    'int2c2e_dr40'              : (81, 81),
    'int2c2e_dr31'              : (81, 81),
    'int2c2e_dr22'              : (81, 81),
    'int2c2e_dr13'              : (81, 81),
    'int2c2e_dr04'              : (81, 81),
    'int1e_r_dr40'              : (243, 243),
    'int1e_r_dr31'              : (243, 243),
    'int1e_r_dr22'              : (243, 243),
    'int1e_r_dr13'              : (243, 243),
    'int1e_r_dr04'              : (243, 243),
    'int1e_rr_dr40'             : (729, 729),
    'int1e_rr_dr31'             : (729, 729),
    'int1e_rr_dr22'             : (729, 729),
    'int1e_rr_dr13'             : (729, 729),
    'int1e_rr_dr04'             : (729, 729),
    'int1e_rrr_dr40'            : (2187, 2187),
    'int1e_rrr_dr31'            : (2187, 2187),
    'int1e_rrr_dr22'            : (2187, 2187),
    'int1e_rrr_dr13'            : (2187, 2187),
    'int1e_rrr_dr04'            : (2187, 2187),
    'int1e_r2_dr40'             : (81, 81),
    'int1e_r2_dr31'             : (81, 81),
    'int1e_r2_dr22'             : (81, 81),
    'int1e_r2_dr13'             : (81, 81),
    'int1e_r2_dr04'             : (81, 81),
    'int2e_dr4000'              : (81, 81),
    'int2e_dr3100'              : (81, 81),
    'int2e_dr3010'              : (81, 81),
    'int2e_dr2200'              : (81, 81),
    'int2e_dr2110'              : (81, 81),
    'int2e_dr2020'              : (81, 81),
    'int2e_dr2011'              : (81, 81),
    'int2e_dr1300'              : (81, 81),
    'int2e_dr1210'              : (81, 81),
    'int2e_dr1120'              : (81, 81),
    'int2e_dr1111'              : (81, 81),
    'int2e_dr1030'              : (81, 81),
    'int2e_dr1021'              : (81, 81),
    'int2e_dr1012'              : (81, 81),
    'int2e_dr0040'              : (81, 81),
    'int2e_dr0031'              : (81, 81),
    'int2e_dr0022'              : (81, 81),
    'int2e_dr0013'              : (81, 81),
})

def _get_intor_and_comp(intor_name, comp=None):
    intor_name = ascint3(intor_name)
    if comp is None:
        try:
            if '_spinor' in intor_name:
                fname = intor_name.replace('_spinor', '')
                comp = _INTOR_FUNCTIONS[fname][1]
            else:
                fname = intor_name.replace('_sph', '').replace('_cart', '')
                comp = _INTOR_FUNCTIONS[fname][0]
        except KeyError:
            warnings.warn(f'Function {intor_name} not found.  Set its comp to 1')
            comp = 1
    return intor_name, comp

make_cintopt = replace_source_code(molint.make_cintopt, locals(),
                                   'libcgto', 'libcgto_vjp')
getints2c = replace_source_code(molint.getints2c, locals(),
                                'libcgto', 'libcgto_vjp')
_getints3c = replace_source_code(molint.getints3c, locals(),
                                 'libcgto', 'libcgto_vjp')
getints4c = replace_source_code(molint.getints4c, locals(),
                                'libcgto', 'libcgto_vjp')
getints_by_shell = replace_source_code(molint.getints_by_shell, locals(),
                                       'libcgto', 'libcgto_vjp')

# libcint uses a signed 32-bit product of the output dimensions as the
# component stride. Large DF derivative tensors can overflow it even when
# enough memory is available. Bound each native call, retaining the public
# layout and the caller's output buffer.
_INT3C_MAX_COMPONENT_SIZE = numpy.iinfo(numpy.int32).max
_INT3C_BLOCK_BYTES = 256 * 1024**2


@wraps(molint.getints3c)
def getints3c(intor_name, atm, bas, env, shls_slice=None, comp=1,
             aosym='s1', ao_loc=None, cintopt=None, out=None):
    atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(env, dtype=numpy.double, order='C')
    nbas = len(bas)
    if shls_slice is None:
        shls_slice = (0, nbas, 0, nbas, 0, nbas)
        if 'ssc' in intor_name or 'spinor' in intor_name:
            bas = numpy.asarray(numpy.vstack((bas, bas)), dtype=numpy.int32)
            shls_slice = (0, nbas, 0, nbas, nbas, nbas*2)
    i0, i1, j0, j1, k0, k1 = shls_slice[:6]
    if ao_loc is None:
        ao_loc = make_loc(bas, intor_name)
        if 'ssc' in intor_name:
            ao_loc[k0:] = ao_loc[k0] + make_loc(bas[k0:], 'cart')
        elif 'spinor' in intor_name:
            ao_loc[k0:] = ao_loc[k0] + make_loc(bas[k0:], 'sph')
    ni, nj, nk = (int(ao_loc[b])-int(ao_loc[a])
                  for a, b in ((i0, i1), (j0, j1), (k0, k1)))
    if aosym == 's1':
        leading_shape = (ni, nj)
        plane = ni * nj
    else:
        ai0, ai1 = int(ao_loc[i0]), int(ao_loc[i1])
        plane = ai1*(ai1+1)//2 - ai0*(ai0+1)//2
        leading_shape = (plane,)
    if plane * nk <= _INT3C_MAX_COMPONENT_SIZE:
        return _getints3c(intor_name, atm, bas, env, shls_slice, comp,
                         aosym, ao_loc, cintopt, out)

    dtype = numpy.complex128 if 'spinor' in intor_name else numpy.double
    max_aux = min(_INT3C_MAX_COMPONENT_SIZE // plane,
                  max(1, _INT3C_BLOCK_BYTES // (plane*comp*numpy.dtype(dtype).itemsize)))
    # A shell cannot be split in this driver. Check before allocating the
    # full output so a truly oversized shell fails without a native crash.
    if int(max(numpy.diff(ao_loc[k0:k1+1]))) * plane > _INT3C_MAX_COMPONENT_SIZE:
        raise OverflowError('A three-center integral shell exceeds the native indexing limit')
    if cintopt is None:
        opt_bas = bas[:max(i1, j1)] if '3c2e' in intor_name else bas
        cintopt = make_cintopt(atm, opt_bas, env, intor_name)
    mat = numpy.ndarray((*leading_shape, nk, comp), dtype, out, order='F')
    result = numpy.rollaxis(mat, -1, 0)
    if comp == 1:
        result = result[0]
    start = k0
    while start < k1:
        stop = int(numpy.searchsorted(ao_loc, int(ao_loc[start])+max_aux, side='right'))-1
        stop = min(k1, max(start+1, stop))
        block = _getints3c(intor_name, atm, bas, env,
                          (i0, i1, j0, j1, start, stop), comp,
                          aosym, ao_loc, cintopt)
        begin, end = int(ao_loc[start]-ao_loc[k0]), int(ao_loc[stop]-ao_loc[k0])
        result[..., begin:end] = block
        del block
        start = stop
    return result


@wraps(molint.getints)
def getints(intor_name, atm, bas, env, shls_slice=None, comp=None, hermi=0,
            aosym='s1', ao_loc=None, cintopt=None, out=None):
    intor_name, comp = _get_intor_and_comp(intor_name, comp)
    if any(bas[:,ANG_OF] > 12):
        raise NotImplementedError('cint library does not support high angular (l>12) GTOs')

    if ao_loc is not None:
        ao_loc = numpy.asarray(ao_loc, dtype=numpy.int32)

    if (intor_name.startswith('int1e') or
        intor_name.startswith('ECP') or
        intor_name.startswith('int2c2e')):
        return getints2c(intor_name, atm, bas, env, shls_slice, comp,
                         hermi, ao_loc, cintopt, out)
    elif intor_name.startswith('int2e') or intor_name.startswith('int4c1e'):
        return getints4c(intor_name, atm, bas, env, shls_slice, comp,
                         aosym, ao_loc, cintopt, out)
    elif intor_name.startswith('int3c'):
        return getints3c(intor_name, atm, bas, env, shls_slice, comp,
                         aosym, ao_loc, cintopt, out)
    else:
        raise KeyError(f'Unknown intor {intor_name}')
