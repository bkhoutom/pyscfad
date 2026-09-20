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

"""Shell-direct local DF transformation and its analytic reverse contractions."""

import os
from types import FunctionType, SimpleNamespace
from functools import partial, reduce
import numpy
import jax
import jax.scipy.linalg as jsp_linalg
import scipy.linalg as scipy_linalg
from pyscf import lib as pyscf_lib
from pyscf.ao2mo.outcore import balance_partition
from pyscfad import numpy as np
from pyscfad.ao2mo import _ao2mo
from pyscfad.df import _cderi_vjp
from pyscfad.df import _int3c_cross_opt


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def _tree_add(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return jax.tree_util.tree_map(lambda a, b: a + b, x, y)


def _local_direct_int3c_block_mb():
    """Memory target for the two raw-int3c buffers in direct local AO2MO."""
    try:
        return max(
            float(os.environ.get('PYSCFAD_LNO_LOCAL_DIRECT_INT3C_BLOCK_MB', 256.0)),
            1.0,
        )
    except ValueError:
        return 256.0


def _local_direct_effective_max_memory(max_memory, naux, nkl):
    """Cap PySCF's direct-DF shell buffer without shrinking the Lov result.

    ``_init_mp_df_eris_direct`` subtracts the resident ``ovL`` array before
    assigning 70 percent of the remaining memory to two AO-pair buffers.  A
    large process-wide memory limit would therefore make a single ED request
    multi-gigabyte work buffers.  Supplying this tighter, current-RSS-aware
    limit keeps the block near the explicit target while retaining the same
    integral-direct algorithm.
    """
    current_mb = float(pyscf_lib.current_memory()[0])
    lov_mb = float(naux) * float(nkl) * numpy.dtype(numpy.float64).itemsize / 1e6
    requested_mb = current_mb + lov_mb + _local_direct_int3c_block_mb() / 0.7
    return max(current_mb + lov_mb, min(float(max_memory), requested_mb))


def _local_direct_nr_e2_impl(mol, auxmol, mo_coeff, max_memory, orbs_slice):
    """Build fitted three-center MO integrals without an AO-pair CDERI.

    This is PySCF's integral-direct DF-MP2 transformation applied to an ED:
    raw ``(P|mu nu)`` integrals are generated in auxiliary-shell blocks and
    transformed to ``(P|ia)`` before the auxiliary-metric solve.  The returned
    layout is ``(naux, n_i*n_a)`` to match :func:`_ao2mo.nr_e2`.
    """
    # Import lazily because this is a private PySCF helper used only by the
    # opt-in path; normal LNO imports should not depend on its availability.
    from pyscf.mp.dfmp2 import _init_mp_df_eris_direct

    k0, k1, l0, l1 = map(int, orbs_slice)
    if not (0 <= k0 <= k1 <= mo_coeff.shape[1]):
        raise ValueError('Invalid first-orbital slice for direct local AO2MO')
    if not (0 <= l0 <= l1 <= mo_coeff.shape[1]):
        raise ValueError('Invalid second-orbital slice for direct local AO2MO')

    coeff_k = numpy.asarray(jax.device_get(mo_coeff[:, k0:k1]), order='F')
    coeff_l = numpy.asarray(jax.device_get(mo_coeff[:, l0:l1]), order='F')
    nkl = int(coeff_k.shape[1] * coeff_l.shape[1])
    effective_memory = _local_direct_effective_max_memory(
        max_memory, auxmol.nao, nkl
    )
    holder = SimpleNamespace(mol=mol, auxmol=auxmol)
    ovl = _init_mp_df_eris_direct(
        holder,
        coeff_k,
        coeff_l,
        effective_memory,
        log=pyscf_lib.logger.new_logger(mol),
    )
    return np.asarray(numpy.asarray(ovl.T))


def _local_direct_raw_int3c_blocks(
        mol, auxmol, z_aux_block_max_rows=None):
    """Yield disjoint ``(auxiliary AO slice, raw int3c block)`` triples."""
    npair = mol.nao * (mol.nao + 1) // 2
    target_bytes = _local_direct_int3c_block_mb() * 1024.0**2
    aux_blksize = max(
        1,
        min(
            auxmol.nao,
            int(target_bytes // max(npair * numpy.dtype(numpy.float64).itemsize, 1)),
        ),
    )
    if z_aux_block_max_rows is not None:
        z_aux_block_max_rows = int(z_aux_block_max_rows)
        if z_aux_block_max_rows <= 0:
            raise ValueError('z_aux_block_max_rows must be positive')
        aux_blksize = min(aux_blksize, z_aux_block_max_rows)
    for shl0, shl1, _ in balance_partition(auxmol.ao_loc, aux_blksize):
        p0 = int(auxmol.ao_loc[shl0])
        p1 = int(auxmol.ao_loc[shl1])
        shls_slice = (
            0, mol.nbas, 0, mol.nbas,
            mol.nbas + shl0, mol.nbas + shl1,
        )
        ints = _int3c_cross_opt.int3c_cross(
            mol,
            auxmol,
            intor='int3c2e',
            comp=1,
            aosym='s2ij',
            shls_slice=shls_slice,
        )
        raw_ints = numpy.asarray(jax.device_get(ints)).T
        if z_aux_block_max_rows is None or p1 - p0 <= z_aux_block_max_rows:
            yield p0, p1, raw_ints
        else:
            # A single shell may contain more AOs than the requested cap.
            # Generate it once and expose bounded, disjoint logical blocks.
            for read0, read1 in _cderi_vjp._iter_auxiliary_subranges(
                    p0, p1, z_aux_block_max_rows):
                raw0 = read0 - p0
                raw1 = read1 - p0
                yield read0, read1, raw_ints[raw0:raw1]


def _local_direct_mo_coeff_vjp_from_z_reader(
        mol, auxmol, mo_coeff, read_z_aux_block, orbs_slice,
        z_aux_block_max_rows=None):
    """MO-coefficient pullback with bounded z reads per integral block."""
    mo_coeff_bar = numpy.zeros_like(numpy.asarray(jax.device_get(mo_coeff)))
    k0, k1, l0, l1 = map(int, orbs_slice)
    kl_count = (k1 - k0) * (l1 - l0)
    if kl_count == 0:
        return np.asarray(mo_coeff_bar)
    for p0, p1, raw_ints in _local_direct_raw_int3c_blocks(
            mol, auxmol, z_aux_block_max_rows=z_aux_block_max_rows):
        z_block = numpy.asarray(read_z_aux_block(p0, p1))
        expected_shape = (p1 - p0, kl_count)
        if z_block.shape != expected_shape:
            raise ValueError(
                'z auxiliary block has incompatible shape: '
                f'got {z_block.shape}, expected {expected_shape}'
            )
        mo_coeff_bar += numpy.asarray(
            _ao2mo.nr_e2_mo_coeff_vjp(
                raw_ints,
                mo_coeff,
                z_block,
                orbs_slice,
                aosym='s2',
                mosym='s1',
            )
        )
    return np.asarray(mo_coeff_bar)


def _local_direct_mo_coeff_vjp(
        mol, auxmol, mo_coeff, z, orbs_slice,
        z_aux_block_max_rows=None):
    """MO-coefficient pullback from a full z array or auxiliary reader."""
    if callable(z):
        read_z_aux_block = z
    else:
        k0, k1, l0, l1 = map(int, orbs_slice)
        z_array = numpy.asarray(jax.device_get(z)).reshape(
            auxmol.nao, (k1 - k0) * (l1 - l0)
        )
        read_z_aux_block = lambda p0, p1: z_array[p0:p1, :]
    return _local_direct_mo_coeff_vjp_from_z_reader(
        mol,
        auxmol,
        mo_coeff,
        read_z_aux_block,
        orbs_slice,
        z_aux_block_max_rows=z_aux_block_max_rows,
    )


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _local_direct_nr_e2(mol, auxmol, mo_coeff, max_memory, orbs_slice):
    return _local_direct_nr_e2_impl(
        mol, auxmol, mo_coeff, max_memory, orbs_slice
    )


def _local_direct_nr_e2_fwd(mol, auxmol, mo_coeff, max_memory, orbs_slice):
    fitted_mo = _local_direct_nr_e2(
        mol, auxmol, mo_coeff, max_memory, orbs_slice
    )
    return fitted_mo, (mol, auxmol, mo_coeff, fitted_mo)


def _local_direct_nr_e2_bwd(max_memory, orbs_slice, res, fitted_mo_bar):
    del max_memory
    mol, auxmol, mo_coeff, fitted_mo = res
    del res
    if mol.exp is not None or mol.ctr_coeff is not None or mol.r0 is not None:
        raise NotImplementedError(
            'Integral-direct local Lov currently supports coordinate and '
            'MO-coefficient derivatives, but not AO basis-parameter derivatives.'
        )
    if (
        auxmol.exp is not None
        or auxmol.ctr_coeff is not None
        or auxmol.r0 is not None
    ):
        raise NotImplementedError(
            'Integral-direct local Lov currently supports coordinate and '
            'MO-coefficient derivatives, but not auxiliary basis-parameter '
            'derivatives.'
        )

    j2c = numpy.asarray(
        jax.device_get(auxmol.intor(auxmol._add_suffix('int2c2e'), hermi=1))
    )
    try:
        low = scipy_linalg.cholesky(j2c, lower=True, check_finite=False)
    except scipy_linalg.LinAlgError as err:
        raise NotImplementedError(
            'Integral-direct local Lov VJP does not yet support the '
            'linear-dependent auxiliary-metric eigenvalue fallback.'
        ) from err
    del j2c

    fitted_mo_np = numpy.asarray(jax.device_get(fitted_mo))
    fitted_mo_bar_np = numpy.asarray(jax.device_get(fitted_mo_bar))
    z = scipy_linalg.solve_triangular(
        low.T,
        fitted_mo_bar_np,
        lower=False,
        check_finite=False,
    )

    # If Y = L^{-1} T and Z = L^{-T} Ybar, then Lbar = -Z Y^T.  Form this
    # small auxiliary-metric cotangent before the streamed integral work so
    # the much larger Y and Ybar buffers can reach their last use here.
    low_bar = -numpy.dot(z, fitted_mo_np.T)
    del fitted_mo_np, fitted_mo_bar_np, fitted_mo, fitted_mo_bar, low

    mo_coeff_bar = _local_direct_mo_coeff_vjp(
        mol, auxmol, mo_coeff, z, orbs_slice
    )
    mol_bar, auxmol_bar = _cderi_vjp._int3c_mo_deriv_coords_vjp(
        mol,
        auxmol,
        mo_coeff,
        z,
        orbs_slice,
        int3c=mol._add_suffix('int3c2e'),
        aosym='s2ij',
        block_memory_mb=_local_direct_int3c_block_mb(),
    )
    del z

    def metric_cholesky(auxmol_):
        return jsp_linalg.cholesky(
            auxmol_.intor(auxmol_._add_suffix('int2c2e'), hermi=1),
            lower=True,
        )

    _, metric_pullback = jax.vjp(metric_cholesky, auxmol)
    # The Cholesky pullback consumes only the stored lower triangle.  Zero the
    # unused half in place rather than allocating another naux-by-naux array.
    _cderi_vjp._zero_strict_upper_inplace(low_bar)
    aux_metric_bar = metric_pullback(np.asarray(low_bar))[0]
    del metric_pullback, low_bar
    auxmol_bar = _tree_add(auxmol_bar, aux_metric_bar)
    return mol_bar, auxmol_bar, mo_coeff_bar


_local_direct_nr_e2.defvjp(_local_direct_nr_e2_fwd, _local_direct_nr_e2_bwd)
