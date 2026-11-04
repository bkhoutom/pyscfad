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
from ._profile import _vjp_progress_section
from . import _df_symmetric


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
            float(os.environ.get('PYSCFAD_LNO_LOCAL_DIRECT_INT3C_BLOCK_MB', 512.0)),
            1.0,
        )
    except ValueError:
        return 512.0


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
    from ._df_shared import matching_scope
    if matching_scope(mol, auxmol) is not None:
        # The shared scope retains the metric and optionally H inputs.
        return _local_direct_wide_nr_e2(
            mol, auxmol, mo_coeff, max_memory, orbs_slice)
    # PySCF's MP2 helper reuses an AO-pair buffer for its MO output.  An
    # all-pair CC transform can be wider, especially after AO projection.
    # Stream those transforms into their own result instead of overflowing
    # that buffer.  The usual occupied-virtual path remains unchanged.
    if nkl > mol.nao * (mol.nao + 1) // 2:
        return _local_direct_wide_nr_e2(
            mol, auxmol, mo_coeff, max_memory, orbs_slice,
        )
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


def _local_direct_wide_nr_e2(mol, auxmol, mo_coeff, max_memory, orbs_slice):
    """Blocked transform for shared inputs or MO pairs wider than AO pairs."""
    from pyscf.ao2mo import _ao2mo as pyscf_ao2mo

    k0, k1, l0, l1 = map(int, orbs_slice)
    nkl = (k1 - k0) * (l1 - l0)
    npair = mol.nao * (mol.nao + 1) // 2
    coeff = numpy.asarray(jax.device_get(mo_coeff), order='F')
    from ._df_shared import matching_scope
    scope = matching_scope(mol, auxmol)
    if scope is None:
        metric = numpy.asarray(jax.device_get(auxmol.intor('int2c2e', hermi=1)))
        low = scipy_linalg.cholesky(metric, lower=True, check_finite=False)
        del metric
    else:
        low = scope.metric_factor()
        if scope.half is not None:
            return scope.half.transform(coeff, orbs_slice, low)
    result = numpy.empty((auxmol.nao, nkl), order='F')
    available_bytes = max(0., (float(max_memory)
        - pyscf_lib.current_memory()[0]) * 1e6 - result.nbytes)
    block_bytes = min(_local_direct_int3c_block_mb() * 1024**2,
                      .7 * available_bytes)
    rows = max(1, int(block_bytes // (8 * (npair + nkl))))
    for p0, p1, raw in _local_direct_raw_int3c_blocks(
            mol, auxmol, z_aux_block_max_rows=rows):
        result[p0:p1] = pyscf_ao2mo.nr_e2(
            numpy.ascontiguousarray(raw), coeff, orbs_slice,
            aosym='s2', mosym='s1',
        )
    del raw
    return np.asarray(scipy_linalg.solve_triangular(
        low, result, lower=True, overwrite_b=True, check_finite=False,
    ))


def _local_direct_raw_int3c_blocks(
        mol, auxmol, z_aux_block_max_rows=None):
    """Generate bounded local integral tiles; never read or spool AO inputs."""
    from ._df_shared import matching_scope
    scope = matching_scope(mol, auxmol)
    if scope is not None:
        pairs = mol.nao * (mol.nao + 1) // 2
        remaining = scope.max_memory_mb - pyscf_lib.current_memory()[0]
        budget = min(scope.workspace_mb, .5 * max(0., remaining),
                     _local_direct_int3c_block_mb()) * 1e6
        # Direct streaming has only one indivisible minimum row. Preserve
        # the scope's advisory fallback when RSS exhausts the target; the
        # resident half-cache performs its strict minimum-tile check first.
        scope._warn_minimum_row(16*pairs, budget, operation='integral')
        rows = max(1, int(budget // (16*pairs)))
        if z_aux_block_max_rows is not None:
            if int(z_aux_block_max_rows) <= 0:
                raise ValueError('z_aux_block_max_rows must be positive')
            rows = min(rows, int(z_aux_block_max_rows))
        z_aux_block_max_rows = rows
        scope.receipt['raw_source_passes'] += 1
        scope.receipt['raw_integral_builds'] += 1
        scope.receipt['raw_integral_bytes'] = 8*auxmol.nao*pairs
    for p0, p1, block in _generate_local_raw_int3c_blocks(
            mol, auxmol, z_aux_block_max_rows):
        if scope is not None:
            scope.receipt['max_raw_rows'] = max(scope.receipt['max_raw_rows'], p1-p0)
        yield p0, p1, block


def _generate_local_raw_int3c_blocks(
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
        z_aux_block_max_rows=None, *, symmetric=False):
    """MO-coefficient pullback with bounded z reads per integral block."""
    mo_coeff_bar = numpy.zeros_like(numpy.asarray(jax.device_get(mo_coeff)))
    k0, k1, l0, l1 = map(int, orbs_slice)
    kl_count = (k1 - k0) * (l1 - l0)
    if kl_count == 0:
        return np.asarray(mo_coeff_bar)
    from ._df_shared import matching_scope
    scope = matching_scope(mol, auxmol)
    if scope is not None and scope.half is not None:
        return np.asarray(scope.half.coefficient_pullback(
            numpy.asarray(jax.device_get(mo_coeff)), read_z_aux_block, orbs_slice,
            symmetric=symmetric))
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
        z_aux_block_max_rows=None, *, symmetric=False):
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
        symmetric=symmetric,
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

    def metric_cholesky(auxmol_):
        return jsp_linalg.cholesky(
            auxmol_.intor(auxmol_._add_suffix('int2c2e'), hermi=1),
            lower=True,
        )

    from ._df_shared import matching_scope
    scope = matching_scope(mol, auxmol)
    if scope is None:
        low, metric_pullback = jax.vjp(metric_cholesky, auxmol)
        low = numpy.asarray(jax.device_get(low))
    else:
        low = scope.metric_factor()
        metric_pullback = None
    if (
        low.shape != (auxmol.nao, auxmol.nao)
        or not numpy.all(numpy.isfinite(low))
        or numpy.any(numpy.diag(low) <= 0)
    ):
        raise NotImplementedError(
            'Integral-direct local Lov VJP does not yet support the '
            'linear-dependent auxiliary-metric eigenvalue fallback.'
        )

    fitted_mo_np = numpy.asarray(jax.device_get(fitted_mo))
    fitted_mo_bar_np = numpy.asarray(jax.device_get(fitted_mo_bar))
    symmetric = _df_symmetric.same_orbital_pair(orbs_slice)
    z, low_bar = _df_symmetric.factor_pullback(
        low, fitted_mo_np, fitted_mo_bar_np, symmetric=symmetric,
        receipt=None if scope is None else scope.receipt)
    del fitted_mo_np, fitted_mo_bar_np, fitted_mo, fitted_mo_bar, low
    if symmetric:
        width = orbs_slice[1] - orbs_slice[0]
        read_z = lambda p0, p1: _df_symmetric.unpack_symmetric(
            z[p0:p1], width).reshape(p1-p0, width*width)
    else:
        read_z = z

    # Consume the metric tape before streaming the three-center integrals.
    # Its pullback uses only the lower triangle of the factor cotangent.
    _cderi_vjp._zero_strict_upper_inplace(low_bar)
    from ._df_shared import defer_geometry
    if defer_geometry(mol, auxmol, mo_coeff, z, low_bar, orbs_slice,
                      symmetric_packed=symmetric):
        # Coefficient bars must remain immediate: LIS orbital derivatives
        # determine the W cotangent that later joins this domain's response.
        del metric_pullback, low_bar
        with _vjp_progress_section('local DF MO-coeff backward'):
            mo_coeff_bar = _local_direct_mo_coeff_vjp(
                mol, auxmol, mo_coeff, read_z, orbs_slice, symmetric=symmetric)
        return (_cderi_vjp._coords_tree_like(mol, numpy.zeros((mol.natm, 3))),
                _cderi_vjp._coords_tree_like(auxmol, numpy.zeros((auxmol.natm, 3))),
                mo_coeff_bar)
    aux_metric_bar = metric_pullback(np.asarray(low_bar))[0]
    del metric_pullback, low_bar

    with _vjp_progress_section('local DF MO-coeff backward'):
        mo_coeff_bar = _local_direct_mo_coeff_vjp(
            mol, auxmol, mo_coeff, read_z, orbs_slice, symmetric=symmetric
        )
    with _vjp_progress_section('local DF integral derivative backward'):
        mol_bar, auxmol_bar = _cderi_vjp._int3c_mo_deriv_coords_vjp(
            mol,
            auxmol,
            mo_coeff,
            read_z,
            orbs_slice,
            int3c=mol._add_suffix('int3c2e'),
            aosym='s2ij',
            block_memory_mb=_local_direct_int3c_block_mb(),
            symmetric=symmetric,
        )
    del z

    auxmol_bar = _tree_add(auxmol_bar, aux_metric_bar)
    return mol_bar, auxmol_bar, mo_coeff_bar


_local_direct_nr_e2.defvjp(_local_direct_nr_e2_fwd, _local_direct_nr_e2_bwd)
