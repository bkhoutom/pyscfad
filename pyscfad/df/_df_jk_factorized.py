"""First-order DF J/K reverse for an explicitly factored real density.

The input is S, with D=S S.T. Its reverse never constructs the generic
AO-square density cotangent or the auxiliary-by-AO-pair integral cotangent.
Coordinate work uses thin occupied factors and one bounded integral pass.

Runtime memory budgets (MiB) can be set with
``PYSCFAD_FACTOR_PANEL_MEMORY_MB`` (combined panel threshold, default 256)
and ``PYSCFAD_SOLVE_MEMORY_MB`` (solve workspace target, default 64).
Finite nonnegative values are accepted; unset or invalid values use the defaults.
Zero forces disk panels or single-column solve slabs, respectively.
"""

from functools import partial
from contextlib import contextmanager
import math
import os
import time

import jax
from jax import custom_vjp
from jax import scipy as jax_scipy
from jax.tree_util import tree_flatten, tree_unflatten
import numpy
import scipy.linalg
from threadpoolctl import threadpool_limits
from pyscf import lib
from pyscf.df import df_jk as pyscf_df_jk

from pyscfad import numpy as np
from pyscfad.df import _cderi_vjp
from pyscfad.lib._threading import dense_blas_threads

_FACTOR_PANEL_MEMORY_MB = 256.0
_SOLVE_MEMORY_MB = 64.0


def _memory_mb(name, default):
    """Read a runtime budget without changing the fallback module constants."""
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        return default
    return value if math.isfinite(value) and value >= 0 else default


@contextmanager
def _factor_panels(naux, nao, rank, with_k):
    """Keep small thin panels in RAM; spill large ones to owned scratch."""
    if not with_k:
        yield None, None
        return
    shape = (naux, nao * rank)
    panel_memory = _memory_mb('PYSCFAD_FACTOR_PANEL_MEMORY_MB', _FACTOR_PANEL_MEMORY_MB)
    if 2 * numpy.prod(shape) * 8 <= panel_memory * 1024**2:
        yield numpy.empty(shape), numpy.empty(shape)
    else:
        # Every later read/solve is a bounded slab. The file is closed and
        # removed on success or failure, and never contains AO-pair bars.
        with lib.H5TmpFile() as scratch:
            chunks = (min(naux, 16), min(shape[1], 1024))
            yield (scratch.create_dataset('T', shape, dtype='f8', chunks=chunks),
                   scratch.create_dataset('R', shape, dtype='f8', chunks=chunks))


@partial(custom_vjp, nondiff_argnums=(2, 3, 4))
def get_jk_from_occ(dfobj, factors, with_j=True, with_k=True, factor_response=True):
    S = numpy.asarray(factors)
    dm = lib.tag_array(S @ S.T, mo_coeff=S,
                       mo_occ=numpy.ones(S.shape[1]))
    J, K = pyscf_df_jk.get_jk(dfobj, dm, hermi=1,
                             with_j=with_j, with_k=with_k)
    zero = numpy.zeros_like(dm)
    return J if with_j else zero, K if with_k else zero


def _forward(dfobj, factors, with_j, with_k, factor_response):
    return get_jk_from_occ(dfobj, factors, with_j, with_k, factor_response), (dfobj, factors)


def _backward(with_j, with_k, factor_response, residual, cotangents):
    dfobj, factors = residual
    nao, rank = factors.shape
    with _factor_panels(dfobj.get_naoaux(), nao, rank, with_k) as panels:
        return _reverse(with_j, with_k, factor_response, residual, cotangents, *panels)


def _reverse(with_j, with_k, factor_response, residual, cotangents, T, Rtilde):
    started = time.perf_counter()
    dfobj, factors = residual
    S = numpy.asarray(factors)
    G, H = (numpy.asarray(bar) for bar in cotangents)
    G, H = (G + G.T) * .5, (H + H.T) * .5
    nao, rank = S.shape
    naux = dfobj.get_naoaux()
    # Bound unpacked AO matrices and their products independently of naux.
    rows = max(1, min(dfobj.blockdim, int(128 * 1024**2 / (8 * nao * nao))))
    Sbar = numpy.zeros_like(S) if factor_response else None
    q = numpy.zeros(naux)
    g = numpy.zeros(naux)
    end = 0
    for packed in dfobj.loop(rows):
        begin, end = end, end + len(packed)
        B = lib.unpack_tril(packed)
        with dense_blas_threads():
            t = B @ S
            if with_j:
                q[begin:end] = numpy.einsum('ar,par->p', S, t)
                g[begin:end] = numpy.einsum('ab,pab->p', G, B)
                if factor_response:
                    Sbar += 2 * numpy.einsum('p,par->ar', g[begin:end], t)
            if with_k:
                r = H @ t
                if factor_response:
                    Sbar += 2 * numpy.sum(B @ r, axis=0)
                T[begin:end] = t.reshape(end - begin, -1)
                Rtilde[begin:end] = r.reshape(end - begin, -1)
        del B, t, packed
        if with_k:
            del r
    if end != naux:
        raise RuntimeError('DF occupied reverse did not cover all auxiliary rows')

    auxmol = dfobj.auxmol
    int2c = dfobj.mol._add_suffix('int2c2e')
    metric = numpy.asarray(auxmol.intor(int2c, hermi=1))
    with dense_blas_threads():
        low = scipy.linalg.cholesky(metric, lower=True, check_finite=False)
    del metric
    lowbar = numpy.zeros_like(low)
    a = b = numpy.zeros(naux)
    if with_j:
        with dense_blas_threads():
            a = scipy.linalg.solve_triangular(low.T, q, lower=False,
                                              check_finite=False)
            b = scipy.linalg.solve_triangular(low.T, g, lower=False,
                                              check_finite=False)
            lowbar -= numpy.outer(a, g) + numpy.outer(b, q)
    if with_k:
        # Each triangular solve requires all auxiliary rows but only a
        # bounded column slab; overwrite the R panels with L^-T R.
        solve_memory = _memory_mb('PYSCFAD_SOLVE_MEMORY_MB', _SOLVE_MEMORY_MB)
        width = max(1, int(solve_memory * 1024**2 / (3 * 8 * naux)))
        for begin in range(0, Rtilde.shape[1], width):
            end = min(begin + width, Rtilde.shape[1])
            with dense_blas_threads():
                z = scipy.linalg.solve_triangular(
                    low.T, numpy.asarray(Rtilde[:, begin:end]),
                    lower=False, check_finite=False)
                lowbar -= 2 * z @ numpy.asarray(T[:, begin:end]).T
            Rtilde[:, begin:end] = z
            del z
    del low, q, g

    with dense_blas_threads():
        D = S @ S.T

    # For real symmetric J inputs, form 2 K + J and symmetrize once at the
    # end. This avoids repeated full-density copies and broadcast products.
    # Keep the original arithmetic for other dtypes or nonexact symmetry;
    # hermi_sum must not introduce conjugation into those cases.
    fused_density = (
        all(x.dtype == numpy.float64 for x in (S, G, H, D))
        and (not with_j or (numpy.array_equal(G, G.T)
                            and numpy.array_equal(D, D.T))))
    doubled_st = (numpy.ascontiguousarray(2 * S.T)
                  if fused_density and with_k else None)
    packed_density = fused_density and _cderi_vjp._symmetric_pack is not None
    # Packing symmetric G,D once gives twice their off-diagonal entries and
    # their unchanged diagonals, precisely the native coordinate weighting.
    packed_jd_columns = (
        _cderi_vjp._pack_tril_symmetric_sum(numpy.stack((G, D))).T
        if packed_density and with_j else None)
    jd_columns = None

    def fill_packed_density(begin, end, out):
        # X=2 R S.T contributes X+X.T off diagonal and X on diagonal.
        # Contract J directly in packed space, avoiding its AO-square block.
        if with_k:
            with dense_blas_threads():
                panel = numpy.asarray(Rtilde[begin:end]).reshape(-1, rank)
                density = (panel @ doubled_st).reshape(end - begin, nao, nao)
            _cderi_vjp._pack_tril_symmetric_sum(density, out=out)
            del density, panel
        if with_j:
            coefficients = numpy.array((a[begin:end], b[begin:end]), order='F')
            # This rank-two update is memory bound. A large BLAS team costs
            # more here and competes with the following OpenMP contraction.
            with threadpool_limits(limits=1, user_api='blas'):
                updated = scipy.linalg.blas.dgemm(
                    1., packed_jd_columns, coefficients, beta=float(with_k),
                    c=out.T, overwrite_c=1)
            # BLAS wrappers may copy their output; preserve the fill contract.
            if not numpy.shares_memory(updated, out):
                out[:] = updated.T
        elif not with_k:
            out.fill(0.)

    def read_density(begin, end):
        nonlocal jd_columns
        # The coordinate integral contracts both AO legs. Supply 2 Z, where
        # Z=L^-T A is the symmetric raw-integral adjoint.
        if fused_density:
            with dense_blas_threads():
                if with_k:
                    panel = numpy.asarray(Rtilde[begin:end]).reshape(-1, rank)
                    density = (panel @ doubled_st).reshape(end - begin, nao, nao)
                else:
                    density = numpy.zeros((end - begin, nao, nao))
                if with_j:
                    if jd_columns is None:
                        jd_columns = numpy.asfortranarray(
                            numpy.stack((G.ravel(), D.ravel()), axis=1))
                    coefficients = numpy.array((a[begin:end], b[begin:end]), order='F')
                    updated = scipy.linalg.blas.dgemm(
                        1., jd_columns, coefficients, beta=1.,
                        c=density.reshape(end - begin, -1).T, overwrite_c=1)
                    # BLAS may copy its output argument; always use its return.
                    density = updated.T.reshape(end - begin, nao, nao)
            return lib.hermi_sum(density, axes=(0, 2, 1), inplace=True)
        with dense_blas_threads():
            if with_k:
                panel = numpy.asarray(Rtilde[begin:end]).reshape(end - begin, nao, rank)
                density = (panel.reshape(-1, rank) @ S.T).reshape(
                    end - begin, nao, nao)
                density += density.swapaxes(1, 2).copy()
            else:
                density = numpy.zeros((end - begin, nao, nao))
            if with_j:
                density += a[begin:end, None, None] * G
                density += b[begin:end, None, None] * D
            density *= 2
        return density

    molbar, auxbar = _cderi_vjp._int3c_mo_deriv_coords_vjp_from_z_reader(
        dfobj.mol, auxmol, None, None, (0, nao, 0, nao),
        int3c=dfobj.mol._add_suffix('int3c2e'),
        read_density_aux_block=read_density,
        fill_packed_density_aux_block=fill_packed_density if packed_density else None,
    )

    def metric_cholesky(aux):
        return jax_scipy.linalg.cholesky(aux.intor(int2c, hermi=1), lower=True)

    _, metric_pullback = jax.vjp(metric_cholesky, auxmol)
    _cderi_vjp._zero_strict_upper_inplace(lowbar)
    auxbar = _cderi_vjp._tree_add(auxbar, metric_pullback(np.asarray(lowbar))[0])
    leaves, tree = tree_flatten(dfobj)
    bars = [numpy.zeros_like(leaf) for leaf in leaves]
    bars[0] = molbar.coords
    bars[1] = auxbar.coords
    _cderi_vjp._profile_msg(
        f'occupied-factor DF J/K reverse rank={rank} '
        f'done {time.perf_counter() - started:.2f} s')
    return tree_unflatten(tree, bars), Sbar


get_jk_from_occ.defvjp(_forward, _backward)
