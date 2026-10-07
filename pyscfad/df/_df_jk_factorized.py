"""First-order DF J/K reverse for an explicitly factored real density.

The input is S, with D=S S.T. Its reverse never constructs the generic
AO-square density cotangent or the auxiliary-by-AO-pair integral cotangent.
Coordinate work uses thin occupied factors and one bounded integral pass.
"""

from functools import partial
from contextlib import contextmanager
import time

import jax
from jax import custom_vjp
from jax import scipy as jax_scipy
from jax.tree_util import tree_flatten, tree_unflatten
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.df import df_jk as pyscf_df_jk

from pyscfad import numpy as np
from pyscfad.df import _cderi_vjp
from pyscfad.lib._threading import dense_blas_threads

_FACTOR_PANEL_MEMORY_MB = 256.0
_SOLVE_MEMORY_MB = 64.0


@contextmanager
def _factor_panels(naux, nao, rank, with_k):
    """Keep small thin panels in RAM; spill large ones to owned scratch."""
    if not with_k:
        yield None, None
        return
    shape = (naux, nao * rank)
    if 2 * numpy.prod(shape) * 8 <= _FACTOR_PANEL_MEMORY_MB * 1024**2:
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
        width = max(1, int(_SOLVE_MEMORY_MB * 1024**2 / (3 * 8 * naux)))
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

    def read_density(begin, end):
        # The coordinate integral contracts both AO legs. Supply 2 Z, where
        # Z=L^-T A is the symmetric raw-integral adjoint.
        with dense_blas_threads():
            if with_k:
                panel = numpy.asarray(Rtilde[begin:end]).reshape(end - begin, nao, rank)
                density = panel @ S.T
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
