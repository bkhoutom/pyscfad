"""Optional geometry preparation using a validated compact exchange seed.

For H=Q C Q.T with diagonal C in {-1,+1}, native packed AO2MO forms
P_p=Q.T B_p S. Solving W=L^-T(C P) suffices: U_p=Q W_p and
bar(L)_K=-2 W_flat P_flat.T. Panels therefore have rank(H)*rank(S)
columns instead of nao*rank(S). Full-rank or unsupported seeds return None
for the ordinary factorized path. Factor/density cotangents remain the
caller's responsibility; this helper prepares geometry only.

Eigenvalues are removed only below the matrix's floating-point resolution,
with a complete reconstruction check. No adjustable rank tolerance is used.
The caller owns the returned contribution and must close it after contraction.
"""
from contextlib import ExitStack

import jax
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.ao2mo import _ao2mo

from pyscfad.df import _cderi_vjp, _df_jk_factorized as factorized
from pyscfad.lib._threading import dense_blas_threads, blas_threads


def _exchange_factors(h):
    """Return machine-precision signed factors, or None without a rank saving."""
    nao = h.shape[0]
    scale = numpy.max(numpy.abs(h))
    if scale == 0:
        return numpy.empty((nao, 0)), numpy.empty(0)
    with dense_blas_threads():
        eigenvalues, eigenvectors = scipy.linalg.eigh(h, check_finite=False)
        threshold = nao * numpy.finfo(numpy.float64).eps * numpy.max(numpy.abs(eigenvalues))
        keep = numpy.abs(eigenvalues) > threshold
        if numpy.count_nonzero(keep) >= nao:
            return None
        q = numpy.asarray(eigenvectors[:, keep] * numpy.sqrt(numpy.abs(eigenvalues[keep])),
                          order='F')
        signs = numpy.sign(eigenvalues[keep])
        residual = numpy.max(numpy.abs(h - (q * signs) @ q.T))
    # Check every matrix element, including entries outside the recovered range.
    if residual > 10 * nao * numpy.finfo(numpy.float64).eps * scale:
        return None
    return q, signs


def prepare_compact(with_j, with_k, residual, cotangents, *, force_disk=False):
    """Return an owned compact geometry contribution, or None for fallback."""
    if not (with_j or with_k):
        return None
    dfobj, factors = residual[:2]
    if any(isinstance(x, jax.core.Tracer) for x in
           jax.tree_util.tree_leaves((dfobj, factors, cotangents))):
        return None
    s = numpy.asarray(factors)
    if (s.ndim != 2 or s.dtype != numpy.float64 or not numpy.isfinite(s).all()
            or s.shape[0] != dfobj.mol.nao or s.shape[1] == 0):
        return None
    nao, occupied_rank = s.shape
    g = numpy.asarray(cotangents[0]) if with_j else numpy.zeros((nao, nao))
    h = numpy.asarray(cotangents[1]) if with_k else numpy.zeros((nao, nao))
    for seed in (g, h):
        if (seed.shape != (nao, nao) or seed.dtype != numpy.float64
                or not numpy.isfinite(seed).all()
                or not numpy.array_equal(seed, seed.T)):
            return None
    for mol in (dfobj.mol, dfobj.auxmol):
        if (mol is None or getattr(mol, 'coords', None) is None
                or len(jax.tree_util.tree_leaves(mol)) != 1):
            return None
    recovered = _exchange_factors(h) if with_k else (numpy.empty((nao, 0)), numpy.empty(0))
    if recovered is None:
        return None
    q_factor, signs = recovered
    exchange_rank = len(signs)
    active_k = exchange_rank > 0
    naux = dfobj.get_naoaux()
    if naux == 0:
        return None

    resources = ExitStack()
    try:
        p_panel, w_panel = resources.enter_context(factorized._factor_panels(
            naux, exchange_rank, occupied_rank, active_k, force_disk=force_disk))
        contribution = _prepare(
            dfobj, s, g, q_factor, signs, with_j, p_panel, w_panel)
        contribution._resources = resources
        return contribution
    except BaseException:
        resources.close()
        raise


def _prepare(dfobj, s, g, q_factor, signs, with_j, p_panel, w_panel):
    nao, occupied_rank = s.shape
    naux, exchange_rank = dfobj.get_naoaux(), len(signs)
    active_k = exchange_rank > 0
    with dense_blas_threads():
        density = s @ s.T
    jd_columns = None
    if with_j:
        # Weight lower triangles once so packed B gives full symmetric traces.
        weighted = lib.pack_tril(numpy.stack((g, density))) * 2
        diagonal = numpy.arange(nao)
        weighted[:, diagonal * (diagonal + 3) // 2] *= .5
        jd_columns = numpy.asarray(weighted.T, order='F')
    q_vec = numpy.zeros(naux)
    g_vec = numpy.zeros(naux)
    if active_k:
        coeff = numpy.asarray(numpy.hstack((q_factor, s)), order='F')
        orbs_slice = (0, exchange_rank, exchange_rank, exchange_rank + occupied_rank)
        column_signs = numpy.repeat(signs, occupied_rank)
    # The native projection consumes packed rows directly; no AO-square B block.
    npair = nao * (nao + 1) // 2
    words_per_row = npair + 2 * exchange_rank * occupied_rank
    rows = max(1, min(dfobj.blockdim, int(128 * 1024**2 / (8 * max(words_per_row, 1)))))
    end = 0
    for packed in dfobj.loop(rows):
        begin, end = end, end + len(packed)
        if with_j:
            with dense_blas_threads():
                vectors = packed @ jd_columns
            g_vec[begin:end], q_vec[begin:end] = vectors[:, 0], vectors[:, 1]
        if active_k:
            projected = _ao2mo.nr_e2(packed, coeff, orbs_slice, aosym='s2')
            p_panel[begin:end] = projected
            w_panel[begin:end] = projected * column_signs
    if end != naux:
        raise RuntimeError('DF compact reverse did not cover all auxiliary rows')

    int2c = dfobj.mol._add_suffix('int2c2e')
    metric = numpy.asarray(dfobj.auxmol.intor(int2c, hermi=1))
    with dense_blas_threads():
        low = scipy.linalg.cholesky(metric, lower=True, check_finite=False)
    del metric
    # BLAS accumulation avoids another auxiliary-square temporary per slab.
    lowbar = numpy.zeros((naux, naux), order='F')
    a = b = numpy.zeros(naux)
    if with_j:
        with dense_blas_threads():
            a = scipy.linalg.solve_triangular(low.T, q_vec, lower=False, check_finite=False)
            b = scipy.linalg.solve_triangular(low.T, g_vec, lower=False, check_finite=False)
            lowbar = scipy.linalg.blas.dger(-1., a, g_vec, a=lowbar, overwrite_a=1)
            lowbar = scipy.linalg.blas.dger(-1., b, q_vec, a=lowbar, overwrite_a=1)
    if active_k:
        budget = factorized._memory_mb('PYSCFAD_SOLVE_MEMORY_MB', factorized._SOLVE_MEMORY_MB)
        width = max(1, int(budget * 1024**2 / (3 * 8 * naux)))
        for begin in range(0, w_panel.shape[1], width):
            end = min(begin + width, w_panel.shape[1])
            with dense_blas_threads():
                transformed = scipy.linalg.solve_triangular(
                    low.T, numpy.asarray(w_panel[:, begin:end]), lower=False,
                    check_finite=False)
                lowbar = scipy.linalg.blas.dgemm(
                    -2., transformed, numpy.asarray(p_panel[:, begin:end]).T,
                    beta=1., c=lowbar, overwrite_c=1)
            w_panel[:, begin:end] = transformed
        del transformed
    del low, q_vec, g_vec
    doubled_st = numpy.asarray(2 * s.T, order='C')

    def exchange_density(begin, end):
        """Return 2 U S.T in a bounded block, before symmetric packing."""
        with dense_blas_threads():
            panel = numpy.asarray(w_panel[begin:end]).reshape(-1, exchange_rank, occupied_rank)
            u = q_factor @ panel
            return (u.reshape(-1, occupied_rank) @ doubled_st).reshape(end-begin, nao, nao)

    def read_density(begin, end):
        out = exchange_density(begin, end) if active_k else numpy.zeros((end-begin, nao, nao))
        if with_j:
            # Dense fallback supplies 2Z; J matrices are already symmetric.
            out += a[begin:end, None, None] * g
            out += b[begin:end, None, None] * density
        return lib.hermi_sum(out, axes=(0, 2, 1), inplace=True)

    def fill_packed_density(begin, end, out):
        if active_k:
            block = exchange_density(begin, end)
            _cderi_vjp._pack_tril_symmetric_sum(block, out=out)
        if with_j:
            coefficients = numpy.array((a[begin:end], b[begin:end]), order='F')
            with blas_threads(1):
                updated = scipy.linalg.blas.dgemm(
                    1., jd_columns, coefficients, beta=float(active_k),
                    c=out.T, overwrite_c=1)
            if not numpy.shares_memory(updated, out):
                out[:] = updated.T
        elif not active_k:
            out.fill(0.)

    contribution = factorized._GeometryContribution(
        dfobj, occupied_rank, lowbar, read_density,
        fill_packed_density if _cderi_vjp._symmetric_pack is not None else None,
        active_k and not isinstance(w_panel, numpy.ndarray))
    contribution.compact_exchange_rank = exchange_rank
    contribution.compact_panel_columns = exchange_rank * occupied_rank
    return contribution
