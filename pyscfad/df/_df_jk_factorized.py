"""First-order DF J/K reverse for an explicitly factored real density.

The input is S, with D=S S.T. Its reverse never constructs the generic
AO-square density cotangent or the auxiliary-by-AO-pair integral cotangent.
Coordinate work uses thin occupied factors and one bounded integral pass.

Retained panels, transformation blocks and solve/read slabs share the DF
workspace, bounded by total-process headroom. PYSCFAD_FACTOR_PANEL_MEMORY_MB
and PYSCFAD_SOLVE_MEMORY_MB are optional MiB caps; unset values use the shared
allowance. The standalone panel helper retains its legacy defaults.
"""

from functools import partial
from contextlib import contextmanager, ExitStack
import math
import os
import time

import jax
from jax import custom_vjp
from jax.custom_derivatives import CustomVJPPrimal, SymbolicZero
from jax import scipy as jax_scipy
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.df import df_jk as pyscf_df_jk
from pyscf.gto.mole import AS_RINV_ORIG_ATOM

from pyscfad import numpy as np
from pyscfad.df import _cderi_vjp
from pyscfad.df import _jk_memory, _response_memory
from pyscfad.lib._threading import dense_blas_threads, blas_threads

_FACTOR_PANEL_MEMORY_MB = 256.0
_SOLVE_MEMORY_MB = 64.0


def _memory_mb(name, default):
    """Read a runtime budget without changing the fallback module constants."""
    from pyscfad.lib import _scf_response_cache
    automatic = (_scf_response_cache.automatic_budget_mb()
                 if name == 'PYSCFAD_FACTOR_PANEL_MEMORY_MB' else None)
    if automatic is not None:
        default = automatic
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        return default
    value = value if math.isfinite(value) and value >= 0 else default
    return value if automatic is None else min(value, automatic)


@contextmanager
def _factor_panels(naux, nao, rank, with_k, *, force_disk=False,
                   workspace=None, reservation=None, scratch_min=0):
    """Keep small thin panels in RAM; spill large ones to owned scratch."""
    if not with_k:
        yield None, None
        return
    shape = (naux, nao * rank)
    size = 2 * int(numpy.prod(shape)) * 8
    if workspace is None:
        budget = _memory_mb('PYSCFAD_FACTOR_PANEL_MEMORY_MB', _FACTOR_PANEL_MEMORY_MB) * 1024**2
    else:
        budget = workspace.available(panels=True, reserve=scratch_min)
    with ExitStack() as owned:
        if workspace is not None:
            reservation = reservation if reservation is not None else workspace.lease(panels=True)
            owned.callback(reservation.close)
        if not force_disk and size <= budget:
            if reservation is not None:
                reservation.resize(size)
            # Do not bind the primal T in this generator: after preparation
            # only the solved R closure should keep a panel alive.
            yield numpy.empty(shape), numpy.empty(shape)
        else:
            scratch = owned.enter_context(lib.H5TmpFile())
            chunks = (min(naux, 16), min(shape[1], 1024))
            yield (scratch.create_dataset('T', shape, dtype='f8', chunks=chunks),
                   scratch.create_dataset('R', shape, dtype='f8', chunks=chunks))


@partial(custom_vjp, nondiff_argnums=(2, 3, 4))
def get_jk_from_occ(dfobj, factors, with_j=True, with_k=True, factor_response=True):
    S = numpy.asarray(factors)
    dm = lib.tag_array(S @ S.T, mo_coeff=S,
                       mo_occ=numpy.ones(S.shape[1]))
    J, K = pyscf_df_jk.get_jk(_jk_memory.forward_df(dfobj), dm, hermi=1,
                             with_j=with_j, with_k=with_k)
    zero = numpy.zeros_like(dm)
    return J if with_j else zero, K if with_k else zero


def _forward(dfobj, factors, with_j, with_k, factor_response):
    def active(value):
        return any(isinstance(x, CustomVJPPrimal) and x.perturbed
                   for x in tree_flatten(value)[0])
    activity = (active(factors), active(dfobj.mol), active(dfobj.auxmol))
    unwrap = lambda x: x.value if isinstance(x, CustomVJPPrimal) else x
    dfobj, factors = tree_map(unwrap, dfobj), unwrap(factors)
    value = get_jk_from_occ(dfobj, factors, with_j, with_k, factor_response)
    return value, (dfobj, factors, activity)


def _backward(with_j, with_k, factor_response, residual, cotangents, *,
              geometry_activity=(True, True)):
    started = time.perf_counter()
    dfobj, factors = residual[:2]
    if len(residual) == 3:
        factor_active, mol_active, aux_active = residual[2]
        factor_response = factor_response and factor_active
        geometry_activity = (geometry_activity[0] and mol_active,
                             geometry_activity[1] and aux_active)
    with_j = with_j and not isinstance(cotangents[0], SymbolicZero)
    with_k = with_k and not isinstance(cotangents[1], SymbolicZero)
    if not (with_j or with_k):
        return None, None
    zero = numpy.zeros((factors.shape[0], factors.shape[0]))
    cotangents = (cotangents[0] if with_j else zero,
                  cotangents[1] if with_k else zero)
    if not any(geometry_activity):
        return None, (factor_pullback(dfobj, factors, *cotangents,
                                     with_j=with_j, with_k=with_k)
                      if factor_response else None)

    from . import _ooc_cotangent
    deferred = _ooc_cotangent.defer_backward(
        with_j, with_k, factor_response, (dfobj, factors), cotangents,
        geometry_activity=geometry_activity)
    if deferred is not None:
        return deferred
    contribution, sbar = _prepare_backward(
        with_j, with_k, factor_response, (dfobj, factors), cotangents)
    try:
        geometry = contribution.contract()
        if not geometry_activity[0]:
            geometry.mol = tree_map(lambda _: None, dfobj.mol)
        if not geometry_activity[1]:
            geometry.auxmol = tree_map(lambda _: None, dfobj.auxmol)
        _cderi_vjp._profile_msg(
            f'occupied-factor DF J/K reverse rank={factors.shape[1]} '
            f'done {time.perf_counter() - started:.2f} s')
        return geometry, sbar
    finally:
        contribution.close()


class _GeometryContribution:
    """Owned, reusable thin-panel seed for a later coordinate contraction."""

    def __init__(self, dfobj, rank, lowbar, read_density, fill_packed_density,
                 disk_panels):
        self.dfobj = dfobj
        self.nao = dfobj.mol.nao
        self.naux = dfobj.get_naoaux()
        self.rank = rank
        self.lowbar = lowbar
        self.read_density_aux_block = read_density
        self.fill_packed_density_aux_block = fill_packed_density
        self.disk_panels = disk_panels
        self._resources = None
        self.closed = False

    def contract(self):
        """Evaluate this seed without consuming it or closing its scratch."""
        return contract_geometry((self,))

    def close(self):
        """Release scratch and retained arrays, including on failed contraction."""
        if self.closed:
            return
        self.closed = True
        try:
            if self._resources is not None:
                self._resources.close()
        finally:
            self._resources = None
            self.lowbar = None
            self.read_density_aux_block = None
            self.fill_packed_density_aux_block = None


def _prepare_backward(with_j, with_k, factor_response, residual, cotangents, *,
                      force_disk=False, workspace=None, fixed_reservation=None):
    """Return a geometry seed and immediate factor bar; caller closes the seed."""
    dfobj, factors = residual[:2]
    nao, rank = factors.shape
    naux = dfobj.get_naoaux()
    workspace = workspace or _response_memory.workspace_for(dfobj)
    resources = ExitStack()
    try:
        fixed = resources.enter_context(
            fixed_reservation if fixed_reservation is not None else workspace.lease())
        fixed.resize(_response_memory.prepare_fixed_bytes(nao, naux, rank))
        panel_reservation = workspace.lease(panels=True)
        row_bytes = _response_memory.transform_row_bytes(nao, rank)
        scratch_min = max(row_bytes * min(16, dfobj.blockdim, naux), 3 * 8 * naux)
        panels = resources.enter_context(_factor_panels(
            naux, nao, rank, with_k, force_disk=force_disk,
            workspace=workspace, reservation=panel_reservation, scratch_min=scratch_min))
        contribution, sbar = _prepare_reverse(
            with_j, with_k, factor_response, (dfobj, factors), cotangents, *panels,
            workspace=workspace)
        panel_reservation.resize(naux * nao * rank * 8
                                 if with_k and isinstance(panels[1], numpy.ndarray) else 0)
        fixed.resize(_response_memory.retained_fixed_bytes(nao, naux, rank))
        contribution.workspace = workspace
        contribution._resources = resources
        return contribution, sbar
    except BaseException:
        resources.close()
        raise


def factor_pullback(dfobj, S, G, H, with_j=True, with_k=True):
    """Return the occupied-factor bar without any metric or geometry response."""
    S, G, H = map(numpy.asarray, (S, G, H))
    G, H = (G + G.T) * .5, (H + H.T) * .5
    sbar = numpy.zeros_like(S)
    if not (with_j or with_k):
        return sbar
    nao = S.shape[0]
    workspace = _response_memory.workspace_for(dfobj)
    row_bytes = _response_memory.transform_row_bytes(nao, S.shape[1])
    rows = workspace.block(row_bytes, min(dfobj.blockdim, dfobj.get_naoaux()), alignment=16)
    with workspace.lease(rows * row_bytes):
        for packed in dfobj.loop(rows):
            b = lib.unpack_tril(packed)
            with dense_blas_threads():
                t = b @ S
                if with_j:
                    g = numpy.einsum('ab,pab->p', G, b)
                    sbar += 2 * numpy.einsum('p,par->ar', g, t)
                if with_k:
                    sbar += 2 * numpy.sum(b @ (H @ t), axis=0)
    return sbar


def _reverse(with_j, with_k, factor_response, residual, cotangents, T, Rtilde):
    """Compatibility entry for callers that own their panel lifetime."""
    contribution, sbar = _prepare_reverse(
        with_j, with_k, factor_response, residual, cotangents, T, Rtilde)
    try:
        return contribution.contract(), sbar
    finally:
        contribution.close()


def _prepare_reverse(with_j, with_k, factor_response, residual, cotangents, T, Rtilde,
                     *, workspace=None):
    dfobj, factors = residual
    S = numpy.asarray(factors)
    G, H = (numpy.asarray(bar) for bar in cotangents)
    G, H = (G + G.T) * .5, (H + H.T) * .5
    nao, rank = S.shape
    naux = dfobj.get_naoaux()
    workspace = workspace or _response_memory.workspace_for(dfobj)
    row_bytes = _response_memory.transform_row_bytes(nao, rank)
    rows = workspace.block(row_bytes, min(dfobj.blockdim, naux), alignment=16)
    _cderi_vjp._profile_msg(
        f'DF response transform rows={rows} workspace_bytes={workspace.limit} '
        f'retained_bytes={workspace.used} scratch_bytes={rows * row_bytes}')
    Sbar = numpy.zeros_like(S) if factor_response else None
    q = numpy.zeros(naux)
    g = numpy.zeros(naux)
    end = 0
    with workspace.lease(rows * row_bytes):
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
    _cderi_vjp._profile_msg(f'DF response transform complete naux={naux}')

    auxmol = dfobj.auxmol
    int2c = dfobj.mol._add_suffix('int2c2e')
    from . import _ooc_cotangent
    shared_metric = _ooc_cotangent.metric_scope(dfobj)
    if shared_metric is not None:
        low = shared_metric.metric_factor()
    else:
        metric = numpy.asarray(auxmol.intor(int2c, hermi=1))
        with dense_blas_threads():
            low = scipy.linalg.cholesky(metric, lower=True, check_finite=False)
        del metric
    lowbar = numpy.zeros_like(low, order='F')
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
        width = workspace.solve_width(naux, Rtilde.shape[1])
        _cderi_vjp._profile_msg(
            f'DF response solve columns={width} disk={not isinstance(Rtilde, numpy.ndarray)} '
            f'available_bytes={workspace.available()}')
        with workspace.lease(3 * 8 * naux * min(width, Rtilde.shape[1])):
            for begin in range(0, Rtilde.shape[1], width):
                end = min(begin + width, Rtilde.shape[1])
                with dense_blas_threads():
                    z = scipy.linalg.solve_triangular(
                        low.T, numpy.asarray(Rtilde[:, begin:end]),
                        lower=False, check_finite=False)
                    lowbar = scipy.linalg.blas.dgemm(
                        -2., z, numpy.asarray(T[:, begin:end]).T,
                        beta=1., c=lowbar, overwrite_c=1)
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
            with blas_threads(1):
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

    return _GeometryContribution(
        dfobj, rank, lowbar, read_density,
        fill_packed_density if packed_density else None,
        Rtilde is not None and not isinstance(Rtilde, numpy.ndarray)), Sbar


def contract_geometry(contributions):
    """Contract summed seeds for one shared DF source in a single block pass.

    The caller groups compatible integral sources and retains ownership of
    the contributions. No contribution or underlying cotangent is consumed.
    """
    contributions = tuple(contributions)
    if not contributions:
        raise ValueError('At least one geometry contribution is required')
    for contribution in contributions:
        if contribution.closed:
            raise RuntimeError('Cannot contract a closed geometry contribution')
    first = contributions[0]
    dfobj, nao, naux = first.dfobj, first.nao, first.naux
    auxmol = dfobj.auxmol
    for contribution in contributions[1:]:
        if contribution.nao != nao or contribution.naux != naux:
            raise ValueError('Geometry contributions have incompatible DF dimensions')
        for left, right in ((dfobj.mol, contribution.dfobj.mol),
                            (auxmol, contribution.dfobj.auxmol)):
            if (left.cart != right.cart
                    or not numpy.array_equal(left.coords, right.coords)
                    or not numpy.array_equal(left._atm, right._atm)
                    or not numpy.array_equal(left._bas, right._bas)
                    or not numpy.array_equal(left._env[:AS_RINV_ORIG_ATOM],
                                             right._env[:AS_RINV_ORIG_ATOM])
                    or not numpy.array_equal(left._env[AS_RINV_ORIG_ATOM+1:],
                                             right._env[AS_RINV_ORIG_ATOM+1:])):
                raise ValueError('Geometry contributions have incompatible integral sources')

    def read_density(begin, end):
        out = first.read_density_aux_block(begin, end)
        for contribution in contributions[1:]:
            out += contribution.read_density_aux_block(begin, end)
        return out

    def fill_packed_density(begin, end, out):
        first.fill_packed_density_aux_block(begin, end, out)
        if len(contributions) > 1:
            scratch = numpy.empty_like(out)
            for contribution in contributions[1:]:
                contribution.fill_packed_density_aux_block(begin, end, scratch)
                out += scratch

    packed_density = all(c.fill_packed_density_aux_block is not None
                         for c in contributions)
    workspace = getattr(first, 'workspace', None) or _response_memory.workspace_for(dfobj)
    # Account for contributions prepared independently instead of silently
    # borrowing the first contribution's unused workspace.
    with ExitStack() as scratch:
        for contribution in contributions:
            if getattr(contribution, 'workspace', None) is not workspace:
                panel = (0 if contribution.disk_panels else getattr(
                    contribution, 'retained_panel_bytes',
                    naux * getattr(contribution, 'compact_panel_columns',
                                   nao * contribution.rank) * 8))
                scratch.enter_context(workspace.lease(panel +
                    _response_memory.retained_fixed_bytes(nao, naux, contribution.rank)))
        scratch.enter_context(workspace.geometry_buffer(4 * 8 * naux**2))
        cap = _response_memory.override_bytes('PYSCFAD_DF_INT3C_MO_VJP_BLOCK_MB')
        available = workspace.available(cap=cap)
        panel_columns = max(getattr(c, 'mo_panel_columns',
                            getattr(c, 'compact_panel_columns', nao * c.rank))
                            for c in contributions)
        words_per_row = 6 * nao**2 + 3 * nao + 2 * max(0, panel_columns - nao**2)
        min_rows = max(1, int(numpy.diff(auxmol.ao_loc).max()))
        if available < min_rows * 8 * words_per_row:
            raise MemoryError('DF response workspace cannot fit one auxiliary shell for geometry')
        _cderi_vjp._profile_msg(
            f'DF response geometry available_bytes={available} retained_bytes={workspace.used}')
        rows = min(naux, available // (8 * words_per_row))
        block_bytes = rows * 8 * words_per_row
        with workspace.lease(block_bytes):
            molbar, auxbar = _cderi_vjp._int3c_mo_deriv_coords_vjp_from_z_reader(
                dfobj.mol, auxmol, None, None, (0, nao, 0, nao),
                int3c=dfobj.mol._add_suffix('int3c2e'),
                block_memory_mb=block_bytes / 1024**2,
                # Also constrain rows: the generic helper has a legacy
                # 1 MiB floor and does not know our retained panel ranks.
                z_aux_block_max_rows=rows,
                read_density_aux_block=read_density,
                fill_packed_density_aux_block=fill_packed_density if packed_density else None,
            )

        lowbar = first.lowbar.copy()
        for contribution in contributions[1:]:
            lowbar += contribution.lowbar
        int2c = dfobj.mol._add_suffix('int2c2e')

        def metric_cholesky(aux):
            return jax_scipy.linalg.cholesky(aux.intor(int2c, hermi=1), lower=True)

        from . import _ooc_cotangent
        shared_metric = _ooc_cotangent.metric_scope(dfobj)
        if shared_metric is None:
            _, metric_pullback = jax.vjp(metric_cholesky, auxmol)
        else:
            shared_metric.metric_factor()
            metric_pullback = shared_metric.metric_pullback
        _cderi_vjp._zero_strict_upper_inplace(lowbar)
        auxbar = _cderi_vjp._tree_add(auxbar, metric_pullback(np.asarray(lowbar))[0])
        leaves, tree = tree_flatten(dfobj)
        bars = [numpy.zeros_like(leaf) for leaf in leaves]
        bars[0] = molbar.coords
        bars[1] = auxbar.coords
        return tree_unflatten(tree, bars)


get_jk_from_occ.defvjp(_forward, _backward, symbolic_zeros=True)
