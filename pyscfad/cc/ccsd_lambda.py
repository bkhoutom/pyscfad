# Copyright 2021-2025 Xing Zhang
#
# Licensed under the Apache License, Version 2.0

"""
Restricted CCSD lambda equations for in-core JAX arrays.

This module mirrors the PySCF RCCSD lambda update, but keeps all intermediates
as JAX-visible arrays.  The implementation is intentionally in-core; it is used
by the experimental DF-CCSD custom response path.
"""

from functools import partial, reduce
import jax
import numpy
from jax.interpreters import ad as jax_ad

from pyscfad import numpy as np
from pyscfad import lib
from pyscfad.lib import logger
from pyscfad.tools import resource_profile


class _IMDS:
    pass


_IMDS_FIELDS = ('w1', 'w2', 'w3', 'w4',
                'woooo', 'wvooo', 'wvvov', 'wVOov', 'wvOOv')

# Cap the two complementary ovvv tiles at roughly 128 MiB.  The second
# wvvov output index can use a wider tile because it does not enlarge ovvv.
_DF_OVVV_MAX_BLKSIZE = 32
_DF_WVVOV_B_BLKSIZE = 32
_DF_OVVV_TILE_BYTES = 128 << 20


def _imds_flatten(imds):
    return tuple(getattr(imds, k, None) for k in _IMDS_FIELDS), None


def _imds_unflatten(_aux, children):
    imds = _IMDS()
    for k, v in zip(_IMDS_FIELDS, children):
        setattr(imds, k, v)
    return imds


jax.tree_util.register_pytree_node(_IMDS, _imds_flatten, _imds_unflatten)


def _get_ovvv_full(eris):
    """Unpack the (ov|vv) integral block to shape (nocc, nvir, nvir, nvir).

    Skips ``eris.get_ovvv``'s optimized numpy path so the result is
    JAX-traceable inside a jit boundary. Triggers the lazy ``eris.ovvv``
    build if Lov/Lvv are stored but ovvv has not been materialized yet.
    """
    if eris.ovvv is None and hasattr(eris, 'get_ovvv_packed'):
        eris.get_ovvv_packed()
    ovvv_packed = eris.ovvv
    nocc, nvir = ovvv_packed.shape[:2]
    ovvv = lib.unpack_tril(
        ovvv_packed.reshape(nocc * nvir, ovvv_packed.shape[-1])
    )
    return ovvv.reshape(nocc, nvir, nvir, nvir)


def _has_df_ovvv_factors(eris):
    return (eris.ovvv is None
            and getattr(eris, 'Lov', None) is not None
            and getattr(eris, 'Lvv', None) is not None)


def _df_ovvv_blksize(eris):
    _, nocc, nvir = eris.Lov.shape
    itemsize = numpy.dtype(eris.Lov.dtype).itemsize
    # _df_ovvv_tiles holds ovvv_a and ovvv_b simultaneously.
    bytes_per_a = max(1, 2 * nocc * nvir * nvir * itemsize)
    return max(1, min(nvir, _DF_OVVV_MAX_BLKSIZE,
                      _DF_OVVV_TILE_BYTES // bytes_per_a))


def _tril_pair_indices(first, second):
    first = numpy.asarray(first, dtype=numpy.intp)[:, None]
    second = numpy.asarray(second, dtype=numpy.intp)[None, :]
    hi = numpy.maximum(first, second)
    lo = numpy.minimum(first, second)
    return hi * (hi + 1) // 2 + lo


def _df_ovvv_tiles(eris, p0, p1):
    """Return complementary ovvv tiles from packed DF factors.

    ``ovvv_a[j,a,b,c]`` slices the first (MO-pair) virtual index ``a``;
    ``ovvv_b[j,c,a,b]`` slices the first virtual index of the packed-vv
    factor.  Together they provide every permutation needed by the lambda
    equations without constructing full ``ovvv``.
    """
    Lov = eris.Lov
    Lvv = eris.Lvv
    naux, nocc, nvir = Lov.shape
    bw = p1 - p0

    Lov_tile = Lov[:, :, p0:p1]
    ovvv_packed = np.einsum('xja,xq->ajq', Lov_tile, Lvv)
    ovvv_a = lib.unpack_tril(
        ovvv_packed.reshape(bw * nocc, -1)
    ).reshape(bw, nocc, nvir, nvir).transpose(1, 0, 2, 3)

    pair = _tril_pair_indices(numpy.arange(p0, p1), numpy.arange(nvir))
    L_ab = np.take(Lvv, pair.ravel(), axis=1).reshape(naux, bw, nvir)
    ovvv_b = np.einsum('xjc,xab->jcab', Lov, L_ab)
    # eris_vvov = eris_ovvv.transpose(2, 3, 0, 1)
    vvov_a = ovvv_b.transpose(2, 3, 0, 1)
    return ovvv_a, ovvv_b, vvov_a


def _voov_composite_tiles(t2, eris, p0, p1):
    """Build the ``VOov``/``vOOv`` intermediates for leading-virtual tile."""
    eris_voov = np.transpose(eris.ovvo, (1, 0, 3, 2))
    voov_a = eris_voov[p0:p1]
    g2voov_a = voov_a * 2 - np.transpose(voov_a, (0, 2, 1, 3))
    VOov_a = np.einsum('bjld,klcd->bjkc', g2voov_a, t2)
    VOov_a -= np.einsum('bjld,kldc->bjkc', voov_a, t2)
    VOov_a += voov_a
    vOOv_a = np.einsum('bljd,kldc->bjkc', voov_a, t2)
    vOOv_a -= np.transpose(eris.oovv, (2, 1, 0, 3))[p0:p1]
    return VOov_a, vOOv_a


def _has_tracer(*xs):
    leaves = []
    for x in xs:
        if x is None:
            continue
        leaves.extend(jax.tree_util.tree_leaves(x))
    return any(_contains_tracer(x) for x in leaves)


def _contains_tracer(x):
    if isinstance(x, (jax.core.Tracer, jax_ad.JVPTracer)):
        return True
    for attr in ('primal', 'val'):
        if hasattr(x, attr):
            try:
                if _contains_tracer(getattr(x, attr)):
                    return True
            except AttributeError:
                pass
    return False


def _make_intermediates_df(mycc, t1, t2, eris):
    """DF lambda intermediates without global ``ovvv`` or ``wvvov``."""
    nocc, nvir = t1.shape
    foo = eris.fock[:nocc, :nocc]
    fov = eris.fock[:nocc, nocc:]
    fvv = eris.fock[nocc:, nocc:]

    imds = _IMDS()
    w1 = fvv - np.einsum('ja,jb->ba', fov, t1)
    w2 = foo + np.einsum('ib,jb->ij', fov, t1)
    w3 = np.einsum('kc,jkbc->bj', fov, t2) * 2 + np.transpose(fov)
    w3 -= np.einsum('kc,kjbc->bj', fov, t2)
    w3 += np.einsum('kc,kb,jc->bj', fov, t1, t1)
    w4 = fov

    woooo = 0
    wvooo = np.zeros((nvir, nocc, nocc, nocc), dtype=t1.dtype)
    wVOov = np.zeros((nvir, nocc, nocc, nvir), dtype=t1.dtype)
    wvOOv = np.zeros_like(wVOov)

    theta_occ = t2 * 2 - np.transpose(t2, (1, 0, 2, 3))
    tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
    blksize = _df_ovvv_blksize(eris)
    for p0 in range(0, nvir, blksize):
        p1 = min(p0 + blksize, nvir)
        ovvv_a, ovvv_b, vvov_a = _df_ovvv_tiles(eris, p0, p1)

        # w1[:,a] = 2 (jc|ba) t_jc - (ja|bc) t_jc
        w1 = w1.at[:, p0:p1].add(
            np.einsum('jcab,jc->ba', ovvv_b, t1 * 2)
            - np.einsum('jabc,jc->ba', ovvv_a, t1)
        )
        # w3[b,j] += theta[j,k,c,d] (kd|cb)
        w3 = w3.at[p0:p1, :].add(
            np.einsum('jkcd,kdac->aj', theta_occ, ovvv_b)
        )
        wVOov = wVOov.at[p0:p1].set(
            np.einsum('jbcd,kd->bjkc', ovvv_a, t1)
        )
        wvOOv = wvOOv.at[p0:p1].set(
            np.einsum('cbjd,kd->cjkb', vvov_a, -t1)
        )
        g2vovv = np.transpose(vvov_a, (0, 2, 1, 3)) * 2
        g2vovv -= np.transpose(vvov_a, (0, 2, 3, 1))
        wvooo = wvooo.at[p0:p1].add(
            np.einsum('cibd,jkbd->ckij', g2vovv, tau)
        )

    eris_ovoo = eris.ovoo
    w2 += np.einsum('kbij,kb->ij', eris_ovoo, t1) * 2
    w2 -= np.einsum('ibkj,kb->ij', eris_ovoo, t1)
    theta = np.transpose(t2, (1, 0, 2, 3)) * 2 - t2
    w3 -= np.einsum('lckj,klcb->bj', eris_ovoo, theta)

    tmp = np.einsum('lc,jcik->ijkl', t1, eris_ovoo)
    woooo += tmp
    woooo += np.transpose(tmp, (1, 0, 3, 2))

    wvOOv += np.einsum('lbjk,lc->bjkc', eris_ovoo, t1)
    wVOov -= np.einsum('jbkl,lc->bjkc', eris_ovoo, t1)
    wvooo += np.transpose(eris_ovoo, (1, 3, 2, 0)) * 2
    wvooo -= np.transpose(eris_ovoo, (1, 0, 2, 3))
    wvooo -= np.einsum('klbc,iblj->ckij', t2, eris_ovoo * 1.5)

    g2ovoo = eris_ovoo * 2 - np.transpose(eris_ovoo, (2, 1, 0, 3))
    theta = t2 * 2 - np.transpose(t2, (1, 0, 2, 3))
    vcjik = np.einsum('jlcb,lbki->cjki', theta, g2ovoo)
    wvooo += np.transpose(vcjik, (0, 3, 2, 1))
    wvooo -= vcjik * .5

    eris_voov = np.transpose(eris.ovvo, (1, 0, 3, 2))
    tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
    woooo += np.einsum('cijd,klcd->ijkl', eris_voov, tau)

    g2voov = eris_voov * 2 - np.transpose(eris_voov, (0, 2, 1, 3))
    tmpw4 = np.einsum('ckld,ld->kc', g2voov, t1)
    w1 -= np.einsum('ckja,kjcb->ba', g2voov, t2)
    w1 -= np.einsum('ja,jb->ba', tmpw4, t1)
    w2 += np.einsum('jkbc,bikc->ij', t2, g2voov)
    w2 += np.einsum('ib,jb->ij', tmpw4, t1)
    w3 += reduce(np.dot, (np.transpose(t1), tmpw4, np.transpose(t1)))
    w4 += tmpw4

    wvOOv += np.einsum('bljd,kd,lc->bjkc', eris_voov, t1, t1)
    wVOov -= np.einsum('bjld,kd,lc->bjkc', eris_voov, t1, t1)

    for p0 in range(0, nvir, blksize):
        p1 = min(p0 + blksize, nvir)
        VOov_a, vOOv_a = _voov_composite_tiles(t2, eris, p0, p1)
        wVOov = wVOov.at[p0:p1].add(VOov_a)
        wvOOv = wvOOv.at[p0:p1].add(vOOv_a)

        ov1_a = vOOv_a * 2 + VOov_a
        ov2_a = VOov_a * 2 + vOOv_a
        wvooo -= np.einsum(
            'ja,aikc->ckij', t1[:, p0:p1], ov1_a
        )
        wvooo += np.einsum(
            'ka,aijc->ckij', t1[:, p0:p1], ov2_a
        )
        w3 += np.einsum(
            'akjb,ka->bj', ov2_a, t1[:, p0:p1]
        )
    imds.wVOov = wVOov
    imds.wvOOv = wvOOv

    # wvvov is O(nocc*nvir**3).  update_lambda reconstructs one (a,b)
    # tile at a time and immediately contracts it with l2.
    imds.wvvov = None

    woooo += np.transpose(eris.oooo, (0, 2, 1, 3))
    imds.woooo = woooo
    imds.wvooo = wvooo

    w3 += np.einsum('bc,jc->bj', w1, t1)
    w3 -= np.einsum('kj,kb->bj', w2, t1)

    imds.w1 = w1
    imds.w2 = w2
    imds.w3 = w3
    imds.w4 = w4
    return imds


#@partial(jax.jit, static_argnums=0)
def make_intermediates(mycc, t1, t2, eris):
    if _has_df_ovvv_factors(eris):
        return _make_intermediates_df(mycc, t1, t2, eris)

    nocc, nvir = t1.shape
    foo = eris.fock[:nocc, :nocc]
    fov = eris.fock[:nocc, nocc:]
    fvv = eris.fock[nocc:, nocc:]

    imds = _IMDS()
    eris_ovvv = _get_ovvv_full(eris)
    eris_vvov = np.transpose(eris_ovvv, (2, 3, 0, 1))

    w1 = fvv - np.einsum('ja,jb->ba', fov, t1)
    w2 = foo + np.einsum('ib,jb->ij', fov, t1)
    w3 = np.einsum('kc,jkbc->bj', fov, t2) * 2 + np.transpose(fov)
    w3 -= np.einsum('kc,kjbc->bj', fov, t2)
    w3 += np.einsum('kc,kb,jc->bj', fov, t1, t1)
    w4 = fov

    woooo = 0
    wvooo = np.zeros((nvir, nocc, nocc, nocc), dtype=t1.dtype)

    w1 += np.einsum('jcba,jc->ba', eris_ovvv, t1 * 2)
    w1 -= np.einsum('jabc,jc->ba', eris_ovvv, t1)
    theta = t2 * 2 - np.transpose(t2, (1, 0, 2, 3))
    w3 += np.einsum('jkcd,kdcb->bj', theta, eris_ovvv)
    wVOov = np.einsum('jbcd,kd->bjkc', eris_ovvv, t1)
    wvOOv = np.einsum('cbjd,kd->cjkb', eris_vvov, -t1)
    g2vovv = np.transpose(eris_vvov, (0, 2, 1, 3)) * 2
    g2vovv -= np.transpose(eris_vvov, (0, 2, 3, 1))
    tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
    wvooo += np.einsum('cibd,jkbd->ckij', g2vovv, tau)

    wvvov = np.einsum('jabd,jkcd->abkc', eris_ovvv, t2) * -1.5
    wvvov += np.transpose(eris_vvov, (0, 3, 2, 1)) * 2
    wvvov -= eris_vvov

    g2vvov = eris_vvov * 2 - np.transpose(eris_ovvv, (1, 2, 0, 3))
    theta = t2 * 2 - np.transpose(t2, (0, 1, 3, 2))
    vackb = np.einsum('acjd,kjbd->ackb', g2vvov, theta)
    wvvov += np.transpose(vackb, (0, 3, 2, 1))
    wvvov -= vackb * .5

    eris_ovoo = eris.ovoo
    w2 += np.einsum('kbij,kb->ij', eris_ovoo, t1) * 2
    w2 -= np.einsum('ibkj,kb->ij', eris_ovoo, t1)
    theta = np.transpose(t2, (1, 0, 2, 3)) * 2 - t2
    w3 -= np.einsum('lckj,klcb->bj', eris_ovoo, theta)

    tmp = np.einsum('lc,jcik->ijkl', t1, eris_ovoo)
    woooo += tmp
    woooo += np.transpose(tmp, (1, 0, 3, 2))

    wvOOv += np.einsum('lbjk,lc->bjkc', eris_ovoo, t1)
    wVOov -= np.einsum('jbkl,lc->bjkc', eris_ovoo, t1)
    wvooo += np.transpose(eris_ovoo, (1, 3, 2, 0)) * 2
    wvooo -= np.transpose(eris_ovoo, (1, 0, 2, 3))
    wvooo -= np.einsum('klbc,iblj->ckij', t2, eris_ovoo * 1.5)

    g2ovoo = eris_ovoo * 2 - np.transpose(eris_ovoo, (2, 1, 0, 3))
    theta = t2 * 2 - np.transpose(t2, (1, 0, 2, 3))
    vcjik = np.einsum('jlcb,lbki->cjki', theta, g2ovoo)
    wvooo += np.transpose(vcjik, (0, 3, 2, 1))
    wvooo -= vcjik * .5

    eris_voov = np.transpose(eris.ovvo, (1, 0, 3, 2))
    tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
    woooo += np.einsum('cijd,klcd->ijkl', eris_voov, tau)

    g2voov = eris_voov * 2 - np.transpose(eris_voov, (0, 2, 1, 3))
    tmpw4 = np.einsum('ckld,ld->kc', g2voov, t1)
    w1 -= np.einsum('ckja,kjcb->ba', g2voov, t2)
    w1 -= np.einsum('ja,jb->ba', tmpw4, t1)
    w2 += np.einsum('jkbc,bikc->ij', t2, g2voov)
    w2 += np.einsum('ib,jb->ij', tmpw4, t1)
    w3 += reduce(np.dot, (np.transpose(t1), tmpw4, np.transpose(t1)))
    w4 += tmpw4

    wvOOv += np.einsum('bljd,kd,lc->bjkc', eris_voov, t1, t1)
    wVOov -= np.einsum('bjld,kd,lc->bjkc', eris_voov, t1, t1)

    VOov = np.einsum('bjld,klcd->bjkc', g2voov, t2)
    VOov -= np.einsum('bjld,kldc->bjkc', eris_voov, t2)
    VOov += eris_voov
    vOOv = np.einsum('bljd,kldc->bjkc', eris_voov, t2)
    vOOv -= np.transpose(eris.oovv, (2, 1, 0, 3))
    wVOov += VOov
    wvOOv += vOOv
    imds.wVOov = wVOov
    imds.wvOOv = wvOOv

    ov1 = vOOv * 2 + VOov
    ov2 = VOov * 2 + vOOv
    wvooo -= np.einsum('jb,bikc->ckij', t1, ov1)
    wvooo += np.einsum('kb,bijc->ckij', t1, ov2)
    w3 += np.einsum('ckjb,kc->bj', ov2, t1)

    wvvov += np.einsum('ajkc,jb->abkc', ov1, t1)
    wvvov -= np.einsum('ajkb,jc->abkc', ov2, t1)

    g2ovoo = eris_ovoo * 2 - np.transpose(eris_ovoo, (2, 1, 0, 3))
    tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
    wvvov += np.einsum('laki,klbc->abic', g2ovoo, tau)
    imds.wvvov = wvvov

    woooo += np.transpose(eris.oooo, (0, 2, 1, 3))
    imds.woooo = woooo
    imds.wvooo = wvooo

    w3 += np.einsum('bc,jc->bj', w1, t1)
    w3 -= np.einsum('kj,kb->bj', w2, t1)

    imds.w1 = w1
    imds.w2 = w2
    imds.w3 = w3
    imds.w4 = w4
    return imds


def _contract_wvvov_l2_df(t1, t2, l2, eris):
    """Evaluate ``wvvov[a,b,k,c] l2[k,i,b,c]`` in DF tiles."""
    nocc, nvir = t1.shape
    eris_ovoo = eris.ovoo
    g2ovoo = eris_ovoo * 2 - np.transpose(eris_ovoo, (2, 1, 0, 3))
    tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
    theta = t2 * 2 - np.transpose(t2, (0, 1, 3, 2))

    out = np.zeros_like(t1)
    blksize = _df_ovvv_blksize(eris)
    b_blksize = max(1, min(_DF_WVVOV_B_BLKSIZE, nvir))
    for a0 in range(0, nvir, blksize):
        a1 = min(a0 + blksize, nvir)
        ovvv_a, _, vvov_a = _df_ovvv_tiles(eris, a0, a1)
        g2vvov_a = vvov_a * 2 - np.transpose(ovvv_a, (1, 2, 0, 3))
        VOov_a, vOOv_a = _voov_composite_tiles(t2, eris, a0, a1)
        ov1_a = vOOv_a * 2 + VOov_a
        ov2_a = VOov_a * 2 + vOOv_a

        for b0 in range(0, nvir, b_blksize):
            b1 = min(b0 + b_blksize, nvir)
            wtile = np.einsum(
                'jabd,jkcd->abkc', ovvv_a[:, :, b0:b1, :], t2
            ) * -1.5
            wtile += np.transpose(
                vvov_a[:, :, :, b0:b1], (0, 3, 2, 1)
            ) * 2
            wtile -= vvov_a[:, b0:b1, :, :]

            # +vackb.transpose(0,3,2,1) -.5*vackb, evaluated only
            # for the requested output-b tile.
            vackb_t = np.einsum(
                'acjd,kjbd->ackb',
                g2vvov_a, theta[:, :, b0:b1, :]
            )
            wtile += np.transpose(vackb_t, (0, 3, 2, 1))
            wtile -= np.einsum(
                'abjd,kjcd->abkc',
                g2vvov_a[:, b0:b1, :, :], theta
            ) * .5

            wtile += np.einsum(
                'ajkc,jb->abkc',
                ov1_a, t1[:, b0:b1]
            )
            wtile -= np.einsum(
                'ajkb,jc->abkc',
                ov2_a[:, :, :, b0:b1], t1
            )
            wtile += np.einsum(
                'laki,klbc->abic',
                g2ovoo[:, a0:a1, :, :], tau[:, :, b0:b1, :]
            )
            out = out.at[:, a0:a1].add(
                np.einsum(
                    'abkc,kibc->ia',
                    wtile, l2[:, :, b0:b1, :]
                )
            )
    return out


#@partial(jax.jit, static_argnums=0)
def update_lambda(mycc, t1, t2, l1, l2, eris=None, imds=None):
    if imds is None:
        imds = make_intermediates(mycc, t1, t2, eris)
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc, nocc:]
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift

    theta = t2 * 2 - np.transpose(t2, (0, 1, 3, 2))
    mba = np.einsum('klca,klcb->ba', l2, theta)
    mij = np.einsum('ikcd,jkcd->ij', l2, theta)
    mba1 = np.einsum('jc,jb->bc', l1, t1) + mba
    mij1 = np.einsum('kb,jb->kj', l1, t1) + mij
    mia1 = t1 + np.einsum('kc,jkbc->jb', l1, t2) * 2
    mia1 -= np.einsum('kc,jkcb->jb', l1, t2)
    mia1 -= reduce(np.dot, (t1, np.transpose(l1), t1))
    mia1 -= np.einsum('bd,jd->jb', mba, t1)
    mia1 -= np.einsum('lj,lb->jb', mij, t1)

    l2new = mycc._add_vvvv(None, l2, eris, with_ovvv=False, t2sym='jiba')
    l1new = np.einsum('ijab,jb->ia', l2new, t1) * 2
    l1new -= np.einsum('jiab,jb->ia', l2new, t1)
    l2new *= .5

    w1 = imds.w1 - np.diag(mo_e_v)
    w2 = imds.w2 - np.diag(mo_e_o)

    l1new += fov
    l1new += np.einsum('ib,ba->ia', l1, w1)
    l1new -= np.einsum('ja,ij->ia', l1, w2)
    l1new -= np.einsum('ik,ka->ia', mij, imds.w4)
    l1new -= np.einsum('ca,ic->ia', mba, imds.w4)
    l1new += np.einsum('ijab,bj->ia', l2, imds.w3) * 2
    l1new -= np.einsum('ijba,bj->ia', l2, imds.w3)

    l2new += np.einsum('ia,jb->ijab', l1, imds.w4)
    l2new += np.einsum('jibc,ca->jiba', l2, w1)
    l2new -= np.einsum('jk,kiba->jiba', w2, l2)

    eris_ovoo = eris.ovoo
    l1new -= np.einsum('iajk,kj->ia', eris_ovoo, mij1) * 2
    l1new += np.einsum('jaik,kj->ia', eris_ovoo, mij1)
    l2new -= np.einsum('jbki,ka->jiba', eris_ovoo, l1)

    tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
    l2tau = np.einsum('ijcd,klcd->ijkl', l2, tau)
    l2t1 = np.einsum('jidc,kc->ijkd', l2, t1)

    l1new -= np.einsum('jb,jiab->ia', l1, eris.oovv)

    if _has_df_ovvv_factors(eris):
        blksize = _df_ovvv_blksize(eris)
        for p0 in range(0, nvir, blksize):
            p1 = min(p0 + blksize, nvir)
            ovvv_a, ovvv_b, _ = _df_ovvv_tiles(eris, p0, p1)
            l1new = l1new.at[:, p0:p1].add(
                np.einsum('iabc,bc->ia', ovvv_a, mba1) * 2
                - np.einsum('ibac,bc->ia', ovvv_b, mba1)
            )
            l2new = l2new.at[:, :, :, p0:p1].add(
                np.einsum('jbac,ic->jiba', ovvv_b, l1)
            )
            m4 = np.einsum('ijkd,kadb->ijab', l2t1, ovvv_a)
            l2new = l2new.at[:, :, p0:p1, :].add(-m4)
            l1new = l1new.at[:, p0:p1].add(
                -np.einsum('ijab,jb->ia', m4, t1) * 2
                + np.einsum('jiab,jb->ia', m4, t1)
            )
            l1new -= np.einsum(
                'ijab,ia->jb', m4, t1[:, p0:p1]
            ) * 2
            l1new += np.einsum(
                'jiab,ia->jb', m4, t1[:, p0:p1]
            )
    else:
        eris_ovvv = _get_ovvv_full(eris)
        l1new += np.einsum('iabc,bc->ia', eris_ovvv, mba1) * 2
        l1new -= np.einsum('ibca,bc->ia', eris_ovvv, mba1)
        l2new += np.einsum('jbac,ic->jiba', eris_ovvv, l1)
        m4 = np.einsum('ijkd,kadb->ijab', l2t1, eris_ovvv)
        l2new -= m4
        l1new -= np.einsum('ijab,jb->ia', m4, t1) * 2
        l1new -= np.einsum('ijab,ia->jb', m4, t1) * 2
        l1new += np.einsum('jiab,jb->ia', m4, t1)
        l1new += np.einsum('jiab,ia->jb', m4, t1)

    eris_voov = np.transpose(eris.ovvo, (1, 0, 3, 2))
    l1new += np.einsum('jb,aijb->ia', l1, eris_voov) * 2
    l2new += np.transpose(eris_voov, (1, 2, 0, 3)) * .5
    l2new -= np.einsum('bjic,ca->jiba', eris_voov, mba1)
    l2new -= np.einsum('bjka,ik->jiba', eris_voov, mij1)
    l1new += np.einsum('aijb,jb->ia', eris_voov, mia1) * 2
    l1new -= np.einsum('bija,jb->ia', eris_voov, mia1)
    m4 = np.einsum('ijkl,aklb->ijab', l2tau, eris_voov)
    l2new += m4 * .5
    l1new += np.einsum('ijab,jb->ia', m4, t1) * 2
    l1new -= np.einsum('ijba,jb->ia', m4, t1)

    l1new -= np.einsum('ckij,jkca->ia', imds.wvooo, l2)
    if imds.wvvov is None and _has_df_ovvv_factors(eris):
        l1new += _contract_wvvov_l2_df(t1, t2, l2, eris)
    else:
        l1new += np.einsum('abkc,kibc->ia', imds.wvvov, l2)

    tmp_voov = imds.wVOov * 2 + imds.wvOOv
    tmp = np.transpose(l2, (0, 2, 1, 3))
    tmp -= np.transpose(l2, (0, 3, 1, 2)) * .5
    l2new += np.einsum('iakc,bjkc->jiba', tmp, tmp_voov)

    tmp = np.einsum('jkca,bikc->jiba', l2, imds.wvOOv)
    l2new += tmp
    l2new += np.transpose(tmp, (1, 0, 2, 3)) * .5

    m3 = np.einsum('ijkl,klab->ijab', imds.woooo, l2)
    l2new += m3 * .5
    l1new += np.einsum('ijab,jb->ia', m3, t1) * 2
    l1new -= np.einsum('ijba,jb->ia', m3, t1)

    eia = mo_e_o[:, None] - mo_e_v[None, :]
    l1new /= eia
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    l2new = l2new + np.transpose(l2new, (1, 0, 3, 2))
    l2new /= eijab
    return l1new, l2new


def kernel(mycc, eris, t1, t2, max_cycle=50, tol=1e-8, verbose=None):
    """Solve the standard CCSD lambda equations (bar_e = 1, no amplitude
    cotangent source).

    For the response gradient used by the DF-CCSD custom VJP, see
    :func:`solve_response_lambda` which handles arbitrary energy and amplitude
    cotangents.
    """
    log = logger.new_logger(mycc, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())

    if isinstance(mycc.diis, lib.diis.DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
        adiis.space = mycc.diis_space
    else:
        adiis = None

    imds = make_intermediates(mycc, t1, t2, eris)
    name = mycc.__class__.__name__
    cput0 = log.timer(f'{name} lambda initialization', *cput0)

    l1, l2 = t1, t2
    conv = False
    for istep in range(max_cycle):
        l1new, l2new = update_lambda(mycc, t1, t2, l1, l2, eris, imds)
        normt = np.linalg.norm(
            mycc.amplitudes_to_vector(l1new, l2new)
            - mycc.amplitudes_to_vector(l1, l2)
        )
        l1, l2 = l1new, l2new
        l1new = l2new = None
        l1, l2 = mycc.run_diis(l1, l2, istep, normt, 0, adiis)
        if not _has_tracer(normt):
            normt_val = float(normt)
            log.info('cycle = %d  norm(lambda1,lambda2) = %.6g',
                     istep + 1, normt_val)
            cput0 = log.timer(f'{name} lambda iter', *cput0)
            if normt_val < tol:
                conv = True
                break
    return conv, l1, l2


def solve_response_lambda(mycc, eris, t1, t2, bar_e, bar_t1, bar_t2,
                          max_cycle=50, tol=1e-8, verbose=None):
    """Solve the CCSD response lambda equation in implicit-diff form.

    Returns the vector ``lambda_vec`` (in the same packing as
    ``mycc.amplitudes_to_vector``) satisfying

        lambda_vec @ (d update_amps/dt - I) = -(bar_e * dE/dt + bar_t),

    so that ``lambda_vec @ d Omega/d eris`` gives the response contribution to
    the ERI cotangent (where ``Omega(t, eris) = update_amps(t, eris) - t``).
    This is the implicit-diff form of the CCSD lambda equations; it reduces to
    the standard CCSD lambda multipliers when ``bar_e = 1`` and the amplitude
    cotangents vanish.

    Picard iteration with optional DIIS acceleration.  Inputs are expected to
    be concrete arrays; the iteration uses a Python convergence check.
    """
    profile_total = resource_profile.start()
    log = logger.new_logger(mycc, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())

    amp = mycc.amplitudes_to_vector(t1, t2)
    nocc, nvir = t1.shape

    def update_fn(amp_):
        t1_, t2_ = mycc.vector_to_amplitudes(amp_)
        t1new, t2new = mycc.update_amps(t1_, t2_, eris)
        return mycc.amplitudes_to_vector(t1new, t2new)

    # Linearize update_amps at the converged amplitudes; vjp_fn(u) returns
    # u @ d(update_amps)/dt.
    profile_build_vjp = resource_profile.start()
    _, vjp_fn = jax.vjp(update_fn, amp)
    resource_profile.finish(
        'ccsd_response.build_update_vjp',
        profile_build_vjp,
        nocc=nocc,
        nvir=nvir,
        response_vector_shape=tuple(amp.shape),
        response_vector_mib=resource_profile.estimated_array_mib(amp),
    )

    # Build the source: bar_e * dE/dt + bar_t (all as a single vector).
    profile_source = resource_profile.start()
    source = np.zeros_like(amp)
    if not _is_scalar_zero(bar_e):
        def scaled_energy(amp_):
            t1_, t2_ = mycc.vector_to_amplitudes(amp_)
            return np.real(bar_e * mycc.energy(t1_, t2_, eris))

        source = source + jax.grad(scaled_energy)(amp)
    if bar_t1 is not None or bar_t2 is not None:
        if bar_t1 is None:
            bar_t1 = np.zeros_like(t1)
        if bar_t2 is None:
            bar_t2 = np.zeros_like(t2)
        source = source + _amplitudes_cotangent_to_vector(bar_t1, bar_t2)
    resource_profile.finish(
        'ccsd_response.build_source',
        profile_source,
        nocc=nocc,
        nvir=nvir,
        source_mib=resource_profile.estimated_array_mib(source),
    )

    cput0 = log.timer(f'{mycc.__class__.__name__} response init', *cput0)

    if isinstance(mycc.diis, lib.diis.DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
        adiis.space = mycc.diis_space
    else:
        adiis = None

    lambda_vec = np.array(amp, copy=True)
    conv = False
    for istep in range(max_cycle):
        profile_iteration = resource_profile.start()
        profile_iteration_vjp = resource_profile.start()
        (lambda_df,) = vjp_fn(lambda_vec)
        resource_profile.finish(
            'ccsd_response.iteration_vjp',
            profile_iteration_vjp,
            iteration=istep + 1,
            nocc=nocc,
            nvir=nvir,
        )
        profile_iteration_update = resource_profile.start()
        lambda_new = lambda_df + source
        diff = mycc.amplitudes_to_vector(
            *mycc.vector_to_amplitudes(lambda_new - lambda_vec)
        )
        normt = np.linalg.norm(diff)
        lambda_vec = lambda_new
        if adiis is not None:
            l1_iter, l2_iter = mycc.vector_to_amplitudes(lambda_vec)
            l1_iter, l2_iter = mycc.run_diis(
                l1_iter, l2_iter, istep, normt, 0, adiis
            )
            lambda_vec = mycc.amplitudes_to_vector(l1_iter, l2_iter)
        normt_val = None
        if not _has_tracer(normt):
            normt_val = float(normt)
            log.info(
                'cycle = %d  norm(response lambda) = %.6g',
                istep + 1, normt_val,
            )
            cput0 = log.timer(
                f'{mycc.__class__.__name__} response iter', *cput0,
            )
        resource_profile.finish(
            'ccsd_response.iteration_update_diis',
            profile_iteration_update,
            iteration=istep + 1,
            norm=normt_val,
            diis=adiis is not None,
        )
        resource_profile.finish(
            'ccsd_response.iteration_total',
            profile_iteration,
            iteration=istep + 1,
            norm=normt_val,
            response_vector_mib=resource_profile.estimated_array_mib(
                lambda_vec
            ),
        )
        if normt_val is not None and normt_val < tol:
            conv = True
            break

    resource_profile.finish(
        'ccsd_response.total',
        profile_total,
        nocc=nocc,
        nvir=nvir,
        iterations=istep + 1 if max_cycle else 0,
        converged=conv,
        response_vector_mib=resource_profile.estimated_array_mib(lambda_vec),
    )
    return conv, lambda_vec


def _is_scalar_zero(x):
    try:
        return bool(np.asarray(x) == 0)
    except Exception:
        return False


def _amplitudes_cotangent_to_vector(bar_t1, bar_t2):
    """Pack amplitude cotangents into the same vector layout as
    ``ccsd.amplitudes_to_vector``.

    ``ccsd.amplitudes_to_vector`` reads t2 from a lower-triangular pair index
    ``[i*nv+a, j*nv+b]`` with ``i*nv+a >= j*nv+b``; each off-diagonal entry of
    t2 is sampled once, so its cotangent must sum contributions from both
    ``(i,a,j,b)`` and the symmetric partner ``(j,b,i,a)``.  Diagonal entries
    contribute only once.
    """
    nocc, nvir = bar_t1.shape
    nov = nocc * nvir
    bar_t2_mat = np.transpose(bar_t2, (0, 2, 1, 3)).reshape(nov, nov)
    idx = numpy.tril_indices(nov)
    lower = bar_t2_mat[idx]
    upper = np.transpose(bar_t2_mat)[idx]
    bar_t2_vec = np.where(idx[0] == idx[1], lower, lower + upper)
    return np.concatenate((bar_t1.ravel(), bar_t2_vec), axis=None)
