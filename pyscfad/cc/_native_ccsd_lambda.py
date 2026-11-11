# Copyright 2021-2026 The PySCFAD Authors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""In-memory NumPy/BLAS translation of the DF lambda contractions.

Derived from pyscfad.cc.ccsd_lambda (itself based on PySCF RCCSD lambda).
All updates use NumPy arrays. Virtual triples intermediates are built in
tiles and cached in RAM when affordable; no HDF5 scratch is created.
"""
from functools import reduce
import numpy as np
import numpy
from pyscf import lib
from ._native_blas import einsum

class _IMDS:
    pass

_DF_OVVV_MAX_BLKSIZE = 32
_DF_WVVOV_B_BLKSIZE = 32
_DF_OVVV_TILE_BYTES = 128 << 20
_WVVOV_CACHE_FRACTION = .25

def _get_ovvv_full(eris):
    """Unpack the (ov|vv) integral block to shape (nocc, nvir, nvir, nvir).

    Triggers the lazy ``eris.ovvv`` build if Lov/Lvv are stored but ovvv
    has not been materialized yet.
    """
    if eris.ovvv is None and hasattr(eris, 'get_ovvv_packed'):
        eris.get_ovvv_packed()
    ovvv_packed = eris.ovvv
    (nocc, nvir) = ovvv_packed.shape[:2]
    ovvv = lib.unpack_tril(ovvv_packed.reshape(nocc * nvir, ovvv_packed.shape[-1]))
    return ovvv.reshape(nocc, nvir, nvir, nvir)


def _has_df_ovvv_factors(eris):
    return getattr(eris, 'Lov', None) is not None and getattr(eris, 'Lvv', None) is not None


def _df_ovvv_blksize(eris):
    (_, nocc, nvir) = eris.Lov.shape
    itemsize = numpy.dtype(eris.Lov.dtype).itemsize
    bytes_per_a = max(1, 2 * nocc * nvir * nvir * itemsize)
    return max(1, min(nvir, _DF_OVVV_MAX_BLKSIZE, _DF_OVVV_TILE_BYTES // bytes_per_a))


def _tril_pair_indices(first, second):
    first = numpy.asarray(first, dtype=numpy.intp)[:, None]
    second = numpy.asarray(second, dtype=numpy.intp)[None, :]
    hi = numpy.maximum(first, second)
    lo = numpy.minimum(first, second)
    return hi * (hi + 1) // 2 + lo


def _df_ovvv_tiles(eris, p0, p1):
    """Return complementary ovvv tiles from the cache, or DF factors.

    ``ovvv_a[j,a,b,c]`` slices the first (MO-pair) virtual index ``a``;
    ``ovvv_b[j,c,a,b]`` slices the first virtual index of the packed-vv
    factor.  Together they provide every permutation needed by the lambda
    equations without constructing full ``ovvv``.
    """
    Lov = eris.Lov
    Lvv = eris.Lvv
    (naux, nocc, nvir) = Lov.shape
    bw = p1 - p0
    pair = _tril_pair_indices(numpy.arange(p0, p1), numpy.arange(nvir))
    if eris.ovvv is not None:
        ovvv_a = lib.unpack_tril(eris.ovvv[:, p0:p1].reshape(nocc * bw, -1)).reshape(nocc, bw, nvir, nvir)
        ovvv_b = np.take(eris.ovvv, pair.ravel(), axis=2).reshape(nocc, nvir, bw, nvir)
        return (ovvv_a, ovvv_b, ovvv_b.transpose(2, 3, 0, 1))
    Lov_tile = Lov[:, :, p0:p1]
    ovvv_packed = einsum('xja,xq->ajq', Lov_tile, Lvv)
    ovvv_a = lib.unpack_tril(ovvv_packed.reshape(bw * nocc, -1)).reshape(bw, nocc, nvir, nvir).transpose(1, 0, 2, 3)
    L_ab = np.take(Lvv, pair.ravel(), axis=1).reshape(naux, bw, nvir)
    ovvv_b = einsum('xjc,xab->jcab', Lov, L_ab)
    vvov_a = ovvv_b.transpose(2, 3, 0, 1)
    return (ovvv_a, ovvv_b, vvov_a)


def _voov_composite_tiles(t2, eris, p0, p1):
    """Build the ``VOov``/``vOOv`` intermediates for leading-virtual tile."""
    eris_voov = np.transpose(eris.ovvo, (1, 0, 3, 2))
    voov_a = eris_voov[p0:p1]
    g2voov_a = voov_a * 2 - np.transpose(voov_a, (0, 2, 1, 3))
    VOov_a = einsum('bjld,klcd->bjkc', g2voov_a, t2)
    VOov_a = VOov_a - einsum('bjld,kldc->bjkc', voov_a, t2)
    VOov_a = VOov_a + voov_a
    vOOv_a = einsum('bljd,kldc->bjkc', voov_a, t2)
    vOOv_a = vOOv_a - np.transpose(eris.oovv, (2, 1, 0, 3))[p0:p1]
    return (VOov_a, vOOv_a)


def _make_intermediates_df(mycc, t1, t2, eris):
    """DF lambda intermediates without full unpacked ``ovvv`` or ``wvvov``."""
    (nocc, nvir) = t1.shape
    foo = eris.fock[:nocc, :nocc]
    fov = eris.fock[:nocc, nocc:]
    fvv = eris.fock[nocc:, nocc:]
    imds = _IMDS()
    w1 = fvv - einsum('ja,jb->ba', fov, t1)
    w2 = foo + einsum('ib,jb->ij', fov, t1)
    w3 = einsum('kc,jkbc->bj', fov, t2) * 2 + np.transpose(fov)
    w3 = w3 - einsum('kc,kjbc->bj', fov, t2)
    w3 = w3 + einsum('kc,kb,jc->bj', fov, t1, t1)
    w4 = fov
    woooo = 0
    wvooo = np.zeros((nvir, nocc, nocc, nocc), dtype=t1.dtype)
    wVOov = np.zeros((nvir, nocc, nocc, nvir), dtype=t1.dtype)
    wvOOv = np.zeros_like(wVOov)
    theta_occ = t2 * 2 - np.transpose(t2, (1, 0, 2, 3))
    tau = t2 + einsum('ia,jb->ijab', t1, t1)
    blksize = _df_ovvv_blksize(eris)
    for p0 in range(0, nvir, blksize):
        p1 = min(p0 + blksize, nvir)
        (ovvv_a, ovvv_b, vvov_a) = _df_ovvv_tiles(eris, p0, p1)
        w1[:, p0:p1] += einsum('jcab,jc->ba', ovvv_b, t1 * 2) - einsum('jabc,jc->ba', ovvv_a, t1)
        w3[p0:p1, :] += einsum('jkcd,kdac->aj', theta_occ, ovvv_b)
        wVOov[p0:p1] = einsum('jbcd,kd->bjkc', ovvv_a, t1)
        wvOOv[p0:p1] = einsum('cbjd,kd->cjkb', vvov_a, -t1)
        g2vovv = np.transpose(vvov_a, (0, 2, 1, 3)) * 2
        g2vovv = g2vovv - np.transpose(vvov_a, (0, 2, 3, 1))
        wvooo[p0:p1] += einsum('cibd,jkbd->ckij', g2vovv, tau)
    eris_ovoo = eris.ovoo
    w2 = w2 + einsum('kbij,kb->ij', eris_ovoo, t1) * 2
    w2 = w2 - einsum('ibkj,kb->ij', eris_ovoo, t1)
    theta = np.transpose(t2, (1, 0, 2, 3)) * 2 - t2
    w3 = w3 - einsum('lckj,klcb->bj', eris_ovoo, theta)
    tmp = einsum('lc,jcik->ijkl', t1, eris_ovoo)
    woooo = woooo + tmp
    woooo = woooo + np.transpose(tmp, (1, 0, 3, 2))
    wvOOv = wvOOv + einsum('lbjk,lc->bjkc', eris_ovoo, t1)
    wVOov = wVOov - einsum('jbkl,lc->bjkc', eris_ovoo, t1)
    wvooo = wvooo + np.transpose(eris_ovoo, (1, 3, 2, 0)) * 2
    wvooo = wvooo - np.transpose(eris_ovoo, (1, 0, 2, 3))
    wvooo = wvooo - einsum('klbc,iblj->ckij', t2, eris_ovoo * 1.5)
    g2ovoo = eris_ovoo * 2 - np.transpose(eris_ovoo, (2, 1, 0, 3))
    theta = t2 * 2 - np.transpose(t2, (1, 0, 2, 3))
    vcjik = einsum('jlcb,lbki->cjki', theta, g2ovoo)
    wvooo = wvooo + np.transpose(vcjik, (0, 3, 2, 1))
    wvooo = wvooo - vcjik * 0.5
    eris_voov = np.transpose(eris.ovvo, (1, 0, 3, 2))
    tau = t2 + einsum('ia,jb->ijab', t1, t1)
    woooo = woooo + einsum('cijd,klcd->ijkl', eris_voov, tau)
    g2voov = eris_voov * 2 - np.transpose(eris_voov, (0, 2, 1, 3))
    tmpw4 = einsum('ckld,ld->kc', g2voov, t1)
    w1 = w1 - einsum('ckja,kjcb->ba', g2voov, t2)
    w1 = w1 - einsum('ja,jb->ba', tmpw4, t1)
    w2 = w2 + einsum('jkbc,bikc->ij', t2, g2voov)
    w2 = w2 + einsum('ib,jb->ij', tmpw4, t1)
    w3 = w3 + reduce(np.dot, (np.transpose(t1), tmpw4, np.transpose(t1)))
    w4 = w4 + tmpw4
    wvOOv = wvOOv + einsum('bljd,kd,lc->bjkc', eris_voov, t1, t1)
    wVOov = wVOov - einsum('bjld,kd,lc->bjkc', eris_voov, t1, t1)
    for p0 in range(0, nvir, blksize):
        p1 = min(p0 + blksize, nvir)
        (VOov_a, vOOv_a) = _voov_composite_tiles(t2, eris, p0, p1)
        wVOov[p0:p1] += VOov_a
        wvOOv[p0:p1] += vOOv_a
        ov1_a = vOOv_a * 2 + VOov_a
        ov2_a = VOov_a * 2 + vOOv_a
        wvooo = wvooo - einsum('ja,aikc->ckij', t1[:, p0:p1], ov1_a)
        wvooo = wvooo + einsum('ka,aijc->ckij', t1[:, p0:p1], ov2_a)
        w3 = w3 + einsum('akjb,ka->bj', ov2_a, t1[:, p0:p1])
    imds.wVOov = wVOov
    imds.wvOOv = wvOOv
    imds.wvvov = None
    cache_bytes = t1.dtype.itemsize * nocc * nvir**3
    available_bytes = max(0., mycc.max_memory-lib.current_memory()[0])*1e6
    if cache_bytes <= _WVVOV_CACHE_FRACTION * available_bytes:
        imds.wvvov = np.empty((nvir, nvir, nocc, nvir), dtype=t1.dtype)
        _contract_wvvov_l2_df(t1, t2, None, eris, store=imds.wvvov)
    woooo = woooo + np.transpose(eris.oooo, (0, 2, 1, 3))
    imds.woooo = woooo
    imds.wvooo = wvooo
    w3 = w3 + einsum('bc,jc->bj', w1, t1)
    w3 = w3 - einsum('kj,kb->bj', w2, t1)
    imds.w1 = w1
    imds.w2 = w2
    imds.w3 = w3
    imds.w4 = w4
    return imds


def _contract_wvvov_l2_df(t1, t2, l2, eris, *, store=None):
    """Build W tiles, either caching them or contracting immediately with l2."""
    (nocc, nvir) = t1.shape
    eris_ovoo = eris.ovoo
    g2ovoo = eris_ovoo * 2 - np.transpose(eris_ovoo, (2, 1, 0, 3))
    tau = t2 + einsum('ia,jb->ijab', t1, t1)
    theta = t2 * 2 - np.transpose(t2, (0, 1, 3, 2))
    out = np.zeros_like(t1)
    blksize = _df_ovvv_blksize(eris)
    b_blksize = max(1, min(_DF_WVVOV_B_BLKSIZE, nvir))
    for a0 in range(0, nvir, blksize):
        a1 = min(a0 + blksize, nvir)
        (ovvv_a, _, vvov_a) = _df_ovvv_tiles(eris, a0, a1)
        g2vvov_a = vvov_a * 2 - np.transpose(ovvv_a, (1, 2, 0, 3))
        (VOov_a, vOOv_a) = _voov_composite_tiles(t2, eris, a0, a1)
        ov1_a = vOOv_a * 2 + VOov_a
        ov2_a = VOov_a * 2 + vOOv_a
        for b0 in range(0, nvir, b_blksize):
            b1 = min(b0 + b_blksize, nvir)
            wtile = einsum('jabd,jkcd->abkc', ovvv_a[:, :, b0:b1, :], t2) * -1.5
            wtile = wtile + np.transpose(vvov_a[:, :, :, b0:b1], (0, 3, 2, 1)) * 2
            wtile = wtile - vvov_a[:, b0:b1, :, :]
            vackb_t = einsum('acjd,kjbd->ackb', g2vvov_a, theta[:, :, b0:b1, :])
            wtile = wtile + np.transpose(vackb_t, (0, 3, 2, 1))
            wtile = wtile - einsum('abjd,kjcd->abkc', g2vvov_a[:, b0:b1, :, :], theta) * 0.5
            wtile = wtile + einsum('ajkc,jb->abkc', ov1_a, t1[:, b0:b1])
            wtile = wtile - einsum('ajkb,jc->abkc', ov2_a[:, :, :, b0:b1], t1)
            wtile = wtile + einsum('laki,klbc->abic', g2ovoo[:, a0:a1, :, :], tau[:, :, b0:b1, :])
            if store is None:
                out[:, a0:a1] += einsum('abkc,kibc->ia', wtile, l2[:, :, b0:b1, :])
            else:
                store[a0:a1, b0:b1] = wtile
    return out


def update_lambda(mycc, t1, t2, l1, l2, eris=None, imds=None):
    if imds is None:
        imds = make_intermediates(mycc, t1, t2, eris)
    if imds.wvvov is not None:
        # Reuse PySCF's blocked native update with our resident intermediates.
        # Only its default intermediate builder uses HDF5; that builder is
        # never called here. The tiled fallback below needs no Wvvov cache.
        from pyscf.cc.ccsd_lambda import update_lambda as pyscf_update
        return pyscf_update(mycc, t1, t2, l1, l2, eris, imds)
    (nocc, nvir) = t1.shape
    fov = eris.fock[:nocc, nocc:]
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift
    theta = t2 * 2 - np.transpose(t2, (0, 1, 3, 2))
    mba = einsum('klca,klcb->ba', l2, theta)
    mij = einsum('ikcd,jkcd->ij', l2, theta)
    mba1 = einsum('jc,jb->bc', l1, t1) + mba
    mij1 = einsum('kb,jb->kj', l1, t1) + mij
    mia1 = t1 + einsum('kc,jkbc->jb', l1, t2) * 2
    mia1 = mia1 - einsum('kc,jkcb->jb', l1, t2)
    mia1 = mia1 - reduce(np.dot, (t1, np.transpose(l1), t1))
    mia1 = mia1 - einsum('bd,jd->jb', mba, t1)
    mia1 = mia1 - einsum('lj,lb->jb', mij, t1)
    l2new = mycc._add_vvvv(None, l2, eris, with_ovvv=False, t2sym='jiba')
    l1new = einsum('ijab,jb->ia', l2new, t1) * 2
    l1new = l1new - einsum('jiab,jb->ia', l2new, t1)
    l2new = l2new * 0.5
    w1 = imds.w1 - np.diag(mo_e_v)
    w2 = imds.w2 - np.diag(mo_e_o)
    l1new = l1new + fov
    l1new = l1new + einsum('ib,ba->ia', l1, w1)
    l1new = l1new - einsum('ja,ij->ia', l1, w2)
    l1new = l1new - einsum('ik,ka->ia', mij, imds.w4)
    l1new = l1new - einsum('ca,ic->ia', mba, imds.w4)
    l1new = l1new + einsum('ijab,bj->ia', l2, imds.w3) * 2
    l1new = l1new - einsum('ijba,bj->ia', l2, imds.w3)
    l2new = l2new + einsum('ia,jb->ijab', l1, imds.w4)
    l2new = l2new + einsum('jibc,ca->jiba', l2, w1)
    l2new = l2new - einsum('jk,kiba->jiba', w2, l2)
    eris_ovoo = eris.ovoo
    l1new = l1new - einsum('iajk,kj->ia', eris_ovoo, mij1) * 2
    l1new = l1new + einsum('jaik,kj->ia', eris_ovoo, mij1)
    l2new = l2new - einsum('jbki,ka->jiba', eris_ovoo, l1)
    tau = t2 + einsum('ia,jb->ijab', t1, t1)
    l2tau = einsum('ijcd,klcd->ijkl', l2, tau)
    l2t1 = einsum('jidc,kc->ijkd', l2, t1)
    l1new = l1new - einsum('jb,jiab->ia', l1, eris.oovv)
    if _has_df_ovvv_factors(eris):
        blksize = _df_ovvv_blksize(eris)
        for p0 in range(0, nvir, blksize):
            p1 = min(p0 + blksize, nvir)
            (ovvv_a, ovvv_b, _) = _df_ovvv_tiles(eris, p0, p1)
            l1new[:, p0:p1] += einsum('iabc,bc->ia', ovvv_a, mba1) * 2 - einsum('ibac,bc->ia', ovvv_b, mba1)
            l2new[:, :, :, p0:p1] += einsum('jbac,ic->jiba', ovvv_b, l1)
            m4 = einsum('ijkd,kadb->ijab', l2t1, ovvv_a)
            l2new[:, :, p0:p1, :] += -m4
            l1new[:, p0:p1] += -einsum('ijab,jb->ia', m4, t1) * 2 + einsum('jiab,jb->ia', m4, t1)
            l1new = l1new - einsum('ijab,ia->jb', m4, t1[:, p0:p1]) * 2
            l1new = l1new + einsum('jiab,ia->jb', m4, t1[:, p0:p1])
    else:
        eris_ovvv = _get_ovvv_full(eris)
        l1new = l1new + einsum('iabc,bc->ia', eris_ovvv, mba1) * 2
        l1new = l1new - einsum('ibca,bc->ia', eris_ovvv, mba1)
        l2new = l2new + einsum('jbac,ic->jiba', eris_ovvv, l1)
        m4 = einsum('ijkd,kadb->ijab', l2t1, eris_ovvv)
        l2new = l2new - m4
        l1new = l1new - einsum('ijab,jb->ia', m4, t1) * 2
        l1new = l1new - einsum('ijab,ia->jb', m4, t1) * 2
        l1new = l1new + einsum('jiab,jb->ia', m4, t1)
        l1new = l1new + einsum('jiab,ia->jb', m4, t1)
    eris_voov = np.transpose(eris.ovvo, (1, 0, 3, 2))
    l1new = l1new + einsum('jb,aijb->ia', l1, eris_voov) * 2
    l2new = l2new + np.transpose(eris_voov, (1, 2, 0, 3)) * 0.5
    l2new = l2new - einsum('bjic,ca->jiba', eris_voov, mba1)
    l2new = l2new - einsum('bjka,ik->jiba', eris_voov, mij1)
    l1new = l1new + einsum('aijb,jb->ia', eris_voov, mia1) * 2
    l1new = l1new - einsum('bija,jb->ia', eris_voov, mia1)
    m4 = einsum('ijkl,aklb->ijab', l2tau, eris_voov)
    l2new = l2new + m4 * 0.5
    l1new = l1new + einsum('ijab,jb->ia', m4, t1) * 2
    l1new = l1new - einsum('ijba,jb->ia', m4, t1)
    l1new = l1new - einsum('ckij,jkca->ia', imds.wvooo, l2)
    if imds.wvvov is None and _has_df_ovvv_factors(eris):
        l1new = l1new + _contract_wvvov_l2_df(t1, t2, l2, eris)
    else:
        l1new = l1new + einsum('abkc,kibc->ia', imds.wvvov, l2)
    tmp_voov = imds.wVOov * 2 + imds.wvOOv
    tmp = np.transpose(l2, (0, 2, 1, 3))
    tmp = tmp - np.transpose(l2, (0, 3, 1, 2)) * 0.5
    l2new = l2new + einsum('iakc,bjkc->jiba', tmp, tmp_voov)
    tmp = einsum('jkca,bikc->jiba', l2, imds.wvOOv)
    l2new = l2new + tmp
    l2new = l2new + np.transpose(tmp, (1, 0, 2, 3)) * 0.5
    m3 = einsum('ijkl,klab->ijab', imds.woooo, l2)
    l2new = l2new + m3 * 0.5
    l1new = l1new + einsum('ijab,jb->ia', m3, t1) * 2
    l1new = l1new - einsum('ijba,jb->ia', m3, t1)
    eia = mo_e_o[:, None] - mo_e_v[None, :]
    l1new = l1new / eia
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    l2new = l2new + np.transpose(l2new, (1, 0, 3, 2))
    l2new = l2new / eijab
    return (l1new, l2new)

make_intermediates = _make_intermediates_df
