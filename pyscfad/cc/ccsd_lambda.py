# Copyright 2021-2025 Xing Zhang
#
# Licensed under the Apache License, Version 2.0

"""
Restricted CCSD lambda equations for in-core JAX arrays.

This module mirrors the PySCF RCCSD lambda update, but keeps all intermediates
as JAX-visible arrays.  The implementation is intentionally in-core; it is used
by the experimental DF-CCSD custom response path.
"""

from functools import reduce
import jax
from jax.interpreters import ad as jax_ad

from pyscfad import numpy as np
from pyscfad import lib


class _IMDS:
    pass


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


def make_intermediates(mycc, t1, t2, eris):
    nocc, nvir = t1.shape
    foo = eris.fock[:nocc, :nocc]
    fov = eris.fock[:nocc, nocc:]
    fvv = eris.fock[nocc:, nocc:]

    imds = _IMDS()
    eris_ovvv = eris.get_ovvv(slice(None), slice(None))
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

    eris_ovvv = eris.get_ovvv(slice(None), slice(None))
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


def kernel(mycc, eris, t1, t2, max_cycle=50, tol=1e-8):
    imds = make_intermediates(mycc, t1, t2, eris)
    l1, l2 = t1, t2
    conv = False
    for _ in range(max_cycle):
        l1new, l2new = update_lambda(mycc, t1, t2, l1, l2, eris, imds)
        normt = np.linalg.norm(mycc.amplitudes_to_vector(l1new - l1, l2new - l2))
        l1, l2 = l1new, l2new
        if not _has_tracer(normt) and float(normt) < tol:
            conv = True
            break
    return conv, l1, l2
