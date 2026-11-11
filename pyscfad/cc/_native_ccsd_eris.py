# Copyright 2021-2026 The PySCFAD Authors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""NumPy/BLAS DF-CCSD integral adjoint and its in-memory intermediates.

Equations mirror cc.ccsd.update_amps and cc.dfccsd's explicit ERI adjoint.
This module performs no JAX operations and creates no disk scratch.
"""
import ctypes
import numpy as np
import numpy
from pyscf import lib
from ._native_blas import einsum
_CCSD_OVVV_BLKSIZE = 32

def _is_zero_cotangent(x):
    return x is None

def update_intermediates(mycc, t1, t2, eris, *, return_intermediates=False):
    """Evaluate the update once to prepare the native integral adjoint.

    Arrays saved here live only for this fragment's integral response.
    Forward convergence is performed by PySCF, not by this helper.
    """
    if mycc.cc2:
        raise NotImplementedError
    saved = {} if return_intermediates else None
    if return_intermediates and (mycc.dcsd or mycc.direct or getattr(eris, 'Lov', None) is None or (getattr(eris, 'Lvv', None) is None)):
        raise NotImplementedError('Saved update intermediates require ordinary DF-CCSD')
    (nocc, nvir) = t1.shape
    nvir_pair = nvir * (nvir + 1) // 2
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift
    t1new = np.zeros_like(t1)
    t2new = mycc._add_vvvv(t1, t2, eris, t2sym='jiba')
    t2new = t2new * 0.5
    fov = fock[:nocc, nocc:].copy()
    t1new = t1new + fov
    foo = fock[:nocc, :nocc] - np.diag(mo_e_o)
    foo = foo + 0.5 * einsum('ia,ja->ij', fock[:nocc, nocc:], t1)
    fvv = fock[nocc:, nocc:] - np.diag(mo_e_v)
    fvv = fvv - 0.5 * einsum('ia,ib->ab', t1, fock[:nocc, nocc:])
    use_df_tile = getattr(eris, 'Lov', None) is not None and getattr(eris, 'Lvv', None) is not None
    if use_df_tile:
        if mycc.direct:
            raise NotImplementedError('DF tiled ovvv update does not support direct AO mode')
        Lov_full = eris.Lov
        Lvv_full = eris.Lvv
        wooVV_flat = np.zeros((nocc, nocc * nvir_pair))
        wVOov = np.zeros((nvir, nocc, nocc, nvir))
        theta = t2.transpose(1, 2, 0, 3) * 2
        theta = theta - t2.transpose(0, 2, 1, 3)
        tau_for_vvvo = t2 + einsum('ia,jb->ijab', t1, t1)
        tmp_acc = np.zeros((nocc, nocc, nvir, nocc))
        for p0 in range(0, nvir, 32):
            p1 = min(p0 + 32, nvir)
            bw = p1 - p0
            packed = eris.ovvv[:, p0:p1].transpose(1, 0, 2)
            vovv = lib.unpack_tril(packed.reshape(bw * nocc, nvir_pair)).reshape(bw, nocc, nvir, nvir)
            wooVV_flat = wooVV_flat - np.dot(t1[:, p0:p1], packed.reshape(bw, -1))
            fvv = fvv + 2 * einsum('kc,ckab->ab', t1[:, p0:p1], vovv)
            fvv[:, p0:p1] = fvv[:, p0:p1] - einsum('kc,bkca->ab', t1, vovv)
            tmp_acc = tmp_acc + einsum('ijcd,cdbk->ijbk', tau_for_vvvo[:, :, p0:p1, :], vovv.transpose(0, 2, 3, 1))
            wVOov[p0:p1] = einsum('biac,jc->bija', vovv, t1)
            t1new = t1new + einsum('icjb,cjba->ia', theta[:, p0:p1], vovv)
        t2new = t2new - einsum('ka,ijbk->ijab', t1, tmp_acc)
        if saved is not None:
            saved.update(theta_ovvv=theta, tau_for_vvvo=tau_for_vvvo, tmp_acc=tmp_acc)
        tmp_acc = None
        tau_for_vvvo = None
        wooVV = lib.unpack_tril(wooVV_flat.reshape(nocc ** 2, nvir_pair))
        wVooV = wooVV.reshape(nocc, nocc, nvir, nvir).transpose(2, 1, 0, 3)
        wooVV_flat = None
    else:
        eris_vovv = eris.ovvv.transpose(1, 0, 2)
        wooVV = -np.dot(t1, eris_vovv.reshape(nvir, -1))
        eris_vovv = lib.unpack_tril(eris_vovv.reshape(nvir * nocc, nvir_pair))
        eris_vovv = eris_vovv.reshape(nvir, nocc, nvir, nvir)
        fvv = fvv + 2 * einsum('kc,ckab->ab', t1, eris_vovv)
        fvv = fvv - einsum('kc,bkca->ab', t1, eris_vovv)
        if not mycc.direct:
            vvvo = eris_vovv.transpose(0, 2, 3, 1)
            tau = t2 + einsum('ia,jb->ijab', t1, t1)
            tmp = einsum('ijcd,cdbk->ijbk', tau, vvvo)
            t2new = t2new - einsum('ka,ijbk->ijab', t1, tmp)
        wVOov = einsum('biac,jc->bija', eris_vovv, t1)
        theta = t2.transpose(1, 2, 0, 3) * 2
        theta = theta - t2.transpose(0, 2, 1, 3)
        t1new = t1new + einsum('icjb,cjba->ia', theta, eris_vovv)
        wooVV = lib.unpack_tril(wooVV.reshape(nocc ** 2, nvir_pair))
        wVooV = wooVV.reshape(nocc, nocc, nvir, nvir).transpose(2, 1, 0, 3)
    woooo = np.asarray(eris.oooo).transpose(0, 2, 1, 3).copy()
    eris_ovoo = eris.ovoo
    eris_oovv = eris.oovv
    foo = foo + einsum('kc,kcji->ij', 2 * t1, eris_ovoo)
    foo = foo + einsum('kc,icjk->ij', -t1, eris_ovoo)
    tmp = einsum('la,jaik->lkji', t1, eris_ovoo)
    woooo = woooo + (tmp + tmp.transpose(1, 0, 3, 2))
    wVOov = wVOov - einsum('jbik,ka->bjia', eris_ovoo, t1)
    t2new = t2new + wVOov.transpose(1, 2, 0, 3)
    wVooV = wVooV + einsum('kbij,ka->bija', eris_ovoo, t1)
    eris_ovvo = eris.ovvo
    t1new = t1new - einsum('jb,jiab->ia', t1, eris_oovv)
    wVooV = wVooV - eris_oovv.transpose(2, 0, 1, 3)
    wVOov = wVOov + wVooV * 0.5
    t2new = t2new + (eris_ovvo * 0.5).transpose(0, 3, 1, 2)
    eris_voov = eris_ovvo.conj().transpose(1, 0, 3, 2)
    t1new = t1new + 2 * einsum('jb,aijb->ia', t1, eris_voov)
    tmp = einsum('ic,kjbc->ibkj', t1, eris_oovv)
    tmp = tmp + einsum('bjkc,ic->jbki', eris_voov, t1)
    if saved is not None:
        saved['tmp_oovv_voov'] = tmp
    t2new = t2new - einsum('ka,jbki->jiba', t1, tmp)
    fov = fov + einsum('kc,aikc->ia', t1, eris_voov) * 2
    fov = fov - einsum('kc,akic->ia', t1, eris_voov)
    tau = einsum('ia,jb->ijab', t1 * 0.5, t1)
    if mycc.dcsd:
        tau = tau + t2 * 0.5
        theta = t2.transpose(1, 0, 2, 3) - t2 * 0.5
        fvv_t2 = -einsum('cjia,cjib->ab', theta.transpose(2, 1, 0, 3), eris_voov)
        foo_t2 = einsum('aikb,kjab->ij', eris_voov, theta)
    else:
        tau = tau + t2
    theta = tau.transpose(1, 0, 2, 3) * 2
    theta = theta - tau
    if saved is not None:
        saved['theta'] = theta
    fvv = fvv - einsum('cjia,cjib->ab', theta.transpose(2, 1, 0, 3), eris_voov)
    foo = foo + einsum('aikb,kjab->ij', eris_voov, theta)
    tau = einsum('ia,jb->ijab', t1, t1)
    if mycc.dcsd:
        woooo_t2 = einsum('ijab,aklb->ijkl', t2, eris_voov)
    else:
        tau = tau + t2
    woooo = woooo + einsum('ijab,aklb->ijkl', tau, eris_voov)
    if saved is not None:
        saved['tau_full'] = tau
    tau = einsum('ia,jb->ijab', t1, t1)
    if not mycc.dcsd:
        tau = tau + t2 * 0.5
    wVooV = wVooV + einsum('bkic,jkca->bija', eris_voov, tau)
    if saved is not None:
        saved['tau_half'] = tau
    tmp = einsum('jkca,ckib->jaib', t2, wVooV)
    t2new = t2new + tmp.transpose(2, 0, 1, 3)
    tmp = tmp * 0.5
    t2new = t2new + tmp.transpose(0, 2, 1, 3)
    wVOov = wVOov + eris_voov
    eris_VOov = -0.5 * eris_voov.transpose(0, 2, 1, 3)
    tau = t2.transpose(1, 3, 0, 2) * 2
    tau = tau - t2.transpose(0, 3, 1, 2)
    tau1 = -einsum('ia,jb->ibja', t1 * 2, t1)
    tau = tau + tau1
    if mycc.dcsd:
        wVOov = wVOov + 0.5 * einsum('aikc,kcjb->aijb', eris_voov, tau)
        wVOov = wVOov + 0.5 * einsum('aikc,kcjb->aijb', eris_VOov, tau1)
    else:
        eris_VOov = eris_VOov + eris_voov
        wVOov = wVOov + 0.5 * einsum('aikc,kcjb->aijb', eris_VOov, tau)
    if saved is not None:
        saved.update(tau_resp=tau, eris_VOov=eris_VOov)
    theta = t2 * 2
    theta = theta - t2.transpose(1, 0, 2, 3)
    if saved is not None:
        saved['theta_t2'] = theta
    t2new = t2new + einsum('kica,ckjb->ijab', theta, wVOov)
    theta = t2.transpose(1, 0, 2, 3) * 2 - t2
    if saved is not None:
        saved['theta_t1'] = theta
    t1new = t1new + einsum('jb,ijba->ia', fov, theta)
    t1new = t1new - einsum('jbki,kjba->ia', eris.ovoo, theta)
    tau = einsum('ia,jb->ijab', t1, t1)
    if mycc.dcsd:
        t2new = t2new + 0.5 * einsum('ijkl,klab->ijab', woooo_t2, tau)
    tau = tau + t2
    t2new = t2new + 0.5 * einsum('ijkl,klab->ijab', woooo, tau)
    if saved is not None:
        saved['tau_final'] = tau
    ft_ij = foo + einsum('ja,ia->ij', 0.5 * t1, fov)
    ft_ab = fvv - einsum('ia,ib->ab', 0.5 * t1, fov)
    t2new = t2new + einsum('ijac,bc->ijab', t2, ft_ab)
    t2new = t2new - einsum('ki,kjab->ijab', ft_ij, t2)
    if mycc.dcsd:
        fvv = fvv + fvv_t2
        foo = foo + foo_t2
    t1new = t1new + einsum('ib,ab->ia', t1, fvv)
    t1new = t1new - einsum('ja,ji->ia', t1, foo)
    t2new = t2new + t2new.transpose(1, 0, 3, 2)
    eia = mo_e_o[:, None] - mo_e_v
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    if saved is not None:
        saved.update(t1work=t1new, t2sym=t2new, eia=eia, eijab=eijab, fov=fov, foo=foo, fvv=fvv, ft_ij=ft_ij, ft_ab=ft_ab, woooo=woooo, wVOov=wVOov, wVooV=wVooV)
    t1new = t1new / eia
    t2new = t2new / eijab
    eia = eijab = None
    if saved is not None:
        return ((t1new, t2new), saved)
    return (t1new, t2new)

def _adjoint_unpack_tril_last2(bar_dense, n, filltriu=lib.HERMITIAN):
    (rows, cols) = numpy.tril_indices(n)
    if filltriu == lib.PLAIN:
        return bar_dense[..., rows, cols]
    if filltriu in (lib.HERMITIAN, lib.SYMMETRIC):
        lower_upper = bar_dense[..., rows, cols] + bar_dense[..., cols, rows]
        diag = numpy.asarray(rows == cols)
        diag_part = numpy.where(diag, bar_dense[..., rows, cols], 0)
        return lower_upper - diag_part
    raise NotImplementedError('unpack_tril adjoint only supports plain/symmetric')


def _amplitudes_to_vector_cotangent(lambda_vec, nocc, nvir):
    nov = nocc * nvir
    l1 = lambda_vec[:nov].reshape(nocc, nvir)
    packed = lambda_vec[nov:]
    mat = np.zeros((nov, nov), dtype=lambda_vec.dtype)
    idx = numpy.tril_indices(nov)
    mat[idx] = packed
    l2 = mat.reshape(nocc, nvir, nocc, nvir).transpose(0, 2, 1, 3)
    return (l1, l2)


def _einsum_operand_bars(subscripts, out_bar, *operands):
    (lhs, out_spec) = subscripts.replace(' ', '').split('->')
    in_specs = lhs.split(',')
    bars = []
    for (i, spec) in enumerate(in_specs):
        other_specs = [s for (j, s) in enumerate(in_specs) if j != i]
        other_vals = [v for (j, v) in enumerate(operands) if j != i]
        grad_subscripts = ','.join([out_spec] + other_specs)
        grad_subscripts = grad_subscripts + ('->' + spec)
        bars.append(einsum(grad_subscripts, out_bar, *other_vals))
    return bars


def _einsum_operand_bar(subscripts, out_bar, operands, operand_index):
    (lhs, out_spec) = subscripts.replace(' ', '').split('->')
    in_specs = lhs.split(',')
    other_specs = [s for (j, s) in enumerate(in_specs) if j != operand_index]
    other_vals = [v for (j, v) in enumerate(operands) if j != operand_index]
    grad_subscripts = ','.join([out_spec] + other_specs)
    grad_subscripts = grad_subscripts + ('->' + in_specs[operand_index])
    return einsum(grad_subscripts, out_bar, *other_vals)


def _adjoint_unpack_t2_tril_jiba(bar, nocc, nvir):
    (idx, idy) = numpy.tril_indices(nocc)
    packed_bar = bar[idx, idy]
    trans_bar = bar[idy, idx].transpose(0, 2, 1)
    packed_bar = packed_bar + trans_bar
    diag = numpy.asarray(idx == idy)
    if numpy.any(diag):
        packed_bar[diag] = trans_bar[diag]
    return packed_bar


def _dfccsd_ovvv_vjp_native(Lov, Lvv, t1, theta, tau, t1new_bar, fvv_bar, wVOov_bar, tmp_acc_bar, wooVV_flat_bar):
    if Lov.dtype != numpy.double or Lvv.dtype != numpy.double:
        raise NotImplementedError('native DF-CCSD ovvv VJP supports float64 only')
    from pyscfadlib import libcc_vjp as libcc
    Lov_c = numpy.asarray(Lov, dtype=numpy.double, order='C')
    Lvv_c = numpy.asarray(Lvv, dtype=numpy.double, order='C')
    t1_c = numpy.asarray(t1, dtype=numpy.double, order='C')
    theta_c = numpy.asarray(theta, dtype=numpy.double, order='C')
    tau_c = numpy.asarray(tau, dtype=numpy.double, order='C')
    t1new_bar_c = numpy.asarray(t1new_bar, dtype=numpy.double, order='C')
    fvv_bar_c = numpy.asarray(fvv_bar, dtype=numpy.double, order='C')
    wVOov_bar_c = numpy.asarray(wVOov_bar, dtype=numpy.double, order='C')
    tmp_acc_bar_c = numpy.asarray(tmp_acc_bar, dtype=numpy.double, order='C')
    wooVV_flat_bar_c = numpy.asarray(wooVV_flat_bar, dtype=numpy.double, order='C')
    (naux, nocc, nvir) = Lov_c.shape
    Lov_bar = numpy.empty_like(Lov_c)
    Lvv_bar = numpy.empty_like(Lvv_c)
    drv = libcc.dfccsd_ovvv_vjp
    drv(Lov_bar.ctypes.data_as(ctypes.c_void_p), Lvv_bar.ctypes.data_as(ctypes.c_void_p), Lov_c.ctypes.data_as(ctypes.c_void_p), Lvv_c.ctypes.data_as(ctypes.c_void_p), t1_c.ctypes.data_as(ctypes.c_void_p), theta_c.ctypes.data_as(ctypes.c_void_p), tau_c.ctypes.data_as(ctypes.c_void_p), t1new_bar_c.ctypes.data_as(ctypes.c_void_p), fvv_bar_c.ctypes.data_as(ctypes.c_void_p), wVOov_bar_c.ctypes.data_as(ctypes.c_void_p), tmp_acc_bar_c.ctypes.data_as(ctypes.c_void_p), wooVV_flat_bar_c.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(naux), ctypes.c_int(nocc), ctypes.c_int(nvir))
    return (Lov_bar, Lvv_bar)


def _dfccsd_update_amps_eris_cotangent(mycc, eris, t1, t2, t1new_bar, t2new_bar, *, intermediates=None):
    if mycc.cc2:
        raise NotImplementedError
    if mycc.direct:
        raise NotImplementedError('DF-CCSD direct AO update is not supported')
    if mycc.dcsd:
        raise NotImplementedError('Explicit DF-CCSD cotangent does not support DCSD')
    if eris.Lov is None or eris.Lvv is None:
        raise NotImplementedError('lagrangian_grad requires DF Lov/Lvv leaves')
    (nocc, nvir) = t1.shape
    nvir_pair = nvir * (nvir + 1) // 2
    fock = eris.fock
    mo_energy = eris.mo_energy
    fock_bar = np.zeros_like(fock)
    mo_energy_bar = np.zeros_like(mo_energy)
    oooo_bar = np.zeros_like(eris.oooo)
    ovoo_bar = np.zeros_like(eris.ovoo)
    oovv_bar = np.zeros_like(eris.oovv)
    ovvo_bar = np.zeros_like(eris.ovvo)
    Lov_bar = np.zeros_like(eris.Lov)
    Lvv_bar = np.zeros_like(eris.Lvv)
    if intermediates is None:
        (_, intermediates) = update_intermediates(mycc, t1, t2, eris, return_intermediates=True)
    (eia, eijab, t1work, t2sym, fov, foo, fvv, ft_ij, ft_ab, woooo, wVOov, wVooV, theta_ovvv, tau_for_vvvo, tmp_acc, tmp_oovv_voov, theta, tau_full, tau_half, tau_resp, eris_VOov, theta_t1, theta_t2, tau_final) = (intermediates[name] for name in ('eia', 'eijab', 't1work', 't2sym', 'fov', 'foo', 'fvv', 'ft_ij', 'ft_ab', 'woooo', 'wVOov', 'wVooV', 'theta_ovvv', 'tau_for_vvvo', 'tmp_acc', 'tmp_oovv_voov', 'theta', 'tau_full', 'tau_half', 'tau_resp', 'eris_VOov', 'theta_t1', 'theta_t2', 'tau_final'))
    (eris_ovoo, eris_oovv) = (eris.ovoo, eris.oovv)
    eris_voov = eris.ovvo.conj().transpose(1, 0, 3, 2)
    tau_vvvv = tau_full[numpy.tril_indices(nocc)]
    blksize = max(1, min(_CCSD_OVVV_BLKSIZE, nvir))
    eia_bar = -t1new_bar * t1work / (eia * eia)
    t1work_bar = t1new_bar / eia
    eijab_bar = -t2new_bar * t2sym / (eijab * eijab)
    t2sym_bar = t2new_bar / eijab
    t2work_bar = t2sym_bar + t2sym_bar.transpose(1, 0, 3, 2)
    eia_bar = eia_bar + eijab_bar.sum(axis=(1, 3))
    eia_bar = eia_bar + eijab_bar.sum(axis=(0, 2))
    mo_energy_bar[:nocc] += eia_bar.sum(axis=1)
    mo_energy_bar[nocc:] += -eia_bar.sum(axis=0)
    fov_bar = np.zeros_like(fov)
    foo_bar = np.zeros_like(foo)
    fvv_bar = np.zeros_like(fvv)
    woooo_bar = np.zeros_like(woooo)
    wVOov_bar = np.zeros_like(wVOov)
    wVooV_bar = np.zeros_like(wVooV)
    eris_voov_bar = np.zeros_like(eris_voov)
    foo_grad = _einsum_operand_bar('ja,ji->ia', t1work_bar, (t1, foo), 1)
    foo_bar = foo_bar - foo_grad
    fvv_grad = _einsum_operand_bar('ib,ab->ia', t1work_bar, (t1, fvv), 1)
    fvv_bar = fvv_bar + fvv_grad
    ft_ij_grad = _einsum_operand_bar('ki,kjab->ijab', -t2work_bar, (ft_ij, t2), 0)
    foo_bar = foo_bar + ft_ij_grad
    ft_ab_grad = _einsum_operand_bar('ijac,bc->ijab', t2work_bar, (t2, ft_ab), 1)
    fvv_bar = fvv_bar + ft_ab_grad
    fov_grad = _einsum_operand_bar('ia,ib->ab', -ft_ab_grad, (0.5 * t1, fov), 1)
    fov_bar = fov_bar + fov_grad
    fov_grad = _einsum_operand_bar('ja,ia->ij', ft_ij_grad, (0.5 * t1, fov), 1)
    fov_bar = fov_bar + fov_grad
    woooo_grad = _einsum_operand_bar('ijkl,klab->ijab', 0.5 * t2work_bar, (woooo, tau_final), 0)
    woooo_bar = woooo_bar + woooo_grad
    ovoo_grad = _einsum_operand_bar('jbki,kjba->ia', -t1work_bar, (eris_ovoo, theta_t1), 0)
    ovoo_bar = ovoo_bar + ovoo_grad
    fov_grad = _einsum_operand_bar('jb,ijba->ia', t1work_bar, (fov, theta_t1), 0)
    fov_bar = fov_bar + fov_grad
    wVOov_grad = _einsum_operand_bar('kica,ckjb->ijab', t2work_bar, (theta_t2, wVOov), 1)
    wVOov_bar = wVOov_bar + wVOov_grad
    eris_VOov_grad = _einsum_operand_bar('aikc,kcjb->aijb', 0.5 * wVOov_bar, (eris_VOov, tau_resp), 0)
    eris_voov_bar = eris_voov_bar + eris_VOov_grad
    eris_voov_bar = eris_voov_bar - 0.5 * eris_VOov_grad.transpose(0, 2, 1, 3)
    eris_voov_bar = eris_voov_bar + wVOov_bar
    tmp_bar = t2work_bar.transpose(1, 2, 0, 3)
    tmp_bar = tmp_bar + 0.5 * t2work_bar.transpose(0, 2, 1, 3)
    wVooV_grad = _einsum_operand_bar('jkca,ckib->jaib', tmp_bar, (t2, wVooV), 1)
    wVooV_bar = wVooV_bar + wVooV_grad
    eris_voov_grad = _einsum_operand_bar('bkic,jkca->bija', wVooV_bar, (eris_voov, tau_half), 0)
    eris_voov_bar = eris_voov_bar + eris_voov_grad
    eris_voov_grad = _einsum_operand_bar('ijab,aklb->ijkl', woooo_bar, (tau_full, eris_voov), 1)
    eris_voov_bar = eris_voov_bar + eris_voov_grad
    eris_voov_grad = _einsum_operand_bar('aikb,kjab->ij', foo_bar, (eris_voov, theta), 0)
    eris_voov_bar = eris_voov_bar + eris_voov_grad
    eris_voov_grad = _einsum_operand_bar('cjia,cjib->ab', -fvv_bar, (theta.transpose(2, 1, 0, 3), eris_voov), 1)
    eris_voov_bar = eris_voov_bar + eris_voov_grad
    eris_voov_grad = _einsum_operand_bar('kc,akic->ia', -fov_bar, (t1, eris_voov), 1)
    eris_voov_bar = eris_voov_bar + eris_voov_grad
    eris_voov_grad = _einsum_operand_bar('kc,aikc->ia', 2 * fov_bar, (t1, eris_voov), 1)
    eris_voov_bar = eris_voov_bar + eris_voov_grad
    tmp_bar = _einsum_operand_bar('ka,jbki->jiba', -t2work_bar, (t1, tmp_oovv_voov), 1)
    oovv_grad = _einsum_operand_bar('ic,kjbc->ibkj', tmp_bar, (t1, eris_oovv), 1)
    oovv_bar = oovv_bar + oovv_grad
    eris_voov_grad = _einsum_operand_bar('bjkc,ic->jbki', tmp_bar, (eris_voov, t1), 0)
    eris_voov_bar = eris_voov_bar + eris_voov_grad
    eris_voov_grad = _einsum_operand_bar('jb,aijb->ia', 2 * t1work_bar, (t1, eris_voov), 1)
    eris_voov_bar = eris_voov_bar + eris_voov_grad
    ovvo_bar = ovvo_bar + 0.5 * t2work_bar.transpose(0, 2, 3, 1)
    wVooV_bar = wVooV_bar + 0.5 * wVOov_bar
    oovv_bar = oovv_bar - wVooV_bar.transpose(1, 2, 0, 3)
    oovv_grad = _einsum_operand_bar('jb,jiab->ia', -t1work_bar, (t1, eris_oovv), 1)
    oovv_bar = oovv_bar + oovv_grad
    ovvo_bar = ovvo_bar + eris_voov_bar.conj().transpose(1, 0, 3, 2)
    ovoo_grad = _einsum_operand_bar('kbij,ka->bija', wVooV_bar, (eris_ovoo, t1), 0)
    ovoo_bar = ovoo_bar + ovoo_grad
    wVOov_bar = wVOov_bar + t2work_bar.transpose(2, 0, 1, 3)
    ovoo_grad = _einsum_operand_bar('jbik,ka->bjia', -wVOov_bar, (eris_ovoo, t1), 0)
    ovoo_bar = ovoo_bar + ovoo_grad
    tmp_bar = woooo_bar + woooo_bar.transpose(1, 0, 3, 2)
    ovoo_grad = _einsum_operand_bar('la,jaik->lkji', tmp_bar, (t1, eris_ovoo), 1)
    ovoo_bar = ovoo_bar + ovoo_grad
    ovoo_grad = _einsum_operand_bar('kc,icjk->ij', foo_bar, (-t1, eris_ovoo), 1)
    ovoo_bar = ovoo_bar + ovoo_grad
    ovoo_grad = _einsum_operand_bar('kc,kcji->ij', foo_bar, (2 * t1, eris_ovoo), 1)
    ovoo_bar = ovoo_bar + ovoo_grad
    oooo_bar = oooo_bar + woooo_bar.transpose(0, 2, 1, 3)
    wooVV_bar = wVooV_bar.transpose(2, 1, 0, 3).reshape(nocc * nocc, nvir, nvir)
    wooVV_flat_bar = _adjoint_unpack_tril_last2(wooVV_bar, nvir).reshape(nocc, nocc * nvir_pair)
    tmp_acc_bar = _einsum_operand_bar('ka,ijbk->ijab', -t2work_bar, (t1, tmp_acc), 1)
    use_native_ovvv = True
    if use_native_ovvv:
        try:
            (Lov_grad, Lvv_grad) = _dfccsd_ovvv_vjp_native(eris.Lov, eris.Lvv, t1, theta_ovvv, tau_for_vvvo, t1work_bar, fvv_bar, wVOov_bar, tmp_acc_bar, wooVV_flat_bar)
            Lov_bar = Lov_bar + Lov_grad
            Lvv_bar = Lvv_bar + Lvv_grad
        except (AttributeError, NotImplementedError):
            use_native_ovvv = False
    if not use_native_ovvv:
        for p0 in range(0, nvir, blksize):
            p1 = min(p0 + blksize, nvir)
            bw = p1 - p0
            Lov_tile = eris.Lov[:, :, p0:p1]
            if eris.ovvv is not None:
                vovv_packed = eris.ovvv[:, p0:p1].transpose(1, 0, 2)
            else:
                vovv_packed = einsum('xia,xb->aib', Lov_tile, eris.Lvv)
            vovv_tile = lib.unpack_tril(vovv_packed.reshape(bw * nocc, nvir_pair)).reshape(bw, nocc, nvir, nvir)
            vovv_tile_bar = np.zeros_like(vovv_tile)
            vovv_grad = _einsum_operand_bar('icjb,cjba->ia', t1work_bar, (theta_ovvv[:, p0:p1, :, :], vovv_tile), 1)
            vovv_tile_bar = vovv_tile_bar + vovv_grad
            vovv_grad = _einsum_operand_bar('biac,jc->bija', wVOov_bar[p0:p1], (vovv_tile, t1), 0)
            vovv_tile_bar = vovv_tile_bar + vovv_grad
            vvvo_grad = _einsum_operand_bar('ijcd,cdbk->ijbk', tmp_acc_bar, (tau_for_vvvo[:, :, p0:p1, :], vovv_tile.transpose(0, 2, 3, 1)), 1)
            vovv_tile_bar = vovv_tile_bar + vvvo_grad.transpose(0, 3, 1, 2)
            vovv_grad = _einsum_operand_bar('kc,bkca->ab', -fvv_bar[:, p0:p1], (t1, vovv_tile), 1)
            vovv_tile_bar = vovv_tile_bar + vovv_grad
            vovv_grad = _einsum_operand_bar('kc,ckab->ab', 2 * fvv_bar, (t1[:, p0:p1], vovv_tile), 1)
            vovv_tile_bar = vovv_tile_bar + vovv_grad
            vovv_packed_bar = _adjoint_unpack_tril_last2(vovv_tile_bar.reshape(bw * nocc, nvir, nvir), nvir).reshape(bw, nocc, nvir_pair)
            vovv_packed_bar = vovv_packed_bar - np.dot(t1[:, p0:p1].T, wooVV_flat_bar).reshape(bw, nocc, nvir_pair)
            (Lov_grad, Lvv_grad) = _einsum_operand_bars('xia,xb->aib', vovv_packed_bar, Lov_tile, eris.Lvv)
            Lov_bar[:, :, p0:p1] += Lov_grad
            Lvv_bar = Lvv_bar + Lvv_grad
    Ht2tril_bar = 0.5 * _adjoint_unpack_t2_tril_jiba(t2work_bar, nocc, nvir)
    Lvv_full = lib.unpack_tril(eris.Lvv, filltriu=lib.SYMMETRIC)
    Lvv_full_bar = _G_Lbar(tau_vvvv, Lvv_full, Ht2tril_bar.transpose(0, 2, 1))
    Lvv_grad = _adjoint_unpack_tril_last2(Lvv_full_bar, nvir)
    Lvv_bar = Lvv_bar + Lvv_grad
    fov_bar = fov_bar + t1work_bar
    fock_bar[:nocc, nocc:] += fov_bar
    fock_bar[nocc:, nocc:] += fvv_bar
    mo_energy_bar[nocc:] += -np.diag(fvv_bar)
    fock_ov_grad = _einsum_operand_bar('ia,ib->ab', -0.5 * fvv_bar, (t1, fock[:nocc, nocc:]), 1)
    fock_bar[:nocc, nocc:] += fock_ov_grad
    fock_bar[:nocc, :nocc] += foo_bar
    mo_energy_bar[:nocc] += -np.diag(foo_bar)
    fock_ov_grad = _einsum_operand_bar('ia,ja->ij', 0.5 * foo_bar, (fock[:nocc, nocc:], t1), 0)
    fock_bar[:nocc, nocc:] += fock_ov_grad
    return {'fock': fock_bar, 'mo_energy': mo_energy_bar, 'oooo': oooo_bar, 'ovoo': ovoo_bar, 'ovov': None, 'oovv': oovv_bar, 'ovvo': ovvo_bar, 'Lov': Lov_bar, 'Lvv': Lvv_bar}


def _energy_eris_cotangent(eris, t1, t2, bar_e):
    (nocc, nvir) = t1.shape
    nmo = nocc + nvir
    dtype = eris.fock.dtype
    fock_bar = np.zeros((nmo, nmo), dtype=dtype)
    ovov_bar = np.zeros_like(eris.ovov)
    if _is_zero_cotangent(bar_e):
        return (fock_bar, ovov_bar)
    tau = t2 + einsum('ia,jb->ijab', t1, t1)
    fock_bar[:nocc, nocc:] += 2 * bar_e * t1
    ovov_bar = ovov_bar + bar_e * (2 * tau.transpose(0, 2, 1, 3) - tau.transpose(0, 3, 1, 2))
    return (fock_bar, ovov_bar)

def _G_Lbar(tau, L, G_bar):
    o2, v, _ = tau.shape
    q = L.shape[0]
    B = np.concatenate((tau, tau.transpose(0, 2, 1)), axis=0)
    C = np.concatenate((G_bar, G_bar.transpose(0, 2, 1)), axis=0).reshape(2*o2, -1)
    BT = np.ascontiguousarray(B.transpose(1, 2, 0))
    L2 = L.reshape(q, -1)
    result = np.empty_like(L)
    for x in range(v):
        result[:, x] = L2 @ (BT[x] @ C).reshape(v*v, v)
    return result
