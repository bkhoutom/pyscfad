# Copyright 2021-2025 Xing Zhang
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

from functools import reduce
import numpy
import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.interpreters import ad as jax_ad
from pyscf.cc import ccsd as pyscf_ccsd
from pyscf.mp.mp2 import _mo_without_core
from pyscfad import numpy as np
from pyscfad import pytree
from pyscfad import ops
from pyscfad import lib
from pyscfad.lib import logger


# Toggle for the analytical (custom_vjp) init_amps path.  Default True so
# the heavy LNO calculations benefit immediately; tests can flip to False
# to get the plain JAX-AD baseline for cross-checking.
_USE_CUSTOM_VJP_INIT_AMPS = True


# ---------------------------------------------------------------------------
# MP2 initial amplitudes wrapped as jax.custom_vjp.
#
# Forward computes the same expressions as ``CCSD.init_amps`` but the backward
# is hand-coded so the JAX trace doesn't have to retain ``eia``, ``eijab``,
# the t2 = ovov / eijab division, or the two emp2 einsums.  Eliminates a
# small but real contribution to the recorded jaxpr around the MP2-init step.
#
# Math (real-valued path used in practice):
#   eia[i,a]    = mo_e[i] - mo_e[nocc+a]
#   eijab[i,j,a,b] = eia[i,a] + eia[j,b]
#   t1[i,a]     = fock_oa[i,a] / eia[i,a]
#   t2[i,j,a,b] = ovov[i,a,j,b] / eijab[i,j,a,b]
#   emp2 = einsum('ijab,iajb', 2*t2 - t2.swapaxes(0,1), ovov)
# ---------------------------------------------------------------------------
@custom_vjp
def _init_amps_jax(mo_energy, fock_oa, ovov):
    nocc = ovov.shape[0]
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    t1 = fock_oa / eia
    t2 = ovov.transpose(0, 2, 1, 3).conj() / eijab
    emp2 = 2 * np.einsum('ijab,iajb', t2, ovov) - np.einsum('jiab,iajb', t2, ovov)
    return emp2.real, t1, t2


def _init_amps_jax_fwd(mo_energy, fock_oa, ovov):
    nocc = ovov.shape[0]
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    t1 = fock_oa / eia
    t2 = ovov.transpose(0, 2, 1, 3).conj() / eijab
    emp2 = 2 * np.einsum('ijab,iajb', t2, ovov) - np.einsum('jiab,iajb', t2, ovov)
    return (emp2.real, t1, t2), (mo_energy, fock_oa, ovov, eia, eijab, t1, t2)


def _is_zero_cot(x):
    return x is None or isinstance(x, jax_ad.Zero)


def _init_amps_jax_bwd(res, cotangent):
    mo_energy, fock_oa, ovov, eia, eijab, t1, t2 = res
    bar_emp2, bar_t1, bar_t2 = cotangent

    # Promote zero cotangents to real arrays.
    if _is_zero_cot(bar_t1):
        bar_t1 = np.zeros_like(t1)
    if _is_zero_cot(bar_t2):
        bar_t2 = np.zeros_like(t2)
    bar_e = 0.0 if _is_zero_cot(bar_emp2) else bar_emp2.real

    # emp2 = einsum('ijab,iajb', 2*t2 - t2.swapaxes(0,1), ovov)
    # d emp2 / d t2[I,J,A,B] = 2*ovov[I,A,J,B] - ovov[J,A,I,B]
    # d emp2 / d ovov[I,A,J,B] = 2*t2[I,J,A,B] - t2[J,I,A,B]
    ovov_T = ovov.transpose(0, 2, 1, 3)             # ovov_T[i,j,a,b] = ovov[i,a,j,b]
    bar_t2 = bar_t2 + bar_e * (2 * ovov_T - ovov_T.transpose(1, 0, 2, 3))

    t2_T = t2.transpose(0, 2, 1, 3)                 # t2_T[i,a,j,b] = t2[i,j,a,b]
    # bar_ovov from emp2:
    #   bar_ovov[I,A,J,B] += bar_e * (2*t2[I,J,A,B] - t2[J,I,A,B])
    bar_ovov_from_emp2 = bar_e * (2 * t2_T - t2_T.transpose(2, 1, 0, 3))

    # bar_ovov from t2 = ovov / eijab:
    #   bar_ovov[I,A,J,B] += bar_t2[I,J,A,B] / eijab[I,J,A,B]
    bar_ovov_from_t2 = bar_t2.transpose(0, 2, 1, 3) / eijab.transpose(0, 2, 1, 3)
    bar_ovov = bar_ovov_from_emp2 + bar_ovov_from_t2

    # bar_eijab from t2 = ovov / eijab:
    #   bar_eijab[I,J,A,B] = -bar_t2[I,J,A,B] * t2[I,J,A,B] / eijab[I,J,A,B]
    bar_eijab = -bar_t2 * t2 / eijab

    # bar_fock_oa from t1 = fock_oa / eia:
    bar_fock_oa = bar_t1 / eia

    # bar_eia from t1 = fock_oa / eia:
    bar_eia_from_t1 = -bar_t1 * t1 / eia

    # bar_eia from eijab[i,j,a,b] = eia[i,a] + eia[j,b]:
    #   bar_eia[I,A] += sum_{j,b} bar_eijab[I,j,A,b]   (first term)
    #               +  sum_{i,a} bar_eijab[i,I,a,A]   (second term)
    bar_eia_from_eijab = bar_eijab.sum(axis=(1, 3)) + bar_eijab.sum(axis=(0, 2))
    bar_eia = bar_eia_from_t1 + bar_eia_from_eijab

    # bar_eia → bar_mo_energy
    nocc = bar_eia.shape[0]
    nmo = mo_energy.shape[0]
    bar_mo_energy = np.zeros(nmo, dtype=mo_energy.dtype)
    bar_mo_energy = ops.index_update(
        bar_mo_energy, ops.index[:nocc], bar_eia.sum(axis=1)
    )
    bar_mo_energy = ops.index_update(
        bar_mo_energy, ops.index[nocc:], -bar_eia.sum(axis=0)
    )

    return bar_mo_energy, bar_fock_oa, bar_ovov


_init_amps_jax.defvjp(_init_amps_jax_fwd, _init_amps_jax_bwd)
#from pyscfad.ops import jit
from pyscfad import config
from pyscfad.implicit_diff import make_implicit_diff
from pyscfad.tools.linear_solver import gen_gmres

# attributes explicitly appearing in :fun:`update_amps` are dynamic
ERI_Tracers = ('fock', 'mo_energy',
               'oooo', 'ovoo', 'ovov', 'oovv', 'ovvo', 'ovvv', 'vvvv')

def _converged_iter(amp, mycc, eris):
    t1, t2 = mycc.vector_to_amplitudes(amp)
    t1, t2 = mycc.update_amps(t1, t2, eris)
    amp = mycc.amplitudes_to_vector(t1, t2)
    del t1, t2
    return amp

def _iter(amp, mycc, eris, *,
          diis=None, max_cycle=50, tol=1e-8,
          tolnormt=1e-6, verbose=None):
    log = logger.new_logger(mycc, verbose)

    t1, t2 = mycc.vector_to_amplitudes(amp)
    eold = 0
    eccsd = mycc.energy(t1, t2, eris)
    log.info('Init E_corr(CCSD) = %.15g', eccsd)
    cput1 = log.timer('initialize CCSD')

    conv = False
    for istep in range(max_cycle):
        t1new, t2new = mycc.update_amps(t1, t2, eris)
        tmpvec = mycc.amplitudes_to_vector(t1new, t2new)
        tmpvec -= mycc.amplitudes_to_vector(t1, t2)
        normt = np.linalg.norm(tmpvec)
        tmpvec = None
        if mycc.iterative_damping < 1.0:
            alpha = mycc.iterative_damping
            t1new = (1-alpha) * t1 + alpha * t1new
            t2new *= alpha
            t2new += (1-alpha) * t2
        t1, t2 = t1new, t2new
        t1new = t2new = None
        t1, t2 = mycc.run_diis(t1, t2, istep, normt, eccsd-eold, diis)
        eold, eccsd = eccsd, mycc.energy(t1, t2, eris)
        log.info('cycle = %d  E_corr(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep+1, eccsd, eccsd - eold, normt)
        cput1 = log.timer('CCSD iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    amp = mycc.amplitudes_to_vector(t1, t2)
    t1 = t2 = None
    del log
    return amp, conv


def kernel(mycc, eris=None, t1=None, t2=None, max_cycle=50, tol=1e-8,
           tolnormt=1e-6, verbose=None):
    log = logger.new_logger(mycc, verbose)
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    if t1 is None and t2 is None:
        t1, t2 = mycc.get_init_guess(eris)
    elif t2 is None:
        t2 = mycc.get_init_guess(eris)[1]

    if isinstance(mycc.diis, lib.diis.DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
        adiis.space = mycc.diis_space
    else:
        adiis = None

    vec = mycc.amplitudes_to_vector(t1, t2)

    vec, conv = make_implicit_diff(_iter, config.ccsd_implicit_diff,
                                   optimality_cond=_converged_iter,
                                   solver=gen_gmres(), has_aux=True)(
                                        vec, mycc, eris,
                                        diis=adiis, max_cycle=max_cycle, tol=tol,
                                        tolnormt=tolnormt, verbose=log)

    t1, t2 = mycc.vector_to_amplitudes(vec)
    eccsd = mycc.energy(t1, t2, eris)
    log.timer('CCSD')
    vec = None
    del adiis, log
    return conv, eccsd, t1, t2


#@jit
def update_amps(mycc, t1, t2, eris):
    if mycc.cc2:
        raise NotImplementedError

    nocc, nvir = t1.shape
    nvir_pair = nvir * (nvir+1) // 2
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift

    t1new = np.zeros_like(t1)
    t2new = mycc._add_vvvv(t1, t2, eris, t2sym='jiba')
    t2new *= .5  # *.5 because t2+t2.transpose(1,0,3,2) in the end

#** make_inter_F
    fov = fock[:nocc,nocc:].copy()
    t1new += fov

    foo = fock[:nocc,:nocc] - np.diag(mo_e_o)
    foo += .5 * np.einsum('ia,ja->ij', fock[:nocc,nocc:], t1)

    fvv = fock[nocc:,nocc:] - np.diag(mo_e_v)
    fvv -= .5 * np.einsum('ia,ib->ab', t1, fock[:nocc,nocc:])

    # begin _add_ovvv_
    use_df_tile = (
        getattr(eris, 'Lov', None) is not None
        and getattr(eris, 'Lvv', None) is not None
    )

    if use_df_tile:
        if mycc.direct:
            raise NotImplementedError(
                'DF tiled ovvv update does not support direct AO mode')

        # DF-direct path: build the ovvv block (nvir, nocc, nvir, nvir) on the
        # fly per tile from Lov, Lvv -- avoids the 1.9 GB persistent
        # `eris.ovvv` and the ~6 GB transient full unpack of `eris_vovv`.
        # ``eris.Lov`` is stored with shape (naux, nocc, nvir);
        # ``eris.Lvv`` is the tril-packed (naux, nvir*(nvir+1)/2).
        Lov_full = eris.Lov
        Lvv_full = eris.Lvv

        wooVV_flat = np.zeros((nocc, nocc * nvir_pair))
        wVOov = np.zeros((nvir, nocc, nocc, nvir))

        # Constants needed inside the tile (sliced per iteration).
        theta = t2.transpose(1, 2, 0, 3) * 2
        theta -= t2.transpose(0, 2, 1, 3)  # (nocc, nvir, nocc, nvir) ~ (i, c, j, b)

        tau_for_vvvo = t2 + np.einsum('ia,jb->ijab', t1, t1)
        tmp_acc = np.zeros((nocc, nocc, nvir, nocc))

        scan_idx = jnp.arange(nvir)
        t1_scan = jnp.asarray(t1)
        theta_scan = jnp.asarray(theta)
        Lov_scan = jnp.asarray(Lov_full)
        Lvv_scan = jnp.asarray(Lvv_full)
        tau_scan = jnp.asarray(tau_for_vvvo)

        def ovvv_body(carry, aidx):
            fvv, wooVV_flat, wVOov, t1new, tmp_acc = carry
            Lov_col = Lov_scan[:, :, aidx]
            vovv_packed = np.einsum('xi,xb->ib', Lov_col, Lvv_scan)
            wooVV_flat = wooVV_flat - (
                t1_scan[:, aidx][:, None]
                * vovv_packed.reshape(1, nocc * nvir_pair)
            )
            vovv = lib.unpack_tril(vovv_packed).reshape(nocc, nvir, nvir)
            fvv = fvv + 2 * np.einsum('k,kab->ab',
                                      t1_scan[:, aidx], vovv)
            fvv = fvv.at[:, aidx].add(
                -np.einsum('kc,kca->a', t1_scan, vovv)
            )
            vvvo = vovv.transpose(1, 2, 0)
            tmp_acc = tmp_acc + np.einsum(
                'ijd,dbk->ijbk',
                tau_scan[:, :, aidx, :],
                vvvo,
            )
            wVOov = wVOov.at[aidx].set(
                np.einsum('iac,jc->ija', vovv, t1_scan)
            )
            t1new = t1new + np.einsum(
                'ijb,jba->ia', theta_scan[:, aidx, :, :], vovv)
            return (fvv, wooVV_flat, wVOov, t1new, tmp_acc), None

        init = (fvv, wooVV_flat, wVOov, t1new, tmp_acc)
        fvv, wooVV_flat, wVOov, t1new, tmp_acc = jax.lax.scan(
            ovvv_body, init, scan_idx)[0]

        t2new = t2new - np.einsum('ka,ijbk->ijab', t1, tmp_acc)
        tmp_acc = None
        tau_for_vvvo = None

        wooVV = lib.unpack_tril(wooVV_flat.reshape(nocc**2, nvir_pair))
        wVooV = wooVV.reshape(nocc, nocc, nvir, nvir).transpose(2, 1, 0, 3)
        wooVV_flat = None
    else:
        # Dense path -- used when eris does not provide L matrices
        # (non-DF CCSD, or DF eris that did not stash `Lov`).
        eris_vovv = eris.ovvv.transpose(1, 0, 2)
        # pylint: disable=invalid-unary-operand-type
        wooVV = -np.dot(t1, eris_vovv.reshape(nvir, -1))

        eris_vovv = lib.unpack_tril(eris_vovv.reshape(nvir*nocc, nvir_pair))
        eris_vovv = eris_vovv.reshape(nvir, nocc, nvir, nvir)

        fvv += 2 * np.einsum('kc,ckab->ab', t1, eris_vovv)
        fvv -= np.einsum('kc,bkca->ab', t1, eris_vovv)

        if not mycc.direct:
            vvvo = eris_vovv.transpose(0, 2, 3, 1)
            tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
            tmp = np.einsum('ijcd,cdbk->ijbk', tau, vvvo)
            t2new -= np.einsum('ka,ijbk->ijab', t1, tmp)

        wVOov = np.einsum('biac,jc->bija', eris_vovv, t1)

        theta = t2.transpose(1, 2, 0, 3) * 2
        theta -= t2.transpose(0, 2, 1, 3)
        t1new += np.einsum('icjb,cjba->ia', theta, eris_vovv)

        wooVV = lib.unpack_tril(wooVV.reshape(nocc**2, nvir_pair))
        wVooV = wooVV.reshape(nocc, nocc, nvir, nvir).transpose(2, 1, 0, 3)
    # end _add_ovvv_

    woooo = np.asarray(eris.oooo).transpose(0,2,1,3).copy()

    eris_ovoo = eris.ovoo
    eris_oovv = eris.oovv
    foo += np.einsum('kc,kcji->ij', 2*t1, eris_ovoo)
    foo += np.einsum('kc,icjk->ij',  -t1, eris_ovoo)
    tmp = np.einsum('la,jaik->lkji', t1, eris_ovoo)
    woooo += tmp + tmp.transpose(1,0,3,2)

    wVOov -= np.einsum('jbik,ka->bjia', eris_ovoo, t1)
    t2new += wVOov.transpose(1,2,0,3)

    wVooV += np.einsum('kbij,ka->bija', eris_ovoo, t1)

    eris_ovvo = eris.ovvo
    t1new -= np.einsum('jb,jiab->ia', t1, eris_oovv)
    wVooV -= eris_oovv.transpose(2,0,1,3)
    wVOov += wVooV*.5  #: bjia + bija*.5

    t2new += (eris_ovvo*0.5).transpose(0,3,1,2)
    eris_voov = eris_ovvo.conj().transpose(1,0,3,2)
    t1new += 2*np.einsum('jb,aijb->ia', t1, eris_voov)

    tmp  = np.einsum('ic,kjbc->ibkj', t1, eris_oovv)
    tmp += np.einsum('bjkc,ic->jbki', eris_voov, t1)
    t2new -= np.einsum('ka,jbki->jiba', t1, tmp)

    fov += np.einsum('kc,aikc->ia', t1, eris_voov) * 2
    fov -= np.einsum('kc,akic->ia', t1, eris_voov)

    tau = np.einsum('ia,jb->ijab', t1*.5, t1)
    if mycc.dcsd:
        tau += t2 * .5
        theta = t2.transpose(1,0,2,3) - t2 * .5
        fvv_t2 = -np.einsum('cjia,cjib->ab', theta.transpose(2,1,0,3), eris_voov)
        foo_t2 =  np.einsum('aikb,kjab->ij', eris_voov, theta)
    else:
        tau += t2
    theta  = tau.transpose(1,0,2,3) * 2
    theta -= tau
    fvv -= np.einsum('cjia,cjib->ab', theta.transpose(2,1,0,3), eris_voov)
    foo += np.einsum('aikb,kjab->ij', eris_voov, theta)

    tau = np.einsum('ia,jb->ijab', t1, t1)
    if mycc.dcsd:
        woooo_t2 = np.einsum('ijab,aklb->ijkl', t2, eris_voov)
    else:
        tau += t2
    woooo += np.einsum('ijab,aklb->ijkl', tau, eris_voov)

    tau = np.einsum('ia,jb->ijab', t1, t1)
    if not mycc.dcsd:
        tau += t2 * .5
    wVooV += np.einsum('bkic,jkca->bija', eris_voov, tau)

    tmp = np.einsum('jkca,ckib->jaib', t2, wVooV)
    t2new += tmp.transpose(2,0,1,3)
    tmp *= .5
    t2new += tmp.transpose(0,2,1,3)

    wVOov += eris_voov
    eris_VOov = -.5 * eris_voov.transpose(0,2,1,3)
    tau  =  t2.transpose(1,3,0,2) * 2
    tau -=  t2.transpose(0,3,1,2)
    tau1 = -np.einsum('ia,jb->ibja', t1*2, t1)
    tau +=  tau1
    if mycc.dcsd:
        wVOov += .5 * np.einsum('aikc,kcjb->aijb', eris_voov, tau)
        wVOov += .5 * np.einsum('aikc,kcjb->aijb', eris_VOov, tau1)
    else:
        eris_VOov += eris_voov
        wVOov += .5 * np.einsum('aikc,kcjb->aijb', eris_VOov, tau)

    theta  = t2 * 2
    theta -= t2.transpose(1,0,2,3)
    t2new += np.einsum('kica,ckjb->ijab', theta, wVOov)

    theta = t2.transpose(1,0,2,3) * 2 - t2
    t1new += np.einsum('jb,ijba->ia', fov, theta)
    t1new -= np.einsum('jbki,kjba->ia', eris.ovoo, theta)

    tau = np.einsum('ia,jb->ijab', t1, t1)
    if mycc.dcsd:
        t2new += .5 * np.einsum('ijkl,klab->ijab', woooo_t2, tau)
    tau += t2
    t2new += .5 * np.einsum('ijkl,klab->ijab', woooo, tau)

    ft_ij = foo + np.einsum('ja,ia->ij', .5*t1, fov)
    ft_ab = fvv - np.einsum('ia,ib->ab', .5*t1, fov)
    t2new += np.einsum('ijac,bc->ijab', t2, ft_ab)
    t2new -= np.einsum('ki,kjab->ijab', ft_ij, t2)

    if mycc.dcsd:
        fvv += fvv_t2
        foo += foo_t2
    t1new += np.einsum('ib,ab->ia', t1, fvv)
    t1new -= np.einsum('ja,ji->ia', t1, foo)
    t2new += t2new.transpose(1,0,3,2)

    eia = mo_e_o[:,None] - mo_e_v
    eijab = eia[:,None,:,None] + eia[None,:,None,:]
    t1new /= eia
    t2new /= eijab
    eia = eijab = None
    return t1new, t2new


def _add_vvvv(mycc, t1, t2, eris, out=None, with_ovvv=None, t2sym=None):
    '''t2sym: whether t2 has the symmetry t2[ijab]==t2[jiba] or
    t2[ijab]==-t2[jiab] or t2[ijab]==-t2[jiba]
    '''
    if t2sym in ('jiba', '-jiba', '-jiab'):
        Ht2tril = _add_vvvv_tril(mycc, t1, t2, eris, with_ovvv=with_ovvv)
        nocc, nvir = t2.shape[1:3]
        Ht2 = _unpack_t2_tril(Ht2tril, nocc, nvir, out, t2sym)
    else:
        raise NotImplementedError
    return Ht2

def _add_vvvv_tril(mycc, t1, t2, eris, out=None, with_ovvv=None):
    '''Ht2 = numpy.einsum('ijcd,acdb->ijab', t2, vvvv)
    Using symmetry t2[ijab] = t2[jiba] and Ht2[ijab] = Ht2[jiba], compute the
    lower triangular part of  Ht2
    '''
    if with_ovvv is None:
        with_ovvv = mycc.direct
    nocc, nvir = t2.shape[1:3]

    idx = numpy.tril_indices(nocc)
    tau = t2[idx]
    if t1 is not None:
        tmp = np.einsum('ia,jb->ijab', t1, t1)
        tau += tmp[idx]

    if mycc.direct:   # AO-direct CCSD
        raise NotImplementedError
    else:
        assert not with_ovvv
        Ht2tril = eris._contract_vvvv_t2(mycc, tau, mycc.direct, out)
    del idx
    return Ht2tril

def _unpack_t2_tril(t2tril, nocc, nvir, out=None, t2sym='jiba'):
    t2 = np.empty((nocc,nocc,nvir,nvir), dtype=t2tril.dtype)
    idx, idy = numpy.tril_indices(nocc)
    t2 = ops.index_update(t2, ops.index[idx,idy], t2tril)
    if t2sym == 'jiba':
        t2 = ops.index_update(t2, ops.index[idy,idx], t2tril.transpose(0,2,1))
    elif t2sym == '-jiba':
        t2 = ops.index_update(t2, ops.index[idy,idx], -t2tril.transpose(0,2,1))
    elif t2sym == '-jiab':
        t2 = ops.index_update(t2, ops.index[idy,idx], -t2tril)
        t2 = ops.index_update(t2, ops.index[numpy.diag_indices(nocc)], 0)
    del idx, idy
    return t2

#@jit
def amplitudes_to_vector(t1, t2, out=None):
    nocc, nvir = t1.shape
    nov = nocc * nvir
    vector_t1 = t1.ravel()
    vector_t2 = t2.transpose(0,2,1,3).reshape(nov,nov)[numpy.tril_indices(nov)]
    vector = np.concatenate((vector_t1, vector_t2), axis=None)
    return vector

def vector_to_amplitudes(vector, nmo, nocc):
    nvir = nmo - nocc
    nov = nocc * nvir
    t1 = vector[:nov].reshape((nocc,nvir))
    # filltriu=lib.SYMMETRIC because t2[iajb] == t2[jbia]
    t2 = lib.unpack_tril(vector[nov:], filltriu=lib.SYMMETRIC)
    t2 = t2.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3)
    return t1, t2

#@jit
def energy(cc, t1=None, t2=None, eris=None):
    if t1 is None:
        t1 = cc.t1
    if t2 is None:
        t2 = cc.t2
    if eris is None:
        eris = cc.ao2mo()

    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    e = 2*np.einsum('ia,ia', fov, t1)
    tau  = np.einsum('ia,jb->ijab', t1, t1)
    tau += t2
    eris_ovov = np.asarray(eris.ovov)
    e += 2*np.einsum('ijab,iajb', tau, eris_ovov)
    e +=  -np.einsum('ijab,ibja', tau, eris_ovov)
    #if abs(e.imag) > 1e-4:
    #    logger.warn(cc, 'Non-zero imaginary part found in RCCSD energy %s', e)
    return e.real

class CCSD(pytree.PytreeNode, pyscf_ccsd.CCSD):
    _dynamic_attr = {'_scf'}

    def init_amps(self, eris=None):
        log = logger.new_logger(self)
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        e_hf = self.e_hf
        if e_hf is None:
            e_hf = self.get_e_hf(mo_coeff=self.mo_coeff)
        mo_e = eris.mo_energy
        nocc = self.nocc

        if _USE_CUSTOM_VJP_INIT_AMPS:
            # Hand-coded backward; avoids retaining eia/eijab and the two
            # emp2 einsums in the JAX trace.
            fock_oa = eris.fock[:nocc, nocc:]
            eris_ovov = eris.ovov
            emp2, t1, t2 = _init_amps_jax(mo_e, fock_oa, eris_ovov)
            self.emp2 = emp2
        else:
            # Plain JAX-AD path -- used by tests that need to baseline the
            # custom_vjp against ordinary autodiff.
            eia = mo_e[:nocc, None] - mo_e[None, nocc:]
            t1 = eris.fock[:nocc, nocc:] / eia
            eris_ovov = eris.ovov
            t2 = (eris_ovov.transpose(0, 2, 1, 3).conj()
                  / (eia[:, None, :, None] + eia[None, :, None, :]))
            emp2 = 2 * np.einsum('ijab,iajb', t2, eris_ovov)
            emp2 -= np.einsum('jiab,iajb', t2, eris_ovov)
            self.emp2 = emp2.real

        log.info('Init t2, MP2 energy = %.15g  E_corr(MP2) %.15g',
                 e_hf + self.emp2, self.emp2)
        log.timer('init mp2')
        del log
        return self.emp2, t1, t2

    def amplitudes_to_vector(self, t1, t2, out=None):
        return amplitudes_to_vector(t1, t2, out)

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        return vector_to_amplitudes(vec, nmo, nocc)

    def ccsd(self, t1=None, t2=None, eris=None):
        assert self.mo_coeff is not None
        assert self.mo_occ is not None

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)

        self.e_hf = getattr(eris, 'e_hf', None)
        if self.e_hf is None:
            self.e_hf = self._scf.e_tot

        self.converged, self.e_corr, self.t1, self.t2 = \
                kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                       tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                       verbose=self.verbose)
        self._finalize()
        return self.e_corr, self.t1, self.t2

    def ccsd_t(self, t1=None, t2=None, eris=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        if config.moleintor_opt:
            from pyscfad.cc import ccsd_t
            return ccsd_t.kernel(self, eris, t1, t2, self.verbose)
        else:
            from pyscfad.cc import ccsd_t_slow
            return ccsd_t_slow.kernel(self, eris, t1, t2, self.verbose)

    def ipccsd(self, nroots=1, left=False, koopmans=False, guess=None,
               partition=None, eris=None):
        raise NotImplementedError
        #from pyscfad.cc import eom_rccsd
        #return eom_rccsd.EOMIP(self).kernel(nroots, left, koopmans, guess,
        #                                    partition, eris)

    def eaccsd(self, nroots=1, left=False, koopmans=False, guess=None,
               partition=None, eris=None):
        raise NotImplementedError
        #from pyscfad.cc import eom_rccsd
        #return eom_rccsd.EOMEA(self).kernel(nroots, left, koopmans, guess,
        #                                    partition, eris)

    def eeccsd(self, nroots=1, koopmans=False, guess=None, eris=None):
        raise NotImplementedError
        #from pyscfad.cc import eom_rccsd
        #return eom_rccsd.EOMEE(self).kernel(nroots, koopmans, guess, eris)

    def amplitude_equation(self, t1, t2, eris):
        raise NotImplementedError

    @property
    def dcsd(self):
        return False

    def _finalize(self):
        if self.converged:
            logger.info(self, '%s converged', self.__class__.__name__)
        else:
            logger.note(self, '%s not converged', self.__class__.__name__)
        logger.note(self, 'E(%s) = %.16g  E_corr = %.16g',
                    self.__class__.__name__, self.e_tot, self.e_corr)
        return self

    energy = energy
    update_amps = update_amps
    _add_vvvv = _add_vvvv

class _ChemistsERIs(pytree.PytreeNode, pyscf_ccsd._ChemistsERIs):
    _dynamic_attr = ERI_Tracers

    def _common_init_(self, mycc, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        self.mo_coeff = mo_coeff = _mo_without_core(mycc, mo_coeff)

        dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        vhf = mycc._scf.get_veff(mycc.mol, dm)
        fockao = mycc._scf.get_fock(vhf=vhf, dm=dm)
        self.fock = reduce(np.dot, (mo_coeff.conj().T, fockao, mo_coeff))
        self.e_hf = mycc._scf.energy_tot(dm=dm, vhf=vhf)
        nocc = self.nocc = mycc.nocc
        self.mol = mycc.mol

        mo_e = self.mo_energy = self.fock.diagonal().real
        return self

    def get_ovvv(self, *slices):
        '''To access a subblock of ovvv tensor'''
        if config.moleintor_opt:
            return pyscf_ccsd._ChemistsERIs.get_ovvv(self, *slices)
        else:
            ovw = np.asarray(self.ovvv[slices])
            nocc, nvir, nvir_pair = ovw.shape
            ovvv = lib.unpack_tril(ovw.reshape(nocc*nvir,nvir_pair))
            nvir1 = ovvv.shape[2]
            return ovvv.reshape(nocc,nvir,nvir1,nvir1)
