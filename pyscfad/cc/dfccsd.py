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

import ctypes
import numpy

from pyscf.lib import square_mat_in_trilu_indices
import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.interpreters import ad as jax_ad
from pyscfad import numpy as np
from pyscfad import lib
from pyscfad import config, config_update
from pyscfad.ao2mo import _ao2mo
from pyscfad.cc import ccsd, ccsd_lambda
from pyscfad.lib import logger
from pyscfad.tools import resource_profile

_CCSD_OVVV_BLKSIZE = 32


class RCCSD(ccsd.CCSD):
    _dynamic_attr = _keys = {'with_df'}

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        super().__init__(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)

        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            raise KeyError('The mean-field object has no density fitting.')

    def ao2mo(self, mo_coeff=None):
        return _make_df_eris_incore(self, mo_coeff)

    def ccsd(self, t1=None, t2=None, eris=None):
        if not config.dfccsd_custom_response:
            return super().ccsd(t1=t1, t2=t2, eris=eris)

        assert self.mo_coeff is not None
        assert self.mo_occ is not None

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)

        self.e_hf = getattr(eris, 'e_hf', None)
        if self.e_hf is None:
            self.e_hf = self._scf.e_tot

        if t1 is None and t2 is None:
            t1, t2 = _init_amps_no_side_effect(self, eris)
        elif t2 is None:
            t2 = _init_amps_no_side_effect(self, eris)[1]

        self.converged, self.e_corr, self.t1, self.t2 = \
            _dfccsd_kernel_custom(self, eris, t1, t2)
        self._finalize()
        return self.e_corr, self.t1, self.t2


@custom_vjp
def _contract_vvvv_t2_lowmem(Lvv, t2):
    nvir = t2.shape[-1]
    tril2sq = jnp.asarray(square_mat_in_trilu_indices(nvir))
    x2 = t2.reshape(-1, nvir*nvir)

    def contract_one_a(_, a):
        pair_ac = jax.lax.dynamic_index_in_dim(
            tril2sq, a, axis=0, keepdims=False
        )
        lac = jnp.take(Lvv, pair_ac, axis=1)
        g_cbd_packed = jnp.dot(jnp.transpose(lac), Lvv)
        g_cbd = jnp.take(g_cbd_packed, tril2sq.reshape(-1), axis=1)
        g_cbd = g_cbd.reshape(nvir, nvir, nvir)
        g_cdb = jnp.transpose(g_cbd, (0, 2, 1)).reshape(nvir*nvir, nvir)
        return None, jnp.dot(x2, g_cdb)

    _, H_apb = jax.lax.scan(contract_one_a, None, jnp.arange(nvir))
    return jnp.transpose(H_apb, (1, 0, 2))


def _contract_vvvv_t2_lowmem_fwd(Lvv, t2):
    out = _contract_vvvv_t2_lowmem(Lvv, t2)
    return out, (Lvv, t2)


def _contract_vvvv_t2_lowmem_bwd(res, out_bar):
    Lvv, t2 = res
    nvir = t2.shape[-1]
    naux = Lvv.shape[0]
    tril2sq = jnp.asarray(square_mat_in_trilu_indices(nvir))

    def add_one_a(carry, a):
        Lvv_bar, t2_bar = carry
        pair_ac = jax.lax.dynamic_index_in_dim(
            tril2sq, a, axis=0, keepdims=False
        )
        lac = jnp.take(Lvv, pair_ac, axis=1)
        lbd = jnp.take(Lvv, tril2sq.reshape(-1), axis=1)
        lbd = lbd.reshape(naux, nvir, nvir)
        hbar = jnp.take(out_bar, a, axis=1)
        g_cbd = jnp.einsum('xc,xbd->cbd', lac, lbd)
        t2_bar += jnp.einsum('pb,cbd->pcd', hbar, g_cbd)
        lac_bar = jnp.einsum('pb,pcd,xbd->xc', hbar, t2, lbd)
        lbd_bar = jnp.einsum('pb,pcd,xc->xbd', hbar, t2, lac)
        Lvv_bar = Lvv_bar.at[:, pair_ac].add(lac_bar)
        Lvv_bar = Lvv_bar.at[:, tril2sq.reshape(-1)].add(
            lbd_bar.reshape(naux, nvir*nvir)
        )
        return (Lvv_bar, t2_bar), None

    init = (np.zeros_like(Lvv), np.zeros_like(t2))
    (Lvv_bar, t2_bar), _ = jax.lax.scan(add_one_a, init, jnp.arange(nvir))
    return Lvv_bar, t2_bar


_contract_vvvv_t2_lowmem.defvjp(
    _contract_vvvv_t2_lowmem_fwd,
    _contract_vvvv_t2_lowmem_bwd,
)


def _contract_vvvv_t2(mycc, mol, Lvv, t2, out=None, verbose=None):
    '''Ht2 = numpy.einsum('ijcd,acbd->ijab', t2, vvvv)
    '''
    return _contract_vvvv_t2_lowmem(Lvv, t2)

class _ChemistsERIs(ccsd._ChemistsERIs):
    _dynamic_attr = {'Loo', 'Lvv', 'Lov'}

    def __init__(self, mol=None):
        super().__init__(mol=mol)
        self.naux = None
        self.Loo = None
        self.Lvv = None
        self.Lov = None

    def _contract_vvvv_t2(self, mycc, t2, direct=False, out=None, verbose=None):
        assert not direct
        return _contract_vvvv_t2(mycc, self.mol, self.Lvv, t2)

    def get_ovvv_packed(self):
        """Return the packed ``(ov|vv)`` block of shape (nocc, nvir, nvir_pair).

        Builds lazily from ``Lov`` and ``Lvv`` on first access and caches on
        ``self.ovvv``.  Forward DF-CCSD's ``update_amps`` no longer needs this
        tensor (it builds tiles from ``Lov`` directly), so the packed form is
        only materialized when the lambda or (T) paths actually need it.
        """
        if self.ovvv is not None:
            return self.ovvv
        if self.Lov is None or self.Lvv is None:
            raise ValueError(
                'Cannot lazily build eris.ovvv: Lov/Lvv missing on eris.'
            )
        Lov_flat = self.Lov.reshape(self.Lov.shape[0], -1)
        nocc = self.nocc
        self.ovvv = np.dot(np.transpose(Lov_flat), self.Lvv).reshape(
            nocc, -1, self.Lvv.shape[1]
        )
        return self.ovvv

    def get_ovvv(self, *slices):
        if self.ovvv is None:
            self.get_ovvv_packed()
        return super().get_ovvv(*slices)


def _make_df_eris_incore(cc, mo_coeff=None):
    eris = _ChemistsERIs()
    eris._common_init_(cc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    with_df = cc.with_df
    naux = with_df.get_naoaux()

    mo = np.asarray(eris.mo_coeff)
    ijslice = (0, nmo, 0, nmo)
    has_outcore = (
        hasattr(with_df, '_has_outcore_cderi_placeholder')
        and with_df._has_outcore_cderi_placeholder()
    )
    if has_outcore:
        # Stream cderi from disk and route the AO->MO transform through the
        # outcore custom_vjp so the (naux, nao_pair) tensor never enters the
        # JAX trace.  The custom_vjp's backward analytically propagates
        # mo_coeff_bar via streamed nr_e2 VJP and mol/auxmol_bar via the
        # chunked cholesky_eri VJP.
        from pyscfad.lno.lno_base import _outcore_nr_e2
        cderi_source = with_df._get_cderi_source()
        Lpq = _outcore_nr_e2(
            with_df.mol, with_df.auxmol, mo, cderi_source,
            max(with_df.max_memory, 4096), ijslice, 's2'
        ).reshape(-1, nmo, nmo)
    else:
        eri1 = with_df._cderi
        # pylint: disable=too-many-function-args
        Lpq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1').reshape(-1,nmo,nmo)
    Loo = Lpq[:,:nocc,:nocc].reshape(naux,-1)
    Lov = Lpq[:,:nocc,nocc:].reshape(naux,-1)
    eris.Loo = Loo.reshape(naux, nocc, nocc)
    eris.Lov = Lov.reshape(naux, nocc, nvir)
    eris.Lvv = Lvv = lib.pack_tril(Lpq[:,nocc:,nocc:])

    eris.oooo = np.dot(np.transpose(Loo), Loo).reshape(nocc,nocc,nocc,nocc)
    eris.ovoo = np.dot(np.transpose(Lov), Loo).reshape(nocc,nvir,nocc,nocc)
    ovov = np.dot(np.transpose(Lov), Lov).reshape(nocc,nvir,nocc,nvir)
    eris.ovov = ovov
    eris.ovvo = np.transpose(ovov, (0,1,3,2))

    oovv = np.dot(np.transpose(Loo), Lvv)
    eris.oovv = lib.unpack_tril(oovv).reshape(nocc,nocc,nvir,nvir)
    # eris.ovvv is built lazily via eris.get_ovvv_packed() on first access so
    # the forward DF-CCSD path (which builds tiles from Lov/Lvv directly) does
    # not pay the persistent nocc*nvir*nvir_pair allocation.  The (T) and
    # lambda paths trigger the build the first time they read eris.ovvv.
    return eris


def _is_zero_cotangent(x):
    return x is None or isinstance(x, jax_ad.Zero)


def _init_amps_no_side_effect(mycc, eris):
    mo_e = eris.mo_energy
    nocc = mycc.nocc
    eia = mo_e[:nocc, None] - mo_e[None, nocc:]
    t1 = eris.fock[:nocc, nocc:] / eia
    eris_ovov = eris.ovov
    t2 = (
        np.conj(np.transpose(eris_ovov, (0, 2, 1, 3)))
        / (eia[:, None, :, None] + eia[None, :, None, :])
    )
    return t1, t2


def _dfccsd_kernel_plain(mycc, eris, t1=None, t2=None):
    with config_update('pyscfad_ccsd_implicit_diff', False):
        return ccsd.kernel(
            mycc,
            eris=eris,
            t1=t1,
            t2=t2,
            max_cycle=mycc.max_cycle,
            tol=mycc.conv_tol,
            tolnormt=mycc.conv_tol_normt,
            verbose=mycc.verbose,
        )


@custom_vjp
def _dfccsd_kernel_custom(mycc, eris, t1=None, t2=None):
    return _dfccsd_kernel_plain(mycc, eris, t1, t2)


def _dfccsd_kernel_custom_fwd(mycc, eris, t1=None, t2=None):
    out = _dfccsd_kernel_plain(mycc, eris, t1, t2)
    _, _, t1_out, t2_out = out
    return out, (mycc, eris, t1_out, t2_out)


def _coerce_eris_tensors_for_bwd(eris):
    """Re-wrap host-side TypedNdArray literals on eris as proper jnp arrays.

    JAX saves non-dynamic eris attributes as static aux data and re-emits them
    as TypedNdArray host literals in the bwd; those lack .transpose() and other
    array methods needed by update_amps / lagrangian_grad.
    """
    import copy
    out = copy.copy(eris)
    for attr in ('fock', 'mo_energy', 'oooo', 'ovoo', 'ovov', 'oovv',
                 'ovvo', 'ovvv', 'Lvv', 'Lov', 'mo_coeff'):
        val = getattr(out, attr, None)
        if val is None or hasattr(val, 'transpose'):
            continue
        setattr(out, attr, jnp.asarray(val))
    return out


def _dfccsd_kernel_custom_bwd(res, cotangent):
    """Backward of the DF-CCSD custom-VJP via implicit-diff-form lambda equations.

    Given upstream cotangents (bar_e on the CCSD energy, bar_t1/bar_t2 on the
    converged amplitudes), solve a single response lambda equation that
    incorporates both, then evaluate the ERI cotangent as the gradient of the
    Lagrangian ``bar_e * E_cc(t, eris) + lambda . Omega(t, eris)`` with
    ``Omega = update_amps(t, eris) - t``.
    """
    profile_total = resource_profile.start()
    profile_prepare = resource_profile.start()
    mycc, eris, t1, t2 = res
    _, bar_e, bar_t1, bar_t2 = cotangent
    eris = _coerce_eris_tensors_for_bwd(eris)
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)
    nocc, nvir = t1.shape
    amplitudes_mib = resource_profile.estimated_array_mib(t1, t2)
    eris_mib = resource_profile.estimated_array_mib(
        *(getattr(eris, attr, None) for attr in (
            'fock', 'mo_energy', 'oooo', 'ovoo', 'ovov', 'oovv',
            'ovvo', 'ovvv', 'Lvv', 'Lov', 'mo_coeff',
        ))
    )
    resource_profile.finish(
        'ccsd_bwd.prepare_saved_tensors',
        profile_prepare,
        nocc=nocc,
        nvir=nvir,
        t1_shape=tuple(t1.shape),
        t2_shape=tuple(t2.shape),
        amplitudes_mib=amplitudes_mib,
        saved_eris_mib=eris_mib,
    )

    if (
        _is_zero_cotangent(bar_e)
        and _is_zero_cotangent(bar_t1)
        and _is_zero_cotangent(bar_t2)
    ):
        resource_profile.finish(
            'ccsd_bwd.total',
            profile_total,
            nocc=nocc,
            nvir=nvir,
            zero_cotangent=True,
        )
        return None, None, None, None

    bar_e_val = 0.0 if _is_zero_cotangent(bar_e) else np.asarray(bar_e)
    bar_t1_val = (
        None if _is_zero_cotangent(bar_t1) else np.asarray(bar_t1)
    )
    bar_t2_val = (
        None if _is_zero_cotangent(bar_t2) else np.asarray(bar_t2)
    )

    profile_lambda = resource_profile.start()
    lambda_converged, lambda_vec = ccsd_lambda.solve_response_lambda(
        mycc, eris, t1, t2,
        bar_e_val, bar_t1_val, bar_t2_val,
        max_cycle=mycc.max_cycle,
        tol=mycc.conv_tol_normt,
        verbose=mycc.verbose,
    )
    resource_profile.finish(
        'ccsd_bwd.response_lambda',
        profile_lambda,
        nocc=nocc,
        nvir=nvir,
        response_vector_shape=tuple(lambda_vec.shape),
        response_vector_mib=resource_profile.estimated_array_mib(lambda_vec),
        converged=lambda_converged,
    )
    if not lambda_converged:
        raise RuntimeError('CCSD response lambda equation did not converge')
    profile_lagrangian = resource_profile.start()
    eris_bar = lagrangian_grad(mycc, eris, t1, t2, bar_e_val, lambda_vec)
    resource_profile.finish(
        'ccsd_bwd.lagrangian_gradient',
        profile_lagrangian,
        nocc=nocc,
        nvir=nvir,
    )
    resource_profile.finish(
        'ccsd_bwd.total',
        profile_total,
        nocc=nocc,
        nvir=nvir,
        amplitudes_mib=amplitudes_mib,
        saved_eris_mib=eris_mib,
    )
    return None, eris_bar, None, None


_dfccsd_kernel_custom.defvjp(
    _dfccsd_kernel_custom_fwd,
    _dfccsd_kernel_custom_bwd,
)


def _adjoint_unpack_tril_last2(bar_dense, n, filltriu=lib.HERMITIAN):
    rows, cols = numpy.tril_indices(n)
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
    mat = mat.at[idx].set(packed)
    l2 = mat.reshape(nocc, nvir, nocc, nvir).transpose(0, 2, 1, 3)
    return l1, l2


def _einsum_operand_bars(subscripts, out_bar, *operands):
    lhs, out_spec = subscripts.replace(' ', '').split('->')
    in_specs = lhs.split(',')
    bars = []
    for i, spec in enumerate(in_specs):
        other_specs = [s for j, s in enumerate(in_specs) if j != i]
        other_vals = [v for j, v in enumerate(operands) if j != i]
        grad_subscripts = ','.join([out_spec] + other_specs)
        grad_subscripts += '->' + spec
        bars.append(np.einsum(grad_subscripts, out_bar, *other_vals))
    return bars


def _einsum_operand_bar(subscripts, out_bar, operands, operand_index):
    lhs, out_spec = subscripts.replace(' ', '').split('->')
    in_specs = lhs.split(',')
    other_specs = [s for j, s in enumerate(in_specs) if j != operand_index]
    other_vals = [v for j, v in enumerate(operands) if j != operand_index]
    grad_subscripts = ','.join([out_spec] + other_specs)
    grad_subscripts += '->' + in_specs[operand_index]
    return np.einsum(grad_subscripts, out_bar, *other_vals)


def _adjoint_unpack_t2_tril_jiba(bar, nocc, nvir):
    idx, idy = numpy.tril_indices(nocc)
    packed_bar = bar[idx, idy]
    trans_bar = bar[idy, idx].transpose(0, 2, 1)
    packed_bar = packed_bar + trans_bar
    diag = numpy.asarray(idx == idy)
    if numpy.any(diag):
        packed_bar = packed_bar.at[diag].set(trans_bar[diag])
    return packed_bar


def _dfccsd_ovvv_vjp_native(Lov, Lvv, t1, theta, tau, t1new_bar,
                            fvv_bar, wVOov_bar, tmp_acc_bar,
                            wooVV_flat_bar):
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
    wooVV_flat_bar_c = numpy.asarray(
        wooVV_flat_bar, dtype=numpy.double, order='C')

    naux, nocc, nvir = Lov_c.shape
    Lov_bar = numpy.empty_like(Lov_c)
    Lvv_bar = numpy.empty_like(Lvv_c)

    drv = libcc.dfccsd_ovvv_vjp
    drv(Lov_bar.ctypes.data_as(ctypes.c_void_p),
        Lvv_bar.ctypes.data_as(ctypes.c_void_p),
        Lov_c.ctypes.data_as(ctypes.c_void_p),
        Lvv_c.ctypes.data_as(ctypes.c_void_p),
        t1_c.ctypes.data_as(ctypes.c_void_p),
        theta_c.ctypes.data_as(ctypes.c_void_p),
        tau_c.ctypes.data_as(ctypes.c_void_p),
        t1new_bar_c.ctypes.data_as(ctypes.c_void_p),
        fvv_bar_c.ctypes.data_as(ctypes.c_void_p),
        wVOov_bar_c.ctypes.data_as(ctypes.c_void_p),
        tmp_acc_bar_c.ctypes.data_as(ctypes.c_void_p),
        wooVV_flat_bar_c.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(naux), ctypes.c_int(nocc), ctypes.c_int(nvir))
    return Lov_bar, Lvv_bar


def _dfccsd_update_amps_eris_cotangent(mycc, eris, t1, t2,
                                       t1new_bar, t2new_bar):
    if mycc.cc2:
        raise NotImplementedError
    if mycc.direct:
        raise NotImplementedError('DF-CCSD direct AO update is not supported')
    if mycc.dcsd:
        raise NotImplementedError('Explicit DF-CCSD cotangent does not support DCSD')
    if eris.Lov is None or eris.Lvv is None:
        raise NotImplementedError('lagrangian_grad requires DF Lov/Lvv leaves')

    nocc, nvir = t1.shape
    nmo = nocc + nvir
    nvir_pair = nvir * (nvir + 1) // 2
    dtype = t1.dtype

    fock = eris.fock
    mo_energy = eris.mo_energy
    mo_e_o = mo_energy[:nocc]
    mo_e_v = mo_energy[nocc:] + mycc.level_shift

    fock_bar = np.zeros_like(fock)
    mo_energy_bar = np.zeros_like(mo_energy)
    oooo_bar = np.zeros_like(eris.oooo)
    ovoo_bar = np.zeros_like(eris.ovoo)
    oovv_bar = np.zeros_like(eris.oovv)
    ovvo_bar = np.zeros_like(eris.ovvo)
    Lov_bar = np.zeros_like(eris.Lov)
    Lvv_bar = np.zeros_like(eris.Lvv)

    # Forward replay of update_amps, keeping only the intermediates needed for
    # the explicit adjoint equations below.
    idx_occ = numpy.tril_indices(nocc)
    tau_vvvv = t2[idx_occ]
    tau_vvvv = tau_vvvv + np.einsum('ia,jb->ijab', t1, t1)[idx_occ]
    Ht2tril = _contract_vvvv_t2_lowmem(eris.Lvv, tau_vvvv)
    t2work = ccsd._unpack_t2_tril(Ht2tril, nocc, nvir, t2sym='jiba') * .5

    t1work = np.zeros_like(t1)
    fov = fock[:nocc, nocc:].copy()
    t1work = t1work + fov
    foo = fock[:nocc, :nocc] - np.diag(mo_e_o)
    foo = foo + .5 * np.einsum('ia,ja->ij', fock[:nocc, nocc:], t1)
    fvv = fock[nocc:, nocc:] - np.diag(mo_e_v)
    fvv = fvv - .5 * np.einsum('ia,ib->ab', t1, fock[:nocc, nocc:])

    blksize = max(1, min(_CCSD_OVVV_BLKSIZE, nvir))
    wooVV_flat = np.zeros((nocc, nocc * nvir_pair), dtype=dtype)
    wVOov = np.zeros((nvir, nocc, nocc, nvir), dtype=dtype)
    theta_ovvv = t2.transpose(1, 2, 0, 3) * 2
    theta_ovvv = theta_ovvv - t2.transpose(0, 2, 1, 3)
    tau_for_vvvo = t2 + np.einsum('ia,jb->ijab', t1, t1)
    tmp_acc = np.zeros((nocc, nocc, nvir, nocc), dtype=dtype)

    for p0 in range(0, nvir, blksize):
        p1 = min(p0 + blksize, nvir)
        bw = p1 - p0
        Lov_tile = eris.Lov[:, :, p0:p1]
        vovv_packed = np.einsum('xia,xb->aib', Lov_tile, eris.Lvv)
        wooVV_flat = wooVV_flat - np.dot(
            t1[:, p0:p1],
            vovv_packed.reshape(bw, nocc * nvir_pair),
        )
        vovv_tile = lib.unpack_tril(
            vovv_packed.reshape(bw * nocc, nvir_pair)
        )
        vovv_tile = vovv_tile.reshape(bw, nocc, nvir, nvir)
        fvv = fvv + 2 * np.einsum('kc,ckab->ab',
                                   t1[:, p0:p1], vovv_tile)
        fvv = fvv.at[:, p0:p1].add(
            -np.einsum('kc,bkca->ab', t1, vovv_tile)
        )
        vvvo_tile = vovv_tile.transpose(0, 2, 3, 1)
        tmp_acc = tmp_acc + np.einsum(
            'ijcd,cdbk->ijbk',
            tau_for_vvvo[:, :, p0:p1, :],
            vvvo_tile,
        )
        wVOov = wVOov.at[p0:p1].set(
            np.einsum('biac,jc->bija', vovv_tile, t1)
        )
        t1work = t1work + np.einsum(
            'icjb,cjba->ia',
            theta_ovvv[:, p0:p1, :, :],
            vovv_tile,
        )

    t2work = t2work - np.einsum('ka,ijbk->ijab', t1, tmp_acc)
    wooVV = lib.unpack_tril(wooVV_flat.reshape(nocc**2, nvir_pair))
    wVooV = wooVV.reshape(nocc, nocc, nvir, nvir).transpose(2, 1, 0, 3)

    woooo = np.asarray(eris.oooo).transpose(0, 2, 1, 3).copy()
    eris_ovoo = eris.ovoo
    eris_oovv = eris.oovv
    foo = foo + np.einsum('kc,kcji->ij', 2 * t1, eris_ovoo)
    foo = foo + np.einsum('kc,icjk->ij', -t1, eris_ovoo)
    tmp = np.einsum('la,jaik->lkji', t1, eris_ovoo)
    woooo = woooo + tmp + tmp.transpose(1, 0, 3, 2)
    wVOov = wVOov - np.einsum('jbik,ka->bjia', eris_ovoo, t1)
    t2work = t2work + wVOov.transpose(1, 2, 0, 3)
    wVooV = wVooV + np.einsum('kbij,ka->bija', eris_ovoo, t1)

    eris_ovvo = eris.ovvo
    t1work = t1work - np.einsum('jb,jiab->ia', t1, eris_oovv)
    wVooV = wVooV - eris_oovv.transpose(2, 0, 1, 3)
    wVOov = wVOov + wVooV * .5
    t2work = t2work + (eris_ovvo * .5).transpose(0, 3, 1, 2)
    eris_voov = eris_ovvo.conj().transpose(1, 0, 3, 2)
    t1work = t1work + 2 * np.einsum('jb,aijb->ia', t1, eris_voov)

    tmp = np.einsum('ic,kjbc->ibkj', t1, eris_oovv)
    tmp = tmp + np.einsum('bjkc,ic->jbki', eris_voov, t1)
    tmp_oovv_voov = tmp
    t2work = t2work - np.einsum('ka,jbki->jiba', t1, tmp)
    fov = fov + np.einsum('kc,aikc->ia', t1, eris_voov) * 2
    fov = fov - np.einsum('kc,akic->ia', t1, eris_voov)

    tau_theta = np.einsum('ia,jb->ijab', t1 * .5, t1) + t2
    theta = tau_theta.transpose(1, 0, 2, 3) * 2 - tau_theta
    fvv = fvv - np.einsum(
        'cjia,cjib->ab', theta.transpose(2, 1, 0, 3), eris_voov)
    foo = foo + np.einsum('aikb,kjab->ij', eris_voov, theta)

    tau_full = np.einsum('ia,jb->ijab', t1, t1) + t2
    woooo = woooo + np.einsum('ijab,aklb->ijkl', tau_full, eris_voov)
    tau_half = np.einsum('ia,jb->ijab', t1, t1) + t2 * .5
    wVooV = wVooV + np.einsum('bkic,jkca->bija', eris_voov, tau_half)

    tmp = np.einsum('jkca,ckib->jaib', t2, wVooV)
    t2work = t2work + tmp.transpose(2, 0, 1, 3)
    t2work = t2work + (tmp * .5).transpose(0, 2, 1, 3)

    wVOov = wVOov + eris_voov
    eris_VOov = -.5 * eris_voov.transpose(0, 2, 1, 3)
    tau_resp = t2.transpose(1, 3, 0, 2) * 2
    tau_resp = tau_resp - t2.transpose(0, 3, 1, 2)
    tau1_resp = -np.einsum('ia,jb->ibja', t1 * 2, t1)
    tau_resp = tau_resp + tau1_resp
    eris_VOov = eris_VOov + eris_voov
    wVOov = wVOov + .5 * np.einsum('aikc,kcjb->aijb',
                                    eris_VOov, tau_resp)

    theta_t2 = t2 * 2 - t2.transpose(1, 0, 2, 3)
    t2work = t2work + np.einsum('kica,ckjb->ijab', theta_t2, wVOov)
    theta_t1 = t2.transpose(1, 0, 2, 3) * 2 - t2
    t1work = t1work + np.einsum('jb,ijba->ia', fov, theta_t1)
    t1work = t1work - np.einsum('jbki,kjba->ia', eris_ovoo, theta_t1)

    tau_final = np.einsum('ia,jb->ijab', t1, t1) + t2
    t2work = t2work + .5 * np.einsum('ijkl,klab->ijab', woooo, tau_final)
    ft_ij = foo + np.einsum('ja,ia->ij', .5 * t1, fov)
    ft_ab = fvv - np.einsum('ia,ib->ab', .5 * t1, fov)
    t2work = t2work + np.einsum('ijac,bc->ijab', t2, ft_ab)
    t2work = t2work - np.einsum('ki,kjab->ijab', ft_ij, t2)
    t1work = t1work + np.einsum('ib,ab->ia', t1, fvv)
    t1work = t1work - np.einsum('ja,ji->ia', t1, foo)

    eia = mo_e_o[:, None] - mo_e_v
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    t2sym = t2work + t2work.transpose(1, 0, 3, 2)

    # Reverse replay.
    eia_bar = -t1new_bar * t1work / (eia * eia)
    t1work_bar = t1new_bar / eia
    eijab_bar = -t2new_bar * t2sym / (eijab * eijab)
    t2sym_bar = t2new_bar / eijab
    t2work_bar = t2sym_bar + t2sym_bar.transpose(1, 0, 3, 2)
    eia_bar = eia_bar + eijab_bar.sum(axis=(1, 3))
    eia_bar = eia_bar + eijab_bar.sum(axis=(0, 2))
    mo_energy_bar = mo_energy_bar.at[:nocc].add(eia_bar.sum(axis=1))
    mo_energy_bar = mo_energy_bar.at[nocc:].add(-eia_bar.sum(axis=0))

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
    ft_ij_grad = _einsum_operand_bar('ki,kjab->ijab',
                                     -t2work_bar, (ft_ij, t2), 0)
    foo_bar = foo_bar + ft_ij_grad
    ft_ab_grad = _einsum_operand_bar('ijac,bc->ijab',
                                     t2work_bar, (t2, ft_ab), 1)
    fvv_bar = fvv_bar + ft_ab_grad
    fov_grad = _einsum_operand_bar('ia,ib->ab',
                                   -ft_ab_grad, (.5 * t1, fov), 1)
    fov_bar = fov_bar + fov_grad
    fov_grad = _einsum_operand_bar('ja,ia->ij',
                                   ft_ij_grad, (.5 * t1, fov), 1)
    fov_bar = fov_bar + fov_grad
    woooo_grad = _einsum_operand_bar('ijkl,klab->ijab',
                                     .5 * t2work_bar, (woooo, tau_final), 0)
    woooo_bar = woooo_bar + woooo_grad
    ovoo_grad = _einsum_operand_bar('jbki,kjba->ia',
                                    -t1work_bar, (eris_ovoo, theta_t1), 0)
    ovoo_bar = ovoo_bar + ovoo_grad
    fov_grad = _einsum_operand_bar('jb,ijba->ia',
                                   t1work_bar, (fov, theta_t1), 0)
    fov_bar = fov_bar + fov_grad
    wVOov_grad = _einsum_operand_bar('kica,ckjb->ijab',
                                     t2work_bar, (theta_t2, wVOov), 1)
    wVOov_bar = wVOov_bar + wVOov_grad

    eris_VOov_grad = _einsum_operand_bar('aikc,kcjb->aijb',
                                         .5 * wVOov_bar,
                                         (eris_VOov, tau_resp), 0)
    eris_voov_bar = eris_voov_bar + eris_VOov_grad
    eris_voov_bar = eris_voov_bar - .5 * eris_VOov_grad.transpose(0, 2, 1, 3)
    eris_voov_bar = eris_voov_bar + wVOov_bar

    tmp_bar = t2work_bar.transpose(1, 2, 0, 3)
    tmp_bar = tmp_bar + .5 * t2work_bar.transpose(0, 2, 1, 3)
    wVooV_grad = _einsum_operand_bar('jkca,ckib->jaib',
                                     tmp_bar, (t2, wVooV), 1)
    wVooV_bar = wVooV_bar + wVooV_grad
    eris_voov_grad = _einsum_operand_bar('bkic,jkca->bija',
                                         wVooV_bar, (eris_voov, tau_half), 0)
    eris_voov_bar = eris_voov_bar + eris_voov_grad
    eris_voov_grad = _einsum_operand_bar('ijab,aklb->ijkl',
                                         woooo_bar, (tau_full, eris_voov), 1)
    eris_voov_bar = eris_voov_bar + eris_voov_grad
    eris_voov_grad = _einsum_operand_bar('aikb,kjab->ij',
                                         foo_bar, (eris_voov, theta), 0)
    eris_voov_bar = eris_voov_bar + eris_voov_grad
    eris_voov_grad = _einsum_operand_bar(
        'cjia,cjib->ab',
        -fvv_bar, (theta.transpose(2, 1, 0, 3), eris_voov), 1)
    eris_voov_bar = eris_voov_bar + eris_voov_grad
    eris_voov_grad = _einsum_operand_bar('kc,akic->ia',
                                         -fov_bar, (t1, eris_voov), 1)
    eris_voov_bar = eris_voov_bar + eris_voov_grad
    eris_voov_grad = _einsum_operand_bar('kc,aikc->ia',
                                         2 * fov_bar, (t1, eris_voov), 1)
    eris_voov_bar = eris_voov_bar + eris_voov_grad

    tmp_bar = _einsum_operand_bar('ka,jbki->jiba',
                                  -t2work_bar, (t1, tmp_oovv_voov), 1)
    oovv_grad = _einsum_operand_bar('ic,kjbc->ibkj',
                                    tmp_bar, (t1, eris_oovv), 1)
    oovv_bar = oovv_bar + oovv_grad
    eris_voov_grad = _einsum_operand_bar('bjkc,ic->jbki',
                                         tmp_bar, (eris_voov, t1), 0)
    eris_voov_bar = eris_voov_bar + eris_voov_grad
    eris_voov_grad = _einsum_operand_bar('jb,aijb->ia',
                                         2 * t1work_bar, (t1, eris_voov), 1)
    eris_voov_bar = eris_voov_bar + eris_voov_grad
    ovvo_bar = ovvo_bar + .5 * t2work_bar.transpose(0, 2, 3, 1)
    wVooV_bar = wVooV_bar + .5 * wVOov_bar
    oovv_bar = oovv_bar - wVooV_bar.transpose(1, 2, 0, 3)
    oovv_grad = _einsum_operand_bar('jb,jiab->ia',
                                    -t1work_bar, (t1, eris_oovv), 1)
    oovv_bar = oovv_bar + oovv_grad
    ovvo_bar = ovvo_bar + eris_voov_bar.conj().transpose(1, 0, 3, 2)

    ovoo_grad = _einsum_operand_bar('kbij,ka->bija',
                                    wVooV_bar, (eris_ovoo, t1), 0)
    ovoo_bar = ovoo_bar + ovoo_grad
    wVOov_bar = wVOov_bar + t2work_bar.transpose(2, 0, 1, 3)
    ovoo_grad = _einsum_operand_bar('jbik,ka->bjia',
                                    -wVOov_bar, (eris_ovoo, t1), 0)
    ovoo_bar = ovoo_bar + ovoo_grad
    tmp_bar = woooo_bar + woooo_bar.transpose(1, 0, 3, 2)
    ovoo_grad = _einsum_operand_bar('la,jaik->lkji',
                                    tmp_bar, (t1, eris_ovoo), 1)
    ovoo_bar = ovoo_bar + ovoo_grad
    ovoo_grad = _einsum_operand_bar('kc,icjk->ij',
                                    foo_bar, (-t1, eris_ovoo), 1)
    ovoo_bar = ovoo_bar + ovoo_grad
    ovoo_grad = _einsum_operand_bar('kc,kcji->ij',
                                    foo_bar, (2 * t1, eris_ovoo), 1)
    ovoo_bar = ovoo_bar + ovoo_grad
    oooo_bar = oooo_bar + woooo_bar.transpose(0, 2, 1, 3)

    # Reverse the DF-specific tiled ovvv construction.
    wooVV_bar = wVooV_bar.transpose(2, 1, 0, 3).reshape(
        nocc * nocc, nvir, nvir)
    wooVV_flat_bar = _adjoint_unpack_tril_last2(
        wooVV_bar, nvir).reshape(nocc, nocc * nvir_pair)
    tmp_acc_bar = _einsum_operand_bar('ka,ijbk->ijab',
                                      -t2work_bar, (t1, tmp_acc), 1)

    use_native_ovvv = True
    if use_native_ovvv:
        try:
            Lov_grad, Lvv_grad = _dfccsd_ovvv_vjp_native(
                eris.Lov, eris.Lvv, t1, theta_ovvv, tau_for_vvvo,
                t1work_bar, fvv_bar, wVOov_bar, tmp_acc_bar,
                wooVV_flat_bar,
            )
            Lov_bar = Lov_bar + Lov_grad
            Lvv_bar = Lvv_bar + Lvv_grad
        except (AttributeError, NotImplementedError):
            use_native_ovvv = False

    if not use_native_ovvv:
        for p0 in range(0, nvir, blksize):
            p1 = min(p0 + blksize, nvir)
            bw = p1 - p0
            Lov_tile = eris.Lov[:, :, p0:p1]
            vovv_packed = np.einsum('xia,xb->aib', Lov_tile, eris.Lvv)
            vovv_tile = lib.unpack_tril(
                vovv_packed.reshape(bw * nocc, nvir_pair)
            ).reshape(bw, nocc, nvir, nvir)
            vovv_tile_bar = np.zeros_like(vovv_tile)

            vovv_grad = _einsum_operand_bar(
                'icjb,cjba->ia',
                t1work_bar, (theta_ovvv[:, p0:p1, :, :], vovv_tile), 1)
            vovv_tile_bar = vovv_tile_bar + vovv_grad
            vovv_grad = _einsum_operand_bar('biac,jc->bija',
                                            wVOov_bar[p0:p1],
                                            (vovv_tile, t1), 0)
            vovv_tile_bar = vovv_tile_bar + vovv_grad
            vvvo_grad = _einsum_operand_bar(
                'ijcd,cdbk->ijbk',
                tmp_acc_bar, (tau_for_vvvo[:, :, p0:p1, :],
                              vovv_tile.transpose(0, 2, 3, 1)), 1)
            vovv_tile_bar = vovv_tile_bar + vvvo_grad.transpose(0, 3, 1, 2)
            vovv_grad = _einsum_operand_bar(
                'kc,bkca->ab',
                -fvv_bar[:, p0:p1], (t1, vovv_tile), 1)
            vovv_tile_bar = vovv_tile_bar + vovv_grad
            vovv_grad = _einsum_operand_bar(
                'kc,ckab->ab',
                2 * fvv_bar, (t1[:, p0:p1], vovv_tile), 1)
            vovv_tile_bar = vovv_tile_bar + vovv_grad

            vovv_packed_bar = _adjoint_unpack_tril_last2(
                vovv_tile_bar.reshape(bw * nocc, nvir, nvir),
                nvir,
            ).reshape(bw, nocc, nvir_pair)
            vovv_packed_bar = vovv_packed_bar - np.dot(
                t1[:, p0:p1].T, wooVV_flat_bar,
            ).reshape(bw, nocc, nvir_pair)
            Lov_grad, Lvv_grad = _einsum_operand_bars(
                'xia,xb->aib', vovv_packed_bar, Lov_tile, eris.Lvv)
            Lov_bar = Lov_bar.at[:, :, p0:p1].add(Lov_grad)
            Lvv_bar = Lvv_bar + Lvv_grad

    # Reverse the initial vvvv contraction and one-electron intermediates.
    Ht2tril_bar = .5 * _adjoint_unpack_t2_tril_jiba(
        t2work_bar, nocc, nvir)
    Lvv_grad, _ = _contract_vvvv_t2_lowmem_bwd(
        (eris.Lvv, tau_vvvv), Ht2tril_bar)
    Lvv_bar = Lvv_bar + Lvv_grad

    fov_bar = fov_bar + t1work_bar
    fock_bar = fock_bar.at[:nocc, nocc:].add(fov_bar)
    fock_bar = fock_bar.at[nocc:, nocc:].add(fvv_bar)
    mo_energy_bar = mo_energy_bar.at[nocc:].add(-np.diag(fvv_bar))
    fock_ov_grad = _einsum_operand_bar('ia,ib->ab',
                                       -0.5 * fvv_bar,
                                       (t1, fock[:nocc, nocc:]), 1)
    fock_bar = fock_bar.at[:nocc, nocc:].add(fock_ov_grad)
    fock_bar = fock_bar.at[:nocc, :nocc].add(foo_bar)
    mo_energy_bar = mo_energy_bar.at[:nocc].add(-np.diag(foo_bar))
    fock_ov_grad = _einsum_operand_bar('ia,ja->ij',
                                       .5 * foo_bar,
                                       (fock[:nocc, nocc:], t1), 0)
    fock_bar = fock_bar.at[:nocc, nocc:].add(fock_ov_grad)

    return {
        'fock': fock_bar,
        'mo_energy': mo_energy_bar,
        'oooo': oooo_bar,
        'ovoo': ovoo_bar,
        'ovov': None,
        'oovv': oovv_bar,
        'ovvo': ovvo_bar,
        'Lov': Lov_bar,
        'Lvv': Lvv_bar,
    }


def _zero_like_leaf(x):
    return None if x is None else np.zeros_like(x)


def _add_bar(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def _energy_eris_cotangent(eris, t1, t2, bar_e):
    nocc, nvir = t1.shape
    nmo = nocc + nvir
    dtype = eris.fock.dtype
    fock_bar = np.zeros((nmo, nmo), dtype=dtype)
    ovov_bar = np.zeros_like(eris.ovov)
    if _is_zero_cotangent(bar_e):
        return fock_bar, ovov_bar

    tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
    fock_bar = fock_bar.at[:nocc, nocc:].add(2 * bar_e * t1)
    ovov_bar = ovov_bar + bar_e * (
        2 * tau.transpose(0, 2, 1, 3)
        - tau.transpose(0, 3, 1, 2)
    )
    return fock_bar, ovov_bar


def lagrangian_grad(mycc, eris, t1, t2, bar_e_val, lambda_vec):
    """Hand-coded ERI cotangent of the DF-CCSD response Lagrangian."""
    nocc, nvir = t1.shape
    t1new_bar, t2new_bar = _amplitudes_to_vector_cotangent(
        lambda_vec, nocc, nvir
    )
    update_bars = _dfccsd_update_amps_eris_cotangent(
        mycc, eris, t1, t2, t1new_bar, t2new_bar
    )
    e_fock_bar, e_ovov_bar = _energy_eris_cotangent(
        eris, t1, t2, bar_e_val
    )

    eris_bar = jax.tree_util.tree_map(_zero_like_leaf, eris)
    eris_bar.fock = _add_bar(update_bars['fock'], e_fock_bar)
    eris_bar.mo_energy = update_bars['mo_energy']
    eris_bar.oooo = update_bars['oooo']
    eris_bar.ovoo = update_bars['ovoo']
    eris_bar.ovov = _add_bar(update_bars['ovov'], e_ovov_bar)
    eris_bar.oovv = update_bars['oovv']
    eris_bar.ovvo = update_bars['ovvo']
    eris_bar.Lov = update_bars['Lov']
    eris_bar.Lvv = update_bars['Lvv']
    return eris_bar
