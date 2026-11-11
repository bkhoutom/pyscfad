# Copyright 2021-2026 The PySCFAD Authors
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
from functools import partial
import numpy

from pyscf.lib import current_memory
import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.custom_derivatives import SymbolicZero
from jax.interpreters import ad as jax_ad
from pyscfad import numpy as np
from pyscfad import lib, ops
from pyscfad import config, config_update
from pyscfad.ao2mo import _ao2mo
from pyscfad.cc import ccsd, ccsd_lambda
from pyscfad.lib import logger
from pyscfad.tools import resource_profile

_CCSD_OVVV_BLKSIZE = 32


class RCCSD(ccsd.CCSD):
    _dynamic_attr = {'with_df'}
    _keys = {'with_df', 'vvvv_direct', 'native_cpu'}

    vvvv_direct = False
    native_cpu = False

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        super().__init__(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)

        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            raise KeyError('The mean-field object has no density fitting.')

    def ao2mo(self, mo_coeff=None):
        return _maybe_cache_vvvv(self, _make_df_eris_incore(self, mo_coeff))

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


@jax.jit
def _build_eris_vvvv(Lvv):
    #: (ab|cd) + (cb|ad) and (ab|cd) - (cb|ad) for a >= c, d >= b
    naux, nvir, _ = Lvv.shape
    npair = nvir * (nvir + 1) // 2
    # GEMM operands are materialized in the layout the GEMM reads (XLA's CPU
    # dot is ~2x slower on a transposed operand); optimization_barrier keeps
    # XLA from folding the transpose back into the dot
    Lv2T = jax.lax.optimization_barrier(Lvv.reshape(naux, nvir * nvir).T)

    def body(V, a):
        # X[c,d,b] = (ab|cd)
        X = (Lv2T @ Lvv[:, a, :]).reshape(nvir, nvir, nvir)
        Xt = X.transpose(0, 2, 1)
        W = np.stack((lib.pack_tril(X + Xt), lib.pack_tril(X - Xt)))
        # rows (a,c) of c > a are overwritten by the next a
        return jax.lax.dynamic_update_slice(V, W, (0, a * (a + 1) // 2, 0)), None

    V = np.zeros((2, npair, npair), dtype=Lvv.dtype)
    return jax.lax.scan(body, V, np.arange(nvir))[0]

def _G_dense(tau, V):
    # the parts of tau symmetric and antisymmetric in the virtual indices,
    # over a >= c; the diagonal is counted twice in V[0]
    nvir = tau.shape[-1]
    taut = tau.transpose(0, 2, 1)
    tau_s = lib.pack_tril(tau + taut) * (.5 - .25 * lib.pack_tril(np.eye(nvir)))
    tau_a = lib.pack_tril(tau - taut) * .5
    # V[0] and V[1] are symmetric; one batched dot_general reads V as stored
    Gs, Ga = jax.lax.dot_general(np.stack((tau_s, tau_a)), V, (((2,), (2,)), ((0,), (0,))))
    return (lib.unpack_tril(Gs, filltriu=lib.SYMMETRIC)
            + lib.unpack_tril(Ga, filltriu=lib.ANTIHERMI))

def _G_direct(tau, L):
    o2, nvir, _ = tau.shape
    v2 = nvir * nvir
    naux = L.shape[0]
    L2 = L.reshape(naux, v2)
    tau2 = tau.reshape(o2, v2)
    LT = jax.lax.optimization_barrier(L.transpose(2, 1, 0))

    @jax.checkpoint
    def slab(LTx):
        Vx = (LTx @ L2).reshape(v2, nvir)
        return tau2 @ Vx

    out = jax.lax.map(slab, LT)
    return out.transpose(1, 2, 0)

def _G_direct_acc(tau, L):
    o2, nvir, _ = tau.shape
    v2 = nvir * nvir
    naux = L.shape[0]
    L2 = L.reshape(naux, v2)
    LT = jax.lax.optimization_barrier(L.transpose(2, 1, 0))
    tauT = jax.lax.optimization_barrier(tau.transpose(2, 0, 1))

    # as _G_direct, accumulated over the slabs: G[p,(y,z)] = sum_{x,w}
    # tau[p,w,x] (yx|zw). Contracting the short axis of a slab is faster
    # than contracting its long axis as _G_direct does.
    @jax.checkpoint
    def slab(G, xs):
        LTx, tau_x = xs
        Vx = (LTx @ L2).reshape(v2, nvir)
        return G + tau_x @ Vx.T, None

    G = np.zeros((o2, v2), dtype=tau.dtype)
    G = jax.lax.scan(slab, G, (LT, tauT))[0]
    return G.reshape(o2, nvir, nvir)

@partial(jax.jit, static_argnames=('tangent',))
def _G(tau, L, V, tangent=False):
    if V is None:
        # reverse mode transposes the slabs of _G_direct into those of
        # _G_direct_acc, so the tangent of tau is computed by _G_direct
        if tangent:
            return _G_direct(tau, L)
        return _G_direct_acc(tau, L)
    return _G_dense(tau, V)

@jax.jit
@jax.checkpoint
def _G_Ldot(tau, L, L_dot):
    '''Tangent of ``G`` for a tangent ``L_dot`` of the full DF factor,
    ``sum_{z,w} tau[p,z,w] (L_dot[P,z,x] L[P,w,y] + L[P,z,x] L_dot[P,w,y])``.

    Accumulated over slabs ``T_z[w,(y,x)] = sum_P L[P,w,y] L_dot[P,z,x]``;
    the second term is the first one with tau and the result transposed in
    the virtual indices, so ``B`` stacks tau with its transpose. The transpose
    of this loop contracts tau with the cotangent of G one slab at a time.
    '''
    o2, nvir, _ = tau.shape
    naux = L.shape[0]
    L2T = jax.lax.optimization_barrier(L.reshape(naux, nvir * nvir).T) # [(w,y), P]
    B = np.concatenate((tau, tau.transpose(0, 2, 1)), axis=0)
    BT = jax.lax.optimization_barrier(B.transpose(1, 0, 2)) # BT[z] = B[:,z,:]

    def slab(H, xs):
        Bz, L_dot_z = xs
        Tz = (L2T @ L_dot_z).reshape(nvir, nvir * nvir) # [w, (y,x)]
        return H + Bz @ Tz, None

    H0 = np.zeros((2 * o2, nvir * nvir), dtype=tau.dtype)
    H = jax.lax.scan(slab, H0, (BT, np.moveaxis(L_dot, 1, 0)))[0]
    H = H.reshape(2 * o2, nvir, nvir)
    return H[:o2] + H[o2:].transpose(0, 2, 1)

@jax.jit
def _G_Lbar(tau, L, G_bar):
    '''Cotangent of the full DF factor for a cotangent ``G_bar`` of ``G``,
    ``sum_{p,c,d} B[p,x,c] L[P,c,d] C[p,d,y]``, where ``B`` and ``C`` stack tau
    and G_bar with their transposes (L enters ``G`` twice). Unlike the
    transpose of :func:`_G_Ldot`, both GEMMs of a slab read their operands
    as stored.
    '''
    o2, nvir, _ = tau.shape
    naux = L.shape[0]
    B = np.concatenate((tau, tau.transpose(0, 2, 1)), axis=0)
    C = np.concatenate((G_bar, G_bar.transpose(0, 2, 1)), axis=0).reshape(2 * o2, -1)
    BT = jax.lax.optimization_barrier(B.transpose(1, 2, 0)) # BT[x] = B[:,x,:].T
    L2 = L.reshape(naux, -1)

    def slab(Bx):
        Yx = (Bx @ C).reshape(nvir * nvir, nvir) # [(c,d), y]
        return L2 @ Yx

    return jax.lax.map(slab, BT).transpose(1, 0, 2)

@jax.custom_jvp
def ladder(tau, L, V):
    #: G[p,d,b] = sum_{a,c} tau[p,a,c] (ab|cd)``.
    return _G(tau, L, V)

@partial(ladder.defjvp, symbolic_zeros=True)
def _ladder_jvp(primals, tangents):
    tau, L, V = primals
    tau_dot, L_dot, V_dot = tangents
    if not (V_dot is None or isinstance(V_dot, SymbolicZero)):
        raise NotImplementedError('the dense cache vvvv is not differentiated; '
                                  'build it from stop_gradient(Lvv)')
    G_dot = []
    if not isinstance(tau_dot, SymbolicZero):
        G_dot.append(_G(tau_dot, L, V, tangent=True))
    if not isinstance(L_dot, SymbolicZero):
        G_dot.append(_G_Ldot(tau, L, L_dot))
    return _G(tau, L, V), (sum(G_dot[1:], G_dot[0]) if G_dot
                           else np.zeros_like(tau))

@jax.custom_vjp
def ladder_rev(tau, L, V):
    '''Reverse-mode only :func:`ladder`.'''
    return _G(tau, L, V)

def _ladder_rev_fwd(tau, L, V):
    # NOTE with symbolic_zeros, the inputs are CustomVJPPrimal, telling
    # _ladder_rev_bwd which cotangents are needed
    perturbed = (tau.perturbed, L.perturbed)
    tau, L = tau.value, L.value
    if V is not None:
        V = V.value
    return _G(tau, L, V), (tau, L, V, perturbed)

def _ladder_rev_bwd(res, G_bar):
    tau, L, V, (tau_perturbed, L_perturbed) = res
    tau_bar = L_bar = None
    if isinstance(G_bar, SymbolicZero):
        return tau_bar, L_bar, None
    # G is a self-adjoint map of tau: its cotangent is the same contraction
    if tau_perturbed:
        tau_bar = _G(G_bar, L, V)
    if L_perturbed:
        L_bar = _G_Lbar(tau, L, G_bar)
    return tau_bar, L_bar, None

ladder_rev.defvjp(_ladder_rev_fwd, _ladder_rev_bwd, symbolic_zeros=True)

def _contract_vvvv_t2(mycc, mol, Lvv, t2, out=None, verbose=None, vvvv=None):
    '''Ht2 = numpy.einsum('ijcd,acbd->ijab', t2, vvvv)
    '''
    nvir = t2.shape[-1]
    Lvv = lib.unpack_tril(Lvv, filltriu=lib.SYMMETRIC)
    # Runtime data for JIT reuse across ERIs; factor derivatives enter via Lvv.
    if vvvv is not None:
        vvvv = ops.stop_grad(vvvv)
    fn = ladder_rev if config.moleintor_opt else ladder
    Ht2 = fn(t2.reshape(-1, nvir, nvir), Lvv, vvvv).transpose(0, 2, 1)
    return Ht2.reshape(t2.shape)


class _ChemistsERIs(ccsd._ChemistsERIs):
    _dynamic_attr = {'Loo', 'Lvv', 'Lov', '_vvvv_pair'}

    def __init__(self, mol=None):
        super().__init__(mol=mol)
        self.naux = None
        self.Loo = None
        self.Lvv = None
        self.Lov = None
        self._vvvv_pair = None

    def _contract_vvvv_t2(self, mycc, t2, direct=False, out=None, verbose=None):
        assert not direct
        return _contract_vvvv_t2(mycc, self.mol, self.Lvv, t2,
                                 vvvv=self._vvvv_pair)

    def get_ovvv_packed(self):
        """Return the packed ``(ov|vv)`` block of shape (nocc, nvir, nvir_pair).

        DF ERI builders populate ``self.ovvv`` once. CCSD and its response
        read tiles from this cache without repeating the auxiliary contraction
        or unpacking the full tensor. Manually supplied factor-only ERIs can
        still populate the cache on first access.
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


def _maybe_cache_vvvv(mycc, eris):
    """Cache the virtual ladder when it fits in 60% of remaining memory.

    The pair tensor is runtime pytree data with stopped derivatives. Its
    physical factor dependence enters through the ladder's custom derivatives.
    Canonical and LNO ERI builders share this policy and retain the same ERIs.
    """
    eris._vvvv_pair = None
    from . import _dfccsd_native
    if _dfccsd_native.enabled(mycc):
        # The native host solve applies the same memory policy to its own
        # packed vvvv layout. Avoid retaining a second, JAX-specific cache.
        return eris
    if mycc.vvvv_direct:
        return eris
    npair = eris.Lvv.shape[-1]
    # Two float64 pair matrices, as in the upstream memory guard.
    cache_mb = 16 * npair**2 / 1e6
    remaining_mb = mycc.max_memory - current_memory()[0]
    if cache_mb > .6 * remaining_mb:
        logger.warn(mycc, 'Not enough memory to store the virtual ladder cache')
    else:
        Lvv = lib.unpack_tril(ops.stop_grad(eris.Lvv), filltriu=lib.SYMMETRIC)
        eris._vvvv_pair = _build_eris_vvvv(Lvv)
    return eris


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
        from pyscfad.lno._df_outcore import _outcore_nr_e2
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
    eris.get_ovvv_packed()
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
    from . import _dfccsd_native
    if _dfccsd_native.enabled(mycc):
        return _dfccsd_native.forward(mycc, eris, t1, t2)
    return _dfccsd_kernel_plain(mycc, eris, t1, t2)


def _dfccsd_kernel_custom_fwd(mycc, eris, t1=None, t2=None):
    from . import _dfccsd_native
    native = _dfccsd_native.enabled(mycc)
    out = (_dfccsd_native.forward(mycc, eris, t1, t2) if native
           else _dfccsd_kernel_plain(mycc, eris, t1, t2))
    _, _, t1_out, t2_out = out
    return out, (mycc, eris, t1_out, t2_out, native)


def _coerce_eris_tensors_for_bwd(eris):
    """Re-wrap host-side TypedNdArray literals on eris as proper jnp arrays.

    JAX saves non-dynamic eris attributes as static aux data and re-emits them
    as TypedNdArray host literals in the bwd; those lack .transpose() and other
    array methods needed by update_amps / lagrangian_grad.
    """
    import copy
    out = copy.copy(eris)
    for attr in ('fock', 'mo_energy', 'oooo', 'ovoo', 'ovov', 'oovv',
                 'ovvo', 'ovvv', 'Lvv', 'Lov', 'mo_coeff', '_vvvv_pair'):
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
    mycc, eris, t1, t2, native = res
    _, bar_e, bar_t1, bar_t2 = cotangent
    eris = _coerce_eris_tensors_for_bwd(eris)
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)
    nocc, nvir = t1.shape
    amplitudes_mib = resource_profile.estimated_array_mib(t1, t2)
    eris_mib = resource_profile.estimated_array_mib(
        *(getattr(eris, attr, None) for attr in (
            'fock', 'mo_energy', 'oooo', 'ovoo', 'ovov', 'oovv',
            'ovvo', 'ovvv', 'Lvv', 'Lov', 'mo_coeff', '_vvvv_pair',
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

    if native:
        from . import _dfccsd_native
        eris_bar = _dfccsd_native.backward(
            mycc, eris, t1, t2, np.asarray(bar_e_val),
            np.zeros_like(t1) if bar_t1_val is None else bar_t1_val,
            np.zeros_like(t2) if bar_t2_val is None else bar_t2_val)
        resource_profile.finish('ccsd_bwd.total', profile_total,
                                nocc=nocc, nvir=nvir, backend='numpy')
        return None, eris_bar, None, None

    profile_lambda = resource_profile.start()
    lambda_converged, lambda_vec, intermediates = ccsd_lambda.solve_response_lambda(
        mycc, eris, t1, t2,
        bar_e_val, bar_t1_val, bar_t2_val,
        max_cycle=mycc.max_cycle,
        tol=mycc.conv_tol_normt,
        verbose=mycc.verbose,
        return_intermediates=True,
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
    profile_lagrangian = resource_profile.start()
    eris_bar = lagrangian_grad(mycc, eris, t1, t2, bar_e_val, lambda_vec,
                              intermediates=intermediates)
    resource_profile.finish(
        'ccsd_bwd.lagrangian_gradient',
        profile_lagrangian,
        nocc=nocc,
        nvir=nvir,
        reused_update_intermediates=True,
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
                                       t1new_bar, t2new_bar, *, intermediates=None):
    if mycc.cc2:
        raise NotImplementedError
    if mycc.direct:
        raise NotImplementedError('DF-CCSD direct AO update is not supported')
    if mycc.dcsd:
        raise NotImplementedError('Explicit DF-CCSD cotangent does not support DCSD')
    if eris.Lov is None or eris.Lvv is None:
        raise NotImplementedError('lagrangian_grad requires DF Lov/Lvv leaves')

    nocc, nvir = t1.shape
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

    # The response linearization already evaluated this exact update. Its
    # auxiliary outputs share the JAX residual arrays and live only until the
    # integral adjoint finishes. Standalone callers can still prepare them.
    if intermediates is None:
        _, intermediates = mycc.update_amps(
            t1, t2, eris, return_intermediates=True)
    (eia, eijab, t1work, t2sym, fov, foo, fvv, ft_ij, ft_ab,
     woooo, wVOov, wVooV, theta_ovvv, tau_for_vvvo, tmp_acc,
     tmp_oovv_voov, theta, tau_full, tau_half, tau_resp, eris_VOov,
     theta_t1, theta_t2, tau_final) = (
        intermediates[name] for name in (
            'eia', 'eijab', 't1work', 't2sym', 'fov', 'foo', 'fvv', 'ft_ij', 'ft_ab',
            'woooo', 'wVOov', 'wVooV', 'theta_ovvv', 'tau_for_vvvo', 'tmp_acc',
            'tmp_oovv_voov', 'theta', 'tau_full', 'tau_half', 'tau_resp', 'eris_VOov',
            'theta_t1', 'theta_t2', 'tau_final'))
    eris_ovoo, eris_oovv = eris.ovoo, eris.oovv
    eris_voov = eris.ovvo.conj().transpose(1, 0, 3, 2)
    tau_vvvv = tau_full[numpy.tril_indices(nocc)]
    blksize = max(1, min(_CCSD_OVVV_BLKSIZE, nvir))

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
            if eris.ovvv is not None:
                vovv_packed = eris.ovvv[:, p0:p1].transpose(1, 0, 2)
            else:
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
    Lvv_full = lib.unpack_tril(eris.Lvv, filltriu=lib.SYMMETRIC)
    Lvv_full_bar = _G_Lbar(tau_vvvv, Lvv_full,
                           Ht2tril_bar.transpose(0, 2, 1))
    Lvv_grad = _adjoint_unpack_tril_last2(Lvv_full_bar, nvir)
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


def lagrangian_grad(mycc, eris, t1, t2, bar_e_val, lambda_vec, *, intermediates=None):
    """Hand-coded ERI cotangent of the DF-CCSD response Lagrangian."""
    nocc, nvir = t1.shape
    t1new_bar, t2new_bar = _amplitudes_to_vector_cotangent(
        lambda_vec, nocc, nvir
    )
    update_bars = _dfccsd_update_amps_eris_cotangent(
        mycc, eris, t1, t2, t1new_bar, t2new_bar, intermediates=intermediates
    )
    e_fock_bar, e_ovov_bar = _energy_eris_cotangent(
        eris, t1, t2, bar_e_val
    )

    # None denotes a zero cotangent to custom_vjp. The analytical ovvv
    # response already enters through Lov/Lvv, so its cache must not add a
    # second gradient path or allocate a cache-sized zero here.
    import copy
    eris_without_cache = copy.copy(eris)
    eris_without_cache._vvvv_pair = None
    eris_without_cache.ovvv = None
    eris_bar = jax.tree_util.tree_map(_zero_like_leaf, eris_without_cache)
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
