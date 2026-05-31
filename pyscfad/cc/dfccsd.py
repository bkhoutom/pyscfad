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
    _dynamic_attr = {'Lvv', 'Lov'}

    def __init__(self, mol=None):
        super().__init__(mol=mol)
        self.naux = None
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


def _dfccsd_kernel_custom_bwd(res, cotangent):
    """Backward of the DF-CCSD custom-VJP via implicit-diff-form lambda equations.

    Given upstream cotangents (bar_e on the CCSD energy, bar_t1/bar_t2 on the
    converged amplitudes), solve a single response lambda equation that
    incorporates both, then evaluate the ERI cotangent as the gradient of the
    Lagrangian ``bar_e * E_cc(t, eris) + lambda . Omega(t, eris)`` with
    ``Omega = update_amps(t, eris) - t``.
    """
    mycc, eris, t1, t2 = res
    _, bar_e, bar_t1, bar_t2 = cotangent
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)

    if (
        _is_zero_cotangent(bar_e)
        and _is_zero_cotangent(bar_t1)
        and _is_zero_cotangent(bar_t2)
    ):
        return None, None, None, None

    bar_e_val = 0.0 if _is_zero_cotangent(bar_e) else np.asarray(bar_e)
    bar_t1_val = None if _is_zero_cotangent(bar_t1) else np.asarray(bar_t1)
    bar_t2_val = None if _is_zero_cotangent(bar_t2) else np.asarray(bar_t2)

    _, lambda_vec = ccsd_lambda.solve_response_lambda(
        mycc, eris, t1, t2,
        bar_e_val, bar_t1_val, bar_t2_val,
        max_cycle=mycc.max_cycle,
        tol=mycc.conv_tol_normt,
        verbose=mycc.verbose,
    )

    def lagrangian(eris_):
        # Lambda already absorbs bar_e * dE/dt + bar_t via the response solve,
        # so the Lagrangian scales only the energy term by bar_e.
        ecc = mycc.energy(t1, t2, eris_)
        t1new, t2new = mycc.update_amps(t1, t2, eris_)
        omega_vec = mycc.amplitudes_to_vector(t1new - t1, t2new - t2)
        return np.real(bar_e_val * ecc + np.vdot(lambda_vec, omega_vec))

    eris_bar = jax.grad(lagrangian)(eris)
    return None, eris_bar, None, None


_dfccsd_kernel_custom.defvjp(
    _dfccsd_kernel_custom_fwd,
    _dfccsd_kernel_custom_bwd,
)
