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
from jax import custom_vjp
from jax.interpreters import ad as jax_ad
from pyscfad import numpy as np
from pyscfad import lib
from pyscfad import config, config_update
from pyscfad.ao2mo import _ao2mo
from pyscfad.cc import ccsd, ccsd_lambda
from pyscfad.tools.linear_solver import gen_gmres

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

def _contract_vvvv_t2(mycc, mol, Lvv, t2, out=None, verbose=None):
    '''Ht2 = numpy.einsum('ijcd,acbd->ijab', t2, vvvv)
    '''
    nvir = t2.shape[-1]
    nvir2 = nvir * nvir
    x2 = t2.reshape(-1, nvir2)

    tril2sq = square_mat_in_trilu_indices(nvir)
    tmp = lib.unpack_tril(np.dot(np.transpose(Lvv), Lvv))
    tmp1 = np.transpose(tmp[tril2sq], (0, 2, 1, 3)).reshape(nvir2,nvir2)
    Ht2tril = np.dot(x2, tmp1)
    tril2sq = None
    return Ht2tril.reshape(t2.shape)

class _ChemistsERIs(ccsd._ChemistsERIs):
    _dynamic_attr = {'Lvv'}

    def __init__(self, mol=None):
        super().__init__(mol=mol)
        self.naux = None
        self.Lvv = None

    def _contract_vvvv_t2(self, mycc, t2, direct=False, out=None, verbose=None):
        assert not direct
        return _contract_vvvv_t2(mycc, self.mol, self.Lvv, t2)


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
    eri1 = with_df._cderi
    # pylint: disable=too-many-function-args
    Lpq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1').reshape(-1,nmo,nmo)
    Loo = Lpq[:,:nocc,:nocc].reshape(naux,-1)
    Lov = Lpq[:,:nocc,nocc:].reshape(naux,-1)
    eris.Lvv = Lvv = lib.pack_tril(Lpq[:,nocc:,nocc:])

    eris.oooo = np.dot(np.transpose(Loo), Loo).reshape(nocc,nocc,nocc,nocc)
    eris.ovoo = np.dot(np.transpose(Lov), Loo).reshape(nocc,nvir,nocc,nocc)
    ovov = np.dot(np.transpose(Lov), Lov).reshape(nocc,nvir,nocc,nvir)
    eris.ovov = ovov
    eris.ovvo = np.transpose(ovov, (0,1,3,2))

    oovv = np.dot(np.transpose(Loo), Lvv)
    eris.oovv = lib.unpack_tril(oovv).reshape(nocc,nocc,nvir,nvir)
    eris.ovvv = np.dot(np.transpose(Lov), Lvv).reshape(nocc,nvir,-1)
    return eris


def _is_zero_cotangent(x):
    return x is None or isinstance(x, jax_ad.Zero)


def _has_concrete_nonzero_cotangent(x):
    if _is_zero_cotangent(x):
        return False
    for leaf in jax.tree_util.tree_leaves(x):
        try:
            if bool(np.any(np.asarray(leaf))):
                return True
        except Exception:  # Traced cotangents cannot be inspected here.
            pass
    return False


def _has_tracer(x):
    if _is_zero_cotangent(x):
        return False
    for leaf in jax.tree_util.tree_leaves(x):
        if isinstance(leaf, (jax.core.Tracer, jax_ad.JVPTracer)):
            return True
        for attr in ('primal', 'val'):
            if hasattr(leaf, attr):
                if _has_tracer(getattr(leaf, attr)):
                    return True
    return False


def _cotangent_is_zero(x):
    if _is_zero_cotangent(x):
        return True
    for leaf in jax.tree_util.tree_leaves(x):
        try:
            if bool(np.any(np.asarray(leaf))):
                return False
        except Exception:  # Traced cotangents should take the general path.
            return False
    return True


def _cotangent_or_zeros(x, ref):
    if _is_zero_cotangent(x):
        return np.zeros_like(ref)
    return np.asarray(x)


def _tree_add(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return jax.tree_util.tree_map(lambda a, b: a + b, x, y)


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

    if (
        _has_tracer(eris)
        or _has_tracer(t1)
        or _has_tracer(t2)
        or _has_tracer(bar_e)
        or _has_concrete_nonzero_cotangent(bar_t1)
        or _has_concrete_nonzero_cotangent(bar_t2)
        or not _cotangent_is_zero(bar_t1)
        or not _cotangent_is_zero(bar_t2)
    ):
        eris_bar = _dfccsd_kernel_custom_bwd_with_amplitudes(
            mycc, eris, t1, t2, bar_e, bar_t1, bar_t2
        )
        return None, eris_bar, None, None

    _, l1, l2 = ccsd_lambda.kernel(
        mycc,
        eris,
        t1,
        t2,
        max_cycle=mycc.max_cycle,
        tol=mycc.conv_tol_normt,
    )

    def lagrangian(eris_):
        ecc = mycc.energy(t1, t2, eris_)
        t1new, t2new = mycc.update_amps(t1, t2, eris_)
        nocc = t1.shape[0]
        mo_e = eris_.mo_energy
        eia = mo_e[:nocc, None] - mo_e[None, nocc:]
        eijab = eia[:, None, :, None] + eia[None, :, None, :]
        amp_dot = 2 * np.vdot(l1, (t1new - t1) * eia)
        l2_metric = l2 * 2 - np.transpose(l2, (1, 0, 2, 3))
        amp_dot += np.vdot(l2_metric, (t2new - t2) * eijab)
        return np.real(bar_e * (ecc + amp_dot))

    eris_bar = jax.grad(lagrangian)(eris)
    return None, eris_bar, None, None


def _dfccsd_kernel_custom_bwd_with_amplitudes(
    mycc, eris, t1, t2, bar_e, bar_t1, bar_t2
):
    amp = mycc.amplitudes_to_vector(t1, t2)
    eris_bar = None

    bar_amp = np.zeros_like(amp)
    if not _cotangent_is_zero(bar_t1) or not _cotangent_is_zero(bar_t2):
        bar_t1 = _cotangent_or_zeros(bar_t1, t1)
        bar_t2 = _cotangent_or_zeros(bar_t2, t2)
        _, amplitudes_vjp = jax.vjp(mycc.vector_to_amplitudes, amp)
        bar_amp += amplitudes_vjp((bar_t1, bar_t2))[0]

    if not _cotangent_is_zero(bar_e):
        def scaled_energy(amp_, eris_):
            t1_, t2_ = mycc.vector_to_amplitudes(amp_)
            return np.real(bar_e * mycc.energy(t1_, t2_, eris_))

        bar_amp_e, eris_bar_e = jax.grad(scaled_energy, argnums=(0, 1))(amp, eris)
        bar_amp += bar_amp_e
        eris_bar = _tree_add(eris_bar, eris_bar_e)

    if _cotangent_is_zero(bar_amp):
        return eris_bar

    def fixed_point(amp_, eris_):
        t1_, t2_ = mycc.vector_to_amplitudes(amp_)
        t1new, t2new = mycc.update_amps(t1_, t2_, eris_)
        return mycc.amplitudes_to_vector(t1new, t2new)

    def optimality_amp(amp_):
        return fixed_point(amp_, eris) - amp_

    _, amp_vjp = jax.vjp(optimality_amp, amp)

    def matvec(u):
        return amp_vjp(u)[0]

    solver = gen_gmres(tol=mycc.conv_tol_normt)
    lambda_bar = solver(matvec, -bar_amp)[0]

    def optimality_eris(eris_):
        return fixed_point(amp, eris_) - amp

    _, eris_vjp = jax.vjp(optimality_eris, eris)
    eris_bar_response = eris_vjp(lambda_bar)[0]
    return _tree_add(eris_bar, eris_bar_response)


_dfccsd_kernel_custom.defvjp(
    _dfccsd_kernel_custom_fwd,
    _dfccsd_kernel_custom_bwd,
)
