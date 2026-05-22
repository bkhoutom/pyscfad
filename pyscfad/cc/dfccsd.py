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

from dataclasses import dataclass
from functools import partial
from pyscf.lib import square_mat_in_trilu_indices
import numpy
import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.interpreters import ad as jax_ad
from pyscfad import numpy as np
from pyscfad import lib
from pyscfad import config, config_update
from pyscfad.ao2mo import _ao2mo
from pyscfad.cc import ccsd, ccsd_lambda
from pyscfad.tools.linear_solver import gen_gmres


@dataclass(frozen=True)
class _DFCCSDResponseContext:
    nmo: int
    nocc: int
    level_shift: float
    cc2: bool
    direct: bool
    dcsd: bool

    def vector_to_amplitudes(self, vec):
        return ccsd.vector_to_amplitudes(vec, self.nmo, self.nocc)

    def amplitudes_to_vector(self, t1, t2, out=None):
        return ccsd.amplitudes_to_vector(t1, t2, out=out)

    def _add_vvvv(self, t1, t2, eris, out=None, with_ovvv=None, t2sym=None):
        return ccsd._add_vvvv(self, t1, t2, eris, out=out,
                              with_ovvv=with_ovvv, t2sym=t2sym)


class _DFCCSDResponseERIs:
    def __init__(self, fock, mo_energy, oooo, ovoo, ovov, oovv, ovvo,
                 ovvv, Lvv):
        self.fock = fock
        self.mo_energy = mo_energy
        self.oooo = oooo
        self.ovoo = ovoo
        self.ovov = ovov
        self.oovv = oovv
        self.ovvo = ovvo
        self.ovvv = ovvv
        self.vvvv = None
        self.Lvv = Lvv
        self.mol = None

    def _contract_vvvv_t2(self, mycc, t2, direct=False, out=None,
                          verbose=None):
        assert not direct
        return _contract_vvvv_t2(mycc, self.mol, self.Lvv, t2)


@partial(jax.jit, static_argnums=0)
def _dfccsd_response_optimality(ctx, amp, fock, mo_energy, oooo, ovoo, ovov,
                                oovv, ovvo, ovvv, Lvv):
    eris = _DFCCSDResponseERIs(fock, mo_energy, oooo, ovoo, ovov, oovv, ovvo,
                               ovvv, Lvv)
    t1, t2 = ctx.vector_to_amplitudes(amp)
    t1new, t2new = ccsd.update_amps(ctx, t1, t2, eris)
    return ctx.amplitudes_to_vector(t1new, t2new) - amp


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
        g_acbd = lib.unpack_tril(jnp.dot(jnp.transpose(lac), Lvv))
        g_cdb = jnp.transpose(g_acbd, (0, 2, 1)).reshape(nvir*nvir, nvir)
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


def _amplitudes_cotangent_to_vector(bar_t1, bar_t2):
    nocc, nvir = bar_t1.shape
    nov = nocc * nvir
    bar_t2_mat = np.transpose(bar_t2, (0, 2, 1, 3)).reshape(nov, nov)
    idx = numpy.tril_indices(nov)
    lower = bar_t2_mat[idx]
    upper = np.transpose(bar_t2_mat)[idx]
    bar_t2_vec = np.where(idx[0] == idx[1], lower, lower + upper)
    return np.concatenate((bar_t1.ravel(), bar_t2_vec), axis=None)


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
        bar_amp += _amplitudes_cotangent_to_vector(bar_t1, bar_t2)

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

    def optimality(amp_, eris_):
        return fixed_point(amp_, eris_) - amp_

    eris_leaves, eris_treedef = jax.tree_util.tree_flatten(eris)
    if len(eris_leaves) != 9:
        def optimality_from_eris_leaves(amp_, *eris_leaves_):
            eris_ = jax.tree_util.tree_unflatten(eris_treedef, eris_leaves_)
            return optimality(amp_, eris_)
    else:
        ctx = _DFCCSDResponseContext(
            nmo=int(mycc.nmo),
            nocc=int(mycc.nocc),
            level_shift=float(mycc.level_shift),
            cc2=bool(mycc.cc2),
            direct=bool(mycc.direct),
            dcsd=bool(mycc.dcsd),
        )

        def optimality_from_eris_leaves(amp_, *eris_leaves_):
            return _dfccsd_response_optimality(ctx, amp_, *eris_leaves_)

    _, optimality_vjp = jax.vjp(
        optimality_from_eris_leaves, amp, *eris_leaves
    )

    def matvec(u):
        return optimality_vjp(u)[0]

    solver = gen_gmres(tol=mycc.conv_tol_normt)
    lambda_bar = solver(matvec, -bar_amp)[0]
    response_bars = optimality_vjp(lambda_bar)
    eris_bar_response = jax.tree_util.tree_unflatten(
        eris_treedef, response_bars[1:]
    )
    return _tree_add(eris_bar, eris_bar_response)


_dfccsd_kernel_custom.defvjp(
    _dfccsd_kernel_custom_fwd,
    _dfccsd_kernel_custom_bwd,
)
