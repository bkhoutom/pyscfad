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
import time

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

    log = logger.new_logger(mycc)
    t0 = time.perf_counter()
    log.info('CCSD response backward: lambda equations start')
    _, lambda_vec = ccsd_lambda.solve_response_lambda(
        mycc, eris, t1, t2,
        bar_e_val, bar_t1_val, bar_t2_val,
        max_cycle=mycc.max_cycle,
        tol=mycc.conv_tol_normt,
        verbose=mycc.verbose,
    )
    log.info('CCSD response backward: lambda equations done in %.2f s',
             time.perf_counter() - t0)

    t0 = time.perf_counter()
    log.info('CCSD response backward: ERI/Lagrangian VJP start')
    eris_bar = lagrangian_grad(mycc, eris, t1, t2, bar_e_val, lambda_vec)
    log.info('CCSD response backward: ERI/Lagrangian VJP done in %.2f s',
             time.perf_counter() - t0)
    return None, eris_bar, None, None


_dfccsd_kernel_custom.defvjp(
    _dfccsd_kernel_custom_fwd,
    _dfccsd_kernel_custom_bwd,
)


def _unbroadcast(bar, shape):
    if shape == ():
        return np.sum(bar)
    while bar.ndim > len(shape):
        bar = np.sum(bar, axis=0)
    axes = [i for i, size in enumerate(shape)
            if size == 1 and bar.shape[i] != 1]
    if axes:
        bar = np.sum(bar, axis=tuple(axes), keepdims=True)
    return bar.reshape(shape)


class _ReverseVar:
    __array_priority__ = 1000

    def __init__(self, val, tape):
        self.val = val
        self.bar = None
        self.tape = tape

    @property
    def shape(self):
        return self.val.shape

    @property
    def dtype(self):
        return self.val.dtype

    @property
    def ndim(self):
        return self.val.ndim

    @property
    def T(self):
        return self.transpose()

    def add_bar(self, bar):
        if bar is None:
            return
        self.bar = bar if self.bar is None else self.bar + bar

    def copy(self):
        out = _ReverseVar(self.val.copy(), self.tape)

        def bwd(bar):
            self.add_bar(bar)

        self.tape.append((out, bwd))
        return out

    def conj(self):
        out = _ReverseVar(np.conj(self.val), self.tape)

        def bwd(bar):
            self.add_bar(np.conj(bar))

        self.tape.append((out, bwd))
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        old_shape = self.val.shape
        out = _ReverseVar(self.val.reshape(*shape), self.tape)

        def bwd(bar):
            self.add_bar(bar.reshape(old_shape))

        self.tape.append((out, bwd))
        return out

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], tuple):
            axes = axes[0]
        elif not axes:
            axes = tuple(reversed(range(self.val.ndim)))
        inv_axes = numpy.argsort(axes)
        out = _ReverseVar(self.val.transpose(*axes), self.tape)

        def bwd(bar):
            self.add_bar(bar.transpose(*inv_axes))

        self.tape.append((out, bwd))
        return out

    def __getitem__(self, idx):
        out = _ReverseVar(self.val[idx], self.tape)
        shape = self.val.shape
        dtype = self.val.dtype

        def bwd(bar):
            base = np.zeros(shape, dtype=dtype)
            self.add_bar(base.at[idx].add(bar))

        self.tape.append((out, bwd))
        return out

    def __add__(self, other):
        other_val = _rv_value(other)
        out = _ReverseVar(self.val + other_val, self.tape)

        def bwd(bar):
            self.add_bar(_unbroadcast(bar, self.val.shape))
            if _is_rv(other):
                other.add_bar(_unbroadcast(bar, other.val.shape))

        self.tape.append((out, bwd))
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other_val = _rv_value(other)
        out = _ReverseVar(self.val - other_val, self.tape)

        def bwd(bar):
            self.add_bar(_unbroadcast(bar, self.val.shape))
            if _is_rv(other):
                other.add_bar(_unbroadcast(-bar, other.val.shape))

        self.tape.append((out, bwd))
        return out

    def __rsub__(self, other):
        other_val = _rv_value(other)
        out = _ReverseVar(other_val - self.val, self.tape)

        def bwd(bar):
            if _is_rv(other):
                other.add_bar(_unbroadcast(bar, other.val.shape))
            self.add_bar(_unbroadcast(-bar, self.val.shape))

        self.tape.append((out, bwd))
        return out

    def __neg__(self):
        out = _ReverseVar(-self.val, self.tape)

        def bwd(bar):
            self.add_bar(-bar)

        self.tape.append((out, bwd))
        return out

    def __mul__(self, other):
        other_val = _rv_value(other)
        out = _ReverseVar(self.val * other_val, self.tape)

        def bwd(bar):
            self.add_bar(_unbroadcast(bar * other_val, self.val.shape))
            if _is_rv(other):
                other.add_bar(_unbroadcast(bar * self.val, other.val.shape))

        self.tape.append((out, bwd))
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other_val = _rv_value(other)
        out = _ReverseVar(self.val / other_val, self.tape)

        def bwd(bar):
            self.add_bar(_unbroadcast(bar / other_val, self.val.shape))
            if _is_rv(other):
                other.add_bar(_unbroadcast(
                    -bar * self.val / (other_val * other_val),
                    other.val.shape,
                ))

        self.tape.append((out, bwd))
        return out

    def __rtruediv__(self, other):
        other_val = _rv_value(other)
        out = _ReverseVar(other_val / self.val, self.tape)

        def bwd(bar):
            if _is_rv(other):
                other.add_bar(_unbroadcast(bar / self.val, other.val.shape))
            self.add_bar(_unbroadcast(
                -bar * other_val / (self.val * self.val),
                self.val.shape,
            ))

        self.tape.append((out, bwd))
        return out


def _is_rv(x):
    return isinstance(x, _ReverseVar)


def _rv_value(x):
    return x.val if _is_rv(x) else x


def _rv_add_bar(x, bar):
    if _is_rv(x):
        x.add_bar(bar)


def _rv_leaf(val, tape):
    return _ReverseVar(val, tape)


def _rv_zeros(shape, dtype):
    return np.zeros(shape, dtype=dtype)


def _rv_zeros_like(x):
    return np.zeros_like(_rv_value(x))


def _rv_asarray(x):
    return x


def _rv_diag(x):
    if not _is_rv(x):
        return np.diag(x)
    out = _ReverseVar(np.diag(x.val), x.tape)

    def bwd(bar):
        x.add_bar(np.diag(bar))

    x.tape.append((out, bwd))
    return out


def _rv_dot(a, b):
    aval = _rv_value(a)
    bval = _rv_value(b)
    if not (_is_rv(a) or _is_rv(b)):
        return np.dot(aval, bval)
    tape = a.tape if _is_rv(a) else b.tape
    out = _ReverseVar(np.dot(aval, bval), tape)

    def bwd(bar):
        if _is_rv(a):
            a.add_bar(np.dot(bar, np.swapaxes(bval, -1, -2)))
        if _is_rv(b):
            b.add_bar(np.dot(np.swapaxes(aval, -1, -2), bar))

    tape.append((out, bwd))
    return out


def _rv_einsum(subscripts, *operands):
    vals = [_rv_value(x) for x in operands]
    if not any(_is_rv(x) for x in operands):
        return np.einsum(subscripts, *vals)

    lhs, out_spec = subscripts.replace(' ', '').split('->')
    in_specs = lhs.split(',')
    tape = next(x.tape for x in operands if _is_rv(x))
    out = _ReverseVar(np.einsum(subscripts, *vals), tape)

    def bwd(bar):
        for i, operand in enumerate(operands):
            if not _is_rv(operand):
                continue
            other_specs = [s for j, s in enumerate(in_specs) if j != i]
            other_vals = [v for j, v in enumerate(vals) if j != i]
            grad_subscripts = ','.join([out_spec] + other_specs)
            grad_subscripts += '->' + in_specs[i]
            grad = np.einsum(grad_subscripts, bar, *other_vals)
            operand.add_bar(_unbroadcast(grad, operand.val.shape))

    tape.append((out, bwd))
    return out


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


def _rv_unpack_tril(packed, filltriu=lib.HERMITIAN):
    if not _is_rv(packed):
        return lib.unpack_tril(packed, filltriu=filltriu)
    out_val = lib.unpack_tril(packed.val, filltriu=filltriu)
    out = _ReverseVar(out_val, packed.tape)
    n = out_val.shape[-1]

    def bwd(bar):
        packed.add_bar(_adjoint_unpack_tril_last2(bar, n, filltriu))

    packed.tape.append((out, bwd))
    return out


def _rv_scatter_add(base, idx, update):
    base_val = _rv_value(base)
    upd_val = _rv_value(update)
    if not (_is_rv(base) or _is_rv(update)):
        return base_val.at[idx].add(upd_val)
    tape = base.tape if _is_rv(base) else update.tape
    out = _ReverseVar(base_val.at[idx].add(upd_val), tape)

    def bwd(bar):
        if _is_rv(base):
            base.add_bar(bar)
        if _is_rv(update):
            update.add_bar(bar[idx])

    tape.append((out, bwd))
    return out


def _rv_scatter_set(base, idx, update):
    base_val = _rv_value(base)
    upd_val = _rv_value(update)
    if not (_is_rv(base) or _is_rv(update)):
        return base_val.at[idx].set(upd_val)
    tape = base.tape if _is_rv(base) else update.tape
    out = _ReverseVar(base_val.at[idx].set(upd_val), tape)

    def bwd(bar):
        if _is_rv(base):
            base.add_bar(bar.at[idx].set(0))
        if _is_rv(update):
            update.add_bar(bar[idx])

    tape.append((out, bwd))
    return out


def _rv_contract_vvvv_t2(Lvv, t2):
    if not (_is_rv(Lvv) or _is_rv(t2)):
        return _contract_vvvv_t2_lowmem(Lvv, t2)
    Lvv_val = _rv_value(Lvv)
    t2_val = _rv_value(t2)
    tape = Lvv.tape if _is_rv(Lvv) else t2.tape
    out = _ReverseVar(_contract_vvvv_t2_lowmem(Lvv_val, t2_val), tape)

    def bwd(bar):
        Lvv_bar, t2_bar = _contract_vvvv_t2_lowmem_bwd((Lvv_val, t2_val), bar)
        _rv_add_bar(Lvv, Lvv_bar)
        _rv_add_bar(t2, t2_bar)

    tape.append((out, bwd))
    return out


def _rv_unpack_t2_tril(t2tril, nocc, nvir, t2sym='jiba'):
    if not _is_rv(t2tril):
        return ccsd._unpack_t2_tril(t2tril, nocc, nvir, t2sym=t2sym)
    out_val = ccsd._unpack_t2_tril(t2tril.val, nocc, nvir, t2sym=t2sym)
    out = _ReverseVar(out_val, t2tril.tape)
    idx, idy = numpy.tril_indices(nocc)

    def bwd(bar):
        if t2sym != 'jiba':
            raise NotImplementedError('Only jiba t2 symmetry is supported')
        packed_bar = bar[idx, idy]
        trans_bar = bar[idy, idx].transpose(0, 2, 1)
        packed_bar = packed_bar + trans_bar
        diag = numpy.asarray(idx == idy)
        if numpy.any(diag):
            packed_bar = packed_bar.at[diag].set(trans_bar[diag])
        t2tril.add_bar(packed_bar)

    t2tril.tape.append((out, bwd))
    return out


def _amplitudes_to_vector_cotangent(lambda_vec, nocc, nvir):
    nov = nocc * nvir
    l1 = lambda_vec[:nov].reshape(nocc, nvir)
    packed = lambda_vec[nov:]
    mat = np.zeros((nov, nov), dtype=lambda_vec.dtype)
    idx = numpy.tril_indices(nov)
    mat = mat.at[idx].set(packed)
    l2 = mat.reshape(nocc, nvir, nocc, nvir).transpose(0, 2, 1, 3)
    return l1, l2


class _ReverseERIs:
    pass


def _dfccsd_update_amps_eris_cotangent_tape(mycc, eris, t1, t2,
                                            t1new_bar, t2new_bar):
    if mycc.cc2:
        raise NotImplementedError
    if mycc.direct:
        raise NotImplementedError('DF-CCSD direct AO update is not supported')

    nocc, nvir = t1.shape
    nvir_pair = nvir * (nvir + 1) // 2
    tape = []

    eris_v = _ReverseERIs()
    for key in ('fock', 'mo_energy', 'oooo', 'ovoo', 'ovov',
                'oovv', 'ovvo', 'Lov', 'Lvv'):
        val = getattr(eris, key, None)
        setattr(eris_v, key, None if val is None else _rv_leaf(val, tape))

    fock = eris_v.fock
    mo_e_o = eris_v.mo_energy[:nocc]
    mo_e_v = eris_v.mo_energy[nocc:] + mycc.level_shift

    idx_occ = numpy.tril_indices(nocc)
    tau_vvvv = t2[idx_occ]
    tau_vvvv = tau_vvvv + np.einsum('ia,jb->ijab', t1, t1)[idx_occ]
    Ht2tril = _rv_contract_vvvv_t2(eris_v.Lvv, tau_vvvv)
    t2new = _rv_unpack_t2_tril(Ht2tril, nocc, nvir, t2sym='jiba')
    t2new = t2new * .5

    t1new = _rv_zeros_like(t1)
    fov = fock[:nocc, nocc:].copy()
    t1new = t1new + fov

    foo = fock[:nocc, :nocc] - _rv_diag(mo_e_o)
    foo = foo + .5 * _rv_einsum('ia,ja->ij', fock[:nocc, nocc:], t1)
    fvv = fock[nocc:, nocc:] - _rv_diag(mo_e_v)
    fvv = fvv - .5 * _rv_einsum('ia,ib->ab', t1, fock[:nocc, nocc:])

    if eris_v.Lov is None or eris_v.Lvv is None:
        raise NotImplementedError('lagrangian_grad requires DF Lov/Lvv leaves')

    Lov_full = eris_v.Lov
    Lvv_full = eris_v.Lvv
    blksize = max(1, min(_CCSD_OVVV_BLKSIZE, nvir))

    wooVV_flat = _rv_zeros((nocc, nocc * nvir_pair), t1.dtype)
    wVOov = _rv_zeros((nvir, nocc, nocc, nvir), t1.dtype)

    theta = t2.transpose(1, 2, 0, 3) * 2
    theta = theta - t2.transpose(0, 2, 1, 3)

    if not mycc.direct:
        tau_for_vvvo = t2 + np.einsum('ia,jb->ijab', t1, t1)
        tmp_acc = _rv_zeros((nocc, nocc, nvir, nocc), t1.dtype)

    for p0 in range(0, nvir, blksize):
        p1 = min(p0 + blksize, nvir)
        bw = p1 - p0

        Lov_tile = Lov_full[:, :, p0:p1]
        vovv_packed = _rv_einsum('xia,xb->aib', Lov_tile, Lvv_full)
        wooVV_flat = wooVV_flat - _rv_dot(
            t1[:, p0:p1],
            vovv_packed.reshape(bw, nocc * nvir_pair),
        )

        vovv_tile = _rv_unpack_tril(
            vovv_packed.reshape(bw * nocc, nvir_pair)
        )
        vovv_tile = vovv_tile.reshape(bw, nocc, nvir, nvir)

        fvv = fvv + 2 * _rv_einsum('kc,ckab->ab',
                                   t1[:, p0:p1], vovv_tile)
        fvv = _rv_scatter_add(
            fvv,
            (slice(None), slice(p0, p1)),
            -_rv_einsum('kc,bkca->ab', t1, vovv_tile),
        )

        if not mycc.direct:
            vvvo_tile = vovv_tile.transpose(0, 2, 3, 1)
            tmp_acc = tmp_acc + _rv_einsum(
                'ijcd,cdbk->ijbk',
                tau_for_vvvo[:, :, p0:p1, :],
                vvvo_tile,
            )

        wVOov = _rv_scatter_set(
            wVOov,
            slice(p0, p1),
            _rv_einsum('biac,jc->bija', vovv_tile, t1),
        )

        t1new = t1new + _rv_einsum(
            'icjb,cjba->ia',
            theta[:, p0:p1, :, :],
            vovv_tile,
        )

    if not mycc.direct:
        t2new = t2new - _rv_einsum('ka,ijbk->ijab', t1, tmp_acc)

    wooVV = _rv_unpack_tril(wooVV_flat.reshape(nocc**2, nvir_pair))
    wVooV = wooVV.reshape(nocc, nocc, nvir, nvir).transpose(2, 1, 0, 3)

    woooo = _rv_asarray(eris_v.oooo).transpose(0, 2, 1, 3).copy()
    eris_ovoo = eris_v.ovoo
    eris_oovv = eris_v.oovv

    foo = foo + _rv_einsum('kc,kcji->ij', 2 * t1, eris_ovoo)
    foo = foo + _rv_einsum('kc,icjk->ij', -t1, eris_ovoo)
    tmp = _rv_einsum('la,jaik->lkji', t1, eris_ovoo)
    woooo = woooo + tmp + tmp.transpose(1, 0, 3, 2)

    wVOov = wVOov - _rv_einsum('jbik,ka->bjia', eris_ovoo, t1)
    t2new = t2new + wVOov.transpose(1, 2, 0, 3)

    wVooV = wVooV + _rv_einsum('kbij,ka->bija', eris_ovoo, t1)

    eris_ovvo = eris_v.ovvo
    t1new = t1new - _rv_einsum('jb,jiab->ia', t1, eris_oovv)
    wVooV = wVooV - eris_oovv.transpose(2, 0, 1, 3)
    wVOov = wVOov + wVooV * .5

    t2new = t2new + (eris_ovvo * .5).transpose(0, 3, 1, 2)
    eris_voov = eris_ovvo.conj().transpose(1, 0, 3, 2)
    t1new = t1new + 2 * _rv_einsum('jb,aijb->ia', t1, eris_voov)

    tmp = _rv_einsum('ic,kjbc->ibkj', t1, eris_oovv)
    tmp = tmp + _rv_einsum('bjkc,ic->jbki', eris_voov, t1)
    t2new = t2new - _rv_einsum('ka,jbki->jiba', t1, tmp)

    fov = fov + _rv_einsum('kc,aikc->ia', t1, eris_voov) * 2
    fov = fov - _rv_einsum('kc,akic->ia', t1, eris_voov)

    tau = np.einsum('ia,jb->ijab', t1 * .5, t1)
    if mycc.dcsd:
        tau = tau + t2 * .5
        theta = t2.transpose(1, 0, 2, 3) - t2 * .5
        fvv_t2 = -_rv_einsum(
            'cjia,cjib->ab', theta.transpose(2, 1, 0, 3), eris_voov)
        foo_t2 = _rv_einsum('aikb,kjab->ij', eris_voov, theta)
    else:
        tau = tau + t2
    theta = tau.transpose(1, 0, 2, 3) * 2
    theta = theta - tau
    fvv = fvv - _rv_einsum(
        'cjia,cjib->ab', theta.transpose(2, 1, 0, 3), eris_voov)
    foo = foo + _rv_einsum('aikb,kjab->ij', eris_voov, theta)

    tau = np.einsum('ia,jb->ijab', t1, t1)
    if mycc.dcsd:
        woooo_t2 = _rv_einsum('ijab,aklb->ijkl', t2, eris_voov)
    else:
        tau = tau + t2
    woooo = woooo + _rv_einsum('ijab,aklb->ijkl', tau, eris_voov)

    tau = np.einsum('ia,jb->ijab', t1, t1)
    if not mycc.dcsd:
        tau = tau + t2 * .5
    wVooV = wVooV + _rv_einsum('bkic,jkca->bija', eris_voov, tau)

    tmp = _rv_einsum('jkca,ckib->jaib', t2, wVooV)
    t2new = t2new + tmp.transpose(2, 0, 1, 3)
    tmp_half = tmp * .5
    t2new = t2new + tmp_half.transpose(0, 2, 1, 3)

    wVOov = wVOov + eris_voov
    eris_VOov = -.5 * eris_voov.transpose(0, 2, 1, 3)
    tau = t2.transpose(1, 3, 0, 2) * 2
    tau = tau - t2.transpose(0, 3, 1, 2)
    tau1 = -np.einsum('ia,jb->ibja', t1 * 2, t1)
    tau = tau + tau1
    if mycc.dcsd:
        wVOov = wVOov + .5 * _rv_einsum('aikc,kcjb->aijb',
                                         eris_voov, tau)
        wVOov = wVOov + .5 * _rv_einsum('aikc,kcjb->aijb',
                                         eris_VOov, tau1)
    else:
        eris_VOov = eris_VOov + eris_voov
        wVOov = wVOov + .5 * _rv_einsum('aikc,kcjb->aijb',
                                         eris_VOov, tau)

    theta = t2 * 2
    theta = theta - t2.transpose(1, 0, 2, 3)
    t2new = t2new + _rv_einsum('kica,ckjb->ijab', theta, wVOov)

    theta = t2.transpose(1, 0, 2, 3) * 2 - t2
    t1new = t1new + _rv_einsum('jb,ijba->ia', fov, theta)
    t1new = t1new - _rv_einsum('jbki,kjba->ia', eris_ovoo, theta)

    tau = np.einsum('ia,jb->ijab', t1, t1)
    if mycc.dcsd:
        t2new = t2new + .5 * _rv_einsum('ijkl,klab->ijab',
                                         woooo_t2, tau)
    tau = tau + t2
    t2new = t2new + .5 * _rv_einsum('ijkl,klab->ijab', woooo, tau)

    ft_ij = foo + _rv_einsum('ja,ia->ij', .5 * t1, fov)
    ft_ab = fvv - _rv_einsum('ia,ib->ab', .5 * t1, fov)
    t2new = t2new + _rv_einsum('ijac,bc->ijab', t2, ft_ab)
    t2new = t2new - _rv_einsum('ki,kjab->ijab', ft_ij, t2)

    if mycc.dcsd:
        fvv = fvv + fvv_t2
        foo = foo + foo_t2
    t1new = t1new + _rv_einsum('ib,ab->ia', t1, fvv)
    t1new = t1new - _rv_einsum('ja,ji->ia', t1, foo)
    t2new = t2new + t2new.transpose(1, 0, 3, 2)

    eia = mo_e_o[:, None] - mo_e_v
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    t1new = t1new / eia
    t2new = t2new / eijab

    t1new.add_bar(t1new_bar)
    t2new.add_bar(t2new_bar)
    for out, bwd in reversed(tape):
        if out.bar is not None:
            bwd(out.bar)

    return {key: getattr(eris_v, key).bar
            for key in ('fock', 'mo_energy', 'oooo', 'ovoo', 'ovov',
                        'oovv', 'ovvo', 'Lov', 'Lvv')}


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


def _dfccsd_update_amps_eris_cotangent_explicit(mycc, eris, t1, t2,
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


def _dfccsd_update_amps_eris_cotangent(mycc, eris, t1, t2,
                                       t1new_bar, t2new_bar):
    return _dfccsd_update_amps_eris_cotangent_tape(
        mycc, eris, t1, t2, t1new_bar, t2new_bar)


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
