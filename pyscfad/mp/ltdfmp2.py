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

from functools import wraps
import numpy

import jax
from pyscf import __config__
from pyscf.mp import dfmp2 as pyscf_dfmp2

from pyscfad import numpy as np
from pyscfad.lib import logger
from pyscfad.mp import dfmp2

WITH_T2 = False
NLAP = getattr(__config__, 'mp_ltdfmp2_nlap', 9)
C_OS = getattr(__config__, 'mp_ltdfmp2_c_os', 1.3)
TMIN_FACTOR = getattr(__config__, 'mp_ltdfmp2_tmin_factor', 1e-3)
TMAX_FACTOR = getattr(__config__, 'mp_ltdfmp2_tmax_factor', 30.0)
QUADRATURE = getattr(__config__, 'mp_ltdfmp2_quadrature', 'fit')
FIT_GRID_SIZE = getattr(__config__, 'mp_ltdfmp2_fit_grid_size', 800)
FIT_RATIO = getattr(__config__, 'mp_ltdfmp2_fit_ratio', None)
SOS = getattr(__config__, 'mp_ltdfmp2_sos', False)

_FITTED_EXP_SUMS = {
    # Positive 9-term relative least-squares exponential-sum fits for
    # 1/y on [1, R].  For 1/x on [dmin, dmax], use t=b/dmin and w=a/dmin.
    9: (
        (16.0,
         (0.025010067147824246, 0.13466509901496815,
          0.34478387050149473, 0.68209483468285848,
          1.1970560387386868, 1.9803114757977995,
          3.191375700416522, 5.1239541258540102,
          8.4529256235177943),
         (0.064504241927554634, 0.15684062838125024,
          0.26769721428217313, 0.41506829990485561,
          0.62952197736378002, 0.96267660389809429,
          1.5056253691394528, 2.456844140659344,
          4.5400818957860816)),
        (32.0,
         (0.014537768596405248, 0.078916127112281478,
          0.20535288642176194, 0.41726240563036121,
          0.76155082155002873, 1.3259087346781004,
          2.2683217312740798, 3.8837433449384893,
          6.8473749675756075),
         (0.037562998216650122, 0.0928439584378489,
          0.1637615997657354, 0.26759430389104744,
          0.43514882730631721, 0.71896892093570564,
          1.2122475804043642, 2.1164123370608876,
          4.1448917668336849)),
        (64.0,
         (0.0082811652165765241, 0.045389272809550664,
          0.12047754865638573, 0.25308205236680109,
          0.48455040754621814, 0.89538712637603601,
          1.6365652244001485, 3.0004046016482757,
          5.6642771516445825),
         (0.021442609610451111, 0.054045372432098823,
          0.099216565411731725, 0.17260724591357049,
          0.30327209242801439, 0.54255277168376803,
          0.98570007296642259, 1.8404901141605845,
          3.8188562421066052)),
        (128.0,
         (0.0046450505358859697, 0.025748180174878885,
          0.069989036026774878, 0.15301276805195829,
          0.30962381320808613, 0.61079306215611162,
          1.1971414644919456, 2.3551761911088751,
          4.7648475813339948),
         (0.01205691827229978, 0.031091276658501185,
          0.059840972877868298, 0.11174342917624189,
          0.21279812179995489, 0.41187086405454426,
          0.80564497247956224, 1.6095791289724142,
          3.5407245675894612)),
        (256.0,
         (0.0025740718534039675, 0.014454985595537571,
          0.040408457471100681, 0.092533577244430579,
          0.19905941783904305, 0.42075652696245586,
          0.88609355387812894, 1.8728373369829343,
          4.0625785851172527),
         (0.0066998552914333855, 0.017738111054904616,
          0.036049730122669071, 0.072656378847361119,
          0.14998999917497177, 0.31362166439388606,
          0.66025790483217917, 1.4127402073074888,
          3.2983831331085907)),
        (512.0,
         (0.0014125743671613001, 0.008050975115958767,
          0.023248984028988985, 0.056090018300554226,
          0.12882822223822069, 0.2924167645329136,
          0.66238153000976896, 1.5051347785645248,
          3.5020235108548863),
         (0.0036880640501202951, 0.010061969312075623,
          0.021735459905227874, 0.047423698826167371,
          0.10600479918350204, 0.23916324949848861,
          0.54188840926631321, 1.2430299029920524,
          3.0840379866284833)),
    ),
}


def laplace_quadrature(dmin, dmax, n=NLAP, tmin_factor=TMIN_FACTOR,
                       tmax_factor=TMAX_FACTOR):
    if n < 1:
        raise ValueError('n must be positive')

    tmin = tmin_factor / dmax
    tmax = tmax_factor / dmin
    if n == 1:
        t = np.sqrt(tmin * tmax)[None]
        return t, t

    logs = np.linspace(np.log(tmin), np.log(tmax), n)
    t = np.exp(logs)
    w0 = 0.5 * t[0] * (logs[1] - logs[0])
    w1 = 0.5 * t[-1] * (logs[-1] - logs[-2])
    wm = 0.5 * t[1:-1] * (logs[2:] - logs[:-2])
    w = np.concatenate((w0[None], wm, w1[None]))
    return t, w


def fitted_laplace_quadrature(dmin, dmax, n=NLAP, grid_size=FIT_GRID_SIZE,
                              fit_ratio=FIT_RATIO):
    del grid_size
    if fit_ratio is None:
        dmin_f = float(numpy.asarray(dmin))
        dmax_f = float(numpy.asarray(dmax))
        ratio = dmax_f / dmin_f
    else:
        ratio = float(fit_ratio)

    for rmax, b, weights in _FITTED_EXP_SUMS.get(int(n), ()):
        if ratio <= rmax:
            break
    else:
        raise ValueError(
            f'No hard-coded LT-DFMP2 fitted quadrature for n={n}, '
            f'dmax/dmin={ratio:.6g}.'
        )
    b = np.asarray(b)
    weights = np.asarray(weights)
    return b / dmin, weights / dmin


def _contract_laplace_os(Lov, mo_energy, nocc, nvir, nlap=NLAP, c_os=C_OS,
                         tmin_factor=TMIN_FACTOR, tmax_factor=TMAX_FACTOR,
                         quadrature=QUADRATURE, fit_grid_size=FIT_GRID_SIZE,
                         fit_ratio=FIT_RATIO):
    Lov = Lov.reshape((-1, nocc, nvir))
    naux = Lov.shape[0]
    eo = mo_energy[:nocc]
    ev = mo_energy[nocc:nocc+nvir]

    dmin = 2.0 * (np.min(ev) - np.max(eo))
    dmax = 2.0 * (np.max(ev) - np.min(eo))
    if quadrature == 'fit':
        t, w = fitted_laplace_quadrature(
            dmin, dmax, n=nlap, grid_size=fit_grid_size,
            fit_ratio=fit_ratio,
        )
    elif quadrature == 'logtrap':
        t, w = laplace_quadrature(
            dmin, dmax, n=nlap,
            tmin_factor=tmin_factor, tmax_factor=tmax_factor,
        )
    else:
        raise ValueError(f'Unknown LT-DFMP2 quadrature: {quadrature}')

    emp2_os = np.zeros((), dtype=Lov.dtype)
    for tq, wq in zip(t, w):
        so = np.exp(0.5 * eo * tq)
        sv = np.exp(-0.5 * ev * tq)
        Lov_q = Lov * so[None, :, None] * sv[None, None, :]
        Lov_q = Lov_q.reshape(naux, nocc*nvir)
        metric_q = np.dot(Lov_q, Lov_q.T)
        emp2_os -= wq * np.einsum('PQ,PQ->', metric_q, metric_q)

    return (c_os * emp2_os).real


def _laplace_quadrature_for_mp2(mo_energy, nocc, nvir, nlap=NLAP,
                                tmin_factor=TMIN_FACTOR,
                                tmax_factor=TMAX_FACTOR,
                                quadrature=QUADRATURE,
                                fit_grid_size=FIT_GRID_SIZE,
                                fit_ratio=FIT_RATIO):
    eo = mo_energy[:nocc]
    ev = mo_energy[nocc:nocc+nvir]
    dmin = 2.0 * (np.min(ev) - np.max(eo))
    dmax = 2.0 * (np.max(ev) - np.min(eo))
    if quadrature == 'fit':
        return fitted_laplace_quadrature(
            dmin, dmax, n=nlap, grid_size=fit_grid_size,
            fit_ratio=fit_ratio,
        )
    if quadrature == 'logtrap':
        return laplace_quadrature(
            dmin, dmax, n=nlap,
            tmin_factor=tmin_factor, tmax_factor=tmax_factor,
        )
    raise ValueError(f'Unknown LT-DFMP2 quadrature: {quadrature}')


def _contract_laplace_mp2(Lov, mo_energy, nocc, nvir, nlap=NLAP,
                          tmin_factor=TMIN_FACTOR,
                          tmax_factor=TMAX_FACTOR,
                          quadrature=QUADRATURE,
                          fit_grid_size=FIT_GRID_SIZE,
                          fit_ratio=FIT_RATIO):
    Lov = Lov.reshape((-1, nocc, nvir))
    naux = Lov.shape[0]
    eo = mo_energy[:nocc]
    ev = mo_energy[nocc:nocc+nvir]
    t, w = _laplace_quadrature_for_mp2(
        mo_energy, nocc, nvir, nlap=nlap,
        tmin_factor=tmin_factor, tmax_factor=tmax_factor,
        quadrature=quadrature, fit_grid_size=fit_grid_size,
        fit_ratio=fit_ratio,
    )

    e_direct = np.zeros((), dtype=Lov.dtype)
    e_exchange = np.zeros((), dtype=Lov.dtype)
    for tq, wq in zip(t, w):
        so = np.exp(0.5 * eo * tq)
        sv = np.exp(-0.5 * ev * tq)
        Lov_q = Lov * so[None, :, None] * sv[None, None, :]
        Lov_q_flat = Lov_q.reshape(naux, nocc*nvir)
        for i in range(nocc):
            gi = np.dot(Lov_q[:, i].T, Lov_q_flat).reshape(nvir, nocc, nvir)
            gi = gi.transpose(1, 0, 2)
            e_direct -= 2.0 * wq * np.einsum('jab,jab->', gi, gi)
            e_exchange += wq * np.einsum('jab,jba->', gi, gi)

    emp2 = e_direct + e_exchange
    return emp2.real, e_direct.real, e_exchange.real


def _contract_laplace_projected_occ(
        Lov, mo_energy, occ_projector, nocc, nvir, nlap=NLAP,
        tmin_factor=TMIN_FACTOR, tmax_factor=TMAX_FACTOR,
        quadrature=QUADRATURE, fit_grid_size=FIT_GRID_SIZE,
        fit_ratio=FIT_RATIO):
    Lov = Lov.reshape((-1, nocc, nvir))
    naux = Lov.shape[0]
    eo = mo_energy[:nocc]
    ev = mo_energy[nocc:nocc+nvir]
    occ_projector = np.asarray(occ_projector)
    single_orbital = occ_projector.ndim == 1
    if single_orbital:
        occ_projector = occ_projector[:, None]

    t, w = _laplace_quadrature_for_mp2(
        mo_energy, nocc, nvir, nlap=nlap,
        tmin_factor=tmin_factor, tmax_factor=tmax_factor,
        quadrature=quadrature, fit_grid_size=fit_grid_size,
        fit_ratio=fit_ratio,
    )

    nlo = occ_projector.shape[1]
    Lov_flat = Lov.reshape(naux, nocc*nvir)

    @jax.checkpoint
    def _contract_one_local(occ_vec):
        Lov_local = np.einsum('i,Lia->La', occ_vec, Lov)
        g_local = np.dot(Lov_local.T, Lov_flat)
        g_local = g_local.reshape(nvir, nocc, nvir).transpose(1, 0, 2)
        g_local_ex = g_local.transpose(0, 2, 1)

        @jax.checkpoint
        def _contract_one_t(carry, x):
            e_direct, e_exchange = carry
            tq, wq = x
            so = np.exp(eo * tq)
            sv = np.exp(-ev * tq)
            Lov_t = Lov * so[None, :, None] * sv[None, None, :]
            Lov_t_flat = Lov_t.reshape(naux, nocc*nvir)
            Lov_local_t = np.einsum(
                'i,Lia->La',
                occ_vec * so,
                Lov,
            )
            Lov_local_t = Lov_local_t * sv[None, :]
            gi = np.dot(Lov_local_t.T, Lov_t_flat)
            gi = gi.reshape(nvir, nocc, nvir).transpose(1, 0, 2)
            e_direct += -2.0 * wq * np.einsum('jab,jab->', gi, g_local)
            e_exchange += wq * np.einsum('jab,jab->', gi, g_local_ex)
            return (e_direct, e_exchange), None

        e0 = np.zeros((), dtype=Lov.dtype)
        (e_direct, e_exchange), _ = jax.lax.scan(
            _contract_one_t, (e0, e0), (t, w)
        )
        return e_direct, e_exchange

    @jax.checkpoint
    def _contract_one_projector(carry, occ_vec):
        return carry, _contract_one_local(occ_vec)

    _, (e_direct, e_exchange) = jax.lax.scan(
        _contract_one_projector,
        np.zeros((), dtype=Lov.dtype),
        occ_projector.T,
    )

    emp2 = e_direct + e_exchange
    if single_orbital:
        return emp2[0].real, e_direct[0].real, e_exchange[0].real
    return emp2.real, e_direct.real, e_exchange.real


@wraps(pyscf_dfmp2.kernel)
def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
           verbose=None):
    log = logger.new_logger(mp, verbose)

    if mo_energy is not None or mo_coeff is not None:
        assert (mp.frozen == 0 or mp.frozen is None)

    if eris is None:
        eris = mp.ao2mo(mo_coeff)
    if mo_energy is None:
        mo_energy = eris.mo_energy
    if mo_coeff is None:
        mo_coeff = eris.mo_coeff

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    Lov = mp.loop_ao2mo(mo_coeff, nocc, False)
    kwargs = {
        'nlap': getattr(mp, 'nlap', NLAP),
        'tmin_factor': getattr(mp, 'tmin_factor', TMIN_FACTOR),
        'tmax_factor': getattr(mp, 'tmax_factor', TMAX_FACTOR),
        'quadrature': getattr(mp, 'quadrature', QUADRATURE),
        'fit_grid_size': getattr(mp, 'fit_grid_size', FIT_GRID_SIZE),
        'fit_ratio': getattr(mp, 'fit_ratio', FIT_RATIO),
    }
    if getattr(mp, 'sos', SOS):
        emp2 = _contract_laplace_os(
            Lov, mo_energy, nocc, nvir,
            c_os=getattr(mp, 'c_os', C_OS), **kwargs,
        )
        e_direct = emp2
        e_exchange = np.zeros_like(emp2)
        e_corr_os = emp2 / getattr(mp, 'c_os', C_OS)
        e_corr_ss = np.zeros_like(emp2)
    else:
        emp2, e_direct, e_exchange = _contract_laplace_mp2(
            Lov, mo_energy, nocc, nvir, **kwargs,
        )
        e_corr_os = 0.5 * e_direct
        e_corr_ss = e_corr_os + e_exchange

    mp.e_corr_direct = e_direct
    mp.e_corr_exchange = e_exchange
    mp.e_corr_ss = e_corr_ss
    mp.e_corr_os = e_corr_os

    if with_t2:
        log.warn(
            '%s does not build conventional MP2 t2 amplitudes; returning None for t2.',
            mp.__class__.__name__,
        )
    return emp2, None


class MP2(dfmp2.MP2):
    """Laplace-transformed density-fitted MP2.

    The public interface mirrors :mod:`pyscfad.mp.dfmp2`: construct from a
    density-fitted mean-field object and call ``kernel``/``init_amps``.  The
    method evaluates direct and exchange LT-DF-MP2 energy contributions and
    does not build conventional ``t2`` amplitudes.  Set ``sos=True`` for the
    older scaled-opposite-spin-only value.
    """

    _dynamic_attr = _keys = dfmp2.MP2._keys.union(
        {'nlap', 'c_os', 'tmin_factor', 'tmax_factor'}
        | {'quadrature', 'fit_grid_size', 'fit_ratio', 'sos'}
    )

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None,
                 nlap=NLAP, c_os=C_OS, tmin_factor=TMIN_FACTOR,
                 tmax_factor=TMAX_FACTOR, quadrature=QUADRATURE,
                 fit_grid_size=FIT_GRID_SIZE, fit_ratio=FIT_RATIO,
                 sos=SOS):
        super().__init__(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)
        self.nlap = nlap
        self.c_os = c_os
        self.tmin_factor = tmin_factor
        self.tmax_factor = tmax_factor
        self.quadrature = quadrature
        self.fit_grid_size = fit_grid_size
        self.fit_ratio = fit_ratio
        self.sos = sos

    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        if self.verbose >= logger.WARN:
            self.check_sanity()

        self.dump_flags()
        self.e_hf = self.get_e_hf(mo_coeff=mo_coeff)

        if eris is None:
            eris = self.ao2mo(mo_coeff)

        if self._scf.converged:
            self.e_corr, self.t2 = self.init_amps(
                mo_energy, mo_coeff, eris, with_t2
            )
        else:
            raise NotImplementedError

        self._finalize()
        return self.e_corr, self.t2

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None,
                  with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

    def projected_occ_energy(self, occ_projector, mo_energy=None,
                             mo_coeff=None, eris=None):
        if eris is None:
            eris = self.ao2mo(mo_coeff)
        if mo_energy is None:
            mo_energy = eris.mo_energy
        if mo_coeff is None:
            mo_coeff = eris.mo_coeff

        nocc = self.nocc
        nvir = self.nmo - nocc
        Lov = self.loop_ao2mo(mo_coeff, nocc, False)
        return _contract_laplace_projected_occ(
            Lov,
            mo_energy,
            occ_projector,
            nocc,
            nvir,
            nlap=self.nlap,
            tmin_factor=self.tmin_factor,
            tmax_factor=self.tmax_factor,
            quadrature=self.quadrature,
            fit_grid_size=self.fit_grid_size,
            fit_ratio=self.fit_ratio,
        )


RMP2 = MP2
DFMP2 = MP2


def sos_lt_mp2(mf, auxbasis=None, nlap=NLAP, c_os=C_OS, frozen=0,
               tmin_factor=TMIN_FACTOR, tmax_factor=TMAX_FACTOR,
               quadrature=QUADRATURE, fit_grid_size=FIT_GRID_SIZE,
               fit_ratio=FIT_RATIO):
    if getattr(mf, 'with_df', None) is None:
        mf = mf.density_fit(auxbasis=auxbasis)
    elif auxbasis is not None:
        mf = mf.copy()
        mf.with_df = mf.with_df.copy()
        mf.with_df.auxbasis = auxbasis

    mymp = MP2(
        mf, frozen=frozen, nlap=nlap, c_os=c_os,
        tmin_factor=tmin_factor, tmax_factor=tmax_factor,
        quadrature=quadrature, fit_grid_size=fit_grid_size,
        fit_ratio=fit_ratio, sos=True,
    )
    emp2, _ = mymp.kernel(with_t2=False)
    return emp2


del WITH_T2
