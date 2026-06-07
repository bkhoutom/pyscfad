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
FIT_TOL = getattr(__config__, 'mp_ltdfmp2_fit_tol', 1e-5)
SOS = getattr(__config__, 'mp_ltdfmp2_sos', False)

_FITTED_EXP_SUMS = {
    # Positive n-term relative least-squares exponential-sum fits for
    # 1/y on [1, R].  For 1/x on [dmin, dmax], use t=b/dmin and w=a/dmin.
    5: (
        (16.0,
         (0.043214791264697766, 0.24443381747300758,
          0.69450698465857641, 1.6386994231937171,
          3.7493828270214222),
         (0.11264945529225082, 0.30265680355422392,
          0.63497272608503097, 1.3515815567007103,
          3.2023578695242501)),
        (32.0,
         (0.025116946239207912, 0.14620240309477134,
          0.44235895050866281, 1.1493564663062739,
          2.942420348815308),
         (0.065859423222699248, 0.18748376978060291,
          0.44054864371177999, 1.0717847703803129,
          2.8505433397868201)),
        (64.0,
         (0.014317136254427503, 0.086327843416961272,
          0.28198241346269382, 0.81709232768984452,
          2.3561884811233464),
         (0.037804539114554822, 0.11543996253302105,
          0.3080385479215188, 0.85792876340687374,
          2.561310404099201)),
        (128.0,
         (0.0080419210929727133, 0.05058375779961171,
          0.18055320034771991, 0.58830065926754926,
          1.9177745973667051),
         (0.021408825889386095, 0.070971802237417211,
          0.21668411300192778, 0.69001111158006034,
          2.3152141360782559)),
        (256.0,
         (0.00446601104644344, 0.0295187472472761,
          0.11630379812543112, 0.42826305026432421,
          1.5814658309047136),
         (0.0120015509187351, 0.043671898978209688,
          0.15301189359757186, 0.55618506913887267,
          2.1012798043536649)),
        (512.0,
         (0.0024581146139687083, 0.017198907035328101,
          0.075396587917587063, 0.31467126716057542,
          1.3179466492209408),
         (0.0066769220966653511, 0.026925169605295202,
          0.10828646512588126, 0.44871475942866429,
          1.9125390742131394)),
    ),
    6: (
        (16.0,
         (0.036546476981663009, 0.20221590487167404,
          0.54765681640088426, 1.1912957240826354,
          2.4071365115278822, 4.8719833102299281),
         (0.094824450722173811, 0.24363698576885945,
          0.46571873990035467, 0.86431023389547812,
          1.6661209466902847, 3.594826940249976)),
        (32.0,
         (0.021236109533915481, 0.1197791948131334,
          0.33809984951512917, 0.78707394524186791,
          1.7362251366745725, 3.8674715728103597),
         (0.055324771138266338, 0.14778776358131085,
          0.3060048836400901, 0.63380718769983779,
          1.3641613555315175, 3.2303498655035177)),
        (64.0,
         (0.012098494862093976, 0.06985125564697385,
          0.20753636969197892, 0.52353756249634464,
          1.2724804965774839, 3.1339902784197786),
         (0.031671334859536313, 0.088689019004331265,
          0.20174205549008406, 0.46935003785608931,
          1.1275699875556486, 2.930209449171723)),
        (128.0,
         (0.0067898066881266891, 0.040307621525283902,
          0.12728835981186995, 0.35129075449773445,
          0.94586983947201841, 2.5820525033720831),
         (0.017873737641186063, 0.052920589581236958,
          0.13365080608133612, 0.34960530983080024,
          0.93662689181305747, 2.6744478357754131)),
        (256.0,
         (0.0037661302451952143, 0.023099960443949245,
          0.078239307382959242, 0.23782302972260591,
          0.71151154281054263, 2.1558548491557592),
         (0.0099776135132127429, 0.031505302845005731,
          0.088939774062935156, 0.2612248295661535,
          0.78003901218921734, 2.451832834952945)),
        (512.0,
         (0.0020695761358664848, 0.013182364231716712,
          0.048270895516208612, 0.16233976339595238,
          0.5405245947211289, 1.8195542058134919),
         (0.0055227578823836988, 0.018752417080896436,
          0.059388415238834889, 0.19548272918074772,
          0.65050897837192878, 2.2551766195533491)),
    ),
    7: (
        (16.0,
         (0.031674178716742607, 0.17298024068107404,
          0.45575515041603576, 0.94591400603237796,
          1.7831626985257547, 3.2545039465321524,
          6.0357269493616696),
         (0.081949495949113174, 0.20504731167034848,
          0.37092048606054034, 0.63161235445765029,
          1.0879647094769884, 1.9529375330268393,
          3.9419929834239826)),
        (32.0,
         (0.01840511817178481, 0.10190832104100671,
          0.27636159836438323, 0.60198323347473037,
          1.2135363967939043, 2.3967052064427277,
          4.8317894450795285),
         (0.047761706287277585, 0.12288582659282282,
          0.23544859339428617, 0.43714955174794001,
          0.83072298665332034, 1.6348604917953624,
          3.5662202914519217)),
        (64.0,
         (0.010484086654220869, 0.059020683281387903,
          0.16588759437289863, 0.38329925808470694,
          0.83475257840146677, 1.7961952780456847,
          3.9494853414253006),
         (0.027302439570005051, 0.07264860036183593,
          0.14918742563615844, 0.30504782836370881,
          0.64077701178993884, 1.3816883625251928,
          3.256518780193252)),
        (128.0,
         (0.0058818139866383791, 0.033767786103665938,
          0.099084377644678243, 0.24512181570268154,
          0.58067140311565046, 1.3664793623122,
          3.282644790144321),
         (0.015379653972580978, 0.04257986674407329,
          0.094706307488201191, 0.21425416656465338,
          0.49695448088270339, 1.1738109570273059,
          2.9924232262001822)),
        (256.0,
         (0.0032607140249028299, 0.019153327163870391,
          0.059101939914531332, 0.15771684868338712,
          0.40813721295704358, 1.052543504496674,
          2.7652497952217518),
         (0.0085655789669314367, 0.024830908775073105,
          0.060322641038251114, 0.15113334757644956,
          0.38644916807406349, 1.0001874112375275,
          2.7623906895889681)),
        (512.0,
         (0.0017904903762768804, 0.010798697886940715,
          0.035286591602089384, 0.1021499816676717,
          0.28949225484931534, 0.81904210732309068,
          2.354962953317528),
         (0.0047280373714461621, 0.014444203878885155,
          0.038556022064348702, 0.10688259132646089,
          0.30089574808475544, 0.85376474095196997,
          2.5590329659991786)),
    ),
    8: (
        (16.0,
         (0.027954328966445011, 0.15138529688224645,
          0.39205720791051341, 0.79045801605826072,
          1.4270643279957014, 2.4542863863080164,
          4.1642010576309358, 7.2313752768086932),
         (0.072190680298570578, 0.1775780888633364,
          0.31022952185158059, 0.49948265260566177,
          0.79818233475711753, 1.3022024708280604,
          2.2152593842803086, 4.2543167136915994)),
        (32.0,
         (0.016245353747424651, 0.088893765732441635,
          0.23514525225900285, 0.49116273533716437,
          0.93315448039723781, 1.7106059610337838,
          3.1160293041709513, 5.8266195004286416),
         (0.042049196655321253, 0.1056288378970941,
          0.19259077185518342, 0.33162583472281071,
          0.57628834027062192, 1.0245897685182765,
          1.8847151955160999, 3.8682435319380875)),
        (64.0,
         (0.0092539378555823639, 0.051269423146861073,
          0.13920888296003153, 0.30380231597064317,
          0.61340582866738425, 1.2082552209311785,
          2.3752900056943931, 4.7943025382910651),
         (0.024017302054052336, 0.061868893175566431,
          0.1188238805041153, 0.22123129661997115,
          0.42027389681599864, 0.81434052681043234,
          1.6188648430821047, 3.54991601668375)),
        (128.0,
         (0.0051911443322581489, 0.029182460507450256,
          0.081768077060534905, 0.18795992878719833,
          0.40638103214793131, 0.86438131983099054,
          1.839005346576966, 4.0115190451422489),
         (0.013514655431117394, 0.035857780307808548,
          0.073199191565253452, 0.14842741733109702,
          0.30846034997231475, 0.65062010116501501,
          1.3980978901783176, 3.2783837034661936)),
        (256.0,
         (0.002877177257841982, 0.01644978769353752,
          0.047830096589294234, 0.11664910675035886,
          0.27150121665265176, 0.6253600738618591,
          1.4420568854144165, 3.4020144487950352),
         (0.0075168037900391112, 0.020637645119017514,
          0.045146821953525755, 0.10006512087309337,
          0.22720471385572022, 0.52116983917383186,
          1.2114677624879899, 3.0418075748514717)),
        (512.0,
         (0.0015793126666724198, 0.0092062101432536115,
          0.027935481391319742, 0.072720953649299883,
          0.18285156360476648, 0.45678607787157594,
          1.1426361338264568, 2.9168943466937343),
         (0.0041423503324614086, 0.011825867244819372,
          0.027913262895887511, 0.067695996564498873,
          0.16766252310065111, 0.41801147855677467,
          1.0520124283506656, 2.8325873420388001)),
    ),
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


_FITTED_EXP_SUM_MAX_REL_ERR = {
    5: (
        (16.0, 1.552387186616322e-04),
        (32.0, 8.138651258131624e-04),
        (64.0, 2.768770413615451e-03),
        (128.0, 7.076658825332371e-03),
        (256.0, 1.4828638508376457e-02),
        (512.0, 2.6919638603723928e-02),
    ),
    6: (
        (16.0, 1.7138907859926e-05),
        (32.0, 1.2622115980909232e-04),
        (64.0, 5.539923958667314e-04),
        (128.0, 1.7262558662440863e-03),
        (256.0, 4.238184423733804e-03),
        (512.0, 8.7581852347427e-03),
    ),
    7: (
        (16.0, 1.8413269577965963e-06),
        (32.0, 1.905240083754922e-05),
        (64.0, 1.0789250314924281e-04),
        (128.0, 4.09896673730481e-04),
        (256.0, 1.179335994845987e-03),
        (512.0, 2.7747940356848133e-03),
    ),
    8: (
        (16.0, 1.9383395299943373e-07),
        (32.0, 2.818103120638682e-06),
        (64.0, 2.0592173740952013e-05),
        (128.0, 9.538852370094553e-05),
        (256.0, 3.216346199398368e-04),
        (512.0, 8.61685817743929e-04),
    ),
    9: (
        (16.0, 1.9458147648698798e-08),
        (32.0, 3.9834164045071674e-07),
        (64.0, 3.7669357364489287e-06),
        (128.0, 2.1331905786681205e-05),
        (256.0, 8.448674090844044e-05),
        (512.0, 2.5823753828357887e-04),
    ),
}


def _fit_ratio_value(dmin, dmax, fit_ratio=FIT_RATIO):
    if fit_ratio is None:
        dmin_f = float(numpy.asarray(dmin))
        dmax_f = float(numpy.asarray(dmax))
        return dmax_f / dmin_f
    return float(fit_ratio)


def _fitted_laplace_entry(n, ratio):
    for rmax, b, weights in _FITTED_EXP_SUMS.get(int(n), ()):
        if ratio <= rmax:
            return rmax, b, weights
    return None


def _fitted_laplace_error(n, ratio):
    for rmax, err in _FITTED_EXP_SUM_MAX_REL_ERR.get(int(n), ()):
        if ratio <= rmax:
            return rmax, err
    return None


def _fitted_laplace_actual_error(n, ratio, grid_size=FIT_GRID_SIZE):
    entry = _fitted_laplace_entry(n, ratio)
    if entry is None:
        return None
    _, b, weights = entry
    y = numpy.exp(numpy.linspace(0.0, numpy.log(ratio), int(grid_size)))
    fit = numpy.exp(-numpy.outer(y, numpy.asarray(b))) @ numpy.asarray(weights)
    return numpy.max(numpy.abs(y * fit - 1.0))


def _is_auto_nlap(n):
    return n is None or (isinstance(n, str) and n.lower() == 'auto')


def select_fitted_laplace_nlap(dmin, dmax, target=FIT_TOL,
                               fit_ratio=FIT_RATIO,
                               grid_size=FIT_GRID_SIZE):
    ratio = _fit_ratio_value(dmin, dmax, fit_ratio=fit_ratio)
    available = []
    for n in sorted(_FITTED_EXP_SUMS):
        table_entry = _fitted_laplace_entry(n, ratio)
        if table_entry is None:
            continue
        err = _fitted_laplace_actual_error(n, ratio, grid_size=grid_size)
        rmax = table_entry[0]
        available.append((n, rmax, err))
        if err <= target:
            return n
    if not available:
        raise ValueError(
            'No hard-coded LT-DFMP2 fitted quadrature covers '
            f'dmax/dmin={ratio:.6g}.'
        )
    return available[-1][0]


def laplace_quadrature(dmin, dmax, n=NLAP, tmin_factor=TMIN_FACTOR,
                       tmax_factor=TMAX_FACTOR):
    if _is_auto_nlap(n):
        raise ValueError("nlap='auto' is only available for fitted quadrature")
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
                              fit_ratio=FIT_RATIO, fit_tol=FIT_TOL):
    ratio = _fit_ratio_value(dmin, dmax, fit_ratio=fit_ratio)
    if _is_auto_nlap(n):
        n = select_fitted_laplace_nlap(
            dmin, dmax, target=fit_tol, fit_ratio=fit_ratio,
            grid_size=grid_size,
        )

    entry = _fitted_laplace_entry(n, ratio)
    if entry is None:
        raise ValueError(
            f'No hard-coded LT-DFMP2 fitted quadrature for n={n}, '
            f'dmax/dmin={ratio:.6g}.'
        )
    _, b, weights = entry
    b = np.asarray(b)
    weights = np.asarray(weights)
    return b / dmin, weights / dmin


def _contract_laplace_os(Lov, mo_energy, nocc, nvir, nlap=NLAP, c_os=C_OS,
                         tmin_factor=TMIN_FACTOR, tmax_factor=TMAX_FACTOR,
                         quadrature=QUADRATURE, fit_grid_size=FIT_GRID_SIZE,
                         fit_ratio=FIT_RATIO, fit_tol=FIT_TOL):
    Lov = Lov.reshape((-1, nocc, nvir))
    naux = Lov.shape[0]
    eo = mo_energy[:nocc]
    ev = mo_energy[nocc:nocc+nvir]

    dmin = 2.0 * (np.min(ev) - np.max(eo))
    dmax = 2.0 * (np.max(ev) - np.min(eo))
    if quadrature == 'fit':
        t, w = fitted_laplace_quadrature(
            dmin, dmax, n=nlap, grid_size=fit_grid_size,
            fit_ratio=fit_ratio, fit_tol=fit_tol,
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
                                fit_ratio=FIT_RATIO,
                                fit_tol=FIT_TOL):
    eo = mo_energy[:nocc]
    ev = mo_energy[nocc:nocc+nvir]
    dmin = 2.0 * (np.min(ev) - np.max(eo))
    dmax = 2.0 * (np.max(ev) - np.min(eo))
    if quadrature == 'fit':
        return fitted_laplace_quadrature(
            dmin, dmax, n=nlap, grid_size=fit_grid_size,
            fit_ratio=fit_ratio, fit_tol=fit_tol,
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
                          fit_ratio=FIT_RATIO, fit_tol=FIT_TOL):
    Lov = Lov.reshape((-1, nocc, nvir))
    naux = Lov.shape[0]
    eo = mo_energy[:nocc]
    ev = mo_energy[nocc:nocc+nvir]
    t, w = _laplace_quadrature_for_mp2(
        mo_energy, nocc, nvir, nlap=nlap,
        tmin_factor=tmin_factor, tmax_factor=tmax_factor,
        quadrature=quadrature, fit_grid_size=fit_grid_size,
        fit_ratio=fit_ratio, fit_tol=fit_tol,
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
        fit_ratio=FIT_RATIO, fit_tol=FIT_TOL):
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
        fit_ratio=fit_ratio, fit_tol=fit_tol,
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
        'fit_tol': getattr(mp, 'fit_tol', FIT_TOL),
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
        | {'quadrature', 'fit_grid_size', 'fit_ratio', 'fit_tol', 'sos'}
    )

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None,
                 nlap=NLAP, c_os=C_OS, tmin_factor=TMIN_FACTOR,
                 tmax_factor=TMAX_FACTOR, quadrature=QUADRATURE,
                 fit_grid_size=FIT_GRID_SIZE, fit_ratio=FIT_RATIO,
                 fit_tol=FIT_TOL, sos=SOS):
        super().__init__(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)
        self.nlap = nlap
        self.c_os = c_os
        self.tmin_factor = tmin_factor
        self.tmax_factor = tmax_factor
        self.quadrature = quadrature
        self.fit_grid_size = fit_grid_size
        self.fit_ratio = fit_ratio
        self.fit_tol = fit_tol
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
            fit_tol=self.fit_tol,
        )


RMP2 = MP2
DFMP2 = MP2


def sos_lt_mp2(mf, auxbasis=None, nlap=NLAP, c_os=C_OS, frozen=0,
               tmin_factor=TMIN_FACTOR, tmax_factor=TMAX_FACTOR,
               quadrature=QUADRATURE, fit_grid_size=FIT_GRID_SIZE,
               fit_ratio=FIT_RATIO, fit_tol=FIT_TOL):
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
        fit_ratio=fit_ratio, fit_tol=fit_tol, sos=True,
    )
    emp2, _ = mymp.kernel(with_t2=False)
    return emp2


del WITH_T2
