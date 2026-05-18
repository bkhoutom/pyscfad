# Copyright 2023-2026 The PySCFAD Authors
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

'''Impurity MP2 solver.
'''

from functools import reduce
import numpy
import time
from pyscf.lib import logger
from pyscfad import numpy as np
from pyscfad.ao2mo import _ao2mo
from pyscfad.cc import dfccsd
from pyscfad.df import addons as df_addons
from pyscfad.ops import is_array
from pyscfad.lno import lno_base
from pyscfad.lno.ccsd import (
    _ChemistsERIs,
    mp2_fragment_energy,
    get_maskact,
)

class RCCSD(dfccsd.RCCSD):
    def ao2mo(self, mo_coeff=None, fockao=None):
        return _make_df_eris_incore(self, mo_coeff, fockao)

def _make_df_eris_incore(cc, mo_coeff=None, fockao=None):
    eris = _ChemistsERIs()
    eris._common_init_(cc, mo_coeff, fockao)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc

    mo = np.asarray(eris.mo_coeff)
    ijslice = (0, nocc, nocc, nmo)
    atmlst = getattr(cc, '_domain_atmlst', None)
    Lov = lno_base.transform_df_to_mo(
        cc._scf, mo, ijslice, aosym='s2', mosym='s1', atmlst=atmlst
    )

    eris.ovov = np.dot(Lov.T, Lov).reshape(nocc,nvir,nocc,nvir)
    return eris

def impurity_solve(mf, mo_coeff, lo_coeff, eris=None, frozen=None,
                   frag_prescreen=None,
                   verbose_imp=0, profile_info=None):
    log = logger.new_logger(mf)
    maskocc = mf.mo_occ > lno_base.THRESH_OCC
    nocc = numpy.count_nonzero(maskocc)
    nmo = mf.mo_occ.size

    frozen, maskact = get_maskact(frozen, nmo)

    orbfrzocc = mo_coeff[:,~maskact& maskocc]
    orbactocc = mo_coeff[:, maskact& maskocc]
    orbactvir = mo_coeff[:, maskact&~maskocc]
    orbfrzvir = mo_coeff[:,~maskact&~maskocc]
    nfrzocc, nactocc, nactvir, nfrzvir = [orb.shape[1]
                                          for orb in [orbfrzocc,orbactocc,
                                                      orbactvir,orbfrzvir]]
    nlo = lo_coeff.shape[1]
    s1e = eris.s1e
    prjlo = reduce(np.dot, (lo_coeff.T, s1e, orbactocc))

    log.info('    impsol:  %d LOs  %d/%d MOs  %d occ  %d vir',
             nlo, nactocc+nactvir, nmo, nactocc, nactvir)

    # solve impurity problem
    mcc = RCCSD(mf, mo_coeff=mo_coeff, frozen=frozen)
    mcc._domain_atmlst = None if frag_prescreen is None else frag_prescreen.get('extended_primary_domain')
    mcc.e_hf = mf.e_tot  #avoid MP2 recompute e_hf
    total_start = time.perf_counter()
    phase_start = time.perf_counter()
    imp_eris = mcc.ao2mo(fockao=eris.fock)
    phase_times = {'ao2mo_s': time.perf_counter() - phase_start}

    # MP2 fragment energy
    phase_start = time.perf_counter()
    t1, t2 = mcc.init_amps(eris=imp_eris)[1:]
    elcorr_pt2 = mp2_fragment_energy(imp_eris, t2, prjlo)
    phase_times['mp2_s'] = time.perf_counter() - phase_start
    phase_times['ccsd_s'] = 0.0
    phase_times['triples_s'] = 0.0
    phase_times['total_s'] = time.perf_counter() - total_start
    if profile_info is not None:
        profile_info['solver_occ'] = int(nactocc)
        profile_info['solver_vir'] = int(nactvir)
        profile_info['solver_mo'] = int(nactocc + nactvir)
        profile_info['phase_times'] = phase_times

    t1 = t2 = imp_eris = mcc = None
    del log
    return (elcorr_pt2,)

class LNOMP2(lno_base.LNO):
    def __init__(self, mf, thresh=1e-4, frozen=None, **kwargs):
        super().__init__(mf, thresh=thresh, frozen=frozen, **kwargs)
        self.efrag_pt2 = None

    def impurity_solve(self, mf, mo_coeff, lo_coeff, eris=None, frozen=None,
                       frag_prescreen=None, profile_info=None):
        return impurity_solve(mf, mo_coeff, lo_coeff, eris=eris, frozen=frozen,
                              frag_prescreen=frag_prescreen,
                              verbose_imp=self.verbose_imp,
                              profile_info=profile_info)

    def _post_proc(self, frag_res, frag_wghtlist):
        ''' Post processing results returned by `impurity_solve` collected in `frag_res`.
        '''
        efrag_pt2 = 0.0
        for i, res in enumerate(frag_res):
            if res is not None:
                efrag_pt2 += res[0] * frag_wghtlist[i]
        self.efrag_pt2  = efrag_pt2

    @property
    def e_corr(self):
        return self.efrag_pt2
