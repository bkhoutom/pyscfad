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

from pyscfad.lno.ccsd import LNOCCSD


class DLNOCCSD(LNOCCSD):
    """CCSD with domain-restricted LNO prescreening.

    A thin subclass of :class:`pyscfad.lno.ccsd.LNOCCSD` whose constructor
    accepts ``dlno_prescreen_data`` directly and enables
    ``use_dlno_prescreen`` by default.  All solver logic (``impurity_solve``,
    ``kernel``, energy accessors, ``e_corr_pt2corrected``, ...) is inherited
    unchanged.  Use this class whenever a calculation needs the DLNO
    prescreening; use the parent :class:`LNOCCSD` for vanilla LNO.
    """

    def __init__(self, mf, thresh=1e-4, frozen=None, fock=None, s1e=None,
                 dlno_prescreen_data=None, **kwargs):
        super().__init__(mf, thresh=thresh, frozen=frozen, fock=fock,
                         s1e=s1e, **kwargs)
        self.use_dlno_prescreen = True
        self.dlno_prescreen_data = dlno_prescreen_data
