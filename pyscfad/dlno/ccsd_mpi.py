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

from pyscfad.lno import lno_base_mpi as lno_base
from pyscfad.dlno import ccsd as dlno_ccsd


class DLNOCCSD(lno_base.LNO, dlno_ccsd.DLNOCCSD):
    """MPI variant of :class:`pyscfad.dlno.ccsd.DLNOCCSD`.

    Mirrors the structure of :class:`pyscfad.lno.ccsd_mpi.LNOCCSD` — a
    plain mixin of the MPI-flavored ``lno_base`` with the DLNO-aware
    single-rank CCSD subclass.  All DLNO behavior (the
    ``use_dlno_prescreen``/``dlno_prescreen_data`` defaults and the
    constructor accepting ``dlno_prescreen_data=...``) is inherited from
    :class:`pyscfad.dlno.ccsd.DLNOCCSD`.
    """
    pass
