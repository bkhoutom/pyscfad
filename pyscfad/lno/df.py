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

"""Local density-fitting integrals and orbital transformations for LNO/DLNO."""

import warnings
import os
import numpy
import jax
from pyscfad import numpy as np
from pyscfad.ops import is_array
from pyscfad import df as df_mod
from pyscfad.ao2mo import _ao2mo
from pyscfad.df import addons as df_addons
from pyscfad.lno import tools as lno_tools
from pyscfad.gto._mole_helper import setup_exp, setup_ctr_coeff
from ._df_direct import (
    _local_direct_nr_e2,
)
from ._df_h5 import (
    LocalLovH5Info,
    _build_local_Lov_h5_impl,
)
from ._df_outcore import (
    _global_pair_indices_for_local_ao,
    _outcore_local_nr_e2_from_global_cderi,
    _outcore_nr_e2,
)


def _local_domain_atmlst(mf, atmlst):
    if atmlst is None or not hasattr(mf, 'with_df') or mf.with_df is None:
        return None
    atmlst = numpy.asarray(atmlst, dtype=numpy.int32).ravel()
    if atmlst.size == 0:
        return None
    return atmlst


def make_local_mol(mol, atmlst):
    fake_mol = lno_tools.fake_mol_by_atom(mol, atmlst)
    if getattr(mol, 'coords', None) is not None:
        fake_mol.coords = np.asarray(mol.atom_coords()[numpy.asarray(atmlst, dtype=numpy.int32)])
    if getattr(mol, 'exp', None) is not None:
        fake_mol.exp = np.asarray(setup_exp(fake_mol)[0])
    else:
        fake_mol.exp = None
    if getattr(mol, 'ctr_coeff', None) is not None:
        fake_mol.ctr_coeff = np.asarray(setup_ctr_coeff(fake_mol)[0])
    else:
        fake_mol.ctr_coeff = None
    return fake_mol


def get_local_df(mf, atmlst):
    atmlst = tuple(map(int, numpy.asarray(atmlst).ravel()))
    cache = getattr(mf.with_df, '_lno_local_df_cache', None)
    if cache is None:
        cache = {}
        mf.with_df._lno_local_df_cache = cache
    if atmlst in cache:
        return cache[atmlst]

    fake_mol = make_local_mol(mf.mol, atmlst)
    # Keep the local auxiliary basis and local AO-pair space unchanged, but
    # follow the parent DF storage policy.  Large EDs can otherwise
    # materialize one dense local CDERI per cached fragment in memory.
    local_df = df_mod.DF(
        fake_mol,
        auxbasis=mf.with_df.auxbasis,
        incore=getattr(mf.with_df, 'incore', True),
    )
    local_df.max_memory = mf.with_df.max_memory
    local_df.build()
    ao_idx = lno_tools.ao_index_by_atom(mf.mol, numpy.asarray(atmlst, dtype=numpy.int32))
    cache[atmlst] = (fake_mol, local_df, ao_idx)
    return cache[atmlst]


def build_local_Lov_h5(
    mf,
    mo_coeff_local,
    nocc: int,
    atmlst,
    path: str | os.PathLike,
) -> LocalLovH5Info:
    """Write local ``Lov`` directly to a contiguous HDF5 dataset.

    This helper has no in-memory backend: the fitted pair-major factors are
    written to ``/lov`` at ``path`` and represented only by metadata on
    return.
    """
    atmlst = numpy.asarray(atmlst, dtype=numpy.int32).ravel()
    ao_idx = lno_tools.ao_index_by_atom(mf.mol, atmlst)
    mo_coeff_local = numpy.asarray(
        jax.device_get(mo_coeff_local), order='F'
    )
    if mo_coeff_local.ndim != 2 or mo_coeff_local.shape[0] != ao_idx.size:
        raise ValueError(
            'mo_coeff_local must have one row for each AO in atmlst'
        )
    nmo = mo_coeff_local.shape[1]
    if not 0 <= nocc <= nmo:
        raise ValueError('nocc must lie between zero and the local MO count')

    fake_mol = make_local_mol(mf.mol, atmlst)
    auxmol = df_addons.make_auxmol(fake_mol, mf.with_df.auxbasis)
    return _build_local_Lov_h5_impl(
        fake_mol,
        auxmol,
        mo_coeff_local,
        (0, nocc, nocc, nmo),
        path,
        mf.with_df.max_memory,
    )


def transform_df_to_mo(mf, mo_coeff, orbs_slice, aosym='s2', mosym='s1', atmlst=None):
    atmlst = _local_domain_atmlst(mf, atmlst)
    if atmlst is not None:
        ao_idx = lno_tools.ao_index_by_atom(mf.mol, atmlst)
        s1e = mf.get_ovlp()
        s21 = s1e[ao_idx]
        s22 = s1e[np.ix_(ao_idx, ao_idx)]
        mo_coeff = lno_tools.project_mo(mo_coeff, s21, s22)
        get_cderi = getattr(mf.with_df, '_get_cderi_source', None)
        cderi = get_cderi() if get_cderi is not None else mf.with_df._cderi
        has_outcore_cderi = (
            hasattr(mf.with_df, '_has_outcore_cderi_placeholder')
            and mf.with_df._has_outcore_cderi_placeholder()
        )
        if has_outcore_cderi:
            if mf.with_df.auxmol is None:
                mf.with_df.auxmol = df_mod.addons.make_auxmol(
                    mf.with_df.mol, mf.with_df.auxbasis
            )
            pair_idx = tuple(
                _global_pair_indices_for_local_ao(ao_idx, mf.mol.nao).tolist()
            )
            return _outcore_local_nr_e2_from_global_cderi(
                mf.with_df.mol, mf.with_df.auxmol, mo_coeff, cderi,
                mf.with_df.max_memory, orbs_slice, aosym, pair_idx
            )

        fake_mol, local_df, _ = get_local_df(mf, atmlst)
        get_cderi = getattr(local_df, '_get_cderi_source', None)
        cderi = get_cderi() if get_cderi is not None else local_df._cderi
        has_outcore_cderi = (
            hasattr(local_df, '_has_outcore_cderi_placeholder')
            and local_df._has_outcore_cderi_placeholder()
        )
        if has_outcore_cderi:
            return _outcore_nr_e2(
                fake_mol, local_df.auxmol, mo_coeff, cderi,
                local_df.max_memory, orbs_slice, aosym
            )
    else:
        get_cderi = getattr(mf.with_df, '_get_cderi_source', None)
        cderi = get_cderi() if get_cderi is not None else mf.with_df._cderi
        has_outcore_cderi = (
            hasattr(mf.with_df, '_has_outcore_cderi_placeholder')
            and mf.with_df._has_outcore_cderi_placeholder()
        )
        if has_outcore_cderi:
            if mf.with_df.auxmol is None:
                mf.with_df.auxmol = df_mod.addons.make_auxmol(
                    mf.with_df.mol, mf.with_df.auxbasis
                )
            return _outcore_nr_e2(
                mf.with_df.mol, mf.with_df.auxmol, mo_coeff, cderi,
                mf.with_df.max_memory, orbs_slice, aosym
            )

    with df_addons.load(cderi, 'j3c') as eri1:
        if not is_array(eri1):
            eri1 = numpy.asarray(eri1)
        return _ao2mo.nr_e2(eri1, mo_coeff, orbs_slice, aosym=aosym, mosym=mosym)


def get_local_Lov(mf, mo_coeff_local, nocc, atmlst,
                  integral_direct=None):
    """Transform a local-domain CDERI using local AO coefficients directly.

    ``get_Lov(..., atmlst=...)`` accepts full-molecule AO coefficients and
    projects them into the requested atom domain.  Domain builders that
    already hold coefficients in that local AO basis should use this helper;
    it avoids zero-padding to the global AO dimension only to project back to
    the same local coefficients.

    If ``integral_direct`` is true, raw three-center integrals are transformed
    to the occupied-virtual basis before applying the auxiliary fitting
    metric.  This avoids constructing the much larger local AO-pair CDERI.
    The default is false for compatibility; callers can enable it globally on
    a DF object with ``with_df._lno_local_lov_integral_direct = True``.
    """
    atmlst = numpy.asarray(atmlst, dtype=numpy.int32).ravel()
    ao_idx = lno_tools.ao_index_by_atom(mf.mol, atmlst)
    mo_coeff_local = np.asarray(mo_coeff_local)
    if mo_coeff_local.ndim != 2 or mo_coeff_local.shape[0] != ao_idx.size:
        raise ValueError(
            'mo_coeff_local must have one row for each AO in atmlst'
        )
    nmo = mo_coeff_local.shape[1]
    if not 0 <= nocc <= nmo:
        raise ValueError('nocc must lie between zero and the local MO count')
    if integral_direct is None:
        integral_direct = bool(getattr(
            mf.with_df, '_lno_local_lov_integral_direct', False
        ))
    if integral_direct:
        fake_mol = make_local_mol(mf.mol, atmlst)
        auxmol = df_addons.make_auxmol(fake_mol, mf.with_df.auxbasis)
        if nocc == 0 or nocc == nmo:
            return np.zeros(
                (auxmol.nao, nocc, nmo - nocc), dtype=mo_coeff_local.dtype
            )
        try:
            lov = _local_direct_nr_e2(
                fake_mol,
                auxmol,
                mo_coeff_local,
                mf.with_df.max_memory,
                (0, nocc, nocc, nmo),
            )
        except ImportError as err:
            # PySCF exposes the shell-direct DF transformation through a
            # private helper whose availability is not guaranteed across the
            # full supported PySCF version range.  Retain a correct (but more
            # memory-intensive) compatibility path rather than making the
            # optimization a hard dependency.
            warnings.warn(
                "integral-direct local Lov is unavailable in this PySCF "
                f"version ({err}); falling back to packed local CDERI",
                RuntimeWarning,
            )
        else:
            return lov.reshape((-1, nocc, nmo - nocc))

    fake_mol, local_df, _ = get_local_df(mf, atmlst)
    get_cderi = getattr(local_df, '_get_cderi_source', None)
    cderi = get_cderi() if get_cderi is not None else local_df._cderi
    ijslice = (0, nocc, nocc, nmo)
    has_outcore_cderi = (
        hasattr(local_df, '_has_outcore_cderi_placeholder')
        and local_df._has_outcore_cderi_placeholder()
    )
    if has_outcore_cderi:
        lov = _outcore_nr_e2(
            fake_mol, local_df.auxmol, mo_coeff_local, cderi,
            local_df.max_memory, ijslice, 's2'
        )
    else:
        with df_addons.load(cderi, 'j3c') as eri1:
            if not is_array(eri1):
                eri1 = numpy.asarray(eri1)
            lov = _ao2mo.nr_e2(
                eri1, mo_coeff_local, ijslice, aosym='s2', mosym='s1'
            )
    return lov.reshape((-1, nocc, nmo - nocc))


def get_Lov(mf, mo_coeff, nocc, atmlst=None):
    assert hasattr(mf, 'with_df')
    nmo = mo_coeff.shape[-1]
    nvir = nmo - nocc
    ijslice = (0, nocc, nocc, nmo)
    Lov = transform_df_to_mo(mf, mo_coeff, ijslice, aosym='s2', mosym='s1', atmlst=atmlst)
    naux = Lov.shape[0]
    Lov = Lov.reshape((naux, nocc, nvir))
    return Lov


