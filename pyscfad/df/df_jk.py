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
from jax.interpreters import ad as jax_ad
from jax.tree_util import tree_flatten
from pyscf.df import df_jk as pyscf_df_jk
from pyscfad import config
from pyscfad import numpy as np
from pyscfad import pytree
from .addons import restore
from ._df_jk_opt import get_jk as get_jk_opt


def _has_jvp_tracer(*xs):
    leaves = []
    for x in xs:
        if x is None:
            continue
        x_leaves, _ = tree_flatten(x)
        leaves.extend(x_leaves)
    return any(_contains_jvp_tracer(x) for x in leaves)


def _contains_jvp_tracer(x):
    if isinstance(x, jax_ad.JVPTracer):
        return True
    for attr in ('primal', 'val'):
        if hasattr(x, attr):
            try:
                if _contains_jvp_tracer(getattr(x, attr)):
                    return True
            except AttributeError:
                pass
    return False


def _has_outcore_cderi(dfobj):
    fn = getattr(dfobj, '_has_outcore_cderi_placeholder', None)
    return bool(fn() if callable(fn) else False)


def get_jk(dfobj, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-13):
    # When CDERI lives on disk (outcore placeholder), always route through the
    # custom_vjp opt path: its forward streams cderi blocks from disk via
    # ``dfobj.loop()`` and its backward computes mol/auxmol cotangents
    # analytically via ``_cderi_vjp``.  ``get_jk_gen`` would try to einsum the
    # ``(0, 0)`` placeholder and die.
    if _has_outcore_cderi(dfobj):
        return get_jk_opt(dfobj, dm, hermi=hermi,
                          with_j=with_j, with_k=with_k,
                          direct_scf_tol=direct_scf_tol)
    if config.moleintor_opt and not _has_jvp_tracer(dfobj, dm):
        return get_jk_opt(dfobj, dm, hermi=hermi,
                          with_j=with_j, with_k=with_k,
                          direct_scf_tol=direct_scf_tol)
    else:
        return get_jk_gen(dfobj, dm, hermi=hermi,
                          with_j=with_j, with_k=with_k,
                          direct_scf_tol=direct_scf_tol)

def get_jk_gen(dfobj, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-13):
    nao = dfobj.mol.nao
    dms = dm.reshape(-1, nao, nao)
    Lpq = restore('s1', dfobj._cderi, nao)

    vj = vk = 0
    if with_j:
        tmp = np.einsum('Lpq,xpq->xL', Lpq, dms)
        vj = np.einsum('Lpq,xL->xpq', Lpq, tmp)
        vj = vj.reshape(dm.shape)
    if with_k:
        tmp = np.einsum('Lij,xjk->xLki', Lpq, dms)
        vk = np.einsum('Lki,xLkj->xij', Lpq, tmp)
        vk = vk.reshape(dm.shape)
    return vj, vk

@wraps(pyscf_df_jk.density_fit)
def density_fit(mf, auxbasis=None, with_df=None, only_dfj=False):
    # pylint: disable=import-outside-toplevel
    from pyscfad import scf
    from .df import DF # pylint: disable=cyclic-import
    assert isinstance(mf, scf.hf.SCF)

    if with_df is None:
        with_df = DF(mf.mol)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis

    if isinstance(mf, _DFHF):
        if mf.with_df is None:
            mf.with_df = with_df
        elif getattr(mf.with_df, 'auxbasis', None) != auxbasis:
            mf = mf.copy()
            mf.with_df = with_df
            mf.only_dfj = only_dfj
        return mf

    _DFHF.__bases__ = (pyscf_df_jk._DFHF, mf.__class__)
    dfmf = _DFHF(mf, with_df, only_dfj)
    return dfmf

class _DFHF(pytree.PytreeNode, pyscf_df_jk._DFHF):
    # mo_coeff, mo_energy, e_tot are stateful attributes ``kernel()`` writes
    # onto the SCF object.  If they sit as aux data (i.e. not in
    # ``_dynamic_attr``), the JAX pytree machinery preserves them verbatim
    # across flatten/unflatten and they carry outer-trace tracer references
    # into any custom_vjp body that re-traces with the SCF as an input.
    # That breaks both the readers (UnexpectedTracerError on attribute
    # reads) and the equality checks JAX uses for treedef matching
    # (``data_for_hash`` equality comparing tracer leaves).
    _dynamic_attr = {'mol', 'with_df', 'mo_coeff', 'mo_energy', 'e_tot'}

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        if dm is None:
            dm = self.make_rdm1()

        if not self.with_df:
            return super().get_jk(mol, dm, hermi, with_j, with_k, omega)

        with_dfk = with_k and not self.only_dfj

        #TODO GHF
        vj, vk = self.with_df.get_jk(dm, hermi, with_j, with_dfk,
                                     self.direct_scf_tol, omega)
        if with_k and not with_dfk:
            vk = super().get_jk(mol, dm, hermi, False, True, omega)[1]
        return vj, vk
