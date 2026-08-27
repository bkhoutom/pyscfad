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

import itertools
import tempfile
import numpy
from pyscf import __config__
from pyscf import lib as pyscf_lib
from pyscf.lib import logger
from pyscf.df import df as pyscf_df
from pyscf.df import outcore
from pyscfad import util
from pyscfad.ops import is_array
from pyscfad.df import addons, incore, df_jk

_OUTCORE_CDERI_PLACEHOLDER_SHAPE = (0, 0)
_FAST_EXCHANGE_CACHE_TOKENS = itertools.count()

@util.pytree_node(['mol', 'auxmol', '_cderi'], num_args=1)
class DF(pyscf_df.DF):
    # pylint: disable=redefined-outer-name
    _keys = pyscf_df.DF._keys.union({'incore'})

    def __init__(self, mol, auxbasis=None, incore=True, **kwargs):
        pyscf_df.DF.__init__(self, mol, auxbasis=auxbasis)
        self.incore = incore
        # Unlike Python identity, this token survives JAX pytree
        # flatten/unflatten.  It lets custom-VJP residual DF objects find the
        # orbital data cached by their originating SCF calculation without
        # aliasing a second DF object that happens to share the same CDERI.
        self._fast_exchange_cache_token = next(_FAST_EXCHANGE_CACHE_TOKENS)
        self.__dict__.update(kwargs)

    def _has_outcore_cderi_placeholder(self):
        cderi = getattr(self, '_cderi', None)
        return (
            getattr(self, '_cderi_to_save', None) is not None
            and hasattr(cderi, 'shape')
            and tuple(cderi.shape) == _OUTCORE_CDERI_PLACEHOLDER_SHAPE
        )

    def _get_cderi_source(self):
        if (
            getattr(self, '_prefer_cderi_to_save', False)
            and getattr(self, '_cderi_to_save', None) is not None
        ):
            return self._cderi_to_save
        if self._has_outcore_cderi_placeholder():
            return self._cderi_to_save
        return self._cderi

    def build(self):
        log = logger.new_logger(self)

        self.check_sanity()
        self.dump_flags()

        if self._cderi is not None and self.auxmol is None:
            log.info('Skip DF.build(). Existing _cderi will be used.')
            return self

        # Caller has already wired up the outcore-placeholder pattern
        # (``_cderi_to_save`` points at an existing on-disk j3c file, ``_cderi``
        # is the zero-shape placeholder, ``_prefer_cderi_to_save`` is True).
        # Skip the build entirely — re-running ``outcore.cholesky_eri`` here
        # would overwrite the file and, under JAX tracing, fail because
        # pyscf's outcore routine cannot consume a traced ``mol``.
        if self._has_outcore_cderi_placeholder() and self.auxmol is not None:
            log.info('Skip DF.build(). Outcore CDERI at %s will be used.',
                     self._cderi_to_save)
            return self

        mol = self.mol
        if self.auxmol is None:
            self.auxmol = addons.make_auxmol(self.mol, self.auxbasis)
        auxmol = self.auxmol
        nao = mol.nao
        naux = auxmol.nao
        nao_pair = nao*nao

        max_memory = self.max_memory - pyscf_lib.current_memory()[0]
        int3c = mol._add_suffix('int3c2e')
        int2c = mol._add_suffix('int2c2e')
        need_disk = (
            isinstance(self._cderi_to_save, str)
            or not self.incore
        )
        if not need_disk:
            self._cderi = incore.cholesky_eri(mol, auxmol=auxmol,
                                              int3c=int3c, int2c=int2c,
                                              max_memory=max_memory, verbose=log)
        else:
            if self._cderi_to_save is None:
                # pylint: disable=consider-using-with
                self._cderi_to_save = tempfile.NamedTemporaryFile(dir=pyscf_lib.param.TMPDIR)
            cderi_file = self._cderi_to_save
            cderi_name = cderi_file if isinstance(cderi_file, str) else cderi_file.name
            log.info('_cderi_to_save = %s', cderi_name)
            outcore.cholesky_eri(
                mol,
                cderi_file,
                dataname='j3c',
                int3c=int3c,
                int2c=int2c,
                auxmol=auxmol,
                max_memory=max_memory,
                verbose=log,
            )
            # Keep the actual cderi on disk.  The zero-size leaf preserves the
            # traced DF pytree and signals the get_jk VJP to recompute cderi.
            self._cderi = numpy.zeros(_OUTCORE_CDERI_PLACEHOLDER_SHAPE)
            self._prefer_cderi_to_save = True

        del log
        return self

    def attach_outcore_cderi(self, path):
        """Wire this DF object up to read an existing on-disk j3c file.

        After this call, ``self.build()`` will be a no-op, and ``get_jk`` /
        ``transform_df_to_mo`` will stream cderi blocks from ``path`` via
        their custom_vjp paths.  Use to reuse a cderi file built once (e.g.
        by an eager reference SCF) across many traced gradient evaluations.
        """
        if self.auxmol is None:
            self.auxmol = addons.make_auxmol(self.mol, self.auxbasis)
        self._cderi_to_save = path
        self._cderi = numpy.zeros(_OUTCORE_CDERI_PLACEHOLDER_SHAPE)
        self._prefer_cderi_to_save = True
        return self

    def reset(self, mol=None):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        if mol is not None:
            self.mol = mol
        # NOTE resetting auxmol will lose its tracing,
        # but its contribution should be included in mol
        self.auxmol = None
        self._cderi = None
        if not isinstance(self._cderi_to_save, str) and not self.incore:
            # pylint: disable=consider-using-with
            self._cderi_to_save = tempfile.NamedTemporaryFile(dir=pyscf_lib.param.TMPDIR)
        self._vjopt = None
        self._rsh_df = {}
        return self

    def get_naoaux(self):
        if self._cderi is None:
            self.build()
        with addons.load(self._get_cderi_source(), 'j3c') as feri:
            return feri.shape[0]

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True,
               direct_scf_tol=getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13),
               omega=None):
        if self._cderi is None:
            self.build()
        if omega is None:
            return df_jk.get_jk(self, dm, hermi, with_j, with_k, direct_scf_tol)
        else:
            raise NotImplementedError

    def loop(self, blksize=None, to_numpy=True):
        # NOTE By default (to_numpy=True)
        # we do not trace this function so that it can be used by pyscf
        if self._cderi is None:
            self.build()
        if blksize is None:
            blksize = self.blockdim

        with addons.load(self._get_cderi_source(), 'j3c') as feri:
            if is_array(feri):
                naoaux = feri.shape[0]
                for b0, b1 in self.prange(0, naoaux, blksize):
                    if to_numpy:
                        out = numpy.asarray(feri[b0:b1], order='C')
                    else:
                        out = feri[b0:b1]
                    yield out
            elif hasattr(feri, 'shape'):
                naoaux = feri.shape[0]
                for b0, b1 in self.prange(0, naoaux, blksize):
                    yield numpy.asarray(feri[b0:b1], order='C')
            else:
                raise NotImplementedError

    to_pyscf = util.to_pyscf
