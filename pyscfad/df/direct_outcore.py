# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
# Portions copyright 2026 The PySCFAD Authors
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

"""Serial DF construction directly into a contiguous final-layout dataset.

Integral generation/whitening follows pyscf.df.outcore.cholesky_eri_b. There
is no on-disk block tensor to assemble: each panel is written to its final
columns. The single new file is renamed into place only after completion.
"""

from contextlib import closing
import math
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
import scipy.linalg
from pyscf import gto, lib
from pyscf.df.addons import make_auxmol
from pyscf.df.outcore import _guess_shell_ranges
from pyscf.lib import logger
from pyscfad.gto._mole_helper import setup_exp, setup_ctr_coeff
from pyscfad.lib._threading import dense_blas_threads


def _concrete_mol(mol):
    """Install eager AD coordinates/basis leaves before calling libcint."""
    if not hasattr(mol, 'to_pyscf'):
        return mol
    concrete = mol.to_pyscf().copy()
    try:
        coords = np.asarray(mol.atom_coords(), dtype=float)
        concrete.set_geom_(coords, unit='Bohr')
        for name, setup in (('exp', setup_exp), ('ctr_coeff', setup_ctr_coeff)):
            value = getattr(mol, name, None)
            if value is not None:
                _, _, indices = setup(concrete)
                concrete._env[indices] = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError('direct DF construction requires concrete molecules; '
                        'prebuild eagerly and attach_outcore_cderi before tracing') from error
    return concrete


def _write_panel(dataset, panel, col0, tile_bytes):
    """Bound the C-order conversion needed by h5py for strided panels."""
    rows, cols = panel.shape
    tile_cols = min(cols, max(1, tile_bytes // panel.dtype.itemsize))
    tile_rows = max(1, tile_bytes // (panel.dtype.itemsize * tile_cols))
    for r0, r1 in lib.prange(0, rows, tile_rows):
        for c0, c1 in lib.prange(0, cols, tile_cols):
            tile = np.ascontiguousarray(panel[r0:r1, c0:c1])
            dataset[r0:r1, col0+c0:col0+c1] = tile
            del tile


def _workspace_plan(mol, naux, aosym, max_memory):
    """Reserve metric, five live panel equivalents, and a bounded write tile.

    Two input buffers coexist with the previous prefetched result and the next
    whitening operation's layout/solve copies. Twenty percent is left for
    libcint, BLAS and HDF5 overhead; this is not an OS-enforced RSS limit.
    """
    budget = int(max_memory * 1e6)
    usable = int(.8 * budget)
    metric_bytes = 8 * naux * naux
    ao_loc = mol.ao_loc_nr().astype(np.int64)
    pair_offsets = (ao_loc * mol.nao_nr() if aosym == 's1'
                    else ao_loc * (ao_loc+1) // 2)
    min_columns = int(np.diff(pair_offsets).max())
    # Make the staging tile shrink with the allowance, rather than imposing
    # a fixed minimum that can exceed a small user's workspace.
    tile_bytes = max(8, min(64 * 1024**2, int(.08 * budget)))
    minimum = max(3 * metric_bytes,
                  metric_bytes + 5 * min_columns * naux * 8 + tile_bytes)
    if usable < minimum:
        # Include the proportional tile in the lower-bound calculation.
        minimum_mb = max(3 * metric_bytes / .8,
                         (metric_bytes + 5 * min_columns * naux * 8 + 8) / .72) / 1e6
        minimum_mb = math.ceil(minimum_mb * 1000) / 1000
        raise MemoryError(f'direct DF workspace {max_memory:g} MB is too small '
                          f'for the metric and one shell panel; provide at least '
                          f'{minimum_mb:.3f} MB (rounded up)')
    columns = min(int(pair_offsets[-1]),
                  (usable - metric_bytes - tile_bytes) // (5 * naux * 8))
    shranges = _guess_shell_ranges(mol, columns, aosym)
    largest = max(x[2] for x in shranges)
    planned_peak = max(3 * metric_bytes,
                       metric_bytes + 5 * largest * naux * 8 + tile_bytes)
    if planned_peak > usable:
        raise MemoryError('direct DF shell partition exceeds the DF workspace')
    return shranges, tile_bytes, planned_peak


def cholesky_eri(mol, erifile, auxbasis='weigend+etb', dataname='j3c',
                 int3c='int3c2e', aosym='s2ij', int2c='int2c2e', comp=1,
                 max_memory=2000, auxmol=None, verbose=logger.NOTE):
    """Build packed DF factors with one new tensor on disk, without repacking.

    ``erifile`` is a standalone filename (or named temporary file), replaced
    atomically on success. Unrelated datasets in an existing file are not
    preserved. While replacing an existing file, that old file also occupies
    disk until the rename. Python exceptions remove the unfinished new file;
    SIGKILL can leave a hidden ``.<filename>.direct-*.h5`` file for cleanup.

    ``max_memory`` is a workspace allowance in decimal MB, not a process limit.
    Planning includes the metric, prefetched integral panels, whitening copies
    and HDF5 conversion tiles (at most 64 MiB or 8% of the allowance), reserving
    20% for native-library overhead. An allowance too small for the metric or
    an indivisible shell panel raises MemoryError before integral generation.

    This eager builder requires a full-rank positive-definite auxiliary metric
    and preserves the Cholesky gauge used by the existing DF response. It
    supports real scalar integrals and s1/s2ij storage, without MPI. Prebuild
    before JAX tracing, then use ``DF.attach_outcore_cderi`` for differentiation.
    """
    if aosym not in ('s1', 's2ij') or comp != 1:
        raise NotImplementedError('direct DF supports scalar s1/s2ij integrals only')
    if not np.isfinite(max_memory) or max_memory <= 0:
        raise ValueError('direct DF max_memory must be finite and positive')
    filename = erifile if isinstance(erifile, (str, os.PathLike)) else erifile.name
    path = Path(filename).absolute()
    mol = _concrete_mol(mol)
    auxmol = make_auxmol(mol, auxbasis) if auxmol is None else _concrete_mol(auxmol)
    if mol.cart != auxmol.cart:
        raise NotImplementedError('direct DF requires matching orbital/auxiliary cart flags')
    log = logger.new_logger(mol, verbose)
    started = logger.process_clock(), logger.perf_counter()
    shranges, tile_bytes, planned_peak = _workspace_plan(
        mol, auxmol.nao_nr(), aosym, max_memory)
    metric = auxmol.intor(int2c, hermi=1)
    if np.iscomplexobj(metric):
        raise NotImplementedError('direct DF requires real integrals')
    try:
        with dense_blas_threads():
            low = scipy.linalg.cholesky(metric, lower=True)
    except scipy.linalg.LinAlgError as error:
        raise RuntimeError('direct DF requires a full-rank positive-definite '
                           'auxiliary metric; rank-truncated ED factors are not '
                           'supported by this gradient path') from error
    del metric

    int3c = gto.moleintor.ascint3(mol._add_suffix(int3c))
    atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                    auxmol._atm, auxmol._bas, auxmol._env)
    ao_loc = gto.moleintor.make_loc(bas, int3c)
    nao = int(ao_loc[mol.nbas])
    naux = int(ao_loc[-1] - nao)
    nao_pair = nao * nao if aosym == 's1' else nao * (nao+1) // 2
    if low.shape != (naux, naux) or nao_pair == 0:
        raise ValueError('inconsistent or empty DF integral dimensions')
    cintopt = gto.moleintor.make_cintopt(atm, bas, env, int3c)
    bufs1 = np.empty((max(x[2] for x in shranges), naux))
    bufs2 = np.empty_like(bufs1)
    log.info('Direct DF: %.3f GB final tensor, %d AO-pair panels; no repacking',
             naux * nao_pair * 8 / 1e9, len(shranges))
    log.info('Direct DF: planned arrays %.3f MB within %.3f MB workspace '
             '(20%% reserved for native overhead)', planned_peak / 1e6, max_memory)

    def process(sh_range):
        nonlocal bufs1, bufs2
        bufs2, bufs1 = bufs1, bufs2
        bstart, bend, _ = sh_range
        shls = (bstart, bend, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas)
        raw = gto.moleintor.getints3c(int3c, atm, bas, env, shls, comp,
                                    aosym, ao_loc, cintopt, out=bufs1)
        if raw.ndim == 3 and raw.flags.f_contiguous:
            raw = lib.transpose(raw.T, axes=(0, 2, 1)).reshape(naux, -1)
        else:
            raw = raw.reshape((-1, naux)).T
        with dense_blas_threads():
            if raw.flags.c_contiguous:
                trsm, = scipy.linalg.get_blas_funcs(('trsm',), (low, raw))
                return trsm(1., low, raw.T, lower=True, trans_a=1, side=1,
                            overwrite_b=True).T
            return scipy.linalg.solve_triangular(low, raw, lower=True,
                                                 overwrite_b=True, check_finite=False)

    # This file already has the final layout. Renaming it never copies data.
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.direct-', suffix='.h5',
                                     dir=path.parent)
    os.close(fd)
    try:
        with h5py.File(temporary, 'w') as handle:
            dataset = handle.create_dataset(dataname, (naux, nao_pair), dtype='f8')
            handle.attrs['pyscfad_direct_df_complete'] = False
            handle.attrs['decomposition'] = 'CD'
            handle.attrs['ao_pair_panels'] = len(shranges)
            handle.attrs['workspace_mb'] = max_memory
            handle.attrs['planned_array_peak_bytes'] = planned_peak
            handle.attrs['write_tile_bytes'] = tile_bytes
            col0 = 0
            with closing(lib.map_with_prefetch(process, shranges)) as panels:
                for istep, panel in enumerate(panels):
                    width = shranges[istep][2]
                    if panel.shape != (naux, width):
                        raise RuntimeError('direct DF integral panel shape mismatch')
                    _write_panel(dataset, panel, col0, tile_bytes)
                    col0 += width
                    panel = None
                    log.debug('Direct DF panel %d/%d, columns through %d/%d',
                              istep+1, len(shranges), col0, nao_pair)
            if col0 != nao_pair:
                raise RuntimeError('direct DF did not cover every AO pair')
            handle.attrs['pyscfad_direct_df_complete'] = True
            handle.flush()
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    log.timer('direct cholesky_eri', *started)
    return erifile
