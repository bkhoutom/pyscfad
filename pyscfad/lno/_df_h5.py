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

"""Pair-major HDF5 local DF production and streamed reverse contractions."""

import os
import shutil
import time
import tempfile
from types import FunctionType, SimpleNamespace
from contextlib import contextmanager
from dataclasses import dataclass, replace
import h5py
import numpy
import jax
import jax.scipy.linalg as jsp_linalg
import scipy.linalg as scipy_linalg
from pyscf import lib as pyscf_lib
from pyscfad import numpy as np
from pyscfad.df import _cderi_vjp
from pyscfad.tools import resource_profile
from ._df_direct import (
    _local_direct_int3c_block_mb,
    _local_direct_mo_coeff_vjp,
    _tree_add,
)

_LOCAL_LOV_H5_Z_AUX_READ_TARGET_MB = 64.0


@dataclass(frozen=True)
class LocalLovH5Info:
    path: str
    naux: int
    nocc: int
    nvir: int
    dtype: str
    lov_dataset: str = "lov"
    lov_bar_dataset: str = "lov_bar"
    z_dataset: str = "z"
    lov_disk_mib: float = 0.0
    hdf5_bytes_written: int = 0
    hdf5_write_seconds: float = 0.0


class _TimedH5DatasetWriter:
    """Transparent dataset writer that accounts only actual HDF5 writes."""

    def __init__(self, dataset, enabled):
        self.dataset = dataset
        self.enabled = bool(enabled)
        self.bytes_written = 0
        self.write_seconds = 0.0

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __setitem__(self, key, value):
        if not self.enabled:
            self.dataset[key] = value
            return
        value = numpy.asarray(value)
        start = time.perf_counter()
        self.dataset[key] = value
        self.write_seconds += time.perf_counter() - start
        self.bytes_written += int(value.nbytes)


class _LocalLovH5Output:
    """Map PySCF's out-of-core ``ovL`` request onto contiguous ``/lov``."""

    def __init__(self, h5file, profile_io):
        self.h5file = h5file
        self.profile_io = bool(profile_io)
        self.dataset = None
        self.writer = None

    def create_dataset(self, name, shape, dtype=None, **unused_options):
        if name != 'ovL':
            raise ValueError(f"Unexpected PySCF output dataset {name!r}")
        if self.dataset is not None:
            raise ValueError("PySCF requested the fitted Lov dataset twice")
        self.dataset = self.h5file.create_dataset(
            'lov', shape=shape, dtype=dtype, chunks=None, compression=None
        )
        self.writer = _TimedH5DatasetWriter(
            self.dataset, self.profile_io
        )
        return self.writer


class _PyscfLibScratchProxy:
    """Delegate PySCF lib calls while making H5TmpFile's directory explicit."""

    def __init__(self, scratch_dir):
        self.scratch_dir = os.fspath(scratch_dir)
        self.temporary_files = []

    def __getattr__(self, name):
        return getattr(pyscf_lib, name)

    def H5TmpFile(self, *args, **kwargs):
        kwargs.setdefault('dir', self.scratch_dir)
        h5tmp = pyscf_lib.H5TmpFile(*args, **kwargs)
        self.temporary_files.append(h5tmp)
        return h5tmp

    def close_temporary_files(self):
        for h5tmp in reversed(self.temporary_files):
            h5tmp.close()
        self.temporary_files.clear()


def _run_local_direct_h5_producer(
    producer, holder, coeff_occ, coeff_vir, max_memory, output, log
):
    """Run a PySCF producer with a call-local scratch-file factory.

    PySCF's ``H5TmpFile`` captures its default directory when ``pyscf.lib`` is
    imported.  Cloning this private producer with one substituted global keeps
    the override local to this call, avoiding process-global monkeypatching.
    """
    scratch_lib = _PyscfLibScratchProxy(pyscf_lib.param.TMPDIR)
    if isinstance(producer, FunctionType):
        producer_globals = dict(producer.__globals__)
        producer_globals['lib'] = scratch_lib
        call_producer = FunctionType(
            producer.__code__,
            producer_globals,
            producer.__name__,
            producer.__defaults__,
            producer.__closure__,
        )
        call_producer.__kwdefaults__ = producer.__kwdefaults__
    else:
        call_producer = producer
    try:
        return call_producer(
            holder,
            coeff_occ,
            coeff_vir,
            max_memory,
            h5obj=output,
            log=log,
        )
    finally:
        scratch_lib.close_temporary_files()


def _local_direct_h5_effective_max_memory(max_memory):
    """Cap raw integral buffers when the fitted Lov itself is on disk."""
    current_mb = float(pyscf_lib.current_memory()[0])
    requested_mb = current_mb + _local_direct_int3c_block_mb() / 0.7
    return max(current_mb, min(float(max_memory), requested_mb))


def _build_local_Lov_h5_impl(
    fake_mol,
    auxmol,
    mo_coeff,
    orbs_slice,
    path,
    max_memory,
    profile_io=None,
):
    """Write fitted local ``Lov`` directly to uncompressed HDF5 scratch.

    The producer streams raw three-center blocks and never returns a complete
    in-memory ``Lov`` array.  ``path`` is owned by the fragment workspace.
    """
    from pyscf.mp.dfmp2 import _init_mp_df_eris_direct

    if profile_io is None:
        profile_io = resource_profile.enabled()
    profile_io = bool(profile_io)
    path = os.fspath(path)
    mo_coeff = numpy.asarray(jax.device_get(mo_coeff), order='F')
    k0, k1, l0, l1 = map(int, orbs_slice)
    if not (0 <= k0 <= k1 <= mo_coeff.shape[1]):
        raise ValueError('Invalid first-orbital slice for direct local AO2MO')
    if not (0 <= l0 <= l1 <= mo_coeff.shape[1]):
        raise ValueError('Invalid second-orbital slice for direct local AO2MO')
    nocc = k1 - k0
    nvir = l1 - l0
    lov_bytes = (
        int(auxmol.nao)
        * int(nocc)
        * int(nvir)
        * mo_coeff.dtype.itemsize
    )
    required_bytes = int(3.25 * lov_bytes) + 1024**3
    destination_dir = os.path.dirname(os.path.abspath(path)) or os.curdir
    free_bytes = int(shutil.disk_usage(destination_dir).free)
    if free_bytes < required_bytes:
        raise OSError(
            f"Insufficient scratch space for {path}: free bytes={free_bytes}, "
            f"required bytes={required_bytes}"
        )

    fd, staging_path = tempfile.mkstemp(
        prefix=f'.{os.path.basename(path)}.',
        suffix='.tmp',
        dir=destination_dir,
    )
    os.close(fd)
    hdf5_bytes_written = 0
    hdf5_write_seconds = 0.0
    lov_disk_mib = 0.0
    try:
        with h5py.File(staging_path, 'w') as h5file:
            if nocc == 0 or nvir == 0:
                lov = h5file.create_dataset(
                    'lov',
                    shape=(0, int(auxmol.nao)),
                    dtype=mo_coeff.dtype,
                    chunks=None,
                    compression=None,
                )
            else:
                output = _LocalLovH5Output(h5file, profile_io)
                holder = SimpleNamespace(mol=fake_mol, auxmol=auxmol)
                effective_memory = _local_direct_h5_effective_max_memory(
                    max_memory
                )
                _run_local_direct_h5_producer(
                    _init_mp_df_eris_direct,
                    holder,
                    mo_coeff[:, k0:k1],
                    mo_coeff[:, l0:l1],
                    effective_memory,
                    output,
                    pyscf_lib.logger.new_logger(fake_mol),
                )
                if output.dataset is None:
                    raise RuntimeError(
                        "PySCF did not create the fitted Lov dataset"
                    )
                lov = output.dataset
                hdf5_bytes_written = output.writer.bytes_written
                hdf5_write_seconds = output.writer.write_seconds
            naux = int(lov.shape[1])
            dtype = str(lov.dtype)
            if profile_io:
                flush_start = time.perf_counter()
                h5file.flush()
                hdf5_write_seconds += time.perf_counter() - flush_start
                lov_disk_mib = (
                    float(lov.id.get_storage_size()) / 1024.0**2
                )
        os.replace(staging_path, path)
    except BaseException:
        try:
            os.unlink(staging_path)
        except FileNotFoundError:
            pass
        raise

    return LocalLovH5Info(
        path=path,
        naux=naux,
        nocc=int(nocc),
        nvir=int(nvir),
        dtype=dtype,
        lov_disk_mib=lov_disk_mib,
        hdf5_bytes_written=hdf5_bytes_written,
        hdf5_write_seconds=hdf5_write_seconds,
    )


def _local_lov_h5_pair_tile_size(
    naux: int,
    npair: int,
    dtype,
    target_mb: float = 256.0,
) -> int:
    """Choose a pair tile accounting for Lov, Lov-bar, and z storage."""
    if not isinstance(naux, (int, numpy.integer)) or int(naux) <= 0:
        raise ValueError('naux must be a positive integer')
    if not isinstance(npair, (int, numpy.integer)) or int(npair) < 0:
        raise ValueError('npair must be a nonnegative integer')
    try:
        dtype = numpy.dtype(dtype)
    except TypeError as err:
        raise ValueError('dtype must be a fixed-width numeric dtype') from err
    if dtype.hasobject or dtype.itemsize <= 0:
        raise ValueError('dtype must be a fixed-width numeric dtype')
    if not numpy.issubdtype(dtype, numpy.number):
        raise ValueError('dtype must be numeric')
    try:
        target_mb = float(target_mb)
    except (TypeError, ValueError) as err:
        raise ValueError('target_mb must be positive and finite') from err
    if not numpy.isfinite(target_mb) or target_mb <= 0:
        raise ValueError('target_mb must be positive and finite')
    if int(npair) == 0:
        return 1
    bytes_per_pair = 3 * int(naux) * dtype.itemsize
    tile_size = int(target_mb * 1024.0**2 // bytes_per_pair)
    return max(1, min(int(npair), tile_size))


def _local_lov_h5_reverse_dtype(low_dtype, lov_dtype, lov_bar_dtype):
    """Return the widest dtype resident in each three-tile reverse step."""
    return numpy.result_type(low_dtype, lov_dtype, lov_bar_dtype)


def _local_lov_h5_io_profile():
    return {
        'hdf5_bytes_read': 0,
        'hdf5_bytes_written': 0,
        'hdf5_read_seconds': 0.0,
        'hdf5_write_seconds': 0.0,
    }


def _local_lov_h5_timed_read(dataset, key, profile):
    if profile is None:
        return dataset[key]
    start = time.perf_counter()
    value = dataset[key]
    profile['hdf5_read_seconds'] += time.perf_counter() - start
    profile['hdf5_bytes_read'] += int(numpy.asarray(value).nbytes)
    return value


def _local_lov_h5_timed_write(dataset, key, value, profile):
    if profile is None:
        dataset[key] = value
        return
    value = numpy.asarray(value)
    start = time.perf_counter()
    dataset[key] = value
    profile['hdf5_write_seconds'] += time.perf_counter() - start
    profile['hdf5_bytes_written'] += int(value.nbytes)


@contextmanager
def _local_lov_h5_reverse_profile_scope(
    profile_start, h5_path, io_profile
):
    details = None
    if profile_start is not None:
        details = {
            'status': 'failed',
            'fragment_index': -1,
            'lov_h5_path_basename': os.path.basename(h5_path),
            'lov_disk_mib': 0.0,
            'lov_bar_disk_mib': 0.0,
            'z_disk_mib': 0.0,
        }
    try:
        yield details
    except BaseException:
        raise
    else:
        if details is not None:
            details['status'] = 'ok'
    finally:
        if details is not None:
            resource_profile.finish(
                'lno.local_direct_nr_e2_h5_bwd',
                profile_start,
                **details,
                **io_profile,
            )


def _local_lov_h5_z_chunks(naux, npair, dtype, pair_tile_size):
    """Choose internal 2-D z chunks near 8 MiB for both access axes."""
    if npair == 0:
        return None
    dtype = numpy.dtype(dtype)
    target_items = max(1, int(8.0 * 1024.0**2 // dtype.itemsize))
    pair_chunk = int(
        numpy.sqrt(target_items * float(npair) / max(float(naux), 1.0))
    )
    pair_chunk = max(1, min(int(npair), int(pair_tile_size), pair_chunk))
    aux_chunk = max(1, min(int(naux), target_items // pair_chunk))
    return aux_chunk, pair_chunk


def _local_lov_h5_z_aux_read_rows(naux, npair, dtype):
    """Bound HDF5 z reads while keeping nontrivial reads strictly partial."""
    naux = int(naux)
    npair = int(npair)
    if naux <= 0 or npair < 0:
        raise ValueError('z dimensions must be nonnegative with positive naux')
    if npair == 0 or naux == 1:
        return naux
    row_bytes = npair * numpy.dtype(dtype).itemsize
    target_bytes = _LOCAL_LOV_H5_Z_AUX_READ_TARGET_MB * 1024.0**2
    rows = max(1, int(target_bytes // max(row_bytes, 1)))
    return min(rows, naux - 1)


def _local_direct_nr_e2_h5_bwd(
    mol,
    auxmol,
    mo_coeff,
    orbs_slice,
    h5_path: str,
):
    """Reverse pair-major ``Lov``/``Lov_bar`` without materializing ``z``.

    The auxiliary-major ``/z`` workspace is formed in pair tiles and consumed
    in bounded auxiliary ranges by the coefficient and coordinate pullbacks.
    """
    if mol.exp is not None or mol.ctr_coeff is not None or mol.r0 is not None:
        raise NotImplementedError(
            'Integral-direct local Lov currently supports coordinate and '
            'MO-coefficient derivatives, but not AO basis-parameter derivatives.'
        )
    if (
        auxmol.exp is not None
        or auxmol.ctr_coeff is not None
        or auxmol.r0 is not None
    ):
        raise NotImplementedError(
            'Integral-direct local Lov currently supports coordinate and '
            'MO-coefficient derivatives, but not auxiliary basis-parameter '
            'derivatives.'
        )

    mo_coeff = np.asarray(mo_coeff)
    if mo_coeff.ndim != 2 or mo_coeff.shape[0] != mol.nao:
        raise ValueError('mo_coeff shape does not match the molecular AO basis')
    k0, k1, l0, l1 = map(int, orbs_slice)
    if not (0 <= k0 <= k1 <= mo_coeff.shape[1]):
        raise ValueError('Invalid first-orbital slice for direct local AO2MO')
    if not (0 <= l0 <= l1 <= mo_coeff.shape[1]):
        raise ValueError('Invalid second-orbital slice for direct local AO2MO')
    naux = int(auxmol.nao)
    npair = (k1 - k0) * (l1 - l0)
    profile_start = resource_profile.start()
    io_profile = (
        _local_lov_h5_io_profile() if profile_start is not None else None
    )
    h5_path = os.fspath(h5_path)

    def metric_cholesky(auxmol_):
        return jsp_linalg.cholesky(
            auxmol_.intor(auxmol_._add_suffix('int2c2e'), hermi=1),
            lower=True,
        )

    with _local_lov_h5_reverse_profile_scope(
        profile_start, h5_path, io_profile
    ) as profile_details, h5py.File(h5_path, 'r+') as h5file:
        if profile_details is not None:
            profile_details['fragment_index'] = int(
                h5file.attrs.get('pyscfad_fragment_index', -1)
            )
        if 'lov' not in h5file or 'lov_bar' not in h5file:
            raise ValueError('HDF5 reverse requires /lov and /lov_bar datasets')
        lov = h5file['lov']
        lov_bar = h5file['lov_bar']
        expected_shape = (npair, naux)
        if lov.ndim != 2 or tuple(lov.shape) != expected_shape:
            raise ValueError(
                f'/lov shape {lov.shape} does not match {expected_shape}'
            )
        if lov_bar.ndim != 2 or tuple(lov_bar.shape) != expected_shape:
            raise ValueError(
                f'/lov_bar shape {lov_bar.shape} does not match {expected_shape}'
            )
        required_dtype = numpy.dtype(numpy.float64)
        reverse_dtypes = {
            '/lov': numpy.dtype(lov.dtype),
            '/lov_bar': numpy.dtype(lov_bar.dtype),
            'mo_coeff': numpy.dtype(mo_coeff.dtype),
        }
        incompatible = {
            name: dtype for name, dtype in reverse_dtypes.items()
            if dtype != required_dtype
        }
        if incompatible:
            details = ', '.join(
                f'{name}={dtype}' for name, dtype in incompatible.items()
            )
            raise ValueError(
                'Direct local HDF5 Lov reverse requires compatible real '
                f'float64 inputs; got {details}'
            )

        low_ad, metric_pullback = jax.vjp(metric_cholesky, auxmol)
        low = numpy.asarray(jax.device_get(low_ad))
        if (
            low.shape != (naux, naux)
            or not numpy.all(numpy.isfinite(low))
            or numpy.any(numpy.diag(low) <= 0)
        ):
            raise NotImplementedError(
                'Integral-direct local Lov VJP does not yet support the '
                'linear-dependent auxiliary-metric eigenvalue fallback.'
            )

        z_dtype = _local_lov_h5_reverse_dtype(
            low.dtype, lov.dtype, lov_bar.dtype
        )
        z_aux_block_max_rows = _local_lov_h5_z_aux_read_rows(
            naux, npair, z_dtype
        )
        pair_tile_size = _local_lov_h5_pair_tile_size(
            naux, npair, z_dtype
        )
        if profile_details is not None:
            profile_details.update(
                lov_disk_mib=(
                    float(lov.id.get_storage_size()) / 1024.0**2
                ),
                lov_bar_disk_mib=(
                    float(lov_bar.id.get_storage_size()) / 1024.0**2
                ),
                pair_tile_size=pair_tile_size,
                z_aux_block_max_rows=z_aux_block_max_rows,
            )
        z_created = False
        try:
            if 'z' in h5file:
                del h5file['z']
            write_start = (
                time.perf_counter() if io_profile is not None else None
            )
            z = h5file.create_dataset(
                'z',
                shape=(naux, npair),
                dtype=z_dtype,
                chunks=_local_lov_h5_z_chunks(
                    naux, npair, z_dtype, pair_tile_size
                ),
                compression=None,
            )
            if io_profile is not None:
                io_profile['hdf5_write_seconds'] += (
                    time.perf_counter() - write_start
                )
            z_created = True
            low_bar = numpy.zeros(
                (naux, naux),
                dtype=numpy.result_type(low.dtype, lov.dtype, z_dtype),
            )
            for pair0 in range(0, npair, pair_tile_size):
                pair1 = min(pair0 + pair_tile_size, npair)
                lov_tile = numpy.asarray(_local_lov_h5_timed_read(
                    lov,
                    (slice(pair0, pair1), slice(None)),
                    io_profile,
                ))
                lov_bar_tile = numpy.asarray(_local_lov_h5_timed_read(
                    lov_bar,
                    (slice(pair0, pair1), slice(None)),
                    io_profile,
                ))
                z_tile = scipy_linalg.solve_triangular(
                    low.T,
                    lov_bar_tile.T,
                    lower=False,
                    check_finite=False,
                )
                low_bar -= z_tile @ lov_tile
                _local_lov_h5_timed_write(
                    z,
                    (slice(None), slice(pair0, pair1)),
                    z_tile,
                    io_profile,
                )
            flush_start = (
                time.perf_counter() if io_profile is not None else None
            )
            h5file.flush()
            if io_profile is not None:
                io_profile['hdf5_write_seconds'] += (
                    time.perf_counter() - flush_start
                )

            def read_z_aux_block(p0, p1):
                return _local_lov_h5_timed_read(
                    z,
                    (slice(p0, p1), slice(None)),
                    io_profile,
                )

            mo_coeff_bar = _local_direct_mo_coeff_vjp(
                mol,
                auxmol,
                mo_coeff,
                read_z_aux_block,
                orbs_slice,
                z_aux_block_max_rows=z_aux_block_max_rows,
            )
            mol_bar, auxmol_bar = \
                _cderi_vjp._int3c_mo_deriv_coords_vjp(
                    mol,
                    auxmol,
                    mo_coeff,
                    read_z_aux_block,
                    orbs_slice,
                    int3c=mol._add_suffix('int3c2e'),
                    aosym='s2ij',
                    block_memory_mb=_local_direct_int3c_block_mb(),
                    z_aux_block_max_rows=z_aux_block_max_rows,
                )

            _cderi_vjp._zero_strict_upper_inplace(low_bar)
            aux_metric_bar = metric_pullback(np.asarray(low_bar))[0]
            auxmol_bar = _tree_add(auxmol_bar, aux_metric_bar)
            if profile_details is not None:
                profile_details['z_disk_mib'] = (
                    float(z.id.get_storage_size()) / 1024.0**2
                )
            return mol_bar, auxmol_bar, mo_coeff_bar
        except BaseException:
            if z_created and 'z' in h5file:
                del h5file['z']
                h5file.flush()
            raise


