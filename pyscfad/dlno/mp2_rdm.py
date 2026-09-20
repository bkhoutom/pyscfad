"""Target-conditioned MP2 selection densities and their blocked reverse pass.

The target conditions the first occupied line; the selected extended domain
supplies the occupied environment. These are unrelaxed oo/vv selection
densities, not the occupied-virtual response density in lno.mp2_rdm."""

from __future__ import annotations

from pyscfad.lno import _df_h5 as lno_df_h5
from contextvars import ContextVar
from functools import partial
import os
import time
from typing import NamedTuple
import warnings

import h5py
import jax
from jax.interpreters import ad as jax_ad
import numpy

from pyscfad import numpy as np
from pyscfad.tools import resource_profile


__all__ = [
    "MP2Density",
    "target_conditioned_mp2_density_from_amplitudes",
    "strong_domain_mp2_density_from_lov",
]


_H5_DENSITY_IO_PROFILE = ContextVar(
    "pyscfad_iao_lis_h5_density_io_profile", default=None
)


def _new_h5_io_profile():
    return {
        "hdf5_bytes_read": 0,
        "hdf5_bytes_written": 0,
        "hdf5_read_seconds": 0.0,
        "hdf5_write_seconds": 0.0,
    }


def _record_h5_density_io(**details):
    profile = _H5_DENSITY_IO_PROFILE.get()
    if profile is None:
        return
    for key, value in details.items():
        if key in (
            "hdf5_bytes_read",
            "hdf5_bytes_written",
            "hdf5_read_seconds",
            "hdf5_write_seconds",
        ):
            profile[key] += value
        else:
            profile[key] = value


def _h5_dataset_disk_mib(dataset):
    """Return allocated HDF5 storage without reading the dataset."""
    return float(dataset.id.get_storage_size()) / 1024.0**2


def _timed_h5_read(dataset, key, profile):
    if profile is None:
        return dataset[key]
    start = time.perf_counter()
    value = dataset[key]
    elapsed = time.perf_counter() - start
    if profile is not None:
        profile["hdf5_read_seconds"] += elapsed
        profile["hdf5_bytes_read"] += int(numpy.asarray(value).nbytes)
    return value


def _timed_h5_write(dataset, key, value, profile):
    if profile is None:
        dataset[key] = value
        return
    value = numpy.asarray(value)
    start = time.perf_counter()
    dataset[key] = value
    elapsed = time.perf_counter() - start
    if profile is not None:
        profile["hdf5_write_seconds"] += elapsed
        profile["hdf5_bytes_written"] += int(value.nbytes)


def _finish_h5_density_reverse_profile(
    profile_start,
    h5_path,
    io_profile,
    *,
    status,
    fragment_index,
    lov_disk_mib,
    lov_bar_disk_mib,
    block_nvir,
    block_count,
):
    if profile_start is None:
        return
    resource_profile.finish(
        "iao_lis.strong_domain_mp2_density_h5_lov_bwd",
        profile_start,
        status=status,
        fragment_index=fragment_index,
        lov_h5_path_basename=os.path.basename(h5_path),
        lov_disk_mib=lov_disk_mib,
        lov_bar_disk_mib=lov_bar_disk_mib,
        z_disk_mib=0.0,
        block_nvir=block_nvir,
        block_count=block_count,
        **io_profile,
    )


class MP2Density(NamedTuple):
    """Target-conditioned strong-ED MP2 density blocks."""

    occupied: object
    virtual: object


def _hermitize(array):
    return 0.5 * (array + array.T.conj())


def _validate_amplitudes(amplitudes, target_projection):
    amplitudes = np.asarray(amplitudes)
    target_projection = np.asarray(target_projection)
    if amplitudes.ndim != 4:
        raise ValueError("amplitudes must have shape (nocc,nocc,nvir,nvir)")
    nocc, nocc1, nvir, nvir1 = amplitudes.shape
    if nocc1 != nocc or nvir1 != nvir:
        raise ValueError("amplitudes must have shape (nocc,nocc,nvir,nvir)")
    if target_projection.ndim != 2 or target_projection.shape[1] != nocc:
        raise ValueError("target_projection must have shape (ntarget,nocc)")
    return amplitudes, target_projection


def _density_from_target_amplitudes(target_amplitudes):
    """Spin-adapted IE density from ``U[I,r,a,b]``.

    The four contractions are kept in the same form as
    :func:`pyscfad.lno._checkpointed.make_mp2_rdm1_ie`.  Wrapping each target
    slice in ``jax.checkpoint`` makes reverse mode recompute its rank-three
    contractions instead of retaining one set per target IAO.
    """

    target_amplitudes = np.asarray(target_amplitudes)
    if target_amplitudes.ndim != 4:
        raise ValueError(
            "target_amplitudes must have shape (ntarget,nocc,nvir,nvir)"
        )
    ntarget, nocc, nvir, nvir1 = target_amplitudes.shape
    if nvir1 != nvir:
        raise ValueError("the two virtual dimensions must be equal")

    dtype = target_amplitudes.dtype
    dmoo0 = np.zeros((nocc, nocc), dtype=dtype)
    dmvv0 = np.zeros((nvir, nvir), dtype=dtype)

    @jax.checkpoint
    def target_density(amplitude):
        # The established helper uses axes (a,j,b) for one internal target.
        t = amplitude.transpose(1, 0, 2)
        tc = t.conj()

        dmvv = np.dot(t.reshape(nvir, -1), tc.reshape(nvir, -1).T)
        dmvv = dmvv - 0.5 * np.einsum("ajc,cjb->ab", t, tc)
        dmvv = dmvv + np.dot(
            t.reshape(-1, nvir).T,
            tc.reshape(-1, nvir),
        )
        dmvv = dmvv - 0.5 * np.einsum("cja,bjc->ab", t, tc)

        dmoo = np.einsum("aib,ajb->ij", t, tc)
        dmoo = dmoo - 0.5 * np.einsum("aib,bja->ij", t, tc)
        dmoo = dmoo + np.einsum("bia,bja->ij", t, tc)
        dmoo = dmoo - 0.5 * np.einsum("bia,ajb->ij", t, tc)
        return dmoo, dmvv

    def scan_body(carry, amplitude):
        dmoo, dmvv = carry
        term_oo, term_vv = target_density(amplitude)
        return (dmoo + term_oo, dmvv + term_vv), None

    if ntarget == 0:
        return MP2Density(dmoo0, dmvv0)
    (dmoo, dmvv), _ = jax.lax.scan(
        scan_body, (dmoo0, dmvv0), target_amplitudes
    )
    return MP2Density(_hermitize(dmoo), _hermitize(dmvv))


def _density_from_target_amplitude_block(a_block, b_block):
    """Return one contracted-virtual-block contribution to Doo and Dvv."""

    a_block = np.asarray(a_block)
    b_block = np.asarray(b_block)
    if a_block.ndim != 4 or b_block.ndim != 4:
        raise ValueError("amplitude blocks must have rank-4 shapes")
    if a_block.shape != b_block.shape:
        raise ValueError("amplitude blocks must have identical shapes")

    # Put the two full virtual indices first and last, respectively.  The
    # supplied B block has already exchanged those virtual indices.
    a = a_block.transpose(0, 2, 1, 3)
    b = b_block.transpose(0, 2, 1, 3)
    b_first = b.swapaxes(1, 3)

    dmvv = np.dot(
        a.transpose(1, 0, 2, 3).reshape(a.shape[1], -1),
        a.conj().transpose(1, 0, 2, 3).reshape(a.shape[1], -1).T,
    )
    dmvv = dmvv - 0.5 * np.einsum(
        "iajc,icjb->ab", a, b_first.conj()
    )
    dmvv = dmvv + np.dot(
        b_first.reshape(-1, b.shape[1]).T,
        b_first.conj().reshape(-1, b.shape[1]),
    )
    dmvv = dmvv - 0.5 * np.einsum(
        "icja,ibjc->ab", b_first, a.conj()
    )

    dmoo = np.einsum("ixpc,ixqc->pq", a, a.conj())
    dmoo = dmoo - 0.5 * np.einsum(
        "ixpc,icqx->pq", a, b_first.conj()
    )
    dmoo = dmoo + np.einsum(
        "icpx,icqx->pq", b_first, b_first.conj()
    )
    dmoo = dmoo - 0.5 * np.einsum(
        "icpx,ixqc->pq", b_first, a.conj()
    )
    return MP2Density(dmoo, dmvv)


@jax.jit
def _density_from_target_amplitude_block_real_pullback(
    a_block, b_block, density_bar
):
    """Block adjoint for real amplitudes and symmetric Doo/Dvv cotangents.

    For either occupied or virtual matricization, the density is
    A A.T + B B.T - (A B.T + B A.T)/2.  Its adjoint is H(2A-B),
    H(2B-A), where H applies both output cotangents.  This uses four
    contractions and never reconstructs the unused density outputs.
    The HDF5 caller validates real inputs and hermitizes the cotangents.
    """

    def apply_density_bar(block):
        return np.einsum(
            "rs,Isac->Irac", density_bar.occupied, block, optimize=True
        ) + np.einsum(
            "ad,Irdc->Irac", density_bar.virtual, block, optimize=True
        )

    return (
        apply_density_bar(2 * a_block - b_block),
        apply_density_bar(2 * b_block - a_block),
    )


def target_conditioned_mp2_density_from_amplitudes(
    amplitudes,
    target_projection,
):
    """Return the target-conditioned spin-adapted MP2 selection density.

    Parameters
    ----------
    amplitudes
        Semicanonical strong-ED MP2 amplitudes ``T[p,r,a,b]``.  Denominators
        must be applied before the target factor because a general IAO factor
        mixes nondegenerate occupied ED orbitals.
    target_projection
        ``X[I,p]``, the target IAO block projected into the ED occupied frame.

    Returns
    -------
    :class:`MP2Density`
        Occupied and virtual unrelaxed selection-density blocks in the ED
        semicanonical bases.
    """

    amplitudes, target_projection = _validate_amplitudes(
        amplitudes, target_projection
    )
    target_amplitudes = np.einsum(
        "Ip,prab->Irab",
        target_projection.conj(),
        amplitudes,
        optimize=True,
    )
    return _density_from_target_amplitudes(target_amplitudes)


def _automatic_workspace_mb(mf_max_memory_mb: float) -> float:
    """Choose the default MP2-density workspace target from SCF memory."""
    candidate = max(256.0, 0.10 * float(mf_max_memory_mb))
    candidate = min(candidate, 8192.0)
    return max(1.0, min(candidate, 0.25 * float(mf_max_memory_mb)))


def _resolve_mp2_density_block_nvir(
    *,
    naux: int,
    nocc: int,
    nvir: int,
    ntarget: int,
    dtype,
    mf_max_memory_mb: float,
    configured_memory_mb: float | None,
    configured_block_nvir: int | None,
) -> tuple[int, str, float]:
    """Return ``(block_nvir, mode, workspace_target_mib)`` for MP2 density.

    With no override, the workspace target and virtual width are selected
    automatically from ``mf_max_memory_mb`` and the tensor dimensions.
    ``configured_block_nvir`` is an advanced exact-width override (clamped
    only to the available virtual dimension); ``configured_memory_mb`` is an
    optional workspace-target override.  Neither value is a hard process-RSS
    cap.
    """
    naux = int(naux)
    nocc = int(nocc)
    nvir = int(nvir)
    ntarget = int(ntarget)
    if configured_memory_mb is not None and configured_memory_mb <= 0.0:
        raise ValueError("mp2_block_memory_mb must be positive")
    if configured_block_nvir is not None and (
        not isinstance(configured_block_nvir, (int, numpy.integer))
        or isinstance(configured_block_nvir, bool)
        or configured_block_nvir <= 0
    ):
        raise ValueError("mp2_block_nvir must be a positive integer")

    automatic_target_mb = _automatic_workspace_mb(mf_max_memory_mb)
    if configured_block_nvir is not None:
        workspace_target_mb = automatic_target_mb
        mode = "manual_width"
    elif configured_memory_mb is not None:
        workspace_target_mb = float(configured_memory_mb)
        mode = "manual_budget"
    else:
        workspace_target_mb = automatic_target_mb
        mode = "auto"

    itemsize = numpy.dtype(dtype).itemsize
    fixed_elements = (
        2 * naux * nvir
        + nocc * nocc
        + nvir * nvir
    )
    per_c_elements = (
        4 * ntarget * nocc * nvir
        + 4 * nocc * nvir
        + 2 * naux * nocc
    )
    available_bytes = workspace_target_mb * 1024.0**2 - itemsize * fixed_elements
    modeled_block_nvir = int(
        available_bytes // max(itemsize * per_c_elements, 1)
    )
    modeled_block_nvir = max(1, min(max(nvir, 1), modeled_block_nvir))

    if configured_block_nvir is not None:
        block_nvir = min(int(configured_block_nvir), max(nvir, 1))
        if block_nvir > modeled_block_nvir:
            warnings.warn(
                "manual MP2 density block width exceeds the modeled "
                "workspace target; honoring the requested width",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        block_nvir = modeled_block_nvir
        if available_bytes < itemsize * per_c_elements:
            warnings.warn(
                "MP2 density workspace target is below the conservative "
                "one-virtual-block model; using block_nvir=1",
                RuntimeWarning,
                stacklevel=2,
            )
    return block_nvir, mode, workspace_target_mb


def _mp2_density_virtual_block_size(lov, ntarget, max_memory_mb):
    itemsize = numpy.dtype(lov.dtype).itemsize
    nocc = lov.shape[1]
    nvir = lov.shape[2]
    bytes_per_c = itemsize * nocc * nvir * max(2 * ntarget + 6, 1)
    budget = float(max_memory_mb) * 1024.0**2
    block_nvir = int(budget // max(bytes_per_c, 1))
    return max(1, min(nvir, block_nvir))


def strong_domain_mp2_density_from_lov(
    lov,
    occupied_energy,
    virtual_energy,
    target_projection,
    *,
    block_nvir=None,
    max_memory_mb=256.0,
):
    """Build the target-conditioned ED density directly from DF factors.

    The contracted virtual index is processed in fixed-size blocks.  Peak
    target-amplitude storage is ``O(ntarget*nocc*nvir*block_nvir)``.  Nested
    checkpointed scans reconstruct each temporary block during reverse mode
    instead of retaining the complete MP2 or target-amplitude tensor.
    """

    lov = np.asarray(lov)
    occupied_energy = np.asarray(occupied_energy)
    virtual_energy = np.asarray(virtual_energy)
    target_projection = np.asarray(target_projection)
    if lov.ndim != 3:
        raise ValueError("lov must have shape (naux,nocc,nvir)")
    naux, nocc, nvir = lov.shape
    if occupied_energy.shape != (nocc,):
        raise ValueError("occupied_energy must have shape (nocc,)")
    if virtual_energy.shape != (nvir,):
        raise ValueError("virtual_energy must have shape (nvir,)")
    if target_projection.ndim != 2 or target_projection.shape[1] != nocc:
        raise ValueError("target_projection must have shape (ntarget,nocc)")

    if max_memory_mb <= 0:
        raise ValueError("max_memory_mb must be positive")
    ntarget = target_projection.shape[0]
    if block_nvir is None:
        block_nvir = _mp2_density_virtual_block_size(
            lov, ntarget, max_memory_mb
        )
    elif (
        not isinstance(block_nvir, (int, numpy.integer))
        or isinstance(block_nvir, bool)
        or block_nvir <= 0
    ):
        raise ValueError("block_nvir must be a positive integer")
    block_nvir = min(int(block_nvir), nvir) if nvir else 1

    dmoo0 = np.zeros((nocc, nocc), dtype=lov.dtype)
    dmvv0 = np.zeros((nvir, nvir), dtype=lov.dtype)
    if ntarget == 0 or nocc == 0 or nvir == 0:
        return MP2Density(dmoo0, dmvv0)

    eia = occupied_energy[:, None] - virtual_energy[None, :]
    nblock = (nvir + block_nvir - 1) // block_nvir
    block_ids = np.arange(nblock, dtype=numpy.int32)
    block_offsets = np.arange(block_nvir, dtype=numpy.int32)
    occupied_ids = np.arange(nocc, dtype=numpy.int32)

    @jax.checkpoint
    def virtual_block_body(carry, block_id):
        dmoo, dmvv = carry
        indices = block_id * block_nvir + block_offsets
        valid = indices < nvir
        safe_indices = np.minimum(indices, nvir - 1)
        lov_c = np.take(lov, safe_indices, axis=2)
        eia_c = np.take(eia, safe_indices, axis=1)
        lov_c = np.where(valid[None, None, :], lov_c, 0)
        a0 = np.zeros((ntarget, nocc, nvir, block_nvir), dtype=lov.dtype)

        @jax.checkpoint
        def occupied_slice(p):
            la = lov[:, p, :]
            eia_p = eia[p]
            target_column = target_projection[:, p]
            integrals = np.einsum(
                "La,Lrc->rac", la, lov_c, optimize=True
            )
            denominator = eia_p[None, :, None] + eia_c[:, None, :]
            denominator = np.where(
                valid[None, None, :],
                denominator,
                np.ones((), dtype=denominator.dtype),
            )
            amplitudes = integrals / denominator
            amplitudes = np.where(
                valid[None, None, :], amplitudes, 0
            )
            a_increment = np.einsum(
                "I,rac->Irac", target_column.conj(), amplitudes,
                optimize=True,
            )
            b_row = np.einsum(
                "Ir,rac->Iac", target_projection.conj(), amplitudes,
                optimize=True,
            )
            return a_increment, b_row

        def occupied_scan_body(a_block, p):
            a_increment, b_row = occupied_slice(p)
            return a_block + a_increment, b_row

        a_block, b_rows = jax.lax.scan(
            jax.checkpoint(occupied_scan_body), a0, occupied_ids
        )
        b_block = b_rows.transpose(1, 0, 2, 3)
        contribution = _density_from_target_amplitude_block(a_block, b_block)
        return (
            dmoo + contribution.occupied,
            dmvv + contribution.virtual,
        ), None

    (dmoo, dmvv), _ = jax.lax.scan(
        jax.checkpoint(virtual_block_body), (dmoo0, dmvv0), block_ids
    )
    return MP2Density(_hermitize(dmoo), _hermitize(dmvv))


def _target_amplitude_block_from_lov_occupied_slice(
    lov_p,
    lov_c,
    eia_p,
    eia_c,
    target_column,
    target_projection,
):
    """Return one occupied slice of the blocked A/B amplitudes."""

    integrals = np.einsum("La,Lrc->rac", lov_p, lov_c, optimize=True)
    denominator = eia_p[None, :, None] + eia_c[:, None, :]
    amplitudes = integrals / denominator
    a_increment = np.einsum(
        "I,rac->Irac", target_column.conj(), amplitudes, optimize=True
    )
    b_row = np.einsum(
        "Ir,rac->Iac", target_projection.conj(), amplitudes, optimize=True
    )
    return a_increment, b_row


@jax.jit
def _target_amplitude_block_from_lov_occupied_slice_pullback(
    lov_p,
    lov_c,
    eia_p,
    eia_c,
    target_column,
    target_projection,
    a_bar,
    b_row_bar,
):
    """Compile only the slice pullback, discarding unused primal outputs.

    Amplitudes are still recomputed for the energy/projection response.
    Keeping this wrapper outside the occupied loop reuses the executable
    without retaining the amplitude blocks or Lov slices between calls.
    """

    _, pullback = jax.vjp(
        _target_amplitude_block_from_lov_occupied_slice,
        lov_p, lov_c, eia_p, eia_c, target_column, target_projection,
    )
    return pullback((a_bar, b_row_bar))


def _strong_domain_mp2_density_h5_primal_impl(
    h5_path: str,
    occupied_energy,
    virtual_energy,
    target_projection,
    *,
    naux: int,
    nocc: int,
    nvir: int,
    block_nvir: int,
    profile,
) -> MP2Density:
    """Evaluate Doo/Dvv while reading only bounded pair-major Lov slices.

    HDF5 access is host-orchestrated and synchronous; no complete ``Lov`` or
    target-amplitude tensor is materialized.
    """

    dimensions = {"naux": naux, "nocc": nocc, "nvir": nvir}
    for name, value in dimensions.items():
        if (
            not isinstance(value, (int, numpy.integer))
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(f"{name} must be a nonnegative integer")
    if (
        not isinstance(block_nvir, (int, numpy.integer))
        or isinstance(block_nvir, bool)
        or block_nvir <= 0
    ):
        raise ValueError("block_nvir must be a positive integer")
    naux, nocc, nvir = int(naux), int(nocc), int(nvir)
    block_nvir = min(int(block_nvir), nvir) if nvir else 1

    occupied_energy = np.asarray(occupied_energy)
    virtual_energy = np.asarray(virtual_energy)
    target_projection = np.asarray(target_projection)
    if occupied_energy.shape != (nocc,):
        raise ValueError("occupied_energy must have shape (nocc,)")
    if virtual_energy.shape != (nvir,):
        raise ValueError("virtual_energy must have shape (nvir,)")
    if target_projection.ndim != 2 or target_projection.shape[1] != nocc:
        raise ValueError("target_projection must have shape (ntarget,nocc)")

    ntarget = int(target_projection.shape[0])
    nblock = (nvir + block_nvir - 1) // block_nvir
    profiling = profile is not None
    hdf5_read_s = 0.0
    mp2_kernel_s = 0.0
    bytes_read = 0

    with h5py.File(h5_path, "r") as h5file:
        lov_h5 = h5file["lov"]
        expected_shape = (nocc * nvir, naux)
        if lov_h5.shape != expected_shape:
            raise ValueError(
                f"/lov must have shape {expected_shape}, got {lov_h5.shape}"
            )
        dtype = lov_h5.dtype
        lov_disk_mib = _h5_dataset_disk_mib(lov_h5) if profiling else 0.0
        itemsize = int(dtype.itemsize)
        dmoo = np.zeros((nocc, nocc), dtype=dtype)
        dmvv = np.zeros((nvir, nvir), dtype=dtype)

        if ntarget != 0 and nocc != 0 and nvir != 0:
            eia = occupied_energy[:, None] - virtual_energy[None, :]
            for c0 in range(0, nvir, block_nvir):
                c1 = min(c0 + block_nvir, nvir)
                width = c1 - c0
                lov_c_host = numpy.empty((naux, nocc, width), dtype=dtype)
                for r in range(nocc):
                    read_start = time.perf_counter() if profiling else None
                    pair_rows = lov_h5[
                        r * nvir + c0:r * nvir + c1, :
                    ]
                    if profiling:
                        hdf5_read_s += time.perf_counter() - read_start
                    lov_c_host[:, r, :] = numpy.asarray(pair_rows).T
                    del pair_rows
                if profiling:
                    bytes_read += nocc * naux * width * itemsize

                lov_c = np.asarray(lov_c_host)
                del lov_c_host
                eia_c = eia[:, c0:c1]
                a_block = np.zeros(
                    (ntarget, nocc, nvir, width), dtype=dtype
                )
                b_rows = []
                for p in range(nocc):
                    read_start = time.perf_counter() if profiling else None
                    lov_p_host = lov_h5[
                        p * nvir:(p + 1) * nvir, :
                    ]
                    if profiling:
                        hdf5_read_s += time.perf_counter() - read_start
                    lov_p_host = numpy.asarray(lov_p_host)
                    if profiling:
                        bytes_read += naux * nvir * itemsize

                    kernel_start = (
                        time.perf_counter() if profiling else None
                    )
                    a_increment, b_row = (
                        _target_amplitude_block_from_lov_occupied_slice(
                            np.asarray(lov_p_host.T),
                            lov_c,
                            eia[p],
                            eia_c,
                            target_projection[:, p],
                            target_projection,
                        )
                    )
                    a_block = a_block + a_increment
                    jax.block_until_ready((a_block, b_row))
                    if profiling:
                        mp2_kernel_s += time.perf_counter() - kernel_start
                    b_rows.append(b_row)
                    del lov_p_host, a_increment

                kernel_start = time.perf_counter() if profiling else None
                b_block = np.stack(b_rows, axis=1)
                contribution = _density_from_target_amplitude_block(
                    a_block, b_block
                )
                dmoo = dmoo + contribution.occupied
                dmvv = dmvv + contribution.virtual
                jax.block_until_ready((dmoo, dmvv))
                if profiling:
                    mp2_kernel_s += time.perf_counter() - kernel_start

    kernel_start = time.perf_counter() if profiling else None
    density = MP2Density(_hermitize(dmoo), _hermitize(dmvv))
    jax.block_until_ready(density)
    if profiling:
        mp2_kernel_s += time.perf_counter() - kernel_start
    if profile is not None:
        _record_h5_density_io(
            lov_disk_mib=lov_disk_mib,
            lov_bar_disk_mib=0.0,
            z_disk_mib=0.0,
            hdf5_bytes_read=bytes_read,
            hdf5_read_seconds=hdf5_read_s,
        )
        resource_profile.finish(
            "iao_lis.strong_domain_mp2_density_h5_primal",
            profile,
            status="ok",
            lov_h5_path_basename=os.path.basename(h5_path),
            lov_disk_mib=lov_disk_mib,
            lov_bar_disk_mib=0.0,
            z_disk_mib=0.0,
            naux=naux,
            nocc=nocc,
            nvir=nvir,
            ntarget=ntarget,
            block_nvir=block_nvir,
            block_count=nblock,
            hdf5_bytes_read=bytes_read,
            hdf5_bytes_written=0,
            hdf5_read_seconds=hdf5_read_s,
            hdf5_write_seconds=0.0,
            mp2_kernel_seconds=mp2_kernel_s,
        )
    return density


def _strong_domain_mp2_density_h5_primal(
    h5_path: str,
    occupied_energy,
    virtual_energy,
    target_projection,
    *,
    naux: int,
    nocc: int,
    nvir: int,
    block_nvir: int,
) -> MP2Density:
    """Profile-safe wrapper for the bounded HDF5 density primal."""
    profile = resource_profile.start()
    try:
        return _strong_domain_mp2_density_h5_primal_impl(
            h5_path,
            occupied_energy,
            virtual_energy,
            target_projection,
            naux=naux,
            nocc=nocc,
            nvir=nvir,
            block_nvir=block_nvir,
            profile=profile,
        )
    except BaseException:
        if profile is not None:
            resource_profile.finish(
                "iao_lis.strong_domain_mp2_density_h5_primal",
                profile,
                status="failed",
                lov_h5_path_basename=os.path.basename(h5_path),
                hdf5_bytes_read=0,
                hdf5_bytes_written=0,
                hdf5_read_seconds=0.0,
                hdf5_write_seconds=0.0,
            )
        raise


def _validate_strong_domain_mp2_density_h5_inputs(
    local_coeff,
    occupied_energy,
    virtual_energy,
    target_projection,
    nocc,
    block_nvir,
):
    """Validate the real-float64 contract of the fused disk reverse."""

    if (
        not isinstance(nocc, (int, numpy.integer))
        or isinstance(nocc, bool)
        or nocc < 0
    ):
        raise ValueError("nocc must be a nonnegative integer")
    if (
        not isinstance(block_nvir, (int, numpy.integer))
        or isinstance(block_nvir, bool)
        or block_nvir <= 0
    ):
        raise ValueError("block_nvir must be a positive integer")
    if local_coeff.ndim != 2 or nocc > local_coeff.shape[1]:
        raise ValueError("local_coeff and nocc define an invalid orbital split")
    nvir = int(local_coeff.shape[1]) - int(nocc)
    if occupied_energy.shape != (int(nocc),):
        raise ValueError("occupied_energy must have shape (nocc,)")
    if virtual_energy.shape != (nvir,):
        raise ValueError("virtual_energy must have shape (nvir,)")
    if (
        target_projection.ndim != 2
        or target_projection.shape[1] != int(nocc)
    ):
        raise ValueError("target_projection must have shape (ntarget,nocc)")
    arrays = {
        "local_coeff": local_coeff,
        "occupied_energy": occupied_energy,
        "virtual_energy": virtual_energy,
        "target_projection": target_projection,
    }
    incompatible = {
        name: numpy.dtype(value.dtype)
        for name, value in arrays.items()
        if numpy.dtype(value.dtype) != numpy.dtype(numpy.float64)
    }
    if incompatible:
        details = ", ".join(
            f"{name}={dtype}" for name, dtype in incompatible.items()
        )
        raise ValueError(
            "Fused HDF5 MP2 density reverse requires real float64 inputs; "
            f"got {details}"
        )
    return nvir


def _materialize_density_output_bar(bar, shape, dtype):
    """Turn an absent/AD-zero output cotangent into a concrete array."""

    if bar is None or isinstance(bar, jax_ad.Zero):
        return np.zeros(shape, dtype=dtype)
    bar = np.asarray(bar)
    if bar.shape != shape:
        raise ValueError(
            f"density cotangent has shape {bar.shape}, expected {shape}"
        )
    return bar


def _h5_density_hermitized_output_bars(
    density_bar, *, nocc, nvir, dtype
):
    """Apply the adjoint of the final occupied/virtual hermitization."""

    if isinstance(density_bar, jax_ad.Zero):
        occupied_bar = virtual_bar = None
    else:
        occupied_bar = getattr(density_bar, "occupied", None)
        virtual_bar = getattr(density_bar, "virtual", None)
    occupied_bar = _materialize_density_output_bar(
        occupied_bar, (nocc, nocc), dtype
    )
    virtual_bar = _materialize_density_output_bar(
        virtual_bar, (nvir, nvir), dtype
    )
    return MP2Density(
        _hermitize(occupied_bar), _hermitize(virtual_bar)
    )


def _read_h5_lov_virtual_block(
    lov_h5, *, naux, nocc, nvir, c0, c1, io_profile=None
):
    """Read one contracted-virtual block into auxiliary-first layout."""

    width = c1 - c0
    lov_c_host = numpy.empty((naux, nocc, width), dtype=lov_h5.dtype)
    for r in range(nocc):
        lov_c_host[:, r, :] = numpy.asarray(
            _timed_h5_read(
                lov_h5,
                (slice(r * nvir + c0, r * nvir + c1), slice(None)),
                io_profile,
            )
        ).T
    return lov_c_host


def _reconstruct_h5_target_amplitude_block(
    lov_h5,
    lov_c,
    eia,
    target_projection,
    *,
    naux,
    nocc,
    nvir,
    c0,
    c1,
    io_profile=None,
):
    """Replay one bounded A/B block without retaining occupied pullbacks."""

    width = c1 - c0
    ntarget = int(target_projection.shape[0])
    a_block = np.zeros(
        (ntarget, nocc, nvir, width), dtype=lov_h5.dtype
    )
    b_block_host = numpy.empty(
        (ntarget, nocc, nvir, width), dtype=lov_h5.dtype
    )
    eia_c = eia[:, c0:c1]
    for p in range(nocc):
        lov_p_host = numpy.asarray(
            _timed_h5_read(
                lov_h5,
                (slice(p * nvir, (p + 1) * nvir), slice(None)),
                io_profile,
            )
        )
        a_increment, b_row = (
            _target_amplitude_block_from_lov_occupied_slice(
                np.asarray(lov_p_host.T),
                lov_c,
                eia[p],
                eia_c,
                target_projection[:, p],
                target_projection,
            )
        )
        a_block = a_block + a_increment
        jax.block_until_ready((a_block, b_row))
        b_block_host[:, p, :, :] = numpy.asarray(jax.device_get(b_row))
        del lov_p_host, a_increment, b_row
    return a_block, np.asarray(b_block_host)


def _remove_h5_density_derivative_datasets(h5_path):
    """Best-effort removal of derivative-only datasets after a failure."""

    try:
        with h5py.File(os.fspath(h5_path), "r+") as h5file:
            changed = False
            for name in ("lov_bar", "z"):
                if name in h5file:
                    del h5file[name]
                    changed = True
            if changed:
                h5file.flush()
    except (FileNotFoundError, OSError):
        pass


def _strong_domain_mp2_density_h5_lov_bwd(
    h5_path,
    occupied_energy,
    virtual_energy,
    target_projection,
    density_bar,
    *,
    naux,
    nocc,
    nvir,
    block_nvir,
):
    """Write pair-major ``/lov_bar`` and return energy/projection bars."""

    h5_path = os.fspath(h5_path)
    naux, nocc, nvir = int(naux), int(nocc), int(nvir)
    if min(naux, nocc, nvir) < 0:
        raise ValueError("HDF5 Lov dimensions must be nonnegative")
    if (
        not isinstance(block_nvir, (int, numpy.integer))
        or isinstance(block_nvir, bool)
        or block_nvir <= 0
    ):
        raise ValueError("block_nvir must be a positive integer")
    block_nvir = min(int(block_nvir), nvir) if nvir else 1
    occupied_energy = np.asarray(occupied_energy)
    virtual_energy = np.asarray(virtual_energy)
    target_projection = np.asarray(target_projection)
    if occupied_energy.shape != (nocc,):
        raise ValueError("occupied_energy must have shape (nocc,)")
    if virtual_energy.shape != (nvir,):
        raise ValueError("virtual_energy must have shape (nvir,)")
    if target_projection.ndim != 2 or target_projection.shape[1] != nocc:
        raise ValueError("target_projection must have shape (ntarget,nocc)")

    occupied_bar_host = numpy.zeros_like(
        numpy.asarray(jax.device_get(occupied_energy))
    )
    virtual_bar_host = numpy.zeros_like(
        numpy.asarray(jax.device_get(virtual_energy))
    )
    target_bar_host = numpy.zeros_like(
        numpy.asarray(jax.device_get(target_projection))
    )
    npair = nocc * nvir
    profile_start = resource_profile.start()
    io_profile = (
        _new_h5_io_profile() if profile_start is not None else None
    )
    lov_disk_mib = lov_bar_disk_mib = 0.0
    fragment_index = -1
    block_count = (nvir + block_nvir - 1) // block_nvir
    try:
        with h5py.File(h5_path, "r+") as h5file:
            if io_profile is not None:
                fragment_index = int(
                    h5file.attrs.get("pyscfad_fragment_index", -1)
                )
            if "lov" not in h5file:
                raise ValueError("HDF5 density reverse requires /lov")
            lov_h5 = h5file["lov"]
            expected_shape = (npair, naux)
            if lov_h5.ndim != 2 or tuple(lov_h5.shape) != expected_shape:
                raise ValueError(
                    f"/lov must have shape {expected_shape}, got "
                    f"{lov_h5.shape}"
                )
            if numpy.dtype(lov_h5.dtype) != numpy.dtype(numpy.float64):
                raise ValueError(
                    "Fused HDF5 MP2 density reverse requires real float64 "
                    f"/lov; got {lov_h5.dtype}"
                )
            if io_profile is not None:
                lov_disk_mib = _h5_dataset_disk_mib(lov_h5)
            for name in ("lov_bar", "z"):
                if name in h5file:
                    del h5file[name]
            write_start = (
                time.perf_counter() if io_profile is not None else None
            )
            lov_bar_h5 = h5file.create_dataset(
                "lov_bar",
                shape=expected_shape,
                dtype=lov_h5.dtype,
                chunks=None,
                compression=None,
                fillvalue=0.0,
            )
            if io_profile is not None:
                io_profile["hdf5_write_seconds"] += (
                    time.perf_counter() - write_start
                )
            hermitized_bar = _h5_density_hermitized_output_bars(
                density_bar, nocc=nocc, nvir=nvir, dtype=lov_h5.dtype
            )

            ntarget = int(target_projection.shape[0])
            if ntarget != 0 and nocc != 0 and nvir != 0:
                eia = occupied_energy[:, None] - virtual_energy[None, :]
                for c0 in reversed(range(0, nvir, block_nvir)):
                    c1 = min(c0 + block_nvir, nvir)
                    width = c1 - c0
                    lov_c_host = _read_h5_lov_virtual_block(
                        lov_h5,
                        naux=naux,
                        nocc=nocc,
                        nvir=nvir,
                        c0=c0,
                        c1=c1,
                        io_profile=io_profile,
                    )
                    lov_c = np.asarray(lov_c_host)
                    del lov_c_host
                    a_block, b_block = (
                        _reconstruct_h5_target_amplitude_block(
                            lov_h5,
                            lov_c,
                            eia,
                            target_projection,
                            naux=naux,
                            nocc=nocc,
                            nvir=nvir,
                            c0=c0,
                            c1=c1,
                            io_profile=io_profile,
                        )
                    )
                    a_bar, b_bar = (
                        _density_from_target_amplitude_block_real_pullback(
                            a_block, b_block, hermitized_bar
                        )
                    )
                    jax.block_until_ready((a_bar, b_bar))
                    del a_block, b_block

                    lov_c_bar_host = numpy.zeros(
                        (naux, nocc, width), dtype=lov_h5.dtype
                    )
                    eia_c = eia[:, c0:c1]
                    for p in range(nocc):
                        lov_p_host = numpy.asarray(
                            _timed_h5_read(
                                lov_h5,
                                (
                                    slice(p * nvir, (p + 1) * nvir),
                                    slice(None),
                                ),
                                io_profile,
                            )
                        )
                        slice_inputs = (
                            np.asarray(lov_p_host.T),
                            lov_c,
                            eia[p],
                            eia_c,
                            target_projection[:, p],
                            target_projection,
                        )
                        slice_bars = (
                            _target_amplitude_block_from_lov_occupied_slice_pullback(
                                *slice_inputs, a_bar, b_bar[:, p, :, :]
                            )
                        )
                        slice_bars_host = tuple(
                            numpy.asarray(jax.device_get(bar))
                            for bar in slice_bars
                        )
                        jax.block_until_ready(slice_bars)
                        del slice_inputs, slice_bars
                        (
                            lov_p_bar,
                            lov_c_bar,
                            eia_p_bar,
                            eia_c_bar,
                            target_column_bar,
                            target_projection_bar,
                        ) = slice_bars_host

                        pair0, pair1 = p * nvir, (p + 1) * nvir
                        lov_bar_row = numpy.asarray(
                            _timed_h5_read(
                                lov_bar_h5,
                                (slice(pair0, pair1), slice(None)),
                                io_profile,
                            )
                        )
                        lov_bar_row += lov_p_bar.T
                        _timed_h5_write(
                            lov_bar_h5,
                            (slice(pair0, pair1), slice(None)),
                            lov_bar_row,
                            io_profile,
                        )
                        lov_c_bar_host += lov_c_bar
                        occupied_bar_host[p] += numpy.sum(eia_p_bar)
                        virtual_bar_host -= eia_p_bar
                        occupied_bar_host += numpy.sum(eia_c_bar, axis=1)
                        virtual_bar_host[c0:c1] -= numpy.sum(
                            eia_c_bar, axis=0
                        )
                        target_bar_host[:, p] += target_column_bar
                        target_bar_host += target_projection_bar
                        del (
                            lov_p_host,
                            slice_bars_host,
                            lov_p_bar,
                            lov_c_bar,
                            eia_p_bar,
                            eia_c_bar,
                            target_column_bar,
                            target_projection_bar,
                            lov_bar_row,
                        )

                    for r in range(nocc):
                        pair0 = r * nvir + c0
                        pair1 = r * nvir + c1
                        lov_bar_block = numpy.asarray(
                            _timed_h5_read(
                                lov_bar_h5,
                                (slice(pair0, pair1), slice(None)),
                                io_profile,
                            )
                        )
                        lov_bar_block += lov_c_bar_host[:, r, :].T
                        _timed_h5_write(
                            lov_bar_h5,
                            (slice(pair0, pair1), slice(None)),
                            lov_bar_block,
                            io_profile,
                        )
                    del (
                        a_bar,
                        b_bar,
                        lov_c,
                        lov_c_bar_host,
                    )
            flush_start = (
                time.perf_counter() if io_profile is not None else None
            )
            h5file.flush()
            if io_profile is not None:
                io_profile["hdf5_write_seconds"] += (
                    time.perf_counter() - flush_start
                )
                lov_bar_disk_mib = _h5_dataset_disk_mib(lov_bar_h5)
    except BaseException:
        try:
            _remove_h5_density_derivative_datasets(h5_path)
        finally:
            _finish_h5_density_reverse_profile(
                profile_start,
                h5_path,
                io_profile,
                status="failed",
                fragment_index=fragment_index,
                lov_disk_mib=lov_disk_mib,
                lov_bar_disk_mib=lov_bar_disk_mib,
                block_nvir=block_nvir,
                block_count=block_count,
            )
        raise

    _finish_h5_density_reverse_profile(
        profile_start,
        h5_path,
        io_profile,
        status="ok",
        fragment_index=fragment_index,
        lov_disk_mib=lov_disk_mib,
        lov_bar_disk_mib=lov_bar_disk_mib,
        block_nvir=block_nvir,
        block_count=block_count,
    )
    return (
        np.asarray(occupied_bar_host),
        np.asarray(virtual_bar_host),
        np.asarray(target_bar_host),
    )


def _strong_domain_mp2_density_h5_impl(
    fake_mol,
    auxmol,
    local_coeff,
    occupied_energy,
    virtual_energy,
    target_projection,
    nocc,
    h5_path,
    max_memory,
    block_nvir,
):
    """Build disk Lov and evaluate its blocked density eagerly."""

    h5_path = os.fspath(h5_path)
    max_memory = float(max_memory)
    local_coeff = np.asarray(local_coeff)
    occupied_energy = np.asarray(occupied_energy)
    virtual_energy = np.asarray(virtual_energy)
    target_projection = np.asarray(target_projection)
    nvir = _validate_strong_domain_mp2_density_h5_inputs(
        local_coeff,
        occupied_energy,
        virtual_energy,
        target_projection,
        nocc,
        block_nvir,
    )
    nocc = int(nocc)
    block_nvir = int(block_nvir)
    if local_coeff.shape[0] != fake_mol.nao:
        raise ValueError("local_coeff rows must match the local AO basis")
    info = lno_df_h5._build_local_Lov_h5_impl(
        fake_mol,
        auxmol,
        local_coeff,
        (0, nocc, nocc, local_coeff.shape[1]),
        h5_path,
        max_memory,
        profile_io=(_H5_DENSITY_IO_PROFILE.get() is not None),
    )
    if (info.naux, info.nocc, info.nvir) != (auxmol.nao, nocc, nvir):
        raise RuntimeError("direct HDF5 Lov metadata is inconsistent")
    _record_h5_density_io(
        lov_disk_mib=info.lov_disk_mib,
        lov_bar_disk_mib=0.0,
        z_disk_mib=0.0,
        hdf5_bytes_written=info.hdf5_bytes_written,
        hdf5_write_seconds=info.hdf5_write_seconds,
    )
    return _strong_domain_mp2_density_h5_primal(
        h5_path,
        occupied_energy,
        virtual_energy,
        target_projection,
        naux=info.naux,
        nocc=info.nocc,
        nvir=info.nvir,
        block_nvir=block_nvir,
    )


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9))
def _strong_domain_mp2_density_h5(
    fake_mol,
    auxmol,
    local_coeff,
    occupied_energy,
    virtual_energy,
    target_projection,
    nocc,
    h5_path,
    max_memory,
    block_nvir,
):
    """Fused eager local-Lov construction and disk-backed MP2 density."""

    return _strong_domain_mp2_density_h5_impl(
        fake_mol,
        auxmol,
        local_coeff,
        occupied_energy,
        virtual_energy,
        target_projection,
        nocc,
        h5_path,
        max_memory,
        block_nvir,
    )


def _strong_domain_mp2_density_h5_fwd(
    fake_mol,
    auxmol,
    local_coeff,
    occupied_energy,
    virtual_energy,
    target_projection,
    nocc,
    h5_path,
    max_memory,
    block_nvir,
):
    density = _strong_domain_mp2_density_h5(
        fake_mol,
        auxmol,
        local_coeff,
        occupied_energy,
        virtual_energy,
        target_projection,
        nocc,
        h5_path,
        max_memory,
        block_nvir,
    )
    residual = (
        fake_mol,
        auxmol,
        local_coeff,
        occupied_energy,
        virtual_energy,
        target_projection,
    )
    return density, residual


def _strong_domain_mp2_density_h5_bwd(
    nocc,
    h5_path,
    max_memory,
    block_nvir,
    residual,
    density_bar,
):
    del max_memory
    (
        fake_mol,
        auxmol,
        local_coeff,
        occupied_energy,
        virtual_energy,
        target_projection,
    ) = residual
    nocc = int(nocc)
    nvir = int(local_coeff.shape[1]) - nocc
    orbs_slice = (0, nocc, nocc, local_coeff.shape[1])
    try:
        energy_projection_bars = (
            _strong_domain_mp2_density_h5_lov_bwd(
                h5_path,
                occupied_energy,
                virtual_energy,
                target_projection,
                density_bar,
                naux=auxmol.nao,
                nocc=nocc,
                nvir=nvir,
                block_nvir=block_nvir,
            )
        )
        fake_mol_bar, auxmol_bar, local_coeff_bar = (
            lno_df_h5._local_direct_nr_e2_h5_bwd(
                fake_mol,
                auxmol,
                local_coeff,
                orbs_slice,
                h5_path,
            )
        )
    except BaseException:
        _remove_h5_density_derivative_datasets(h5_path)
        raise
    return (
        fake_mol_bar,
        auxmol_bar,
        local_coeff_bar,
        *energy_projection_bars,
    )


_strong_domain_mp2_density_h5.defvjp(
    _strong_domain_mp2_density_h5_fwd,
    _strong_domain_mp2_density_h5_bwd,
)
