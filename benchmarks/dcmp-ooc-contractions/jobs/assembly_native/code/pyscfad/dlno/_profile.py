"""Instrumentation around DLNO density construction; no selection mathematics."""
import os
import h5py
import numpy
from pyscfad.lno._df_direct import _local_direct_int3c_block_mb
from pyscfad.tools import resource_profile
from .mp2_rdm import _H5_DENSITY_IO_PROFILE, _new_h5_io_profile


def density_call(function, arguments, *, fragment_index, coeff,
                 naux, nocc, nvir, ntarget, block_nvir, block_mode,
                 workspace_target_mb, h5_path):
    profile_density = resource_profile.start()
    h5_io_profile = (
        _new_h5_io_profile() if profile_density is not None else None
    )
    profile_token = _H5_DENSITY_IO_PROFILE.set(
        h5_io_profile
    )
    try:
        try:
            density = function(*arguments)
        finally:
            _H5_DENSITY_IO_PROFILE.reset(profile_token)
    except BaseException:
        if profile_density is not None:
            resource_profile.finish(
                "iao_lis.strong_domain_mp2_density",
                profile_density,
                status="failed",
                fragment_index=fragment_index,
                lov_h5_path_basename=os.path.basename(h5_path),
                **h5_io_profile,
            )
        raise
    if profile_density is not None:
        try:
            if os.path.isfile(h5_path):
                with h5py.File(h5_path, "r+") as h5file:
                    h5file.attrs["pyscfad_fragment_index"] = fragment_index
        except BaseException:
            resource_profile.finish(
                "iao_lis.strong_domain_mp2_density",
                profile_density,
                status="failed",
                fragment_index=fragment_index,
                lov_h5_path_basename=os.path.basename(h5_path),
                **h5_io_profile,
            )
            raise
        itemsize = numpy.dtype(coeff.dtype).itemsize
        lov_mib = naux * nocc * nvir * itemsize / 1024.0**2
        resource_profile.finish(
            "iao_lis.strong_domain_mp2_density",
            profile_density,
            status="ok",
            fragment_index=fragment_index,
            coeff_shape=tuple(coeff.shape),
            lov_shape=(naux, nocc, nvir),
            naux=naux,
            nocc=nocc,
            nvir=nvir,
            ntarget=ntarget,
            lov_mib=lov_mib,
            block_nvir=block_nvir,
            block_count=(nvir + block_nvir - 1) // block_nvir,
            block_mode=block_mode,
            workspace_target_mib=workspace_target_mb,
            full_target_amplitudes_mib=(
                ntarget * nocc * nvir * nvir * itemsize / 1024.0**2
            ),
            block_target_amplitudes_mib=(
                2 * ntarget * nocc * nvir * block_nvir * itemsize / 1024.0**2
            ),
            estimated_block_workspace_mib=(
                itemsize
                * (
                    2 * naux * nvir
                    + nocc * nocc
                    + nvir * nvir
                    + block_nvir
                    * (
                        4 * ntarget * nocc * nvir
                        + 4 * nocc * nvir
                        + 2 * naux * nocc
                    )
                )
                / 1024.0**2
            ),
            occupied_density_shape=tuple(density.occupied.shape),
            virtual_density_shape=tuple(density.virtual.shape),
            density_mib=resource_profile.estimated_array_mib(
                density.occupied, density.virtual
            ),
            lov_h5_path_basename=os.path.basename(h5_path),
            lov_disk_mib=h5_io_profile.get("lov_disk_mib", lov_mib),
            lov_bar_disk_mib=h5_io_profile.get("lov_bar_disk_mib", 0.0),
            z_disk_mib=h5_io_profile.get("z_disk_mib", 0.0),
            hdf5_bytes_read=h5_io_profile["hdf5_bytes_read"],
            hdf5_bytes_written=h5_io_profile["hdf5_bytes_written"],
            hdf5_read_seconds=h5_io_profile["hdf5_read_seconds"],
            hdf5_write_seconds=h5_io_profile["hdf5_write_seconds"],
            local_direct_block_mb=_local_direct_int3c_block_mb(),
        )
    return density
