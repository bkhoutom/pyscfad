# Copyright 2026 The PySCFAD Authors
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

"""MPI construction of molecular out-of-core Cholesky CDERI.

The two-centre auxiliary metric is constructed and Cholesky-factorized on
``root``.  Its factor is broadcast once.  Packed AO shell-pair columns are
then divided into contiguous ranges; every rank evaluates all auxiliary
functions for its own columns, applies the common metric factor, and writes a
private HDF5 shard.  Root assembles those shards, in bounded auxiliary-row
blocks, into an ordinary ``j3c`` dataset and atomically installs the result.

Partitioning the final auxiliary rows during construction would be wrong:
the metric whitening transformation mixes the complete auxiliary space.  AO
pair columns are independent before and after that transformation and are the
same work units used by :func:`pyscf.df.outcore.cholesky_eri_b`.

This first implementation deliberately requires a full-rank Cholesky metric.
PySCFAD's molecular CDERI coordinate VJP reconstructs that same Cholesky
gauge; an eigenvalue-truncated metric would require a matching ED pullback.
The build is eager, real, float64, packed ``s2ij``, and collective over the
supplied communicator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback
import warnings

import h5py
from mpi4py import MPI
import numpy
from pyscf import gto, lib
from pyscf.df import addons as pyscf_df_addons
from pyscf.df import outcore
import scipy.linalg

from pyscfad.gto._mole_helper import setup_ctr_coeff, setup_exp


__all__ = [
    "MPICDERIBuildResult",
    "MPICDERIRankManifest",
    "build_cderi",
]


_FORMAT_VERSION = 1


@dataclass(frozen=True)
class MPICDERIRankManifest:
    """One rank's stable work and timing diagnostics."""

    rank: int
    pair_start: int
    pair_stop: int
    block_start: int
    block_stop: int
    block_count: int
    pair_columns: int
    byte_count: int
    integral_seconds: float
    checksum: str


@dataclass(frozen=True)
class MPICDERIBuildResult:
    """Collective result returned identically on every MPI rank."""

    path: str
    decomposition: str
    naux_raw: int
    naux: int
    nao_pair: int
    nblocks: int
    nproc: int
    metric_seconds: float
    assembly_seconds: float
    wall_seconds: float
    manifests: tuple[MPICDERIRankManifest, ...]


def _progress_enabled(progress):
    if progress is None or progress is False:
        return False
    if progress is True or callable(progress):
        return True
    raise TypeError("progress must be a bool, callable, or None")


def _progress_reporter(progress, *, rank, root):
    if not _progress_enabled(progress) or rank != root:
        return None
    if callable(progress):
        return progress

    def report(message):
        print(message, flush=True)

    return report


def _report(reporter, message):
    if reporter is not None:
        try:
            reporter(f"[MPI-DF] {message}")
        except Exception as error:  # A root-only callback must not strand MPI.
            warnings.warn(
                f"MPI CDERI progress callback failed and was ignored: {error!r}",
                RuntimeWarning,
                stacklevel=2,
            )


def _exception_text(stage, rank):
    return (
        f"MPI CDERI {stage} failed on rank {int(rank)}:\n"
        + traceback.format_exc()
    )


def _cleanup_shared_scratch(comm, scratch_path, *, rank, root):
    if rank == root:
        shutil.rmtree(scratch_path, ignore_errors=True)
    comm.Barrier()


def _raise_if_any_rank_failed(
    comm,
    local_error,
    *,
    scratch_path=None,
    rank=None,
    root=None,
):
    errors = comm.allgather(local_error)
    failures = tuple(error for error in errors if error is not None)
    if failures:
        if scratch_path is not None:
            _cleanup_shared_scratch(
                comm, scratch_path, rank=rank, root=root
            )
        raise RuntimeError("\n".join(failures))


def _as_eager_float64(value, label):
    try:
        array = numpy.asarray(value)
    except Exception as error:
        raise TypeError(
            f"MPI CDERI construction is an eager preprocessing step; {label} "
            "must have concrete values and cannot be a JAX tracer"
        ) from error
    if numpy.iscomplexobj(array):
        raise ValueError(f"{label} must contain real values")
    array = numpy.asarray(array, dtype=numpy.float64)
    if not numpy.all(numpy.isfinite(array)):
        raise ValueError(f"{label} must contain finite real values")
    return array


def _concrete_pyscf_mol(mol):
    """Copy a PySCFAD molecule with dynamic leaves installed in ``_env``."""

    coords = _as_eager_float64(mol.atom_coords(unit="Bohr"), "mol.coords")
    expected_coords = (int(mol.natm), 3)
    if coords.shape != expected_coords:
        raise ValueError(
            f"mol.coords has shape {coords.shape}, expected {expected_coords}"
        )

    if hasattr(mol, "to_pyscf"):
        concrete = mol.to_pyscf().copy()
    else:
        concrete = mol.copy()
    concrete.set_geom_(coords, unit="Bohr")

    # PySCFAD normally differentiates at the basis stored in _env. Installing
    # concrete dynamic leaves here also makes this eager helper safe for a
    # caller that explicitly replaced an exponent or coefficient leaf.
    for name, setup in (("exp", setup_exp), ("ctr_coeff", setup_ctr_coeff)):
        value = getattr(mol, name, None)
        if value is None:
            continue
        array = _as_eager_float64(value, f"mol.{name}").reshape(-1)
        _, _, env_indices = setup(concrete)
        if array.shape != env_indices.shape:
            raise ValueError(
                f"mol.{name} has shape {array.shape}, expected "
                f"{env_indices.shape}"
            )
        concrete._env[env_indices] = array
    return concrete


def _array_digest(digest, array):
    array = numpy.ascontiguousarray(numpy.asarray(array))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())


def _array_hexdigest(array):
    digest = hashlib.sha256()
    _array_digest(digest, array)
    return digest.hexdigest()


def _molecule_semantic_arrays(mol):
    exponents = []
    coefficients = []
    for shell in mol._bas:
        nprim = int(shell[gto.NPRIM_OF])
        nctr = int(shell[gto.NCTR_OF])
        exp_start = int(shell[gto.PTR_EXP])
        coeff_start = int(shell[gto.PTR_COEFF])
        exponents.append(mol._env[exp_start:exp_start + nprim])
        coefficients.append(
            mol._env[coeff_start:coeff_start + nprim * nctr]
        )
    exponents = (
        numpy.concatenate(exponents) if exponents else numpy.empty(0)
    )
    coefficients = (
        numpy.concatenate(coefficients) if coefficients else numpy.empty(0)
    )
    # The first five shell fields carry atom, angular momentum, primitive,
    # contraction, and kappa data. The remaining fields are offsets into
    # _env. Hash basis values shell-by-shell because equivalent PySCF builds
    # may store their unique environment sections in a different order.
    return {
        "charges": numpy.asarray(mol.atom_charges()),
        "coords": numpy.asarray(mol.atom_coords(unit="Bohr")),
        "shells": numpy.asarray(mol._bas[:, :5]),
        "exp": numpy.asarray(exponents),
        "coeff": numpy.asarray(coefficients),
    }


def _system_component_digests(mol, auxmol):
    components = {}
    for prefix, molecule in (("mol", mol), ("aux", auxmol)):
        for name, array in _molecule_semantic_arrays(molecule).items():
            components[f"{prefix}_{name}"] = _array_hexdigest(array)
    return components


def _system_digest(mol, auxmol):
    digest = hashlib.sha256()
    digest.update(str(bool(mol.cart)).encode("ascii"))
    digest.update(str(bool(auxmol.cart)).encode("ascii"))
    for molecule in (mol, auxmol):
        for array in _molecule_semantic_arrays(molecule).values():
            _array_digest(digest, array)
    return digest.hexdigest()


def _collective_signature(
    output_path,
    mol,
    auxmol,
    *,
    root,
    int3c,
    int2c,
    aosym,
    dataname,
    max_memory,
    min_blocks_per_rank,
    assembly_block_mb,
    overwrite,
    progress_enabled,
):
    return (
        str(output_path),
        int(root),
        str(int3c),
        str(int2c),
        str(aosym),
        str(dataname),
        float(max_memory),
        int(min_blocks_per_rank),
        float(assembly_block_mb),
        bool(overwrite),
        bool(progress_enabled),
        int(mol.natm),
        int(mol.nao_nr()),
        int(auxmol.nao_nr()),
        _system_digest(mol, auxmol),
    )


def _contiguous_block_partition(shranges, size):
    """Return contiguous block-index bounds balanced by pair-column count."""

    nblock = len(shranges)
    if nblock == 0:
        return tuple((0, 0) for _ in range(size))
    weights = numpy.asarray([item[2] for item in shranges], dtype=numpy.int64)
    cumulative = numpy.cumsum(weights)
    total = int(cumulative[-1])
    active = min(int(size), nblock)
    boundaries = [0]
    for part in range(1, active):
        target = total * part / active
        boundary = int(numpy.searchsorted(cumulative, target, side="right"))
        boundary = max(boundary, boundaries[-1] + 1)
        boundary = min(boundary, nblock - (active - part))
        boundaries.append(boundary)
    boundaries.append(nblock)
    partitions = [
        (boundaries[index], boundaries[index + 1])
        for index in range(active)
    ]
    partitions.extend((nblock, nblock) for _ in range(size - active))
    return tuple(partitions)


def _pair_offsets(shranges):
    offsets = [0]
    for _, _, nrow in shranges:
        offsets.append(offsets[-1] + int(nrow))
    return tuple(offsets)


def _metric_cholesky(auxmol, int2c):
    j2c = numpy.asarray(auxmol.intor(int2c, hermi=1))
    if numpy.iscomplexobj(j2c):
        raise NotImplementedError("complex auxiliary metrics are unsupported")
    j2c = numpy.asarray(j2c, dtype=numpy.float64)
    try:
        low = scipy.linalg.cholesky(
            j2c, lower=True, overwrite_a=False, check_finite=False
        )
    except scipy.linalg.LinAlgError as error:
        raise RuntimeError(
            "the auxiliary metric is not full-rank positive definite.  "
            "MPI CDERI gradients currently require the Cholesky gauge; "
            "eigenvalue-truncated ED construction is not yet supported"
        ) from error
    return numpy.ascontiguousarray(low, dtype=numpy.float64)


def _integral_context(mol, auxmol, int3c):
    intor_name = gto.moleintor.ascint3(mol._add_suffix(int3c))
    atm, bas, env = gto.mole.conc_env(
        mol._atm, mol._bas, mol._env,
        auxmol._atm, auxmol._bas, auxmol._env,
    )
    ao_loc = gto.moleintor.make_loc(bas, intor_name)
    nao = int(ao_loc[mol.nbas])
    naux = int(ao_loc[-1] - nao)
    cintopt = gto.moleintor.make_cintopt(atm, bas, env, intor_name)
    return intor_name, atm, bas, env, ao_loc, cintopt, nao, naux


def _transform_int3c(raw, low, naux_raw):
    if raw.ndim == 3 and raw.flags.f_contiguous:
        raw = lib.transpose(raw.T, axes=(0, 2, 1)).reshape(naux_raw, -1)
    else:
        raw = raw.reshape((-1, naux_raw)).T
    transformed = scipy.linalg.solve_triangular(
        low,
        raw,
        lower=True,
        overwrite_b=True,
        check_finite=False,
    )
    return numpy.ascontiguousarray(transformed, dtype=numpy.float64)


def _write_rank_shard(
    mol,
    auxmol,
    low,
    shranges,
    block_bounds,
    pair_offsets,
    shard_path,
    *,
    rank,
    int3c,
    aosym,
    dataname,
    system_digest,
):
    block_start, block_stop = map(int, block_bounds)
    pair_start = int(pair_offsets[block_start])
    pair_stop = int(pair_offsets[block_stop])
    local_ranges = tuple(shranges[block_start:block_stop])
    local_npair = pair_stop - pair_start
    naux_fit, naux_raw = map(int, low.shape)
    if naux_fit != naux_raw:
        raise NotImplementedError(
            "only full-rank Cholesky metric factors are supported"
        )

    (
        intor_name,
        atm,
        bas,
        env,
        ao_loc,
        cintopt,
        nao,
        context_naux,
    ) = _integral_context(mol, auxmol, int3c)
    if context_naux != naux_raw:
        raise RuntimeError(
            f"integral context has naux={context_naux}, metric has {naux_raw}"
        )

    digest = hashlib.sha256()
    started = time.perf_counter()
    with h5py.File(shard_path, "w") as shard:
        dataset = shard.create_dataset(
            dataname, (naux_fit, local_npair), dtype=numpy.float64
        )
        shard.attrs["pyscfad_mpi_df_format"] = _FORMAT_VERSION
        shard.attrs["rank"] = int(rank)
        shard.attrs["pair_start"] = pair_start
        shard.attrs["pair_stop"] = pair_stop
        shard.attrs["block_start"] = block_start
        shard.attrs["block_stop"] = block_stop
        shard.attrs["system_digest"] = system_digest

        max_nrow = max((item[2] for item in local_ranges), default=0)
        raw_buffer = (
            numpy.empty((int(max_nrow), naux_raw), dtype=numpy.float64)
            if max_nrow else None
        )
        local_column = 0
        for sh_range in local_ranges:
            shell_start, shell_stop, nrow = map(int, sh_range)
            shls_slice = (
                shell_start,
                shell_stop,
                0,
                mol.nbas,
                mol.nbas,
                mol.nbas + auxmol.nbas,
            )
            raw = gto.moleintor.getints3c(
                intor_name,
                atm,
                bas,
                env,
                shls_slice,
                1,
                aosym,
                ao_loc,
                cintopt,
                out=raw_buffer,
            )
            transformed = _transform_int3c(raw, low, naux_raw)
            if transformed.shape != (naux_fit, nrow):
                raise RuntimeError(
                    "three-centre block has shape "
                    f"{transformed.shape}, expected {(naux_fit, nrow)}"
                )
            if not numpy.all(numpy.isfinite(transformed)):
                raise RuntimeError("three-centre block contains non-finite values")
            dataset[:, local_column:local_column + nrow] = transformed
            digest.update(transformed.tobytes())
            local_column += nrow
        if local_column != local_npair:
            raise RuntimeError(
                f"rank {rank} wrote {local_column} pair columns, "
                f"expected {local_npair}"
            )
        shard.flush()

    return MPICDERIRankManifest(
        rank=int(rank),
        pair_start=pair_start,
        pair_stop=pair_stop,
        block_start=block_start,
        block_stop=block_stop,
        block_count=block_stop - block_start,
        pair_columns=local_npair,
        byte_count=naux_fit * local_npair * numpy.dtype(numpy.float64).itemsize,
        integral_seconds=float(time.perf_counter() - started),
        checksum=digest.hexdigest(),
    )


def _assemble_shards(
    output_path,
    scratch_path,
    manifests,
    *,
    naux,
    nao_pair,
    dataname,
    system_digest,
    assembly_block_mb,
    overwrite,
):
    started = time.perf_counter()
    partial_path = scratch_path / "assembled.partial.h5"
    max_local_pair = max(
        (manifest.pair_columns for manifest in manifests), default=1
    )
    target_bytes = max(float(assembly_block_mb), 1.0) * 1e6
    aux_block = max(
        1,
        min(int(naux), int(target_bytes / (8 * max(max_local_pair, 1)))),
    )

    with h5py.File(partial_path, "w") as output:
        dataset = output.create_dataset(
            dataname, (int(naux), int(nao_pair)), dtype=numpy.float64
        )
        output.attrs["pyscfad_mpi_df_format"] = _FORMAT_VERSION
        output.attrs["pyscfad_mpi_df_complete"] = True
        output.attrs["decomposition"] = "CD"
        output.attrs["system_digest"] = system_digest
        output.attrs["nproc"] = len(manifests)
        output.attrs["naux_raw"] = int(naux)

        expected_pair = 0
        shard_handles = []
        try:
            for manifest in manifests:
                if manifest.pair_start != expected_pair:
                    raise RuntimeError(
                        "rank shard pair ranges are not contiguous: "
                        f"expected {expected_pair}, got {manifest.pair_start}"
                    )
                expected_pair = manifest.pair_stop
                shard_path = scratch_path / f"rank_{manifest.rank:06d}.h5"
                shard = h5py.File(shard_path, "r")
                shard_handles.append(shard)
                source = shard[dataname]
                expected_shape = (int(naux), manifest.pair_columns)
                if tuple(source.shape) != expected_shape:
                    raise RuntimeError(
                        f"rank {manifest.rank} shard shape {source.shape}, "
                        f"expected {expected_shape}"
                    )
                if numpy.dtype(source.dtype) != numpy.dtype(numpy.float64):
                    raise RuntimeError(
                        f"rank {manifest.rank} shard dtype is {source.dtype}"
                    )
                expected_attrs = {
                    "rank": manifest.rank,
                    "pair_start": manifest.pair_start,
                    "pair_stop": manifest.pair_stop,
                    "block_start": manifest.block_start,
                    "block_stop": manifest.block_stop,
                    "system_digest": system_digest,
                }
                for name, expected in expected_attrs.items():
                    if shard.attrs.get(name) != expected:
                        raise RuntimeError(
                            f"rank {manifest.rank} shard attribute {name!r} "
                            f"is {shard.attrs.get(name)!r}, expected {expected!r}"
                        )
            if expected_pair != int(nao_pair):
                raise RuntimeError(
                    f"rank shards cover {expected_pair} pair columns, "
                    f"expected {nao_pair}"
                )

            for q0 in range(0, int(naux), aux_block):
                q1 = min(q0 + aux_block, int(naux))
                for manifest, shard in zip(manifests, shard_handles):
                    if manifest.pair_columns == 0:
                        continue
                    dataset[
                        q0:q1, manifest.pair_start:manifest.pair_stop
                    ] = shard[dataname][q0:q1, :]
            output.flush()
        finally:
            for shard in shard_handles:
                shard.close()

    if overwrite:
        os.replace(partial_path, output_path)
    else:
        # link(2) creates the destination atomically and fails if a competing
        # job published it while this build was running. Unlike rename(2), it
        # therefore enforces the advertised no-clobber behavior.
        os.link(partial_path, output_path)
        partial_path.unlink()
    return float(time.perf_counter() - started)


def _validate_final_file(
    output_path,
    *,
    naux,
    nao_pair,
    dataname,
    system_digest,
):
    with h5py.File(output_path, "r") as handle:
        if dataname not in handle:
            raise RuntimeError(f"completed CDERI file has no {dataname!r} dataset")
        dataset = handle[dataname]
        expected_shape = (int(naux), int(nao_pair))
        if tuple(dataset.shape) != expected_shape:
            raise RuntimeError(
                f"completed CDERI shape is {dataset.shape}, "
                f"expected {expected_shape}"
            )
        if numpy.dtype(dataset.dtype) != numpy.dtype(numpy.float64):
            raise RuntimeError(
                f"completed CDERI dtype is {dataset.dtype}, expected float64"
            )
        if not bool(handle.attrs.get("pyscfad_mpi_df_complete", False)):
            raise RuntimeError("completed CDERI marker is missing")
        if handle.attrs.get("system_digest") != system_digest:
            raise RuntimeError("completed CDERI system digest is inconsistent")
        if naux and nao_pair:
            sample_rows = sorted({0, int(naux) - 1})
            samples = numpy.asarray(dataset[sample_rows, :])
            if not numpy.all(numpy.isfinite(samples)):
                raise RuntimeError("completed CDERI contains non-finite values")


def build_cderi(
    mol,
    erifile,
    *,
    auxbasis="weigend+etb",
    comm=MPI.COMM_WORLD,
    root=0,
    dataname="j3c",
    int3c="int3c2e",
    int2c="int2c2e",
    aosym="s2ij",
    max_memory=2000,
    min_blocks_per_rank=2,
    assembly_block_mb=128.0,
    overwrite=False,
    progress=False,
):
    """Collectively build a standard out-of-core Cholesky CDERI file.

    All ranks must call this function with equivalent molecules, auxiliary
    bases, options, and a path visible to every rank.  Existing output is
    preserved unless ``overwrite=True``; replacement is atomic after a fully
    successful assembly.  Rank-local temporary shards are created beside the
    final file and removed before return.
    """

    rank = int(comm.Get_rank())
    size = int(comm.Get_size())
    started = time.perf_counter()
    local_setup_error = None
    root_int = 0
    progress_enabled = False
    output_path = None
    concrete_mol = None
    auxmol = None
    try:
        if not isinstance(root, (int, numpy.integer)):
            raise TypeError("root must be an integer MPI rank")
        root_int = int(root)
        if root_int < 0 or root_int >= size:
            raise ValueError(f"root={root!r} is invalid for {size} MPI ranks")
        if aosym != "s2ij":
            raise NotImplementedError(
                "MPI CDERI build currently requires aosym='s2ij'"
            )
        max_memory = float(max_memory)
        if not numpy.isfinite(max_memory) or max_memory <= 0:
            raise ValueError("max_memory must be finite and positive")
        min_blocks_per_rank = int(min_blocks_per_rank)
        if min_blocks_per_rank <= 0:
            raise ValueError("min_blocks_per_rank must be positive")
        assembly_block_mb = float(assembly_block_mb)
        if not numpy.isfinite(assembly_block_mb) or assembly_block_mb <= 0:
            raise ValueError("assembly_block_mb must be finite and positive")
        progress_enabled = _progress_enabled(progress)
        if not isinstance(erifile, (str, bytes, os.PathLike)):
            raise TypeError("erifile must be a filesystem path")
        output_path = Path(os.path.abspath(os.fsdecode(os.fspath(erifile))))

        concrete_mol = _concrete_pyscf_mol(mol)
        auxmol = pyscf_df_addons.make_auxmol(concrete_mol, auxbasis)
        if not concrete_mol.cart and auxmol.cart:
            raise NotImplementedError(
                "spherical AO / Cartesian auxiliary is unsupported"
            )
        if concrete_mol.cart and not auxmol.cart:
            raise RuntimeError(
                "Cartesian AO / spherical auxiliary is unsupported"
            )
    except Exception:
        local_setup_error = _exception_text("local input preparation", rank)
    setup_errors = tuple(
        error for error in comm.allgather(local_setup_error)
        if error is not None
    )
    if setup_errors:
        raise RuntimeError("\n".join(setup_errors))
    root = root_int
    reporter = _progress_reporter(progress, rank=rank, root=root)

    signature = _collective_signature(
        output_path,
        concrete_mol,
        auxmol,
        root=root,
        int3c=int3c,
        int2c=int2c,
        aosym=aosym,
        dataname=dataname,
        max_memory=max_memory,
        min_blocks_per_rank=min_blocks_per_rank,
        assembly_block_mb=assembly_block_mb,
        overwrite=overwrite,
        progress_enabled=progress_enabled,
    )
    signatures = comm.allgather(signature)
    if len(set(signatures)) != 1:
        component_digests = comm.allgather(
            _system_component_digests(concrete_mol, auxmol)
        )
        rank_signatures = "\n".join(
            f"rank {index}: {item!r}; components={component_digests[index]!r}"
            for index, item in enumerate(signatures)
        )
        raise ValueError(
            "MPI CDERI path, molecule, auxiliary basis, or build options "
            f"differ across ranks:\n{rank_signatures}"
        )

    setup_error = None
    scratch_path = None
    if rank == root:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists() and not overwrite:
                raise FileExistsError(
                    f"CDERI output already exists: {output_path}; pass "
                    "overwrite=True to replace it atomically"
                )
            scratch_path = Path(tempfile.mkdtemp(
                prefix=f".{output_path.name}.mpi-",
                dir=output_path.parent,
            ))
        except Exception:
            setup_error = _exception_text("shared scratch setup", rank)
    setup_state = comm.bcast(
        (
            setup_error,
            None if scratch_path is None else str(scratch_path),
        ),
        root=root,
    )
    if setup_state[0] is not None:
        raise RuntimeError(setup_state[0])
    scratch_path = Path(setup_state[1])

    visibility_error = None
    try:
        marker = scratch_path / f"visible_rank_{rank:06d}"
        marker.write_text(f"{rank}\n", encoding="ascii")
    except Exception:
        visibility_error = _exception_text("shared-path visibility", rank)
    _raise_if_any_rank_failed(
        comm,
        visibility_error,
        scratch_path=scratch_path,
        rank=rank,
        root=root,
    )
    root_visibility_error = None
    if rank == root:
        try:
            missing = [
                index for index in range(size)
                if not (scratch_path / f"visible_rank_{index:06d}").is_file()
            ]
            if missing:
                raise RuntimeError(
                    "output directory is not shared with MPI ranks "
                    + ", ".join(map(str, missing))
                )
        except Exception:
            root_visibility_error = _exception_text(
                "shared-path verification", rank
            )
    root_visibility_error = comm.bcast(root_visibility_error, root=root)
    if root_visibility_error is not None:
        _cleanup_shared_scratch(
            comm, scratch_path, rank=rank, root=root
        )
        raise RuntimeError(root_visibility_error)

    metric_error = None
    low = None
    metric_seconds = 0.0
    if rank == root:
        try:
            _report(reporter, "auxiliary metric Cholesky: starting on root")
            metric_start = time.perf_counter()
            low = _metric_cholesky(auxmol, int2c)
            metric_seconds = time.perf_counter() - metric_start
        except Exception:
            metric_error = _exception_text("auxiliary metric", rank)
    metric_state = comm.bcast(
        (
            metric_error,
            None if low is None else tuple(low.shape),
            float(metric_seconds),
        ),
        root=root,
    )
    if metric_state[0] is not None:
        _cleanup_shared_scratch(
            comm, scratch_path, rank=rank, root=root
        )
        raise RuntimeError(metric_state[0])
    low_shape = tuple(metric_state[1])
    metric_seconds = float(metric_state[2])

    allocation_error = None
    if rank != root:
        try:
            low = numpy.empty(low_shape, dtype=numpy.float64)
        except Exception:
            allocation_error = _exception_text("metric-buffer allocation", rank)
    _raise_if_any_rank_failed(
        comm,
        allocation_error,
        scratch_path=scratch_path,
        rank=rank,
        root=root,
    )
    comm.Bcast(low, root=root)

    naux_fit, naux_raw = map(int, low.shape)
    if naux_fit != naux_raw:
        raise NotImplementedError("MPI CDERI gradients require full-rank Cholesky")
    nao = int(concrete_mol.nao_nr())
    nao_pair = nao * (nao + 1) // 2
    memory_buflen = min(
        max(int(float(max_memory) * .24e6 / 8 / max(naux_raw, 1)), 1),
        nao_pair,
    )
    target_blocks = max(size * int(min_blocks_per_rank), 1)
    parallel_buflen = max(1, (nao_pair + target_blocks - 1) // target_blocks)
    buflen = min(memory_buflen, parallel_buflen)
    shranges = tuple(
        outcore._guess_shell_ranges(concrete_mol, buflen, aosym)
    )
    pair_offsets = _pair_offsets(shranges)
    if pair_offsets[-1] != nao_pair:
        raise RuntimeError(
            f"AO shell ranges cover {pair_offsets[-1]} pairs, expected {nao_pair}"
        )
    partitions = _contiguous_block_partition(shranges, size)
    system_digest = signature[-1]
    if rank == root:
        _report(
            reporter,
            f"three-centre generation: {len(shranges)} AO-pair blocks, "
            f"{nao_pair} columns on {size} ranks",
        )

    local_error = None
    manifest = None
    try:
        manifest = _write_rank_shard(
            concrete_mol,
            auxmol,
            low,
            shranges,
            partitions[rank],
            pair_offsets,
            scratch_path / f"rank_{rank:06d}.h5",
            rank=rank,
            int3c=int3c,
            aosym=aosym,
            dataname=dataname,
            system_digest=system_digest,
        )
    except Exception:
        local_error = _exception_text("three-centre shard generation", rank)
    generation_records = comm.allgather((local_error, manifest))
    generation_errors = tuple(
        error for error, _ in generation_records if error is not None
    )
    if generation_errors:
        _cleanup_shared_scratch(
            comm, scratch_path, rank=rank, root=root
        )
        raise RuntimeError("\n".join(generation_errors))
    manifests = tuple(record[1] for record in generation_records)

    if rank == root:
        split = ", ".join(
            f"r{item.rank}:{item.pair_columns}"
            for item in manifests
        )
        _report(reporter, f"three-centre generation complete; columns [{split}]")

    assembly_error = None
    assembly_seconds = 0.0
    if rank == root:
        try:
            _report(reporter, "shared CDERI assembly: starting on root")
            assembly_seconds = _assemble_shards(
                output_path,
                scratch_path,
                manifests,
                naux=naux_fit,
                nao_pair=nao_pair,
                dataname=dataname,
                system_digest=system_digest,
                assembly_block_mb=assembly_block_mb,
                overwrite=overwrite,
            )
        except Exception:
            assembly_error = _exception_text("shared-file assembly", rank)
    assembly_state = comm.bcast(
        (assembly_error, float(assembly_seconds)), root=root
    )
    assembly_error, assembly_seconds = assembly_state
    if assembly_error is not None:
        _cleanup_shared_scratch(
            comm, scratch_path, rank=rank, root=root
        )
        raise RuntimeError(assembly_error)

    validation_error = None
    try:
        _validate_final_file(
            output_path,
            naux=naux_fit,
            nao_pair=nao_pair,
            dataname=dataname,
            system_digest=system_digest,
        )
    except Exception:
        validation_error = _exception_text("final shared-file validation", rank)
    validation_errors = comm.allgather(validation_error)
    validation_errors = tuple(
        error for error in validation_errors if error is not None
    )
    _cleanup_shared_scratch(
        comm, scratch_path, rank=rank, root=root
    )
    if validation_errors:
        raise RuntimeError("\n".join(validation_errors))

    wall_seconds = float(comm.allreduce(
        time.perf_counter() - started, op=MPI.MAX
    ))
    result = MPICDERIBuildResult(
        path=str(output_path),
        decomposition="CD",
        naux_raw=naux_raw,
        naux=naux_fit,
        nao_pair=nao_pair,
        nblocks=len(shranges),
        nproc=size,
        metric_seconds=metric_seconds,
        assembly_seconds=float(assembly_seconds),
        wall_seconds=wall_seconds,
        manifests=manifests,
    )
    _report(
        reporter,
        f"complete in {wall_seconds:.2f} s; shape="
        f"({naux_fit}, {nao_pair}), metric={metric_seconds:.2f} s, "
        f"assembly={assembly_seconds:.2f} s",
    )
    return result
