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

"""Root-owned MPI service for molecular density-fitted J/K builds.

This module deliberately does not make MPI a JAX primitive.  Rank ``root``
owns the SCF program and its JAX tape, while the other ranks block in
:meth:`MPIDFJKExecutor.serve`.  A custom VJP on the root broadcasts the small
dense AO operands and distributes the large CDERI contraction over
orthogonalized auxiliary-function rows.

Density-only J/K transposes used by the implicit SCF response are distributed
over auxiliary rows.  In the one complete J/K pullback, root constructs the
CDERI cotangent and performs the globally coupled Cholesky-whitening algebra,
then streams independent AO-pair shell blocks to the MPI ranks for the
three-centre integral coordinate VJPs.  The final two-centre metric/Cholesky
pullback remains on root.

Only real float64 RHF densities and packed ``s2ij`` CDERI are supported.
Every rank must provide a compatible DF object backed by a prebuilt, shared
(or equivalent rank-local) CDERI source.  The service supports eager first
derivatives; JIT staging, forward-mode differentiation, and Hessians are not
part of this initial MPI path.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import ctypes
from enum import Enum
from functools import partial
import itertools
import traceback

from jax import custom_vjp
from jax.tree_util import tree_flatten
import numpy
from mpi4py import MPI
from pyscf import lib
from pyscf.df import df_jk as pyscf_df_jk

from pyscfad._src.implicit_diff import is_implicit_diff_solve_matvec
from pyscfad.df import addons
from pyscfad.df import _cderi_vjp
from pyscfad.df import _df_jk_opt


__all__ = [
    "MPIDFJKExecutor",
    "ServiceExit",
    "get_active_executor",
    "local_aux_range",
    "local_density_vjp",
    "local_jk",
    "mpi_get_jk",
]


_OP_FORWARD = "forward"
_OP_DENSITY_VJP = "density_vjp"
_OP_COORDINATE_VJP = "coordinate_vjp"
_OP_PAUSE = "pause"
_OP_STOP = "stop"

# MPI guarantees tags through at least 32767.  The coordinate service is
# synchronous, so fixed tags are sufficient and avoid pickling large blocks.
_COORDINATE_HEADER_TAG = 27101
_COORDINATE_DATA_TAG = 27102
_COORDINATE_READY_TAG = 27103

_ACTIVE_EXECUTOR = ContextVar("pyscfad_mpi_df_jk_executor", default=None)
_EXECUTOR_TOKENS = itertools.count()
_EXECUTORS = {}


class ServiceExit(Enum):
    """Reason a non-root :meth:`MPIDFJKExecutor.serve` call returned."""

    PAUSED = "paused"
    STOPPED = "stopped"


def get_active_executor():
    """Return the executor active in this Python context, or ``None``.

    ``pyscfad.df.df_jk.get_jk`` can use this hook without placing an MPI
    communicator in a JAX pytree.  The communicator remains process-local
    and the custom VJP carries only a small integer registry token.
    """

    return _ACTIVE_EXECUTOR.get()


def local_aux_range(naux, rank, size):
    """Return the balanced half-open auxiliary-row interval for one rank."""

    naux = int(naux)
    rank = int(rank)
    size = int(size)
    if naux < 0:
        raise ValueError(f"naux must be non-negative, got {naux}")
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if rank < 0 or rank >= size:
        raise ValueError(f"rank={rank} is invalid for size={size}")
    return naux * rank // size, naux * (rank + 1) // size


def _cderi_source(dfobj):
    getter = getattr(dfobj, "_get_cderi_source", None)
    source = getter() if callable(getter) else getattr(dfobj, "_cderi", None)
    if source is None:
        raise RuntimeError(
            "MPI DF-J/K requires a prebuilt CDERI source on every rank"
        )
    # A concrete in-core CDERI leaf returns from a custom-VJP residual as a
    # JAX ``TypedNdArray``.  PySCF's HDF5 loader does not recognize that
    # container even though it has a valid NumPy buffer.  Outcore paths and
    # HDF5 datasets must remain lazy and are therefore left untouched.
    if type(source).__module__.startswith(("jax.", "jaxlib.")):
        source = numpy.asarray(source)
    return source


def _df_metadata(dfobj):
    nao = int(dfobj.mol.nao)
    npair = nao * (nao + 1) // 2
    with addons.load(_cderi_source(dfobj), "j3c") as cderi:
        if not hasattr(cderi, "shape") or len(cderi.shape) != 2:
            raise NotImplementedError(
                "MPI DF-J/K requires a two-dimensional packed CDERI source"
            )
        naux, stored_pair = map(int, cderi.shape)
        cderi_dtype = numpy.dtype(cderi.dtype)
    if cderi_dtype != numpy.dtype(numpy.float64):
        raise NotImplementedError(
            "MPI DF-J/K currently requires float64 CDERI, "
            f"got dtype={cderi_dtype}"
        )
    if stored_pair != npair:
        raise NotImplementedError(
            "MPI DF-J/K supports packed s2ij CDERI only: "
            f"source shape={(naux, stored_pair)}, expected pair size={npair}"
        )
    return nao, naux, npair


def _iter_cderi_range(dfobj, q0, q1, blksize=None):
    """Yield concrete packed CDERI blocks in ``[q0, q1)``."""

    if blksize is None:
        blksize = int(getattr(dfobj, "blockdim", 240))
    blksize = max(int(blksize), 1)
    source = _cderi_source(dfobj)
    with addons.load(source, "j3c") as cderi:
        naux = int(cderi.shape[0])
        if q0 < 0 or q1 < q0 or q1 > naux:
            raise ValueError(
                f"invalid CDERI interval [{q0}, {q1}) for naux={naux}"
            )
        for p0 in range(q0, q1, blksize):
            p1 = min(p0 + blksize, q1)
            yield numpy.asarray(cderi[p0:p1], order="C")


class _LocalDFView:
    """Minimal PySCF DF interface restricted to an auxiliary-row interval."""

    def __init__(self, dfobj, q0, q1):
        self._dfobj = dfobj
        self._q0 = int(q0)
        self._q1 = int(q1)
        # PySCF checks only whether this is None before choosing an integral-
        # direct branch.  Actual data are supplied by ``loop`` below.
        self._cderi = True

    def __getattr__(self, name):
        return getattr(self._dfobj, name)

    def loop(self, blksize=None):
        yield from _iter_cderi_range(
            self._dfobj, self._q0, self._q1, blksize=blksize
        )


def _require_real_rhf_array(name, value, nao=None):
    array = numpy.asarray(value)
    if numpy.iscomplexobj(array):
        raise NotImplementedError(f"MPI DF-J/K does not support complex {name}")
    if array.dtype != numpy.dtype(numpy.float64):
        raise NotImplementedError(
            f"MPI DF-J/K currently requires float64 {name}, "
            f"got dtype={array.dtype}"
        )
    if nao is not None and array.shape[-2:] != (nao, nao):
        raise ValueError(
            f"{name} has trailing shape {array.shape[-2:]}, "
            f"expected {(nao, nao)}"
        )
    return array


def _coordinate_bar_leaf(bar, reference, label):
    """Return one coordinate-only Mole cotangent as a dense host array."""

    expected_shape = (int(reference.natm), 3)
    if bar is None:
        return numpy.zeros(expected_shape, dtype=numpy.float64)
    leaves = tree_flatten(bar)[0]
    if len(leaves) != 1:
        raise NotImplementedError(
            "MPI DF coordinate VJP supports coordinate-only Mole pytrees; "
            f"{label} cotangent has {len(leaves)} differentiable leaves"
        )
    array = numpy.asarray(leaves[0])
    if numpy.iscomplexobj(array) or array.dtype != numpy.dtype(numpy.float64):
        raise NotImplementedError(
            "MPI DF coordinate VJP requires real float64 coordinates; "
            f"{label} cotangent has dtype={array.dtype}"
        )
    if array.shape != expected_shape:
        raise ValueError(
            f"{label} coordinate cotangent has shape {array.shape}, "
            f"expected {expected_shape}"
        )
    return numpy.asarray(array, order="C")


def _pack_coordinate_bars(mol_bar, auxmol_bar, mol, auxmol):
    mol_coords = _coordinate_bar_leaf(mol_bar, mol, "molecule")
    aux_coords = _coordinate_bar_leaf(auxmol_bar, auxmol, "auxiliary molecule")
    if mol_coords.shape != aux_coords.shape:
        raise NotImplementedError(
            "MPI DF coordinate VJP requires the primary and auxiliary "
            "molecules to contain the same atoms"
        )
    return numpy.ascontiguousarray(numpy.stack((mol_coords, aux_coords)))


def _tag_density(dm, mo_coeff, mo_occ):
    dm = numpy.asarray(dm)
    if mo_coeff is None or mo_occ is None:
        return dm
    mo_coeff = _require_real_rhf_array("MO coefficients", mo_coeff)
    mo_occ = _require_real_rhf_array("MO occupations", mo_occ)
    return lib.tag_array(
        dm,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
    )


def local_jk(
    dfobj,
    dm,
    *,
    rank=0,
    size=1,
    hermi=1,
    with_j=True,
    with_k=True,
    direct_scf_tol=1e-13,
    mo_coeff=None,
    mo_occ=None,
    expected_metadata=None,
    local_metadata=None,
):
    """Compute one rank's additive DF-J/K contribution.

    No MPI calls are made.  This helper is public primarily for numerical and
    adjoint tests; production callers should use :class:`MPIDFJKExecutor`.
    """

    if not (with_j or with_k):
        raise ValueError("at least one of with_j and with_k must be true")
    if hermi != 1:
        raise NotImplementedError("MPI DF-J/K currently requires hermi=1")

    metadata = (
        _df_metadata(dfobj)
        if local_metadata is None else tuple(local_metadata)
    )
    nao, naux, npair = metadata
    if expected_metadata is not None and tuple(expected_metadata) != metadata:
        raise RuntimeError(
            "MPI ranks have incompatible DF sources: "
            f"local={metadata}, root={tuple(expected_metadata)}"
        )
    dm = _require_real_rhf_array("density", dm, nao=nao)
    q0, q1 = local_aux_range(naux, rank, size)
    if q0 == q1:
        zeros = numpy.zeros(dm.shape, dtype=dm.dtype)
        return zeros.copy(), zeros.copy()

    local_df = _LocalDFView(dfobj, q0, q1)
    tagged_dm = _tag_density(dm, mo_coeff, mo_occ)
    vj, vk = pyscf_df_jk.get_jk(
        local_df,
        tagged_dm,
        hermi=hermi,
        with_j=with_j,
        with_k=with_k,
        direct_scf_tol=direct_scf_tol,
    )
    zeros = numpy.zeros(dm.shape, dtype=dm.dtype)
    vj = zeros.copy() if not with_j else numpy.asarray(vj).reshape(dm.shape)
    vk = zeros.copy() if not with_k else numpy.asarray(vk).reshape(dm.shape)
    return vj, vk


def _generic_exchange_density_vjp(dm_bars, vk_bar, eri1):
    """Portable NumPy fallback for the real DF exchange transpose."""

    cderi = lib.unpack_tril(eri1)
    for index, output in enumerate(dm_bars):
        output += numpy.einsum(
            "Lac,ab,Ldb->cd",
            cderi,
            vk_bar[index],
            cderi,
            optimize=True,
        )


def local_density_vjp(
    dfobj,
    dm_shape,
    vj_bar,
    vk_bar,
    *,
    rank=0,
    size=1,
    hermi=1,
    with_j=True,
    with_k=True,
    lowrank_factors=None,
    expected_metadata=None,
    local_metadata=None,
):
    """Compute one rank's density cotangent for a DF-J/K call.

    This is the reverse operation needed repeatedly by implicit SCF/CPHF.
    It intentionally does not construct a CDERI, molecule, or auxiliary-
    molecule cotangent.
    """

    if not (with_j or with_k):
        raise ValueError("at least one of with_j and with_k must be true")
    if hermi != 1:
        raise NotImplementedError("MPI DF-J/K currently requires hermi=1")

    metadata = (
        _df_metadata(dfobj)
        if local_metadata is None else tuple(local_metadata)
    )
    nao, naux, npair = metadata
    if expected_metadata is not None and tuple(expected_metadata) != metadata:
        raise RuntimeError(
            "MPI ranks have incompatible DF sources: "
            f"local={metadata}, root={tuple(expected_metadata)}"
        )

    dm_shape = tuple(int(value) for value in dm_shape)
    if len(dm_shape) < 2 or dm_shape[-2:] != (nao, nao):
        raise ValueError(
            f"density shape {dm_shape} is incompatible with nao={nao}"
        )
    nset = int(numpy.prod(dm_shape[:-2], dtype=int)) if len(dm_shape) > 2 else 1
    vj_bar = _require_real_rhf_array("J cotangent", vj_bar, nao=nao)
    vk_bar = _require_real_rhf_array("K cotangent", vk_bar, nao=nao)
    vj_bar = vj_bar.reshape(nset, nao, nao)
    vk_bar = vk_bar.reshape(nset, nao, nao)
    dm_bars = [numpy.zeros((nao, nao), order="F") for _ in range(nset)]

    if with_j:
        diagonal = numpy.arange(nao)
        vj_bar_tril = lib.pack_tril(
            vj_bar + vj_bar.transpose(0, 2, 1)
        )
        vj_bar_tril[:, diagonal * (diagonal + 1) // 2 + diagonal] *= 0.5
    else:
        vj_bar_tril = None

    if lowrank_factors is not None:
        if nset != 1:
            raise NotImplementedError(
                "low-rank exchange transpose requires one RHF density"
            )
        occ_coeff, response_coeff = (
            numpy.asarray(
                _require_real_rhf_array(
                    "occupied exchange factor", lowrank_factors[0]
                ),
                order="F",
            ),
            numpy.asarray(
                _require_real_rhf_array(
                    "response exchange factor", lowrank_factors[1]
                ),
                order="F",
            ),
        )
        if occ_coeff.shape != response_coeff.shape:
            raise ValueError(
                "occupied and response exchange factors have different shapes"
            )
        if occ_coeff.ndim != 2 or occ_coeff.shape[0] != nao:
            raise ValueError(
                "low-rank exchange factors must have shape (nao, rank); "
                f"got {occ_coeff.shape} for nao={nao}"
            )
    else:
        occ_coeff = response_coeff = None

    q0, q1 = local_aux_range(naux, rank, size)
    if q0 == q1:
        return numpy.asarray(dm_bars).reshape(dm_shape)

    native_exchange_vjp = _df_jk_opt._DF_VK_DM_VJP
    for eri1 in _iter_cderi_range(dfobj, q0, q1):
        naux_block, pair_block = eri1.shape
        if pair_block != npair:
            raise RuntimeError(
                f"DF block has pair size {pair_block}, expected {npair}"
            )
        if with_j:
            rho_bar = vj_bar_tril @ eri1.T
            packed_dm_bar = rho_bar @ eri1
            dense_dm_bar = lib.unpack_tril(packed_dm_bar)
            for index in range(nset):
                dm_bars[index] += dense_dm_bar[index]

        if not with_k:
            continue
        if lowrank_factors is not None:
            response_rank = occ_coeff.shape[1]
            buf_occ = numpy.empty((naux_block * response_rank, nao))
            buf_response = numpy.empty_like(buf_occ)
            _df_jk_opt._df_vk_dm_vjp_lowrank(
                dm_bars[0],
                eri1,
                occ_coeff,
                response_coeff,
                buf_occ,
                buf_response,
            )
        elif native_exchange_vjp is not None:
            for index in range(nset):
                vk_bar_block = numpy.asarray(vk_bar[index], order="F")
                native_exchange_vjp(
                    dm_bars[index].ctypes.data_as(ctypes.c_void_p),
                    vk_bar_block.ctypes.data_as(ctypes.c_void_p),
                    eri1.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(naux_block),
                    ctypes.c_int(nao),
                )
        else:
            _generic_exchange_density_vjp(dm_bars, vk_bar, eri1)

    return numpy.asarray(dm_bars).reshape(dm_shape)


def _executor_from_token(token):
    try:
        return _EXECUTORS[int(token)]
    except KeyError as error:
        raise RuntimeError(
            "MPI DF-J/K executor is no longer registered; the SCF pullback "
            "must run before stop_workers()"
        ) from error


@partial(custom_vjp, nondiff_argnums=(0, 3, 4, 5, 6))
def mpi_get_jk(
    executor_token,
    dfobj,
    dm,
    hermi=1,
    with_j=True,
    with_k=True,
    direct_scf_tol=1e-13,
):
    """MPI DF-J/K custom-VJP entry point used on rank ``root``."""

    executor = _executor_from_token(executor_token)
    return executor._forward(
        dfobj,
        dm,
        hermi=hermi,
        with_j=with_j,
        with_k=with_k,
        direct_scf_tol=direct_scf_tol,
    )


def _mpi_get_jk_fwd(
    executor_token,
    dfobj,
    dm,
    hermi,
    with_j,
    with_k,
    direct_scf_tol,
):
    result = mpi_get_jk(
        executor_token,
        dfobj,
        dm,
        hermi,
        with_j,
        with_k,
        direct_scf_tol,
    )
    return result, (dfobj, dm)


def _mpi_get_jk_bwd(
    executor_token,
    hermi,
    with_j,
    with_k,
    direct_scf_tol,
    residual,
    output_bar,
):
    executor = _executor_from_token(executor_token)
    dfobj, dm = residual
    vj_bar, vk_bar = output_bar
    if is_implicit_diff_solve_matvec():
        dm_bar = executor._density_vjp(
            dfobj,
            dm,
            vj_bar,
            vk_bar,
            hermi=hermi,
            with_j=with_j,
            with_k=with_k,
        )
        return None, dm_bar

    return _df_jk_opt.get_jk_bwd(
        hermi,
        with_j,
        with_k,
        direct_scf_tol,
        (dfobj, dm),
        (vj_bar, vk_bar),
        coordinate_vjp=executor._coordinate_vjp,
    )


mpi_get_jk.defvjp(_mpi_get_jk_fwd, _mpi_get_jk_bwd)


def _remote_error(operation, rank):
    return {
        "operation": operation,
        "rank": int(rank),
        "traceback": traceback.format_exc(),
    }


class MPIDFJKExecutor:
    """Coordinate root-owned DF-J/K forward and reverse MPI services.

    The constructor is not collective.  On root, activate this executor
    around the SCF forward or pullback.  On workers, call :meth:`serve` with a
    compatible DF object.  A service may be paused so workers can participate
    in DLNO correlation and resumed later for the SCF pullback::

        # root
        with executor.root_session(final=False):
            mf, scf_pullback = jax.vjp(build_mf, mol)
        ... distributed correlation ...
        with executor.root_session(final=True):
            mol_bar, = scf_pullback(mf_bar)

        # every worker
        assert executor.serve(worker_df) is ServiceExit.PAUSED
        ... distributed correlation ...
        assert executor.serve(worker_df) is ServiceExit.STOPPED
    """

    def __init__(self, comm=MPI.COMM_WORLD, root=0):
        self.comm = comm
        self.root = int(root)
        self.rank = int(comm.Get_rank())
        self.size = int(comm.Get_size())
        if self.root < 0 or self.root >= self.size:
            raise ValueError(
                f"root={self.root} is invalid for {self.size} MPI ranks"
            )
        self.token = next(_EXECUTOR_TOKENS)
        self._closed = False
        self._metadata_cache = {}
        _EXECUTORS[self.token] = self

    @property
    def is_root(self):
        return self.rank == self.root

    def _require_root(self):
        if not self.is_root:
            raise RuntimeError("this MPI DF-J/K operation is root-only")
        if self._closed:
            raise RuntimeError("MPI DF-J/K executor has been stopped")

    @contextmanager
    def activate(self):
        """Make this executor visible to the root's DF ``get_jk`` hook."""

        self._require_root()
        previous = get_active_executor()
        if previous is not None and previous is not self:
            raise RuntimeError("a different MPI DF-J/K executor is active")
        context_token = _ACTIVE_EXECUTOR.set(self)
        # Import lazily so serial PySCFAD never imports mpi4py merely to build
        # J/K.  ``df_jk`` owns only a lightweight ContextVar hook and has no
        # dependency on this module.
        from pyscfad.df import df_jk as df_jk_module

        hook_var = getattr(df_jk_module, "_MPI_GET_JK_HOOK", None)
        hook_token = None if hook_var is None else hook_var.set(self.get_jk)
        try:
            yield self
        finally:
            if hook_token is not None:
                hook_var.reset(hook_token)
            _ACTIVE_EXECUTOR.reset(context_token)

    @contextmanager
    def root_session(self, *, final=False):
        """Activate root work, then pause or stop the matching worker serve."""

        self._require_root()
        try:
            with self.activate():
                yield self
        except BaseException:
            # Release workers waiting for another command before propagating
            # the root exception.  The executor cannot safely be reused.
            self.stop_workers()
            raise
        else:
            if final:
                self.stop_workers()
            else:
                self.pause_workers()

    def get_jk(
        self,
        dfobj,
        dm,
        hermi=1,
        with_j=True,
        with_k=True,
        direct_scf_tol=1e-13,
    ):
        """Call the registered MPI custom VJP on root."""

        self._require_root()
        return mpi_get_jk(
            self.token,
            dfobj,
            dm,
            hermi,
            with_j,
            with_k,
            direct_scf_tol,
        )

    def _metadata(self, dfobj):
        # DF's cache token survives JAX pytree reconstruction, unlike Python
        # object identity.  It avoids retaining every residual DF instance
        # merely to cache three scalar dimensions.
        stable_token = getattr(dfobj, "_fast_exchange_cache_token", None)
        key = (
            ("token", int(stable_token))
            if stable_token is not None
            else ("object", id(dfobj))
        )
        metadata = self._metadata_cache.get(key)
        if metadata is None:
            metadata = _df_metadata(dfobj)
            self._metadata_cache[key] = metadata
        return metadata

    def _root_metadata(self, dfobj):
        return self._metadata(dfobj)

    def _collective_errors(self, local_error):
        errors = self.comm.allgather(local_error)
        return [error for error in errors if error is not None]

    def _raise_remote_errors(self, errors):
        if not errors:
            return
        details = "\n".join(
            f"rank {error['rank']} during {error['operation']}:\n"
            f"{error['traceback']}"
            for error in errors
        )
        raise RuntimeError("MPI DF-J/K worker failure:\n" + details)

    def _reduce_sum_array(self, send_buffer, receive_buffer=None):
        """Reduce prepared dense buffers without Python-object pickling."""

        self.comm.Reduce(
            send_buffer, receive_buffer, op=MPI.SUM, root=self.root
        )
        return receive_buffer

    def _forward(
        self,
        dfobj,
        dm,
        *,
        hermi,
        with_j,
        with_k,
        direct_scf_tol,
    ):
        self._require_root()
        if hermi != 1:
            raise NotImplementedError("MPI DF-J/K currently requires hermi=1")

        dm_fast = _df_jk_opt._tag_dm_for_fast_exchange(dfobj, dm)
        dm_host = _require_real_rhf_array("density", dm_fast)
        mo_coeff = getattr(dm_fast, "mo_coeff", None)
        mo_occ = getattr(dm_fast, "mo_occ", None)
        payload = {
            "op": _OP_FORWARD,
            "dm": dm_host,
            "mo_coeff": None if mo_coeff is None else numpy.asarray(mo_coeff),
            "mo_occ": None if mo_occ is None else numpy.asarray(mo_occ),
            "hermi": int(hermi),
            "with_j": bool(with_j),
            "with_k": bool(with_k),
            "direct_scf_tol": float(direct_scf_tol),
            "metadata": self._root_metadata(dfobj),
        }
        if self.size > 1:
            self.comm.bcast(payload, root=self.root)
        try:
            local_result = numpy.ascontiguousarray(
                numpy.stack(self._execute_forward(dfobj, payload))
            )
            reduced = (
                numpy.empty_like(local_result) if self.size > 1 else None
            )
            local_error = None
        except BaseException:  # synchronize failures before any reduction
            local_result = None
            reduced = None
            local_error = _remote_error(_OP_FORWARD, self.rank)
        errors = self._collective_errors(local_error) if self.size > 1 else (
            [] if local_error is None else [local_error]
        )
        self._raise_remote_errors(errors)
        if self.size == 1:
            return local_result[0], local_result[1]
        self._reduce_sum_array(local_result, reduced)
        return reduced[0], reduced[1]

    def _execute_forward(self, dfobj, payload):
        return local_jk(
            dfobj,
            payload["dm"],
            rank=self.rank,
            size=self.size,
            hermi=payload["hermi"],
            with_j=payload["with_j"],
            with_k=payload["with_k"],
            direct_scf_tol=payload["direct_scf_tol"],
            mo_coeff=payload["mo_coeff"],
            mo_occ=payload["mo_occ"],
            expected_metadata=payload["metadata"],
            local_metadata=self._metadata(dfobj),
        )

    def _density_vjp(
        self,
        dfobj,
        dm,
        vj_bar,
        vk_bar,
        *,
        hermi,
        with_j,
        with_k,
    ):
        self._require_root()
        if hermi != 1:
            raise NotImplementedError("MPI DF-J/K currently requires hermi=1")
        dm_host = _require_real_rhf_array("density", dm)
        vj_bar = _require_real_rhf_array("J cotangent", vj_bar)
        vk_bar = _require_real_rhf_array("K cotangent", vk_bar)
        lowrank_factors = _df_jk_opt._implicit_lowrank_exchange_factors(
            dfobj,
            vj_bar.reshape(-1, dm_host.shape[-1], dm_host.shape[-1]),
            vk_bar.reshape(-1, dm_host.shape[-1], dm_host.shape[-1]),
            hermi,
            with_j,
            with_k,
        )
        payload = {
            "op": _OP_DENSITY_VJP,
            "dm_shape": tuple(dm_host.shape),
            "vj_bar": vj_bar,
            "vk_bar": vk_bar,
            "hermi": int(hermi),
            "with_j": bool(with_j),
            "with_k": bool(with_k),
            "lowrank_factors": lowrank_factors,
            "metadata": self._root_metadata(dfobj),
        }
        if self.size > 1:
            self.comm.bcast(payload, root=self.root)
        try:
            local_result = numpy.ascontiguousarray(
                self._execute_density_vjp(dfobj, payload)
            )
            reduced = (
                numpy.empty_like(local_result) if self.size > 1 else None
            )
            local_error = None
        except BaseException:  # synchronize failures before the reduction
            local_result = None
            reduced = None
            local_error = _remote_error(_OP_DENSITY_VJP, self.rank)
        errors = self._collective_errors(local_error) if self.size > 1 else (
            [] if local_error is None else [local_error]
        )
        self._raise_remote_errors(errors)
        if self.size == 1:
            return local_result
        self._reduce_sum_array(local_result, reduced)
        return reduced

    def _execute_density_vjp(self, dfobj, payload):
        return local_density_vjp(
            dfobj,
            payload["dm_shape"],
            payload["vj_bar"],
            payload["vk_bar"],
            rank=self.rank,
            size=self.size,
            hermi=payload["hermi"],
            with_j=payload["with_j"],
            with_k=payload["with_k"],
            lowrank_factors=payload["lowrank_factors"],
            expected_metadata=payload["metadata"],
            local_metadata=self._metadata(dfobj),
        )

    def _execute_coordinate_vjp_block(
        self, dfobj, payload, shls_slice, ints_bar
    ):
        """Evaluate one independent three-centre coordinate pullback."""

        return _cderi_vjp._int3c_coordinate_vjp_block(
            dfobj.mol,
            dfobj.auxmol,
            ints_bar,
            int3c=payload["int3c"],
            shls_slice=tuple(shls_slice),
            naoaux=payload["metadata"][1],
        )

    def _coordinate_block_owner(self, active_block_index):
        # The integral derivative dominates once domains become appreciable,
        # so include root in the cyclic schedule as well as the workers.  Start
        # at the first worker: this both launches remote work promptly and
        # ensures even a single nonzero block exercises the distributed path.
        # The root joins on the last slot of each cycle.  Use the active
        # (nonzero) block index so sparse cotangents remain balanced.
        return (self.root + 1 + int(active_block_index)) % self.size

    def _dispatch_coordinate_vjp_block(
        self, dfobj, payload, block_index, active_block_index,
        shls_slice, ints_bar
    ):
        """Run a root-owned block locally or stream it to its MPI owner."""

        owner = self._coordinate_block_owner(active_block_index)
        if owner == self.root:
            return self._execute_coordinate_vjp_block(
                dfobj, payload, shls_slice, ints_bar
            )

        ints_bar = _require_real_rhf_array(
            "three-centre integral cotangent", ints_bar
        )
        if ints_bar.flags.c_contiguous:
            storage_order = "C"
        elif ints_bar.flags.f_contiguous:
            # scipy.linalg.solve_triangular normally returns this layout.
            # MPI can transmit it directly, avoiding another potentially
            # hundreds-of-MiB block copy on root.
            storage_order = "F"
        else:
            ints_bar = numpy.ascontiguousarray(ints_bar)
            storage_order = "C"
        header = {
            "stop": False,
            "block_index": int(block_index),
            "shls_slice": tuple(int(value) for value in shls_slice),
            "shape": tuple(int(value) for value in ints_bar.shape),
            "order": storage_order,
        }
        self.comm.send(
            header, dest=owner, tag=_COORDINATE_HEADER_TAG
        )
        ready = self.comm.recv(
            source=owner, tag=_COORDINATE_READY_TAG
        )
        if int(ready.get("block_index", -1)) != int(block_index):
            raise RuntimeError(
                "MPI DF coordinate worker acknowledged the wrong block: "
                f"sent={block_index}, received={ready!r}"
            )
        if not ready.get("ready"):
            # The worker records the local traceback, drains subsequent
            # headers, and reports the failure in the collective error step.
            return None, None
        self.comm.Send(
            ints_bar, dest=owner, tag=_COORDINATE_DATA_TAG
        )
        return None, None

    def _finish_coordinate_dispatch(self):
        if self.size <= 1:
            return
        message = {"stop": True}
        for worker_rank in range(self.size):
            if worker_rank != self.root:
                self.comm.send(
                    message,
                    dest=worker_rank,
                    tag=_COORDINATE_HEADER_TAG,
                )

    def _coordinate_vjp(self, dfobj, cderi_bar_block_fn):
        """Distribute the three-centre part of one full CDERI coordinate VJP."""

        self._require_root()
        metadata = self._root_metadata(dfobj)
        payload = {
            "op": _OP_COORDINATE_VJP,
            "metadata": metadata,
            "int3c": dfobj.mol._add_suffix("int3c2e"),
        }
        if self.size > 1:
            self.comm.bcast(payload, root=self.root)

        local_result = None
        reduced = None
        try:
            # Request at least two shell blocks per rank when shell structure
            # permits it.  This improves load balance without increasing the
            # peak block memory used by the serial implementation.
            max_pair_block = None
            if self.size > 1:
                npair = int(metadata[2])
                max_pair_block = max(
                    8, (npair + 2 * self.size - 1) // (2 * self.size)
                )

            active_block_count = 0

            def block_vjp(block_index, shls_slice, ints_bar):
                nonlocal active_block_count
                active_block_index = active_block_count
                active_block_count += 1
                return self._dispatch_coordinate_vjp_block(
                    dfobj,
                    payload,
                    block_index,
                    active_block_index,
                    shls_slice,
                    ints_bar,
                )

            mol_bar, auxmol_bar = (
                _df_jk_opt._cderi_mol_aux_vjp_from_block_fn(
                    dfobj,
                    cderi_bar_block_fn,
                    int3c_block_vjp=block_vjp,
                    max_pair_block=max_pair_block,
                )
            )
            local_result = _pack_coordinate_bars(
                mol_bar, auxmol_bar, dfobj.mol, dfobj.auxmol
            )
            reduced = (
                numpy.empty_like(local_result) if self.size > 1 else None
            )
            local_error = None
        except BaseException:
            local_error = _remote_error(_OP_COORDINATE_VJP, self.rank)
        finally:
            # Workers receive until this explicit sentinel, including when
            # root's block loop or metric pullback fails.
            self._finish_coordinate_dispatch()

        errors = self._collective_errors(local_error) if self.size > 1 else (
            [] if local_error is None else [local_error]
        )
        self._raise_remote_errors(errors)
        if self.size == 1:
            return local_result[0], local_result[1]
        self._reduce_sum_array(local_result, reduced)
        return reduced[0], reduced[1]

    def _execute_coordinate_vjp_worker(self, dfobj, payload):
        """Receive assigned integral cotangent blocks until root's sentinel."""

        mol_bar = None
        auxmol_bar = None
        first_error = None
        try:
            local_metadata = self._metadata(dfobj)
            if tuple(local_metadata) != tuple(payload["metadata"]):
                raise RuntimeError(
                    "MPI ranks have incompatible DF sources during the "
                    "coordinate VJP: "
                    f"local={local_metadata}, root={payload['metadata']}"
                )
            if dfobj.auxmol is None:
                raise RuntimeError(
                    "MPI DF coordinate VJP requires an auxiliary molecule "
                    "on every worker"
                )
        except BaseException:
            # Still drain headers and acknowledge them as rejected; otherwise
            # root could block before the collective error report.
            first_error = traceback.format_exc()
        while True:
            header = self.comm.recv(
                source=self.root, tag=_COORDINATE_HEADER_TAG
            )
            if header.get("stop"):
                break
            block_index = int(header["block_index"])
            if first_error is not None:
                self.comm.send(
                    {"ready": False, "block_index": block_index},
                    dest=self.root,
                    tag=_COORDINATE_READY_TAG,
                )
                continue
            try:
                ints_bar = numpy.empty(
                    tuple(header["shape"]),
                    dtype=numpy.float64,
                    order=header["order"],
                )
            except BaseException:
                first_error = traceback.format_exc()
                self.comm.send(
                    {"ready": False, "block_index": block_index},
                    dest=self.root,
                    tag=_COORDINATE_READY_TAG,
                )
                continue
            self.comm.send(
                {"ready": True, "block_index": block_index},
                dest=self.root,
                tag=_COORDINATE_READY_TAG,
            )
            self.comm.Recv(
                ints_bar, source=self.root, tag=_COORDINATE_DATA_TAG
            )
            try:
                mol_blk_bar, auxmol_blk_bar = (
                    self._execute_coordinate_vjp_block(
                        dfobj,
                        payload,
                        header["shls_slice"],
                        ints_bar,
                    )
                )
                mol_bar = _cderi_vjp._tree_add(mol_bar, mol_blk_bar)
                auxmol_bar = _cderi_vjp._tree_add(
                    auxmol_bar, auxmol_blk_bar
                )
            except BaseException:
                # Drain the remaining messages before reporting the error so
                # root cannot become stranded in a blocking Send.
                first_error = traceback.format_exc()

        if first_error is not None:
            raise RuntimeError(
                "MPI DF coordinate block pullback failed:\n" + first_error
            )
        return _pack_coordinate_bars(
            mol_bar, auxmol_bar, dfobj.mol, dfobj.auxmol
        )

    def pause_workers(self):
        """Return workers from the current serve call without closing."""

        self._require_root()
        if self.size > 1:
            self.comm.bcast({"op": _OP_PAUSE}, root=self.root)

    def stop_workers(self):
        """Return workers from serve and permanently close this executor."""

        if not self.is_root:
            raise RuntimeError("stop_workers is root-only")
        if self._closed:
            return
        if self.size > 1:
            self.comm.bcast({"op": _OP_STOP}, root=self.root)
        self.close_local()

    def close_local(self):
        """Release process-local state when no rank is inside ``serve``.

        This method performs no MPI communication.  It is intended for a
        synchronized failure between the paused forward and reverse service
        sessions; normal completion must use :meth:`stop_workers`.
        """

        self._closed = True
        self._metadata_cache.clear()
        _EXECUTORS.pop(self.token, None)

    def serve(self, dfobj):
        """Run the blocking DF-J/K service on a non-root rank.

        Returns :class:`ServiceExit.PAUSED` or
        :class:`ServiceExit.STOPPED`.  The same DF object may be supplied to a
        later call after a pause.
        """

        if self.is_root:
            if self.size == 1:
                return ServiceExit.STOPPED if self._closed else ServiceExit.PAUSED
            raise RuntimeError("root must execute the SCF program, not serve()")
        if self._closed:
            return ServiceExit.STOPPED

        while True:
            payload = self.comm.bcast(None, root=self.root)
            operation = payload.get("op")
            if operation == _OP_PAUSE:
                return ServiceExit.PAUSED
            if operation == _OP_STOP:
                self.close_local()
                return ServiceExit.STOPPED
            try:
                if operation == _OP_FORWARD:
                    local_result = numpy.ascontiguousarray(
                        numpy.stack(self._execute_forward(dfobj, payload))
                    )
                elif operation == _OP_DENSITY_VJP:
                    local_result = numpy.ascontiguousarray(
                        self._execute_density_vjp(dfobj, payload)
                    )
                elif operation == _OP_COORDINATE_VJP:
                    local_result = numpy.ascontiguousarray(
                        self._execute_coordinate_vjp_worker(dfobj, payload)
                    )
                else:
                    raise RuntimeError(
                        f"unknown MPI DF-J/K service operation {operation!r}"
                    )
                local_error = None
            except BaseException:  # report to root without stranding ranks
                local_result = None
                local_error = _remote_error(operation, self.rank)

            errors = self._collective_errors(local_error)
            if errors:
                # Root raises and should release us through root_session's
                # exception path.  Wait for that PAUSE/STOP command.
                continue
            self._reduce_sum_array(local_result)
