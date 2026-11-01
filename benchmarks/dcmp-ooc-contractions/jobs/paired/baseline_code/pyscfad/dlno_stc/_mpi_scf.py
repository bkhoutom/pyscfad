"""Paused DF J/K service around shared STC reference and response phases.

The executor remains registered while owners perform independent correlation
work. Its hook is active only during root initialization and final response.
Workers retain a separate DF object whose source survives correlation setup.
"""

from functools import partial
import hashlib
import time
from types import SimpleNamespace

import jax
import numpy

_SOURCE_CHUNK_BYTES = 8 * 1024**2


def _source(dfobj):
    getter = getattr(dfobj, '_get_cderi_source', None)
    return getter() if callable(getter) else getattr(dfobj, '_cderi', None)


def _source_fingerprint(dfobj):
    """Hash logical packed factors in bounded row blocks, including disk data."""
    from pyscfad.df import addons

    source = _source(dfobj)
    if source is None:
        raise ValueError('MPI DF-J/K requires a built density-fitting source')
    digest = hashlib.sha256()
    with addons.load(source, 'j3c') as factors:
        shape, dtype = tuple(factors.shape), numpy.dtype(factors.dtype)
        nao = int(dfobj.mol.nao)
        if (len(shape) != 2 or shape[1] != nao * (nao + 1) // 2
                or dtype != numpy.dtype(numpy.float64)):
            raise ValueError('MPI DF-J/K requires real float64 packed s2ij CDERI')
        rows = max(1, _SOURCE_CHUNK_BYTES // max(1, shape[1] * dtype.itemsize))
        for start in range(0, shape[0], rows):
            block = numpy.ascontiguousarray(factors[start:start + rows])
            if not numpy.isfinite(block).all():
                raise ValueError('MPI DF-J/K source contains nonfinite factors')
            digest.update(memoryview(block).cast('B'))
    return shape, dtype.str, digest.hexdigest()


def _packed_source(dfobj):
    """Keep packed sources lazy and normalize full symmetric factors once."""
    from pyscfad.df import addons

    source = _source(dfobj)
    if source is None:
        raise ValueError('MPI DF-J/K requires built factors')
    nao = int(dfobj.mol.nao)
    npair = nao * (nao + 1) // 2
    with addons.load(source, 'j3c') as factors:
        shape, dtype = tuple(factors.shape), numpy.dtype(factors.dtype)
        if len(shape) != 2 or dtype != numpy.dtype(numpy.float64):
            raise ValueError('MPI DF-J/K requires real float64 two-dimensional factors')
        if shape[1] == npair:
            return source
        if shape[1] != nao * nao:
            raise ValueError('MPI DF-J/K source has an unsupported AO-pair dimension')
        packed = numpy.empty((shape[0], npair), dtype=dtype)
        lower = numpy.tril_indices(nao)
        rows = max(1, _SOURCE_CHUNK_BYTES // max(1, shape[1] * dtype.itemsize))
        for start in range(0, shape[0], rows):
            block = numpy.asarray(factors[start:start + rows]).reshape(-1, nao, nao)
            if not numpy.allclose(block, block.swapaxes(1, 2), rtol=1e-10, atol=1e-11):
                raise ValueError('full DF factors must be symmetric in their AO indices')
            packed[start:start + len(block)] = block[:, lower[0], lower[1]]
    return packed


def _registered_source(executor_token, df_token):
    from pyscfad.df.mpi_df_jk import _executor_from_token
    return _executor_from_token(executor_token).sources[df_token]


def _copy_df(dfobj):
    # PySCF's pickle hooks intentionally omit caches including auxmol. An
    # ordinary copy.copy therefore changes the differentiable subtree.
    duplicate = object.__new__(type(dfobj))
    duplicate.__dict__.update(dfobj.__dict__)
    return duplicate


def _source_view(dfobj, getter):
    """Keep molecular leaves live while replacing the factor AD path."""
    view = _copy_df(dfobj)
    view._cderi = numpy.zeros((0, 0), dtype=numpy.float64)
    view._get_cderi_source = getter
    return view


@partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4, 5, 6))
def _service_get_jk(token, dfobj, dm, hermi=1, with_j=True, with_k=True,
                    direct_scf_tol=1e-13):
    from pyscfad.df.mpi_df_jk import _executor_from_token
    executor = _executor_from_token(token)
    view = executor.source_view(dfobj)
    return executor._forward(view, dm, hermi=hermi, with_j=with_j,
                             with_k=with_k, direct_scf_tol=direct_scf_tol)


def _service_get_jk_fwd(token, dfobj, dm, hermi, with_j, with_k, direct_scf_tol):
    from pyscfad.df.mpi_df_jk import _executor_from_token
    from pyscfad.df import _df_jk_opt
    dfobj, dm, activity = _df_jk_opt._unwrap_jk_primals(dfobj, dm)
    # A service view closes over physical factors while keeping geometry live.
    activity = (activity[0], False, activity[2], activity[3])
    executor = _executor_from_token(token)
    view = executor.source_view(dfobj)
    result = executor._forward(view, dm, hermi=hermi, with_j=with_j,
                               with_k=with_k, direct_scf_tol=direct_scf_tol)
    return result, (dfobj, view, dm, activity)


def _service_get_jk_bwd(token, hermi, with_j, with_k, direct_scf_tol, residual, bars):
    from pyscfad.df.mpi_df_jk import _mpi_get_jk_bwd
    original, view, dm, activity = residual
    view_bar, dm_bar = _mpi_get_jk_bwd(
        token, hermi, with_j, with_k, direct_scf_tol, (view, dm, activity), bars)
    if view_bar is None:
        return None, dm_bar
    # The native placeholder pullback returns explicit molecular/auxiliary
    # bars through its distributed coordinate service. Map only those live
    # leaves back; the original physical factor leaf receives a zero bar.
    view_bar = jax.tree_util.tree_map(
        lambda primal, bar: (None if primal is None else
                             numpy.zeros_like(primal) if bar is None else bar),
        view, view_bar, is_leaf=lambda value: value is None)
    _, pullback = jax.vjp(partial(_source_view, getter=view._get_cderi_source), original)
    original_bar, = pullback(view_bar)
    return original_bar, dm_bar


_service_get_jk.defvjp(_service_get_jk_fwd, _service_get_jk_bwd, symbolic_zeros=True)


def _verify_sources(comm, dfobj):
    """Check actual molecules and integral contents before the first contraction."""
    from ._mpi import (
        _collective_call, _reference_signature, _validate_molecule_signature,
    )

    def stage():
        return (_reference_signature(SimpleNamespace(mol=dfobj.mol, with_df=dfobj)),
                _source_fingerprint(dfobj))

    local = _collective_call(comm, stage, 'SCF service reference staging')
    references = comm.allgather(local)

    def validate():
        expected, source = references[0]
        for rank, (actual, actual_source) in enumerate(references):
            if actual_source != source:
                raise ValueError(f'density-fitting integral contents differ on rank {rank}')
            if actual['auxbasis'] != expected['auxbasis']:
                raise ValueError(f'SCF service auxiliary basis differs on rank {rank}')
            for name in ('mol', 'df_mol', 'auxmol'):
                left, right = actual[name], expected[name]
                if left is None or right is None:
                    if left is not None or right is not None:
                        raise ValueError(f'SCF service {name} subtree differs on rank {rank}')
                else:
                    _validate_molecule_signature(left, right, f'SCF service rank {rank} {name}')

    _collective_call(comm, validate, 'SCF service reference validation')


def _make_executor(comm):
    # Keep the dependency, including mpi4py initialization, off serial imports.
    from pyscfad.df.mpi_df_jk import MPIDFJKExecutor

    class _VerifiedExecutor(MPIDFJKExecutor):
        def __init__(self):
            super().__init__(comm=comm, root=0)
            self.source_verified = False
            self.sources = {}
            self.source_verification_seconds = 0.0
            self.operations = dict(forward_calls=0, density_vjp_calls=0,
                                   coordinate_vjp_blocks=0)

        def source_view(self, dfobj):
            stable = getattr(dfobj, '_fast_exchange_cache_token', None)
            key = int(stable) if stable is not None else id(dfobj)
            if key not in self.sources:
                self.sources[key] = _packed_source(dfobj)
            return _source_view(dfobj, partial(_registered_source, self.token, key))

        def get_jk(self, dfobj, dm, hermi=1, with_j=True, with_k=True,
                   direct_scf_tol=1e-13):
            self._require_root()
            return _service_get_jk(self.token, dfobj, dm, hermi, with_j, with_k,
                                   direct_scf_tol)

        def close_local(self):
            self.sources.clear()
            super().close_local()

        def _execute_forward(self, dfobj, payload):
            if not self.source_verified:
                start = time.perf_counter()
                _verify_sources(self.comm, dfobj)
                self.source_verification_seconds = time.perf_counter() - start
                self.source_verified = True
            result = super()._execute_forward(dfobj, payload)
            self.operations['forward_calls'] += 1
            return result

        def _execute_density_vjp(self, dfobj, payload):
            result = super()._execute_density_vjp(dfobj, payload)
            self.operations['density_vjp_calls'] += 1
            return result

        def _execute_coordinate_vjp_block(self, dfobj, payload, shls_slice, ints_bar):
            result = super()._execute_coordinate_vjp_block(dfobj, payload, shls_slice, ints_bar)
            self.operations['coordinate_vjp_blocks'] += 1
            return result

    return _VerifiedExecutor()


class DomainSCFService:
    """Own one DF executor across its initial PAUSE and final STOP sessions."""

    def __init__(self, comm, worker_initializer):
        from ._mpi import _collective_call

        self.comm = comm
        self.rank = comm.Get_rank()
        self.executor = self.worker_df = None
        start = time.perf_counter()
        try:
            def setup():
                self.executor = _make_executor(comm)
                if self.rank == 0:
                    return
                from pyscfad.lib._compilation_cache import configure_compilation_cache
                from pyscfad.df.addons import make_auxmol
                configure_compilation_cache()
                if not callable(worker_initializer):
                    raise TypeError('parallel_scf_jk needs an unrun worker MF initializer')
                worker = worker_initializer()
                dfobj = getattr(worker, 'with_df', None)
                if dfobj is None:
                    raise ValueError('SCF service worker needs a density-fitted MF skeleton')
                if int(worker.mol.spin) != 0 or int(worker.mol.nelectron) % 2:
                    raise NotImplementedError('MPI DF-J/K supports closed-shell RHF references only')
                # A separate DF object keeps its CDERI/auxmol alive when the
                # lightweight correlation MF clears or replaces those fields.
                self.worker_df = _copy_df(dfobj)
                if _source(self.worker_df) is None:
                    self.worker_df.build()
                elif self.worker_df.auxmol is None:
                    self.worker_df.auxmol = make_auxmol(self.worker_df.mol, self.worker_df.auxbasis)
                self.worker_df = self.executor.source_view(self.worker_df)
                _source_fingerprint(self.worker_df)

            _collective_call(comm, setup, 'SCF service worker setup')
        except BaseException:
            self.close_local()
            raise
        self.setup_seconds = time.perf_counter() - start

    def _run(self, function, *, final, phase):
        from pyscfad.df.mpi_df_jk import ServiceExit
        from pyscfad.tools.linear_solver import eager_gmres
        from ._mpi import _collective_call

        def execute():
            if self.rank == 0:
                # root_session sends STOP on every root exception before the
                # outer status exchange; workers never wait behind that check.
                with eager_gmres(), self.executor.root_session(final=final):
                    result = function()
                    jax.block_until_ready(result)
                return result
            exit_reason = self.executor.serve(self.worker_df)
            expected = ServiceExit.STOPPED if final else ServiceExit.PAUSED
            if exit_reason is not expected:
                raise RuntimeError(f'{phase} service exited {exit_reason}; expected {expected}')
            return None

        return _collective_call(self.comm, execute, phase)

    def run_initial(self, initializer):
        return self._run(initializer, final=False, phase='initial SCF/common service')

    def run_final(self, finish):
        return self._run(finish, final=True, phase='final common/SCF response service')

    def diagnostics(self):
        return dict(enabled=True, setup_seconds=self.setup_seconds,
                    source_verified=self.executor.source_verified,
                    source_verification_seconds=self.executor.source_verification_seconds,
                    **self.executor.operations)

    def close_local(self):
        """Release tokens locally only when every rank is outside serve()."""
        if self.executor is not None:
            self.executor.close_local()
