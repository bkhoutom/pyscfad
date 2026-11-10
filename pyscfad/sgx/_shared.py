"""Join eager SGX geometry seeds at their common coordinate producer.

Density pullbacks, including GMRES matvecs, remain immediate. Geometry seeds
are held only within an explicitly owned complete reverse, then consumed at
the identity VJP inserted before SCF. That boundary preserves upstream
coordinate mappings and prevents partial/repeated VJPs from losing gradients.
As with shared DF geometry, intermediate cotangents must reach that producer
unchanged: apply weighting to the output seeds, before calling their VJPs.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
import itertools
import time

import jax
import numpy as np
from pyscf.gto.mole import AS_RINV_ORIG_ATOM
from pyscfad.lib._threading import fixed_threads

_capture = ContextVar('pyscfad_sgx_capture_geometry', default=False)
_scope = ContextVar('pyscfad_sgx_shared_geometry', default=None)
_ids = itertools.count()


@contextmanager
def capture_producers(enabled=True):
    token = _capture.set(enabled)
    try:
        yield
    finally:
        _capture.reset(token)


def _source_key(mol):
    # Nuclear-attraction reverse mutates this scratch selector, which has no
    # effect on SGX/DF-J. All other integral controls and coordinates must match.
    return (mol._atm.tobytes(), mol._bas.tobytes(),
            mol._env[:AS_RINV_ORIG_ATOM].tobytes(),
            mol._env[AS_RINV_ORIG_ATOM+1:].tobytes(),
            np.asarray(mol.atom_coords()).tobytes(), mol.cart)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _producer(mol, identity):
    return mol


def _producer_fwd(mol, identity):
    return mol, mol


def _producer_bwd(identity, mol, bar):
    scope = _scope.get()
    if scope is not None and identity == scope.identity and not scope.finished:
        if _source_key(mol) != scope.key:
            raise RuntimeError('SGX geometry producer changed inside the captured SCF')
        from pyscfad.tools import resource_profile
        with resource_profile.section('sgx.shared_scf_geometry'):
            bar = scope.finish(bar)
    return (bar,)


_producer.defvjp(_producer_fwd, _producer_bwd)


def register_producer(mf):
    backend = mf._sgx_backend
    if (not _capture.get() or getattr(backend, '_geometry_producer', None) is not None
            or not isinstance(mf.mol.coords, jax.core.Tracer)):
        return
    backend._geometry_producer = next(_ids)
    mf.mol = _producer(mf.mol, backend._geometry_producer)


class GeometryScope:
    def __init__(self, mf):
        self.backend = mf._sgx_backend
        self.identity = self.backend._geometry_producer
        self.key = _source_key(mf.mol)
        self.pending = []
        self.finished = False
        self.stats = dict(consumers=0, seed_groups=0, j_integral_passes=0,
                          k_integral_passes=0, retained_seed_bytes=0,
                          peak_seed_bytes=0, geometry_seconds=0.,
                          j_geometry_seconds=0., k_geometry_seconds=0.,
                          geometry_batches=[])

    def add(self, mol, dm, jbar, kbar, tol):
        self.stats['consumers'] += 1
        densities = dm.reshape(-1, *dm.shape[-2:])
        seeds = [([None]*len(densities) if b is None else b.reshape(densities.shape))
                 for b in (jbar, kbar)]
        for d, jb, kb in zip(densities, *seeds):
            if not np.any(d) or not any(b is not None and np.any(b) for b in (jb, kb)):
                continue
            group = next((g for g in self.pending
                          if g['tol'] == tol and np.array_equal(g['dm'], d)), None)
            if group is None:
                group = dict(mol=mol, tol=tol, dm=d.copy(), j=None, k=None)
                self.pending.append(group)
                self.stats['retained_seed_bytes'] += d.nbytes
            for name, b in (('j', jb), ('k', kb)):
                if b is None or not np.any(b):
                    continue
                if group[name] is None:
                    group[name] = b.copy()
                    self.stats['retained_seed_bytes'] += b.nbytes
                else:
                    group[name] += b
        self.stats['seed_groups'] = len(self.pending)
        self.stats['peak_seed_bytes'] = max(self.stats['peak_seed_bytes'],
                                            self.stats['retained_seed_bytes'])

    @fixed_threads()
    def finish(self, bar):
        from . import _dfj, _sgx_grad
        started = time.perf_counter()
        result = np.zeros_like(bar.coords)
        for tol in dict.fromkeys(g['tol'] for g in self.pending):
            groups = [g for g in self.pending if g['tol'] == tol]
            obj = self.backend.native(groups[0]['mol'], tol)
            for name, kernel in (('j', _dfj.geometry_many), ('k', _sgx_grad.geometry_many)):
                seeds = [(g['dm'], g[name]) for g in groups
                         if g[name] is not None and np.any(g[name])]
                if seeds:
                    from pyscfad.tools import resource_profile
                    tick = time.perf_counter()
                    with resource_profile.section(f'sgx.shared_{name}_geometry'):
                        result += kernel(obj, seeds, *(() if name == 'j' else (tol,)))
                    self.stats[name+'_geometry_seconds'] += time.perf_counter()-tick
                    self.stats[name+'_integral_passes'] += 1
                    details = (obj._dfj_geometry_stats if name == 'j'
                               else obj._pjs_data._geometry_stats)
                    self.stats['geometry_batches'].append(dict(
                        kind=name, tolerance=tol, **details))
        answer = jax.tree_util.tree_map(lambda x: x, bar)
        answer.coords = answer.coords + result
        self.pending.clear()
        self.stats['retained_seed_bytes'] = 0
        self.stats['geometry_seconds'] = time.perf_counter()-started
        self.finished = True
        return answer


@contextmanager
def producer_scope(mf):
    backend = getattr(mf, '_sgx_backend', None)
    if (_scope.get() is not None or backend is None
            or getattr(backend, '_geometry_producer', None) is None):
        yield None
        return
    scope = GeometryScope(mf)
    token = _scope.set(scope)
    try:
        yield scope
        if scope.pending:
            raise RuntimeError('unconsumed SGX geometry: scope must include the complete SCF pullback')
    finally:
        scope.pending.clear()
        scope.stats['retained_seed_bytes'] = 0
        _scope.reset(token)


def defer(backend, mol, dm, jbar, kbar, tol):
    scope = _scope.get()
    if (scope is None or scope.finished or scope.backend is not backend
            or _source_key(mol) != scope.key):
        return False
    scope.add(mol, dm, jbar, kbar, tol)
    return True
