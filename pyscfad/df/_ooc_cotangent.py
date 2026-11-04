"""Explicit first-order OOC geometry accumulation across staged reverses.

Within ``geometry_scope(df_input)``, call the complete SCF pullback, optionally
preceded by MP2's pullback, and pass the DF-input cotangent to ``scope.finish``
before its final DF attachment reverse. The scope
asserts that geometry cotangents reach that input unchanged. Intermediate
rescaling or interception is unsupported; scale the output seed instead.
Ordinary unscoped VJPs retain their immediate geometry response.

Compact S/G/H seeds persist until the implicit-response seed arrives. MP2's
raw transformed factors and seeds are owned in bounded RAM or HDF5 scratch.
Exactly equal S matrices may share panel preparation; different factors share
only the final derivative-integral pass. No derivative tensor is retained.
"""
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
import hashlib
import itertools
import os
from pathlib import Path
import time

import jax
import numpy
from pyscf import lib
from pyscf.gto.mole import AS_RINV_ORIG_ATOM

from pyscfad.lib import _cotangent_buffer

_scope = ContextVar('pyscfad_ooc_geometry_scope', default=None)
_capture = ContextVar('pyscfad_ooc_geometry_producers', default=False)
_producer_ids = itertools.count()


@contextmanager
def capture_producers(enabled=True):
    """Mark a complete traced SCF build's DF coordinate dependency boundary."""
    enabled = enabled and os.environ.get('PYSCFAD_STC_COMBINE_SCF_DF', '1').lower() in (
        '1', 'true', 'yes', 'on')
    token = _capture.set(enabled)
    try:
        yield
    finally:
        _capture.reset(token)


@jax.custom_vjp
def _geometry_producer(obj):
    return obj


def _producer_forward(obj):
    return obj, obj


def _producer_backward(obj, bar):
    scope = _scope.get()
    if (scope is not None and scope.automatic_producer and not scope.finished
            and getattr(obj, '_ooc_geometry_producer', None) is not None
            and obj._ooc_geometry_producer == getattr(scope.df_input, '_ooc_geometry_producer', None)):
        from pyscfad.tools import resource_profile
        with resource_profile.section('df.shared_scf_geometry'):
            bar = scope.finish(bar)
    return (bar,)


_geometry_producer.defvjp(_producer_forward, _producer_backward)


def register_producer(obj):
    """Insert an identity VJP before consumers, preserving both geometry maps.

    The final summed DF coordinate seed rejoins the original mol/auxmol
    construction here, including nonidentity auxiliary coordinate mappings.
    Outside the explicitly owned SCF trace, ordinary DF objects are unchanged.
    """
    if (not _capture.get() or not obj._has_outcore_cderi_placeholder()
            or getattr(obj, '_ooc_geometry_producer', None) is not None
            or not any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(obj))):
        return
    # PySCF's pickle/copy protocol deliberately drops auxmol. A pytree copy
    # preserves every differentiable leaf and this producer's source token.
    original = jax.tree_util.tree_map(lambda x: x, obj)
    original._ooc_geometry_producer = next(_producer_ids)
    linked = _geometry_producer(original)
    obj.mol, obj.auxmol = linked.mol, linked.auxmol
    obj._ooc_geometry_producer = original._ooc_geometry_producer


@contextmanager
def producer_scope(obj):
    """Own automatic geometry accumulation only when its producer is live."""
    if (_scope.get() is not None or obj is None
            or getattr(obj, '_ooc_geometry_producer', None) is None):
        yield None
    else:
        with geometry_scope(obj) as scope:
            scope.automatic_producer = True
            yield scope


class _RawMOSeed:
    """Own raw transformed factors and their seed until final DF preparation."""
    def __init__(self, obj, coeff, orbs, output, bar, mask, disk, slab_bytes):
        self.obj, self.orbs, self.mask = obj, tuple(orbs), tuple(mask)
        self.coeff = numpy.array(coeff, copy=True)
        self.resources = ExitStack()
        self.disk = disk
        self.ram_bytes = 0 if disk else output.size*16
        self.closed = False
        try:
            if disk:
                file = self.resources.enter_context(lib.H5TmpFile())
                shape = output.shape
                chunks = (min(shape[0],16), min(shape[1],1024))
                self.output = file.create_dataset('primal',shape,dtype='f8',chunks=chunks)
                self.bar = file.create_dataset('cotangent',shape,dtype='f8',chunks=chunks)
                width = max(1, int(slab_bytes/(2*8*shape[0])))
                for begin in range(0,shape[1],width):
                    end = min(begin+width,shape[1])
                    self.output[:,begin:end] = numpy.asarray(output[:,begin:end])
                    self.bar[:,begin:end] = numpy.asarray(bar[:,begin:end])
            else:
                self.output = numpy.array(output,copy=True)
                self.bar = numpy.array(bar,copy=True)
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self.closed:
            self.output = self.bar = self.coeff = None
            self.resources.close()
            self.closed = True


def _retained_panel_bytes(contribution, with_k=True):
    if contribution.disk_panels:
        return 0
    if hasattr(contribution,'retained_panel_bytes'):
        return contribution.retained_panel_bytes
    columns = getattr(contribution,'compact_panel_columns',
                      contribution.nao*contribution.rank if with_k else 0)
    return contribution.naux*columns*8


def _array_identity(value):
    if isinstance(value, jax.core.Tracer):
        return None
    if isinstance(value, numpy.ndarray):
        pointer = value.__array_interface__['data'][0]
        layout = tuple(value.strides)
    elif hasattr(value, 'unsafe_buffer_pointer'):
        pointer = value.unsafe_buffer_pointer()
        layout = None
    else:
        return None
    return pointer, tuple(value.shape), str(value.dtype), layout


def _source_key(obj):
    """Match one logical producer and coordinate dependency, not file equality."""
    if (not obj._has_outcore_cderi_placeholder()
            or getattr(obj, '_fast_exchange_cache_token', None) is None):
        return None
    source = obj._get_cderi_source()
    if not isinstance(source, (str, os.PathLike)):
        source = getattr(source, 'name', None)
    if source is None:
        return None
    path = Path(source).resolve()
    try:
        stat = path.stat()
    except OSError:
        return None
    molecules = []
    for mol in (obj.mol, obj.auxmol):
        if mol is None or len(jax.tree_util.tree_leaves(mol)) != 1:
            return None
        identity = _array_identity(mol.coords)
        if identity is None or str(mol.coords.dtype) != 'float64':
            return None
        digest = hashlib.sha256()
        for name in ('_atm', '_bas'):
            digest.update(numpy.asarray(getattr(mol, name)).tobytes())
        # Nuclear-attraction reverse changes this scratch atom selector.
        # It does not enter int3c2e/int2c2e. Preserve all other controls,
        # particularly range omega, as well as basis and coordinate data.
        digest.update(numpy.asarray(mol._env[:AS_RINV_ORIG_ATOM]).tobytes())
        digest.update(numpy.asarray(mol._env[AS_RINV_ORIG_ATOM+1:]).tobytes())
        digest.update(numpy.asarray(mol.coords).tobytes())
        molecules.append((identity, digest.digest(), bool(mol.cart)))
    return (obj._fast_exchange_cache_token, str(path), stat.st_dev, stat.st_ino,
            stat.st_size, stat.st_mtime_ns, tuple(molecules))


class GeometryScope:
    def __init__(self, df_input, enabled, coalesce, force_disk):
        self.df_input = df_input
        self.enabled = enabled
        self.coalesce = coalesce
        self.force_disk = force_disk
        self.key = _source_key(df_input) if enabled else None
        self.pending = []
        self.mo_pending = []
        self.raw_ram_bytes = 0
        self.finished = False
        self.automatic_producer = False
        self.low = self.metric_pullback = None
        self.stats = dict(enabled=enabled, consumers=0, seed_groups=0,
                          geometry_passes=0, disk_contributions=0,
                          compact_contributions=0, compact_exchange_ranks=[],
                          mo_consumers=0, mo_raw_disk_seeds=0,
                          mo_raw_peak_ram_bytes=0, mo_prepare_seconds=0.,
                          factor_response_seconds=0., prepare_seconds=0.,
                          contract_seconds=0., metric_builds=0, cholesky_builds=0)

    def metric_factor(self):
        if self.low is None:
            from jax import scipy as jsp
            def factor(aux):
                return jsp.linalg.cholesky(
                    aux.intor(self.df_input.mol._add_suffix('int2c2e'), hermi=1), lower=True)
            low, self.metric_pullback = jax.vjp(factor, self.df_input.auxmol)
            self.low = numpy.asarray(low)
            self.stats['metric_builds'] += 1
            self.stats['cholesky_builds'] += 1
        return self.low

    def finish(self, df_bar):
        """Add deferred geometry to the original DF-input cotangent exactly once."""
        if self.finished:
            raise RuntimeError('OOC geometry scope already finalized')
        if (self.pending or self.mo_pending) and _source_key(self.df_input) != self.key:
            raise RuntimeError('OOC source changed before geometry finalization')
        from . import _df_jk_factorized as factorized
        groups = []
        for obj, s, g, h, with_j, with_k, mask in self.pending:
            match = next((x for x in groups if self.coalesce and x['mask'] == mask
                          and numpy.array_equal(x['s'], s)), None)
            if match is None:
                groups.append(dict(obj=obj, s=s, g=g.copy(), h=h.copy(),
                                   with_j=with_j, with_k=with_k, mask=mask))
            else:
                match['g'] += g
                match['h'] += h
                match['with_j'] |= with_j
                match['with_k'] |= with_k
        self.stats['seed_groups'] = len(groups)
        # Reserve at most one shared panel budget across retained contributions.
        budget = factorized._memory_mb('PYSCFAD_FACTOR_PANEL_MEMORY_MB',
                                       factorized._FACTOR_PANEL_MEMORY_MB)*1024**2
        answer = jax.tree_util.tree_map(lambda x: x, df_bar)
        with ExitStack() as owned:
            masked = {}
            retained = 0
            started = time.perf_counter()
            for raw in self.mo_pending:
                from . import _cderi_vjp
                z_bytes = raw.output.size*8
                spill = self.force_disk or self.raw_ram_bytes+retained+z_bytes>budget
                contribution = _cderi_vjp._prepare_outcore_mo_geometry(
                    raw.obj,raw.coeff,raw.bar,raw.orbs,raw.output,force_disk=spill)
                owned.callback(contribution.close)
                masked.setdefault(raw.mask,[]).append(contribution)
                retained += _retained_panel_bytes(contribution)
                self.stats['disk_contributions'] += int(contribution.disk_panels)
                self.raw_ram_bytes -= raw.ram_bytes
                raw.close()
            self.mo_pending.clear()
            self.stats['mo_prepare_seconds'] = time.perf_counter()-started
            for x in groups:
                panel_bytes = x['obj'].get_naoaux()*x['s'].size*8 if x['with_k'] else 0
                # Only solved U survives each preparation. Compact seeds can
                # retain much less than the conservative allocation estimate.
                spill = self.force_disk or retained+2*panel_bytes>budget
                contribution = None
                if os.environ.get('PYSCFAD_DF_OOC_COMPACT', '0').lower() in (
                        '1', 'true', 'yes', 'on'):
                    from . import _df_jk_compact
                    contribution = _df_jk_compact.prepare_compact(
                        x['with_j'], x['with_k'], (x['obj'], x['s']),
                        (x['g'], x['h']), force_disk=spill)
                    if contribution is not None:
                        self.stats['compact_contributions'] += 1
                        self.stats['compact_exchange_ranks'].append(
                            contribution.compact_exchange_rank)
                if contribution is None:
                    contribution, _ = factorized._prepare_backward(
                        x['with_j'], x['with_k'], False, (x['obj'], x['s']),
                        (x['g'], x['h']), force_disk=spill)
                owned.callback(contribution.close)
                masked.setdefault(x['mask'], []).append(contribution)
                retained += _retained_panel_bytes(contribution,x['with_k'])
                self.stats['disk_contributions'] += int(contribution.disk_panels)
            self.stats['prepare_seconds'] = time.perf_counter()-started
            started = time.perf_counter()
            for mask, contributions in masked.items():
                bar = factorized.contract_geometry(contributions)
                if mask[0]:
                    answer.mol = jax.tree_util.tree_map(lambda x,y:x+y,answer.mol,bar.mol)
                if mask[1]:
                    answer.auxmol = jax.tree_util.tree_map(lambda x,y:x+y,answer.auxmol,bar.auxmol)
                self.stats['geometry_passes'] += 1
            self.stats['contract_seconds'] = time.perf_counter()-started
        self.pending.clear()
        self.finished = True
        return answer


@contextmanager
def geometry_scope(df_input, *, enabled=True, coalesce=True, force_disk=False):
    """Own pending geometry for one complete eager first-order reverse chain."""
    if _scope.get() is not None:
        raise RuntimeError('nested OOC geometry scopes are unsupported')
    scope = GeometryScope(df_input, enabled, coalesce, force_disk)
    token = _scope.set(scope)
    try:
        yield scope
        if scope.pending or scope.mo_pending:
            raise RuntimeError('unconsumed OOC geometry: call scope.finish on the DF input cotangent')
    finally:
        scope.pending.clear()
        for seed in scope.mo_pending:
            seed.close()
        scope.mo_pending.clear()
        scope.raw_ram_bytes = 0
        scope.low = scope.metric_pullback = None
        _scope.reset(token)


def metric_scope(obj):
    scope = _scope.get()
    if (scope is not None and scope.enabled and not scope.finished
            and scope.key is not None and _source_key(obj) == scope.key):
        return scope
    return None


def defer_backward(with_j, with_k, factor_response, residual, cotangents,
                   *, geometry_activity=(True, True)):
    """Return immediate bars, or None when the ordinary path must be used."""
    scope = _scope.get()
    if (scope is None or not scope.enabled or scope.finished
            or _cotangent_buffer._suspended.get() or not any(geometry_activity)):
        return None
    obj, factors = residual[:2]
    if scope.key is None or _source_key(obj) != scope.key:
        return None
    if any(isinstance(x, jax.core.Tracer) for x in
           jax.tree_util.tree_leaves((factors, cotangents))):
        return None
    s = numpy.asarray(factors)
    g, h = (numpy.asarray(x) for x in cotangents)
    if (any(x.dtype != numpy.float64 for x in (s,g,h))
            or s.ndim != 2 or g.shape != (s.shape[0],)*2 or h.shape != g.shape):
        return None
    g = (g+g.T)*.5 if with_j else numpy.zeros_like(g)
    h = (h+h.T)*.5 if with_k else numpy.zeros_like(h)
    from . import _df_jk_factorized as factorized
    started = time.perf_counter()
    sbar = (factorized.factor_pullback(obj,s,g,h,with_j,with_k)
            if factor_response else None)
    scope.stats['factor_response_seconds'] += time.perf_counter()-started
    scope.pending.append((obj,s.copy(),g,h,with_j,with_k,tuple(geometry_activity)))
    scope.stats['consumers'] += 1
    return jax.tree_util.tree_map(numpy.zeros_like,obj), sbar


def defer_mo_geometry(source_token,cderi_source,mol,auxmol,mo_coeff,orbs_slice,
                      saved_output,ybar,*,geometry_activity=(True,True)):
    """Own an MP2 AO-to-MO geometry seed; leave its coefficient bar immediate."""
    scope = _scope.get()
    if (scope is None or not scope.enabled or scope.finished or scope.key is None
            or _cotangent_buffer._suspended.get() or not any(geometry_activity)
            or source_token != scope.df_input._fast_exchange_cache_token):
        return False
    if any(isinstance(x,jax.core.Tracer) for x in
           jax.tree_util.tree_leaves((mol,auxmol,mo_coeff,saved_output,ybar))):
        return False
    if (any(str(x.dtype)!='float64' for x in (mo_coeff,saved_output,ybar))
            or len(mo_coeff.shape)!=2 or mo_coeff.shape[0]!=mol.nao
            or len(saved_output.shape)!=2 or ybar.shape!=saved_output.shape
            or len(orbs_slice)!=4):
        return False
    k0,k1,l0,l1 = orbs_slice
    if (not 0<=k0<k1<=mo_coeff.shape[1]
            or not 0<=l0<l1<=mo_coeff.shape[1]
            or saved_output.shape!=(auxmol.nao,(k1-k0)*(l1-l0))):
        return False
    obj = jax.tree_util.tree_map(lambda x:x,scope.df_input)
    obj.mol,obj.auxmol = mol,auxmol
    obj._cderi_to_save = cderi_source
    if _source_key(obj)!=scope.key:
        return False
    from . import _df_jk_factorized as factorized
    budget = factorized._memory_mb('PYSCFAD_FACTOR_PANEL_MEMORY_MB',
                                  factorized._FACTOR_PANEL_MEMORY_MB)*1024**2
    workspace = factorized._memory_mb('PYSCFAD_SOLVE_MEMORY_MB',
                                     factorized._SOLVE_MEMORY_MB)*1024**2
    disk = scope.force_disk or scope.raw_ram_bytes+saved_output.size*16>budget
    seed = _RawMOSeed(obj,mo_coeff,orbs_slice,saved_output,ybar,
                      geometry_activity,disk,workspace)
    scope.mo_pending.append(seed)
    scope.raw_ram_bytes += seed.ram_bytes
    scope.stats['mo_consumers'] += 1
    scope.stats['mo_raw_disk_seeds'] += int(disk)
    scope.stats['mo_raw_peak_ram_bytes'] = max(scope.stats['mo_raw_peak_ram_bytes'],scope.raw_ram_bytes)
    return True
