# Copyright 2026 The PySCFAD Authors
# Licensed under the Apache License, Version 2.0 (the "License");
"""Native resident RCCSD driver and independent PySCF reference implementation."""
from functools import update_wrapper
from types import FunctionType

import numpy as np
from pyscf.lib import logger
from ._native_kernels import _ccsd_kernels
from pyscf import lib
from pyscf.cc import ccsd


def ovvv_singles(theta, vovv):
    """Contract ``icjb,cjba->ia`` without permuting the large ovvv operand.

    The unpacked integral block is C-contiguous in c,j,b,a order. Keeping
    c,j,b as the GEMM reduction order makes its matrix representation a view.
    PySCF's generic sorted b,c,j order instead copies the entire integral
    block. Only the smaller theta tensor needs to be packed here.
    """
    i, c, j, b = theta.shape
    if vovv.ndim != 4 or vovv.shape[:3] != (c, j, b):
        raise ValueError('inconsistent ovvv singles contraction dimensions')
    a = vovv.shape[3]
    left = np.ascontiguousarray(theta).reshape(i, c*j*b)
    right = np.ascontiguousarray(vovv).reshape(c*j*b, a)
    return lib.dot(left, right)


class _ContractionLib:
    """Delegate everything except the real float64 singles contraction."""

    def __getattr__(self, name):
        return getattr(lib, name)

    def einsum(self, subscripts, *operands, **kwargs):
        if (subscripts == 'icjb,cjba->ia' and len(operands) == 2 and not kwargs
                and all(isinstance(a, np.ndarray) and a.ndim == 4
                        and a.dtype == np.float64 for a in operands)):
            return ovvv_singles(*operands)
        return lib.einsum(subscripts, *operands, **kwargs)


def _with_globals(function, **overrides):
    private = FunctionType(function.__code__, dict(function.__globals__, **overrides),
                           function.__name__, function.__defaults__, function.__closure__)
    private.__kwdefaults__ = function.__kwdefaults__
    return update_wrapper(private, function)


_add_ovvv = _with_globals(ccsd._add_ovvv_, lib=_ContractionLib())
_reference_update_amps = _with_globals(ccsd.update_amps, _add_ovvv_=_add_ovvv)


def _ovvv_scratch_bytes(o, v):
    """Conservative high-water bound for the accumulating ovvv contractions."""
    tile = min(32, v)
    return 8*(max(o*o*v*v, tile*o*v*v, tile*o*o*v) + o*v + tile*v + 24)


def _ovov_scratch_bytes(o, v):
    """Two packed operands plus a result, or the larger occupied intermediate."""
    return 8*(max(3*o*o*v*v, o*o*v*v+o**3*v+o*v) + 24)


def _residual_scratch_bytes(o, v):
    """The phases share one allocation; reserve their maximum, not their sum."""
    ladder = (_ccsd_kernels.ladder_scratch_bytes(o, v)
              if hasattr(_ccsd_kernels, 'ladder_scratch_bytes') else 0)
    return max(ladder, _ovvv_scratch_bytes(o, v), _ovov_scratch_bytes(o, v))


def _native_plan(mycc, t1, t2, eris, *, reserve_bytes=None):
    """Return bounded ladder workspace and virtual tile size, or use fallback."""
    if (not getattr(mycc, 'cpp_kernels', False)
            or _ccsd_kernels is None or not hasattr(_ccsd_kernels, 'residual')
            or mycc.cc2 or mycc.direct or getattr(mycc, 'dcsd', False)
            or mycc.async_io or not mycc.incore_complete
            or getattr(eris, 'vvvv', None) is None
            or not isinstance(eris, ccsd._ChemistsERIs)):
        return None
    arrays = (t1, t2, eris.fock, eris.mo_energy, eris.oooo, eris.ovoo,
              eris.oovv, eris.ovvo, eris.ovvv, eris.vvvv)
    if any(not isinstance(a, np.ndarray) or a.dtype != np.float64 for a in arrays):
        return None
    o, v = t1.shape
    if not o or not v:
        return None
    if hasattr(_ccsd_kernels, 'ladder_memory'):
        cached = _cached_workspace(eris)
        existing = cached.memory_info() if cached is not None else None
        return _plan_for(mycc, t1, t2, eris, resident=cached is not None,
                         existing=existing, reserve_bytes=reserve_bytes)
    # Older extensions retain their conservative compatibility allowance.
    remaining = int(max(0., mycc.max_memory-lib.current_memory()[0])*1e6)
    remaining -= _future_bytes(mycc, reserve_bytes)
    # Packed tau + packed ladder result + expanded result coexist briefly.
    ladder_reserved = (o*(o+1)*v*v + o*o*v*v + v*v)*8
    if not eris.vvvv.flags.c_contiguous:
        ladder_reserved += eris.vvvv.nbytes
    workspace = remaining-ladder_reserved
    # Conservative allowance for resident intermediates, outputs, and the
    # amplitude/layout caches and packing buffers; tiles use the remaining RAM.
    fixed = 16*t2.nbytes + 2*o**4*8 + _residual_scratch_bytes(o, v)
    per_virtual = 8*(7*o*o*v + o**3 + 3*o*v*v)
    block = min(v, (remaining-fixed)//max(1, per_virtual))
    if workspace < 8*v*v or block < 1:
        return None
    return workspace, block


def _future_bytes(mycc, reserve_bytes=None):
    return int(getattr(mycc, '_native_ccsd_future_bytes', 0)
               if reserve_bytes is None else reserve_bytes)


def _workspace_key(eris):
    return tuple((id(a), a.shape, a.strides) for a in (eris.ovvv, eris.ovoo))


def _cached_workspace(eris):
    cached = getattr(eris, '_native_residual_workspace', None)
    if cached is not None and cached[0] == _workspace_key(eris):
        return cached[1]
    eris._native_residual_workspace = None
    return None


def _plan_for(mycc, t1, t2, eris, *, resident, existing=None, reserve_bytes=None):
    from ._native_ccsd_memory import residual_memory
    o, v = t1.shape
    available = int((mycc.max_memory-lib.current_memory()[0])*1e6)
    available -= _future_bytes(mycc, reserve_bytes)
    if available <= 0:
        return None
    copy_bytes = 0 if eris.vvvv.flags.c_contiguous else eris.vvvv.nbytes
    plans = [_ccsd_kernels.ladder_memory(o, v)]
    minimum = _ccsd_kernels.ladder_memory(o, v, 8*v*v)
    if minimum['panel_bytes'] != plans[0]['panel_bytes']:
        plans.append(minimum)
    # Prefer normal parallel ladder panels. Try the one-panel path only when
    # even one residual virtual slice cannot fit with the normal panel plan.
    for ladder in plans:
        for block in range(v, 0, -1):
            estimate = residual_memory(o, v, block, ladder, resident=resident,
                existing=existing, vvvv_copy_bytes=copy_bytes,
                damping=mycc.iterative_damping < 1.,
                short_lifetimes=bool(getattr(_ccsd_kernels, 'ovov_reuse_version', 0)))
            if estimate['incremental_bytes'] <= available:
                return ladder['panel_bytes'], block
    return None


def _native_update(mycc, t1, t2, eris, plan, *, profile=False):
    """Python owns resource policy; both numerical phases execute in C++."""
    from threadpoolctl import threadpool_limits
    workspace_cache = _resident_workspace(mycc, t1, t2, eris)
    available = _native_plan(mycc, t1, t2, eris)
    if available is None and workspace_cache is not None:
        eris._native_residual_workspace = None
        workspace_cache = None
        available = _native_plan(mycc, t1, t2, eris)
    if available is None:
        if profile:
            raise MemoryError('insufficient memory for native residual')
        return _reference_update_amps(mycc, t1, t2, eris)
    plan = min(plan[0], available[0]), min(plan[1], available[1])
    workspace, block = plan
    time0 = logger.process_clock(), logger.perf_counter()
    # Native kernels own OpenMP teams with serial BLAS in each worker. Restore
    # the caller's BLAS pool after both phases, including on exceptions.
    with threadpool_limits(1, user_api='blas'):
        args = (t1, t2, eris.fock, eris.mo_energy, eris.oooo, eris.ovoo,
                eris.oovv, eris.ovvo, eris.ovvv)
        kw = {'workspace': workspace_cache} if workspace_cache is not None else {}
        if hasattr(_ccsd_kernels, 'residual_from_integrals'):
            result = _ccsd_kernels.residual_from_integrals(
                *args, np.ascontiguousarray(eris.vvvv), workspace,
                mycc.level_shift, block, profile, **kw)
        else:
            seed = _ccsd_kernels.ladder_residual(
                t1, t2, np.ascontiguousarray(eris.vvvv), workspace)
            result = _ccsd_kernels.residual(
                *args, seed, mycc.level_shift, block, profile, **kw)
    logger.timer_debug1(mycc, 'native CCSD update', *time0)
    return result


def _resident_workspace(mycc, t1, t2, eris):
    """Retain layouts of immutable host ERIs and refill amplitude workspaces.

    This cache lives with the host solve, like its packed vvvv cache. Replacing
    an ERI array/view rebuilds it; callers editing ERIs in place must clear
    ``eris._native_residual_workspace`` before the next update.
    """
    if not hasattr(_ccsd_kernels, 'ResidualWorkspace'):
        return None
    cached = _cached_workspace(eris)
    if cached is not None:
        return cached
    # Resident planning requires allocation/population metadata from the new
    # extension. Older binaries retain the bounded nonresident path.
    if (not hasattr(_ccsd_kernels, 'ladder_memory') or
            _plan_for(mycc, t1, t2, eris, resident=True) is None):
        return None
    workspace = _ccsd_kernels.ResidualWorkspace(eris.ovvv, eris.ovoo)
    eris._native_residual_workspace = _workspace_key(eris), workspace
    return workspace


def update_amps(mycc, t1, t2, eris):
    """Evaluate resident real RCCSD entirely with native numerical kernels.

    PySCF remains the reference and fallback for uncached/direct integrals,
    unsupported solver modes, unavailable extensions, or insufficient memory.
    Python retains solver policy; dense solver numerics use separate kernels.
    """
    plan = _native_plan(mycc, t1, t2, eris)
    if plan is None and getattr(eris, '_native_residual_workspace', None) is not None:
        # A reference/tiled fallback must not retain an unused native cache.
        eris._native_residual_workspace = None
        plan = _native_plan(mycc, t1, t2, eris)
    if plan is None:
        return _reference_update_amps(mycc, t1, t2, eris)
    return _native_update(mycc, t1, t2, eris, plan)
