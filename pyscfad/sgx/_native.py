# Copyright 2026 The PySCFAD Authors. Apache License, Version 2.0.
"""Optional portable s--g integral/contraction backend.

FP64-only and mixed precision share the integral plan and contraction driver.
Density-dependent screening is configured independently of precision.
"""
import ctypes
from functools import lru_cache
import weakref

import numpy as np
from pyscf import gto, lib


@lru_cache(None)
def library():
    from pyscfadlib.lib import load_library
    try:
        dll = load_library('libsgx_ad')
    except OSError:
        return None
    ptr, integer, size, real = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_double
    dll.SGX_generated_version.restype = integer
    version = dll.SGX_generated_version()
    if version not in (1, 2, 3, 4):
        return None
    signatures = {
        'SGX_generated_error': (ctypes.c_char_p, []),
        'SGX_generated_new': (ptr, [integer, ptr, ptr, ptr, ptr, ptr, real, integer, size]),
        'SGX_generated_delete': (None, [ptr]),
        'SGX_generated_bytes': (size, [ptr]),
        'SGX_generated_scratch': (size, [ptr, integer]),
        'SGX_generated_contract': (integer, [ptr, integer, integer, ptr, ptr, ptr, integer, integer]),
        'SGX_generated_pair': (integer, [ptr, integer, integer, integer, ptr, ptr, integer]),
        'SGX_generated_boys': (integer, [integer, integer, ptr, ptr]),
    }
    if version >= 2:
        signatures.update({
            'SGX_generated_scratch_deriv': (size, [ptr, integer]),
            'SGX_generated_contract_deriv': (integer, [ptr, integer, integer, ptr, ptr, ptr, ptr, integer]),
            'SGX_generated_pair_deriv': (integer, [ptr, integer, integer, integer, ptr, ptr]),
        })
    if version >= 3:
        signatures.update({
            'SGX_generated_scratch_mixed': (size, [ptr, integer, integer]),
            'SGX_generated_contract_adaptive': (integer, [ptr, integer, integer,
                ptr, ptr, ptr, ptr, ptr, ptr, real, real, real, integer, integer, ptr]),
            'SGX_generated_pair_float': (integer, [ptr, integer, integer, integer, ptr, ptr, integer]),
            'SGX_generated_boys_float': (integer, [integer, integer, ptr, ptr]),
        })
    for name, (restype, argtypes) in signatures.items():
        fn = getattr(dll, name)
        fn.restype, fn.argtypes = restype, argtypes
    # Additive optional ABI: old generated libraries keep the NumPy fallback.
    if hasattr(dll, 'SGX_contract_geometry'):
        dll.SGX_contract_geometry.restype = integer
        dll.SGX_contract_geometry.argtypes = [integer]*4 + [ptr]*10 + [integer]
    return dll


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def reduce_geometry(dao, fg, gv, dgv, weights, start, stop, index, tmp,
                    atom_ao, xed, dxed):
    """Accumulate one compact geometry subpanel; False selects old-library fallback."""
    dll = library()
    if dll is None or not hasattr(dll, 'SGX_contract_geometry'):
        return False
    nao, ngrids = fg.shape
    arrays = (weights, fg, gv, dgv, dao, index, tmp, atom_ao, xed, dxed)
    if dll.SGX_contract_geometry(nao, ngrids, start, stop-start,
                                 *map(_ptr, arrays), lib.num_threads()):
        raise MemoryError('Unable to allocate SGX geometry contraction scratch')
    return True


class IntegralPlan:
    # TILE in libsgx_ad ABI versions 1--4. Independent of PySCF's 224-point jobs.
    grid_batch_points = 128

    def __init__(self, mol, bounds, cutoff, max_bytes):
        self.dll = library()
        if self.dll is None:
            raise ImportError('Generated SGX requires pyscfadlib with libsgx_ad')
        if (np.any(mol._bas[:, gto.ANG_OF] > 4) or
                mol._env[gto.PTR_RANGE_OMEGA] != 0 or mol._env[gto.PTR_RINV_ZETA] != 0):
            raise NotImplementedError('Generated SGX supports ordinary Coulomb s--g integrals')
        transforms = []
        for shell in range(mol.nbas):
            l = mol.bas_angular(shell)
            nc = (l+1)*(l+2)//2
            transforms.append((np.eye(nc) if mol.cart else
                               gto.cart2sph(l, normalized='sp')).ravel())
        transforms = np.ascontiguousarray(np.concatenate(transforms), dtype=np.float64)
        bas = np.ascontiguousarray(mol._bas, dtype=np.int32)
        atm = np.ascontiguousarray(mol._atm, dtype=np.int32)
        env = np.ascontiguousarray(mol._env, dtype=np.float64)
        bounds = np.ascontiguousarray(bounds, dtype=np.float64)
        if bounds.shape != (mol.nbas, mol.nbas):
            raise ValueError('SGX shell-pair bounds must have shape (nbas, nbas)')
        handle = self.dll.SGX_generated_new(mol.nbas, _ptr(bas), _ptr(atm), _ptr(env),
            _ptr(transforms), _ptr(bounds), float(cutoff), int(mol.cart), int(max_bytes))
        if not handle:
            raise MemoryError(self.dll.SGX_generated_error().decode())
        self.handle = handle
        self.has_derivatives = self.dll.SGX_generated_version() >= 2
        self._finalizer = weakref.finalize(self, self.dll.SGX_generated_delete, handle)
        self.nao = mol.nao
        self.cart_loc = mol.ao_loc_nr(cart=True)
        self.bytes = self.dll.SGX_generated_bytes(handle)
        self.accuracy = None
        self.stats = np.zeros(6, dtype=np.uint64)

    def configure_accuracy(self, *, screening=False, mixed=False, energy_tol=1e-13,
                           potential_tol=1e-13**.5, fp32_multiplier=1e5, ngrids=1,
                           screening_mode='cumulative', average_weight=1.):
        """Set cumulative budgets or independent per-pair thresholds.

        Per-pair thresholds are nominal tolerances times the whole grid's mean
        absolute weight times tile_points/1024. They do not divide by the
        total grid size. Neither mode certifies a molecular force error.
        """
        if screening_mode not in ('cumulative', 'per_pair'):
            raise ValueError('SGX screening_mode must be cumulative or per_pair')
        if not screening and not mixed:
            self.accuracy = None
            return
        if self.dll.SGX_generated_version() < 3:
            raise NotImplementedError('Adaptive/mixed SGX requires libsgx_ad version 3')
        if screening_mode == 'per_pair' and self.dll.SGX_generated_version() < 4:
            raise NotImplementedError('Per-pair SGX requires libsgx_ad version 4')
        values = (energy_tol, potential_tol, fp32_multiplier, average_weight)
        if any(not np.isfinite(v) or v <= 0 for v in values) or ngrids < 1:
            raise ValueError('SGX screening tolerances, multiplier and grid count must be positive')
        self.accuracy = dict(screening=bool(screening), mixed=bool(mixed),
            energy_tol=float(energy_tol), potential_tol=float(potential_tol),
            fp32_multiplier=float(fp32_multiplier), ngrids=int(ngrids),
            screening_mode=screening_mode, average_weight=float(average_weight))

    def reset_stats(self):
        self.stats[:] = 0

    def screening_stats(self):
        return dict(zip(('skipped_tiles', 'fp32_tiles', 'fp64_tiles',
                         'skipped_primitive_points', 'fp32_primitive_points',
                         'fp64_primitive_points'), map(int, self.stats)))

    def scratch_bytes(self, nset, derivative=False):
        if self.accuracy is not None:
            return int(self.dll.SGX_generated_scratch_mixed(self.handle, nset, derivative))
        function = (self.dll.SGX_generated_scratch_deriv if derivative and self.has_derivatives
                    else self.dll.SGX_generated_scratch)
        return int(function(self.handle, nset))

    def _inputs(self, coords, fg):
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        fg = np.asarray(fg)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError('SGX coordinates must have shape (ngrids, 3)')
        if fg.ndim != 3 or fg.shape[1:] != (self.nao, len(coords)) or fg.dtype != np.float64:
            raise ValueError('SGX F must be real float64 with shape (nset, nao, ngrids)')
        fg = np.ascontiguousarray(fg)
        return coords, fg

    def contract(self, mol, coords, fg, weights=None, b0=None, *, algorithm=0, ao_l1=None):
        coords, fg = self._inputs(coords, fg)
        out = np.empty_like(fg)
        if self.accuracy is not None:
            if algorithm:
                raise ValueError('Adaptive SGX selects its recurrence algorithm automatically')
            self._adaptive(coords, fg, weights, ao_l1, out, None)
            return out
        if self.dll.SGX_generated_contract(self.handle, len(fg), len(coords),
                _ptr(coords), _ptr(fg), _ptr(out), algorithm, lib.num_threads()):
            raise MemoryError(self.dll.SGX_generated_error().decode())
        return out

    def contract_with_grad(self, mol, coords, fg, weights=None, b0=None, *, ao_l1=None):
        """Return G and its bra spatial derivative, matching int1e_grids_ip.

        This is the integral contribution only. Nuclear-center derivatives
        have the opposite sign; the response driver handles that sign, grid
        motion, partition weights, AO derivatives, and overlap fitting.
        """
        if not self.has_derivatives:
            raise NotImplementedError('Generated SGX derivatives require libsgx_ad version 2')
        coords, fg = self._inputs(coords, fg)
        out = np.empty_like(fg)
        grad = np.empty((len(fg), 3, self.nao, len(coords)))
        if self.accuracy is not None:
            self._adaptive(coords, fg, weights, ao_l1, out, grad)
            return out, grad
        if self.dll.SGX_generated_contract_deriv(self.handle, len(fg), len(coords),
                _ptr(coords), _ptr(fg), _ptr(out), _ptr(grad), lib.num_threads()):
            raise MemoryError(self.dll.SGX_generated_error().decode())
        return out, grad

    def _adaptive(self, coords, fg, weights, ao_l1, out, grad):
        if weights is None or ao_l1 is None:
            raise ValueError('Adaptive SGX needs grid weights and AO absolute sums')
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        ao_l1 = np.ascontiguousarray(ao_l1, dtype=np.float64)
        if weights.shape != (len(coords),) or ao_l1.shape != weights.shape:
            raise ValueError('Adaptive SGX grid metadata must have shape (ngrids,)')
        if not np.isfinite(weights).all() or not np.isfinite(ao_l1).all() or np.any(ao_l1 < 0):
            raise ValueError('Adaptive SGX requires finite weights and nonnegative AO sums')
        cfg = self.accuracy
        per_pair = cfg['screening_mode'] == 'per_pair'
        scale = cfg['average_weight']/1024 if per_pair else 1/cfg['ngrids']
        counts = np.zeros(6, dtype=np.uint64)
        status = self.dll.SGX_generated_contract_adaptive(self.handle, len(fg), len(coords),
            _ptr(coords), _ptr(fg), _ptr(weights), _ptr(ao_l1), _ptr(out),
            None if grad is None else _ptr(grad), cfg['energy_tol']*scale,
            cfg['potential_tol']*scale, cfg['fp32_multiplier'] if cfg['mixed'] else 0.,
            (2 if per_pair else 1) if cfg['screening'] else 0, lib.num_threads(), _ptr(counts))
        if status:
            raise MemoryError(self.dll.SGX_generated_error().decode())
        if not np.isfinite(out).all() or (grad is not None and not np.isfinite(grad).all()):
            raise FloatingPointError('Nonfinite mixed SGX result; use the FP64 reference')
        self.stats += counts

    def shell_pair(self, ia, ib, coords, *, algorithm=0, precision='double'):
        """Cartesian shell integrals for independent validation (AO, AO, grid)."""
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        if not 0 <= ia < len(self.cart_loc)-1 or not 0 <= ib < len(self.cart_loc)-1:
            raise ValueError('Shell index out of range')
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError('SGX coordinates must have shape (ngrids, 3)')
        shape = (self.cart_loc[ia+1]-self.cart_loc[ia],
                 self.cart_loc[ib+1]-self.cart_loc[ib], len(coords))
        out = np.empty(shape)
        if precision == 'float':
            if self.dll.SGX_generated_version() < 3:
                raise NotImplementedError('FP32 SGX requires version 3')
            if self.dll.SGX_generated_pair_float(self.handle, ia, ib, len(coords),
                                                _ptr(coords), _ptr(out), 0):
                raise RuntimeError(self.dll.SGX_generated_error().decode())
            return out
        if precision != 'double':
            raise ValueError('precision must be double or float')
        if self.dll.SGX_generated_pair(self.handle, ia, ib, len(coords),
                                      _ptr(coords), _ptr(out), algorithm):
            raise RuntimeError(self.dll.SGX_generated_error().decode())
        return out

    def shell_pair_with_grad(self, ia, ib, coords, *, precision='double'):
        """Return Cartesian values and both shells' spatial derivatives.

        Shapes are (AOa, AOb, grid) and (2, 3, AOa, AOb, grid). The first
        derivative is libcint's int1e_grids_ip convention; the second is its
        shell-swapped counterpart. Basis centers have the opposite sign.
        """
        if not self.has_derivatives:
            raise NotImplementedError('Generated SGX derivatives require libsgx_ad version 2')
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        if not 0 <= ia < len(self.cart_loc)-1 or not 0 <= ib < len(self.cart_loc)-1:
            raise ValueError('Shell index out of range')
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError('SGX coordinates must have shape (ngrids, 3)')
        shape = (self.cart_loc[ia+1]-self.cart_loc[ia],
                 self.cart_loc[ib+1]-self.cart_loc[ib], len(coords))
        out = np.empty((7, *shape))
        if precision == 'float':
            if self.dll.SGX_generated_version() < 3:
                raise NotImplementedError('FP32 SGX requires version 3')
            status = self.dll.SGX_generated_pair_float(self.handle, ia, ib, len(coords),
                                                      _ptr(coords), _ptr(out), 1)
        elif precision == 'double':
            status = self.dll.SGX_generated_pair_deriv(self.handle, ia, ib, len(coords),
                                                      _ptr(coords), _ptr(out))
        else:
            raise ValueError('precision must be double or float')
        if status:
            raise RuntimeError(self.dll.SGX_generated_error().decode())
        return out[0], out[1:].reshape(2, 3, *shape)


def _configure_accuracy(plan, data):
    """Resolve public screening/precision settings for one native plan.

    Static FP64 needs no adaptive driver. All other combinations use the same
    driver with independent skipping and mixed-precision flags.
    """
    accuracy = getattr(data, '_sgx_accuracy', None)
    if not accuracy:
        return
    screening = accuracy['screening'] != 'static'
    mixed = accuracy['mixed_precision']
    if not screening and not mixed:
        return
    if plan is None:
        raise NotImplementedError('Adaptive/mixed SGX requires the generated backend')
    per_pair = accuracy['screening'] == 'per_pair'
    energy = accuracy['energy_tol']
    if energy == 'auto':
        energy = 1e-10 if per_pair else data._itol
    potential = accuracy['potential_tol']
    if potential == 'auto':
        potential = 1e-8 if per_pair else energy**.5
    plan.configure_accuracy(
        screening=screening, mixed=mixed, energy_tol=energy,
        potential_tol=potential, fp32_multiplier=accuracy['fp32_multiplier'],
        ngrids=len(data.grids.weights),
        screening_mode='per_pair' if per_pair else 'cumulative',
        average_weight=float(np.mean(np.abs(data.grids.weights))))


def get_plan(data):
    policy = getattr(data, '_integral_backend', 'auto')
    if policy == 'libcint' or data.use_dm_screening:
        return None
    if hasattr(data, '_generated_plan'):
        return data._generated_plan
    total = getattr(data, '_response_total_memory_mb', data.max_memory)
    workspace = getattr(data, '_response_workspace_bytes', data.max_memory*1e6)
    # Cache only primitive-pair metadata, with a separate bounded allowance.
    headroom = max(0., (total-lib.current_memory()[0])*1e6)
    from ._sgx_grad import _system_available_bytes
    available = _system_available_bytes()
    if available is not None:
        headroom = min(headroom, available)
    limit = int(min(.1*workspace, .1*headroom))
    try:
        plan = IntegralPlan(data.mol, data._mbar_ij, data._itol, limit)
    except (ImportError, NotImplementedError, MemoryError) as exc:
        if policy == 'generated':
            raise
        data._generated_fallback_reason = str(exc)
        plan = None
    _configure_accuracy(plan, data)
    data._generated_plan = plan
    return plan
