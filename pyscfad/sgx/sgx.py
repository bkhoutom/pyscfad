# Copyright 2026 The PySCFAD Authors
# Licensed under the Apache License, Version 2.0.
"""First-order coordinate and density response for direct DF-J + SGX-K.

With static screening and FP64, the symmetric overlap fit makes K a
self-adjoint linear map of a symmetric density. Density-dependent screening
and mixed precision approximate this map. Density pullbacks follow the selected
precision mode while disabling density-dependent skipping. A bilinear analytic
kernel supplies arbitrary geometry pullbacks including grid motion, partition
weights and overlap-fit response. DF-J geometry derivatives contract shell
integrals directly, without AO-pair panels. No geometry finite differences
are used by this primitive.
"""
import copy
from functools import partial
import math

import jax
from jax.custom_derivatives import CustomVJPPrimal, SymbolicZero
import numpy as np
from pyscf import lib, scf as pscf
from pyscf.sgx import sgx as native_sgx
from pyscf.sgx.grad import rhf as sgx_grad
from . import _dfj, _sgx_grad


def _configure(mol, auxbasis, grid_level, fit_ovlp, max_memory, workspace,
               integral_backend='auto', accuracy=None):
    if not hasattr(sgx_grad, 'get_k_grad_only'):
        raise ImportError('DF-J/SGX-K requires PySCF with optimized SGX gradients (2.14+)')
    obj = _sgx_grad.SGX(mol, auxbasis=copy.deepcopy(auxbasis))
    obj._response_workspace_bytes = int(workspace * 1e6)
    obj._response_total_memory_mb = float(mol.max_memory)
    obj._integral_backend = integral_backend
    obj._sgx_accuracy = copy.deepcopy(accuracy)
    obj.grids_level_i = obj.grids_level_f = grid_level
    obj.dfj = obj.optk = obj.use_opt_grids = True
    obj.fit_ovlp = fit_ovlp
    obj._symm_ovlp_fit = True
    obj._grid_response = True
    # The strict reference uses fixed masks. Optional adaptive screening is
    # implemented in our generated driver, including the bilinear response;
    # upstream PySCF's forward-only screening must remain disabled here.
    obj.sgx_tol_energy = obj.sgx_tol_potential = None
    obj.max_memory = max_memory
    return obj


def sgx_fit(mf, auxbasis=None, *, grid_level=2, fit_ovlp=True,
            max_memory=None, integral_backend='auto', screening='static',
            mixed_precision=False, energy_tol='auto', potential_tol='auto',
            fp32_multiplier=1e5):
    """Return RHF using integral-direct DF-J and optimized symmetric SGX-K.

    Accepts PySCFAD or native PySCF RHF, so reference and differentiable SCFs
    can use identical settings. On PySCFAD, ``with_df`` remains the separate
    correlation DF object. ``auxbasis`` specifies the Coulomb fitting basis;
    an existing correlation auxiliary basis is preserved.

    There are two precision modes, controlled by ``mixed_precision``:
    ``False`` uses FP64 throughout; ``True`` selects FP32 or FP64 for forward,
    density-response and geometry kernels. Mixed geometry may promote eligible
    FP32 work to FP64 when the derivative crossover predicts faster execution.
    There is no all-FP32 mode. Screening is independent of precision: use
    ``screening='per_pair'`` in both modes to compare at the same skip cutoffs.
    The default remains static screening with FP64 only.

    ``max_memory`` is the J/K working-memory allowance in MB (default 4000).
    No global three-index tensor or AO-pair derivative panel is built.
    Grid panels grow with the thread count and AO dimension, targeting four
    native grid jobs per thread within the workspace and current memory
    headroom. Eager GMRES workspace is reserved using its live dimension,
    dtype and restart setting. Planning reserves AO matrices and native scratch,
    and leaves 20 percent of headroom for untracked work. The total memory
    ceiling is ``mol.max_memory`` (or an enclosing SCF-response scope).
    Metric derivative tiles remain bounded at 32 MiB.

    ``integral_backend='auto'`` uses portable generated FP64 s--g value and
    first-derivative kernels when libsgx_ad is installed, falling back to
    libcint otherwise. Version 1 libraries accelerate values only.
    ``'generated'`` requires those kernels; ``'libcint'`` selects the reference
    engine. Version 2 combines values and integral derivatives in one native
    call. Both engines use the same density-independent shell-pair screen,
    grid/weight response, and overlap fitting.

    ``screening='adaptive'`` uses cumulative density-dependent screening budgets
    for shell-pair/grid tiles. ``mixed_precision=True`` evaluates small contributions
    and accumulates their G/dG buffers in FP32 before merging into FP64. Both
    options require version 3 of the generated library. The default retains
    the original static-screened FP64 reference. ``energy_tol='auto'`` uses
    ``direct_scf_tol`` (normally 1e-13); ``potential_tol='auto'`` uses its square
    root. Estimated screening budgets are distributed over the grid and spent
    across pairs. ``fp32_multiplier`` multiplies each tile's budgets to select
    FP32 work. These settings do not certify a molecular force error: validate
    against the strict reference for the intended system and property.

    ``screening='per_pair'`` (library version 4) instead tests each shell pair
    independently. Its auto tolerances are energy=1e-10 and potential=1e-8.
    Each is multiplied by the whole grid's mean absolute weight and by
    tile_points/1024, with no total-grid-size divisor. Keep these nominal
    tolerances fixed across system sizes; accumulated errors still require
    validation. The FP32 multiplier and derivative guard apply in both modes.

    Adaptive selection makes the computed density map approximately linear.
    Reverse-mode derivatives neglect discrete changes in selection, as in
    ordinary screened integral algorithms. Shared geometry calls use the union
    of all density/adjoint requirements and a tighter derivative criterion.
    Density pullbacks disable density-dependent skipping for both J and K,
    while retaining a fixed Schwarz integral screen for both DF-J passes.
    DF-J geometry uses separate AO-derivative and auxiliary-derivative
    Schwarz bounds at the SCF integral tolerance.
    With ``mixed_precision=True``, response K uses the same FP32 classification
    thresholds as forward K; otherwise it uses static FP64. Mixed arithmetic
    remains approximately linear. There is no separate response-precision
    setting; an FP64 reference uses a separate ``sgx_fit`` object with
    ``mixed_precision=False``. Forward and geometry calls retain the selected
    accuracy settings.

    Forward J/K and eager geometry kernels use the current PySCF thread
    budget with dynamic OpenMP/MKL shrinking disabled, restoring the caller's
    settings afterward. Modern native contractions retain all density pairs.
    The STC driver combines complete SCF geometry seeds into one J pass and
    one K traversal; ordinary standalone VJPs also support immediate response.

    Supported: real closed-shell RHF, reverse-mode first derivatives of
    coordinates and densities, implicit SCF or unrolled SCF iterations.
    JIT, forward-mode/higher derivatives, basis-parameter derivatives,
    range separation and distributed SCF J/K are not supported.
    """
    from pyscfad.scf.hf import RHF
    if not isinstance(mf, pscf.hf.RHF) or hasattr(mf, 'xc') or mf.mol.spin:
        raise NotImplementedError('DF-J/SGX-K supports real closed-shell RHF only')
    if isinstance(grid_level, bool) or not isinstance(grid_level, int) or not 0 <= grid_level <= 7:
        raise ValueError('SGX grid_level must be an integer from 0 to 7')
    memory = 4000. if max_memory is None else float(max_memory)
    if not math.isfinite(memory) or memory <= 0:
        raise ValueError('SGX max_memory must be finite and positive')
    if not isinstance(fit_ovlp, bool):
        raise TypeError('fit_ovlp must be a bool')
    if integral_backend not in ('auto', 'generated', 'libcint'):
        raise ValueError('integral_backend must be auto, generated, or libcint')
    if screening not in ('static', 'adaptive', 'per_pair'):
        raise ValueError('SGX screening must be static, adaptive or per_pair')
    if not isinstance(mixed_precision, bool):
        raise TypeError('mixed_precision must be a bool')
    for name, value in (('energy_tol', energy_tol), ('potential_tol', potential_tol)):
        if value != 'auto' and (not isinstance(value, (int, float)) or
                                not math.isfinite(value) or value <= 0):
            raise ValueError(f'{name} must be auto or a finite positive number')
    if not math.isfinite(fp32_multiplier) or fp32_multiplier <= 0:
        raise ValueError('fp32_multiplier must be finite and positive')
    if integral_backend == 'libcint' and (screening != 'static' or mixed_precision):
        raise ValueError('Adaptive/mixed SGX requires the generated backend')
    accuracy = dict(screening=screening, mixed_precision=mixed_precision,
                    energy_tol=energy_tol, potential_tol=potential_tol,
                    fp32_multiplier=fp32_multiplier)
    if auxbasis is None and getattr(mf, 'with_df', None) is not None:
        auxbasis = mf.with_df.auxbasis
    if isinstance(mf, RHF):
        _check_mol(mf.mol)
        mf = mf.copy()
        if getattr(mf, 'with_df', None) is None:
            mf = mf.density_fit(auxbasis=auxbasis)
        mf._sgx_backend = _Backend(auxbasis, grid_level, fit_ovlp, memory, integral_backend, accuracy)
        mf._eri = None
        mf.direct_scf = False
        from . import _shared
        _shared.register_producer(mf)
        return mf
    obj = _configure(mf.mol, auxbasis, grid_level, fit_ovlp,
                     lib.current_memory()[0] + memory, memory, integral_backend, accuracy)
    result = native_sgx.sgx_fit(mf, with_df=obj, pjs=True)
    result.direct_scf = False
    return result


def _check_mol(mol):
    if any(getattr(mol, name, None) is not None for name in ('exp', 'ctr_coeff', 'r0')):
        raise NotImplementedError('SGX supports coordinate derivatives only; build the '
                                  'molecule with trace_exp=False, trace_ctr_coeff=False')


def _host(x):
    if isinstance(x, jax.core.Tracer):
        raise NotImplementedError('DF-J/SGX-K supports eager first-order reverse AD only '
                                  '(no JIT, JVP or higher derivatives)')
    return np.asarray(x)


def _sym(x):
    return (x + x.swapaxes(-1, -2)) * .5


class _Backend:
    """One geometry cache per SCF, shared by its pytree copies and response."""
    def __init__(self, auxbasis, grid_level, fit_ovlp, memory, integral_backend='auto', accuracy=None):
        self.auxbasis = copy.deepcopy(auxbasis)
        self.grid_level, self.fit_ovlp, self.memory = grid_level, fit_ovlp, memory
        self.integral_backend = integral_backend
        self.accuracy = copy.deepcopy(accuracy)
        self._key = self._native = None
        self._primal_cache = []
        self.primal_cache_stats = dict(hits=0, misses=0, j_builds=0, k_builds=0,
                                      retained_bytes=0, peak_bytes=0)

    def native(self, mol, tol):
        from ._shared import _source_key
        _check_mol(mol)
        coords = _host(mol.atom_coords())
        # Include the static basis/environment too: reset/scanner calls can
        # change the molecule while retaining this backend. The nuclear
        # attraction atom selector is scratch, irrelevant to J/K integrals.
        key = (*_source_key(mol), tol)
        budget = lib.current_memory()[0] + self.memory
        if key != self._key:
            self._primal_cache.clear()
            self.primal_cache_stats['retained_bytes'] = 0
            native = mol.to_pyscf().copy()
            native.set_geom_(coords, unit='Bohr')
            obj = _configure(native, self.auxbasis, self.grid_level,
                             self.fit_ovlp, budget, self.memory, self.integral_backend, self.accuracy)
            obj.build()
            obj._build_pjs(tol)
            self._native, self._key = obj, key
        obj = self._native
        # These PySCF kernels subtract RSS from max_memory. Treat the user's
        # allowance as workspace even when correlation intermediates survive.
        obj.max_memory = obj._pjs_data.max_memory = budget
        obj._response_total_memory_mb = obj._pjs_data._response_total_memory_mb = float(mol.max_memory)
        return obj

    def primal_jk(self, mol, dm, with_j, with_k, tol):
        """Reuse exact forward densities across the final SCF and root VJPs.

        Two entries cover the densities before and after the final update.
        Only primal values are cached: the custom VJP still differentiates
        every call, and Krylov trial densities do not evict these entries.
        Geometry/tolerance changes invalidate the cache in native().
        """
        obj = self.native(mol, tol)
        dm = _sym(_host(dm))
        stats = self.primal_cache_stats
        entry = next((e for e in self._primal_cache
                      if np.array_equal(e['dm'], dm)), None)
        need_j = with_j and (entry is None or entry['j'] is None)
        need_k = with_k and (entry is None or entry['k'] is None)
        if not need_j and not need_k:
            stats['hits'] += 1
        else:
            stats['misses'] += 1
            j, k = obj.get_jk(dm, hermi=1, with_j=need_j, with_k=need_k,
                              direct_scf_tol=tol)
            stats['j_builds'] += int(need_j)
            stats['k_builds'] += int(need_k)
            if entry is None:
                entry = dict(dm=dm.copy(), j=None, k=None)
            if need_j:
                entry['j'] = np.array(j, copy=True)
            if need_k:
                entry['k'] = np.array(k, copy=True)
        # Keep optional O(nao**2) storage within live memory, including when
        # the caller reduces its budget between calls. Never retain grids.
        headroom = max(0., (float(mol.max_memory)-lib.current_memory()[0])*1e6)
        available = _sgx_grad._system_available_bytes()
        if available is not None:
            headroom = min(headroom, available)
        cap = min(.05*self.memory*1e6, .05*headroom)
        self._primal_cache = [e for e in self._primal_cache if e is not entry]
        self._primal_cache.append(entry)
        size = lambda e: sum(x.nbytes for x in e.values() if x is not None)
        retained = sum(map(size, self._primal_cache))
        while self._primal_cache and (len(self._primal_cache) > 2 or retained > cap):
            retained -= size(self._primal_cache.pop(0))
        stats['retained_bytes'] = retained
        stats['peak_bytes'] = max(stats['peak_bytes'], retained)
        # Protect cached values from caller-side mutation.
        return (entry['j'].copy() if with_j else None,
                entry['k'].copy() if with_k else None)

    def get_jk(self, mol, dm, hermi, with_j, with_k, tol, omega):
        from pyscfad.df.df_jk import _MPI_GET_JK_HOOK
        if _MPI_GET_JK_HOOK.get() is not None:
            raise NotImplementedError('SGX requires parallel_scf_jk=False')
        if hermi != 1 or omega not in (None, 0):
            raise NotImplementedError('SGX requires hermi=1 and no range separation')
        if np.iscomplexobj(dm) or dm.dtype != np.float64:
            raise NotImplementedError('SGX requires real float64 densities')
        if dm.ndim < 2 or dm.shape[-2:] != (mol.nao, mol.nao):
            raise ValueError('SGX density must have trailing shape (nao, nao)')
        _check_mol(mol)
        return _get_jk(self, mol, dm, with_j, with_k, tol)


@partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4, 5))
def _get_jk(backend, mol, dm, with_j, with_k, tol):
    if not with_j and not with_k:
        return None, None
    return backend.primal_jk(mol, dm, with_j, with_k, tol)


def _fwd(backend, mol, dm, with_j, with_k, tol):
    active = lambda tree: any(isinstance(x, CustomVJPPrimal) and x.perturbed
                             for x in jax.tree_util.tree_leaves(tree))
    activity = (active(mol), active(dm))
    unwrap = lambda x: x.value if isinstance(x, CustomVJPPrimal) else x
    mol, dm = jax.tree_util.tree_map(unwrap, (mol, dm))
    return _get_jk(backend, mol, dm, with_j, with_k, tol), (mol, dm, activity)


def _j_geometry(obj, dm, bar):
    return _dfj.geometry(obj, dm, bar)


def _k_geometry(obj, dm, bar, tol):
    return _sgx_grad.geometry(obj, dm, bar, tol)


def _bwd(backend, with_j, with_k, tol, residual, bars):
    mol, dm, (geometry_active, dm_active) = residual
    obj = backend.native(mol, tol)
    dm = _sym(_host(dm))
    dbar, rbar = np.zeros_like(dm), np.zeros((mol.natm, 3))
    geometry_bars = [None, None]
    for k, (enabled, bar) in enumerate(zip((with_j, with_k), bars)):
        if not enabled or bar is None or isinstance(bar, SymbolicZero):
            continue
        bar = _sym(_host(bar))
        if not np.any(bar):
            continue
        if dm_active:
            response = (_dfj.density_response(obj, bar, tol) if k == 0 else
                        _sgx_grad.density_response(obj, bar, tol))
            dbar += response
        if geometry_active:
            geometry_bars[k] = bar
    if any(b is not None for b in geometry_bars):
        from . import _shared
        deferred = _shared.defer(backend, mol, dm, *geometry_bars, tol)
        if not deferred:
            for k, bar in enumerate(geometry_bars):
                if bar is None:
                    continue
                contributions = list(zip(dm.reshape(-1, mol.nao, mol.nao),
                                         bar.reshape(-1, mol.nao, mol.nao)))
                if len(contributions) == 1:
                    d, b = contributions[0]
                    rbar += (_j_geometry(obj, d, b) if k == 0
                             else _k_geometry(obj, d, b, tol))
                    continue
                if k == 0:
                    rbar += _dfj.geometry_many(obj, contributions)
                else:
                    rbar += _sgx_grad.geometry_many(obj, contributions, tol)
    molbar = jax.tree_util.tree_map(lambda x: np.zeros(x.shape, x.dtype), mol)
    if mol.coords is not None:
        molbar.coords = rbar
    return molbar, dbar


_get_jk.defvjp(_fwd, _bwd, symbolic_zeros=True)
