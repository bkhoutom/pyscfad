"""Physical Boys-fragment CCSD(T) corrections with shared W energy.

Discrete domains and LIS ranks are fixed. Native W chooses each LIS in its
STC frame. The solver uses normalized domain projections of the original
physical Boys targets.
"""
from dataclasses import dataclass, replace
import gc
import os

import jax
import numpy
from pyscfad.tools import resource_profile
from pyscfad import numpy as np
from pyscfad.lno.ccsd import _impurity_solve_core

from . import backend, driver, _stc_precision
from ._profile import response as profile_response
from ._memory import memory_calculation
from ._cleanup import release_memory as _release_memory
from ._selection import (DomainSelections, _physical_groups, _strong_preparation_key,
                         build_domain_groups, physical_fragments_in_group as _group_cc_fragments)
from ._df_basis import (basis_kwargs, basis_identity, selection_auxbasis,
                        validate_selection_basis)
from .adjoint import apply_input_pullback
from .controls import (normalize_matrix_controls, normalize_triples_controls,
                       DEFAULT_TRIPLES_TRACE_EXPONENT)
from .domain import build_boys_ed_frame
from .protocol import validate_input_cotangents


CC_SELECTIONS_SCHEMA_VERSION = 19


def _clear_jax_caches(stage, *live_results, **metadata):
    """Optionally trade recompilation for lower residency between eager stages.

    Saved VJP residuals remain live: only JAX's compilation caches are cleared.
    This does not impose a hard memory limit or change the selected spaces.
    """
    if os.environ.get('PYSCFAD_STC_CLEAR_JAX_CACHES', '0').lower() not in (
            '1', 'true', 'yes', 'on'):
        return
    _release_memory(stage, *live_results, clear_caches=True,
                    profile_prefix='stc.cc', **metadata)


@dataclass(frozen=True)
class CCSDSelections:
    """Fixed LIS choices and the scientific settings used to select them."""
    lis_static: object
    controls_identity: tuple
    schema_version: int = CC_SELECTIONS_SCHEMA_VERSION
    mp2_fragment_energies: tuple | None = None
    selection_options: dict | None = None
    correlation_basis_identity: str | None = None
    mp2_grid: tuple | None = None
    mp2_grid_diagnostics: dict | None = None

    @property
    def mp2_static(self):
        return self.lis_static.mp2_static

    @property
    def fragments(self):
        return self.lis_static.fragments


@dataclass(frozen=True)
class FragmentComponents:
    fragment_index: int
    e_mp2_lis: float
    e_ccsd: float
    e_triples: float
    matrix_energy_diagnostic: float
    nocc_lis: int | None = None
    nvir_lis: int | None = None
    direct_host_bar_norm: float | None = None
    native_host_bar_norm: float | None = None
    matrix_diagnostics: dict | None = None
    triples_method: str = 'none'
    triples_standard_error: float = 0.
    triples_samples: int = 0
    triples_timings: dict | None = None
    triples_diagnostics: dict | None = None
    ccsd_method: str = 'deterministic'
    ccsd_diagnostics: dict | None = None
    auxiliary_compression: dict | None = None

    @property
    def e_ccsd_t(self):
        return self.e_triples


@dataclass(frozen=True)
class CCSDComponents:
    """Correlation accounting; W supplies its own strong-domain MP2 energy."""
    e_mp2_base: float
    e_mp2_weak: float
    fragments: tuple
    matrix_kind: str
    baseline_standard_error: float

    @property
    def e_ccsd(self):
        return sum(f.e_ccsd for f in self.fragments)

    @property
    def e_mp2_lis(self):
        return sum(f.e_mp2_lis for f in self.fragments)

    @property
    def e_triples(self):
        return sum(f.e_triples for f in self.fragments)

    @property
    def e_ccsd_t(self):
        return self.e_triples

    @property
    def e_corr(self):
        return self.e_mp2_base + self.e_ccsd + self.e_triples - self.e_mp2_lis

    @property
    def triples_standard_error(self):
        """Energy SE for independently sampled LIS triples, excluding D error."""
        return sum(f.triples_standard_error**2 for f in self.fragments)**.5


    @property
    def stc_standard_error(self):
        """Empirical SE of independent final projected replica means, or None."""
        statistics = [(f.ccsd_diagnostics or {}).get('precision') for f in self.fragments]
        if not statistics or any(item is None for item in statistics):
            return None
        return sum(item['standard_error']**2 for item in statistics)**.5

    @property
    def stc_target_standard_error(self):
        """Requested SE over selected physical fragments, in Hartree, or None."""
        statistics = [(f.ccsd_diagnostics or {}).get('precision') for f in self.fragments]
        if not statistics or any(item is None for item in statistics):
            return None
        return sum(item['fragment_target_standard_error']**2 for item in statistics)**.5

    @property
    def stc_target_met(self):
        """Whether the empirical global SE meets its target; no bias guarantee."""
        error, target = self.stc_standard_error, self.stc_target_standard_error
        return None if error is None else error <= target


def _cc_solver_options(cc_solver, stc_cc_controls, lis_aux_svd_tol=None):
    """Validate the optional backend without importing external STC packages."""
    from ._aux_compress import normalize_tolerance
    tolerance = normalize_tolerance(lis_aux_svd_tol)
    options = {} if tolerance is None else {'lis_aux_svd_tol': tolerance}
    if cc_solver not in ('deterministic', 'stc'):
        raise ValueError("cc_solver must be 'deterministic' or 'stc'")
    if cc_solver == 'deterministic':
        if stc_cc_controls is not None:
            raise ValueError("stc_cc_controls requires cc_solver='stc'")
        return options
    from .stc_ccsd import normalize_controls
    return dict(options, cc_solver='stc', stc_cc_controls=normalize_controls(stc_cc_controls))


def _normalize_controls(baseline_controls, matrix_controls, mp2_grid=None):
    """Resolve one MP2/W quadrature, independently of the fitted triples grid."""
    from .controls import resolve_mp2_grid
    grid = resolve_mp2_grid(mp2_grid)
    def with_grid(controls):
        if controls is None:
            controls = {}
        if not isinstance(controls, dict):
            raise TypeError('controls must be a dictionary')
        if {'laplace_roots', 'laplace_weights'} & controls.keys():
            raise ValueError('CC MP2/W quadrature belongs in mp2_grid=(roots, weights), '
                             'not baseline_controls or matrix_controls')
        return dict(controls, **grid)
    baseline = driver._driver_controls(with_grid(baseline_controls))
    baseline.setdefault('mode', 'deterministic')
    baseline.setdefault('global_seed', 0)
    matrix = normalize_matrix_controls(with_grid(matrix_controls))
    return baseline, matrix


def _scientific_identity(baseline, matrix):
    # Memory/thread knobs do not alter the scientific selection. Sampling
    # allocation, seeds, batching and the Laplace grid do and remain recorded.
    operational = {'threads', 'memory_limit_mb', 'virtual_block_size'}
    def freeze(value):
        if isinstance(value, dict):
            return tuple((key, freeze(item)) for key, item in sorted(value.items())
                         if key not in operational)
        if isinstance(value, (list, tuple, numpy.ndarray)):
            return tuple(freeze(item) for item in value)
        return value.item() if isinstance(value, numpy.generic) else value
    grid_keys = {'laplace_roots', 'laplace_weights'}
    return (freeze({key: baseline[key] for key in grid_keys}),
            freeze({key: value for key, value in baseline.items() if key not in grid_keys}),
            freeze({key: value for key, value in matrix.items() if key not in grid_keys}))


def _validate_run(static, baseline_controls, matrix_controls, mp2_grid=None):
    if not isinstance(static, CCSDSelections):
        raise TypeError('static must be CCSDSelections from build_static_selections')
    if vars(static).get('schema_version') != CC_SELECTIONS_SCHEMA_VERSION:
        raise ValueError('incompatible CC selection schema; rebuild domain-space LIS selections')
    if static.lis_static.smooth_frames is None or static.lis_static.triples_grid is None:
        raise ValueError('only smooth LIS treatment is supported; rebuild smooth selections')
    if static.mp2_grid is None:
        raise ValueError('CC selections have no saved MP2 grid; rebuild selections')
    baseline, matrix = _normalize_controls(
        baseline_controls, matrix_controls, static.mp2_grid if mp2_grid is None else mp2_grid)
    if static.selection_options is None and any(f is None for f in static.fragments):
        raise ValueError('CC selections must include every physical fragment; rebuild selections')
    if _scientific_identity(baseline, matrix) != static.controls_identity:
        raise ValueError('scientific controls differ from the saved LIS selection')
    return baseline, matrix


def _all_fragment_indices(static):
    """Every physical fragment contributes one CC correction."""
    return tuple(range(len(static.fragments)))



def _fragment_dimensions(static, index):
    if not isinstance(static, CCSDSelections):
        return None, None
    item = static.fragments[index]
    if item is None:
        return None, None
    parent = item.parent_dimensions
    occ = (parent[0] if item.full_occupied_space else
           len(item.internal_occ_keep) + len(item.occupied_lno_keep))
    vir = (parent[1] if item.full_virtual_space else
           len(item.internal_vir_keep) + len(item.virtual_lno_keep))
    return occ, vir


def _bar_norm(*trees):
    squares = [np.sum(np.abs(leaf)**2) for leaf in jax.tree_util.tree_leaves(trees)
               if hasattr(leaf, 'dtype') and leaf.dtype != jax.dtypes.float0]
    return float(np.sqrt(sum(squares)))


def build_static_selections(mf, mp2_static, *, thresh_occ, thresh_vir,
                            baseline_controls=None, matrix_controls=None, mp2_grid=None,
                            internal_rank_threshold=1e-6, merge_identical=True,
                            triples_laplace_points=4,
                            correlation_auxbasis=None):
    """Record physical-fragment LIS cutoffs and controls for the W pipeline.

    Use the run's ``correlation_auxbasis`` for W factors. ``None`` uses the
    SCF fit, or inherits the fit inside a driver-owned selection callback.
    ``mp2_static`` is a concrete Boys ``DomainSelections``. Absolute W
    eigenvalue cutoffs select the occupied and virtual spaces.

    Smooth anchored frames with full occupied and virtual Fock matrices are
    the only supported treatment.
    W subspace and AO overlap response are retained; there are no internal
    Fock cluster projectors. Ranks, anchors and quadratures are held fixed
    during differentiation.
    ``mp2_grid=(roots, weights)`` overrides the global fitted MP2/W grid
    saved by domain selection. ``None`` reuses that saved grid. Control dictionaries contain no
    quadrature keys. ``triples_laplace_points`` independently fits four points
    per LIS by default, using triple-excitation denominators.
    This function records selection settings without building W. Each fragment
    builds W once in production, selects its LIS and fits its independent (T)
    grid, then consumes the same W in CCSD(T) and the combined pullback.
    Discrete labels and both grids remain fixed during that fragment's response.
    Occupied and virtual LNO eigenproblems always
    use compact extended-domain coordinates, preserving the projected target
    occupied span and all domain-frame/target-projector responses.

    LIS orbitals are selected directly in the orthonormal ED, retaining
    normalized domain projections of the physical Boys targets. Domain space
    is the only supported definition. Zero cutoffs retain the full ED.
    Resident H_ED reuse is the default for energies and gradients; set
    PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE=direct for regeneration.
    """
    correlation_auxbasis = selection_auxbasis(mf, correlation_auxbasis)
    validate_selection_basis(mf, mp2_static, correlation_auxbasis)
    identity = (None if correlation_auxbasis is None else
                basis_identity(mf, correlation_auxbasis))
    from .lis import LISSelections, _validate_thresholds
    _validate_thresholds(thresh_occ, thresh_vir, internal_rank_threshold)
    if mp2_grid is None:
        saved_grid = getattr(mp2_static, 'mp2_grid', None)
        grid_diagnostics = getattr(mp2_static, 'mp2_grid_diagnostics', None)
        if saved_grid is None:
            raise ValueError('domain selections have no saved MP2 grid; rebuild selections')
    else:
        from .laplace import select_mp2_grid
        spectrum = numpy.asarray(jax.lax.stop_gradient(mf.mo_energy))
        saved_grid, grid_diagnostics = select_mp2_grid(
            spectrum[mp2_static.active_occ_indices], spectrum[mp2_static.active_vir_indices],
            grid=mp2_grid)
    baseline, matrix = _normalize_controls(baseline_controls, matrix_controls, saved_grid)
    saved_grid = (baseline['laplace_roots'].copy(), baseline['laplace_weights'].copy())
    indices = tuple(range(len(_physical_groups(mp2_static))))
    physical = _physical_groups(mp2_static)
    selections = [None] * len(physical)
    frames, grids = [None] * len(physical), [None] * len(physical)
    lis = LISSelections(mp2_static, thresh_occ, thresh_vir,
        internal_rank_threshold, tuple(selections),
        tuple(frames), tuple(grids))
    return CCSDSelections(lis, _scientific_identity(baseline, matrix),
        selection_options=dict(indices=tuple(indices),
            triples_laplace_points=triples_laplace_points),
        correlation_basis_identity=identity,
        mp2_grid=saved_grid, mp2_grid_diagnostics=grid_diagnostics)


def load_static_selections(checkpoint_dir, mol, *, domain_options=None,
                           selection_options=None):
    """Load saved Boys/domain/LIS choices without constructing new selections.

    Geometry, elements, charge, spin and basis must match the checkpoint.
    Pass requested domain settings (e.g. frozen, thresholds, boys_grouping,
    lo_kwargs) and LIS settings (thresh_occ, thresh_vir,
    internal_rank_threshold, triples_laplace_points) in the two mappings to
    validate them explicitly. Omitted options use the saved configuration.
    The calculation still validates SCF/DF/SGX, solver and implementation
    identity before restoring contributions, and rebuilds continuous response.
    """
    from ._ccsd_restart_reference import load_selections
    return load_selections(checkpoint_dir, mol, domain_options=domain_options,
                           selection_options=selection_options)


def _resume_boys_options(checkpoint_dir, mf, static):
    from ._ccsd_restart_reference import load_boys_reference
    coeff, exact = load_boys_reference(checkpoint_dir, mf, static.mp2_static,
                                       with_identity=True)
    # A restored SCF keeps the saved root without any optimizer. A caller
    # that explicitly reconverges SCF may need continuous Boys refinement
    # at that slightly different root; its discrete selections stay fixed.
    return dict(stationary_boys=exact, boys_reference_coeff=coeff)


def _select_fragment_lis(mf, common, domains, index, group, frame, forward, *,
                         thresh_occ, thresh_vir, internal_rank_threshold,
                         triples_laplace_points, prepared_lis=None, with_grad=False):
    """Select labels and build the solver LIS, optionally retaining its VJP.

    Anchors/grid are frozen choices; numerical orbitals and their response are
    returned in the call-local ``prepared_lis`` holder for solver reuse.
    """
    from .lis import LISSelections, build_fragment_lis_selection, build_fragment_lis
    matrices = {name: np.asarray(forward[name]) for name in ('occupied', 'virtual')}
    with resource_profile.section('stc.lis.select_ranks', frag=index+1):
        selection = build_fragment_lis_selection(
            mf, domains, index, matrices=matrices,
            common=common, frame=frame, frame_fragment_index=group.representative_index,
            thresh_occ=thresh_occ, thresh_vir=thresh_vir,
            internal_rank_threshold=internal_rank_threshold)
    frames = grid = None
    from .smooth import select_frame
    from .smooth_triples import make_grid
    selections = [None] * len(_physical_groups(domains))
    selections[index] = selection
    partial_lis = LISSelections(domains, thresh_occ, thresh_vir,
                               internal_rank_threshold, tuple(selections))
    chosen_frames = []
    def choose_frame(coeff, fock):
        anchor = select_frame(jax.lax.stop_gradient(coeff), jax.lax.stop_gradient(fock))
        chosen_frames.append(anchor)
        return anchor

    def build(current, data, blocks, coeff):
        current_frame = frame._replace(occupied_coeff=coeff[0], virtual_coeff=coeff[1])
        selected = build_fragment_lis(current, data, partial_lis, index,
            matrices=blocks, frame=current_frame, frame_fragment_index=group.representative_index,
            frame_selector=choose_frame)
        return _SolverLIS(selected.mo_coeff, selected.fragment_iao_coeff,
            selected.active_occupied_coeff, selected.active_virtual_coeff,
            tuple(int(i) for i in selected.frozen))

    with resource_profile.section('stc.lis.build', frag=index+1):
        arguments = (mf, common, matrices, (frame.occupied_coeff, frame.virtual_coeff))
        if with_grad and prepared_lis is not None:
            selected, pullback = jax.vjp(build, *arguments)
            prepared_lis['pullback'] = pullback
        else:
            selected = build(*arguments)
        if prepared_lis is not None:
            prepared_lis['value'] = selected
        coefficients = (selected.active_occupied_coeff, selected.active_virtual_coeff)
        jax.block_until_ready(coefficients)
        if all(coeff.shape[1] for coeff in coefficients):
            frames = tuple(chosen_frames)
            spectra = [numpy.linalg.eigvalsh(item.anchor.T @
                numpy.asarray(common.fock) @ item.anchor) for item in frames]
    if frames is not None:
        with resource_profile.section('stc.lis.fit_grid', frag=index+1):
            grid = make_grid(*spectra, points=triples_laplace_points)
    return selection, frames, grid


def _select_current_fragment(state, static, index, group, frame, forward, *,
                             prepared_lis=None, with_grad=False):
    """Create this fragment's immutable selection while its input VJP is live."""
    if static.fragments[index] is not None:
        return static
    options = static.selection_options
    with resource_profile.section('stc.cc.lis_selection', frag=index+1):
        selection, frames, grid = _select_fragment_lis(
            state['mf'], state['common'], static.mp2_static, index, group, frame, forward,
            thresh_occ=static.lis_static.thresh_occ,
            thresh_vir=static.lis_static.thresh_vir,
            internal_rank_threshold=static.lis_static.internal_rank_threshold,
            triples_laplace_points=options['triples_laplace_points'],
            prepared_lis=prepared_lis, with_grad=with_grad)
    fragments = list(static.fragments)
    fragments[index] = selection
    lis = replace(static.lis_static, fragments=tuple(fragments))
    all_frames, all_grids = list(lis.smooth_frames), list(lis.triples_grid)
    all_frames[index], all_grids[index] = frames, grid
    lis = replace(lis, smooth_frames=tuple(all_frames), triples_grid=tuple(all_grids))
    return replace(static, lis_static=lis, selection_options=None)


def _allocate_triples_controls(controls, nfragments, *, static=None, indices=None):
    """Allocate a total triples SE budget over globally requested physical fragments."""
    smooth = getattr(getattr(static, 'lis_static', None), 'smooth_frames', None) is not None
    if controls is not None and controls.get('integral_transform') == 'rotate' and not smooth:
        raise ValueError('integral_transform=rotate requires smooth triples selections')
    if controls is not None and controls.get('mode') == 'stochastic':
        if not smooth:
            raise ValueError('stochastic triples requires full-LIS smooth frames; '
                             'canonical triples sampling has been removed')
        controls = dict(controls)
        controls.setdefault('trace_exponent', DEFAULT_TRIPLES_TRACE_EXPONENT)
    if controls is None or 'energy_tolerance' not in controls:
        return controls
    plan = getattr(static, 'selection_options', None)
    allocation = controls.get('variance_allocation', 'occupied_count' if plan else 'mp2_energy')
    if plan and allocation != 'occupied_count':
        raise ValueError('the W pipeline requires occupied_count variance allocation')
    if allocation == 'occupied_count':
        if static is None or 'fragment_energies' in controls:
            raise ValueError('occupied_count allocation requires domains and no energy proxies')
        domains = static.mp2_static
        groups = _physical_groups(domains)
        counts = [sum(len(domains.frag_lolist[i]) for i in members) for members in groups]
        total = len(domains.active_occ_indices)
        if total <= 0 or min(counts) <= 0 or sum(counts) != total:
            raise ValueError('physical occupied targets must partition the active occupied space')
        indices = tuple(range(len(groups))) if indices is None else tuple(indices)
        fractions = {i: counts[i] / total for i in indices}
        return dict(controls, variance_allocation='occupied_count',
            fragment_variance_fractions=fractions,
            fragment_standard_errors={i: controls['energy_tolerance'] * value**.5
                                      for i, value in fractions.items()},
            fragment_occupied_counts={i: counts[i] for i in indices}, total_active_occupied=total)
    energies = controls.get('fragment_energies', getattr(static, 'mp2_fragment_energies', None))
    if energies is not None:
        if static is None or len(energies) != len(static.fragments):
            raise ValueError('fragment_energies must contain one energy per physical S')
        indices = tuple(range(len(energies))) if indices is None else tuple(indices)
        if not indices:
            return dict(controls, fragment_standard_errors={}, fragment_variance_fractions={})
        selected = numpy.asarray(energies, dtype=float)[list(indices)]
        if not numpy.isfinite(selected).all():
            raise ValueError('fragment energy proxies must be finite')
        weights = numpy.abs(selected)
        largest = weights.max()
        # A tiny floor keeps support if an MP2 contribution happens to vanish.
        weights = numpy.maximum(weights / largest, 1e-12) if largest else numpy.ones_like(weights)
        fractions = weights / weights.sum()
        targets = controls['energy_tolerance'] * numpy.sqrt(fractions)
        if numpy.any(targets == 0.):
            raise ValueError('energy_tolerance is too small for a nonzero per-fragment target')
        return dict(controls,
                    fragment_standard_errors=dict(zip(indices, map(float, targets))),
                    fragment_variance_fractions=dict(zip(indices, map(float, fractions))),
                    fragment_reference_energies=dict(zip(indices, map(float, selected))))
    target = controls['energy_tolerance'] / numpy.sqrt(nfragments) if nfragments else 0.
    if nfragments and target == 0.:
        raise ValueError('energy_tolerance is too small for a nonzero per-fragment target')
    return dict(controls, target_standard_error=float(target))


def _fragment_triples_controls(controls, index):
    if controls is None or 'fragment_standard_errors' not in controls:
        return controls
    local = dict(controls, target_standard_error=controls['fragment_standard_errors'][index],
                 fragment_variance_fraction=controls['fragment_variance_fractions'][index])
    if 'fragment_reference_energies' in controls:
        local['fragment_reference_energy'] = controls['fragment_reference_energies'][index]
    if 'fragment_occupied_counts' in controls:
        local['fragment_occupied_count'] = controls['fragment_occupied_counts'][index]
    return local


def _triples_fragment_seed(global_seed, fragment_index):
    """Derive a triples stream from fragment identity, independent of ownership."""
    sequence = numpy.random.SeedSequence([global_seed, 0x54524950, fragment_index])
    return int(sequence.generate_state(1, dtype=numpy.uint64)[0])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _SolverLIS:
    """Compact occupied/virtual LIS inputs with frozen labels kept static."""
    mo_coeff: object
    fragment_iao_coeff: object
    active_occupied_coeff: object
    active_virtual_coeff: object
    frozen: tuple

    def tree_flatten(self):
        return ((self.mo_coeff, self.fragment_iao_coeff,
                 self.active_occupied_coeff, self.active_virtual_coeff), self.frozen)

    @classmethod
    def tree_unflatten(cls, frozen, arrays):
        return cls(*arrays, frozen)


def _build_solver_lis(mf, common, static, fragment_index, matrices, frame, representative):
    from .lis import build_fragment_lis
    if not resource_profile.enabled():
        return build_fragment_lis(mf, common, static, fragment_index,
            matrices=matrices, frame=frame, frame_fragment_index=representative)
    # Differentiate only numerical outputs used by the solver. Frozen labels
    # come from the existing constructor and remain static pytree metadata.
    frame_metadata = frame._replace(occupied_coeff=None, virtual_coeff=None)
    def build(current, data, blocks, occupied, virtual):
        current_frame = frame_metadata._replace(occupied_coeff=occupied, virtual_coeff=virtual)
        lis = build_fragment_lis(current, data, static, fragment_index,
            matrices=blocks, frame=current_frame, frame_fragment_index=representative)
        return _SolverLIS(lis.mo_coeff, lis.fragment_iao_coeff,
            lis.active_occupied_coeff, lis.active_virtual_coeff,
            tuple(int(index) for index in lis.frozen))
    with resource_profile.section('stc.lis.build', frag=fragment_index+1):
        lis = profile_response('stc.lis.pullback', build, mf, common, matrices,
            frame.occupied_coeff, frame.virtual_coeff, frag=fragment_index+1)
        jax.block_until_ready(lis)
    return lis


def _solve_fragment(mf, common, static, fragment_index, matrices, group, *,
                    verbose_imp, ccsd_t, triples_controls=None, with_grad=False,
                    vvvv_direct=False, cc_solver='deterministic', stc_cc_controls=None,
                    frame=None, correlation_auxbasis=None, df_compressor=None,
                    sampling_frame=None, prepared_lis=None):
    """Return MP2-LIS, CCSD and triples; stochastic calls also return diagnostics."""
    representative = group.representative_index
    if frame is None:
        frame = build_boys_ed_frame(common, static.mp2_static, representative)
    lis = prepared_lis
    if lis is None:
        lis = _build_solver_lis(mf, common, static.lis_static, fragment_index,
                                matrices, frame, representative)
    stochastic = triples_controls is not None and triples_controls['mode'] == 'stochastic'
    if not lis.active_occupied_coeff.shape[1] or not lis.active_virtual_coeff.shape[1]:
        zero = np.zeros((), dtype=common.s1e.dtype)
        values = (zero, zero, zero)
        return ((values, {}) if df_compressor is not None or cc_solver == 'stc' or stochastic or
                ccsd_t else values)
    metadata = {'fragment_index': int(fragment_index),
                'extended_primary_domain': numpy.asarray(
                    static.mp2_static.fragments[representative].extended_atoms, dtype=numpy.int32),
                'auxiliary_domain': numpy.asarray(
                    static.mp2_static.fragments[representative].pao_center_atoms, dtype=numpy.int32)}
    options = basis_kwargs(correlation_auxbasis)
    options['active_nocc'] = lis.active_occupied_coeff.shape[1]
    if df_compressor is not None:
        options['df_compressor'] = df_compressor
    if cc_solver == 'stc':
        from .stc_ccsd import make_solver
        sampling_options = {} if sampling_frame is None else {'sampling_frame': sampling_frame}
        options['amplitude_solver'] = make_solver(
            stc_cc_controls, fragment_index=fragment_index, **sampling_options)
    from .smooth import noncanonical_mp2
    from .smooth_triples import make_solver
    options['mp2_solver'] = noncanonical_mp2
    if resource_profile.enabled():
        options['mp2_solver'] = lambda eris, projection: profile_response(
            'solver.mp2_reference_pullback', noncanonical_mp2, eris, projection,
            frag=fragment_index+1)
    if ccsd_t:
        local_controls = dict(triples_controls)
        local_controls['global_seed'] = _triples_fragment_seed(
            local_controls.get('global_seed', 71), fragment_index)
        options['triples_solver'] = make_solver(local_controls,
            static.lis_static.triples_grid[fragment_index],
            with_grad=with_grad)
    result = _impurity_solve_core(
        mf, lis.mo_coeff, lis.fragment_iao_coeff, common.fock, common.s1e,
        frozen=lis.frozen, frag_prescreen=metadata, verbose_imp=verbose_imp,
        ccsd_t=ccsd_t, dcsd=False, profile_info=None,
        vvvv_direct=vvvv_direct, **options)
    if df_compressor is None:
        return result
    if 'amplitude_solver' in options or 'triples_solver' in options:
        values, diagnostics = result
    else:
        values, diagnostics = result, {}
    return values, diagnostics


def _run_group(state, static, group, baseline_controls, matrix_controls, *,
               with_grad, verbose_imp, ccsd_t, triples_controls=None,
               vvvv_direct=False,
               cc_solver='deterministic', stc_cc_controls=None, lis_aux_svd_tol=None):
    """Optionally join all local DF producer responses for this domain."""
    from pyscfad.lno import _df_shared
    options = dict(with_grad=with_grad, verbose_imp=verbose_imp, ccsd_t=ccsd_t,
                   triples_controls=triples_controls,
                   vvvv_direct=vvvv_direct,
                   cc_solver=cc_solver, stc_cc_controls=stc_cc_controls)
    if lis_aux_svd_tol:
        options['lis_aux_svd_tol'] = lis_aux_svd_tol
    # Replica averaging scales returned VJP trees. Deferred integral bars in an
    # outer shared scope would escape that scaling, so use ordinary responses.
    replica_precision = cc_solver == 'stc' and _stc_precision.enabled(stc_cc_controls)
    if not hasattr(state['mf'], 'mol') or with_grad and replica_precision:
        return _run_group_responses(state, static, group, baseline_controls,
                                    matrix_controls, **options)
    atoms = static.mp2_static.fragments[group.representative_index].extended_atoms
    with _df_shared.reverse_scope(
            state['mf'], atoms, with_grad=with_grad,
            aux_atoms=static.mp2_static.fragments[group.representative_index].pao_center_atoms,
            **basis_kwargs(state.get('correlation_auxbasis'))) as shared:
        if state.get('common') is not None:
            from ._df_inputs import configure_cache
            configure_cache(shared)
        records = _run_group_responses(state, static, group, baseline_controls,
                                       matrix_controls, **options)
        with resource_profile.section('stc.cc.shared_df_integral_pullback',
                                      domain_representative=group.representative_index+1):
            coords = shared.finish()
            if with_grad:
                state['mf_bar'].mol.coords = state['mf_bar'].mol.coords + np.asarray(coords)
                jax.block_until_ready(state['mf_bar'].mol.coords)
        driver.lib.logger.info(state['mf'], 'Shared local DF reverse: %s', shared.receipt)
    return records


def _run_group_responses(state, static, group, baseline_controls, matrix_controls, *,
                        with_grad, verbose_imp, ccsd_t, triples_controls=None,
                        vvvv_direct=False,
                        cc_solver='deterministic', stc_cc_controls=None, lis_aux_svd_tol=None):
    """Complete one owned group's preparation, baseline and physical-fragment responses.

    All process-local VJPs and matrix replay records are consumed here. Only
    scalar component records and the accumulated numerical host bars survive.
    One preparation pullback consumes the sum of its input cotangents and
    direct LIS frame cotangents. The domain frame and its reverse are shared
    by domain MP2/W and every physical-fragment correction in this group.
    """
    from pyscfad.lno._df_shared import release_consumed_inputs
    with resource_profile.section('stc.cc.prepare'):
        inputs, host, prepare_pullback, frame = driver._prepare(
            state, static.mp2_static, group, baseline_controls, with_grad,
            return_frame=True)
    domains = state.pop('_prepared_domain_static', static.mp2_static)
    if domains is not static.mp2_static:
        static = replace(static, lis_static=replace(static.lis_static, mp2_static=domains))
    frame_coeff = (frame.occupied_coeff, frame.virtual_coeff)
    frame_bars = tuple(np.zeros_like(coeff) for coeff in frame_coeff) if with_grad else None
    input_bars = None
    records = []
    for index, targets in _group_cc_fragments(static, group):
        with resource_profile.section('stc.cc.matrix_forward', frag=index+1):
            forward = backend.solve_matrices(
                host, {'fragment_id': index, 'target_indices': targets},
                matrix_controls)
        state['energy'] = state['energy'] + forward['energy']
        state['baseline_energy'] = state.get('baseline_energy', 0.) + float(forward['energy'])
        state['variance'] += float(forward['diagnostics']['energy_standard_error']) ** 2
        matrices = {name: np.asarray(forward[name]) for name in ('occupied', 'virtual')}
        # Call-local orbitals and tape: share selection's construction with all
        # solver replicas, without freezing the physical LIS response.
        prepared_lis = {}
        fragment_static = _select_current_fragment(state, static, index, group, frame, forward,
            prepared_lis=prepared_lis, with_grad=with_grad)
        _clear_jax_caches('before_correction', matrices, frame_coeff, frag=index+1)
        local_triples = _fragment_triples_controls(triples_controls, index)
        sampling_frame = None
        if cc_solver == 'stc':
            from .stc_ccsd import SamplingFrame
            # Share only numerical gauge rotations across this LIS's replicas.
            # Physical factors and their responses remain live in each VJP;
            # neither another fragment nor another geometry inherits the cache.
            sampling_frame = SamplingFrame()
        pilot_mf = None
        def evaluate(configuration, need_grad):
            nonlocal pilot_mf
            solver_options = (dict(cc_solver=cc_solver, stc_cc_controls=configuration,
                                   sampling_frame=sampling_frame)
                              if cc_solver == 'stc' else {})
            compressor = None
            if lis_aux_svd_tol:
                from ._aux_compress import make_compressor
                compressor = make_compressor(lis_aux_svd_tol)
                solver_options['df_compressor'] = compressor
            def correction(mf, common, blocks, coeff, lis):
                current_frame = frame._replace(occupied_coeff=coeff[0], virtual_coeff=coeff[1])
                lis_options = {} if lis is None else {'prepared_lis': lis}
                values = _solve_fragment(mf, common, fragment_static, index, blocks, group,
                                         verbose_imp=verbose_imp, ccsd_t=ccsd_t,
                                         triples_controls=local_triples, with_grad=need_grad,
                                         vvvv_direct=vvvv_direct, frame=current_frame, **solver_options,
                                         **basis_kwargs(state.get('correlation_auxbasis')), **lis_options)
                if lis_aux_svd_tol or cc_solver == 'stc' or ccsd_t:
                    values, diagnostics = values
                else:
                    diagnostics = {}
                emp2, ecc, et = values
                return ecc + et - emp2, (values, diagnostics)
            if not need_grad:
                reference = state['mf']
                if with_grad:
                    # Plain forward pilots populate local-DF caches. Keep those
                    # mutations off the live MF whose response trees are saved.
                    if pilot_mf is None:
                        pilot_mf = jax.tree.map(lambda value: value, reference)
                        df = getattr(pilot_mf, 'with_df', None)
                        cache = getattr(df, '_lno_local_df_cache', None)
                        if cache is not None:
                            df._lno_local_df_cache = dict(cache)
                    reference = pilot_mf
                energy, auxiliary = correction(reference, state['common'], matrices, frame_coeff,
                                                prepared_lis.get('value'))
                if compressor is not None:
                    auxiliary[1]['auxiliary_compression'] = dict(compressor.diagnostics)
                return energy, auxiliary, None
            with resource_profile.section('stc.cc.correction_forward', frag=index+1):
                energy, pullback, auxiliary = jax.vjp(
                    correction, state['mf'], state['common'], matrices, frame_coeff,
                    prepared_lis.get('value'), has_aux=True)
                jax.block_until_ready((energy, auxiliary))
            _clear_jax_caches('before_correction_pullback', energy, auxiliary, frag=index+1)
            with resource_profile.section('stc.cc.correction_pullback', frag=index+1):
                bars = pullback(np.ones_like(energy))
                jax.block_until_ready(bars)
            if compressor is not None:
                auxiliary[1]['auxiliary_compression'] = dict(compressor.diagnostics)
            return energy, auxiliary, bars
        if cc_solver == 'stc' and _stc_precision.enabled(stc_cc_controls):
            nfragments = len(_all_fragment_indices(static))
            energy, auxiliary, correction_bars = _stc_precision.run_replicas(
                evaluate, stc_cc_controls, index, nfragments, with_grad=with_grad)
        else:
            energy, auxiliary, correction_bars = evaluate(stc_cc_controls, with_grad)
        if with_grad:
            direct_bars, lis_bar = correction_bars[:4], correction_bars[4]
            if lis_bar is not None:
                # Sum replica cotangents before applying the retained LIS tape
                # once. This includes W, overlap, targets and domain orbitals.
                with resource_profile.section('stc.lis.pullback', frag=index+1):
                    lis_bars = prepared_lis['pullback'](lis_bar)
                    direct_bars = jax.tree.map(lambda a, b: a + b, direct_bars, lis_bars)
                    jax.block_until_ready(direct_bars)
                del lis_bars
            mf_bar, common_bar, matrix_bars, coeff_bars = direct_bars
            del direct_bars, lis_bar
            prepared_lis.clear()
            frame_bars = tuple(left + right for left, right in zip(frame_bars, coeff_bars))
            jax.block_until_ready((mf_bar, common_bar, matrix_bars, frame_bars))
            # The direct host response holds the shared frame fixed; apply its
            # accumulated frame contribution once through preparation below.
            direct_norm = _bar_norm(mf_bar, common_bar)
            driver._add_bars(state, mf_bar, common_bar)
            # These bars have been accumulated into state/frame_bars. Release
            # their separate trees and completed CC response executables before
            # the native W reverse allocates domain-sized factor buffers.
            del correction_bars, mf_bar, common_bar, coeff_bars
            _clear_jax_caches('before_matrix_pullback', state['mf_bar'],
                              state['common_bar'], frame_bars, matrix_bars,
                              frag=index+1)
            with resource_profile.section('stc.cc.matrix_pullback', frag=index+1):
                bars = backend.pullback_matrices(
                    host, forward, energy_bar=1., occupied_bar=matrix_bars['occupied'],
                    virtual_bar=matrix_bars['virtual'])
            validate_input_cotangents(bars, inputs)
            if input_bars is None:
                # Native outputs are owned and writable, including the empty
                # fast path. Adopt them without duplicating the domain-sized
                # B bar; diagnostic wrappers may return read-only snapshots.
                input_bars = {name: value if value.flags.writeable else value.copy()
                              for name, value in bars.items()}
            else:
                for name in inputs:
                    input_bars[name] += bars[name]
            del matrix_bars, bars
        else:
            direct_norm = None
        values, diagnostics = auxiliary
        def diagnostic_value(value):
            value = numpy.asarray(value)
            return value.item() if value.ndim == 0 else value.tolist()
        diagnostics = jax.tree_util.tree_map(diagnostic_value, diagnostics)
        cc_diagnostics = diagnostics.pop('ccsd', None)
        compression_diagnostics = diagnostics.pop('auxiliary_compression', None)
        if local_triples is not None:
            diagnostics.update({key: local_triples[key] for key in (
                'target_standard_error', 'fragment_variance_fraction',
                'fragment_reference_energy', 'fragment_occupied_count',
                'total_active_occupied', 'variance_allocation') if key in local_triples})
        state['energy'] = state['energy'] + energy
        jax.block_until_ready((state['energy'], state['mf_bar'], state['common_bar']))
        records.append(FragmentComponents(index, *(float(value) for value in values),
                                           float(forward['energy']),
                                           *_fragment_dimensions(fragment_static, index),
                                           direct_norm, None,
                                           {key: value for key, value in forward['diagnostics'].items()
                                            if key not in ('occupied_standard_error',
                                                           'virtual_standard_error')},
                                           triples_method=(triples_controls['mode']
                                               if ccsd_t and triples_controls is not None else
                                               'deterministic' if ccsd_t else 'none'),
                                           triples_standard_error=float(diagnostics.get('standard_error', 0.)),
                                           triples_samples=int(diagnostics.get('samples', 0)),
                                           triples_timings={key: value for key, value in diagnostics.items()
                                                            if key.endswith('_seconds')} or None,
                                           triples_diagnostics=diagnostics or None,
                                           ccsd_method=cc_solver,
                                           ccsd_diagnostics=cc_diagnostics,
                                           auxiliary_compression=compression_diagnostics))
        del forward, matrices, evaluate, pilot_mf, energy, values, auxiliary, diagnostics, fragment_static
        del prepared_lis
        release_consumed_inputs(forward_only=not with_grad)
        gc.collect()
    if input_bars is not None:
        with resource_profile.section('stc.cc.input_pullback'):
            # AD adds the coefficient bars from DF/Fock construction to the
            # explicit LIS bars before traversing the frame producer once.
            combined_pullback = lambda bars: prepare_pullback((bars, frame_bars))
            driver._add_bars(state, *apply_input_pullback(combined_pullback, inputs, input_bars))
            del input_bars
            jax.block_until_ready((state['mf_bar'], state['common_bar']))
        release_consumed_inputs()
    return records


def _run_weak(state, static, *, with_grad, pairs=None):
    """Complete separately owned baseline weak terms once."""
    pairs = state['weak_pairs'] if pairs is None else pairs
    if not pairs:
        return
    if with_grad:
        energy, bars = driver._weak_energy_and_grad(state, static.mp2_static, pairs)
        driver._add_bars(state, *bars)
    else:
        energy = driver.weak_pair_energy_sum(
            state['mf'], state['common'], static.mp2_static, pairs=pairs)
    state['energy'] = state['energy'] + energy
    state['baseline_energy'] = state.get('baseline_energy', 0.) + float(energy)
    state['weak_energy'] = state.get('weak_energy', 0.) + float(energy)


def _run_serial(state, static, baseline, matrix, *, with_grad, include_hf,
                merge_identical, verbose_imp, ccsd_t, return_components,
                triples_controls=None,
                checkpoint_dir=None, resume=False, vvvv_direct=False,
                cc_solver='deterministic', stc_cc_controls=None, lis_aux_svd_tol=None):
    from ._ccsd_checkpoint import open_checkpoint, restore_progress, save_progress
    groups = build_domain_groups(static.mp2_static, merge_identical=merge_identical)
    options = dict(with_grad=with_grad, include_hf=include_hf, ccsd_t=ccsd_t,
                   triples_controls=triples_controls, vvvv_direct=vvvv_direct,

                   merge_identical=merge_identical)
    solver_options = (dict(cc_solver=cc_solver, stc_cc_controls=stc_cc_controls)
                      if cc_solver == 'stc' else {})
    if lis_aux_svd_tol:
        solver_options['lis_aux_svd_tol'] = lis_aux_svd_tol
    options.update(solver_options)
    checkpoint = open_checkpoint(state, static, groups, checkpoint_dir=checkpoint_dir,
                                  resume=resume, options=options)
    try:
        start, weak_done, records = restore_progress(checkpoint, state, with_grad=with_grad)
        if weak_done:
            state.pop('_prepared_weak', None)
        if not 0 <= start <= len(groups):
            raise ValueError('CC checkpoint progress is outside the saved group worklist')
        if not weak_done:
            _run_weak(state, static, with_grad=with_grad)
            save_progress(checkpoint, state, records, start, True, with_grad=with_grad)
        for position in range(start, len(groups)):
            group = groups[position]
            records.extend(_run_group(
                state, static, group, baseline, matrix, with_grad=with_grad,
                verbose_imp=verbose_imp, ccsd_t=ccsd_t,
                triples_controls=triples_controls, vvvv_direct=vvvv_direct,

                **solver_options))
            _clear_jax_caches('after_group', state['energy'], state['mf_bar'],
                              state['common_bar'], frag=group.representative_index+1)
            save_progress(checkpoint, state, records, position + 1, True, with_grad=with_grad)
            gc.collect()
        from pyscfad.tools._response_checkpoint import scope
        with scope(checkpoint):
            return _finish(state, static, records, with_grad=with_grad, include_hf=include_hf,
                           return_components=return_components)
    finally:
        if checkpoint is not None:
            checkpoint.close()


def _finish(state, static, records, *, with_grad, include_hf, return_components):
    components = CCSDComponents(state.get('baseline_energy', 0.), state.get('weak_energy', 0.),
                                tuple(sorted(records, key=lambda item: item.fragment_index)),
                                'W', state['variance'] ** .5)
    if components.stc_standard_error is not None:
        driver.lib.logger.info(state['mf'],
            'STC final selected-correction empirical SE %.6g mEh; target %.6g mEh; met %s',
            components.stc_standard_error * 1000., components.stc_target_standard_error * 1000.,
            components.stc_target_met)
    result = driver._finish(
        state, with_grad=with_grad, include_hf=include_hf,
        uncertainty_label='MP2 baseline standard error')
    if not with_grad and include_hf:
        result = result + state['mf'].e_tot
    if return_components:
        return (*result, components) if with_grad else (result, components)
    return result


def _run(initializer, worker_initializer, comm, *, checkpoint_dir, resume,
         parallel_scf_jk=True, scf_worker_initializer=None, **options):
    if comm is not None and comm.Get_size() > 1:
        from ._mpi_ccsd import _run_mpi
        def check_options():
            if checkpoint_dir is not None or resume:
                raise NotImplementedError('CC progress checkpoints currently require serial execution')
        driver._root_call(comm, comm.Get_rank(), check_options, 'CC checkpoint option validation')
        return _run_mpi(initializer, worker_initializer, comm,
                        parallel_scf_jk=parallel_scf_jk,
                        scf_worker_initializer=scf_worker_initializer, **options)
    if not isinstance(options.get('vvvv_direct', False), bool):
        raise TypeError('vvvv_direct must be a bool')
    state, static, baseline, matrix = initializer()
    _clear_jax_caches('initialization', state.get('mf'), state.get('common'))
    options['triples_controls'] = state.pop('cc_triples_controls')
    options.update(state.pop('cc_solver_options', {}))
    return _run_serial(state, static, baseline, matrix,
                       checkpoint_dir=checkpoint_dir, resume=resume, **options)


@memory_calculation
def kernel(mf, static=None, *, baseline_controls=None, matrix_controls=None, mp2_grid=None,
           ccsd_t=False, triples_controls=None, verbose_imp=0, include_hf=False,
           return_components=False,
           merge_identical=True, comm=None,
           checkpoint_dir=None, resume=False, parallel_scf_jk=True, vvvv_direct=False,
           cc_solver='deterministic', stc_cc_controls=None, correlation_auxbasis=None,
           lis_aux_svd_tol=1e-5):
    """Return correlation energy, or total energy with ``include_hf=True``.

    ``correlation_auxbasis`` selects local MP2, W and CC fitting; HF/Fock
    retain the SCF fit. Use the same basis for selection and execution.
    ``lis_aux_svd_tol`` compresses only LIS auxiliary factors with the strict
    squared-singular-value cutoff ``s**2 > tol`` (default ``1e-5``).
    Pass ``None`` or zero to disable compression.
    Domain MP2, W and orbital selection remain unchanged. Compression
    diagnostics are included in each fragment's ``auxiliary_compression``.
    ``return_components=True`` returns ``(energy, CCSDComponents)``. With W,
    strong-domain MP2 energy is read directly from its traces, in deterministic
    and stochastic modes. Matrix
    controls set its precision; there is no separate strong-domain MP2 solve.
    ``mp2_grid=(roots, weights)`` supplies MP2/W quadrature and must match
    selection. ``None`` reuses the grid saved in the selections. The separately fitted
    (T) grids come from the saved selections. The MP2 baseline includes weak
    completion.
    With ``ccsd_t=True``, omitted ``triples_controls`` selects deterministic
    MRCC symmetric half-Laplace triples (2024+ spin expression).
    It shares one connected numerator and its adjoint per quadrature point.
    Individual fragment energies generally differ between partitions; the
    whole-space energy agrees for identical spaces/amplitudes. Stochastic
    evaluation currently supports only the asymmetric partition and requires
    integer ``nsamples >= 2``. A positive
    ``energy_tolerance`` targets the total triples energy SE in Hartree and
    divides variance according to ``variance_allocation``. ``occupied_count``
    uses each physical fragment's occupied targets divided by all active
    occupied orbitals; it is required by
    the W pipeline. The legacy ``mp2_energy`` allocation uses saved
    absolute MP2 fragment energies; ``fragment_energies`` overrides those
    proxies. In the smooth path, ``pilot_samples`` defaults to 10000 per
    point and determines independent production counts.
    Omitted/zero tolerance uses fixed production counts.
    Every physical fragment contributes a correction; the complete MP2 baseline
    and weak completion are always included.
    Deterministic fragment RCCSD uses the in-memory PySCF/NumPy CPU path by
    default. Set ``pyscfad_dfccsd_native=False`` for the JAX reference path;
    ``vvvv_direct=False`` (default) caches the virtual ladder when it fits
    within 60% of remaining total memory, for both backends. ``True`` skips
    that cache and contracts resident DF factors in blocks. Native caches
    use PySCF's packed layout and live only for each host solve.
    Multi-rank calls distribute complete correlation groups and their
    fragment-local LIS selections. Shared preparation uses distributed DF J/K by default.
    Each rank must supply a matching MF skeleton; preceding SCF is caller-owned.
    ``parallel_scf_jk=True`` also distributes common-preparation DF J/K work. Serial and
    single-rank calls keep their existing behavior.

    ``cc_solver='stc'`` selects the experimental external ``stc_cc_jax``
    amplitude solver; ``stc_cc_controls`` contains its sampling/localization
    settings (unsuffixed energy fields: Hartree; explicit ``*_mEh`` aliases:
    millihartree). All other stages retain their independent controls.
    STC defaults to ``localize=True, df_rotations=True, shared_local_frame=True``: differentiable
    averaged amplitudes, projected CCSD energy and smooth triples use one fixed
    private joint DF frame with consistently transformed integrals and projector.
    ``df_rotations=False`` restores Boys-only localization; ``localize=False``
    keeps the supplied orbital and auxiliary frames.
    Triples then requires a smooth selection. ``shared_local_frame=False``
    restores amplitude backrotation to the original LIS basis. Fragment CC
    diagnostics are separate from triples and baseline uncertainty. The default
    deterministic solver does not import the optional dependency.
    ``stc_cc_controls['energy_tolerance_mEh']`` enables independent complete-solve
    replicas, allocating equal variance across the globally selected physical LIS.
    Independent pilots choose a fixed production count per LIS. Returned energies
    and response cotangents are production means; ``components.stc_standard_error``
    is their empirical combined SE. Baseline, matrices and triples contractions
    must be deterministic. This SE excludes iteration/approximation bias and does
    not bound forces; cap/unmet-target diagnostics remain explicit.
    """
    if resume and (comm is None or comm.Get_size() == 1):
        from ._ccsd_restart_reference import prepare_resume
        resume = prepare_resume(checkpoint_dir, resume)
        if resume and static is None:
            static = load_static_selections(checkpoint_dir, mf.mol)
    def initialize():
        solver_options = _cc_solver_options(cc_solver, stc_cc_controls, lis_aux_svd_tol)
        triples = normalize_triples_controls(triples_controls, ccsd_t=ccsd_t)
        selected = static
        if resume and checkpoint_dir is not None:
            from ._ccsd_restart_reference import resume_selections
            selected = resume_selections(checkpoint_dir, selected)
        baseline, matrix = _validate_run(selected, baseline_controls, matrix_controls, mp2_grid)
        indices = _all_fragment_indices(selected)
        _stc_precision.validate_stages(solver_options.get('stc_cc_controls'), baseline, matrix, triples)
        _stc_precision.validate_seed_independence(solver_options.get('stc_cc_controls'), indices)
        triples = _allocate_triples_controls(triples, len(indices), static=selected, indices=indices)
        if any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(mf)):
            raise TypeError('STC CC kernel is forward-only; use value_and_grad')
        validate_selection_basis(mf, selected, correlation_auxbasis)
        boys_options = _resume_boys_options(checkpoint_dir, mf, selected) if resume else {}
        state = driver._initialize(mf, selected.mp2_static, baseline, with_grad=False,
                                   **boys_options,
                                   **basis_kwargs(correlation_auxbasis))
        state['cc_triples_controls'] = triples
        if solver_options:
            state['cc_solver_options'] = solver_options
        return state, selected, baseline, matrix

    def initialize_worker(canonical):
        if mf is None:
            raise ValueError('CC MPI kernel requires a matching MF skeleton on every rank')
        for name, value in canonical.items():
            setattr(mf, name, np.asarray(value))
        mf.converged = True
        return mf

    return _run(initialize, initialize_worker, comm, with_grad=False, include_hf=include_hf,
                merge_identical=merge_identical, verbose_imp=verbose_imp,
                ccsd_t=ccsd_t, return_components=return_components,
                vvvv_direct=vvvv_direct,

                checkpoint_dir=checkpoint_dir, resume=resume,
                parallel_scf_jk=parallel_scf_jk, scf_worker_initializer=lambda: mf)


@memory_calculation
def value_and_grad(mol, build_mf, static=None, *, baseline_controls=None, matrix_controls=None,
                   mp2_grid=None, ccsd_t=False, triples_controls=None,
                   verbose_imp=0, include_hf=False,
                   return_components=False, merge_identical=True, comm=None, build_static=None,
                   checkpoint_dir=None, resume=False,
                   parallel_scf_jk=True, vvvv_direct=False,
                   cc_solver='deterministic', stc_cc_controls=None, correlation_auxbasis=None,
                   lis_aux_svd_tol=1e-5):
    """Return ``(energy, mol_bar)``; ``mol_bar.coords`` is a nuclear gradient.

    ``correlation_auxbasis`` follows :func:`kernel`; ``build_static`` inherits
    it on its detached reference. The live SCF DF object remains unchanged.
    ``lis_aux_svd_tol`` defaults to ``1e-5`` and follows :func:`kernel`,
    including the response of the retained auxiliary subspace at its fixed rank.
    Pass ``None`` or zero to disable compression.
    Include HF once with ``include_hf=True``. Append ``CCSDComponents`` with
    ``return_components=True``. A force is the negated coordinate gradient.
    ``build_static(mf)`` may select ranks once on the detached SCF reference.
    Common-data and implicit SCF response are accumulated once.
    ``triples_controls`` has the same independent sampling semantics as in
    :func:`kernel`; the reported triples SE estimates energy uncertainty only.
    ``vvvv_direct`` has the memory/performance semantics of :func:`kernel`.
    Multi-rank calls distribute complete correlation groups, including fragment-local
    fragment-local LIS selection. The default ``parallel_scf_jk=True``
    also distributes DF J/K work in SCF, common
    preparation and their coordinate pullbacks. Worker ``build_mf`` must accept
    ``mo_coeff_init``, ``mo_energy_init``, ``mo_occ_init`` and ``e_tot_init``,
    adopt these values and skip SCF. An attached out-of-core CDERI source
    must be available on every rank. ``parallel_scf_jk=False`` retains
    root-only SCF and shared response. All root-owned calls use the same
    DF geometry fusion, response caching, opt-in resident DF buffer and
    BLAS-controller scopes as the MP2 driver;
    the complete SCF pullback must reach its DF producer. See
    README_SERIAL_DF_SCF.md for activation and prebuilt-factor ownership.

    ``cc_solver`` and ``stc_cc_controls`` match :func:`kernel`. STC response
    uses its stochastic Lambda solve at the averaged amplitudes; energy-error
    controls do not bound nuclear-gradient errors. Physical domain/LIS response
    remains active, including the original fragment projector.
    """
    if resume and (comm is None or comm.Get_size() == 1):
        from ._ccsd_restart_reference import prepare_resume
        resume = prepare_resume(checkpoint_dir, resume)
        if resume and static is None and build_static is None:
            static = load_static_selections(checkpoint_dir, mol)
    def initialize():
        solver_options = _cc_solver_options(cc_solver, stc_cc_controls, lis_aux_svd_tol)
        triples = normalize_triples_controls(triples_controls, ccsd_t=ccsd_t)
        driver._coordinate_only(mol)
        prechecked_indices = None
        if build_static is not None:
            if static is not None:
                raise ValueError('supply either static or build_static, not both')
            if not callable(build_static):
                raise TypeError('build_static must be callable')
        else:
            early_baseline, early_matrix = _validate_run(static, baseline_controls, matrix_controls, mp2_grid)
            _stc_precision.validate_stages(solver_options.get('stc_cc_controls'), early_baseline, early_matrix, triples)
            if _stc_precision.enabled(solver_options.get('stc_cc_controls')):
                prechecked_indices = _all_fragment_indices(static)
                _stc_precision.validate_seed_independence(solver_options['stc_cc_controls'], prechecked_indices)
        local_response = driver._root_owned_scf_response(comm, parallel_scf_jk)
        with resource_profile.section('stc.domain.scf'):
            mf, scf_pullback = driver._initialize_scf(
                mol, build_mf, combine_df=local_response, share_scf_data=True)
            if resource_profile.enabled():
                jax.block_until_ready(mf)
        if local_response:
            scf_pullback = driver._serial_scf_pullback(scf_pullback)
        selected = static
        scf_data = getattr(scf_pullback, 'scf_data', None)
        scf_data_pullback = prepared_domain = None
        if build_static is not None:
            selected, scf_data, scf_data_pullback, prepared_domain = driver._select_domain_reference(
                mf, build_static, reuse_weak=not resume,
                scf_data=scf_data,
                **basis_kwargs(correlation_auxbasis))
        if resume and checkpoint_dir is not None:
            from ._ccsd_restart_reference import resume_selections
            selected = resume_selections(checkpoint_dir, selected)
            # Rebuild against the saved Boys anchors and quadratures. Never
            # apply stored cotangents to a newly selected orbital frame.
            prepared_domain = None
        validate_selection_basis(mf, selected, correlation_auxbasis)
        baseline, matrix = _validate_run(selected, baseline_controls, matrix_controls, mp2_grid)
        indices = (_all_fragment_indices(selected)
                   if prechecked_indices is None else prechecked_indices)
        _stc_precision.validate_stages(solver_options.get('stc_cc_controls'), baseline, matrix, triples)
        if build_static is not None:
            _stc_precision.validate_seed_independence(solver_options.get('stc_cc_controls'), indices)
        triples = _allocate_triples_controls(triples, len(indices), static=selected, indices=indices)
        boys_options = _resume_boys_options(checkpoint_dir, mf, selected) if resume else {}
        state = driver._initialize(mf, selected.mp2_static, baseline, with_grad=True,
                                   scf_pullback=scf_pullback, scf_data=scf_data,
                                   scf_data_pullback=scf_data_pullback,
                                   prepared_domain=prepared_domain,
                                   **boys_options,
                                   **basis_kwargs(correlation_auxbasis))
        state['cc_triples_controls'] = triples
        if solver_options:
            state['cc_solver_options'] = solver_options
        return state, selected, baseline, matrix

    def initialize_worker(canonical):
        driver._coordinate_only(mol)
        return build_mf(mol, **{name + '_init': value for name, value in canonical.items()})

    def initialize_scf_worker():
        nao = mol.nao
        occupation = numpy.zeros(nao)
        occupation[:mol.nelectron // 2] = 2.0
        return initialize_worker({
            'mo_coeff': numpy.eye(nao), 'mo_energy': numpy.zeros(nao),
            'mo_occ': occupation, 'e_tot': numpy.asarray(0.0),
        })

    return _run(initialize, initialize_worker, comm, with_grad=True, include_hf=include_hf,
                merge_identical=merge_identical, verbose_imp=verbose_imp,
                ccsd_t=ccsd_t, return_components=return_components,
                vvvv_direct=vvvv_direct,

                checkpoint_dir=checkpoint_dir, resume=resume,
                parallel_scf_jk=parallel_scf_jk,
                scf_worker_initializer=initialize_scf_worker)
