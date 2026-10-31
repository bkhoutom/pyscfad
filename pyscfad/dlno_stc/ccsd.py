"""Physical Boys-fragment CCSD(T) corrections to the separately controlled STC baseline.

Discrete domains and LIS ranks are fixed. Native D/W chooses each LIS in its
STC frame; the shared Python solver retains the original Boys energy target.
"""
from dataclasses import dataclass
import gc

import jax
import numpy
from pyscfad.tools import resource_profile
from pyscfad import numpy as np
from pyscfad.lno.ccsd import _impurity_solve_core

from . import backend, driver
from ._memory import memory_calculation
from ._selection import (DomainSelections, _physical_groups, _strong_preparation_key,
                         build_domain_groups)
from ._workflow import allocate_domain_controls
from .adjoint import apply_input_pullback
from .controls import (normalize_matrix_controls, normalize_triples_controls,
                       DEFAULT_TRIPLES_TRACE_EXPONENT)
from .domain import build_boys_ed_frame
from .protocol import validate_full_result, validate_input_cotangents


CC_SELECTIONS_SCHEMA_VERSION = 6


@dataclass(frozen=True)
class CCSDSelections:
    """Fixed LIS choices and the scientific settings used to select them."""
    lis_static: object
    controls_identity: tuple
    schema_version: int = CC_SELECTIONS_SCHEMA_VERSION
    mp2_fragment_energies: tuple | None = None

    @property
    def mp2_static(self):
        return self.lis_static.mp2_static

    @property
    def matrix_kind(self):
        return self.lis_static.matrix_kind

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

    @property
    def e_ccsd_t(self):
        return self.e_triples


@dataclass(frozen=True)
class CCSDComponents:
    """Forward correlation accounting; matrix energies are diagnostics only."""
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


def _normalize_controls(baseline_controls, matrix_controls, matrix_kind):
    baseline = driver._driver_controls(baseline_controls, scope='domain')
    baseline.setdefault('mode', 'deterministic')
    baseline.setdefault('global_seed', 0)
    return baseline, normalize_matrix_controls(matrix_controls, matrix_kind)


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
    return freeze(baseline), freeze(matrix)


def _validate_run(static, baseline_controls, matrix_controls, matrix_kind):
    if not isinstance(static, CCSDSelections):
        raise TypeError('static must be CCSDSelections from build_static_selections')
    if vars(static).get('schema_version') != CC_SELECTIONS_SCHEMA_VERSION:
        raise ValueError('incompatible CC selections; rebuild selections for full-Fock smooth frames')
    kind = static.matrix_kind if matrix_kind is None else matrix_kind
    if kind != static.matrix_kind:
        raise ValueError('matrix_kind differs from the saved LIS selection')
    baseline, matrix = _normalize_controls(baseline_controls, matrix_controls, kind)
    if _scientific_identity(baseline, matrix) != static.controls_identity:
        raise ValueError('scientific controls differ from the saved LIS selection')
    return baseline, matrix


def _fragment_indices(static, indices, *, selecting=False):
    fragments = (_physical_groups(static) if selecting and isinstance(static, DomainSelections)
                 else static.fragments)
    available = tuple(i for i, fragment in enumerate(fragments)
                      if selecting or fragment is not None)
    if indices is None:
        return available
    indices = tuple(indices)
    if any(isinstance(i, (bool, numpy.bool_)) or not isinstance(i, (int, numpy.integer))
           for i in indices):
        raise TypeError('fragment_indices must contain integers')
    if len(set(indices)) != len(indices) or any(i not in available for i in indices):
        raise ValueError('fragment_indices must be unique available LIS selections')
    return tuple(int(i) for i in indices)


def _group_cc_fragments(static, group):
    """Map physical IDs to target slots in their first member's preparation job.

    Preparation merging affects ownership only. Every physical member must
    have exactly the representative's ordered strong frame, even when the
    preparation jobs themselves are left unmerged.
    """
    domains = static.mp2_static if hasattr(static, 'mp2_static') else static
    if not isinstance(domains, DomainSelections):
        raise TypeError('physical CC grouping requires STC Boys DomainSelections')
    physical = _physical_groups(domains)
    owned = [(index, members) for index, members in enumerate(physical)
             if members[0] in group.fragment_indices]
    if not owned:
        return ()
    labels = tuple(int(columns[0]) for columns in domains.frag_lolist)
    representative = group.representative_index
    frame_key, _, _ = _strong_preparation_key(
        domains, domains.fragments[representative], representative, labels)
    partners = tuple(int(i) for i in domains.fragments[representative].strong_fragments)
    result = []
    for index, members in owned:
        targets = []
        for member in members:
            key, _, _ = _strong_preparation_key(domains, domains.fragments[member], member, labels)
            if key != frame_key or partners.count(member) != 1:
                raise ValueError(f'physical CC fragment {index} has incompatible preparation frames; rebuild domains')
            targets.append(partners.index(member))
        if len(set(targets)) != len(members):
            raise ValueError(f'physical CC fragment {index} must map each target exactly once')
        result.append((index, tuple(targets)))
    return tuple(result)


def _fragment_dimensions(static, index):
    if not isinstance(static, CCSDSelections):
        return None, None
    item = static.fragments[index]
    occ = (len(static.mp2_static.active_occ_indices) if item.full_occupied_space else
           len(item.internal_occ_keep) + len(item.occupied_lno_keep))
    vir = (len(static.mp2_static.active_vir_indices) if item.full_virtual_space else
           len(item.internal_vir_keep) + len(item.virtual_lno_keep))
    return occ, vir


def _bar_norm(*trees):
    squares = [np.sum(np.abs(leaf)**2) for leaf in jax.tree_util.tree_leaves(trees)
               if hasattr(leaf, 'dtype') and leaf.dtype != jax.dtypes.float0]
    return float(np.sqrt(sum(squares)))


def build_static_selections(mf, mp2_static, *, matrix_kind, thresh_occ, thresh_vir,
                            baseline_controls, matrix_controls,
                            internal_rank_threshold=1e-6, merge_identical=True, fragment_indices=None,
                            smooth=True, smooth_laplace_points=10):
    """Select physical-fragment LIS ranks from explicit D or W cutoffs and controls.

    ``mp2_static`` is a concrete Boys ``DomainSelections``. D and W use
    separately supplied absolute-eigenvalue cutoffs, with no conversion.
    ``fragment_indices`` selects only those LIS ranks; unselected entries
    remain absent while all domain metadata is retained.

    Smooth anchored frames with full occupied and virtual Fock matrices are
    the default. Set ``smooth=False`` only for a deterministic semicanonical reference.
    D/W subspace and AO overlap response are retained; there are no internal
    Fock cluster projectors. Ranks, anchors and quadratures are fixed here.
    ``smooth_laplace_points`` defaults to ten fitted points per LIS.
    MP2 fragment proxies from D/W selection allocate triples variance budgets.
    """
    from .lis import LISSelections, build_fragment_lis_selection, build_fragment_lis
    if not isinstance(smooth, (bool, numpy.bool_)):
        raise TypeError('smooth must be a boolean')
    baseline, matrix = _normalize_controls(baseline_controls, matrix_controls, matrix_kind)
    state = driver._initialize(mf, mp2_static, baseline, with_grad=False)
    indices = _fragment_indices(mp2_static, fragment_indices, selecting=True)
    physical = _physical_groups(mp2_static)
    selections = [None] * len(physical)
    frames, grids = [None] * len(physical), [None] * len(physical)
    mp2_energies = [0.] * len(physical)
    groups = build_domain_groups(mp2_static, merge_identical=merge_identical)
    driver.lib.logger.info(mf, 'STC CC physical fragments %d; Boys grouping %s; preparation groups %d',
                           len(physical), mp2_static.boys_grouping, len(groups))
    for index, members in enumerate(physical):
        targets = tuple(int(mp2_static.frag_lolist[member][0]) for member in members)
        driver.lib.logger.info(mf, 'STC CC fragment %d; orbital records %s; targets %s; n_F=%d',
                               index, members, targets, len(members))
    for group in groups:
        corrections = _group_cc_fragments(mp2_static, group)
        if not any(index in indices for index, _ in corrections):
            continue
        inputs, host, _ = driver._prepare(state, mp2_static, group, baseline, False)
        frame = build_boys_ed_frame(state['common'], mp2_static, group.representative_index)
        for index, targets in corrections:
            if index not in indices:
                continue
            forward = backend.solve_matrices(host, {'fragment_id': index, 'target_indices': targets},
                                             matrix, matrix_kind=matrix_kind)
            mp2_energies[index] = float(forward['energy'])
            selections[index] = build_fragment_lis_selection(
                mf, mp2_static, index,
                matrices={name: np.asarray(forward[name]) for name in ('occupied', 'virtual')},
                matrix_kind=matrix_kind, common=state['common'], frame=frame,
                frame_fragment_index=group.representative_index,
                thresh_occ=thresh_occ, thresh_vir=thresh_vir,
                internal_rank_threshold=internal_rank_threshold)
            if smooth:
                from .smooth import select_frame
                from .smooth_triples import make_grid
                partial_lis = LISSelections(mp2_static, matrix_kind, thresh_occ, thresh_vir,
                                           internal_rank_threshold, tuple(selections))
                selected = build_fragment_lis(mf, state['common'], partial_lis, index,
                    matrices={name: np.asarray(forward[name]) for name in ('occupied', 'virtual')},
                    frame=frame, frame_fragment_index=group.representative_index)
                coefficients = (selected.active_occupied_coeff, selected.active_virtual_coeff)
                if all(coeff.shape[1] for coeff in coefficients):
                    frames[index] = tuple(select_frame(coeff, state['common'].fock)
                                          for coeff in coefficients)
                    spectra = [numpy.linalg.eigvalsh(item.anchor.T @
                        numpy.asarray(state['common'].fock) @ item.anchor) for item in frames[index]]
                    grids[index] = make_grid(*spectra, points=smooth_laplace_points)
                del selected, partial_lis
            del forward
        del inputs, host, frame
        gc.collect()
    lis = LISSelections(mp2_static=mp2_static, matrix_kind=matrix_kind,
                        thresh_occ=thresh_occ, thresh_vir=thresh_vir,
                        internal_rank_threshold=internal_rank_threshold,
                        fragments=tuple(selections),
                        smooth_frames=tuple(frames) if smooth else None,
                        smooth_grid=tuple(grids) if smooth else None)
    static = CCSDSelections(lis, _scientific_identity(baseline, matrix),
                            mp2_fragment_energies=tuple(mp2_energies))
    for index in indices:
        driver.lib.logger.info(mf, 'STC CC fragment %d; LIS occupied %d; LIS virtual %d',
                               index, *_fragment_dimensions(static, index))
    return static


def _allocate_triples_controls(controls, nfragments, *, static=None, indices=None):
    """Allocate a total triples SE budget over globally requested physical fragments."""
    smooth = getattr(getattr(static, 'lis_static', None), 'smooth_frames', None) is not None
    if controls is not None and controls.get('mode') == 'stochastic':
        if not smooth:
            raise ValueError('stochastic triples requires full-LIS smooth frames; '
                             'canonical triples sampling has been removed')
        controls = dict(controls)
        controls.setdefault('trace_exponent', DEFAULT_TRIPLES_TRACE_EXPONENT)
    if controls is None or 'energy_tolerance' not in controls:
        return controls
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
    return dict(controls, target_standard_error=controls['fragment_standard_errors'][index],
                fragment_variance_fraction=controls['fragment_variance_fractions'][index],
                fragment_reference_energy=controls['fragment_reference_energies'][index])


def _triples_fragment_seed(global_seed, fragment_index):
    """Derive a triples stream from fragment identity, independent of ownership."""
    sequence = numpy.random.SeedSequence([global_seed, 0x54524950, fragment_index])
    return int(sequence.generate_state(1, dtype=numpy.uint64)[0])


def _solve_fragment(mf, common, static, fragment_index, matrices, group, *,
                    verbose_imp, ccsd_t, triples_controls=None, with_grad=False):
    """Return MP2-LIS, CCSD and triples; stochastic calls also return diagnostics."""
    from .lis import build_fragment_lis
    representative = group.representative_index
    frame = build_boys_ed_frame(common, static.mp2_static, representative)
    lis = build_fragment_lis(mf, common, static.lis_static, fragment_index,
                             matrices=matrices, frame=frame,
                             frame_fragment_index=representative)
    stochastic = triples_controls is not None and triples_controls['mode'] == 'stochastic'
    if not lis.active_occupied_coeff.shape[1] or not lis.active_virtual_coeff.shape[1]:
        zero = np.zeros((), dtype=common.s1e.dtype)
        values = (zero, zero, zero)
        return (values, {}) if stochastic or (ccsd_t and static.lis_static.smooth_frames is not None) else values
    metadata = {'fragment_index': int(fragment_index),
                'extended_primary_domain': numpy.asarray(
                    static.mp2_static.fragments[representative].extended_atoms, dtype=numpy.int32)}
    options = {}
    if static.lis_static.smooth_frames is not None:
        from .smooth import noncanonical_mp2
        from .smooth_triples import make_solver
        options['mp2_solver'] = noncanonical_mp2
        if ccsd_t:
            local_controls = dict(triples_controls)
            local_controls['global_seed'] = _triples_fragment_seed(
                local_controls.get('global_seed', 71), fragment_index)
            options['triples_solver'] = make_solver(local_controls,
                static.lis_static.smooth_grid[fragment_index],
                with_grad=with_grad)
    elif stochastic:
        raise ValueError('canonical triples sampling has been removed; rebuild smooth selections')
    return _impurity_solve_core(
        mf, lis.mo_coeff, lis.fragment_iao_coeff, common.fock, common.s1e,
        frozen=lis.frozen, frag_prescreen=metadata, verbose_imp=verbose_imp,
        ccsd_t=ccsd_t, dcsd=False, profile_info=None, **options)


def _run_group(state, static, group, baseline_controls, matrix_controls, *,
               with_grad, verbose_imp, ccsd_t, triples_controls=None,
               include_baseline=True, fragment_indices=None):
    """Complete one owned group's preparation, baseline and physical-fragment responses.

    All process-local VJPs and matrix replay records are consumed here. Only
    scalar component records and the accumulated numerical host bars survive.
    One preparation pullback consumes the sum of its input cotangents.
    """
    with resource_profile.section('stc.cc.prepare'):
        inputs, host, prepare_pullback = driver._prepare(
            state, static.mp2_static, group, baseline_controls, with_grad)
    metadata = {'fragment_id': group.representative_index,
                'target_indices': group.target_indices}
    input_bars = None
    if include_baseline:
        with resource_profile.section('stc.cc.baseline'):
            baseline = backend.solve(host, metadata, baseline_controls, with_grad=with_grad)
        state['energy'] = state['energy'] + baseline['energy']
        state['baseline_energy'] = state.get('baseline_energy', 0.) + float(baseline['energy'])
        state['variance'] += float(baseline['energy_standard_error']) ** 2
        if with_grad:
            validate_full_result(baseline, inputs)
            input_bars = {name: numpy.array(value, dtype=numpy.float64, copy=True)
                          for name, value in baseline['cotangents'].items()}
        del baseline
    records = []
    for index, targets in _group_cc_fragments(static, group):
        if fragment_indices is not None and index not in fragment_indices:
            continue
        with resource_profile.section('stc.cc.matrix_forward', frag=index+1):
            forward = backend.solve_matrices(
                host, {'fragment_id': index, 'target_indices': targets},
                matrix_controls, matrix_kind=static.matrix_kind)
        matrices = {name: np.asarray(forward[name]) for name in ('occupied', 'virtual')}
        local_triples = _fragment_triples_controls(triples_controls, index)
        def correction(mf, common, blocks):
            values = _solve_fragment(mf, common, static, index, blocks, group,
                                     verbose_imp=verbose_imp, ccsd_t=ccsd_t,
                                     triples_controls=local_triples, with_grad=with_grad)
            if ccsd_t and (getattr(getattr(static, 'lis_static', None), 'smooth_frames', None) is not None or
                          triples_controls is not None and triples_controls['mode'] == 'stochastic'):
                values, diagnostics = values
            else:
                diagnostics = {}
            emp2, ecc, et = values
            return ecc + et - emp2, (values, diagnostics)
        if with_grad:
            with resource_profile.section('stc.cc.correction_forward', frag=index+1):
                energy, correction_pullback, auxiliary = jax.vjp(
                    correction, state['mf'], state['common'], matrices, has_aux=True)
                jax.block_until_ready((energy, auxiliary))
            with resource_profile.section('stc.cc.correction_pullback', frag=index+1):
                mf_bar, common_bar, matrix_bars = correction_pullback(np.ones_like(energy))
                jax.block_until_ready((mf_bar, common_bar, matrix_bars))
            direct_norm = _bar_norm(mf_bar, common_bar)
            driver._add_bars(state, mf_bar, common_bar)
            with resource_profile.section('stc.cc.matrix_pullback', frag=index+1):
                bars = backend.pullback_matrices(
                    host, forward, energy_bar=0., occupied_bar=matrix_bars['occupied'],
                    virtual_bar=matrix_bars['virtual'])
            validate_input_cotangents(bars, inputs)
            if input_bars is None:
                input_bars = {name: numpy.array(value, dtype=numpy.float64, copy=True)
                              for name, value in bars.items()}
            else:
                for name in inputs:
                    input_bars[name] += bars[name]
            del correction_pullback, mf_bar, common_bar, matrix_bars, bars
        else:
            direct_norm = None
            energy, auxiliary = correction(state['mf'], state['common'], matrices)
        values, diagnostics = auxiliary
        def diagnostic_value(value):
            value = numpy.asarray(value)
            return value.item() if value.ndim == 0 else value.tolist()
        diagnostics = jax.tree_util.tree_map(diagnostic_value, diagnostics)
        if local_triples is not None:
            diagnostics.update({key: local_triples[key] for key in (
                'target_standard_error', 'fragment_variance_fraction',
                'fragment_reference_energy') if key in local_triples})
        state['energy'] = state['energy'] + energy
        jax.block_until_ready((state['energy'], state['mf_bar'], state['common_bar']))
        records.append(FragmentComponents(index, *(float(value) for value in values),
                                           float(forward['energy']),
                                           *_fragment_dimensions(static, index),
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
                                           triples_diagnostics=diagnostics or None))
        del forward, matrices, correction, energy, values, auxiliary, diagnostics
        gc.collect()
    if input_bars is not None:
        with resource_profile.section('stc.cc.input_pullback'):
            driver._add_bars(state, *apply_input_pullback(prepare_pullback, inputs, input_bars))
            del input_bars
            jax.block_until_ready((state['mf_bar'], state['common_bar']))
    return records


def _run_weak(state, static, *, with_grad, pairs=None):
    """Complete separately owned baseline weak terms once."""
    pairs = state['weak_pairs'] if pairs is None else pairs
    if not pairs:
        return
    if with_grad:
        energy, bars = driver.weak_pair_energy_and_grad(
            state['mf'], state['common'], static.mp2_static, pairs=pairs)
        driver._add_bars(state, *bars)
    else:
        energy = driver.weak_pair_energy_sum(
            state['mf'], state['common'], static.mp2_static, pairs=pairs)
    state['energy'] = state['energy'] + energy
    state['baseline_energy'] = state.get('baseline_energy', 0.) + float(energy)
    state['weak_energy'] = state.get('weak_energy', 0.) + float(energy)


def _run_serial(state, static, baseline, matrix, *, with_grad, include_hf,
                merge_identical, verbose_imp, ccsd_t, return_components,
                triples_controls=None, fragment_indices=None, include_baseline=True,
                checkpoint_dir=None, resume=False):
    from ._ccsd_checkpoint import open_checkpoint, restore_progress, save_progress
    groups = build_domain_groups(static.mp2_static, merge_identical=merge_identical)
    allocated = allocate_domain_controls(baseline, len(groups))
    options = dict(with_grad=with_grad, include_hf=include_hf, ccsd_t=ccsd_t,
                   triples_controls=triples_controls,
                   include_baseline=include_baseline, fragment_indices=fragment_indices,
                   merge_identical=merge_identical)
    checkpoint = open_checkpoint(state, static, groups, checkpoint_dir=checkpoint_dir,
                                  resume=resume, options=options)
    try:
        start, weak_done, records = restore_progress(checkpoint, state, with_grad=with_grad)
        if not 0 <= start <= len(groups):
            raise ValueError('CC checkpoint progress is outside the saved group worklist')
        for position in range(start, len(groups)):
            group = groups[position]
            corrections = _group_cc_fragments(static, group)
            if include_baseline or any(fragment_indices is None or index in fragment_indices
                                       for index, _ in corrections):
                records.extend(_run_group(
                    state, static, group, allocated, matrix, with_grad=with_grad,
                    verbose_imp=verbose_imp, ccsd_t=ccsd_t,
                    triples_controls=triples_controls,
                    include_baseline=include_baseline, fragment_indices=fragment_indices))
            save_progress(checkpoint, state, records, position + 1, False, with_grad=with_grad)
            gc.collect()
        if not weak_done:
            if include_baseline:
                _run_weak(state, static, with_grad=with_grad)
            save_progress(checkpoint, state, records, len(groups), True, with_grad=with_grad)
        return _finish(state, static, records, with_grad=with_grad, include_hf=include_hf,
                       return_components=return_components)
    finally:
        if checkpoint is not None:
            checkpoint.close()


def _finish(state, static, records, *, with_grad, include_hf, return_components):
    components = CCSDComponents(state.get('baseline_energy', 0.), state.get('weak_energy', 0.),
                                tuple(sorted(records, key=lambda item: item.fragment_index)),
                                static.matrix_kind, state['variance'] ** .5)
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
    state, static, baseline, matrix = initializer()
    options['fragment_indices'] = state.pop('cc_fragment_indices')
    options['triples_controls'] = state.pop('cc_triples_controls')
    return _run_serial(state, static, baseline, matrix,
                       checkpoint_dir=checkpoint_dir, resume=resume, **options)


@memory_calculation
def kernel(mf, static, *, baseline_controls, matrix_controls, matrix_kind=None,
           ccsd_t=False, triples_controls=None, verbose_imp=0, include_hf=False,
           return_components=False,
           merge_identical=True, comm=None, fragment_indices=None, include_baseline=True,
           checkpoint_dir=None, resume=False, parallel_scf_jk=True):
    """Return correlation energy, or total energy with ``include_hf=True``.

    ``return_components=True`` returns ``(energy, CCSDComponents)``. The MP2
    baseline includes weak completion and is independent of matrix precision.
    With ``ccsd_t=True``, omitted ``triples_controls`` preserves deterministic
    triples; stochastic mode requires integer ``nsamples >= 2``. A positive
    ``energy_tolerance`` targets the total triples energy SE in Hartree and
    divides variance in proportion to saved absolute MP2 fragment energies.
    ``fragment_energies`` overrides those proxies with reference correlation
    energies. In the smooth path, ``pilot_samples`` defaults to 10000 per
    point and determines independent production counts. In the conventional
    path it overrides the pilot count otherwise supplied by ``nsamples``.
    Omitted/zero tolerance uses fixed production counts.
    ``fragment_indices`` restricts corrections (default: all saved LIS ranks).
    The complete baseline remains included unless ``include_baseline=False``;
    that option returns only selected CC-minus-MP2-LIS corrections and forbids HF.
    Multi-rank calls distribute common-preparation DF J/K work by default.
    Each rank must supply a matching MF skeleton; preceding SCF is caller-owned.
    ``parallel_scf_jk=False`` retains root-only shared work. Serial and
    single-rank calls keep their existing behavior.
    """
    def initialize():
        triples = normalize_triples_controls(triples_controls, ccsd_t=ccsd_t)
        if include_hf and not include_baseline:
            raise ValueError('include_hf requires include_baseline=True')
        baseline, matrix = _validate_run(static, baseline_controls, matrix_controls, matrix_kind)
        indices = _fragment_indices(static, fragment_indices)
        triples = _allocate_triples_controls(triples, len(indices), static=static, indices=indices)
        if any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(mf)):
            raise TypeError('STC CC kernel is forward-only; use value_and_grad')
        state = driver._initialize(mf, static.mp2_static, baseline, with_grad=False)
        state['cc_fragment_indices'] = indices
        state['cc_triples_controls'] = triples
        return state, static, baseline, matrix

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
                fragment_indices=fragment_indices, include_baseline=include_baseline,
                checkpoint_dir=checkpoint_dir, resume=resume,
                parallel_scf_jk=parallel_scf_jk, scf_worker_initializer=lambda: mf)


@memory_calculation
def value_and_grad(mol, build_mf, static=None, *, baseline_controls, matrix_controls,
                   matrix_kind=None, ccsd_t=False, triples_controls=None,
                   verbose_imp=0, include_hf=False,
                   return_components=False, merge_identical=True, comm=None, build_static=None,
                   fragment_indices=None, include_baseline=True, checkpoint_dir=None, resume=False,
                   parallel_scf_jk=True):
    """Return ``(energy, mol_bar)``; ``mol_bar.coords`` is a nuclear gradient.

    Include HF once with ``include_hf=True``. Append ``CCSDComponents`` with
    ``return_components=True``. A force is the negated coordinate gradient.
    ``build_static(mf)`` may select ranks once on the detached SCF reference.
    Subset and ``include_baseline`` semantics match :func:`kernel`; even a
    correction-only subset closes common-data and implicit SCF response once.
    ``triples_controls`` has the same independent sampling semantics as in
    :func:`kernel`; the reported triples SE estimates energy uncertainty only.
    Multi-rank calls distribute DF J/K work in SCF, common preparation and
    their coordinate pullbacks by default. Worker ``build_mf`` must accept
    ``mo_coeff_init``, ``mo_energy_init``, ``mo_occ_init`` and ``e_tot_init``,
    adopt these values and skip SCF. An attached out-of-core CDERI source
    must be available on every rank. ``parallel_scf_jk=False`` retains
    root-only SCF and shared response. Serial and single-rank calls keep
    their existing behavior.
    """
    def initialize():
        triples = normalize_triples_controls(triples_controls, ccsd_t=ccsd_t)
        driver._coordinate_only(mol)
        if include_hf and not include_baseline:
            raise ValueError('include_hf requires include_baseline=True')
        if build_static is not None:
            if static is not None:
                raise ValueError('supply either static or build_static, not both')
            if not callable(build_static):
                raise TypeError('build_static must be callable')
        else:
            _validate_run(static, baseline_controls, matrix_controls, matrix_kind)
        mf, scf_pullback = driver._initialize_scf(mol, build_mf)
        selected = static
        scf_data = scf_data_pullback = None
        if build_static is not None:
            selected, scf_data, scf_data_pullback = driver._select_domain_reference(mf, build_static)
        baseline, matrix = _validate_run(selected, baseline_controls, matrix_controls, matrix_kind)
        indices = _fragment_indices(selected, fragment_indices)
        triples = _allocate_triples_controls(triples, len(indices), static=selected, indices=indices)
        state = driver._initialize(mf, selected.mp2_static, baseline, with_grad=True,
                                   scf_pullback=scf_pullback, scf_data=scf_data,
                                   scf_data_pullback=scf_data_pullback)
        state['cc_fragment_indices'] = indices
        state['cc_triples_controls'] = triples
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
                fragment_indices=fragment_indices, include_baseline=include_baseline,
                checkpoint_dir=checkpoint_dir, resume=resume,
                parallel_scf_jk=parallel_scf_jk,
                scf_worker_initializer=initialize_scf_worker)
