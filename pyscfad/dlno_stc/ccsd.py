"""Singleton Boys CCSD(T) corrections to the separately controlled STC baseline.

Discrete domains and LIS ranks are fixed. Native D/W chooses each LIS in its
STC frame; the shared Python solver retains the original Boys energy target.
"""
from dataclasses import dataclass
import gc

import jax
import numpy
from pyscfad import numpy as np
from pyscfad.lno.ccsd import _impurity_solve_core
from pyscfad.ops import stop_trace

from . import backend, driver
from ._selection import build_domain_groups
from ._workflow import allocate_domain_controls
from .adjoint import apply_input_pullback, apply_selected_pullback
from .controls import normalize_matrix_controls
from .domain import build_boys_ed_frame


@dataclass(frozen=True)
class CCSDSelections:
    """Fixed LIS choices and the scientific settings used to select them."""
    lis_static: object
    controls_identity: tuple

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
    kind = static.matrix_kind if matrix_kind is None else matrix_kind
    if kind != static.matrix_kind:
        raise ValueError('matrix_kind differs from the saved LIS selection')
    baseline, matrix = _normalize_controls(baseline_controls, matrix_controls, kind)
    if _scientific_identity(baseline, matrix) != static.controls_identity:
        raise ValueError('scientific controls differ from the saved LIS selection')
    return baseline, matrix


def _fragment_indices(static, indices, *, selecting=False):
    available = tuple(i for i, fragment in enumerate(static.fragments)
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
                            internal_rank_threshold=1e-6, merge_identical=True, fragment_indices=None):
    """Select singleton LIS ranks from explicit D or W cutoffs and controls.

    ``mp2_static`` is a concrete Boys ``DomainSelections``. D and W use
    separately supplied absolute-eigenvalue cutoffs, with no conversion.
    ``fragment_indices`` selects only those LIS ranks; unselected entries
    remain absent while all domain metadata is retained.
    """
    from .lis import LISSelections, build_fragment_lis_selection
    baseline, matrix = _normalize_controls(baseline_controls, matrix_controls, matrix_kind)
    state = driver._initialize(mf, mp2_static, baseline, with_grad=False)
    indices = _fragment_indices(mp2_static, fragment_indices, selecting=True)
    selections = [None] * len(mp2_static.fragments)
    for group in build_domain_groups(mp2_static, merge_identical=merge_identical):
        if not set(group.fragment_indices).intersection(indices):
            continue
        inputs, host, _ = driver._prepare(state, mp2_static, group, baseline, False)
        for index, target in zip(group.fragment_indices, group.target_indices):
            if index not in indices:
                continue
            frame = build_boys_ed_frame(state['common'], mp2_static, index)
            forward = backend.solve_matrices(host, {'fragment_id': index, 'target_indices': (target,)},
                                             matrix, matrix_kind=matrix_kind)
            selections[index] = build_fragment_lis_selection(
                mf, mp2_static, index,
                matrices={name: np.asarray(forward[name]) for name in ('occupied', 'virtual')},
                matrix_kind=matrix_kind, common=state['common'], frame=frame,
                thresh_occ=thresh_occ, thresh_vir=thresh_vir,
                internal_rank_threshold=internal_rank_threshold)
            del forward
        del inputs, host, frame
        gc.collect()
    lis = LISSelections(mp2_static=mp2_static, matrix_kind=matrix_kind,
                        thresh_occ=thresh_occ, thresh_vir=thresh_vir,
                        internal_rank_threshold=internal_rank_threshold,
                        fragments=tuple(selections))
    return CCSDSelections(lis, _scientific_identity(baseline, matrix))


def _solve_fragment(mf, common, static, fragment_index, matrices, group, *,
                    verbose_imp, ccsd_t):
    """Return exact-denominator MP2-LIS, CCSD and optional triples terms."""
    from .lis import build_fragment_lis
    frame = build_boys_ed_frame(common, static.mp2_static, fragment_index)
    lis = build_fragment_lis(mf, common, static.lis_static, fragment_index,
                             matrices=matrices, frame=frame)
    if not lis.active_occupied_coeff.shape[1] or not lis.active_virtual_coeff.shape[1]:
        zero = np.zeros((), dtype=common.s1e.dtype)
        return zero, zero, zero
    metadata = {'fragment_index': int(fragment_index),
                'extended_primary_domain': numpy.asarray(
                    static.mp2_static.fragments[fragment_index].extended_atoms, dtype=numpy.int32)}
    return _impurity_solve_core(
        mf, lis.mo_coeff, lis.fragment_iao_coeff, common.fock, common.s1e,
        frozen=lis.frozen, frag_prescreen=metadata, verbose_imp=verbose_imp,
        ccsd_t=ccsd_t, dcsd=False, profile_info=None)


def _run_group(state, static, group, baseline_controls, matrix_controls, *,
               with_grad, verbose_imp, ccsd_t, include_baseline=True, fragment_indices=None):
    """Complete one owned group's preparation, baseline and singleton responses.

    All process-local VJPs and matrix replay records are consumed here. Only
    scalar component records and the accumulated numerical host bars survive.
    """
    inputs, host, prepare_pullback = driver._prepare(
        state, static.mp2_static, group, baseline_controls, with_grad)
    metadata = {'fragment_id': group.representative_index,
                'target_indices': group.target_indices}
    if include_baseline:
        baseline = backend.solve(host, metadata, baseline_controls, with_grad=with_grad)
        state['energy'] = state['energy'] + baseline['energy']
        state['baseline_energy'] = state.get('baseline_energy', 0.) + float(baseline['energy'])
        state['variance'] += float(baseline['energy_standard_error']) ** 2
        if with_grad:
            driver._add_bars(state, *apply_selected_pullback(prepare_pullback, inputs, baseline))
        del baseline
    records = []
    for index, target in zip(group.fragment_indices, group.target_indices):
        if fragment_indices is not None and index not in fragment_indices:
            continue
        forward = backend.solve_matrices(
            host, {'fragment_id': index, 'target_indices': (target,)},
            matrix_controls, matrix_kind=static.matrix_kind)
        matrices = {name: np.asarray(forward[name]) for name in ('occupied', 'virtual')}
        def correction(mf, common, blocks):
            values = _solve_fragment(mf, common, static, index, blocks, group,
                                     verbose_imp=verbose_imp, ccsd_t=ccsd_t)
            emp2, ecc, et = values
            return ecc + et - emp2, values
        if with_grad:
            energy, correction_pullback, values = jax.vjp(
                correction, state['mf'], state['common'], matrices, has_aux=True)
            mf_bar, common_bar, matrix_bars = correction_pullback(np.ones_like(energy))
            direct_norm = _bar_norm(mf_bar, common_bar)
            driver._add_bars(state, mf_bar, common_bar)
            bars = backend.pullback_matrices(
                host, forward, energy_bar=0., occupied_bar=matrix_bars['occupied'],
                virtual_bar=matrix_bars['virtual'])
            native_bars = apply_input_pullback(prepare_pullback, inputs, bars)
            native_norm = _bar_norm(*native_bars)
            driver._add_bars(state, *native_bars)
            del native_bars
            del correction_pullback, mf_bar, common_bar, matrix_bars, bars
        else:
            direct_norm = native_norm = None
            energy, values = correction(state['mf'], state['common'], matrices)
        state['energy'] = state['energy'] + energy
        jax.block_until_ready((state['energy'], state['mf_bar'], state['common_bar']))
        records.append(FragmentComponents(index, *(float(value) for value in values),
                                           float(forward['energy']),
                                           *_fragment_dimensions(static, index),
                                           direct_norm, native_norm,
                                           {key: value for key, value in forward['diagnostics'].items()
                                            if key not in ('occupied_standard_error',
                                                           'virtual_standard_error')}))
        del forward, matrices, correction, energy, values
        gc.collect()
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
                fragment_indices=None, include_baseline=True,
                checkpoint_dir=None, resume=False):
    from ._ccsd_checkpoint import open_checkpoint, restore_progress, save_progress
    groups = build_domain_groups(static.mp2_static, merge_identical=merge_identical)
    allocated = allocate_domain_controls(baseline, len(groups))
    options = dict(with_grad=with_grad, include_hf=include_hf, ccsd_t=ccsd_t,
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
            if include_baseline or set(group.fragment_indices).intersection(fragment_indices):
                records.extend(_run_group(
                    state, static, group, allocated, matrix, with_grad=with_grad,
                    verbose_imp=verbose_imp, ccsd_t=ccsd_t,
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


def _run(initializer, worker_initializer, comm, *, checkpoint_dir, resume, **options):
    if comm is not None and comm.Get_size() > 1:
        from ._mpi_ccsd import _run_mpi
        def check_options():
            if checkpoint_dir is not None or resume:
                raise NotImplementedError('CC progress checkpoints currently require serial execution')
        driver._root_call(comm, comm.Get_rank(), check_options, 'CC checkpoint option validation')
        return _run_mpi(initializer, worker_initializer, comm, **options)
    state, static, baseline, matrix = initializer()
    options['fragment_indices'] = state.pop('cc_fragment_indices')
    return _run_serial(state, static, baseline, matrix,
                       checkpoint_dir=checkpoint_dir, resume=resume, **options)


def kernel(mf, static, *, baseline_controls, matrix_controls, matrix_kind=None,
           ccsd_t=False, verbose_imp=0, include_hf=False, return_components=False,
           merge_identical=True, comm=None, fragment_indices=None, include_baseline=True,
           checkpoint_dir=None, resume=False):
    """Return correlation energy, or total energy with ``include_hf=True``.

    ``return_components=True`` returns ``(energy, CCSDComponents)``. The MP2
    baseline includes weak completion and is independent of matrix precision.
    ``fragment_indices`` restricts corrections (default: all saved LIS ranks).
    The complete baseline remains included unless ``include_baseline=False``;
    that option returns only selected CC-minus-MP2-LIS corrections and forbids HF.
    """
    def initialize():
        if include_hf and not include_baseline:
            raise ValueError('include_hf requires include_baseline=True')
        baseline, matrix = _validate_run(static, baseline_controls, matrix_controls, matrix_kind)
        indices = _fragment_indices(static, fragment_indices)
        if any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(mf)):
            raise TypeError('STC CC kernel is forward-only; use value_and_grad')
        state = driver._initialize(mf, static.mp2_static, baseline, with_grad=False)
        state['cc_fragment_indices'] = indices
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
                checkpoint_dir=checkpoint_dir, resume=resume)


def value_and_grad(mol, build_mf, static=None, *, baseline_controls, matrix_controls,
                   matrix_kind=None, ccsd_t=False, verbose_imp=0, include_hf=False,
                   return_components=False, merge_identical=True, comm=None, build_static=None,
                   fragment_indices=None, include_baseline=True, checkpoint_dir=None, resume=False):
    """Return ``(energy, mol_bar)``; ``mol_bar.coords`` is a nuclear gradient.

    Include HF once with ``include_hf=True``. Append ``CCSDComponents`` with
    ``return_components=True``. A force is the negated coordinate gradient.
    ``build_static(mf)`` may select ranks once on the detached SCF reference.
    Subset and ``include_baseline`` semantics match :func:`kernel`; even a
    correction-only subset closes common-data and implicit SCF response once.
    """
    def initialize():
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
        selected = stop_trace(build_static)(mf) if build_static is not None else static
        baseline, matrix = _validate_run(selected, baseline_controls, matrix_controls, matrix_kind)
        indices = _fragment_indices(selected, fragment_indices)
        state = driver._initialize(mf, selected.mp2_static, baseline, with_grad=True,
                                   scf_pullback=scf_pullback)
        state['cc_fragment_indices'] = indices
        return state, selected, baseline, matrix

    def initialize_worker(canonical):
        driver._coordinate_only(mol)
        return build_mf(mol, **{name + '_init': value for name, value in canonical.items()})

    return _run(initialize, initialize_worker, comm, with_grad=True, include_hf=include_hf,
                merge_identical=merge_identical, verbose_imp=verbose_imp,
                ccsd_t=ccsd_t, return_components=return_components,
                fragment_indices=fragment_indices, include_baseline=include_baseline,
                checkpoint_dir=checkpoint_dir, resume=resume)
