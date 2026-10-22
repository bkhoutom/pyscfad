"""Domain and whole-system STC-MP2 energies and molecular gradients.

    Discrete domains, ranks, partner sets and Laplace grid are fixed. Nuclear
    coordinate gradients require coordinate-only molecule leaves and a
    positive-definite local auxiliary fitting metric.
"""

from functools import partial
from contextlib import nullcontext
import gc
import math
import warnings

import jax
import numpy
from pyscf import df as pyscf_df, lib
from pyscfad import config_update, numpy as np
from pyscfad.ops import stop_trace
from pyscfad.lib._compilation_cache import configure_compilation_cache
from pyscfad.tools import resource_profile
from ._selection import DomainSelections, _extract_active_indices, build_domain_groups
from .domain import rebuild_domain_data
from ._selection import _validate_boys_target_groups as validate_boys_target_groups
from .adjoint import (
    add_cotangent as _add_cotangent,
    fix_restart_mo_phases as _fix_restart_mo_phases,
    zero_term_cotangents as _zero_term_cotangents,
    zero_mf_cotangent as _zero_mf_cotangent,
)
from ._pairs import weak_pair_energy_and_grad, weak_pair_energy_sum, weak_pair_indices


from . import backend
from ._memory import memory_calculation
from ._workflow import check_system_memory, check_domain_memory, allocate_domain_controls
from .adjoint import apply_full_pullback, apply_selected_pullback
from .controls import DEFAULT_ENERGY_TOLERANCE, default_laplace_grid
from .domain import (
    select_system_boys_reference, select_system_virtual_anchor_columns,
)
from .parallel import run_backend
from .prepare import prepare_system_inputs, prepare_boys_group_inputs
from .protocol import ARRAY_NAMES


def _root_call(comm, rank, function, phase):
    """Complete a root-only phase and announce failure before continuing."""
    if comm is None:
        return function()
    result = None
    if rank == 0:
        try:
            result = function()
            error = None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    else:
        error = None
    error = comm.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(f"STC {phase} failed on root: {error}")
    return result


def _validate_static(static):
    if not isinstance(static, DomainSelections):
        raise TypeError("static must be DomainSelections")
    if static.frozen is None:
        raise ValueError("STC requires an explicit frozen-space setting")
    nocc = len(static.active_occ_indices)
    if (static.lo_type != "boys" or len(static.fragments) != nocc
            or static.frag_lolist is None):
        raise ValueError("Boys targets must form a complete singleton partition")
    groups = validate_boys_target_groups(static.frag_lolist, nocc)
    mask = numpy.asarray(static.strong_mask, dtype=bool)
    if (mask.shape != (nocc, nocc) or not numpy.array_equal(mask, mask.T)
            or not numpy.diag(mask).all()):
        raise ValueError("strong_mask must be symmetric with every target self-strong")
    for index, fragment in enumerate(static.fragments):
        if not numpy.array_equal(fragment.iao_indices, groups[index]):
            raise ValueError("fragment target does not match the singleton partition")
        partners = numpy.asarray(fragment.strong_fragments)
        if (partners.ndim != 1 or partners.dtype.kind not in "iu"
                or not numpy.array_equal(numpy.sort(partners), numpy.flatnonzero(mask[index]))):
            raise ValueError("strong partner selections disagree with strong_mask")


def _driver_controls(controls, *, scope=None):
    """Copy root-owned options and apply high-level defaults before preparation."""
    if not isinstance(controls, dict):
        raise TypeError("controls must be a dictionary")
    controls = controls.copy()
    if "laplace_roots" not in controls and "laplace_weights" not in controls:
        controls["laplace_roots"], controls["laplace_weights"] = default_laplace_grid()
    if controls.get("mode", "deterministic") == "stochastic" and "production_samples" not in controls:
        controls.setdefault("energy_tolerance", DEFAULT_ENERGY_TOLERANCE)
    _validate_controls(controls)
    if controls.get("canonical_fock", False):
        raise ValueError(
            "high-level STC uses full local Fock blocks; canonical_fock=True "
            "is reserved for low-level canonical reference calculations"
        )
    return controls


def _validate_controls(controls):
    if not isinstance(controls, dict):
        raise TypeError("controls must be a dictionary")
    if "canonical_fock" in controls and not isinstance(controls["canonical_fock"], bool):
        raise TypeError("canonical_fock must be a bool")
    for name in ("laplace_roots", "laplace_weights"):
        if name not in controls:
            raise ValueError(f"controls must supply {name}")
        values = numpy.asarray(controls[name], dtype=numpy.float64)
        if values.ndim != 1 or not values.size or not numpy.isfinite(values).all():
            raise ValueError(f"{name} must be a nonempty finite vector")
    roots = numpy.asarray(controls["laplace_roots"])
    weights = numpy.asarray(controls["laplace_weights"])
    if roots.shape != weights.shape or numpy.any(roots < 0):
        raise ValueError("Laplace roots/weights must match and roots must be nonnegative")
    if controls.get("mode", "deterministic") not in ("deterministic", "stochastic"):
        raise ValueError("mode must be deterministic or stochastic")
    if "energy_tolerance" in controls:
        try:
            tolerance = float(controls["energy_tolerance"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("energy_tolerance must be positive and finite") from exc
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("energy_tolerance must be positive and finite")
    block = controls.get("virtual_block_size", 16)
    if (not isinstance(block, (int, numpy.integer))
            or isinstance(block, (bool, numpy.bool_)) or block <= 0):
        raise ValueError("virtual_block_size must be a positive integer")


def _validate_reference(mf, static):
    _system_active_indices(mf, static.frozen)
    for name, expected in zip(
        ("active_occ_indices", "active_vir_indices", "pao_projected_out_indices"),
        _extract_active_indices(mf, static.frozen),
    ):
        if not numpy.array_equal(getattr(static, name), expected):
            raise ValueError(f"saved {name} does not match this SCF/frozen setting")
    # The portable path must never silently fall back to cached AO-pair CDERI.
    try:
        from pyscf.mp.dfmp2 import _init_mp_df_eris_direct
    except ImportError as exc:
        raise RuntimeError("STC requires PySCF's integral-direct local Lov helper") from exc
    if not callable(_init_mp_df_eris_direct):
        raise RuntimeError("STC requires PySCF's integral-direct local Lov helper")


def _coordinate_only(mol):
    if any(getattr(mol, name, None) is not None for name in ("exp", "ctr_coeff", "r0")):
        raise NotImplementedError(
            "STC molecular gradients differentiate nuclear coordinates only; "
            "build mol with trace_exp=False and trace_ctr_coeff=False, and no r0 leaf"
        )


def _initialize(mf, static, controls, *, with_grad, scf_pullback=None):
    _validate_static(static)
    _validate_controls(controls)
    _validate_reference(mf, static)
    configure_compilation_cache()
    with resource_profile.section('stc.domain.common_prepare'):
        if with_grad:
            common, common_pullback = jax.vjp(lambda mf_: rebuild_domain_data(mf_, static), mf)
            mf_bar, common_bar = _zero_term_cotangents(mf, common)
        else:
            common = rebuild_domain_data(mf, static)
            common_pullback = mf_bar = common_bar = None
        jax.block_until_ready(common)
    return {"mf": mf, "static": static, "common": common, "common_pullback": common_pullback,
            "scf_pullback": scf_pullback, "mf_bar": mf_bar, "common_bar": common_bar,
            "energy": np.zeros((), dtype=common.s1e.dtype), "variance": 0.0,
            "weak_pairs": tuple(weak_pair_indices(static))}


def _prepare(state, static, group, controls, with_grad):
    memory_controls = ({"mode": "deterministic",
                        "target_batch_size": controls.get("target_batch_size", 1),
                        "threads": controls.get("threads")}
                       if controls.get("mode", "deterministic") == "deterministic" else {})
    check_domain_memory(state["mf"], static, group.representative_index,
                        with_grad=with_grad,
                        virtual_block_size=controls.get("virtual_block_size", 16),
                        **memory_controls)
    prepare = lambda mf_, common_: prepare_boys_group_inputs(mf_, common_, static, group)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="integral-direct local Lov is unavailable.*",
                                category=RuntimeWarning)
        if with_grad:
            inputs, pullback = jax.vjp(prepare, state["mf"], state["common"])
        else:
            inputs, pullback = prepare(state["mf"], state["common"]), None
    jax.block_until_ready(inputs)
    return inputs, backend.host_inputs(inputs), pullback


def _add_bars(state, mf_bar, common_bar):
    state["mf_bar"] = jax.tree_util.tree_map(_add_cotangent, state["mf_bar"], mf_bar)
    state["common_bar"] = jax.tree_util.tree_map(_add_cotangent,
                                              state["common_bar"], common_bar)


def _finish(state, *, with_grad, include_hf,
            uncertainty_label="stochastic standard error"):
    energy = state["energy"]
    lib.logger.info(state["mf"], "STC correlation energy %.12g; %s %.6g",
                    float(energy), uncertainty_label, math.sqrt(state["variance"]))
    if not with_grad:
        jax.block_until_ready(energy)
        return energy
    mf_bar = state["mf_bar"]
    if state.get("common_pullback") is not None:
        with resource_profile.section('stc.domain.common_pullback'):
            common_mf_bar, = state["common_pullback"](state["common_bar"])
            mf_bar = jax.tree_util.tree_map(_add_cotangent, mf_bar, common_mf_bar)
            jax.block_until_ready(mf_bar)
    return _finish_response(state, energy, mf_bar, include_hf=include_hf)


def _finish_response(state, energy, mf_bar, *, include_hf):
    """Add HF once, then apply the single implicit SCF response."""
    if include_hf:
        hf_energy = state["mf"].e_tot
        mf_bar.e_tot = _add_cotangent(mf_bar.e_tot, np.ones_like(hf_energy))
        energy = energy + hf_energy
    scope = 'system' if 'fock_from_scf' in state else 'domain'
    with resource_profile.section(f'stc.{scope}.scf_response'):
        mol_bar, = state["scf_pullback"](mf_bar)
        jax.block_until_ready((energy, mol_bar))
    return energy, mol_bar


def _run_domain(initializer, static, controls, comm, *, with_grad, include_hf=False,
                merge_identical=True):
    rank = 0 if comm is None else comm.Get_rank()
    with resource_profile.section('stc.domain.initialize'):
        state = _root_call(comm, rank, initializer, "initial preparation")
    if rank == 0 and static is None:
        static = state["static"]
    groups = _root_call(comm, rank,
                        lambda: build_domain_groups(static, merge_identical=merge_identical),
                        "group selection")
    pairs = state["weak_pairs"] if rank == 0 else None
    if comm is not None:
        groups, pairs = comm.bcast((groups, pairs), root=0)
    settings = allocate_domain_controls(controls, len(groups)) if rank == 0 else None
    for group in groups:
        fragment_id = group.representative_index
        with resource_profile.section('stc.domain.prepare'):
            prepared = _root_call(comm, rank,
                lambda: _prepare(state, static, group, controls, with_grad),
                f"group {fragment_id} preparation")
        metadata = ({"fragment_id": fragment_id, "target_indices": group.target_indices,
                     "fragment_indices": group.fragment_indices,
                     "global_target_orbitals": tuple(int(static.frag_lolist[i][0])
                                                     for i in group.fragment_indices),
                     "budget": {"total_tolerance": controls.get("energy_tolerance"),
                                "effective_tolerance": settings.get("energy_tolerance"),
                                "original_target_count": len(static.fragments),
                                "group_count": len(groups)}} if rank == 0 else None)
        with resource_profile.section('stc.domain.native'):
            result = run_backend(prepared[1] if rank == 0 else None, metadata, settings,
                                 partial(backend.solve, with_grad=with_grad), comm=comm)

        def apply_group():
            if with_grad:
                _add_bars(state, *apply_selected_pullback(prepared[2], prepared[0], result))
            state["energy"] = state["energy"] + result["energy"]
            state["variance"] += float(result["energy_standard_error"]) ** 2
            jax.block_until_ready((state["energy"], state["mf_bar"], state["common_bar"]))

        with resource_profile.section('stc.domain.input_pullback'):
            _root_call(comm, rank, apply_group, f"group {fragment_id} pullback")
        del prepared, result, apply_group
        gc.collect()
    if pairs:
        def apply_weak():
            if with_grad:
                energy, bars = weak_pair_energy_and_grad(
                    state["mf"], state["common"], static, pairs=pairs)
                _add_bars(state, *bars)
            else:
                energy = weak_pair_energy_sum(
                    state["mf"], state["common"], static, pairs=pairs)
            state["energy"] = state["energy"] + energy
            jax.block_until_ready((state["energy"], state["mf_bar"], state["common_bar"]))
        with resource_profile.section('stc.domain.weak'):
            _root_call(comm, rank, apply_weak, "weak endpoint response")
        del apply_weak
        if not with_grad:
            # The gradient helper already collects after releasing its tapes.
            gc.collect()
    with resource_profile.section('stc.domain.response'):
        return _root_call(comm, rank, lambda: _finish(state, with_grad=with_grad,
                         include_hf=include_hf), "final response")


def _select_scope(static, scope, frozen, lo_kwargs=None):
    """Validate root-owned call options before any scientific scope branch."""
    if scope not in ("domain", "system"):
        raise ValueError("scope must be domain or system")
    if scope == "domain" and frozen is not None:
        raise ValueError("domain frozen setting must come from static; frozen must be None")
    if scope == "domain" and lo_kwargs is not None:
        raise ValueError("domain localization options must come from static.lo_kwargs")
    if scope == "system" and static is not None:
        raise ValueError("system scope requires static=None")
    return scope


def _shared_scope(static, scope, frozen, comm, lo_kwargs=None):
    rank = 0 if comm is None else comm.Get_rank()
    selected = _root_call(comm, rank, lambda: _select_scope(static, scope, frozen, lo_kwargs),
                          "scope selection")
    return selected if comm is None else comm.bcast(selected, root=0)


def _system_active_indices(mf, frozen):
    """Validate restricted DF and frozen spaces before inherited selection."""
    if getattr(mf, "with_df", None) is None:
        raise ValueError("system STC requires a density-fitted SCF object")
    if not getattr(mf, "converged", False):
        raise ValueError("system STC requires a converged SCF reference")
    coeff, occ = numpy.asarray(mf.mo_coeff), numpy.asarray(mf.mo_occ)
    if numpy.iscomplexobj(coeff) or numpy.iscomplexobj(occ):
        raise NotImplementedError("system STC supports real orbitals only")
    if (coeff.ndim != 2 or occ.ndim != 1 or occ.size != coeff.shape[1]
            or not numpy.all((numpy.abs(occ) < 1e-12) | (numpy.abs(occ - 2) < 1e-12))):
        raise NotImplementedError("system STC supports restricted closed-shell references only")
    if not numpy.isfinite(coeff).all() or not numpy.isfinite(occ).all():
        raise ValueError("system SCF orbitals and occupations must be finite")
    noccupied, nmo = int(numpy.count_nonzero(occ > 1)), occ.size
    if frozen is None:
        frozen = 0
    if isinstance(frozen, (int, numpy.integer)) and not isinstance(frozen, (bool, numpy.bool_)):
        if not 0 <= frozen <= noccupied:
            raise ValueError("frozen integer must lie between zero and occupied count")
        frozen = int(frozen)
    elif isinstance(frozen, (list, tuple, numpy.ndarray)):
        values = list(frozen)
        if any(not isinstance(i, (int, numpy.integer)) or isinstance(i, (bool, numpy.bool_))
               for i in values):
            raise ValueError("frozen list must contain integer MO indices")
        if any(i < 0 or i >= nmo for i in values) or len(set(values)) != len(values):
            raise ValueError("frozen MO indices must be unique and within the MO range")
        frozen = [int(i) for i in values]
    else:
        raise TypeError("frozen must be None, an integer, or an integer MO index list")
    occupied, virtual, _ = _extract_active_indices(mf, frozen)
    return occupied, virtual


def _initialize_scf(mol, build_mf, *, fix_phases=True):
    """Build one implicit SCF VJP shared by both correlation scopes."""
    _coordinate_only(mol)
    if not callable(build_mf):
        raise TypeError("build_mf must be callable")
    with (config_update("pyscfad_scf_implicit_diff", True),
          config_update("pyscfad_scf_first_order_custom", False)):
        build = (lambda mol_: _fix_restart_mo_phases(build_mf(mol_))) if fix_phases else build_mf
        return jax.vjp(build, mol)


def _initialize_system(mf, frozen, controls, *, with_grad, scf_pullback=None,
                       lo_kwargs=None, fock_from_scf=False):
    _validate_controls(controls)
    occupied, virtual = _system_active_indices(mf, frozen)
    requested_fock_from_scf = fock_from_scf
    if fock_from_scf and len(occupied) and len(virtual):
        energies = numpy.asarray(mf.mo_energy)
        if (energies.shape != (mf.mo_coeff.shape[1],) or numpy.iscomplexobj(energies)
                or not numpy.isfinite(energies).all()):
            raise ValueError("SCF Fock projection requires real finite orbital energies "
                             "matching the canonical coefficient columns")
        # SCF's generalized-eigh response masks rotations at gaps <= 1e-9.
        # Reconstructing F from those outputs would lose off-diagonal Fock
        # response inside a degenerate active block. Select the general path
        # outside the AD tape, using the same cutoff as the SCF eigensolver.
        if any(numpy.any(numpy.diff(numpy.sort(energies[indices])) <= 1e-9)
               for indices in (occupied, virtual)):
            fock_from_scf = False
    resource_profile.checkpoint('stc.system.fock_state',
        requested_fock_from_scf=requested_fock_from_scf, fock_from_scf=fock_from_scf,
        degeneracy_fallback=requested_fock_from_scf and not fock_from_scf)
    dtype = mf.mo_coeff.dtype
    return {"mf": mf, "occupied": occupied, "virtual": virtual,
            "lo_kwargs": lo_kwargs, "fock_from_scf": fock_from_scf,
            "scf_pullback": scf_pullback, "energy": np.zeros((), dtype=dtype),
            "variance": 0.0,
            "mf_bar": (_zero_mf_cotangent(mf, dtype)
                       if with_grad and not (len(occupied) and len(virtual)) else None)}


def _validate_canonical_system(mf):
    """Validate the converged canonical eigenproblem outside the AD tape."""
    coeff = numpy.asarray(mf.mo_coeff)
    energies = numpy.asarray(mf.mo_energy)
    if energies.shape != (coeff.shape[1],) or not numpy.isfinite(energies).all():
        raise ValueError("canonical_fock requires finite canonical orbital energies")
    fock = numpy.asarray(mf.get_fock())
    residual = float(numpy.max(numpy.abs(coeff.T @ fock @ coeff - numpy.diag(energies))))
    grad_tolerance = mf.conv_tol_grad
    if grad_tolerance is None:
        energy_tolerance = float(mf.conv_tol)
        if not math.isfinite(energy_tolerance) or energy_tolerance <= 0:
            raise ValueError("canonical_fock requires a finite positive SCF energy tolerance")
        grad_tolerance = math.sqrt(energy_tolerance)
    grad_tolerance = float(grad_tolerance)
    if not math.isfinite(grad_tolerance) or grad_tolerance <= 0:
        raise ValueError("canonical_fock requires a finite positive SCF gradient tolerance")
    residual_limit = max(1e-7, grad_tolerance)
    if not numpy.isfinite(residual) or residual > residual_limit:
        raise ValueError(
            "canonical_fock requires canonical SCF orbitals and matching energies; "
            f"max |C.T F C - diag(eps)| = {residual:.3g} Eh "
            f"exceeds {residual_limit:.3g} Eh")


def _validate_system_metric(mf):
    """Require the fitting metric used by system preparation to admit Cholesky."""
    get_cderi = getattr(mf.with_df, "_get_cderi_source", None)
    cderi = get_cderi() if get_cderi is not None else mf.with_df._cderi
    auxmol = getattr(mf.with_df, "auxmol", None) if cderi is not None else None
    if auxmol is None:
        auxmol = pyscf_df.addons.make_auxmol(mf.mol.to_pyscf(), mf.with_df.auxbasis)
    elif hasattr(auxmol, "to_pyscf"):
        auxmol = auxmol.to_pyscf()
    metric = auxmol.intor("int2c2e", hermi=1)
    try:
        if not numpy.isfinite(metric).all():
            raise numpy.linalg.LinAlgError("nonfinite metric")
        numpy.linalg.cholesky(metric)
    except numpy.linalg.LinAlgError as exc:
        raise NotImplementedError(
            "system STC requires a positive-definite auxiliary fitting metric"
        ) from exc


def _prepare_system(state, controls, with_grad):
    memory_controls = ({"mode": "deterministic",
                        "target_batch_size": controls.get("target_batch_size", 1),
                        "threads": controls.get("threads")}
                       if controls.get("mode", "deterministic") == "deterministic" else {})
    check_system_memory(state["mf"], len(state["occupied"]), len(state["virtual"]),
                        with_grad=with_grad,
                        virtual_block_size=controls.get("virtual_block_size", 16),
                        auxiliary_group_size=controls.get("auxiliary_group_size", 32),
                        **memory_controls)
    _validate_system_metric(state["mf"])
    # Save discrete labels before building the VJP; rebuild all continuous
    # coefficients inside it. System scope needs no domain selections.
    state["boys_reference"] = select_system_boys_reference(
        state["mf"], state["occupied"], lo_kwargs=state["lo_kwargs"],
    )
    state["virtual_anchor_columns"] = select_system_virtual_anchor_columns(
        state["mf"], state["virtual"],
    )
    prepare = lambda mf_: prepare_system_inputs(
        mf_, state["occupied"], state["virtual"],
        boys_reference=state["boys_reference"],
        virtual_anchor_columns=state["virtual_anchor_columns"],
        fock_from_scf=state["fock_from_scf"],
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="integral-direct local Lov is unavailable.*",
                                category=RuntimeWarning)
        if with_grad:
            inputs, pullback = jax.vjp(prepare, state["mf"])
        else:
            inputs, pullback = prepare(state["mf"]), None
    return inputs, backend.host_inputs(inputs, scope="system"), pullback


def _run_system(initializer, controls, comm, *, with_grad, include_hf=False):
    rank = 0 if comm is None else comm.Get_rank()
    with resource_profile.section('stc.system.scf'):
        state = _root_call(comm, rank, initializer, "initial system preparation")
    has_work = bool(len(state["occupied"]) and len(state["virtual"])) if rank == 0 else None
    if comm is not None:
        has_work = comm.bcast(has_work, root=0)
    if has_work:
        with resource_profile.section('stc.system.prepare'):
            prepared = _root_call(comm, rank, lambda: _prepare_system(state, controls, with_grad),
                                  "system preparation")
        with resource_profile.section('stc.system.native'):
            result = run_backend(
                prepared[1] if rank == 0 else None, {} if rank == 0 else None,
                controls if rank == 0 else None,
                partial(backend.solve, with_grad=with_grad, scope="system"),
                comm=comm, array_names=ARRAY_NAMES,
            )

        def apply_system():
            if with_grad:
                # Validate the existing host mirror without compiling many
                # bounded JAX slices solely for repeated finite-value checks.
                state["mf_bar"], = apply_full_pullback(prepared[2], prepared[1], result)
            state["energy"] = state["energy"] + result["energy"]
            state["variance"] = float(result["energy_standard_error"]) ** 2
            jax.block_until_ready((state["energy"], state["mf_bar"]))

        with resource_profile.section('stc.system.input_pullback'):
            _root_call(comm, rank, apply_system, "system pullback")
        del prepared, result, apply_system
        gc.collect()
    with resource_profile.section('stc.system.response'):
        return _root_call(comm, rank, lambda: _finish(state, with_grad=with_grad,
                                                    include_hf=include_hf), "final response")


@memory_calculation
def kernel(mf, static=None, *, scope="domain", frozen=None, controls, comm=None,
           lo_kwargs=None, merge_identical=True, parallel_scf_jk=True):
    """Return correlation energy on root.

    Domain jobs share identical saved frames by default. ``merge_identical=False``
    runs each singleton separately for diagnostic comparisons.
    With a multi-rank domain communicator, each rank supplies a matching MF
    object; workers may use an unconverged skeleton. Root's canonical state
    is installed on workers without SCF. Only root needs selections/controls.
    ``parallel_scf_jk=True`` distributes common-preparation DF J/K work for
    multi-rank domain calls. SCF preceding this energy-only API is caller-owned.
    System, serial and single-rank calls retain their existing behavior.
    """
    scope = _shared_scope(static, scope, frozen, comm, lo_kwargs)
    rank = 0 if comm is None else comm.Get_rank()
    controls = _root_call(comm, rank, lambda: _driver_controls(controls, scope=scope),
                          "control validation")

    def initialize():
        if any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(mf)):
            raise TypeError("STC kernel is forward-only; use value_and_grad")
        if scope == "system":
            return _initialize_system(mf, frozen, controls, with_grad=False,
                                      lo_kwargs=lo_kwargs)
        return _initialize(mf, static, controls, with_grad=False)

    if scope == "system":
        return _run_system(initialize, controls, comm, with_grad=False)
    if comm is not None and comm.Get_size() > 1:
        from ._mpi import _run_domain_mpi

        def initialize_worker(canonical):
            if mf is None:
                raise ValueError("domain MPI kernel requires an MF skeleton on every rank")
            for name, value in canonical.items():
                setattr(mf, name, np.asarray(value))
            mf.converged = True
            return mf

        return _run_domain_mpi(initialize, initialize_worker, static, controls, comm,
                               with_grad=False, merge_identical=merge_identical,
                               parallel_scf_jk=parallel_scf_jk,
                               scf_worker_initializer=lambda: mf)
    return _run_domain(initialize, static, controls, comm, with_grad=False,
                       merge_identical=merge_identical)


@memory_calculation
def value_and_grad(mol, build_mf, static=None, *, scope="domain", frozen=None,
                   controls, comm=None, include_hf=False, lo_kwargs=None, merge_identical=True,
                   build_static=None, parallel_scf_jk=True):
    """Return ``(energy, mol_bar)`` on root; worker ranks return ``None``.

    Discrete selections and the Laplace grid are fixed. System scope fixes
    Boys reference labels and complete PAO virtual anchors, while coordinate
    response includes localization, projections, Cholesky orthonormalization,
    full Fock blocks, fitted factors and SCF.
    For system scope, ``build_mf`` must return converged canonical orbitals
    and matching orbital energies without subsequent orbital rotations.
    Local Fock blocks use these differentiable energies and rotations to
    avoid a separate AO Fock construction and its preparation pullback.
    Degenerate active occupied or virtual spectra use AO Fock preparation
    because the SCF eigensolver masks their internal rotation response.
    System ``lo_kwargs`` are passed to the shared Boys localization; domain
    calls take their saved localization options from ``static``.
    Domain calls may instead supply ``build_static(mf)``. It runs once on
    a detached, phase-fixed reference after the single differentiated SCF,
    and returns fixed ``DomainSelections``. Selection-only mutations cannot
    change the live SCF tree retained by its pullback. ``static`` and
    ``build_static`` are mutually exclusive; system scope uses neither.
    Identical domain frames share one selected solve/VJP unless the diagnostic
    option ``merge_identical=False`` is supplied. Public targets stay singleton.
    Stochastic energy uncertainty does not bound gradient uncertainty.
    Multi-rank domain calls distribute whole groups and weak endpoints.
    Worker ``build_mf`` must accept ``mo_coeff_init``, ``mo_energy_init``,
    ``mo_occ_init`` and ``e_tot_init``, adopt their supplied values, and skip
    SCF. Workers may pass ``None`` for selections and controls; ``build_static``
    runs only on root. Results are returned only on communicator rank zero.
    Multi-rank domain calls also distribute DF J/K contractions in SCF,
    common-data preparation and their coordinate pullbacks by default.
    Workers first build an unrun reference with placeholder canonical arrays
    to prepare compatible global DF factors, then adopt root's converged state.
    An attached out-of-core CDERI source must be available on every rank.
    ``parallel_scf_jk=False`` retains root-only SCF and common response.
    System, serial and single-rank calls never activate this executor.
    """
    scope = _shared_scope(static, scope, frozen, comm, lo_kwargs)
    rank = 0 if comm is None else comm.Get_rank()
    controls = _root_call(comm, rank, lambda: _driver_controls(controls, scope=scope),
                          "control validation")

    def initialize():
        _coordinate_only(mol)
        if build_static is not None:
            if scope != "domain":
                raise ValueError("build_static is only supported for domain scope")
            if static is not None:
                raise ValueError("supply either static or build_static, not both")
            if not callable(build_static):
                raise TypeError("build_static must be callable")
        elif scope == "domain":
            _validate_static(static)
        with (resource_profile.section('stc.domain.scf') if scope == 'domain' else nullcontext()):
            mf, scf_pullback = _initialize_scf(mol, build_mf)
        if scope == "system":
            return _initialize_system(mf, frozen, controls, with_grad=True,
                                      scf_pullback=scf_pullback, lo_kwargs=lo_kwargs,
                                      fock_from_scf=True)
        selected = static
        if build_static is not None:
            with resource_profile.section('stc.domain.selection'):
                selected = stop_trace(build_static)(mf)
        return _initialize(mf, selected, controls, with_grad=True,
                           scf_pullback=scf_pullback)

    if scope == "system":
        return _run_system(initialize, controls, comm, with_grad=True, include_hf=include_hf)
    if comm is not None and comm.Get_size() > 1:
        from ._mpi import _run_domain_mpi

        def initialize_worker(canonical):
            _coordinate_only(mol)
            try:
                return build_mf(mol, **{name + "_init": value
                                        for name, value in canonical.items()})
            except TypeError as exc:
                raise TypeError(
                    "domain MPI worker build_mf must accept mo_coeff_init, "
                    "mo_energy_init, mo_occ_init, and e_tot_init without running SCF"
                ) from exc

        def initialize_scf_worker():
            nao = mol.nao
            occupation = numpy.zeros(nao)
            occupation[:mol.nelectron // 2] = 2.0
            return initialize_worker({
                "mo_coeff": numpy.eye(nao), "mo_energy": numpy.zeros(nao),
                "mo_occ": occupation, "e_tot": numpy.asarray(0.0),
            })

        return _run_domain_mpi(initialize, initialize_worker, static, controls, comm,
                               with_grad=True, include_hf=include_hf,
                               merge_identical=merge_identical,
                               parallel_scf_jk=parallel_scf_jk,
                               scf_worker_initializer=initialize_scf_worker)
    return _run_domain(initialize, static, controls, comm, with_grad=True, include_hf=include_hf,
                       merge_identical=merge_identical)
