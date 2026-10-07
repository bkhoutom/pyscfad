"""Domain and whole-system STC-MP2 energies and molecular gradients.

    Discrete domains, ranks, partner sets and Laplace grid are fixed. Nuclear
    coordinate gradients require coordinate-only molecule leaves and a
    positive-definite local auxiliary fitting metric.
"""

from functools import partial
import gc
import math
import warnings

import jax
import numpy
from pyscf import df as pyscf_df, lib
from pyscfad import config_update, numpy as np
from pyscfad.tools import resource_profile
from pyscfad.dlno._selection import DomainSelections, _extract_active_indices
from pyscfad.dlno.domain import _check_domain_inputs
from pyscfad.dlno.dlno_base import rebuild_domain_data
from pyscfad.dlno.targets import validate_boys_target_groups
from pyscfad.dlno.mp2 import (
    _add_cotangent, _contains_tracer, _correlation_term_energy,
    _correlation_term_specs, _fix_restart_mo_phases, _zero_term_cotangents,
    _zero_mf_cotangent,
)

from . import backend
from ._workflow import check_system_memory, check_weighted_memory
from .adjoint import apply_full_pullback, apply_weighted_pullback
from .controls import DEFAULT_ENERGY_TOLERANCE, default_laplace_grid
from .domain import select_virtual_anchor_columns
from .parallel import run_backend
from .prepare import prepare_system_inputs, prepare_weighted_inputs
from .protocol import ARRAY_NAMES, WEIGHTED_ARRAY_NAMES


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
        if not numpy.array_equal(fragment.strong_fragments, numpy.flatnonzero(mask[index])):
            raise ValueError("strong partner selections disagree with strong_mask")


def _driver_controls(controls):
    """Copy root-owned options and apply high-level defaults before preparation."""
    if not isinstance(controls, dict):
        raise TypeError("controls must be a dictionary")
    controls = controls.copy()
    if "laplace_roots" not in controls and "laplace_weights" not in controls:
        controls["laplace_roots"], controls["laplace_weights"] = default_laplace_grid()
    if controls.get("mode", "deterministic") == "stochastic" and "production_samples" not in controls:
        controls.setdefault("energy_tolerance", DEFAULT_ENERGY_TOLERANCE)
    _validate_controls(controls)
    return controls


def _validate_controls(controls):
    if not isinstance(controls, dict):
        raise TypeError("controls must be a dictionary")
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
    _check_domain_inputs(mf, static.thresholds)
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
    if with_grad:
        common, common_pullback = jax.vjp(lambda mf_: rebuild_domain_data(mf_, static), mf)
        mf_bar, common_bar = _zero_term_cotangents(mf, common)
    else:
        common = rebuild_domain_data(mf, static)
        common_pullback = mf_bar = common_bar = None
    return {"mf": mf, "common": common, "common_pullback": common_pullback,
            "scf_pullback": scf_pullback, "mf_bar": mf_bar, "common_bar": common_bar,
            "energy": np.zeros((), dtype=common.s1e.dtype), "variance": 0.0,
            "specs": _correlation_term_specs(static)}


def _prepare(state, static, fragment_id, controls, with_grad):
    check_weighted_memory(state["mf"], static, fragment_id, with_grad=with_grad,
                          virtual_block_size=controls.get("virtual_block_size", 16))
    anchors = select_virtual_anchor_columns(state["common"], static, fragment_id)
    prepare = lambda mf_, common_: prepare_weighted_inputs(
        mf_, common_, static, fragment_id, virtual_anchor_columns=anchors,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="integral-direct local Lov is unavailable.*",
                                category=RuntimeWarning)
        if with_grad:
            inputs, pullback = jax.vjp(prepare, state["mf"], state["common"])
        else:
            inputs, pullback = prepare(state["mf"], state["common"]), None
    return inputs, backend.host_inputs(inputs), pullback


def _add_bars(state, mf_bar, common_bar):
    state["mf_bar"] = jax.tree_util.tree_map(_add_cotangent, state["mf_bar"], mf_bar)
    state["common_bar"] = jax.tree_util.tree_map(_add_cotangent,
                                              state["common_bar"], common_bar)


def _finish(state, *, with_grad, include_hf):
    energy = state["energy"]
    lib.logger.info(state["mf"], "STC correlation energy %.12g; stochastic standard error %.6g",
                    float(energy), math.sqrt(state["variance"]))
    if not with_grad:
        jax.block_until_ready(energy)
        return energy
    mf_bar = state["mf_bar"]
    if state.get("common_pullback") is not None:
        common_mf_bar, = state["common_pullback"](state["common_bar"])
        mf_bar = jax.tree_util.tree_map(_add_cotangent, mf_bar, common_mf_bar)
    return _finish_response(state, energy, mf_bar, include_hf=include_hf)


def _finish_response(state, energy, mf_bar, *, include_hf):
    """Add HF once, then apply the single implicit SCF response."""
    if include_hf:
        hf_energy, hf_pullback = jax.vjp(lambda mf_: mf_.e_tot, state["mf"])
        hf_bar, = hf_pullback(np.ones((), dtype=hf_energy.dtype))
        mf_bar = jax.tree_util.tree_map(_add_cotangent, mf_bar, hf_bar)
        energy = energy + hf_energy
    mol_bar, = state["scf_pullback"](mf_bar)
    jax.block_until_ready((energy, mol_bar))
    return energy, mol_bar


def _run_domain(initializer, static, controls, comm, *, with_grad, include_hf=False):
    rank = 0 if comm is None else comm.Get_rank()
    state = _root_call(comm, rank, initializer, "initial preparation")
    specs = state["specs"] if rank == 0 else None
    if comm is not None:
        specs = comm.bcast(specs, root=0)
    domain_controls = controls.copy() if rank == 0 else None
    ndomains = sum(spec[0] == "strong" for spec in specs)
    if rank == 0 and ndomains and "energy_tolerance" in domain_controls:
        total_tolerance = float(domain_controls["energy_tolerance"])
        domain_controls["energy_tolerance"] = total_tolerance / ndomains
        lib.logger.info(state["mf"],
                        "STC requested energy tolerance %.6g Eh; %d domains; per-domain %.6g Eh",
                        total_tolerance, ndomains, domain_controls["energy_tolerance"])
    for spec in specs:
        kind, fragment_id, _ = spec
        if kind == "strong":
            prepared = _root_call(
                comm, rank, lambda: _prepare(state, static, fragment_id, controls, with_grad),
                f"domain {fragment_id} preparation",
            )
            inputs = prepared[1] if rank == 0 else None
            result = run_backend(
                inputs, {"fragment_id": fragment_id} if rank == 0 else None,
                domain_controls if rank == 0 else None, partial(backend.solve, with_grad=with_grad),
                comm=comm, array_names=WEIGHTED_ARRAY_NAMES,
            )

            def apply_domain():
                if with_grad:
                    bars = apply_weighted_pullback(prepared[2], prepared[0], result)
                    _add_bars(state, *bars)
                state["energy"] = state["energy"] + result["energy"]
                state["variance"] += float(result["energy_standard_error"]) ** 2
                jax.block_until_ready((state["energy"], state["mf_bar"], state["common_bar"]))

            _root_call(comm, rank, apply_domain, f"domain {fragment_id} pullback")
            del prepared, inputs, result, apply_domain
        else:
            def apply_weak():
                term = lambda mf_, common_: _correlation_term_energy(mf_, common_, static, spec)
                if with_grad:
                    energy, pullback = jax.vjp(term, state["mf"], state["common"])
                    _add_bars(state, *pullback(np.ones((), dtype=energy.dtype)))
                    del pullback
                else:
                    energy = term(state["mf"], state["common"])
                state["energy"] = state["energy"] + energy
                jax.block_until_ready((state["energy"], state["mf_bar"], state["common_bar"]))

            _root_call(comm, rank, apply_weak, f"weak pair {spec[1:]} pullback")
            del apply_weak
        gc.collect()
    return _root_call(comm, rank, lambda: _finish(state, with_grad=with_grad,
                                                include_hf=include_hf), "final response")


def _select_scope(static, scope, frozen):
    """Validate root-owned call options before any scientific scope branch."""
    if scope not in ("domain", "system"):
        raise ValueError("scope must be domain or system")
    if scope == "domain" and frozen is not None:
        raise ValueError("domain frozen setting must come from static; frozen must be None")
    if scope == "system" and static is not None:
        raise ValueError("system scope requires static=None")
    return scope


def _shared_scope(static, scope, frozen, comm):
    rank = 0 if comm is None else comm.Get_rank()
    selected = _root_call(comm, rank, lambda: _select_scope(static, scope, frozen),
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


def _initialize_scf(mol, build_mf):
    """Build one implicit SCF VJP shared by both correlation scopes."""
    _coordinate_only(mol)
    if not callable(build_mf):
        raise TypeError("build_mf must be callable")
    with (config_update("pyscfad_scf_implicit_diff", True),
          config_update("pyscfad_scf_first_order_custom", False)):
        return jax.vjp(lambda mol_: _fix_restart_mo_phases(build_mf(mol_)), mol)


def _initialize_system(mf, frozen, controls, *, with_grad, scf_pullback=None):
    _validate_controls(controls)
    occupied, virtual = _system_active_indices(mf, frozen)
    dtype = mf.mo_coeff.dtype
    return {"mf": mf, "occupied": occupied, "virtual": virtual,
            "scf_pullback": scf_pullback, "energy": np.zeros((), dtype=dtype),
            "variance": 0.0,
            "mf_bar": (_zero_mf_cotangent(mf, dtype)
                       if with_grad and not (len(occupied) and len(virtual)) else None)}


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
    check_system_memory(state["mf"], len(state["occupied"]), len(state["virtual"]),
                        with_grad=with_grad,
                        virtual_block_size=controls.get("virtual_block_size", 16),
                        auxiliary_group_size=controls.get("auxiliary_group_size", 32))
    _validate_system_metric(state["mf"])
    prepare = lambda mf_: prepare_system_inputs(mf_, state["occupied"], state["virtual"])
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
                state["mf_bar"], = apply_full_pullback(prepared[2], prepared[0], result)
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


def kernel(mf, static=None, *, scope="domain", frozen=None, controls, comm=None):
    """Return domain or whole-system correlation energy on root."""
    scope = _shared_scope(static, scope, frozen, comm)
    rank = 0 if comm is None else comm.Get_rank()
    controls = _root_call(comm, rank, lambda: _driver_controls(controls), "control validation")

    def initialize():
        if _contains_tracer(mf):
            raise TypeError("STC kernel is forward-only; use value_and_grad")
        if scope == "system":
            return _initialize_system(mf, frozen, controls, with_grad=False)
        return _initialize(mf, static, controls, with_grad=False)

    if scope == "system":
        return _run_system(initialize, controls, comm, with_grad=False)
    return _run_domain(initialize, static, controls, comm, with_grad=False)


def value_and_grad(mol, build_mf, static=None, *, scope="domain", frozen=None,
                   controls, comm=None, include_hf=False):
    """Return ``(energy, mol_bar)`` on root; worker ranks return ``None``.

    Numerical selections and the Laplace grid are fixed. Coordinate response
    includes the full Fock matrices, orbitals, direct DF factors and SCF.
    Stochastic energy uncertainty does not bound gradient uncertainty.
    """
    scope = _shared_scope(static, scope, frozen, comm)
    rank = 0 if comm is None else comm.Get_rank()
    controls = _root_call(comm, rank, lambda: _driver_controls(controls), "control validation")

    def initialize():
        _coordinate_only(mol)
        if scope == "domain":
            _validate_static(static)
        _validate_controls(controls)
        mf, scf_pullback = _initialize_scf(mol, build_mf)
        if scope == "system":
            return _initialize_system(mf, frozen, controls, with_grad=True,
                                      scf_pullback=scf_pullback)
        return _initialize(mf, static, controls, with_grad=True,
                           scf_pullback=scf_pullback)

    if scope == "system":
        return _run_system(initialize, controls, comm, with_grad=True, include_hf=include_hf)
    return _run_domain(initialize, static, controls, comm, with_grad=True, include_hf=include_hf)
