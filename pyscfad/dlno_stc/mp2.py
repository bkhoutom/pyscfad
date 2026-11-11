"""Grouped selected-target MP2 exchange and molecular replay."""
from pathlib import Path
import gc
import math
import jax
import numpy
from pyscfad import numpy as np
from ._restart import (RestartManager, scientific_digest, grouping_payload,
                       validate_grouping_payload)
from ._selection import DomainSelections, build_domain_groups
from .domain import rebuild_domain_data
from .adjoint import pullback_inputs
from .exchange import read_input, read_result, write_input, write_result
from .parallel import run_backend
from .prepare import prepare_boys_group_inputs
from .protocol import validate_result, canonical_json
from ._workflow import (check_memory, format_domain_dimensions, revision_status,
                        run_payload, allocate_domain_controls)
from ._memory import memory_calculation
from ._df_basis import basis_kwargs

_METHOD = "dlno-stc-selected-mp2"
_ENERGY_KIND = "finite_boys_target_mp2_v1"
_FRAME = "local_before_fock_v3"

def solve_request(input_path, result_path, backend, *, comm=None):
    """Call a backend for one saved domain; only the team root writes output."""
    if comm is None:
        inputs, metadata, controls = read_input(input_path)
        result = run_backend(inputs, metadata, controls, backend)
        validate_result(result, inputs, metadata, controls=controls)
        write_result(result_path, result)
        return result

    rank = comm.Get_rank()
    inputs = metadata = controls = None
    if rank == 0:
        try:
            inputs, metadata, controls = read_input(input_path)
            error = None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    else:
        error = None
    error = comm.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(f"root request read failed: {error}")

    result = run_backend(inputs, metadata, controls, backend, comm=comm)
    if rank == 0:
        try:
            validate_result(result, inputs, metadata, controls=controls)
            write_result(result_path, result)
            error = None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    error = comm.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(f"root result write failed: {error}")
    return result if rank == 0 else None



def _run_payload(mf, frozen, *, correlation_auxbasis=None):
    return run_payload(mf, frozen, driver=_METHOD,
        **basis_kwargs(correlation_auxbasis))


def _budget(controls, effective, static, groups):
    return {"total_tolerance": controls.get("energy_tolerance"),
            "original_target_count": len(static.fragments),
            "group_count": len(groups),
            "effective_tolerance": effective.get("energy_tolerance")}


def _validate_budget(budget, controls, static, groups):
    if not isinstance(budget, dict):
        raise ValueError("missing sampling budget identity")
    total = budget.get("total_tolerance")
    expected = {"total_tolerance": total,
                "original_target_count": len(static.fragments),
                "group_count": len(groups),
                "effective_tolerance": (total if not groups or total is None
                                        else total / math.sqrt(len(groups)))}
    if budget != expected or controls.get("energy_tolerance") != expected["effective_tolerance"]:
        raise ValueError("packet sampling budget does not match grouped controls")


def _run_identity(mf, static, groups, *, scientific_payload=None, correlation_auxbasis=None):
    if scientific_payload is None:
        scientific_payload = _run_payload(mf, static.frozen,
            **basis_kwargs(correlation_auxbasis))
    revision, dirty = revision_status()
    return {"checkpoint_id": scientific_digest(static),
            "molecular_fingerprint": scientific_digest(scientific_payload),
            "grouping_fingerprint": grouping_payload(static, groups)["fingerprint"],
            "code_revision": revision, "dirty_worktree": dirty}


def _packet_metadata(mf, static, group_id, group, inputs, groups, budget, *, identity=None,
                     correlation_auxbasis=None):
    fragment = static.fragments[group.representative_index]
    candidates = [int(static.fragments[i].iao_indices[0]) for i in fragment.strong_fragments]
    if identity is None:
        identity = _run_identity(mf, static, groups, correlation_auxbasis=correlation_auxbasis)
    return {
        "schema_version": 1, "method": "mp2", "energy_kind": _ENERGY_KIND,
        "fragment_id": group.representative_index, "group_id": group_id,
        "target_indices": list(group.target_indices),
        "global_target_orbitals": [int(static.fragments[i].iao_indices[0]) for i in group.fragment_indices],
        "fragment_indices": list(group.fragment_indices),
        "representative_index": group.representative_index,
        "candidate_order": candidates,
        "virtual_anchor_parent_columns": list(fragment.virtual_anchor_columns),
        "frame_convention": _FRAME, "basis_frame": "orthonormal_local", "B_axes": "Pia",
        "units": {"energy": "Eh", "length": "bohr"},
        "frozen": numpy.asarray(static.frozen).tolist(),
        "orbital_order": {"occupied": "saved Boys candidates, metric Cholesky before Fock diagonalization",
                          "virtual": "saved PAO anchors, metric Cholesky before Fock diagonalization"},
        "auxiliary_order": {"atoms": numpy.asarray(fragment.extended_atoms).tolist(),
                            "auxbasis": str(mf.with_df.auxbasis if correlation_auxbasis is None
                                            else correlation_auxbasis)},
        **basis_kwargs(correlation_auxbasis),
        **identity,
        "boys_domain_model": "local_selector_v1", "budget": budget,
        "domain_options": {"support": "finite", "lo_type": "boys",
                           "thresholds": repr(static.thresholds),
                           "extended_ao_indices": numpy.asarray(fragment.extended_ao_indices).tolist(),
                           "strong_fragments": numpy.asarray(fragment.strong_fragments).tolist()},
    }


def _validate_packet_identity(mf, static, group_id, group, inputs, metadata, controls, groups,
                              *, identity=None, correlation_auxbasis=None):
    _validate_budget(metadata.get("budget"), controls, static, groups)
    expected = _packet_metadata(mf, static, group_id, group, inputs, groups,
                                metadata["budget"], identity=identity,
                                correlation_auxbasis=correlation_auxbasis)
    for field, value in expected.items():
        if field not in ("code_revision", "dirty_worktree") and canonical_json(metadata.get(field)) != canonical_json(value):
            raise ValueError(f"request {field} does not match replay")


@memory_calculation
def export_run(mf, static, workdir, *, controls, reporter=None, merge_identical=True,
               correlation_auxbasis=None):
    """Export one selected MP2 packet per exact group and allocate once."""
    if not isinstance(static, DomainSelections):
        raise TypeError("static must be STC DomainSelections")
    if static.frozen is None:
        raise ValueError("export requires an explicit frozen-space setting")
    from .driver import _driver_controls, _validate_static, _validate_reference
    _validate_static(static)
    correlation = basis_kwargs(correlation_auxbasis)
    _validate_reference(mf, static, **correlation)
    controls = _driver_controls(controls, scope="domain")
    groups = build_domain_groups(static, merge_identical=merge_identical)
    effective = allocate_domain_controls(controls, len(groups))
    budget = _budget(controls, effective, static, groups)
    scientific_payload = _run_payload(mf, static.frozen, **correlation)
    identity = _run_identity(mf, static, groups, scientific_payload=scientific_payload)
    checkpoint = RestartManager(workdir, method=_METHOD,
        scientific_payload=scientific_payload, initialize=True)
    try:
        checkpoint.save_static(static)
        checkpoint.save_record("exchange", metadata={
            "grouping": grouping_payload(static, groups), "budget": budget,
            "controls": effective, "merge_identical": merge_identical})
        common = rebuild_domain_data(mf, static)
        rows = []
        from ._workflow import packet_auxiliary_size
        naux = packet_auxiliary_size(mf, **correlation) if groups else None
        for group_id, group in enumerate(groups):
            fragment = static.fragments[group.representative_index]
            nocc = len(fragment.strong_occ_metric_keep)
            nvir = len(fragment.strong_virtual.metric_keep)
            check_memory(mf, nocc, nvir, naux=naux, **correlation)
            inputs = prepare_boys_group_inputs(mf, common, static, group, **correlation)
            metadata = _packet_metadata(mf, static, group_id, group, inputs, groups, budget,
                                        identity=identity, **correlation)
            write_input(Path(workdir) / f"fragment_{group_id:04d}" / "input.h5", inputs, metadata, effective)
            rows.append({"fragment_id": group_id, "fragment_indices": group.fragment_indices,
                         "target_indices": group.target_indices, "ed_ao": len(fragment.extended_ao_indices),
                         "nocc": nocc, "nvir": nvir, "naux": inputs["B"].shape[0]})
            del inputs
            gc.collect()
        if reporter is not None:
            reporter(format_domain_dimensions(rows))
        return rows
    finally:
        checkpoint.close()


def load_static_from_run(mf, workdir, *, frozen, controls=None, correlation_auxbasis=None):
    """Deserialize STC selections and validate saved grouping and budget."""
    if frozen is None:
        raise ValueError("import requires an explicit frozen-space setting")
    checkpoint = RestartManager(workdir, resume=True, method=_METHOD,
        scientific_payload=_run_payload(mf, frozen,
            **basis_kwargs(correlation_auxbasis)), initialize=False)
    try:
        static = checkpoint.load_static(expected_type=DomainSelections)
        if static is None:
            raise FileNotFoundError("run has no completed static selection checkpoint")
        record = checkpoint.load_record("exchange", missing_ok=False)
        md = record.metadata
        groups = build_domain_groups(static, merge_identical=md["merge_identical"])
        validate_grouping_payload(md["grouping"], static, groups)
        _validate_budget(md["budget"], md["controls"], static, groups)
        if controls is not None:
            from .driver import _driver_controls
            controls = _driver_controls(controls, scope="domain")
            requested = dict(md["controls"])
            if md["budget"]["total_tolerance"] is not None:
                requested["energy_tolerance"] = md["budget"]["total_tolerance"]
            if canonical_json(controls) != canonical_json(requested):
                raise ValueError("requested controls do not match saved exchange run")
        return static
    finally:
        checkpoint.close()


def load_disk_packet(workdir, fragment_id):
    folder = Path(workdir) / f"fragment_{int(fragment_id):04d}"
    inputs, metadata, controls = read_input(folder / "input.h5")
    result = read_result(folder / "result.h5", inputs, metadata, controls=controls)
    return inputs, metadata, controls, result


@memory_calculation
def imported_value_and_grad(mol, build_mf, static, packet_loader, *,
                            fragment_ids=None, include_hf=True,
                            include_weak=True, merge_identical=True,
                            correlation_auxbasis=None):
    """Replay saved grouped three-tensor MP2 bars, then weak/common/SCF response.

    Fragment IDs identify exported jobs. Replay consumes effective controls
    without invoking the total-budget allocator again. Initialization and final
    response use the production driver, including joint SCF overlap/Fock data,
    shared DF geometry response and consumption of completed response tapes.
    """
    if not isinstance(static, DomainSelections):
        raise TypeError("static must be STC DomainSelections")
    if not callable(packet_loader):
        raise TypeError("packet_loader must be callable")
    from . import driver
    correlation = basis_kwargs(correlation_auxbasis)
    driver._coordinate_only(mol)
    driver._validate_static(static)
    groups = build_domain_groups(static, merge_identical=merge_identical)
    ids = tuple(range(len(groups))) if fragment_ids is None else tuple(fragment_ids)
    if ((not ids and groups) or len(set(ids)) != len(ids)
            or any(type(i) is not int or i < 0 or i >= len(groups) for i in ids)):
        raise ValueError("fragment_ids contains empty, duplicate or invalid jobs")
    if len(ids) != len(groups) and (include_hf or include_weak):
        raise ValueError("partial import requires include_hf=False and include_weak=False")
    mf, scf_pullback = driver._initialize_scf(
        mol, build_mf, combine_df=True, share_scf_data=True)
    scf_pullback = driver._serial_scf_pullback(scf_pullback)
    # Read the first packet once for common initialization. Empty worklists
    # have no numerical solver controls; the defaults only validate setup.
    packet = packet_loader(ids[0]) if ids else None
    initial_controls = packet[2] if packet is not None else driver._driver_controls({})
    state = driver._initialize(mf, static, initial_controls, with_grad=True,
                               scf_pullback=scf_pullback, **correlation)
    # State owns the only reference to the tape when _finish consumes it.
    del scf_pullback
    identity = _run_identity(mf, static, groups, **correlation)
    run_controls = None
    for position, group_id in enumerate(ids):
        group = groups[group_id]
        saved, metadata, controls, result = packet if position == 0 else packet_loader(group_id)
        packet = None
        _validate_packet_identity(mf, static, group_id, group, saved, metadata, controls, groups,
                                  identity=identity, **correlation)
        control_identity = canonical_json({"controls": controls, "budget": metadata["budget"]})
        if run_controls is not None and control_identity != run_controls:
            raise ValueError("packet controls or budget differ within one run")
        run_controls = control_identity
        validate_result(result, saved, metadata, controls=controls)
        with driver._mp2_df_scope(
                state, static.fragments[group.representative_index].extended_atoms) as shared:
            bars = pullback_inputs(lambda m, c: prepare_boys_group_inputs(m, c, static, group,
                                                                        **correlation),
                                   (mf, state['common']), saved, result, replay_atol=1e-9)
            driver._add_bars(state, *bars)
            del bars
            state['mf_bar'] = driver._finish_mp2_df(shared, state['mf_bar'])
        state['energy'] += np.asarray(result["energy"])
        # Portable packets do not require/save the native scalar uncertainty.
        # Keep unavailable uncertainty explicit rather than reporting zero.
        state['variance'] += float(result.get('energy_standard_error', numpy.nan)) ** 2
        jax.block_until_ready((state['energy'], state['mf_bar'], state['common_bar']))
        del saved, result
        gc.collect()
    if include_weak:
        weak, bars = driver._weak_energy_and_grad(state, static, state['weak_pairs'])
        state['energy'] += weak
        driver._add_bars(state, *bars)
        del weak, bars
    return driver._finish(state, with_grad=True, include_hf=include_hf,
                          uncertainty_label='packet standard error (NaN if unavailable)')
