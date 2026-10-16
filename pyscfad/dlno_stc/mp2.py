"""Grouped selected-target MP2 exchange and molecular replay."""
from pathlib import Path
import gc
import math
import jax
import numpy
from pyscfad import config_update, numpy as np
from ._restart import (RestartManager, scientific_digest, grouping_payload,
                       validate_grouping_payload)
from ._selection import DomainSelections, build_domain_groups
from .domain import rebuild_domain_data
from .adjoint import (pullback_inputs, add_cotangent, fix_restart_mo_phases,
                      zero_term_cotangents)
from ._pairs import weak_pair_energy_and_grad
from .exchange import read_input, read_result, write_input, write_result
from .parallel import run_backend
from .prepare import prepare_boys_group_inputs
from .protocol import validate_result, canonical_json
from ._workflow import (check_memory, format_domain_dimensions, revision_status,
                        run_payload, allocate_domain_controls)

_METHOD = "dlno-stc-selected-mp2"
_ENERGY_KIND = "finite_boys_target_mp2_v1"
_FRAME = "local_before_fock_v2"

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



def _run_payload(mf, frozen):
    return run_payload(mf, frozen, driver=_METHOD)


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


def _run_identity(mf, static, groups, *, scientific_payload=None):
    if scientific_payload is None:
        scientific_payload = _run_payload(mf, static.frozen)
    revision, dirty = revision_status()
    return {"checkpoint_id": scientific_digest(static),
            "molecular_fingerprint": scientific_digest(scientific_payload),
            "grouping_fingerprint": grouping_payload(static, groups)["fingerprint"],
            "code_revision": revision, "dirty_worktree": dirty}


def _packet_metadata(mf, static, group_id, group, inputs, groups, budget, *, identity=None):
    fragment = static.fragments[group.representative_index]
    candidates = [int(static.fragments[i].iao_indices[0]) for i in fragment.strong_fragments]
    if identity is None:
        identity = _run_identity(mf, static, groups)
    return {
        "schema_version": 1, "method": "mp2", "energy_kind": _ENERGY_KIND,
        "fragment_id": group.representative_index, "group_id": group_id,
        "target_indices": list(group.target_indices),
        "global_target_orbitals": [int(static.fragments[i].iao_indices[0]) for i in group.fragment_indices],
        "fragment_indices": list(group.fragment_indices),
        "representative_index": group.representative_index,
        "candidate_order": candidates,
        "frame_convention": _FRAME, "basis_frame": "orthonormal_local", "B_axes": "Pia",
        "units": {"energy": "Eh", "length": "bohr"},
        "frozen": numpy.asarray(static.frozen).tolist(),
        "orbital_order": {"occupied": "saved Boys candidates, metric Cholesky before Fock diagonalization",
                          "virtual": "retained PAO subspace, metric orthonormalization before Fock diagonalization"},
        "auxiliary_order": {"atoms": numpy.asarray(fragment.extended_atoms).tolist(),
                            "auxbasis": str(mf.with_df.auxbasis)},
        **identity,
        "boys_domain_model": "local_selector_v1", "budget": budget,
        "domain_options": {"support": "finite", "lo_type": "boys",
                           "thresholds": repr(static.thresholds),
                           "extended_ao_indices": numpy.asarray(fragment.extended_ao_indices).tolist(),
                           "strong_fragments": numpy.asarray(fragment.strong_fragments).tolist()},
    }


def _validate_packet_identity(mf, static, group_id, group, inputs, metadata, controls, groups,
                              *, identity=None):
    _validate_budget(metadata.get("budget"), controls, static, groups)
    expected = _packet_metadata(mf, static, group_id, group, inputs, groups,
                                metadata["budget"], identity=identity)
    for field, value in expected.items():
        if field not in ("code_revision", "dirty_worktree") and canonical_json(metadata.get(field)) != canonical_json(value):
            raise ValueError(f"request {field} does not match replay")


def export_run(mf, static, workdir, *, controls, reporter=None, merge_identical=True):
    """Export one selected MP2 packet per exact group and allocate once."""
    if not isinstance(static, DomainSelections):
        raise TypeError("static must be STC DomainSelections")
    if static.frozen is None:
        raise ValueError("export requires an explicit frozen-space setting")
    from .driver import _driver_controls, _validate_static, _validate_reference
    _validate_static(static)
    _validate_reference(mf, static)
    controls = _driver_controls(controls, scope="domain")
    groups = build_domain_groups(static, merge_identical=merge_identical)
    effective = allocate_domain_controls(controls, len(groups))
    budget = _budget(controls, effective, static, groups)
    scientific_payload = _run_payload(mf, static.frozen)
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
        for group_id, group in enumerate(groups):
            fragment = static.fragments[group.representative_index]
            nocc = len(fragment.strong_occ_metric_keep)
            nvir = len(fragment.strong_virtual.metric_keep)
            check_memory(mf, nocc, nvir)
            inputs = prepare_boys_group_inputs(mf, common, static, group)
            metadata = _packet_metadata(mf, static, group_id, group, inputs, groups, budget,
                                        identity=identity)
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


def load_static_from_run(mf, workdir, *, frozen, controls=None):
    """Deserialize STC selections and validate saved grouping and budget."""
    if frozen is None:
        raise ValueError("import requires an explicit frozen-space setting")
    checkpoint = RestartManager(workdir, resume=True, method=_METHOD,
        scientific_payload=_run_payload(mf, frozen), initialize=False)
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


def imported_value_and_grad(mol, build_mf, static, packet_loader, *,
                            fragment_ids=None, include_hf=True,
                            include_weak=True, merge_identical=True):
    """Replay saved grouped three-tensor MP2 bars, then weak/common/SCF response.

    Fragment IDs identify exported jobs. Replay consumes effective controls
    without invoking the total-budget allocator again.
    """
    if not isinstance(static, DomainSelections):
        raise TypeError("static must be STC DomainSelections")
    if not callable(packet_loader):
        raise TypeError("packet_loader must be callable")
    from .driver import _coordinate_only, _validate_static, _validate_reference
    _coordinate_only(mol)
    _validate_static(static)
    groups = build_domain_groups(static, merge_identical=merge_identical)
    ids = tuple(range(len(groups))) if fragment_ids is None else tuple(fragment_ids)
    if ((not ids and groups) or len(set(ids)) != len(ids)
            or any(type(i) is not int or i < 0 or i >= len(groups) for i in ids)):
        raise ValueError("fragment_ids contains empty, duplicate or invalid jobs")
    if len(ids) != len(groups) and (include_hf or include_weak):
        raise ValueError("partial import requires include_hf=False and include_weak=False")
    with (config_update("pyscfad_scf_implicit_diff", True),
          config_update("pyscfad_scf_first_order_custom", False)):
        mf, scf_pullback = jax.vjp(lambda m: fix_restart_mo_phases(build_mf(m)), mol)
    _validate_reference(mf, static)
    identity = _run_identity(mf, static, groups)
    common, common_pullback = jax.vjp(lambda m: rebuild_domain_data(m, static), mf)
    mf_bar, common_bar = zero_term_cotangents(mf, common)
    energy = np.zeros((), dtype=common.s1e.dtype)
    run_controls = None
    for group_id in ids:
        group = groups[group_id]
        saved, metadata, controls, result = packet_loader(group_id)
        _validate_packet_identity(mf, static, group_id, group, saved, metadata, controls, groups,
                                  identity=identity)
        control_identity = canonical_json({"controls": controls, "budget": metadata["budget"]})
        if run_controls is not None and control_identity != run_controls:
            raise ValueError("packet controls or budget differ within one run")
        run_controls = control_identity
        validate_result(result, saved, metadata, controls=controls)
        bars = pullback_inputs(lambda m, c: prepare_boys_group_inputs(m, c, static, group),
                               (mf, common), saved, result, replay_atol=1e-9)
        mf_bar, common_bar = jax.tree_util.tree_map(add_cotangent, (mf_bar, common_bar), bars)
        energy += np.asarray(result["energy"])
        del saved, result, bars
        gc.collect()
    if include_weak:
        weak, bars = weak_pair_energy_and_grad(mf, common, static)
        energy += weak
        mf_bar, common_bar = jax.tree_util.tree_map(add_cotangent, (mf_bar, common_bar), bars)
    common_mf_bar, = common_pullback(common_bar)
    mf_bar = jax.tree_util.tree_map(add_cotangent, mf_bar, common_mf_bar)
    if include_hf:
        hf, pullback = jax.vjp(lambda m: m.e_tot, mf)
        hf_bar, = pullback(np.ones_like(hf))
        mf_bar = jax.tree_util.tree_map(add_cotangent, mf_bar, hf_bar)
        energy += hf
    mol_bar, = scf_pullback(mf_bar)
    jax.block_until_ready((energy, mol_bar))
    return energy, mol_bar
