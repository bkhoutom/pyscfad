"""Host workflow for DLNO to external MP2 tensor exchange.

One rank or one supplied communicator processes one domain at a time. The
external backend never enters a JAX transform; its unit cotangents are pulled
through freshly reconstructed numerical preparation on the root process.
"""

from pathlib import Path
import gc

import jax
import numpy
from pyscfad import config_update, numpy as np

from pyscfad.dlno._restart import RestartManager, scientific_digest
from pyscfad.dlno._selection import DomainSelections
from pyscfad.dlno.dlno_base import rebuild_domain_data
from pyscfad.dlno.mp2 import (
    _add_cotangent, _fix_restart_mo_phases, _zero_term_cotangents,
)

from .adjoint import pullback_inputs
from .domain import build_stc_domain
from .exchange import read_input, read_result, write_input, write_result
from .parallel import run_backend
from .prepare import prepare_inputs
from .protocol import validate_result
from ._workflow import (
    check_memory as _check_memory, format_domain_dimensions,
    revision_status as _revision_status, run_payload,
)


_METHOD = "dlno-stc-mp2"


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
    """Keep the established full-support checkpoint identity."""
    return run_payload(mf, frozen, driver=_METHOD)


def _packet_metadata(mf, static, fragment_id, domain, checkpoint_id):
    revision, dirty = _revision_status()
    frozen = numpy.asarray(static.frozen).tolist()
    return {
        "schema_version": 1,
        "method": "mp2",
        "energy_kind": "boys_target_mp2_v1",
        "fragment_id": int(fragment_id),
        "target_index": int(domain.target_index),
        "basis_frame": "orthonormal_local",
        "B_axes": "Pia",
        "units": {"energy": "Eh", "length": "bohr"},
        "frozen": frozen,
        "orbital_order": {
            "occupied": "Boys target order in saved selections",
            "virtual": numpy.asarray(static.active_vir_indices).tolist(),
        },
        "auxiliary_order": {
            "atoms": numpy.asarray(domain.extended_atoms).tolist(),
            "auxbasis": str(mf.with_df.auxbasis),
        },
        "checkpoint_id": checkpoint_id,
        "domain_options": {
            "support": "full", "lo_type": "boys",
            "thresholds": repr(static.thresholds),
        },
        "code_revision": revision,
        "dirty_worktree": dirty,
    }


def export_run(mf, static, workdir, *, controls, reporter=None):
    """Save fixed selections and one complete input request per Boys target."""
    if not isinstance(static, DomainSelections):
        raise TypeError("static must be DomainSelections")
    if static.frozen is None:
        raise ValueError("export requires an explicit frozen-space setting")
    workdir = Path(workdir)
    checkpoint = RestartManager(
        workdir, method=_METHOD, scientific_payload=_run_payload(mf, static.frozen),
        initialize=True,
    )
    try:
        checkpoint.save_static(static)
        common = rebuild_domain_data(mf, static)
        rows = []
        for fragment_id in range(len(static.fragments)):
            domain = build_stc_domain(mf, common, static, fragment_id)
            nocc = domain.occupied_coeff.shape[1]
            nvir = domain.virtual_coeff.shape[1]
            _check_memory(mf, nocc, nvir)
            inputs = prepare_inputs(mf, common, static, fragment_id)
            metadata = _packet_metadata(
                mf, static, fragment_id, domain, checkpoint.static_digest,
            )
            folder = workdir / f"fragment_{fragment_id:04d}"
            write_input(folder / "input.h5", inputs, metadata, controls)
            rows.append({
                "fragment_id": fragment_id,
                "ed_ao": len(domain.extended_ao_indices),
                "nocc": nocc,
                "nvir": nvir,
            })
            del inputs, domain
            gc.collect()
        if reporter is not None:
            reporter(format_domain_dimensions(rows))
        return rows
    finally:
        checkpoint.close()


def load_static_from_run(mf, workdir, *, frozen):
    """Load saved fixed selections against this live molecular preparation."""
    if frozen is None:
        raise ValueError("import requires an explicit frozen-space setting")
    checkpoint = RestartManager(
        workdir, resume=True, method=_METHOD,
        scientific_payload=_run_payload(mf, frozen), initialize=False,
    )
    try:
        static = checkpoint.load_static(expected_type=DomainSelections)
        if static is None:
            raise FileNotFoundError("run has no completed static selection checkpoint")
        return static
    finally:
        checkpoint.close()


def load_disk_packet(workdir, fragment_id):
    """Read one validated completed request/result pair from disk."""
    folder = Path(workdir) / f"fragment_{int(fragment_id):04d}"
    inputs, metadata, controls = read_input(folder / "input.h5")
    result = read_result(folder / "result.h5", inputs, metadata, controls=controls)
    return inputs, metadata, controls, result


def imported_value_and_grad(mol, build_mf, static, packet_loader, *,
                            fragment_ids=None, include_hf=True):
    """Replay one domain at a time and close common and SCF response once.

    ``packet_loader(fragment_id)`` returns saved inputs, request metadata,
    controls, and a result. Both disk and in-memory packets use the same
    request/result validation before the numerical pullback.
    The source may be HDF5 or ordinary in-memory dictionaries. This first
    energy replacement supports only complete Boys domains, so every active
    pair is strong and the target terms sum to full active-space MP2.
    """
    if not callable(packet_loader):
        raise TypeError("packet_loader must be callable")
    all_ids = tuple(range(len(static.fragments)))
    if fragment_ids is None:
        selected_ids = all_ids
    else:
        selected_ids = tuple(fragment_ids)
        if not selected_ids:
            raise ValueError("fragment_ids must not be empty")
        if any(type(index) is not int or index not in all_ids
               for index in selected_ids):
            raise ValueError("fragment_ids contains an invalid fragment")
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("fragment_ids contains a duplicate fragment")
        if set(selected_ids) != set(all_ids) and include_hf:
            raise ValueError("partial import requires include_hf=False")
    with (config_update("pyscfad_scf_implicit_diff", True),
          config_update("pyscfad_scf_first_order_custom", False)):
        mf, scf_pullback = jax.vjp(
            lambda mol_: _fix_restart_mo_phases(build_mf(mol_)), mol,
        )
    common, common_pullback = jax.vjp(
        lambda mf_: rebuild_domain_data(mf_, static), mf,
    )
    mf_bar, common_bar = _zero_term_cotangents(mf, common)
    static_digest = scientific_digest(static)
    expected_frozen = numpy.asarray(static.frozen).tolist()
    energy = np.zeros((), dtype=common.s1e.dtype)
    for fragment_id in selected_ids:
        domain = build_stc_domain(mf, common, static, fragment_id)
        _check_memory(mf, domain.occupied_coeff.shape[1],
                      domain.virtual_coeff.shape[1])
        saved_inputs, request_metadata, controls, result = packet_loader(fragment_id)
        if request_metadata.get("checkpoint_id") != static_digest:
            raise ValueError("request checkpoint does not match fixed selections")
        if request_metadata.get("frozen") != expected_frozen:
            raise ValueError("request frozen setting does not match fixed selections")
        validate_result(result, saved_inputs, request_metadata, controls=controls)
        result_metadata = result["metadata"]
        if result_metadata.get("energy_kind") != "boys_target_mp2_v1":
            raise ValueError("test-only energy cannot replace a Boys target term")
        if (result_metadata.get("fragment_id") != fragment_id
                or result_metadata.get("target_index") != domain.target_index):
            raise ValueError("result fragment or target does not match replay")
        prepare_fn = lambda mf_, common_, _fragment=fragment_id: prepare_inputs(
            mf_, common_, static, _fragment,
        )
        term_mf_bar, term_common_bar = pullback_inputs(
            prepare_fn, (mf, common), saved_inputs, result,
        )
        mf_bar = jax.tree_util.tree_map(_add_cotangent, mf_bar, term_mf_bar)
        common_bar = jax.tree_util.tree_map(
            _add_cotangent, common_bar, term_common_bar,
        )
        energy = energy + np.asarray(result["energy"])
        del domain, saved_inputs, result, term_mf_bar, term_common_bar
        gc.collect()

    common_mf_bar, = common_pullback(common_bar)
    mf_bar = jax.tree_util.tree_map(_add_cotangent, mf_bar, common_mf_bar)
    if include_hf:
        hf_energy, hf_pullback = jax.vjp(lambda mf_: mf_.e_tot, mf)
        hf_bar, = hf_pullback(np.ones((), dtype=hf_energy.dtype))
        mf_bar = jax.tree_util.tree_map(_add_cotangent, mf_bar, hf_bar)
        energy = energy + hf_energy
    mol_bar, = scf_pullback(mf_bar)
    jax.block_until_ready((energy, mol_bar))
    return energy, mol_bar
