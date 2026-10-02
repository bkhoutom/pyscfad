"""Finite DLNO domain packets and molecular pullback for an external solver.

Each packet contains only the full ``foo``, ``fvv``, and ``B`` tensors
in one pre-semicanonical orthonormal finite local frame.  Its scalar energy and three unit cotangents describe the *external*
solver's function of those tensors.  In particular, this is not the weighted
strong/weak energy used by :class:`pyscfad.dlno.mp2.DLNOMP2`.

The saved selections are discrete.  Replay rebuilds every continuous local
orbital, Fock block, and RI factor from the current differentiable SCF state.
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
from .exchange import read_input, read_result, write_input
from ._workflow import (
    check_memory as _check_memory, format_domain_dimensions,
    revision_status as _revision_status, run_payload,
)
from .domain import _validate_virtual_anchor_columns
from .prepare import prepare_finite_export_inputs, prepare_finite_inputs
from .protocol import validate_result


_METHOD = "dlno-stc-finite-three-tensor"
_ENERGY_KIND = "finite_three_tensor_v1"
_ORBITAL_ORDER = {
    "occupied": (
        "projected strong-partner Boys candidates in saved order, then "
        "fixed-rank metric orthonormalization before Fock diagonalization"
    ),
    "virtual": (
        "fixed QR-selected PAO parent columns projected into the retained "
        "PAO subspace, then metric-Cholesky orthonormalized before Fock "
        "diagonalization"
    ),
    "phase": "largest-magnitude ED AO component of each column is positive",
}


def _finite_run_payload(mf, frozen):
    return run_payload(mf, frozen, driver=_METHOD)


def _fragment_ids(static, fragment_ids):
    all_ids = tuple(range(len(static.fragments)))
    if fragment_ids is None:
        return all_ids
    selected = tuple(fragment_ids)
    if not selected:
        raise ValueError("fragment_ids must not be empty")
    if any(type(index) is not int or index not in all_ids for index in selected):
        raise ValueError("fragment_ids contains an invalid fragment")
    if len(set(selected)) != len(selected):
        raise ValueError("fragment_ids contains a duplicate fragment")
    return selected


def _require_singleton_boys_targets(static):
    if static.lo_type != "boys" or any(
        len(fragment.iao_indices) != 1 for fragment in static.fragments
    ):
        raise ValueError(
            "finite three-tensor packets currently require one Boys orbital "
            "per target domain"
        )


def _dimensions(static, fragment_id):
    fragment = static.fragments[fragment_id]
    return {
        "fragment_id": fragment_id,
        "ed_ao": len(fragment.extended_ao_indices),
        "nocc": len(fragment.strong_occ_metric_keep),
        "nvir": len(fragment.strong_virtual.metric_keep),
    }


def _packet_metadata(
    mf, static, fragment_id, inputs, checkpoint_id, virtual_anchor_columns,
):
    """Describe the actual finite ED and its *global* Boys target label.

    ``target_index=None`` is deliberate: a global Boys target is generally a
    mixture of the local occupied orbitals and has no local index.
    """
    fragment = static.fragments[fragment_id]
    dimensions = _dimensions(static, fragment_id)
    revision, dirty = _revision_status()
    return {
        "schema_version": 1,
        "method": "mp2",
        "energy_kind": _ENERGY_KIND,
        "fragment_id": fragment_id,
        "target_index": None,
        "global_target_id": int(fragment.iao_indices[0]),
        "global_target_orbitals": numpy.asarray(fragment.iao_indices).tolist(),
        "virtual_anchor_parent_columns": list(virtual_anchor_columns),
        "basis_frame": "orthonormal_local",
        "B_axes": "Pia",
        "units": {"energy": "Eh", "length": "bohr"},
        "frozen": numpy.asarray(static.frozen).tolist(),
        "orbital_order": dict(_ORBITAL_ORDER),
        "auxiliary_order": {
            "atoms": numpy.asarray(fragment.extended_atoms).tolist(),
            "auxbasis": str(mf.with_df.auxbasis),
        },
        "checkpoint_id": checkpoint_id,
        "domain_options": {
            "support": "finite",
            "lo_type": static.lo_type,
            "thresholds": repr(static.thresholds),
            "extended_ao_indices": numpy.asarray(
                fragment.extended_ao_indices
            ).tolist(),
            "strong_fragments": numpy.asarray(
                fragment.strong_fragments
            ).tolist(),
            "ed_ao": dimensions["ed_ao"],
            "nocc": dimensions["nocc"],
            "nvir": dimensions["nvir"],
            "naux": int(inputs["B"].shape[0]),
        },
        "code_revision": revision,
        "dirty_worktree": dirty,
    }


def _validate_packet_identity(static, fragment_id, saved_inputs, metadata):
    fragment = static.fragments[fragment_id]
    dimensions = _dimensions(static, fragment_id)
    if metadata.get("energy_kind") != _ENERGY_KIND:
        raise ValueError("expected a finite three-tensor request")
    if metadata.get("basis_frame") != "orthonormal_local":
        raise ValueError("request is not in the orthonormal local frame")
    if metadata.get("fragment_id") != fragment_id:
        raise ValueError("request fragment does not match replay")
    if (metadata.get("target_index") is not None
            or metadata.get("global_target_id") != int(fragment.iao_indices[0])
            or metadata.get("global_target_orbitals")
            != numpy.asarray(fragment.iao_indices).tolist()):
        raise ValueError("request target does not match fixed selections")
    anchors = _validate_virtual_anchor_columns(
        metadata.get("virtual_anchor_parent_columns"),
        fragment.strong_virtual.parent_columns, dimensions["nvir"],
    )
    if list(anchors) != metadata["virtual_anchor_parent_columns"]:
        raise ValueError("virtual anchor parent columns must be integer labels")
    if metadata.get("checkpoint_id") != scientific_digest(static):
        raise ValueError("request checkpoint does not match fixed selections")
    if metadata.get("frozen") != numpy.asarray(static.frozen).tolist():
        raise ValueError("request frozen setting does not match fixed selections")
    if metadata.get("orbital_order") != _ORBITAL_ORDER:
        raise ValueError("request orbital frame description does not match replay")
    options = metadata.get("domain_options", {})
    expected_options = {
        "support": "finite",
        "lo_type": static.lo_type,
        "thresholds": repr(static.thresholds),
        "extended_ao_indices": numpy.asarray(
            fragment.extended_ao_indices
        ).tolist(),
        "strong_fragments": numpy.asarray(fragment.strong_fragments).tolist(),
        "ed_ao": dimensions["ed_ao"],
        "nocc": dimensions["nocc"],
        "nvir": dimensions["nvir"],
        "naux": int(saved_inputs["B"].shape[0]),
    }
    if options != expected_options:
        raise ValueError("request finite domain selections do not match replay")
    auxiliary = metadata.get("auxiliary_order")
    if not isinstance(auxiliary, dict) or auxiliary.get("atoms") != numpy.asarray(
            fragment.extended_atoms).tolist():
        raise ValueError("request auxiliary atom order does not match replay")
    if (saved_inputs["foo"].shape != (dimensions["nocc"], dimensions["nocc"])
            or saved_inputs["fvv"].shape
            != (dimensions["nvir"], dimensions["nvir"])
            or saved_inputs["B"].shape[1:]
            != (dimensions["nocc"], dimensions["nvir"])):
        raise ValueError("request tensor dimensions do not match finite domain")


def export_finite_run(mf, static, workdir, *, controls, reporter=None):
    """Write one ``input.h5`` for each saved finite DLNO target domain.

    The caller supplies the same fixed :class:`DomainSelections` used by its
    PySCFAD DLNO run, preferably at the reference geometry.  This function
    does not solve the external energy or evaluate a gradient.
    """
    if not isinstance(static, DomainSelections):
        raise TypeError("static must be DomainSelections")
    if static.frozen is None:
        raise ValueError("export requires an explicit frozen-space setting")
    _require_singleton_boys_targets(static)
    workdir = Path(workdir)
    checkpoint = RestartManager(
        workdir, method=_METHOD,
        scientific_payload=_finite_run_payload(mf, static.frozen),
        initialize=True,
    )
    try:
        checkpoint.save_static(static)
        common = rebuild_domain_data(mf, static)
        rows = []
        for fragment_id in range(len(static.fragments)):
            dimensions = _dimensions(static, fragment_id)
            _check_memory(mf, dimensions["nocc"], dimensions["nvir"])
            inputs, anchors = prepare_finite_export_inputs(
                mf, common, static, fragment_id
            )
            metadata = _packet_metadata(
                mf, static, fragment_id, inputs, checkpoint.static_digest,
                anchors,
            )
            _validate_packet_identity(static, fragment_id, inputs, metadata)
            folder = workdir / f"fragment_{fragment_id:04d}"
            write_input(folder / "input.h5", inputs, metadata, controls)
            rows.append(dimensions)
            del inputs
            gc.collect()
        if reporter is not None:
            reporter(format_domain_dimensions(rows))
        return rows
    finally:
        checkpoint.close()


def load_finite_static_from_run(mf, workdir, *, frozen):
    """Load fixed selections after checking the finite run's SCF identity."""
    if frozen is None:
        raise ValueError("import requires an explicit frozen-space setting")
    checkpoint = RestartManager(
        workdir, resume=True, method=_METHOD,
        scientific_payload=_finite_run_payload(mf, frozen), initialize=False,
    )
    try:
        static = checkpoint.load_static(expected_type=DomainSelections)
        if static is None:
            raise FileNotFoundError("run has no completed static selection checkpoint")
        _require_singleton_boys_targets(static)
        return static
    finally:
        checkpoint.close()


def load_finite_disk_packet(workdir, fragment_id):
    """Read one complete finite request/result pair, checking its hash."""
    folder = Path(workdir) / f"fragment_{int(fragment_id):04d}"
    inputs, metadata, controls = read_input(folder / "input.h5")
    result = read_result(folder / "result.h5", inputs, metadata, controls=controls)
    return inputs, metadata, controls, result


def imported_finite_value_and_grad(
    mol, build_mf, static, packet_loader, *, fragment_ids=None,
    include_hf=False,
):
    """Return an external three-tensor energy and its molecular cotangent.

    ``packet_loader(fragment_id)`` returns ``(inputs, metadata, controls,
    result)``.  Each result must contain a unit-seed scalar and cotangents for
    exactly ``foo``, ``fvv``, and ``B``.  The returned gradient differentiates
    that external scalar through finite local-domain preparation, SCF, and
    Boys localization.  It is not the finite DLNO-MP2 reference gradient:
    that energy also depends on target/partner projections and weak pairs.
    Set ``include_hf=True`` only if the external scalar is a correlation term
    to which the RHF energy should be added.
    """
    if not isinstance(static, DomainSelections):
        raise TypeError("static must be DomainSelections")
    if static.frozen is None:
        raise ValueError("import requires an explicit frozen-space setting")
    if not callable(packet_loader):
        raise TypeError("packet_loader must be callable")
    _require_singleton_boys_targets(static)
    selected_ids = _fragment_ids(static, fragment_ids)
    with (config_update("pyscfad_scf_implicit_diff", True),
          config_update("pyscfad_scf_first_order_custom", False)):
        mf, scf_pullback = jax.vjp(
            lambda mol_: _fix_restart_mo_phases(build_mf(mol_)), mol,
        )
    common, common_pullback = jax.vjp(
        lambda mf_: rebuild_domain_data(mf_, static), mf,
    )
    mf_bar, common_bar = _zero_term_cotangents(mf, common)
    energy = np.zeros((), dtype=common.s1e.dtype)
    for fragment_id in selected_ids:
        dimensions = _dimensions(static, fragment_id)
        _check_memory(mf, dimensions["nocc"], dimensions["nvir"])
        saved_inputs, metadata, controls, result = packet_loader(fragment_id)
        _validate_packet_identity(static, fragment_id, saved_inputs, metadata)
        validate_result(result, saved_inputs, metadata, controls=controls)
        anchors = tuple(metadata["virtual_anchor_parent_columns"])
        prepare_fn = lambda mf_, common_, _fragment=fragment_id, _anchors=anchors: (
            prepare_finite_inputs(
                mf_, common_, static, _fragment,
                virtual_anchor_columns=_anchors,
            )
        )
        # The selected PAO anchor columns fix the local virtual gauge across
        # export and traced replay. Remaining differences are eigensolver
        # roundoff at the retained PAO rank boundary.
        term_mf_bar, term_common_bar = pullback_inputs(
            prepare_fn, (mf, common), saved_inputs, result,
            replay_atol=1e-9,
        )
        mf_bar = jax.tree_util.tree_map(_add_cotangent, mf_bar, term_mf_bar)
        common_bar = jax.tree_util.tree_map(
            _add_cotangent, common_bar, term_common_bar,
        )
        energy = energy + np.asarray(result["energy"])
        del saved_inputs, result, term_mf_bar, term_common_bar
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
