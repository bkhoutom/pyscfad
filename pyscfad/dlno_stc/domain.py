"""Full-support and finite DLNO orbital frames for STC tensor packets."""

from typing import NamedTuple

import jax
import numpy
import scipy.linalg as scipy_linalg

from pyscfad import numpy as np, scipy
from pyscfad.dlno import dlno_base, tools


class SystemBoysReference(NamedTuple):
    """Fixed initializer and orbital labels for active-occupied Boys replay."""

    target_reference_coeff: object
    target_reference_coords: object
    lo_kwargs: dict | None = None


class SystemLocalFrame(NamedTuple):
    """Complete active spaces, with discrete Boys and AO-anchor metadata."""

    occupied_coeff: object
    virtual_coeff: object
    boys_reference: SystemBoysReference | None
    virtual_anchor_columns: tuple[int, ...]


def _require_concrete_system_selection(mf):
    if any(isinstance(leaf, jax.core.Tracer)
           for leaf in jax.tree_util.tree_leaves((mf.mol, mf.mo_coeff))):
        raise ValueError(
            "select the system Boys reference and virtual anchor columns "
            "from concrete data before tracing preparation"
        )


def select_system_boys_reference(mf, active_occ_indices, *, lo_kwargs=None):
    """Select active-occupied Boys labels using the shared domain setup."""
    from pyscfad.dlno.targets import build_targets, resolve_target_options

    _require_concrete_system_selection(mf)
    occupied = mf.mo_coeff[:, active_occ_indices]
    if not occupied.shape[1]:
        return None
    _, options = resolve_target_options("boys", lo_kwargs)
    # Match the Boys-domain benchmark: explicit atomic initialization, the
    # standard derived gradient tolerance, and its localization response solve.
    options = {"init_guess": "atomic", "conv_tol": 1e-10,
               "gmres_options": {"restart": 120, "maxiter": 100}, **options}
    localized = build_targets(mf.mol, occupied, lo_type="boys", lo_kwargs=options)
    return SystemBoysReference(
        numpy.array(dlno_base._host_array(localized), copy=True),
        numpy.array(dlno_base._host_array(mf.mol.atom_coords()), copy=True),
        options,
    )


def select_system_virtual_anchor_columns(mf, active_vir_indices):
    """Select exactly nvir independent projected AO columns, without domains."""
    _require_concrete_system_selection(mf)
    virtual = mf.mo_coeff[:, active_vir_indices]
    nvir = virtual.shape[1]
    if not nvir:
        return ()
    overlap = mf.mol.intor_symmetric("int1e_ovlp")
    cross = dlno_base._host_array(virtual.T.conj() @ overlap)
    _, triangular, pivots = scipy_linalg.qr(
        cross, pivoting=True, mode="economic", check_finite=True,
    )
    diagonal = numpy.abs(numpy.diag(triangular))
    if diagonal.size < nvir or numpy.min(diagonal[:nvir]) < 1e-10:
        raise ValueError("system projected AO anchors do not span the active virtual space")
    return tuple(int(column) for column in pivots[:nvir])


def build_system_local_frame(
    mf, active_occ_indices, active_vir_indices, *, boys_reference=None,
    virtual_anchor_columns=None,
):
    """Rebuild one complete Boys/PAO frame with only discrete choices fixed.

    Frozen occupied and virtual columns must already be excluded by the
    supplied active indices. Concrete calls may select their reference and
    anchors; a preparation VJP must receive both selections from its caller.
    Current Boys coefficients, AO projections, overlap and Cholesky remain
    differentiable. No ED support selection or virtual rank truncation occurs.
    """
    occupied = mf.mo_coeff[:, active_occ_indices]
    virtual = mf.mo_coeff[:, active_vir_indices]
    if boys_reference is None and occupied.shape[1]:
        boys_reference = select_system_boys_reference(mf, active_occ_indices)
    if virtual_anchor_columns is None:
        virtual_anchor_columns = select_system_virtual_anchor_columns(mf, active_vir_indices)
    columns = _validate_virtual_anchor_columns(
        virtual_anchor_columns, numpy.arange(mf.mol.nao), virtual.shape[1],
    )
    co = (dlno_base._rebuild_boys_targets(mf, occupied, boys_reference)
          if occupied.shape[1] else occupied)
    if virtual.shape[1]:
        overlap = mf.mol.intor_symmetric("int1e_ovlp")
        anchors = virtual @ (virtual.T.conj() @ overlap[:, columns])
        cv = dlno_base._cholesky_metric_orthonormalize(anchors, overlap)
    else:
        cv = virtual
    return SystemLocalFrame(co, cv, boys_reference, columns)


class STCDomain(NamedTuple):
    """One complete active orbital frame; target_index selects its Boys column."""

    occupied_coeff: object
    virtual_coeff: object
    extended_ao_indices: numpy.ndarray
    extended_atoms: numpy.ndarray
    target_index: int


def _fragment_support(static, fragment_index):
    """Use the saved fragment, AO, and atom ordering for either packet frame."""
    fragment_index = int(fragment_index)
    fragment = static.fragments[fragment_index]
    return (
        fragment_index, fragment,
        numpy.asarray(fragment.extended_ao_indices),
        numpy.asarray(fragment.extended_atoms),
    )



def build_stc_domain(mf, common, static, fragment_index):
    """Return the supported complete-space frame without occupied mixing.

    The existing full-target MP2 contract identifies one Boys target by a
    fixed local column. This frame preserves that convention. A finite ED
    stores the target as a projection across its local occupied frame.
    """
    if static.lo_type != "boys":
        raise ValueError("STC target MP2 currently requires Boys targets")
    if static.frozen is None:
        raise ValueError("STC target MP2 requires an explicit frozen-space setting")
    if len(static.fragments) != len(static.active_occ_indices):
        raise ValueError("Boys targets must form a complete singleton partition")
    labels = [int(group[0]) for group in static.frag_lolist if len(group) == 1]
    if sorted(labels) != list(range(len(static.active_occ_indices))):
        raise ValueError("Boys targets must form a complete singleton partition")
    fragment_index, fragment, ao, atoms = _fragment_support(
        static, fragment_index
    )
    natm = mf.mol.natm
    nao = mf.mol.nao
    all_atoms = numpy.arange(natm)
    all_ao = numpy.arange(nao)
    if (not numpy.array_equal(atoms, all_atoms)
            or not numpy.array_equal(ao, all_ao)):
        raise ValueError("STC target MP2 requires full extended atom/AO support")
    if not numpy.array_equal(numpy.asarray(fragment.pao_center_atoms), all_atoms):
        raise ValueError("STC target MP2 requires full PAO-center atom support")
    if (not numpy.asarray(static.strong_mask).all()
            or not numpy.array_equal(numpy.asarray(fragment.strong_fragments),
                                     numpy.arange(len(static.fragments)))):
        raise ValueError("STC target MP2 requires all fragment pairs strong")
    if len(fragment.strong_occ_metric_keep) != len(static.active_occ_indices):
        raise ValueError("STC target MP2 requires all active occupied columns")
    if len(fragment.strong_virtual.metric_keep) != len(static.active_vir_indices):
        raise ValueError("STC target MP2 requires the full active virtual space")
    if (common.iao_coeff.shape != (nao, len(static.active_occ_indices))
            or common.virtual_coeff.shape != (nao, len(static.active_vir_indices))):
        raise ValueError("current orbital dimensions differ from fixed selections")
    if (common.iao_coeff.dtype.kind != "f" or common.iao_coeff.dtype.itemsize != 8
            or common.virtual_coeff.dtype.kind != "f"
            or common.virtual_coeff.dtype.itemsize != 8):
        raise ValueError("STC target MP2 requires real float64 orbitals")
    return STCDomain(
        common.iao_coeff, common.virtual_coeff,
        ao, atoms,
        labels[fragment_index],
    )


def _positive_pivot_phases(coeff):
    """Choose a reproducible real phase for each local orbital column."""
    pivot_rows = np.argmax(np.abs(coeff), axis=0)
    pivot_values = coeff[pivot_rows, np.arange(coeff.shape[1])]
    return np.where(pivot_values < 0.0, -1.0, 1.0)


def _all_columns(indices, count):
    return numpy.array_equal(
        numpy.asarray(indices), numpy.arange(count, dtype=numpy.int32)
    )


def _rebuild_local_virtual_subspace(
    common, selection, ao, occupied_local, *, boys_metric_threshold=None,
):
    """Retain the DLNO PAO subspace without rotating fully retained modes.

    The canonical PAO rank selection is unchanged. A support-overlap
    eigensolver is needed only when its output columns are subsequently
    truncated; when every column survives, that rotation adds no subspace
    information and has an unstable gauge near an identity overlap.
    """
    overlap = common.s1e
    parent = common.pao_coeff[:, selection.parent_columns]
    gram = parent.T.conj() @ overlap @ parent
    gram = 0.5 * (gram + gram.T.conj())
    _, canonical_vectors = scipy.linalg.eigh(
        gram, deg_thresh=max(dlno_base._PAO_ORTH_THRESHOLD, 1e-9)
    )
    pao_orth = dlno_base._cholesky_metric_orthonormalize(
        parent @ canonical_vectors[:, selection.canonical_keep], overlap
    )
    if (_all_columns(selection.overlap_keep, pao_orth.shape[1])
            and _all_columns(selection.completeness_keep, pao_orth.shape[1])):
        candidate = pao_orth
    else:
        support_ao = numpy.asarray(selection.support_ao_indices)
        tmp = overlap[support_ao] @ pao_orth
        support_metric = overlap[numpy.ix_(support_ao, support_ao)]
        projected_overlap = tmp.T.conj() @ np.linalg.solve(
            support_metric, tmp
        )
        projected_overlap = 0.5 * (
            projected_overlap + projected_overlap.T.conj()
        )
        _, overlap_vectors = scipy.linalg.eigh(
            projected_overlap,
            deg_thresh=max(dlno_base._PAO_ORTH_THRESHOLD, 1e-9),
        )
        candidate = pao_orth @ overlap_vectors[:, selection.overlap_keep]
        candidate = candidate[:, selection.completeness_keep]

    overlap21 = overlap[ao]
    overlap22 = overlap[numpy.ix_(ao, ao)]
    candidate = tools.project_mo(candidate, overlap21, overlap22)
    candidate = tools.orthogonalize(
        occupied_local, candidate, overlap22
    )
    if boys_metric_threshold is not None:
        dlno_base._validate_boys_metric_rank(
            candidate, overlap22, selection.metric_keep,
            boys_metric_threshold, "virtual ED",
        )
    return dlno_base._fixed_metric_orthonormalize(
        candidate, overlap22, selection.metric_keep
    )


def _validate_virtual_anchor_columns(columns, parent, nvir):
    try:
        columns = tuple(columns)
    except TypeError as err:
        raise ValueError("virtual anchor columns must be a sequence") from err
    allowed = set(int(index) for index in parent)
    if (len(columns) != nvir or len(set(columns)) != nvir
            or any(type(index) not in (int, numpy.int32, numpy.int64)
                   or int(index) not in allowed for index in columns)):
        raise ValueError(
            "virtual anchor columns must be distinct selected PAO parents "
            "with one column per retained virtual orbital"
        )
    return tuple(int(index) for index in columns)


def _stable_virtual_frame(
    common, fragment, ao, overlap22, virtual_subspace,
    virtual_anchor_columns=None,
):
    """Fix the retained PAO gauge using selected, well-conditioned parents."""
    parent = numpy.asarray(fragment.strong_virtual.parent_columns)
    parent_local = tools.project_mo(
        common.pao_coeff[:, parent], common.s1e[ao], overlap22
    )
    nvir = virtual_subspace.shape[1]
    if virtual_anchor_columns is None:
        cross = virtual_subspace.T.conj() @ overlap22 @ parent_local
        cross = dlno_base._host_array(jax.lax.stop_gradient(cross))
        _, triangular, pivots = scipy_linalg.qr(
            cross, pivoting=True, mode="economic", check_finite=True
        )
        if nvir and numpy.min(numpy.abs(numpy.diag(triangular)[:nvir])) < 1e-10:
            raise RuntimeError("selected PAO parent anchors lost rank")
        columns = tuple(int(parent[index]) for index in pivots[:nvir])
    else:
        columns = _validate_virtual_anchor_columns(
            virtual_anchor_columns, parent, nvir
        )
    position = {int(column): index for index, column in enumerate(parent)}
    anchors = parent_local[:, [position[column] for column in columns]]
    projected = virtual_subspace @ (
        virtual_subspace.T.conj() @ overlap22 @ anchors
    )
    return dlno_base._cholesky_metric_orthonormalize(
        projected, overlap22
    ), columns


class LocalStrongDomain(NamedTuple):
    """Finite orthonormal ED frame before occupied/virtual Fock rotations."""

    occupied_coeff: object
    virtual_coeff: object
    target_projection: object
    target_weight: object
    partner_weight: object


def _build_local_strong_ed_domain_and_anchors(
    common, static, fragment_index, virtual_anchor_columns=None,
):
    """Rebuild one finite ED in its pre-semicanonical local orbital frame.

    Fixed AO, partner, PAO, and rank selections are reused from ``static``.
    The continuous projection and metric orthonormalization remain traced.
    This deliberately returns no orbital energies: the projected Fock blocks
    are generally non-diagonal and their diagonals are not MP2 denominators.
    """
    fragment_index, fragment, ao, _ = _fragment_support(
        static, fragment_index
    )
    overlap = common.s1e
    overlap22 = overlap[numpy.ix_(ao, ao)]
    overlap21 = overlap[ao]

    if static.lo_type == "boys":
        occupied_candidate = np.concatenate([
            common.fragment_occupied_data[int(partner)].iao_coeff
            for partner in fragment.strong_fragments
        ], axis=1)
    else:
        partner_weight = sum(
            common.fragment_occupied_data[int(partner)].occupied_weight
            for partner in fragment.strong_fragments
        )
        partner_weight = 0.5 * (partner_weight + partner_weight.T.conj())
        _, partner_vector = scipy.linalg.eigh(
            partner_weight, deg_thresh=1e-9,
        )
        occupied_candidate = common.occupied_coeff @ (
            partner_vector[:, fragment.strong_occ_union_keep]
        )
    occupied_local = tools.project_mo(
        occupied_candidate, overlap21, overlap22
    )
    if static.lo_type == "boys":
        dlno_base._validate_boys_metric_rank(
            occupied_local, overlap22, fragment.strong_occ_metric_keep,
            static.thresholds.metric_rank, "occupied target/partner ED",
        )
    occupied_local = dlno_base._fixed_metric_orthonormalize(
        occupied_local, overlap22, fragment.strong_occ_metric_keep
    )
    occupied_local = occupied_local * _positive_pivot_phases(occupied_local)

    virtual_local = _rebuild_local_virtual_subspace(
        common, fragment.strong_virtual, ao, occupied_local,
        boys_metric_threshold=(
            static.thresholds.metric_rank if static.lo_type == "boys" else None
        ),
    )
    virtual_local, anchors = _stable_virtual_frame(
        common, fragment, ao, overlap22, virtual_local,
        virtual_anchor_columns,
    )
    virtual_local = virtual_local * _positive_pivot_phases(virtual_local)

    target_coeff = common.fragment_occupied_data[
        fragment_index
    ].iao_coeff
    partner_coeff = np.concatenate([
        common.fragment_occupied_data[int(partner)].iao_coeff
        for partner in fragment.strong_fragments
    ], axis=1)
    overlap_to_local = overlap[:, ao] @ occupied_local
    target_projection = target_coeff.T.conj() @ overlap_to_local
    partner_projection = partner_coeff.T.conj() @ overlap_to_local
    target_weight = target_projection.T.conj() @ target_projection
    partner_weight = partner_projection.T.conj() @ partner_projection
    if static.lo_type == "boys":
        retained_target = dlno_base._host_array(
            jax.lax.stop_gradient(target_weight)
        )
        if (not numpy.isfinite(retained_target).all()
                or numpy.trace(retained_target).real
                <= static.thresholds.metric_rank):
            raise RuntimeError(
                f"Boys target {fragment_index} is lost in its retained ED; "
                "rebuild domains"
            )
    domain = LocalStrongDomain(
        occupied_coeff=occupied_local,
        virtual_coeff=virtual_local,
        target_projection=target_projection,
        target_weight=target_weight,
        partner_weight=partner_weight,
    )
    return domain, anchors


def build_local_strong_ed_domain(
    common, static, fragment_index, *, virtual_anchor_columns=None,
):
    """Return a finite ED with its virtual frame fixed by selected PAOs."""
    domain, _ = _build_local_strong_ed_domain_and_anchors(
        common, static, fragment_index, virtual_anchor_columns
    )
    return domain


def select_virtual_anchor_columns(common, static, fragment_index):
    """Select a fixed, well-conditioned PAO parent gauge for one ED."""
    _, columns = _build_local_strong_ed_domain_and_anchors(
        common, static, fragment_index
    )
    return columns
