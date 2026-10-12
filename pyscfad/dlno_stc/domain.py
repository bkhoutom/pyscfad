"""STC-owned finite Boys and complete-system orbital frames."""

from typing import NamedTuple

import jax
import numpy
import scipy.linalg as scipy_linalg

from pyscfad import numpy as np, scipy
from pyscfad.lno import tools
from pyscfad.lno.lno_base import SEMICANONICAL_DEG_THRESH
from ._selection import (
    DomainSelections, _PAO_ORTH_THRESHOLD, _host_array, _hermitian_numpy,
    _orthogonalize, _build_boys_targets, _resolve_boys_options,
)


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

    _require_concrete_system_selection(mf)
    occupied = mf.mo_coeff[:, active_occ_indices]
    if not occupied.shape[1]:
        return None
    options = _resolve_boys_options(lo_kwargs)
    # Match the Boys-domain benchmark: explicit atomic initialization, the
    # standard derived gradient tolerance, and its localization response solve.
    options = {"init_guess": "atomic", "conv_tol": 1e-10,
               "gmres_options": {"restart": 120, "maxiter": 100}, **options}
    localized = _build_boys_targets(mf.mol, occupied, options)
    return SystemBoysReference(
        numpy.array(_host_array(localized), copy=True),
        numpy.array(_host_array(mf.mol.atom_coords()), copy=True),
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
    cross = _host_array(virtual.T.conj() @ overlap)
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
    co = (_rebuild_boys_targets(mf, occupied, boys_reference)
          if occupied.shape[1] else occupied)
    if virtual.shape[1]:
        overlap = mf.mol.intor_symmetric("int1e_ovlp")
        anchors = virtual @ (virtual.T.conj() @ overlap[:, columns])
        cv = _cholesky_metric_orthonormalize(anchors, overlap)
    else:
        cv = virtual
    return SystemLocalFrame(co, cv, boys_reference, columns)


def _fragment_support(static, fragment_index):
    """Use the saved fragment, AO, and atom ordering for either packet frame."""
    fragment_index = int(fragment_index)
    fragment = static.fragments[fragment_index]
    return (
        fragment_index, fragment,
        numpy.asarray(fragment.extended_ao_indices),
        numpy.asarray(fragment.extended_atoms),
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
        gram, deg_thresh=max(_PAO_ORTH_THRESHOLD, 1e-9)
    )
    pao_orth = _cholesky_metric_orthonormalize(
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
            deg_thresh=max(_PAO_ORTH_THRESHOLD, 1e-9),
        )
        candidate = pao_orth @ overlap_vectors[:, selection.overlap_keep]
        candidate = candidate[:, selection.completeness_keep]

    overlap21 = overlap[ao]
    overlap22 = overlap[numpy.ix_(ao, ao)]
    candidate = tools.project_mo(candidate, overlap21, overlap22)
    candidate = _orthogonalize(
        occupied_local, candidate, overlap22
    )
    if boys_metric_threshold is not None:
        _validate_boys_metric_rank(
            candidate, overlap22, selection.metric_keep,
            boys_metric_threshold, "virtual ED",
        )
    return _fixed_metric_orthonormalize(
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
        cross = _host_array(jax.lax.stop_gradient(cross))
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
    return _cholesky_metric_orthonormalize(
        projected, overlap22
    ), columns


# Boys replay and metric operations copied from dlno/dlno_base.py.
def _rebuild_boys_targets(mf, occupied, static):
    """Replay a saved Boys branch with the existing implicit-response wrapper.

    Only the initializer and permutation/sign choices are concrete. Current
    orbitals and their response always come from ``boys.boys``. Like the
    production SCF/localization driver this boundary is eager, not JITted.
    """
    from pyscf import gto as pyscf_gto
    from scipy.optimize import linear_sum_assignment

    reference = static.target_reference_coeff
    if reference is None:
        raise ValueError("Boys static selections require a reference orbital frame")
    reference = numpy.asarray(reference)
    if reference.shape != occupied.shape:
        raise ValueError("Boys reference and current active occupied spaces differ")
    coords = _host_array(jax.lax.stop_gradient(mf.mol.atom_coords()))
    current_mol = mf.mol.to_pyscf().set_geom_(coords, unit="Bohr", inplace=False)
    reference_mol = current_mol.set_geom_(
        numpy.asarray(static.target_reference_coords), unit="Bohr", inplace=False
    )
    cross_overlap = pyscf_gto.intor_cross("int1e_ovlp", current_mol, reference_mol)
    current_occupied = _host_array(jax.lax.stop_gradient(occupied))
    projection = current_occupied.T @ cross_overlap @ reference
    left, singular, right = scipy_linalg.svd(projection, check_finite=True)
    if singular.min() < 0.5:
        raise ValueError("Boys reference no longer spans the current occupied space; rebuild selections")
    options = dict(static.lo_kwargs or {})
    # Rebase the optimizer at the reference-aligned occupied frame. Solving
    # for the small remaining rotation avoids nearly singular exponential
    # coordinates for a large canonical-to-local rotation. The concrete
    # alignment is a coordinate choice; current occupied coefficients and
    # their localization response remain differentiated.
    options["init_guess"] = numpy.eye(occupied.shape[1])
    localized = _build_boys_targets(mf.mol, occupied @ (left @ right), options)
    current_localized = _host_array(jax.lax.stop_gradient(localized))
    overlaps = reference.T @ cross_overlap.T @ current_localized
    magnitude = numpy.abs(overlaps)
    rows, permutation = linear_sum_assignment(-magnitude)
    matched = magnitude[rows, permutation]
    alternatives = magnitude.copy()
    alternatives[rows, permutation] = -numpy.inf
    if (numpy.any(matched < 0.5)
            or numpy.any(matched - alternatives.max(axis=1) < 1e-6)):
        raise ValueError("ambiguous Boys orbital match; rebuild selections at this geometry")
    signs = numpy.where(overlaps[rows, permutation] < 0.0, -1.0, 1.0)
    return localized[:, permutation] * signs[None, :]


def _cholesky_metric_orthonormalize(coeff, overlap):
    """Return a smooth metric-orthonormal frame for a full-rank column set."""
    metric = coeff.T.conj() @ overlap @ coeff
    metric = 0.5 * (metric + metric.T.conj())
    metric_cholesky = np.linalg.cholesky(metric)
    return np.linalg.solve(
        metric_cholesky, coeff.T.conj()
    ).T.conj()


def _fixed_metric_orthonormalize(coeff, overlap, keep):
    coeff = np.asarray(coeff)
    keep = numpy.asarray(keep, dtype=numpy.int32)
    if coeff.shape[1] == 0 or keep.size == 0:
        return np.zeros((coeff.shape[0], 0), dtype=coeff.dtype)
    if (
        keep.size == coeff.shape[1]
        and numpy.array_equal(keep, numpy.arange(keep.size))
    ):
        # For full rank, Cholesky gives Q = C L^{-H}, G = L L^H.
        # It keeps the full metric derivative without a degenerate-eigenvector
        # gauge or the regularized eigh JVP's dropped rotations.
        return _cholesky_metric_orthonormalize(coeff, overlap)
    metric = coeff.T.conj() @ overlap @ coeff
    metric = 0.5 * (metric + metric.T.conj())
    eigenvalue, eigenvector = scipy.linalg.eigh(
        metric, deg_thresh=1e-9
    )
    del eigenvalue
    retained = coeff @ eigenvector[:, keep]
    # Eigh is needed only for the cross-boundary response of the retained
    # range.  Cholesky normalization supplies the complete Frechet response
    # within that range and is primal-equivalent to division by sqrt(lambda).
    return _cholesky_metric_orthonormalize(
        retained, overlap
    )


def _fixed_semicanonicalize(coeff, fock):
    coeff = np.asarray(coeff)
    if coeff.shape[1] == 0:
        return np.zeros((0,), dtype=np.real(coeff).dtype), coeff
    projected = coeff.T.conj() @ fock @ coeff
    projected = 0.5 * (projected + projected.T.conj())
    if coeff.shape[1] == 1:
        return np.real(projected).reshape(-1), coeff
    energy, rotation = scipy.linalg.eigh(
        projected, deg_thresh=SEMICANONICAL_DEG_THRESH
    )
    return np.real(energy), coeff @ rotation


def _validate_boys_metric_rank(coeff, overlap, keep, threshold, label):
    """Reject a changed domain rank at the eager fixed-selection boundary."""
    metric = coeff.T.conj() @ overlap @ coeff
    metric = _host_array(jax.lax.stop_gradient(metric))
    if not numpy.isfinite(metric).all():
        raise RuntimeError(f"Boys {label} metric rank is unusable; rebuild domains")
    values = scipy_linalg.eigvalsh(_hermitian_numpy(metric), check_finite=False)
    current_keep = numpy.flatnonzero(values > threshold)
    if not numpy.array_equal(current_keep, numpy.asarray(keep)):
        raise RuntimeError(f"Boys {label} metric rank changed; rebuild domains")


def _rebuild_selected_pao_space(
    common,
    selection,
    representation_ao_indices,
    occupied_for_projection,
    *,
    boys_metric_threshold=None,
):
    """Replay continuous PAO transformations with all labels held fixed."""

    s1e = common.s1e
    pao_parent = common.pao_coeff[:, selection.parent_columns]
    gram = pao_parent.T.conj() @ s1e @ pao_parent
    gram = 0.5 * (gram + gram.T.conj())
    gram_e, gram_v = scipy.linalg.eigh(
        gram, deg_thresh=max(_PAO_ORTH_THRESHOLD, 1e-9)
    )
    canonical_keep = selection.canonical_keep
    del gram_e
    pao_orth = _cholesky_metric_orthonormalize(
        pao_parent @ gram_v[:, canonical_keep], s1e
    )

    support_ao = selection.support_ao_indices
    tmp = s1e[support_ao] @ pao_orth
    support_metric = s1e[numpy.ix_(support_ao, support_ao)]
    projected_overlap = tmp.T.conj() @ np.linalg.solve(
        support_metric, tmp
    )
    projected_overlap = 0.5 * (
        projected_overlap + projected_overlap.T.conj()
    )
    _, overlap_v = scipy.linalg.eigh(
        projected_overlap, deg_thresh=max(_PAO_ORTH_THRESHOLD, 1e-9)
    )
    candidate = pao_orth @ overlap_v[:, selection.overlap_keep]
    candidate = candidate[:, selection.completeness_keep]

    representation_ao = numpy.asarray(
        representation_ao_indices, dtype=numpy.int32
    )
    s21 = s1e[representation_ao]
    s22 = s1e[numpy.ix_(representation_ao, representation_ao)]
    candidate = tools.project_mo(candidate, s21, s22)
    candidate = _orthogonalize(
        occupied_for_projection, candidate, s22
    )
    if boys_metric_threshold is not None:
        _validate_boys_metric_rank(
            candidate, s22, selection.metric_keep,
            boys_metric_threshold, "virtual ED",
        )
    return _fixed_metric_orthonormalize(
        candidate, s22, selection.metric_keep
    )


class TargetOccupiedData(NamedTuple):
    """Current singleton Boys coefficients; labels are held in selections."""

    iao_coeff: object


class DomainData(NamedTuple):
    """Common differentiable Boys/PAO arrays, without occupied weights."""

    s1e: object
    fock: object
    occupied_coeff: object
    virtual_coeff: object
    occupied_energy: object
    virtual_energy: object
    iao_coeff: object
    pao_coeff: object
    fragment_occupied_data: tuple[TargetOccupiedData, ...]


def rebuild_domain_data(mf, static):
    """Rebuild current Boys and PAOs with the saved concrete choices.

    Copied from the Boys branch of dlno_base.rebuild_domain_data, omitting
    raw occupied overlaps and all additive orbital projection weights.
    """
    if not isinstance(static, DomainSelections) or static.lo_type != "boys":
        raise TypeError("static must be STC Boys DomainSelections")
    overlap = mf.mol.intor_symmetric("int1e_ovlp")
    mo_coeff, mo_energy = np.asarray(mf.mo_coeff), np.asarray(mf.mo_energy)
    occupied, virtual = (mo_coeff[:, indices] for indices in (static.active_occ_indices, static.active_vir_indices))
    targets = (_rebuild_boys_targets(mf, occupied, static)
               if occupied.shape[1] else occupied)
    projected_out = mo_coeff[:, static.pao_projected_out_indices]
    raw_pao = np.eye(mf.mol.nao, dtype=mo_coeff.dtype) - projected_out @ (projected_out.T.conj() @ overlap)
    raw_pao = raw_pao[:, static.pao_parent_ao_indices]
    norm2 = np.real(np.einsum("ui,uv,vi->i", raw_pao.conj(), overlap, raw_pao, optimize=True))
    return DomainData(overlap, mf.get_fock(), occupied, virtual,
                      mo_energy[static.active_occ_indices], mo_energy[static.active_vir_indices],
                      targets, raw_pao / np.sqrt(norm2)[None, :],
                      tuple(TargetOccupiedData(targets[:, indices]) for indices in static.frag_lolist))


class BoysEDFrame(NamedTuple):
    """Ordered metric-Cholesky ED; target_index labels its local column."""

    occupied_coeff: object
    virtual_coeff: object
    extended_ao_indices: numpy.ndarray
    extended_atoms: numpy.ndarray
    target_index: int
    virtual_anchor_columns: tuple[int, ...]


def build_boys_ed_frame(common, static, fragment_index, *, virtual_anchor_columns=None):
    """Replay a finite Boys ED with full ordered occupied label retention.

    The target slot is the position of its fragment in strong_fragments. It
    labels an orthogonalized column, not the original projected Boys orbital.
    Saved reference orbitals, supports and anchors do not freeze response.
    """
    if not isinstance(static, DomainSelections) or static.lo_type != "boys":
        raise TypeError("static must be STC Boys DomainSelections")
    fragment_index, fragment, ao, atoms = _fragment_support(static, fragment_index)
    partners = numpy.asarray(fragment.strong_fragments)
    if (partners.ndim != 1 or partners.dtype.kind not in "iu" or len(set(partners.tolist())) != partners.size
            or numpy.any(partners < 0) or numpy.any(partners >= len(static.fragments))):
        raise ValueError(f"Boys domain {fragment_index} has invalid ordered strong partners")
    target_positions = numpy.flatnonzero(partners == fragment_index)
    if (target_positions.size != 1 or fragment.target_index != int(target_positions[0])):
        raise ValueError(f"Boys domain {fragment_index} must retain its unique local target label")
    nocc = partners.size
    if not numpy.array_equal(fragment.strong_occ_metric_keep, numpy.arange(nocc)):
        raise RuntimeError(f"Boys domain {fragment_index} loses occupied target/partner labels; rebuild domains with tighter support")
    overlap22 = common.s1e[numpy.ix_(ao, ao)]
    candidate = np.concatenate([common.fragment_occupied_data[int(partner)].iao_coeff for partner in partners], axis=1)
    if candidate.shape[1] != nocc:
        raise ValueError("Boys targets require a complete singleton partition")
    occupied = tools.project_mo(candidate, common.s1e[ao], overlap22)
    _validate_boys_metric_rank(occupied, overlap22, fragment.strong_occ_metric_keep,
                               static.thresholds.metric_rank, f"domain {fragment_index} occupied target/partner ED")
    occupied = _cholesky_metric_orthonormalize(occupied, overlap22)
    occupied = occupied * _positive_pivot_phases(occupied)
    virtual = _rebuild_local_virtual_subspace(common, fragment.strong_virtual, ao, occupied,
                                              boys_metric_threshold=static.thresholds.metric_rank)
    if virtual_anchor_columns is None:
        virtual_anchor_columns = fragment.virtual_anchor_columns
    if virtual_anchor_columns is None and any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(common)):
        raise ValueError("select virtual anchors from concrete data before tracing preparation")
    virtual, columns = _stable_virtual_frame(common, fragment, ao, overlap22, virtual, virtual_anchor_columns)
    virtual = virtual * _positive_pivot_phases(virtual)
    return BoysEDFrame(occupied, virtual, ao, atoms, fragment.target_index, columns)
