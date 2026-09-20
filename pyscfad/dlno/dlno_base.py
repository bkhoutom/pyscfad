"""Current target, PAO, and strong/weak domain orbitals for DLNO.

Selection labels belong to _selection; global-HF local interacting spaces
belong to lis. These routines rebuild continuous arrays with those labels fixed.
"""
from __future__ import annotations

from typing import NamedTuple
import jax
import numpy
import scipy.linalg as scipy_linalg
from pyscfad import numpy as np, scipy
from pyscfad.lno import lno_base
from . import tools
from ._selection import (
    _PAO_ORTH_THRESHOLD, _WEIGHT_DEGENERACY_TOLERANCE,
    DomainSelections, _host_array, _hermitian_numpy,
)

class TargetOccupiedData(NamedTuple):
    """Current occupied projection and weight of one target block."""

    iao_coeff: object
    iao_occ_overlap: object
    occupied_projection: object
    occupied_weight: object


class DomainData(NamedTuple):
    """Common differentiable arrays rebuilt once for all fragments."""

    s1e: object
    fock: object
    occupied_coeff: object
    virtual_coeff: object
    occupied_energy: object
    virtual_energy: object
    iao_coeff: object
    pao_coeff: object
    fragment_occupied_data: tuple[TargetOccupiedData, ...]


class StrongDomain(NamedTuple):
    """Differentiable semicanonical orbitals and weights in one strong ED."""

    occupied_coeff: object
    virtual_coeff: object
    occupied_energy: object
    virtual_energy: object
    target_projection: object
    target_weight: object
    partner_weight: object


class WeakScreen(NamedTuple):
    """Differentiable weighted occupied modes and PAOs for one weak screen."""

    weights: object
    occupied_energy: object
    occupied_coeff: object
    virtual_energy: object
    virtual_coeff: object


def _rebuild_boys_targets(mf, occupied, static):
    """Replay a saved Boys branch with the existing implicit-response wrapper.

    Only the initializer and permutation/sign choices are concrete. Current
    orbitals and their response always come from ``boys.boys``. Like the
    production SCF/localization driver this boundary is eager, not JITted.
    """
    from pyscf import gto as pyscf_gto
    from scipy.optimize import linear_sum_assignment
    from .targets import build_targets

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
    # Polar alignment is an initial guess only; converged LMOs are never
    # mixed by a post-localization Procrustes rotation.
    options["init_guess"] = left @ right
    localized = build_targets(mf.mol, occupied, lo_type="boys", lo_kwargs=options)
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


def rebuild_domain_data(mf, static, *, iao_coeff=None):
    """Rebuild all common continuous IAO-MP2 arrays from ``mf``.

    ``static`` supplies fixed selections and optional Boys label references.
    With ``iao_coeff=None``, current IAO or Boys targets are rebuilt from the
    active occupied orbitals and carry AO-integral, SCF and localization response.
    A supplied ``iao_coeff`` remains differentiable if it is a traced array,
    but the caller is responsible for constructing it at the current geometry.
    """

    if not isinstance(static, DomainSelections):
        raise TypeError("static must be DomainSelections")
    mol = mf.mol
    s1e = mol.intor_symmetric("int1e_ovlp")
    fock = mf.get_fock()
    mo_coeff = np.asarray(mf.mo_coeff)
    mo_energy = np.asarray(mf.mo_energy)
    occupied = mo_coeff[:, static.active_occ_indices]
    virtual = mo_coeff[:, static.active_vir_indices]
    occupied_energy = mo_energy[static.active_occ_indices]
    virtual_energy = mo_energy[static.active_vir_indices]

    if iao_coeff is None:
        if static.lo_type == "boys":
            iao_coeff = _rebuild_boys_targets(mf, occupied, static)
        else:
            from .targets import build_targets
            iao_coeff = build_targets(
                mol, occupied, lo_type=static.lo_type, lo_kwargs=static.lo_kwargs
            )
    else:
        iao_coeff = np.asarray(iao_coeff)
    if iao_coeff.ndim != 2:
        raise ValueError("iao_coeff must be a rank-2 array")

    fragment_data = []
    for indices in static.frag_lolist:
        fragment_iao = iao_coeff[:, indices]
        overlap_occ = fragment_iao.T.conj() @ s1e @ occupied
        occupied_projection = occupied @ overlap_occ.T.conj()
        occupied_weight = overlap_occ.T.conj() @ overlap_occ
        fragment_data.append(TargetOccupiedData(
            iao_coeff=fragment_iao,
            iao_occ_overlap=overlap_occ,
            occupied_projection=occupied_projection,
            occupied_weight=occupied_weight,
        ))

    projected_out = mo_coeff[:, static.pao_projected_out_indices]
    raw_pao = np.eye(mol.nao, dtype=mo_coeff.dtype) - projected_out @ (
        projected_out.T.conj() @ s1e
    )
    raw_pao = raw_pao[:, static.pao_parent_ao_indices]
    norm2 = np.real(np.einsum(
        "ui,uv,vi->i", raw_pao.conj(), s1e, raw_pao, optimize=True
    ))
    pao_coeff = raw_pao / np.sqrt(norm2)[None, :]

    return DomainData(
        s1e=s1e,
        fock=fock,
        occupied_coeff=occupied,
        virtual_coeff=virtual,
        occupied_energy=occupied_energy,
        virtual_energy=virtual_energy,
        iao_coeff=iao_coeff,
        pao_coeff=pao_coeff,
        fragment_occupied_data=tuple(fragment_data),
    )


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
        projected, deg_thresh=lno_base.SEMICANONICAL_DEG_THRESH
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
    candidate = tools.orthogonalize(
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


def build_strong_ed_domain(common, static, fragment_index):
    """Rebuild one strong-pair ED with fixed atom lists and retained ranks."""

    fragment = static.fragments[int(fragment_index)]
    s1e = common.s1e
    fock = common.fock
    ao_indices = fragment.extended_ao_indices
    s21 = s1e[ao_indices]
    s22 = s1e[numpy.ix_(ao_indices, ao_indices)]
    fock22 = fock[numpy.ix_(ao_indices, ao_indices)]

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
            partner_weight,
            deg_thresh=1e-9,
        )
        occupied_candidate = common.occupied_coeff @ (
            partner_vector[:, fragment.strong_occ_union_keep]
        )
    occupied_local = tools.project_mo(occupied_candidate, s21, s22)
    if static.lo_type == "boys":
        _validate_boys_metric_rank(
            occupied_local, s22, fragment.strong_occ_metric_keep,
            static.thresholds.metric_rank, "occupied target/partner ED",
        )
    occupied_local = _fixed_metric_orthonormalize(
        occupied_local, s22, fragment.strong_occ_metric_keep
    )
    occupied_energy, occupied_local = _fixed_semicanonicalize(
        occupied_local, fock22
    )

    virtual_local = _rebuild_selected_pao_space(
        common,
        fragment.strong_virtual,
        ao_indices,
        occupied_local,
        boys_metric_threshold=(
            static.thresholds.metric_rank if static.lo_type == "boys" else None
        ),
    )
    virtual_energy, virtual_local = _fixed_semicanonicalize(
        virtual_local, fock22
    )

    target_coeff = common.fragment_occupied_data[
        int(fragment_index)
    ].iao_coeff
    partner_coeff = np.concatenate([
        common.fragment_occupied_data[int(partner)].iao_coeff
        for partner in fragment.strong_fragments
    ], axis=1)
    overlap_to_local = s1e[:, ao_indices] @ occupied_local
    target_projection = target_coeff.T.conj() @ overlap_to_local
    partner_projection = partner_coeff.T.conj() @ overlap_to_local
    target_weight = target_projection.T.conj() @ target_projection
    partner_weight = partner_projection.T.conj() @ partner_projection
    if static.lo_type == "boys":
        # Diagnose an unusable fixed domain at the eager primal boundary;
        # the physical projection and its derivatives remain untouched.
        retained_target = _host_array(jax.lax.stop_gradient(target_weight))
        if (not numpy.isfinite(retained_target).all()
                or numpy.trace(retained_target).real <= static.thresholds.metric_rank):
            raise RuntimeError(
                f"Boys target {fragment_index} is lost in its retained ED; rebuild domains"
            )

    return StrongDomain(
        occupied_coeff=occupied_local,
        virtual_coeff=virtual_local,
        occupied_energy=occupied_energy,
        virtual_energy=virtual_energy,
        target_projection=target_projection,
        target_weight=target_weight,
        partner_weight=partner_weight,
    )


def _rebuild_fragment_modes(common, fragment):
    occupied_weight = common.fragment_occupied_data[
        fragment.fragment_index
    ].occupied_weight
    hermitian_weight = 0.5 * (
        occupied_weight + occupied_weight.T.conj()
    )
    eigenvalue, eigenvector = scipy.linalg.eigh(
        hermitian_weight,
        deg_thresh=max(_WEIGHT_DEGENERACY_TOLERANCE, 1e-9),
    )
    retained = fragment.weak_weight_eigen_indices
    values = np.real(eigenvalue[retained])
    modes = eigenvector[:, retained]
    output_modes = []
    output_weights = []
    for start, stop in fragment.weak_weight_degenerate_blocks:
        block = modes[:, start:stop]
        if stop - start > 1:
            projected_fock = block.T.conj() @ (
                common.occupied_energy[:, None] * block
            )
            projected_fock = 0.5 * (
                projected_fock + projected_fock.T.conj()
            )
            _, rotation = scipy.linalg.eigh(
                projected_fock,
                deg_thresh=max(_WEIGHT_DEGENERACY_TOLERANCE, 1e-9),
            )
            block = block @ rotation
        output_modes.append(block)
        output_weights.append(
            np.ones((stop - start,), dtype=values.dtype)
            * np.mean(values[start:stop])
        )
    return np.concatenate(output_weights), np.concatenate(
        output_modes, axis=1
    )


def build_weak_multipole_screen(common, static, fragment_index):
    """Rebuild one primary-domain weak-pair multipole screen.

    Returns ``None`` when the reference construction had no usable weak
    virtual space.  Such a fragment is marked forced-strong by the reference
    topology, so it must never be used in a final weak-pair correction.
    """

    fragment = static.fragments[int(fragment_index)]
    if not fragment.has_weak_screen:
        return None

    if static.lo_type == "boys":
        occupied_global = common.fragment_occupied_data[
            fragment.fragment_index
        ].iao_coeff
        weight = np.ones((1,), dtype=occupied_global.dtype)
    else:
        weight, occupied_mode = _rebuild_fragment_modes(common, fragment)
        occupied_global = common.occupied_coeff @ occupied_mode
    ao_indices = fragment.primary_ao_indices
    s1e = common.s1e
    s21 = s1e[ao_indices]
    s22 = s1e[numpy.ix_(ao_indices, ao_indices)]
    fock22 = common.fock[numpy.ix_(ao_indices, ao_indices)]
    occupied_local = tools.project_mo(occupied_global, s21, s22)

    norm2 = np.real(np.einsum(
        "ui,uv,vi->i",
        occupied_local.conj(),
        s22,
        occupied_local,
        optimize=True,
    ))
    norm_keep = fragment.weak_occ_norm_keep
    if static.lo_type == "boys":
        current_norm = _host_array(jax.lax.stop_gradient(norm2))
        if (not numpy.isfinite(current_norm).all() or not numpy.array_equal(
                numpy.flatnonzero(current_norm > static.thresholds.metric_rank), norm_keep)):
            raise RuntimeError("Boys screening target rank changed; rebuild domains")
    occupied_local = (
        occupied_local[:, norm_keep]
        / np.sqrt(norm2[norm_keep])[None, :]
    )
    weight = weight[norm_keep]
    occupied_energy = np.real(np.einsum(
        "ui,uv,vi->i",
        occupied_local.conj(),
        fock22,
        occupied_local,
        optimize=True,
    ))

    occupied_span = _fixed_metric_orthonormalize(
        occupied_local,
        s22,
        fragment.weak_occ_span_metric_keep,
    )
    virtual_local = _rebuild_selected_pao_space(
        common,
        fragment.weak_virtual,
        ao_indices,
        occupied_span,
        boys_metric_threshold=(
            static.thresholds.metric_rank if static.lo_type == "boys" else None
        ),
    )
    virtual_energy, virtual_local = _fixed_semicanonicalize(
        virtual_local, fock22
    )
    return WeakScreen(
        weights=weight,
        occupied_energy=occupied_energy,
        occupied_coeff=occupied_local,
        virtual_energy=virtual_energy,
        virtual_coeff=virtual_local,
    )


