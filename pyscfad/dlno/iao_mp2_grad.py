"""Fixed-topology differentiable orbital rebuilds for IAO-fragment MP2.

The energy-only implementation in :mod:`pyscfad.dlno.iao_mp2` intentionally
stores both discrete domain choices and geometry-dependent arrays in one
validation container.  Nuclear derivatives must not reuse the latter.  This
module extracts every thresholded choice at a concrete reference geometry and
then reconstructs the continuous quantities from a current SCF object.

Only atom/index lists, retained ranks, and retained eigenvector labels are
frozen.  Overlaps, Fock matrices, active orbitals, IAOs, PAOs, fragment
weights, projected domain orbitals, semicanonical rotations, strong-pair MP2
energies, and weak multipole energies remain on the AD tape.  The progressive
pullback routines expose the resulting SCF cotangent for reuse by a
DLNO-CCSD(T) driver.

The current implementation targets the same real, restricted, density-fitted
references as :mod:`pyscfad.dlno.iao_mp2`.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import numpy as onp
import scipy.linalg as onp_scipy_linalg
from pyscf.mp.mp2 import _mo_splitter

from pyscfad import numpy as np
from pyscfad import scipy
from pyscfad.lno import lno_base
from pyscfad.mp import dfmp2

from . import pao as dlno_pao
from . import util
from . import mp2 as dlno_mp2
from .fragment_mp2 import fragment_pair_energy_from_lov_jax
from .iao_mp2 import IAOFragmentTopology


__all__ = [
    "FixedPAOSubspaceSelection",
    "IAOMP2FragmentStaticSelection",
    "IAOFragmentMP2StaticSelections",
    "IAOFragmentOccupiedData",
    "IAOFragmentMP2ContinuousData",
    "IAOMP2StrongDomain",
    "IAOMP2WeakScreen",
    "IAOMP2FragmentDimensions",
    "IAOMP2TermResult",
    "IAOMP2GradientTiming",
    "IAOMP2Decomposition",
    "build_iao_mp2_static_selections",
    "rebuild_iao_mp2_common",
    "build_strong_ed_domain",
    "build_weak_multipole_screen",
    "strong_domain_energy",
    "weak_screen_pair_energy",
    "strong_fragment_energy",
    "correlation_energy",
    "correlation_value_and_grad_from_common",
    "correlation_value_and_grad",
    "correlation_value_and_grad_with_iao",
]


_PAO_ORTH_THRESHOLD = 1e-6
_WEIGHT_DEGENERACY_TOLERANCE = 1e-10


@dataclass(frozen=True)
class FixedPAOSubspaceSelection:
    """Discrete selections made while constructing one local PAO subspace.

    ``parent_columns`` index the globally retained, normalized PAOs.  The
    remaining arrays index ascending eigenvalue order at the corresponding
    fixed-rank step, except ``completeness_keep``, which indexes the overlap-
    selected PAO columns directly.
    """

    parent_columns: onp.ndarray
    support_ao_indices: onp.ndarray
    canonical_keep: onp.ndarray
    overlap_keep: onp.ndarray
    completeness_keep: onp.ndarray
    metric_keep: onp.ndarray


@dataclass(frozen=True)
class IAOMP2FragmentStaticSelection:
    """Fixed-shape selections for one fragment's strong and weak spaces."""

    fragment_index: int
    iao_indices: onp.ndarray
    fragment_atoms: onp.ndarray | None
    strong_fragments: onp.ndarray
    extended_atoms: onp.ndarray
    extended_ao_indices: onp.ndarray
    pao_center_atoms: onp.ndarray
    strong_occ_union_keep: onp.ndarray
    strong_occ_metric_keep: onp.ndarray
    strong_virtual: FixedPAOSubspaceSelection
    primary_atoms: onp.ndarray
    primary_ao_indices: onp.ndarray
    primary_bp_atoms: onp.ndarray
    weak_weight_eigen_indices: onp.ndarray
    weak_weight_degenerate_blocks: tuple[tuple[int, int], ...]
    weak_occ_norm_keep: onp.ndarray
    weak_occ_span_metric_keep: onp.ndarray
    weak_virtual: FixedPAOSubspaceSelection | None

    @property
    def has_weak_screen(self):
        return self.weak_virtual is not None


@dataclass(frozen=True)
class IAOFragmentMP2StaticSelections:
    """All non-differentiable metadata for a fixed IAO-fragment topology."""

    frozen: object
    thresholds: object
    active_occ_indices: onp.ndarray
    active_vir_indices: onp.ndarray
    pao_projected_out_indices: onp.ndarray
    pao_parent_ao_indices: onp.ndarray
    ao2pao_map: onp.ndarray
    frag_lolist: tuple[onp.ndarray, ...]
    frag_atmlist: tuple[onp.ndarray | None, ...]
    strong_mask: onp.ndarray
    fragments: tuple[IAOMP2FragmentStaticSelection, ...]


class IAOFragmentOccupiedData(NamedTuple):
    """Geometry-dependent occupied projection and weight of one IAO block."""

    iao_coeff: object
    iao_occ_overlap: object
    occupied_projection: object
    occupied_weight: object


class IAOFragmentMP2ContinuousData(NamedTuple):
    """Common differentiable arrays rebuilt once for all fragments."""

    s1e: object
    fock: object
    occupied_coeff: object
    virtual_coeff: object
    occupied_energy: object
    virtual_energy: object
    iao_coeff: object
    pao_coeff: object
    fragment_occupied_data: tuple[IAOFragmentOccupiedData, ...]


class IAOMP2StrongDomain(NamedTuple):
    """Differentiable semicanonical orbitals and weights in one strong ED."""

    occupied_coeff: object
    virtual_coeff: object
    occupied_energy: object
    virtual_energy: object
    target_projection: object
    target_weight: object
    partner_weight: object


class IAOMP2WeakScreen(NamedTuple):
    """Differentiable weighted occupied modes and PAOs for one weak screen."""

    weights: object
    occupied_energy: object
    occupied_coeff: object
    virtual_energy: object
    virtual_coeff: object


@dataclass(frozen=True)
class IAOMP2FragmentDimensions:
    """Fixed dimensions of one strong extended-domain (ED) calculation.

    ``strong_fragments`` includes the target fragment itself.  The occupied
    and virtual ranks are the fixed ranks selected at the reference geometry;
    they are therefore also the actual array dimensions rebuilt during a
    fixed-topology energy or gradient evaluation.
    """

    fragment_index: int
    strong_fragments: tuple[int, ...]
    extended_atoms: tuple[int, ...]
    n_domain_atoms: int
    n_domain_ao: int
    n_domain_occ: int
    n_domain_vir: int


@dataclass(frozen=True)
class IAOMP2TermResult:
    """Scalar result retained after one progressive local-MP2 pullback.

    A strong term is one weighted ED row and has ``right_fragment=None``.
    A weak term is one unordered distant fragment pair.  Only host scalars
    and integer labels are retained; no orbital frame, pullback, or AD tape is
    stored in this record.

    In the serial evaluator, frame construction is part of the term VJP and
    is consequently included in ``forward_seconds``/``reverse_seconds``.  In
    MPI, root builds and replays the common-to-frame map separately, and those
    times are reported in ``frame_build_seconds`` and
    ``frame_replay_seconds``.
    """

    kind: str
    left_fragment: int
    right_fragment: int | None
    energy: float
    forward_seconds: float
    reverse_seconds: float
    frame_build_seconds: float = 0.0
    frame_replay_seconds: float = 0.0
    worker_rank: int = 0


@dataclass(frozen=True)
class IAOMP2GradientTiming:
    """Wall/work timing of a progressive local-MP2 correlation pullback.

    On one process all entries are ordinary wall seconds.  In MPI the strong
    and weak forward/reverse entries are sums of the worker term times and are
    thus *work seconds*, while ``total_seconds`` is the collective elapsed
    wall time (the maximum over ranks).  The common and frame fields are root
    wall seconds.  This distinction exposes both parallel load and critical
    elapsed time without pretending their sum is a serial wall clock.
    """

    common_forward_seconds: float = 0.0
    strong_forward_seconds: float = 0.0
    strong_reverse_seconds: float = 0.0
    weak_forward_seconds: float = 0.0
    weak_reverse_seconds: float = 0.0
    frame_build_seconds: float = 0.0
    frame_replay_seconds: float = 0.0
    common_reverse_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass(frozen=True)
class IAOMP2Decomposition:
    """Energy, topology, dimensions, and timing from one gradient run.

    ``n_strong_pairs`` and ``n_weak_pairs`` count unordered *interfragment*
    pairs and sum to ``n_fragments * (n_fragments - 1) // 2``.  The strong
    energy is evaluated as one weighted ED row per fragment, so
    ``n_strong_ed_terms`` is instead always ``n_fragments`` and includes the
    intrafragment contribution.  Reporting both counts avoids identifying ED
    rows with pair-list entries.
    """

    e_corr: float
    e_strong: float
    e_weak: float
    n_fragments: int
    n_strong_pairs: int
    n_weak_pairs: int
    n_strong_ed_terms: int
    n_weak_pair_terms: int
    fragments: tuple[IAOMP2FragmentDimensions, ...]
    terms: tuple[IAOMP2TermResult, ...]
    timing: IAOMP2GradientTiming


def _fragment_dimensions(static):
    """Return host-only ED dimensions encoded by fixed selections."""
    rows = []
    for fragment in static.fragments:
        extended_atoms = tuple(
            int(value) for value in onp.asarray(fragment.extended_atoms)
        )
        rows.append(IAOMP2FragmentDimensions(
            fragment_index=int(fragment.fragment_index),
            strong_fragments=tuple(
                int(value)
                for value in onp.asarray(fragment.strong_fragments)
            ),
            extended_atoms=extended_atoms,
            n_domain_atoms=len(extended_atoms),
            n_domain_ao=int(onp.asarray(
                fragment.extended_ao_indices
            ).size),
            n_domain_occ=int(onp.asarray(
                fragment.strong_occ_metric_keep
            ).size),
            n_domain_vir=int(onp.asarray(
                fragment.strong_virtual.metric_keep
            ).size),
        ))
    return tuple(rows)


def _make_decomposition(static, energy, terms, timing):
    """Build a host-only diagnostic result from completed scalar terms."""
    terms = tuple(terms)
    strong_terms = tuple(term for term in terms if term.kind == "strong")
    weak_terms = tuple(term for term in terms if term.kind == "weak")
    nfragment = len(static.fragments)
    strong_mask = onp.asarray(static.strong_mask, dtype=bool)
    n_strong_pairs = int(onp.count_nonzero(onp.triu(
        strong_mask, k=1
    )))
    n_weak_pairs = nfragment * (nfragment - 1) // 2 - n_strong_pairs
    if len(strong_terms) != nfragment or len(weak_terms) != n_weak_pairs:
        raise RuntimeError(
            "local-MP2 term records do not match the fixed pair topology"
        )
    return IAOMP2Decomposition(
        # Preserve the evaluator's original mixed term-addition order for the
        # reported total rather than recomputing it as strong + weak.
        e_corr=float(jax.device_get(energy)),
        e_strong=float(sum(term.energy for term in strong_terms)),
        e_weak=float(sum(term.energy for term in weak_terms)),
        n_fragments=nfragment,
        n_strong_pairs=n_strong_pairs,
        n_weak_pairs=n_weak_pairs,
        n_strong_ed_terms=len(strong_terms),
        n_weak_pair_terms=len(weak_terms),
        fragments=_fragment_dimensions(static),
        terms=terms,
        timing=timing,
    )


def _host_array(value, dtype=None):
    array = onp.asarray(jax.device_get(value))
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _hermitian_numpy(matrix):
    matrix = onp.asarray(matrix)
    return 0.5 * (matrix + matrix.T.conj())


def _eigh_keep_numpy(matrix, threshold):
    eigenvalue, eigenvector = onp_scipy_linalg.eigh(
        _hermitian_numpy(matrix), check_finite=False
    )
    keep = onp.where(onp.real(eigenvalue) > threshold)[0].astype(onp.int32)
    return eigenvalue, eigenvector, keep


def _metric_orthonormalize_numpy(coeff, overlap, keep):
    coeff = onp.asarray(coeff)
    keep = onp.asarray(keep, dtype=onp.int32)
    if coeff.shape[1] == 0 or keep.size == 0:
        return onp.zeros((coeff.shape[0], 0), dtype=coeff.dtype)
    metric = coeff.T.conj() @ onp.asarray(overlap) @ coeff
    eigenvalue, eigenvector = onp_scipy_linalg.eigh(
        _hermitian_numpy(metric), check_finite=False
    )
    return coeff @ (
        eigenvector[:, keep]
        / onp.sqrt(onp.real(eigenvalue[keep]))[None, :]
    )


def _semicanonicalize_numpy(coeff, fock):
    coeff = onp.asarray(coeff)
    if coeff.shape[1] <= 1:
        return coeff
    projected = coeff.T.conj() @ onp.asarray(fock) @ coeff
    _, rotation = onp_scipy_linalg.eigh(
        _hermitian_numpy(projected), check_finite=False
    )
    return coeff @ rotation


def _project_numpy(coeff, s21, s22):
    return onp.linalg.solve(onp.asarray(s22), onp.asarray(s21) @ coeff)


def _orthogonalize_numpy(occupied, candidate, overlap):
    return candidate - occupied @ (
        occupied.T.conj() @ onp.asarray(overlap) @ candidate
    )


def _pao_columns_on_atoms(mol, atoms, ao2pao_map):
    atoms = onp.asarray(atoms, dtype=onp.int32).reshape(-1)
    aoslices = onp.asarray(mol.aoslice_by_atom())[:, 2:]
    columns = []
    for atom in atoms:
        p0, p1 = map(int, aoslices[int(atom)])
        mapped = onp.asarray(ao2pao_map[p0:p1], dtype=onp.int32)
        columns.extend(mapped[mapped >= 0].tolist())
    return onp.asarray(columns, dtype=onp.int32)


def _compute_completeness_numpy(candidate, overlap, ao_indices):
    candidate = onp.asarray(candidate)
    if candidate.shape[1] == 0:
        return onp.zeros((0,), dtype=float)
    overlap = onp.asarray(overlap)
    ao_indices = onp.asarray(ao_indices, dtype=onp.int32)
    values = overlap[ao_indices] @ candidate
    recovered = onp.linalg.solve(
        overlap[onp.ix_(ao_indices, ao_indices)], values
    )
    return onp.real(onp.sum(recovered.conj() * values, axis=0))


def _reference_pao_overlap_selection(
    mol,
    pao_coeff,
    ao2pao_map,
    overlap,
    *,
    parent_atoms,
    support_atoms,
    representation_atoms,
    completeness_threshold,
    overlap_threshold,
    occupied_for_projection,
    metric_threshold,
):
    """Reproduce a PAO build once and record every thresholded label."""

    overlap = onp.asarray(overlap)
    parent_columns = _pao_columns_on_atoms(
        mol, parent_atoms, ao2pao_map
    )
    pao_parent = onp.asarray(pao_coeff)[:, parent_columns]

    gram = pao_parent.T.conj() @ overlap @ pao_parent
    gram_e, gram_v, canonical_keep = _eigh_keep_numpy(
        gram, _PAO_ORTH_THRESHOLD
    )
    if canonical_keep.size:
        pao_orth = pao_parent @ (
            gram_v[:, canonical_keep]
            / onp.sqrt(onp.real(gram_e[canonical_keep]))[None, :]
        )
    else:
        pao_orth = onp.zeros((mol.nao, 0), dtype=pao_parent.dtype)

    support_ao = util.ao_index_by_atom(
        mol, onp.asarray(support_atoms, dtype=onp.int32)
    ).astype(onp.int32, copy=False)
    if pao_orth.shape[1]:
        tmp = overlap[support_ao] @ pao_orth
        local_metric = overlap[onp.ix_(support_ao, support_ao)]
        projected_overlap = tmp.T.conj() @ onp.linalg.solve(local_metric, tmp)
        _, overlap_v, overlap_keep = _eigh_keep_numpy(
            projected_overlap, overlap_threshold
        )
        candidate = pao_orth @ overlap_v[:, overlap_keep]
    else:
        overlap_keep = onp.zeros((0,), dtype=onp.int32)
        candidate = onp.zeros((mol.nao, 0), dtype=pao_parent.dtype)

    representation_ao = util.ao_index_by_atom(
        mol, onp.asarray(representation_atoms, dtype=onp.int32)
    ).astype(onp.int32, copy=False)
    completeness = _compute_completeness_numpy(
        candidate, overlap, representation_ao
    )
    completeness_keep = onp.where(
        completeness > completeness_threshold
    )[0].astype(onp.int32)
    candidate = candidate[:, completeness_keep]

    if candidate.shape[1]:
        s21 = overlap[representation_ao]
        s22 = overlap[onp.ix_(representation_ao, representation_ao)]
        candidate_local = _project_numpy(candidate, s21, s22)
        candidate_local = _orthogonalize_numpy(
            occupied_for_projection, candidate_local, s22
        )
        metric = candidate_local.T.conj() @ s22 @ candidate_local
        _, _, metric_keep = _eigh_keep_numpy(metric, metric_threshold)
        candidate_local = _metric_orthonormalize_numpy(
            candidate_local, s22, metric_keep
        )
    else:
        metric_keep = onp.zeros((0,), dtype=onp.int32)
        candidate_local = onp.zeros(
            (representation_ao.size, 0), dtype=pao_parent.dtype
        )

    selection = FixedPAOSubspaceSelection(
        parent_columns=parent_columns,
        support_ao_indices=support_ao,
        canonical_keep=canonical_keep,
        overlap_keep=overlap_keep,
        completeness_keep=completeness_keep,
        metric_keep=metric_keep,
    )
    return selection, candidate_local


def _weight_eigenspace_metadata(weight, threshold):
    eigenvalue, _ = onp_scipy_linalg.eigh(
        _hermitian_numpy(weight), check_finite=False
    )
    order = onp.argsort(onp.real(eigenvalue))[::-1]
    retained = order[onp.real(eigenvalue[order]) > threshold]
    retained = onp.asarray(retained, dtype=onp.int32)
    values = onp.real(eigenvalue[retained])

    blocks = []
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and onp.isclose(
            values[stop],
            values[start],
            rtol=_WEIGHT_DEGENERACY_TOLERANCE,
            atol=_WEIGHT_DEGENERACY_TOLERANCE,
        ):
            stop += 1
        blocks.append((start, stop))
        start = stop
    return retained, tuple(blocks)


def _fragment_modes_numpy(
    occupied_weight,
    occupied_energy,
    retained,
    degenerate_blocks,
):
    eigenvalue, eigenvector = onp_scipy_linalg.eigh(
        _hermitian_numpy(occupied_weight), check_finite=False
    )
    values = onp.real(eigenvalue[retained])
    modes = eigenvector[:, retained]
    output_modes = []
    output_weights = []
    for start, stop in degenerate_blocks:
        block = modes[:, start:stop]
        if stop - start > 1:
            projected_fock = block.T.conj() @ (
                onp.asarray(occupied_energy)[:, None] * block
            )
            _, rotation = onp_scipy_linalg.eigh(
                _hermitian_numpy(projected_fock), check_finite=False
            )
            block = block @ rotation
        output_modes.append(block)
        output_weights.append(onp.full(
            stop - start, onp.mean(values[start:stop]), dtype=float
        ))
    if not output_modes:
        nocc = onp.asarray(occupied_weight).shape[0]
        return onp.zeros((0,)), onp.zeros((nocc, 0))
    return onp.concatenate(output_weights), onp.hstack(output_modes)


def _extract_active_indices(mf, frozen):
    pt = dfmp2.MP2(mf, frozen=frozen)
    masks = _mo_splitter(pt)
    active_occ = onp.where(_host_array(masks[1], bool))[0].astype(onp.int32)
    active_vir = onp.where(_host_array(masks[2], bool))[0].astype(onp.int32)
    projected_out = onp.where(
        _host_array(masks[0], bool)
        | _host_array(masks[1], bool)
        | _host_array(masks[3], bool)
    )[0].astype(onp.int32)
    return active_occ, active_vir, projected_out


def build_iao_mp2_static_selections(mf, topology):
    """Extract fixed ranks and index lists from a reference topology.

    Parameters
    ----------
    mf
        The concrete reference SCF object used to build ``topology``.
    topology : :class:`~pyscfad.dlno.iao_mp2.IAOFragmentTopology`
        Energy-validation topology at the same geometry.

    Returns
    -------
    :class:`IAOFragmentMP2StaticSelections`
        A container with no geometry-dependent floating-point coefficients.
    """

    if not isinstance(topology, IAOFragmentTopology):
        raise TypeError("topology must be an IAOFragmentTopology")
    if mf.mol.nao != topology.s1e.shape[0]:
        raise ValueError("mf and topology use different AO spaces")

    thresholds = topology.thresholds
    active_occ, active_vir, projected_out = _extract_active_indices(
        mf, topology.frozen
    )
    reference_mo = _host_array(mf.mo_coeff)
    if not onp.allclose(
        reference_mo[:, active_occ], _host_array(topology.occupied_coeff),
        atol=1e-9, rtol=1e-9,
    ):
        raise ValueError("active occupied MO indices do not match topology")
    if not onp.allclose(
        reference_mo[:, active_vir], _host_array(topology.virtual_coeff),
        atol=1e-9, rtol=1e-9,
    ):
        raise ValueError("active virtual MO indices do not match topology")

    overlap = _host_array(topology.s1e)
    fock = _host_array(topology.fock)
    occupied = _host_array(topology.occupied_coeff)
    occupied_energy = _host_array(topology.occupied_energy)
    pao_coeff = _host_array(topology.pao_coeff)
    ao2pao_map = _host_array(topology.ao2pao_map, onp.int32)
    pao_parent_ao = onp.where(ao2pao_map >= 0)[0].astype(onp.int32)
    if not onp.array_equal(
        ao2pao_map[pao_parent_ao], onp.arange(pao_parent_ao.size)
    ):
        raise ValueError("topology PAO map is not in parent-AO order")

    frag_lolist = tuple(
        _host_array(indices, onp.int32).reshape(-1)
        for indices in topology.frag_lolist
    )
    frag_atmlist = tuple(
        None if atoms is None else _host_array(atoms, onp.int32).reshape(-1)
        for atoms in topology.frag_atmlist
    )
    fragments = []

    for fragment_index, iao_indices in enumerate(frag_lolist):
        partners = _host_array(
            topology.strong_fragments[fragment_index], onp.int32
        ).reshape(-1)
        extended_atoms = _host_array(
            topology.extended_domain[fragment_index], onp.int32
        ).reshape(-1)
        center_atoms = _host_array(
            topology.pao_center_domain[fragment_index], onp.int32
        ).reshape(-1)
        extended_ao = util.ao_index_by_atom(
            mf.mol, extended_atoms
        ).astype(onp.int32, copy=False)
        s21 = overlap[extended_ao]
        s22 = overlap[onp.ix_(extended_ao, extended_ao)]
        fock22 = fock[onp.ix_(extended_ao, extended_ao)]

        partner_overlap = onp.vstack([
            _host_array(
                topology.fragment_occupied_data[partner].iao_occ_overlap
            )
            for partner in partners
        ])
        # The reference implementation obtains this range from an SVD of the
        # stacked thin factors.  Use the mathematically equivalent eigenspace
        # of M^H M here.  The degeneracy-aware JAX Hermitian eigensolver has a
        # well-defined projector response when (as often happens for a full
        # strong union) several singular values are exactly one; differentiating
        # the raw SVD gauge produces NaNs in that case.
        partner_weight = partner_overlap.T.conj() @ partner_overlap
        partner_e, partner_v = onp_scipy_linalg.eigh(
            _hermitian_numpy(partner_weight), check_finite=False
        )
        occ_union_keep = onp.where(
            onp.real(partner_e) > thresholds.occupied_weight
        )[0].astype(onp.int32)
        if occ_union_keep.size == 0:
            occ_union_keep = onp.asarray(
                [int(onp.argmax(onp.real(partner_e)))], dtype=onp.int32
            )
        occupied_candidate = occupied @ partner_v[:, occ_union_keep]
        occupied_local_raw = _project_numpy(
            occupied_candidate, s21, s22
        )
        occupied_metric = (
            occupied_local_raw.T.conj() @ s22 @ occupied_local_raw
        )
        _, _, occ_metric_keep = _eigh_keep_numpy(
            occupied_metric, thresholds.metric_rank
        )
        occupied_local = _metric_orthonormalize_numpy(
            occupied_local_raw, s22, occ_metric_keep
        )
        occupied_local = _semicanonicalize_numpy(
            occupied_local, fock22
        )
        if occupied_local.shape[1] == 0:
            raise RuntimeError(
                f"fragment {fragment_index} reference ED has no occupied space"
            )

        strong_virtual, strong_virtual_ref = (
            _reference_pao_overlap_selection(
                mf.mol,
                pao_coeff,
                ao2pao_map,
                overlap,
                parent_atoms=center_atoms,
                support_atoms=extended_atoms,
                representation_atoms=extended_atoms,
                completeness_threshold=thresholds.ed_pao,
                overlap_threshold=thresholds.domain_pao,
                occupied_for_projection=occupied_local,
                metric_threshold=thresholds.metric_rank,
            )
        )
        if strong_virtual_ref.shape[1] == 0:
            raise RuntimeError(
                f"fragment {fragment_index} reference ED has no virtual space"
            )

        primary_atoms = _host_array(
            topology.primary_domain[fragment_index], onp.int32
        ).reshape(-1)
        primary_bp_atoms = _host_array(
            topology.primary_bp_domain[fragment_index], onp.int32
        ).reshape(-1)
        primary_ao = util.ao_index_by_atom(
            mf.mol, primary_atoms
        ).astype(onp.int32, copy=False)
        primary_s21 = overlap[primary_ao]
        primary_s22 = overlap[onp.ix_(primary_ao, primary_ao)]

        occupied_weight = _host_array(
            topology.fragment_occupied_data[fragment_index].occupied_weight
        )
        weight_indices, weight_blocks = _weight_eigenspace_metadata(
            occupied_weight, thresholds.occupied_weight
        )
        _, weak_modes = _fragment_modes_numpy(
            occupied_weight,
            occupied_energy,
            weight_indices,
            weight_blocks,
        )
        weak_occupied_global = occupied @ weak_modes
        weak_occupied_raw = _project_numpy(
            weak_occupied_global, primary_s21, primary_s22
        )
        norm2 = onp.real(onp.einsum(
            "ui,uv,vi->i",
            weak_occupied_raw.conj(),
            primary_s22,
            weak_occupied_raw,
            optimize=True,
        ))
        weak_norm_keep = onp.where(
            norm2 > thresholds.metric_rank
        )[0].astype(onp.int32)
        if weak_norm_keep.size:
            weak_occupied = (
                weak_occupied_raw[:, weak_norm_keep]
                / onp.sqrt(norm2[weak_norm_keep])[None, :]
            )
            weak_occ_metric = (
                weak_occupied.T.conj() @ primary_s22 @ weak_occupied
            )
            _, _, weak_occ_span_keep = _eigh_keep_numpy(
                weak_occ_metric, thresholds.metric_rank
            )
            weak_occ_span = _metric_orthonormalize_numpy(
                weak_occupied, primary_s22, weak_occ_span_keep
            )
            weak_virtual, weak_virtual_ref = (
                _reference_pao_overlap_selection(
                    mf.mol,
                    pao_coeff,
                    ao2pao_map,
                    overlap,
                    parent_atoms=primary_bp_atoms,
                    support_atoms=primary_bp_atoms,
                    representation_atoms=primary_atoms,
                    completeness_threshold=thresholds.bp_pao,
                    overlap_threshold=thresholds.domain_pao,
                    occupied_for_projection=weak_occ_span,
                    metric_threshold=thresholds.metric_rank,
                )
            )
            if weak_virtual_ref.shape[1] == 0:
                weak_virtual = None
        else:
            weak_occ_span_keep = onp.zeros((0,), dtype=onp.int32)
            weak_virtual = None

        fragments.append(IAOMP2FragmentStaticSelection(
            fragment_index=fragment_index,
            iao_indices=iao_indices.copy(),
            fragment_atoms=(
                None if frag_atmlist[fragment_index] is None
                else frag_atmlist[fragment_index].copy()
            ),
            strong_fragments=partners,
            extended_atoms=extended_atoms,
            extended_ao_indices=extended_ao,
            pao_center_atoms=center_atoms,
            strong_occ_union_keep=occ_union_keep,
            strong_occ_metric_keep=occ_metric_keep,
            strong_virtual=strong_virtual,
            primary_atoms=primary_atoms,
            primary_ao_indices=primary_ao,
            primary_bp_atoms=primary_bp_atoms,
            weak_weight_eigen_indices=weight_indices,
            weak_weight_degenerate_blocks=weight_blocks,
            weak_occ_norm_keep=weak_norm_keep,
            weak_occ_span_metric_keep=weak_occ_span_keep,
            weak_virtual=weak_virtual,
        ))

    return IAOFragmentMP2StaticSelections(
        frozen=topology.frozen,
        thresholds=thresholds,
        active_occ_indices=active_occ,
        active_vir_indices=active_vir,
        pao_projected_out_indices=projected_out,
        pao_parent_ao_indices=pao_parent_ao,
        ao2pao_map=ao2pao_map.copy(),
        frag_lolist=frag_lolist,
        frag_atmlist=frag_atmlist,
        strong_mask=_host_array(topology.strong_mask, bool),
        fragments=tuple(fragments),
    )


def rebuild_iao_mp2_common(mf, static, *, iao_coeff=None):
    """Rebuild all common continuous IAO-MP2 arrays from ``mf``.

    ``static`` supplies only fixed MO/PAO column labels.  With the default
    ``iao_coeff=None``, IAOs are recomputed from the current active occupied
    orbitals and therefore carry both AO-integral and SCF orbital response.
    A supplied ``iao_coeff`` remains differentiable if it is a traced array,
    but the caller is responsible for constructing it at the current geometry.
    """

    if not isinstance(static, IAOFragmentMP2StaticSelections):
        raise TypeError("static must be IAOFragmentMP2StaticSelections")
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
        iao_coeff = lno_base.get_iao(mol, occupied)
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
        fragment_data.append(IAOFragmentOccupiedData(
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

    return IAOFragmentMP2ContinuousData(
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
    keep = onp.asarray(keep, dtype=onp.int32)
    if coeff.shape[1] == 0 or keep.size == 0:
        return np.zeros((coeff.shape[0], 0), dtype=coeff.dtype)
    if (
        keep.size == coeff.shape[1]
        and onp.array_equal(keep, onp.arange(keep.size))
    ):
        # When no metric direction is truncated, an eigendecomposition is
        # only a choice of gauge.  Its regularized JVP drops eigenvector
        # rotations inside a numerically degenerate block, but those terms
        # are required by the Frechet derivative of G^{-1/2}; on large EDs
        # this produced enormous spurious occupied-projector responses.
        # Cholesky gives an equally valid, smooth and rotationally covariant
        # orthonormal frame Q = C L^{-H}, G = L L^H, without eigenvectors.
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


def _rebuild_selected_pao_space(
    common,
    selection,
    representation_ao_indices,
    occupied_for_projection,
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
    support_metric = s1e[onp.ix_(support_ao, support_ao)]
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

    representation_ao = onp.asarray(
        representation_ao_indices, dtype=onp.int32
    )
    s21 = s1e[representation_ao]
    s22 = s1e[onp.ix_(representation_ao, representation_ao)]
    candidate = util.project_mo(candidate, s21, s22)
    candidate = util.orthogonalize(
        occupied_for_projection, candidate, s22
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
    s22 = s1e[onp.ix_(ao_indices, ao_indices)]
    fock22 = fock[onp.ix_(ao_indices, ao_indices)]

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
    occupied_local = util.project_mo(occupied_candidate, s21, s22)
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
    )
    virtual_energy, virtual_local = _fixed_semicanonicalize(
        virtual_local, fock22
    )

    target_iao = common.fragment_occupied_data[
        int(fragment_index)
    ].iao_coeff
    partner_iao = np.concatenate([
        common.fragment_occupied_data[int(partner)].iao_coeff
        for partner in fragment.strong_fragments
    ], axis=1)
    overlap_to_local = s1e[:, ao_indices] @ occupied_local
    target_projection = target_iao.T.conj() @ overlap_to_local
    partner_projection = partner_iao.T.conj() @ overlap_to_local
    target_weight = target_projection.T.conj() @ target_projection
    partner_weight = partner_projection.T.conj() @ partner_projection

    return IAOMP2StrongDomain(
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

    weight, occupied_mode = _rebuild_fragment_modes(common, fragment)
    occupied_global = common.occupied_coeff @ occupied_mode
    ao_indices = fragment.primary_ao_indices
    s1e = common.s1e
    s21 = s1e[ao_indices]
    s22 = s1e[onp.ix_(ao_indices, ao_indices)]
    fock22 = common.fock[onp.ix_(ao_indices, ao_indices)]
    occupied_local = util.project_mo(occupied_global, s21, s22)

    norm2 = np.real(np.einsum(
        "ui,uv,vi->i",
        occupied_local.conj(),
        s22,
        occupied_local,
        optimize=True,
    ))
    norm_keep = fragment.weak_occ_norm_keep
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
    )
    virtual_energy, virtual_local = _fixed_semicanonicalize(
        virtual_local, fock22
    )
    return IAOMP2WeakScreen(
        weights=weight,
        occupied_energy=occupied_energy,
        occupied_coeff=occupied_local,
        virtual_energy=virtual_energy,
        virtual_coeff=virtual_local,
    )


def strong_domain_energy(mf, domain, static, fragment_index):
    """Evaluate one exact two-sided MP2 term in a supplied strong ED frame."""
    fragment_index = int(fragment_index)
    fragment = static.fragments[fragment_index]
    nocc = domain.occupied_coeff.shape[1]
    local_coeff = np.concatenate(
        (domain.occupied_coeff, domain.virtual_coeff), axis=1
    )
    lov = lno_base.get_local_Lov(
        mf,
        local_coeff,
        nocc,
        fragment.extended_atoms,
        integral_direct=True,
    )
    lov = np.reshape(
        lov, (-1, nocc, domain.virtual_coeff.shape[1])
    )
    return fragment_pair_energy_from_lov_jax(
        lov,
        domain.occupied_energy,
        domain.virtual_energy,
        domain.target_projection,
        domain.partner_weight,
        max_memory_mb=static.thresholds.mp2_block_memory_mb,
    )


def strong_fragment_energy(mf, common, static, fragment_index):
    """Differentiate the exact two-sided MP2 energy in one fixed ED.

    The atom list and every retained rank/index come from ``static``.  The
    local occupied/virtual orbitals, their semicanonical energies, the IAO
    target/partner weights, and the local density-fitting factors are rebuilt
    from the current geometry.
    """
    domain = build_strong_ed_domain(common, static, fragment_index)
    return strong_domain_energy(
        mf, domain, static, fragment_index
    )


def _multipole_screen_arguments(screen, atoms):
    """Expand one shared weak PAO space into the per-mode API."""
    nmode = int(screen.occupied_coeff.shape[1])
    return (
        screen.occupied_energy,
        tuple(screen.occupied_coeff[:, index] for index in range(nmode)),
        tuple(screen.virtual_energy for _ in range(nmode)),
        tuple(screen.virtual_coeff for _ in range(nmode)),
        tuple(atoms for _ in range(nmode)),
    )


def _validate_weak_pair(static, left_index, right_index):
    left_index = int(left_index)
    right_index = int(right_index)
    if left_index == right_index:
        raise ValueError("a weak pair must contain two distinct fragments")
    if left_index > right_index:
        left_index, right_index = right_index, left_index
    nfragment = len(static.fragments)
    if not (0 <= left_index < nfragment and 0 <= right_index < nfragment):
        raise IndexError("weak-pair fragment index is out of range")
    if bool(static.strong_mask[left_index, right_index]):
        raise ValueError(
            f"fragment pair ({left_index}, {right_index}) is classified strong"
        )
    return left_index, right_index


def weak_screen_pair_energy(
    mf,
    left,
    right,
    static,
    left_index,
    right_index,
):
    """Evaluate one weak multipole pair in supplied root-gauge screens."""
    left_index, right_index = _validate_weak_pair(
        static, left_index, right_index
    )

    if left is None:
        raise RuntimeError(
            f"fragment {left_index} has fixed weak pairs but no "
            "multipole screen"
        )
    left_fragment = static.fragments[left_index]
    left_args = _multipole_screen_arguments(
        left, left_fragment.primary_atoms
    )

    if right is None:
        raise RuntimeError(
            f"fragment {right_index} has fixed weak pairs but no "
            "multipole screen"
        )
    right_fragment = static.fragments[right_index]
    right_args = _multipole_screen_arguments(
        right, right_fragment.primary_atoms
    )
    pair = dlno_mp2.pair_energy_multipole_cross(
        mf.mol,
        left_args[0],
        left_args[1],
        left_args[2],
        left_args[3],
        right_args[0],
        right_args[1],
        right_args[2],
        right_args[3],
        atmlst_left=left_args[4],
        atmlst_right=right_args[4],
        order=static.thresholds.multipole_order,
    )
    return np.real(np.sum(
        pair * left.weights[:, None] * right.weights[None, :]
    ))


def _weak_pair_energy(mf, common, static, left_index, right_index):
    """Build both screens and evaluate one unordered weak pair."""
    left_index, right_index = _validate_weak_pair(
        static, left_index, right_index
    )
    left = build_weak_multipole_screen(common, static, left_index)
    right = build_weak_multipole_screen(common, static, right_index)
    return weak_screen_pair_energy(
        mf,
        left,
        right,
        static,
        left_index,
        right_index,
    )


def _correlation_term_specs(static):
    """Enumerate every additive strong row and unordered weak pair once."""
    specs = []
    nfragment = len(static.fragments)
    strong_mask = onp.asarray(static.strong_mask, dtype=bool)
    for fragment_index in range(nfragment):
        specs.append(("strong", fragment_index, -1))
        for partner_index in range(fragment_index + 1, nfragment):
            if not strong_mask[fragment_index, partner_index]:
                specs.append(("weak", fragment_index, partner_index))
    return tuple(specs)


def _correlation_term_energy(mf, common, static, spec):
    """Evaluate one additive term from :func:`_correlation_term_specs`."""
    kind, left_index, right_index = spec
    if kind == "strong":
        if int(right_index) != -1:
            raise ValueError("a strong-row term must use right_index=-1")
        return strong_fragment_energy(
            mf, common, static, int(left_index)
        ).total
    if kind == "weak":
        return _weak_pair_energy(
            mf,
            common,
            static,
            int(left_index),
            int(right_index),
        )
    raise ValueError(f"unknown IAO-MP2 correlation term kind {kind!r}")


def _contains_tracer(value):
    return any(
        isinstance(leaf, jax.core.Tracer)
        for leaf in jax.tree_util.tree_leaves(value)
    )


def correlation_energy(mf, static, *, iao_coeff=None):
    """Forward-only local-MP2 energy with fixed ED and pair topology.

    This scalar form is retained for energy checks and finite differences.
    Applying an outer JAX transform would place every local term on one tape,
    so traced inputs are rejected.  Use :func:`correlation_value_and_grad` for
    derivatives; it pulls back each strong ED and unordered weak pair
    separately and releases their intermediates immediately.
    """
    if _contains_tracer((mf, iao_coeff)):
        raise TypeError(
            "correlation_energy is forward-only; use "
            "correlation_value_and_grad for automatic differentiation"
        )
    common = rebuild_iao_mp2_common(
        mf, static, iao_coeff=iao_coeff
    )
    energy = np.zeros((), dtype=common.s1e.dtype)
    for spec in _correlation_term_specs(static):
        energy = energy + _correlation_term_energy(
            mf, common, static, spec
        )
    return np.real(energy)


def _add_cotangent(left, right):
    if left is None:
        return right
    if right is None:
        return left
    if hasattr(left, "dtype") and left.dtype == jax.dtypes.float0:
        return right
    if hasattr(right, "dtype") and right.dtype == jax.dtypes.float0:
        return left
    return left + right


def _restart_is_enabled(restart):
    return restart is not None and bool(getattr(restart, "enabled", False))


def _restart_is_resuming(restart):
    return _restart_is_enabled(restart) and bool(
        getattr(restart, "resume", False)
    )


def _term_result_metadata(term):
    return {
        "kind": str(term.kind),
        "left_fragment": int(term.left_fragment),
        "right_fragment": (
            None
            if term.right_fragment is None
            else int(term.right_fragment)
        ),
        "energy": float(term.energy),
        "forward_seconds": float(term.forward_seconds),
        "reverse_seconds": float(term.reverse_seconds),
        "frame_build_seconds": float(term.frame_build_seconds),
        "frame_replay_seconds": float(term.frame_replay_seconds),
        "worker_rank": int(term.worker_rank),
    }


def _term_result_from_metadata(row):
    return IAOMP2TermResult(
        kind=str(row["kind"]),
        left_fragment=int(row["left_fragment"]),
        right_fragment=(
            None
            if row.get("right_fragment") is None
            else int(row["right_fragment"])
        ),
        energy=float(row["energy"]),
        forward_seconds=float(row["forward_seconds"]),
        reverse_seconds=float(row["reverse_seconds"]),
        frame_build_seconds=float(row.get("frame_build_seconds", 0.0)),
        frame_replay_seconds=float(row.get("frame_replay_seconds", 0.0)),
        worker_rank=int(row.get("worker_rank", 0)),
    )


def _timing_metadata(timing):
    return {
        name: float(getattr(timing, name))
        for name in IAOMP2GradientTiming.__dataclass_fields__
    }


def _timing_from_metadata(row):
    return IAOMP2GradientTiming(**{
        name: float(row.get(name, 0.0))
        for name in IAOMP2GradientTiming.__dataclass_fields__
    })


def _details_restart_metadata(details):
    return {
        "term_results": [
            _term_result_metadata(term) for term in details.terms
        ],
        "timing": _timing_metadata(details.timing),
    }


def _details_from_restart_metadata(static, energy, metadata):
    terms = tuple(
        _term_result_from_metadata(row)
        for row in metadata.get("term_results", ())
    )
    timing = _timing_from_metadata(metadata.get("timing", {}))
    return _make_decomposition(static, energy, terms, timing)


def _zero_mf_cotangent(mf, dtype):
    """Return a live MF-shaped zero tree for restart deserialization."""

    _, pullback = jax.vjp(
        lambda mf_: np.zeros((), dtype=dtype), mf
    )
    mf_bar, = pullback(np.ones((), dtype=dtype))
    return mf_bar


def _zero_term_cotangents(mf, common):
    """Return live zero trees matching one MP2 term's two inputs."""

    _, pullback = jax.vjp(
        lambda mf_, common_: np.zeros((), dtype=common_.s1e.dtype),
        mf,
        common,
    )
    return pullback(np.ones((), dtype=common.s1e.dtype))


def _progressive_correlation_pullback(
    mf, common, static, *, return_details=False, restart=None
):
    """Return correlation energy plus cotangents of ``mf`` and ``common``.

    Strong ED energies and unordered weak pairs are pulled back as separate
    scalar terms.  In particular, do not place every weak partner of one
    fragment on the same reverse-mode tape: order-four AO multipole moments
    are large, and that grouped tape grows linearly with the number of weak
    partners even though the pair energies are independent.  Immediate
    pullback and collection keep the peak tied to one ED or one weak pair.
    """
    work = _correlation_term_specs(static)
    energy = np.zeros((), dtype=common.s1e.dtype)
    mf_bar = None
    common_bar = None
    term_results = []
    completed_count = 0
    previous_elapsed = 0.0
    total_start = time.perf_counter() if return_details else None

    if _restart_is_resuming(restart):
        zero_mf_bar, zero_common_bar = _zero_term_cotangents(mf, common)
        progress = restart.load_record(
            "correlation_progress",
            templates={
                "mf_bar": zero_mf_bar,
                "common_bar": zero_common_bar,
            },
            missing_ok=True,
        )
        if progress is not None:
            completed_count = int(progress.scalars["completed_count"])
            if completed_count < 0 or completed_count > len(work):
                raise RuntimeError(
                    "serial MP2 restart completed-term count is invalid"
                )
            saved_specs = tuple(
                (str(row[0]), int(row[1]), int(row[2]))
                for row in progress.metadata.get("completed_specs", ())
            )
            if saved_specs != work[:completed_count]:
                raise RuntimeError(
                    "serial MP2 restart term prefix does not match the "
                    "fixed topology"
                )
            energy = np.asarray(
                progress.scalars["energy"], dtype=common.s1e.dtype
            )
            mf_bar = progress.trees["mf_bar"]
            common_bar = progress.trees["common_bar"]
            previous_elapsed = float(
                progress.scalars.get("elapsed_seconds", 0.0)
            )
            if return_details:
                term_results = [
                    _term_result_from_metadata(row)
                    for row in progress.metadata.get("term_results", ())
                ]
                if len(term_results) != completed_count:
                    raise RuntimeError(
                        "serial MP2 restart term diagnostics are incomplete"
                    )
        del zero_mf_bar, zero_common_bar

    def accumulate_term(term, spec, term_index):
        nonlocal energy, mf_bar, common_bar
        forward_start = time.perf_counter() if return_details else None
        term_energy, pullback = jax.vjp(term, mf, common)
        if return_details:
            jax.block_until_ready(term_energy)
            forward_seconds = time.perf_counter() - forward_start
            reverse_start = time.perf_counter()
        term_mf_bar, term_common_bar = pullback(
            np.ones((), dtype=term_energy.dtype)
        )
        energy = energy + term_energy
        if mf_bar is None:
            mf_bar = term_mf_bar
            common_bar = term_common_bar
        else:
            mf_bar = jax.tree_util.tree_map(
                _add_cotangent, mf_bar, term_mf_bar
            )
            common_bar = jax.tree_util.tree_map(
                _add_cotangent, common_bar, term_common_bar
            )
        jax.block_until_ready((energy, mf_bar, common_bar))
        if return_details:
            reverse_seconds = time.perf_counter() - reverse_start
            kind, left_index, right_index = spec
            term_results.append(IAOMP2TermResult(
                kind=str(kind),
                left_fragment=int(left_index),
                right_fragment=(
                    None if int(right_index) == -1 else int(right_index)
                ),
                energy=float(jax.device_get(term_energy)),
                forward_seconds=float(forward_seconds),
                reverse_seconds=float(reverse_seconds),
            ))
        if _restart_is_enabled(restart):
            elapsed_seconds = previous_elapsed
            if return_details:
                elapsed_seconds += time.perf_counter() - total_start
            restart.save_record(
                "correlation_progress",
                scalars={
                    "energy": float(jax.device_get(energy)),
                    "completed_count": int(term_index + 1),
                    "elapsed_seconds": float(elapsed_seconds),
                },
                trees={
                    "mf_bar": mf_bar,
                    "common_bar": common_bar,
                },
                metadata={
                    "completed_specs": [
                        [str(kind), int(left), int(right)]
                        for kind, left, right in work[:term_index + 1]
                    ],
                    "term_results": [
                        _term_result_metadata(item) for item in term_results
                    ] if return_details else [],
                },
            )
        del pullback, term_mf_bar, term_common_bar
        # Saved JAX pullbacks contain reference cycles.  Waiting for the
        # generational collector lets several independent multipole tapes
        # coexist and leaves large native allocator arenas at their high-water
        # mark.  Collection here is cheap relative to an order-four pair VJP.
        gc.collect()

    for term_index, spec in enumerate(
        work[completed_count:], start=completed_count
    ):
        def term(mf_, common_, _spec=spec):
            return _correlation_term_energy(
                mf_, common_, static, _spec
            )

        accumulate_term(term, spec, term_index)
    if mf_bar is None:
        raise ValueError("fixed topology must contain at least one fragment")
    if return_details:
        strong_results = tuple(
            item for item in term_results if item.kind == "strong"
        )
        weak_results = tuple(
            item for item in term_results if item.kind == "weak"
        )
        timing = IAOMP2GradientTiming(
            strong_forward_seconds=sum(
                item.forward_seconds for item in strong_results
            ),
            strong_reverse_seconds=sum(
                item.reverse_seconds for item in strong_results
            ),
            weak_forward_seconds=sum(
                item.forward_seconds for item in weak_results
            ),
            weak_reverse_seconds=sum(
                item.reverse_seconds for item in weak_results
            ),
            total_seconds=(
                previous_elapsed + time.perf_counter() - total_start
            ),
        )
        details = _make_decomposition(
            static, energy, term_results, timing
        )
        return energy, mf_bar, common_bar, details
    return energy, mf_bar, common_bar


def correlation_value_and_grad_from_common(
    mf, common, static, *, return_details=False, restart=None
):
    """Return the local-MP2 energy and open ``mf``/``common`` cotangents.

    This is the shared-localization seam used by DLNO-CCSD(T).  The caller
    owns the VJP that built ``common`` and can therefore accumulate MP2,
    LIS, and coupled-cluster cotangents in the same IAO gauge before closing
    that VJP exactly once.  Each strong ED and unordered weak pair is still
    pulled back separately, so exposing this lower boundary does not recreate
    a grouped reverse-mode tape.
    """
    result = _progressive_correlation_pullback(
        mf,
        common,
        static,
        return_details=return_details,
        restart=restart,
    )
    energy, mf_bar, common_bar = result[:3]
    jax.block_until_ready((energy, mf_bar, common_bar))
    if return_details:
        return energy, mf_bar, common_bar, result[3]
    return energy, mf_bar, common_bar


def correlation_value_and_grad(
    mf, static, *, return_details=False, restart=None
):
    """Return ``(E_corr, mf_bar)`` with progressive fixed-topology AD.

    IAOs and other common continuous arrays are rebuilt once under a saved
    VJP.  Each fragment's strong and weak energies are then evaluated and
    pulled back immediately.  This is the reusable layer for a DLNO-CCSD(T)
    PT correction: the caller can add ``mf_bar`` to its existing SCF
    cotangent and perform only one final CPHF/SCF pullback.  That saved SCF
    pullback must use the standard implicit backend.  The experimental
    first-order replay backend is not valid for this general, nonstationary
    local-orbital cotangent.
    """
    if _restart_is_resuming(restart):
        zero_mf_bar = _zero_mf_cotangent(mf, np.asarray(mf.e_tot).dtype)
        closed = restart.load_record(
            "correlation_closed",
            templates={"mf_bar": zero_mf_bar},
            missing_ok=True,
        )
        del zero_mf_bar
        if closed is not None:
            energy = np.asarray(
                closed.scalars["energy"], dtype=np.asarray(mf.e_tot).dtype
            )
            mf_bar = closed.trees["mf_bar"]
            jax.block_until_ready((energy, mf_bar))
            if return_details:
                details = _details_from_restart_metadata(
                    static, energy, closed.metadata
                )
                return energy, mf_bar, details
            return energy, mf_bar

    collect_details = return_details or _restart_is_enabled(restart)
    common_start = time.perf_counter() if collect_details else None
    common, common_pullback = jax.vjp(
        lambda mf_: rebuild_iao_mp2_common(mf_, static), mf
    )
    if collect_details:
        jax.block_until_ready(common)
        common_forward_seconds = time.perf_counter() - common_start
    result = correlation_value_and_grad_from_common(
        mf,
        common,
        static,
        return_details=collect_details,
        restart=restart,
    )
    energy, mf_bar, common_bar = result[:3]
    common_reverse_start = time.perf_counter() if collect_details else None
    common_mf_bar, = common_pullback(common_bar)
    mf_bar = jax.tree_util.tree_map(
        _add_cotangent, mf_bar, common_mf_bar
    )
    jax.block_until_ready((energy, mf_bar))
    if collect_details:
        common_reverse_seconds = (
            time.perf_counter() - common_reverse_start
        )
        details = result[3]
        details = replace(
            details,
            timing=replace(
                details.timing,
                common_forward_seconds=common_forward_seconds,
                common_reverse_seconds=common_reverse_seconds,
                total_seconds=(
                    details.timing.total_seconds
                    + common_forward_seconds
                    + common_reverse_seconds
                ),
            ),
        )
        if _restart_is_enabled(restart):
            restart.save_record(
                "correlation_closed",
                scalars={"energy": float(jax.device_get(energy))},
                trees={"mf_bar": mf_bar},
                metadata=_details_restart_metadata(details),
            )
        if return_details:
            return energy, mf_bar, details
    return energy, mf_bar


def correlation_value_and_grad_with_iao(
    mf, iao_coeff, static, *, return_details=False
):
    """Return ``(E_corr, mf_bar, iao_bar)`` for an externally built IAO.

    This variant lets a DLNO driver reuse its existing IAO/LO transform and
    accumulate the local-MP2 cotangent into the same saved localization VJP.
    As for :func:`correlation_value_and_grad`, the final SCF response must use
    the standard implicit backend rather than the experimental replay path.
    """
    total_start = time.perf_counter() if return_details else None
    common_start = time.perf_counter() if return_details else None
    common, common_pullback = jax.vjp(
        lambda mf_, iao_: rebuild_iao_mp2_common(
            mf_, static, iao_coeff=iao_
        ),
        mf,
        iao_coeff,
    )
    if return_details:
        jax.block_until_ready(common)
        common_forward_seconds = time.perf_counter() - common_start
    result = _progressive_correlation_pullback(
        mf, common, static, return_details=return_details
    )
    energy, mf_bar, common_bar = result[:3]
    common_reverse_start = time.perf_counter() if return_details else None
    common_mf_bar, iao_bar = common_pullback(common_bar)
    mf_bar = jax.tree_util.tree_map(
        _add_cotangent, mf_bar, common_mf_bar
    )
    jax.block_until_ready((energy, mf_bar, iao_bar))
    if return_details:
        details = result[3]
        details = replace(
            details,
            timing=replace(
                details.timing,
                common_forward_seconds=common_forward_seconds,
                common_reverse_seconds=(
                    time.perf_counter() - common_reverse_start
                ),
                total_seconds=time.perf_counter() - total_start,
            ),
        )
        return energy, mf_bar, iao_bar, details
    return energy, mf_bar, iao_bar
