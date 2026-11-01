"""Concrete domain selections: host reference spaces and retained labels.

Only this module chooses thresholded ranks. Current differentiable arrays are
rebuilt in dlno_base; the reference SVD and selection eigenframes stay distinct.
"""
from __future__ import annotations

from dataclasses import dataclass
import jax
import numpy
import scipy.linalg as scipy_linalg
from pyscf.mp.mp2 import _mo_splitter
from pyscfad.mp import dfmp2
from . import tools

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

    parent_columns: numpy.ndarray
    support_ao_indices: numpy.ndarray
    canonical_keep: numpy.ndarray
    overlap_keep: numpy.ndarray
    completeness_keep: numpy.ndarray
    metric_keep: numpy.ndarray


@dataclass(frozen=True)
class FragmentDomainSelection:
    """Fixed-shape selections for one fragment's strong and weak spaces."""

    fragment_index: int
    iao_indices: numpy.ndarray
    fragment_atoms: numpy.ndarray | None
    strong_fragments: numpy.ndarray
    extended_atoms: numpy.ndarray
    extended_ao_indices: numpy.ndarray
    pao_center_atoms: numpy.ndarray
    strong_occ_union_keep: numpy.ndarray
    strong_occ_metric_keep: numpy.ndarray
    strong_virtual: FixedPAOSubspaceSelection
    primary_atoms: numpy.ndarray
    primary_ao_indices: numpy.ndarray
    primary_bp_atoms: numpy.ndarray
    weak_weight_eigen_indices: numpy.ndarray
    weak_weight_degenerate_blocks: tuple[tuple[int, int], ...]
    weak_occ_norm_keep: numpy.ndarray
    weak_occ_span_metric_keep: numpy.ndarray
    weak_virtual: FixedPAOSubspaceSelection | None

    @property
    def has_weak_screen(self):
        return self.weak_virtual is not None


@dataclass(frozen=True)
class DomainSelections:
    """Fixed domain choices and optional Boys reference data.

    Reference Boys orbitals initialize and label the current solution; they
    never replace the differentiable current orbitals. Historical ``iao``
    field names below also serve Boys targets.
    """

    frozen: object
    thresholds: object
    active_occ_indices: numpy.ndarray
    active_vir_indices: numpy.ndarray
    pao_projected_out_indices: numpy.ndarray
    pao_parent_ao_indices: numpy.ndarray
    ao2pao_map: numpy.ndarray
    frag_lolist: tuple[numpy.ndarray, ...]
    frag_atmlist: tuple[numpy.ndarray | None, ...]
    strong_mask: numpy.ndarray
    fragments: tuple[FragmentDomainSelection, ...]
    lo_type: str = "iao"
    lo_kwargs: dict | None = None
    target_reference_coeff: object = None
    target_reference_coords: object = None


def _host_array(value, dtype=None):
    array = numpy.asarray(jax.device_get(value))
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _hermitian_numpy(matrix):
    matrix = numpy.asarray(matrix)
    return 0.5 * (matrix + matrix.T.conj())


def _eigh_keep_numpy(matrix, threshold):
    eigenvalue, eigenvector = scipy_linalg.eigh(
        _hermitian_numpy(matrix), check_finite=False
    )
    keep = numpy.where(numpy.real(eigenvalue) > threshold)[0].astype(numpy.int32)
    return eigenvalue, eigenvector, keep


def _reference_metric_space(coeff, overlap, threshold):
    """Choose a metric rank and build its canonical frame from one eigensolve."""
    coeff = numpy.asarray(coeff)
    metric = coeff.T.conj() @ numpy.asarray(overlap) @ coeff
    eigenvalue, eigenvector, keep = _eigh_keep_numpy(metric, threshold)
    normalized = coeff @ (
        eigenvector[:, keep] / numpy.sqrt(numpy.real(eigenvalue[keep]))[None, :]
    )
    return normalized, keep


def _pao_columns_on_atoms(mol, atoms, ao2pao_map):
    atoms = numpy.asarray(atoms, dtype=numpy.int32).reshape(-1)
    aoslices = numpy.asarray(mol.aoslice_by_atom())[:, 2:]
    columns = []
    for atom in atoms:
        p0, p1 = map(int, aoslices[int(atom)])
        mapped = numpy.asarray(ao2pao_map[p0:p1], dtype=numpy.int32)
        columns.extend(mapped[mapped >= 0].tolist())
    return numpy.asarray(columns, dtype=numpy.int32)


def _compute_completeness_numpy(candidate, overlap, ao_indices):
    candidate = numpy.asarray(candidate)
    if candidate.shape[1] == 0:
        return numpy.zeros((0,), dtype=float)
    overlap = numpy.asarray(overlap)
    ao_indices = numpy.asarray(ao_indices, dtype=numpy.int32)
    values = overlap[ao_indices] @ candidate
    recovered = numpy.linalg.solve(
        overlap[numpy.ix_(ao_indices, ao_indices)], values
    )
    return numpy.real(numpy.sum(recovered.conj() * values, axis=0))


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

    overlap = numpy.asarray(overlap)
    parent_columns = _pao_columns_on_atoms(
        mol, parent_atoms, ao2pao_map
    )
    pao_parent = numpy.asarray(pao_coeff)[:, parent_columns]

    gram = pao_parent.T.conj() @ overlap @ pao_parent
    gram_e, gram_v, canonical_keep = _eigh_keep_numpy(
        gram, _PAO_ORTH_THRESHOLD
    )
    if canonical_keep.size:
        pao_orth = pao_parent @ (
            gram_v[:, canonical_keep]
            / numpy.sqrt(numpy.real(gram_e[canonical_keep]))[None, :]
        )
    else:
        pao_orth = numpy.zeros((mol.nao, 0), dtype=pao_parent.dtype)

    support_ao = tools.ao_index_by_atom(
        mol, numpy.asarray(support_atoms, dtype=numpy.int32)
    ).astype(numpy.int32, copy=False)
    if pao_orth.shape[1]:
        tmp = overlap[support_ao] @ pao_orth
        local_metric = overlap[numpy.ix_(support_ao, support_ao)]
        projected_overlap = tmp.T.conj() @ numpy.linalg.solve(local_metric, tmp)
        _, overlap_v, overlap_keep = _eigh_keep_numpy(
            projected_overlap, overlap_threshold
        )
        candidate = pao_orth @ overlap_v[:, overlap_keep]
    else:
        overlap_keep = numpy.zeros((0,), dtype=numpy.int32)
        candidate = numpy.zeros((mol.nao, 0), dtype=pao_parent.dtype)

    representation_ao = tools.ao_index_by_atom(
        mol, numpy.asarray(representation_atoms, dtype=numpy.int32)
    ).astype(numpy.int32, copy=False)
    completeness = _compute_completeness_numpy(
        candidate, overlap, representation_ao
    )
    completeness_keep = numpy.where(
        completeness > completeness_threshold
    )[0].astype(numpy.int32)
    candidate = candidate[:, completeness_keep]

    if candidate.shape[1]:
        s21 = overlap[representation_ao]
        s22 = overlap[numpy.ix_(representation_ao, representation_ao)]
        candidate_local = tools.project_mo(candidate, s21, s22)
        candidate_local = tools.orthogonalize(
            occupied_for_projection, candidate_local, s22
        )
        candidate_local, metric_keep = _reference_metric_space(
            candidate_local, s22, metric_threshold
        )
    else:
        metric_keep = numpy.zeros((0,), dtype=numpy.int32)
        candidate_local = numpy.zeros(
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
    eigenvalue, _ = scipy_linalg.eigh(
        _hermitian_numpy(weight), check_finite=False
    )
    order = numpy.argsort(numpy.real(eigenvalue))[::-1]
    retained = order[numpy.real(eigenvalue[order]) > threshold]
    retained = numpy.asarray(retained, dtype=numpy.int32)
    values = numpy.real(eigenvalue[retained])

    blocks = []
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and numpy.isclose(
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
    eigenvalue, eigenvector = scipy_linalg.eigh(
        _hermitian_numpy(occupied_weight), check_finite=False
    )
    values = numpy.real(eigenvalue[retained])
    modes = eigenvector[:, retained]
    output_modes = []
    output_weights = []
    for start, stop in degenerate_blocks:
        block = modes[:, start:stop]
        if stop - start > 1:
            projected_fock = block.T.conj() @ (
                numpy.asarray(occupied_energy)[:, None] * block
            )
            _, rotation = scipy_linalg.eigh(
                _hermitian_numpy(projected_fock), check_finite=False
            )
            block = block @ rotation
        output_modes.append(block)
        output_weights.append(numpy.full(
            stop - start, numpy.mean(values[start:stop]), dtype=float
        ))
    if not output_modes:
        nocc = numpy.asarray(occupied_weight).shape[0]
        return numpy.zeros((0,)), numpy.zeros((nocc, 0))
    return numpy.concatenate(output_weights), numpy.hstack(output_modes)


def _extract_active_indices(mf, frozen):
    pt = dfmp2.MP2(mf, frozen=frozen)
    masks = _mo_splitter(pt)
    active_occ = numpy.where(_host_array(masks[1], bool))[0].astype(numpy.int32)
    active_vir = numpy.where(_host_array(masks[2], bool))[0].astype(numpy.int32)
    projected_out = numpy.where(
        _host_array(masks[0], bool)
        | _host_array(masks[1], bool)
        | _host_array(masks[3], bool)
    )[0].astype(numpy.int32)
    return active_occ, active_vir, projected_out


def _reference_occupied_space(
    topology, fragment_index, s21, s22, fock22, *, selection_frame=False,
):
    """Build a concrete occupied ED and return its two retained-label arrays.

    The energy reference retains its historical thin SVD frame. Selection
    preparation uses the eigenframe of the summed weight, matching smooth
    replay's retained labels. Projection and rank construction are shared.
    """
    partners = topology.strong_fragments[fragment_index]
    thresholds = topology.thresholds
    fragments = topology.fragment_occupied_data
    if topology.lo_type == "boys":
        candidate = numpy.concatenate([
            _host_array(fragments[int(partner)].iao_coeff) for partner in partners
        ], axis=1)
        union_keep = numpy.zeros((0,), dtype=numpy.int32)
    else:
        thin = numpy.vstack([
            _host_array(fragments[int(partner)].iao_occ_overlap) for partner in partners
        ])
        if selection_frame:
            weight = thin.T.conj() @ thin
            values, vectors, union_keep = _eigh_keep_numpy(
                weight, thresholds.occupied_weight
            )
            if union_keep.size == 0:
                union_keep = numpy.asarray([int(numpy.argmax(numpy.real(values)))], dtype=numpy.int32)
        else:
            _, singular, vh = scipy_linalg.svd(thin, full_matrices=False)
            retained = singular**2 > thresholds.occupied_weight
            if not numpy.any(retained):
                retained[numpy.argmax(singular)] = True
            union_keep = numpy.flatnonzero(retained).astype(numpy.int32)
            vectors = vh.conj().T
        candidate = _host_array(topology.occupied_coeff) @ vectors[:, union_keep]
    projected = tools.project_mo(candidate, s21, s22)
    if selection_frame:
        occupied, metric_keep = _reference_metric_space(
            projected, s22, thresholds.metric_rank
        )
    else:
        # Keep the eager reference's finite-input check and canonical frame.
        metric = projected.conj().T @ numpy.asarray(s22) @ projected
        metric = 0.5 * (metric + metric.conj().T)
        values, vectors = scipy_linalg.eigh(metric)
        metric_keep = numpy.flatnonzero(values > thresholds.metric_rank).astype(numpy.int32)
        occupied = projected @ (vectors[:, metric_keep] / numpy.sqrt(values[metric_keep])[None, :])
    projected_fock = occupied.T.conj() @ numpy.asarray(fock22) @ occupied
    if selection_frame and occupied.shape[1] <= 1:
        return numpy.real(numpy.diag(projected_fock)), occupied, union_keep, metric_keep
    projected_fock = 0.5 * (projected_fock + projected_fock.T.conj())
    energy, rotation = scipy_linalg.eigh(projected_fock, check_finite=not selection_frame)
    return numpy.asarray(energy.real), occupied @ rotation, union_keep, metric_keep


def build_domain_selections(mf, topology):
    """Extract fixed ranks and index lists from a reference topology.

    Parameters
    ----------
    mf
        The concrete reference SCF object used to build ``topology``.
    topology : :class:`~pyscfad.dlno.domain.DomainTopology`
        Energy-validation topology at the same geometry.

    Returns
    -------
    :class:`DomainSelections`
        Discrete selections plus an optional Boys reference frame used only
        for initialization and label matching, never as current orbitals.
    """

    from .domain import DomainTopology

    if not isinstance(topology, DomainTopology):
        raise TypeError("topology must be an DomainTopology")
    if mf.mol.nao != topology.s1e.shape[0]:
        raise ValueError("mf and topology use different AO spaces")

    thresholds = topology.thresholds
    active_occ, active_vir, projected_out = _extract_active_indices(
        mf, topology.frozen
    )
    reference_mo = _host_array(mf.mo_coeff)
    if not numpy.allclose(
        reference_mo[:, active_occ], _host_array(topology.occupied_coeff),
        atol=1e-9, rtol=1e-9,
    ):
        raise ValueError("active occupied MO indices do not match topology")
    if not numpy.allclose(
        reference_mo[:, active_vir], _host_array(topology.virtual_coeff),
        atol=1e-9, rtol=1e-9,
    ):
        raise ValueError("active virtual MO indices do not match topology")

    overlap = _host_array(topology.s1e)
    fock = _host_array(topology.fock)
    occupied = _host_array(topology.occupied_coeff)
    occupied_energy = _host_array(topology.occupied_energy)
    pao_coeff = _host_array(topology.pao_coeff)
    ao2pao_map = _host_array(topology.ao2pao_map, numpy.int32)
    pao_parent_ao = numpy.where(ao2pao_map >= 0)[0].astype(numpy.int32)
    if not numpy.array_equal(
        ao2pao_map[pao_parent_ao], numpy.arange(pao_parent_ao.size)
    ):
        raise ValueError("topology PAO map is not in parent-AO order")

    frag_lolist = tuple(
        _host_array(indices, numpy.int32).reshape(-1)
        for indices in topology.frag_lolist
    )
    frag_atmlist = tuple(
        None if atoms is None else _host_array(atoms, numpy.int32).reshape(-1)
        for atoms in topology.frag_atmlist
    )
    fragments = []

    for fragment_index, iao_indices in enumerate(frag_lolist):
        partners = _host_array(
            topology.strong_fragments[fragment_index], numpy.int32
        ).reshape(-1)
        extended_atoms = _host_array(
            topology.extended_domain[fragment_index], numpy.int32
        ).reshape(-1)
        center_atoms = _host_array(
            topology.pao_center_domain[fragment_index], numpy.int32
        ).reshape(-1)
        extended_ao = tools.ao_index_by_atom(
            mf.mol, extended_atoms
        ).astype(numpy.int32, copy=False)
        s21 = overlap[extended_ao]
        s22 = overlap[numpy.ix_(extended_ao, extended_ao)]
        fock22 = fock[numpy.ix_(extended_ao, extended_ao)]

        _, occupied_local, occ_union_keep, occ_metric_keep = _reference_occupied_space(
            topology, fragment_index, s21, s22, fock22, selection_frame=True
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
            topology.primary_domain[fragment_index], numpy.int32
        ).reshape(-1)
        primary_bp_atoms = _host_array(
            topology.primary_bp_domain[fragment_index], numpy.int32
        ).reshape(-1)
        primary_ao = tools.ao_index_by_atom(
            mf.mol, primary_atoms
        ).astype(numpy.int32, copy=False)
        primary_s21 = overlap[primary_ao]
        primary_s22 = overlap[numpy.ix_(primary_ao, primary_ao)]

        if topology.lo_type == "boys":
            weight_indices = numpy.zeros((0,), dtype=numpy.int32)
            weight_blocks = ()
            weak_occupied_global = _host_array(
                topology.fragment_occupied_data[fragment_index].iao_coeff
            )
        else:
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
        weak_occupied_raw = tools.project_mo(
            weak_occupied_global, primary_s21, primary_s22
        )
        norm2 = numpy.real(numpy.einsum(
            "ui,uv,vi->i",
            weak_occupied_raw.conj(),
            primary_s22,
            weak_occupied_raw,
            optimize=True,
        ))
        weak_norm_keep = numpy.where(
            norm2 > thresholds.metric_rank
        )[0].astype(numpy.int32)
        if weak_norm_keep.size:
            weak_occupied = (
                weak_occupied_raw[:, weak_norm_keep]
                / numpy.sqrt(norm2[weak_norm_keep])[None, :]
            )
            weak_occ_span, weak_occ_span_keep = _reference_metric_space(
                weak_occupied, primary_s22, thresholds.metric_rank
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
            weak_occ_span_keep = numpy.zeros((0,), dtype=numpy.int32)
            weak_virtual = None

        fragments.append(FragmentDomainSelection(
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

    return DomainSelections(
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
        lo_type=topology.lo_type,
        lo_kwargs=topology.lo_kwargs,
        target_reference_coeff=(
            _host_array(topology.iao_coeff) if topology.lo_type == "boys" else None
        ),
        target_reference_coords=(
            _host_array(mf.mol.atom_coords()) if topology.lo_type == "boys" else None
        ),
    )


