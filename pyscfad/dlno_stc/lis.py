"""LIS selected through extended-domain occupied and virtual eigenproblems.

The supplied blocks and BoysEDFrame must describe the same native call. The
reference step saves only discrete labels; reconstruction differentiates the
blocks, numerical ED frame and original Boys target overlaps. Every LIS
orbital stays within its occupied or virtual ED parent span.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import jax
from jax.scipy.linalg import solve_triangular
import numpy
import scipy.linalg as scipy_linalg
from pyscfad import numpy as np, scipy
from pyscfad.lno import lno_base
from ._selection import DomainSelections, _physical_groups
from .domain import BoysEDFrame, DomainData, build_boys_ed_frame, rebuild_domain_data

LIS_INTERNAL_RANK_THRESHOLD = 1e-6


def _hermitize(matrix):
    return 0.5 * (matrix + matrix.T.conj())


@dataclass(frozen=True)
class FragmentLISSelection:
    """Fixed rank/label choices for one fragment LIS.

    All index arrays refer to ascending Hermitian-eigenvalue order at the
    corresponding reference eigenproblem. ``internal_occ_keep`` enumerates
    the retained projected Boys targets. ``*_lno_keep`` index the density after
    removing the internal target span. Both LNO index arrays use compact
    extended-domain coordinates. ``occupied_map_columns`` selects a fixed
    independent column basis after removing the occupied target span.
    A physical Boys fragment retains every original occupied target direction
    and has no internal virtual directions.
    """

    fragment_index: int
    internal_occ_keep: numpy.ndarray
    internal_vir_keep: numpy.ndarray
    occupied_lno_keep: numpy.ndarray
    virtual_lno_keep: numpy.ndarray
    full_occupied_space: bool
    full_virtual_space: bool
    occupied_map_columns: numpy.ndarray
    lis_space: str = 'domain'
    parent_dimensions: tuple | None = None


@dataclass(frozen=True)
class LISSelections:
    """Fixed domain and LIS labels chosen at a reference geometry.

    ``mp2_static`` holds :class:`DomainSelections`; its historical field name
    is part of the checkpoint format. Zero occupied or virtual LNO thresholds
    retain the full corresponding ED parent space. ``lis_space`` records the
    domain-only definition for checkpoint validation.
    ``smooth_frames`` and ``smooth_grid``, when present, are tuples indexed
    by physical fragment. Empty or unselected fragments have None entries.
    """

    mp2_static: DomainSelections
    matrix_kind: str
    thresh_occ: float
    thresh_vir: float
    internal_rank_threshold: float
    fragments: tuple[FragmentLISSelection, ...]
    smooth_frames: object = None
    smooth_grid: object = None
    lis_space: str = 'domain'


@dataclass(frozen=True)
class FragmentLIS:
    """One rebuilt fragment LIS and the data needed by an impurity solver.

    All ``*_coeff`` arrays have global AO rows, with zeros outside the ED.
    ``mo_coeff`` contains LIS occupied then virtual columns; ``frozen`` is
    empty. The solver receives the explicit occupied LIS count. Projectors
    and D/W blocks use the orthonormal ED occupied/virtual coordinates.
    ``density_*_active`` are None; no global active-space density is formed.
    Both target fields contain the normalized domain projection of the
    physical Boys block used for the fragment energy partition.
    """

    mo_coeff: object
    frozen: numpy.ndarray
    fragment_occupied_anchor: object
    fragment_iao_coeff: object
    active_occupied_coeff: object
    active_virtual_coeff: object
    occupied_projector: object
    virtual_projector: object
    density_occupied_ed: object
    density_virtual_ed: object
    density_occupied_active: object
    density_virtual_active: object
    domain: BoysEDFrame
    n_internal_occ: int
    n_internal_vir: int
    n_lno_occ: int
    n_lno_vir: int

    @property
    def orbfrag(self):
        """Compatibility alias for the complete impurity MO layout."""

        return self.mo_coeff

    @property
    def frzfrag(self):
        """Compatibility alias for the impurity frozen indices."""

        return self.frozen

    @property
    def orbfragloc(self):
        """Normalized domain target block used by the fragment interface."""

        return self.fragment_iao_coeff


def _physical_target_block(common, static, fragment_index):
    members = _physical_groups(static)[fragment_index]
    return np.concatenate(tuple(common.fragment_occupied_data[member].iao_coeff
                                for member in members), axis=1)


def _external_density(density, internal):
    """Remove the orthonormal internal span from both density indices."""

    if not internal.shape[1]:
        return _hermitize(density)
    identity = np.eye(density.shape[0], dtype=density.dtype)
    projector = identity - internal @ internal.T.conj()
    return _hermitize(projector @ density @ projector)


def _density_keep_numpy(density, internal, threshold, full_space):
    """Select external-density labels with strict ``abs(eigenvalue) > cutoff``."""

    if full_space or density.shape[0] == 0:
        return numpy.zeros((0,), dtype=numpy.int32)
    external = numpy.asarray(jax.device_get(_external_density(density, internal)))
    values = scipy_linalg.eigh(
        0.5 * (external + external.T.conj()),
        eigvals_only=True,
        check_finite=False,
    )
    return numpy.where(numpy.abs(numpy.real(values)) > float(threshold))[0].astype(
        numpy.int32
    )


def _fixed_density_space(density, internal, keep, full_space):
    """Rebuild internal plus retained LNO columns in ED coordinates.

    ``full_space`` returns the whole ED manifold.
    """

    nspace = int(density.shape[0])
    if full_space:
        return np.eye(nspace, dtype=density.dtype)
    keep = numpy.asarray(keep, dtype=numpy.int32)
    if keep.size == 0:
        return internal
    external = _external_density(density, internal)
    _, vectors = scipy.linalg.eigh(
        external, deg_thresh=lno_base.COMPRESS_DEG_THRESH
    )
    lno = vectors[:, keep]
    if internal.shape[1]:
        lno = lno - internal @ (internal.T.conj() @ lno)
    # Numerical projection against the internal space changes only roundoff,
    # but Cholesky normalization makes the rebuilt span explicitly orthonormal.
    metric = _hermitize(lno.T.conj() @ lno)
    chol = np.linalg.cholesky(metric)
    lno = solve_triangular(chol, lno.T.conj(), lower=True).T.conj()
    return np.concatenate((internal, lno), axis=1)


def _semicanonical_space(coeff, fock):
    """Diagonalize the Fock matrix within a possibly empty AO-space block."""

    if coeff.shape[1] == 0:
        return coeff
    return lno_base.semicanonicalize(fock, coeff)[1]


def _validate_kind(matrix_kind):
    if matrix_kind not in ('D', 'W'):
        raise ValueError("matrix_kind must be 'D' or 'W'")
    return matrix_kind


def _validate_thresholds(thresh_occ, thresh_vir, internal_rank_threshold):
    for name, value in (('thresh_occ', thresh_occ), ('thresh_vir', thresh_vir),
                        ('internal_rank_threshold', internal_rank_threshold)):
        if not numpy.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f'{name} must be finite and non-negative')


def _fragment_index(static, fragment_index):
    if not isinstance(static, DomainSelections) or static.lo_type != 'boys':
        raise TypeError('mp2_static must be STC Boys DomainSelections')
    index = int(fragment_index)
    physical = _physical_groups(static)
    if not 0 <= index < len(physical):
        raise IndexError(f'fragment_index={index} is outside [0, {len(physical)})')
    if any(len(static.frag_lolist[member]) != 1 for member in physical[index]):
        raise ValueError('Boys LIS requires one original Boys target per orbital record')
    return index


def _frame_and_matrices(common, static, index, frame, matrices):
    if not isinstance(common, DomainData):
        raise TypeError('common must be STC DomainData')
    if frame is None:
        frame = build_boys_ed_frame(common, static, index)
    if not isinstance(frame, BoysEDFrame):
        raise TypeError('frame must be BoysEDFrame')
    selection = static.fragments[index]
    if (not numpy.array_equal(frame.extended_ao_indices, selection.extended_ao_indices)
            or not numpy.array_equal(frame.extended_atoms, selection.extended_atoms)
            or frame.target_index != selection.target_index):
        raise ValueError('frame support/target does not match the selected fragment')
    ao = numpy.asarray(frame.extended_ao_indices)
    if (ao.ndim != 1 or ao.dtype.kind not in 'iu' or len(numpy.unique(ao)) != len(ao)
            or numpy.any(ao < 0) or numpy.any(ao >= common.s1e.shape[0])):
        raise ValueError('frame extended_ao_indices must be distinct valid AO indices')
    if not isinstance(matrices, Mapping):
        raise TypeError('matrices must supply occupied and virtual blocks')
    blocks = []
    for name, coeff in (('occupied', frame.occupied_coeff), ('virtual', frame.virtual_coeff)):
        if coeff.ndim != 2 or coeff.shape[0] != len(ao):
            raise ValueError(f'frame {name} coefficients must have local ED AO rows')
        if name not in matrices:
            raise ValueError(f'matrices must supply the {name} block')
        block = np.asarray(matrices[name])
        nspace = coeff.shape[1]
        if block.shape != (nspace, nspace):
            raise ValueError(f'matrices {name} block must have shape {(nspace, nspace)}')
        blocks.append(block)
    return frame, blocks[0], blocks[1]


def _validate_lis_space(lis_space):
    if lis_space != 'domain':
        raise ValueError("Only lis_space='domain' is supported; rebuild domain-space LIS selections")
    return lis_space


def _domain_targets(common, domains, index, frame):
    """Coordinates of the physical targets projected into the occupied ED.

    Use the projected original Boys block, not orthogonalized ED column
    labels: Cholesky orthogonalization can mix a target with its partners.
    """
    target = _physical_target_block(common, domains, index)
    cross = common.s1e[frame.extended_ao_indices] @ target
    coordinates = frame.occupied_coeff.T.conj() @ cross
    from .smooth import orthonormalize
    return orthonormalize(coordinates, np.eye(coordinates.shape[0]))


def _build_domain_lis(mf, common, static, index, frame, occupied, virtual):
    """Select and frame LIS orbitals entirely within the orthonormal ED.

    Global AO rows are a zero-padded representation only. There is no global
    HF occupied/virtual projection or completion with frozen HF orbitals.
    The impurity solver receives an explicit local occupied count instead.
    """
    selection = static.fragments[index]
    internal = _domain_targets(common, static.mp2_static, index, frame)
    frames = None if static.smooth_frames is None else static.smooth_frames[index]
    parents, rotations, coefficients = [], [], []
    for spin, (parent, density, target, keep, full) in enumerate((
            (frame.occupied_coeff, occupied, internal, selection.occupied_lno_keep,
             selection.full_occupied_space),
            (frame.virtual_coeff, virtual, np.zeros((virtual.shape[0], 0)),
             selection.virtual_lno_keep, selection.full_virtual_space))):
        embedded = np.zeros((common.s1e.shape[0], parent.shape[1]), dtype=parent.dtype)
        embedded = embedded.at[frame.extended_ao_indices].set(parent)
        if full:
            rotation = np.eye(parent.shape[1], dtype=parent.dtype)
        else:
            mapping = np.eye(parent.shape[1], dtype=parent.dtype)
            if spin == 0:
                # Work in the actual target complement. A projected n_ED by
                # n_ED density contains numerical zero modes along the target,
                # which tiny positive cutoffs must never select as extra LNOs.
                mapping, density = _compact_mapped_density(
                    _external_map(mapping, target), density, selection.occupied_map_columns)
            rotation = _mapped_density_space(density, mapping, target, keep,
                embedded, common.s1e, None if frames is None else frames[spin])
        coeff = embedded @ rotation
        if frames is None:
            coeff = _semicanonical_space(coeff, common.fock)
        else:
            from .smooth import rebuild_frame
            coeff = rebuild_frame(coeff, common.s1e, frames[spin])
        parents.append(embedded)
        rotations.append(rotation)
        coefficients.append(coeff)
    anchor = parents[0] @ internal
    co, cv = coefficients
    return FragmentLIS(
        mo_coeff=np.concatenate((co, cv), axis=1), frozen=numpy.zeros(0, dtype=numpy.int32),
        fragment_occupied_anchor=anchor, fragment_iao_coeff=anchor,
        active_occupied_coeff=co, active_virtual_coeff=cv,
        occupied_projector=_hermitize(rotations[0] @ rotations[0].T.conj()),
        virtual_projector=_hermitize(rotations[1] @ rotations[1].T.conj()),
        density_occupied_ed=occupied, density_virtual_ed=virtual,
        density_occupied_active=None, density_virtual_active=None,
        domain=frame, n_internal_occ=internal.shape[1], n_internal_vir=0,
        n_lno_occ=co.shape[1]-internal.shape[1], n_lno_vir=cv.shape[1])


def _compact_mapped_density(mapping, density, columns=None):
    """Represent R W R^H in the domain-sized orthonormal range of R.

    R need not be isometric: for R^H R = L L^H, Q = R L^-H and
    K = L^H W L give Q K Q^H = R W R^H. Both the metric and Q remain
    differentiable. Dropping either response changes the molecular force.
    For a rank-deficient map, fixed independent ``columns`` span its range;
    all columns still contribute to K through Q^H R. Only the range basis
    construction uses the selected columns.
    """
    if columns is None:
        low = np.linalg.cholesky(_hermitize(mapping.T.conj() @ mapping))
        basis = solve_triangular(low, mapping.T.conj(), lower=True).T.conj()
        coordinates = low.T.conj()
    else:
        selected = mapping[:, columns]
        if not selected.shape[1]:
            return selected, np.zeros((0, 0), dtype=density.dtype)
        # Avoid squaring the condition number of the projected occupied map.
        basis, _ = np.linalg.qr(selected, mode='reduced')
        coordinates = basis.T.conj() @ mapping
    compact = _hermitize(coordinates @ density @ coordinates.T.conj())
    return basis, compact


def _range_columns(mapping):
    """Save independent columns at the reference, discarding roundoff nulls."""
    host = numpy.asarray(jax.device_get(mapping))
    if not host.size:
        return numpy.empty(0, dtype=numpy.int32)
    _, triangular, pivots = scipy_linalg.qr(host, mode='economic', pivoting=True)
    tolerance = numpy.finfo(host.real.dtype).eps * max(host.shape) * max(1., numpy.linalg.norm(host))
    rank = numpy.count_nonzero(numpy.abs(numpy.diag(triangular)) > tolerance)
    return numpy.asarray(pivots[:rank], dtype=numpy.int32)


def _external_map(mapping, internal):
    return mapping - internal @ (internal.T.conj() @ mapping)


def _mapped_density_space(density, mapping, internal, keep, coeff, overlap, anchor):
    """Rebuild the target plus retained domain span, including mapping response."""
    empty = np.zeros((density.shape[0], 0), dtype=density.dtype)
    if anchor is None:
        external = _fixed_density_space(density, empty, keep, False)
        return np.concatenate((internal, mapping @ external), axis=1)
    from ._retained_space import retained_projector
    projected_anchor = coeff.T.conj() @ overlap @ np.asarray(anchor.anchor)
    candidate = internal @ (internal.T.conj() @ projected_anchor)
    if len(keep):
        retained = retained_projector(tuple(int(i) for i in keep))(density)
        candidate = candidate + mapping @ (retained @ (mapping.T.conj() @ projected_anchor))
    # These are orthonormal ED coordinates; avoid a molecular-sized
    # identity matrix and its multiplication during anchor normalization.
    metric = _hermitize(candidate.T.conj() @ candidate)
    low = np.linalg.cholesky(metric)
    return solve_triangular(low, candidate.T.conj(), lower=True).T.conj()


def build_fragment_lis_selection(
    mf, mp2_static, fragment_index, *, matrices, matrix_kind, common=None,
    frame=None, frame_fragment_index=None, thresh_occ=1e-4, thresh_vir=1e-5,
    internal_rank_threshold=LIS_INTERNAL_RANK_THRESHOLD, lis_space='domain',
):
    """Save target rank and LNO eigenvector labels from one supplied D/W pair.

    This concrete reference step runs outside AD. A zero threshold selects
    the full corresponding ED parent manifold; otherwise the strict
    absolute eigenvalue cutoff applies, including signed W eigenvalues.
    W examples should always set its occupied and virtual cutoffs explicitly.
    No density or native matrix provider is invoked by this function.
    """
    _validate_kind(matrix_kind)
    _validate_lis_space(lis_space)
    _validate_thresholds(thresh_occ, thresh_vir, internal_rank_threshold)
    index = _fragment_index(mp2_static, fragment_index)
    if common is None:
        common = rebuild_domain_data(mf, mp2_static)
    representative = (_physical_groups(mp2_static)[index][0] if frame_fragment_index is None
                      else frame_fragment_index)
    frame, occupied, virtual = _frame_and_matrices(common, mp2_static, representative, frame, matrices)
    internal = _domain_targets(common, mp2_static, index, frame)
    full_occ, full_vir = float(thresh_occ) == 0, float(thresh_vir) == 0
    columns = numpy.zeros(0, dtype=numpy.int32)
    occ_keep = numpy.zeros(0, dtype=numpy.int32)
    if not full_occ:
        external = _external_map(np.eye(occupied.shape[0]), internal)
        columns = _range_columns(external)
        _, compact = _compact_mapped_density(external, occupied, columns)
        occ_keep = _density_keep_numpy(compact, np.zeros((compact.shape[0], 0)),
                                       thresh_occ, False)
    vir_keep = _density_keep_numpy(virtual, np.zeros((virtual.shape[0], 0)),
                                   thresh_vir, full_vir)
    return FragmentLISSelection(index, numpy.arange(internal.shape[1], dtype=numpy.int32),
        numpy.zeros(0, dtype=numpy.int32), occ_keep, vir_keep, full_occ, full_vir,
        columns, lis_space='domain',
        parent_dimensions=(occupied.shape[0], virtual.shape[0]))


def build_fragment_lis(mf, common, static, fragment_index, *, matrices,
                       matrix_kind=None, frame=None, frame_fragment_index=None):
    """Replay fixed LIS labels with current D/W blocks and the matching ED frame.

    The output contains only the domain LIS and its normalized target block.
    D/W blocks, ED orbitals, target overlaps and retained orbitals respond to AD.
    """
    if not isinstance(static, LISSelections):
        raise TypeError('static must be LISSelections')
    kind = _validate_kind(static.matrix_kind)
    _validate_lis_space(static.lis_space)
    if matrix_kind is not None and _validate_kind(matrix_kind) != kind:
        raise ValueError('matrix_kind does not match the saved LIS selections')
    _validate_thresholds(static.thresh_occ, static.thresh_vir, static.internal_rank_threshold)
    index = _fragment_index(static.mp2_static, fragment_index)
    selection = static.fragments[index]
    if 'occupied_map_columns' not in vars(selection):
        raise ValueError('obsolete LIS selection; rebuild domain-space selections')
    if selection.fragment_index != index:
        raise ValueError('fragment selection order is inconsistent')
    _validate_lis_space(selection.lis_space)
    representative = (_physical_groups(static.mp2_static)[index][0] if frame_fragment_index is None
                      else frame_fragment_index)
    frame, occupied_ed, virtual_ed = _frame_and_matrices(
        common, static.mp2_static, representative, frame, matrices)
    return _build_domain_lis(mf, common, static, index, frame, occupied_ed, virtual_ed)
