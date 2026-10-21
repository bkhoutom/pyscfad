"""Global active-HF LIS selected from supplied STC D or W matrices.

The supplied blocks and BoysEDFrame must describe the same native call. The
reference step saves only discrete labels; reconstruction differentiates the
blocks, numerical frame, global orbitals and original Boys target overlaps.
The discarded-complement rotation keeps the historical DLNO stopped gauge.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import jax
import numpy
import scipy.linalg as scipy_linalg
from pyscfad import numpy as np, scipy
from pyscfad.ops import stop_grad
from pyscfad.lno import lno_base
from ._selection import DomainSelections
from .domain import BoysEDFrame, DomainData, build_boys_ed_frame, rebuild_domain_data

LIS_INTERNAL_RANK_THRESHOLD = 1e-6


def _hermitize(matrix):
    return 0.5 * (matrix + matrix.T.conj())


@dataclass(frozen=True)
class FragmentLISSelection:
    """Fixed rank/label choices for one fragment LIS.

    All index arrays refer to ascending Hermitian-eigenvalue order at the
    corresponding reference eigenproblem.  ``internal_*_keep`` index the
    target-row Gram matrix. ``*_lno_keep`` index the external density in the
    global active occupied or virtual manifold, after removing the internal
    target span. A Boys target always retains its single occupied direction
    and has no internal virtual directions.
    """

    fragment_index: int
    internal_occ_keep: numpy.ndarray
    internal_vir_keep: numpy.ndarray
    occupied_lno_keep: numpy.ndarray
    virtual_lno_keep: numpy.ndarray
    full_occupied_space: bool
    full_virtual_space: bool


@dataclass(frozen=True)
class LISSelections:
    """Fixed domain and LIS labels chosen at a reference geometry.

    ``mp2_static`` holds :class:`DomainSelections`; its historical field name
    is part of the checkpoint format. Zero occupied or virtual LNO thresholds
    retain the corresponding full global active-HF space.
    """

    mp2_static: DomainSelections
    matrix_kind: str
    thresh_occ: float
    thresh_vir: float
    internal_rank_threshold: float
    fragments: tuple[FragmentLISSelection, ...]


@dataclass(frozen=True)
class FragmentLIS:
    """One rebuilt fragment LIS and the data needed by an impurity solver.

    All ``*_coeff`` arrays and ``fragment_occupied_anchor`` have global AO
    rows. ``mo_coeff`` has shape ``(nao, nmo)`` and contains six column blocks:
    original frozen occupied, discarded active occupied, LIS occupied, LIS
    virtual, discarded active virtual, original frozen virtual. ``frozen``
    indexes all columns outside the two LIS blocks, as required by
    :mod:`pyscfad.lno.ccsd`. The ``active_*_coeff`` fields contain those two
    semicanonical LIS blocks, of shapes ``(nao, nocc_lis)`` and
    ``(nao, nvir_lis)``.

    ``occupied_projector`` and ``density_occupied_active`` have shape
    ``(nocc_active, nocc_active)`` in the global active-HF occupied coordinates
    of ``STC DomainData``. Their virtual counterparts have shape
    ``(nvir_active, nvir_active)``. ``density_*_ed`` instead use the occupied
    or virtual supplied ED coordinates, of dimensions ``nocc_ed`` and
    ``nvir_ed``. The coefficient arrays inside ``domain`` have local ED AO rows.

    ``fragment_iao_coeff`` is the raw target block, shape ``(nao, ntarget)``;
    the historical name also covers Boys targets. ``fragment_occupied_anchor``
    is its projection into the global active occupied space, with the same
    shape and overlaps with the occupied LIS. It supplies the occupied-space
    weight for the fragment energy partition.
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
        """Raw target block used by the historical fragment interface."""

        return self.fragment_iao_coeff


def _fixed_row_space(matrix, keep):
    """Rebuild ``(nspace, nkeep)`` columns from ``(ntarget, nspace)`` rows.

    ``keep`` contains fixed target-row Gram-eigenvector labels. The returned
    columns are orthonormal in global active occupied or virtual coordinates.
    """

    matrix = np.asarray(matrix)
    keep = numpy.asarray(keep, dtype=numpy.int32)
    if matrix.shape[0] == 0 or keep.size == 0:
        return np.zeros((matrix.shape[1], 0), dtype=matrix.dtype)
    gram = _hermitize(matrix @ matrix.T.conj())
    _, vectors = scipy.linalg.eigh(
        gram, deg_thresh=lno_base.COMPRESS_DEG_THRESH
    )
    candidate = matrix.T.conj() @ vectors[:, keep]
    # The fixed reference rank guarantees a nonsingular retained Gram matrix.
    metric = _hermitize(candidate.T.conj() @ candidate)
    chol = np.linalg.cholesky(metric)
    return np.linalg.solve(chol, candidate.T.conj()).T.conj()


def _internal_projection_matrices(common, fragment_index):
    """Original global Boys target overlaps; Boys has no internal virtual span."""
    target = common.fragment_occupied_data[int(fragment_index)].iao_coeff
    occupied = target.T.conj() @ common.s1e @ common.occupied_coeff
    return occupied, np.zeros((0, common.virtual_coeff.shape[1]), dtype=occupied.dtype)


def _external_density(density, internal):
    """Remove the orthonormal internal span from both density indices."""

    identity = np.eye(density.shape[0], dtype=density.dtype)
    projector = identity - internal @ internal.T.conj()
    return _hermitize(projector @ density @ projector)


def _density_keep_numpy(density, internal, threshold, full_space):
    """Select external-density labels with strict ``abs(eigenvalue) > cutoff``."""

    if full_space:
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
    """Rebuild internal plus retained LNO columns in global active-HF coordinates.

    ``full_space`` returns the whole manifold in its original HF gauge.
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
    lno = np.linalg.solve(chol, lno.T.conj()).T.conj()
    return np.concatenate((internal, lno), axis=1)


def _orthonormalize_complement(space, thresh=1e-10):
    """Keep left singular vectors above the complement rank threshold."""

    space = np.asarray(space)
    if space.ndim != 2:
        raise ValueError('Input space must be a rank-2 array.')
    if space.shape[1] == 0:
        return np.zeros((space.shape[0], 0), dtype=space.dtype)
    vectors, singular_values, _ = scipy.linalg.svd(space, full_matrices=False)
    keep = numpy.where(abs(singular_values) > thresh)[0]
    if len(keep) == 0:
        return np.zeros((space.shape[0], 0), dtype=space.dtype)
    return vectors[:, keep]


def _active_complement(selected, threshold):
    """Complete selected columns in global active-HF coordinates.

    The complement rotation has a frozen gauge: its SVD and projector carry no
    response. Multiplication by the current HF coefficients in the caller
    still carries their response into the discarded AO-space orbitals.
    """

    nspace = int(selected.shape[0])
    if selected.shape[1] == nspace:
        return np.zeros((nspace, 0), dtype=selected.dtype)
    identity = np.eye(nspace, dtype=selected.dtype)
    threshold = max(float(threshold), 1e-8)
    if selected.shape[1] == 0:
        return stop_grad(_orthonormalize_complement(identity, thresh=threshold))
    residual = identity - np.dot(
        stop_grad(selected),
        np.dot(stop_grad(selected.T.conj()), stop_grad(identity)),
    )
    return stop_grad(_orthonormalize_complement(residual, thresh=threshold))


def _semicanonical_space(coeff, fock):
    """Diagonalize the Fock matrix within a possibly empty AO-space block."""

    if coeff.shape[1] == 0:
        return coeff
    return lno_base.semicanonicalize(fock, coeff)[1]


def _assemble_full_mo_layout(
    mf,
    domain_selections,
    occupied_coeff,
    virtual_coeff,
    fock,
    occupied_rotation,
    virtual_rotation,
    rank_threshold,
):
    """Complete the LIS to the occupied-first MO layout of the impurity solver.

    ``occupied_coeff`` and ``virtual_coeff`` have global AO rows and span the
    active-HF manifolds. The corresponding ``*_rotation`` arrays have shapes
    ``(nocc_active, nocc_lis)`` and ``(nvir_active, nvir_lis)``. Only the selected
    LIS blocks are semicanonicalized; the discarded complements use a frozen
    rotation gauge, and originally frozen MOs keep their original columns.
    """

    # Identify original HF columns excluded by the global frozen selection.
    mo_coeff = np.asarray(mf.mo_coeff)
    mo_occ_host = numpy.asarray(jax.device_get(mf.mo_occ))
    nmo = int(mo_occ_host.size)
    all_indices = numpy.arange(nmo, dtype=numpy.int32)
    occupied_indices = all_indices[mo_occ_host > lno_base.THRESH_OCC]
    virtual_indices = all_indices[mo_occ_host <= lno_base.THRESH_OCC]
    active_occ_indices = numpy.asarray(
        domain_selections.active_occ_indices, dtype=numpy.int32
    )
    active_vir_indices = numpy.asarray(
        domain_selections.active_vir_indices, dtype=numpy.int32
    )
    frozen_occ_indices = numpy.setdiff1d(
        occupied_indices, active_occ_indices, assume_unique=False
    )
    frozen_vir_indices = numpy.setdiff1d(
        virtual_indices, active_vir_indices, assume_unique=False
    )

    # Complete each active-HF manifold, then express its LIS in the AO basis.
    occ_complement = _active_complement(
        occupied_rotation, rank_threshold
    )
    vir_complement = _active_complement(
        virtual_rotation, rank_threshold
    )
    occupied_lis_coeff = _semicanonical_space(
        occupied_coeff @ occupied_rotation,
        fock,
    )
    virtual_lis_coeff = _semicanonical_space(
        virtual_coeff @ virtual_rotation,
        fock,
    )
    occupied_discarded_coeff = occupied_coeff @ occ_complement
    virtual_discarded_coeff = virtual_coeff @ vir_complement

    # Pack six blocks: frozen occ | discarded occ | LIS occ | LIS vir |
    # discarded vir | frozen vir. The two LIS blocks are contiguous.
    blocks = (
        mo_coeff[:, frozen_occ_indices],
        occupied_discarded_coeff,
        occupied_lis_coeff,
        virtual_lis_coeff,
        virtual_discarded_coeff,
        mo_coeff[:, frozen_vir_indices],
    )
    full_coeff = np.concatenate(blocks, axis=1)
    nocc_total = int(occupied_indices.size)
    n_frozen_occ = int(frozen_occ_indices.size + occ_complement.shape[1])
    n_active_vir = int(virtual_lis_coeff.shape[1])
    # Freeze the leading occupied and trailing virtual columns; the occupied
    # count is unchanged by rotations within the active-HF manifold.
    frozen = numpy.concatenate((
        numpy.arange(n_frozen_occ, dtype=numpy.int32),
        numpy.arange(
            nocc_total + n_active_vir, nmo, dtype=numpy.int32
        ),
    ))
    return full_coeff, frozen, occupied_lis_coeff, virtual_lis_coeff


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
    if not 0 <= index < len(static.fragments):
        raise IndexError(f'fragment_index={index} is outside [0, {len(static.fragments)})')
    if len(static.frag_lolist[index]) != 1:
        raise ValueError('Boys LIS requires one original Boys target per fragment')
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


def _matrices_in_active_spaces(common, frame, occupied, virtual):
    """Embed local-AO ED blocks using S[:, A_ED], retaining both map responses."""
    overlap_columns = common.s1e[:, frame.extended_ao_indices]
    ro = common.occupied_coeff.T.conj() @ overlap_columns @ frame.occupied_coeff
    rv = common.virtual_coeff.T.conj() @ overlap_columns @ frame.virtual_coeff
    return (_hermitize(ro @ occupied @ ro.T.conj()),
            _hermitize(rv @ virtual @ rv.T.conj()))


def build_fragment_lis_selection(
    mf, mp2_static, fragment_index, *, matrices, matrix_kind, common=None,
    frame=None, thresh_occ=1e-4, thresh_vir=1e-5,
    internal_rank_threshold=LIS_INTERNAL_RANK_THRESHOLD,
):
    """Save target rank and LNO eigenvector labels from one supplied D/W pair.

    This concrete reference step runs outside AD. A zero threshold selects
    the full corresponding global active-HF manifold; otherwise the strict
    absolute eigenvalue cutoff applies, including signed W eigenvalues.
    W examples should always set its occupied and virtual cutoffs explicitly.
    No density or native matrix provider is invoked by this function.
    """
    _validate_kind(matrix_kind)
    _validate_thresholds(thresh_occ, thresh_vir, internal_rank_threshold)
    index = _fragment_index(mp2_static, fragment_index)
    if common is None:
        common = rebuild_domain_data(mf, mp2_static)
    frame, occupied, virtual = _frame_and_matrices(common, mp2_static, index, frame, matrices)
    occupied, virtual = _matrices_in_active_spaces(common, frame, occupied, virtual)
    occ_projection, vir_projection = _internal_projection_matrices(common, index)
    if occ_projection.shape[0] != 1:
        raise ValueError('Boys LIS requires one original Boys target per fragment')
    # Boys always retains its occupied target, independent of the rank cutoff.
    occ_keep = numpy.array([0], dtype=numpy.int32)
    vir_keep = numpy.zeros(0, dtype=numpy.int32)
    occ_internal = _fixed_row_space(occ_projection, occ_keep)
    vir_internal = _fixed_row_space(vir_projection, vir_keep)
    full_occ, full_vir = float(thresh_occ) == 0, float(thresh_vir) == 0
    return FragmentLISSelection(index, occ_keep, vir_keep,
        _density_keep_numpy(occupied, occ_internal, thresh_occ, full_occ),
        _density_keep_numpy(virtual, vir_internal, thresh_vir, full_vir),
        full_occ, full_vir)


def build_fragment_lis(mf, common, static, fragment_index, *, matrices,
                       matrix_kind=None, frame=None):
    """Replay fixed LIS labels with current D/W blocks and the matching ED frame.

    The output contains the original global-HF six-block MO layout, original
    Boys target and its occupied anchor. Blocks, ED maps, target overlaps and
    retained orbitals respond to AD. Only the historical discarded rotation
    is stopped; multiplication by parent global-HF coefficients responds.
    """
    if not isinstance(static, LISSelections):
        raise TypeError('static must be LISSelections')
    kind = _validate_kind(static.matrix_kind)
    if matrix_kind is not None and _validate_kind(matrix_kind) != kind:
        raise ValueError('matrix_kind does not match the saved LIS selections')
    _validate_thresholds(static.thresh_occ, static.thresh_vir, static.internal_rank_threshold)
    index = _fragment_index(static.mp2_static, fragment_index)
    selection = static.fragments[index]
    if selection.fragment_index != index:
        raise ValueError('fragment selection order is inconsistent')
    frame, occupied_ed, virtual_ed = _frame_and_matrices(
        common, static.mp2_static, index, frame, matrices)
    occupied, virtual = _matrices_in_active_spaces(common, frame, occupied_ed, virtual_ed)
    occ_projection, vir_projection = _internal_projection_matrices(common, index)
    occ_internal = _fixed_row_space(occ_projection, selection.internal_occ_keep)
    vir_internal = _fixed_row_space(vir_projection, selection.internal_vir_keep)
    occ_rotation = _fixed_density_space(occupied, occ_internal, selection.occupied_lno_keep,
                                        selection.full_occupied_space)
    vir_rotation = _fixed_density_space(virtual, vir_internal, selection.virtual_lno_keep,
                                        selection.full_virtual_space)
    full_coeff, frozen, occ_coeff, vir_coeff = _assemble_full_mo_layout(
        mf, static.mp2_static, common.occupied_coeff, common.virtual_coeff,
        common.fock, occ_rotation, vir_rotation, static.internal_rank_threshold)
    target = common.fragment_occupied_data[index].iao_coeff
    return FragmentLIS(
        mo_coeff=full_coeff, frozen=frozen,
        fragment_occupied_anchor=common.occupied_coeff @ occ_projection.T.conj(),
        fragment_iao_coeff=target, active_occupied_coeff=occ_coeff,
        active_virtual_coeff=vir_coeff,
        occupied_projector=_hermitize(occ_rotation @ occ_rotation.T.conj()),
        virtual_projector=_hermitize(vir_rotation @ vir_rotation.T.conj()),
        density_occupied_ed=occupied_ed, density_virtual_ed=virtual_ed,
        density_occupied_active=occupied, density_virtual_active=virtual,
        domain=frame, n_internal_occ=int(selection.internal_occ_keep.size),
        n_internal_vir=int(selection.internal_vir_keep.size),
        n_lno_occ=int(occ_rotation.shape[1] - selection.internal_occ_keep.size),
        n_lno_vir=int(vir_rotation.shape[1] - selection.internal_vir_keep.size))
