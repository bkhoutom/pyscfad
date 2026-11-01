"""Global-HF local interacting spaces selected from domain MP2 densities.

At the reference geometry, target projections set the internal ranks and the
external MP2 densities select LNO eigenvector labels. Rebuilds hold those labels
fixed while differentiating the orbitals, densities, and semicanonical LIS.
Density diagonalization and LIS selection use global active-HF coordinates;
the impurity solver receives a complete MO layout in the global AO basis.

Public ``common``, ``static``, and ``mp2_static`` keywords are retained for
compatibility. Local names distinguish current ``domain_data`` from fixed
``domain_selections`` and ``lis_selections``.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import tempfile
import jax
import numpy
import scipy.linalg as scipy_linalg
from pyscf import lib as pyscf_lib
from pyscfad import numpy as np, scipy
from pyscfad.df import addons as df_addons
from pyscfad.ops import stop_grad
from pyscfad.lno import lno_base, df as lno_df
from . import _profile
from ._selection import DomainSelections
from .dlno_base import (
    DomainData, StrongDomain, build_strong_ed_domain, rebuild_domain_data,
)
from .mp2_rdm import (
    _hermitize,
    _resolve_mp2_density_block_nvir, _strong_domain_mp2_density_h5,
)

LIS_INTERNAL_RANK_THRESHOLD = 1e-6


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
    of ``DomainData``. Their virtual counterparts have shape
    ``(nvir_active, nvir_active)``. ``density_*_ed`` instead use the occupied
    or virtual semicanonical ED coordinates, of dimensions ``nocc_ed`` and
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
    domain: StrongDomain
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


def strong_domain_mp2_density(
    mf,
    domain,
    static,
    fragment_index,
    *,
    lov_scratch_dir,
):
    """Return target-conditioned MP2 densities in strong-ED MO coordinates.

    ``static`` supplies :class:`DomainSelections`. ``domain`` supplies local
    ED AO coefficients and semicanonical energies; the returned occupied and
    virtual density blocks have shapes ``(nocc_ed, nocc_ed)`` and
    ``(nvir_ed, nvir_ed)``.

    The caller owns ``lov_scratch_dir`` and must keep its ``local_lov.h5``
    intact until the forward calculation and any reverse pass finish. Each
    concurrently live fragment calculation needs a separate directory.
    Virtual blocking uses ``static.thresholds.mp2_block_nvir`` or
    ``mp2_block_memory_mb`` when configured, otherwise an automatic workspace
    target. These options tune temporary workspace, not total process memory.
    """

    domain_selections = static
    if not isinstance(domain_selections, DomainSelections):
        raise TypeError("static must be DomainSelections")
    if not isinstance(domain, StrongDomain):
        raise TypeError("domain must be StrongDomain")
    fragment_index = int(fragment_index)
    fragment_selection = domain_selections.fragments[fragment_index]
    nocc = int(domain.occupied_coeff.shape[1])
    local_coeff = np.concatenate(
        (domain.occupied_coeff, domain.virtual_coeff), axis=1
    )
    nvir = int(domain.virtual_coeff.shape[1])
    if lov_scratch_dir is None:
        raise ValueError(
            "lov_scratch_dir is required when building the MP2 density"
        )
    lov_scratch_dir = os.fspath(lov_scratch_dir)
    if not os.path.isdir(lov_scratch_dir):
        raise FileNotFoundError(
            f"Lov scratch directory does not exist: {lov_scratch_dir}"
        )

    # Match the integral AO rows to the fixed extended-domain atom order.
    local_mol = lno_df.make_local_mol(
        mf.mol, fragment_selection.extended_atoms
    )
    auxmol = df_addons.make_auxmol(local_mol, mf.with_df.auxbasis)
    naux = int(auxmol.nao)
    ntarget = int(domain.target_projection.shape[0])

    block_nvir, block_mode, workspace_target_mb = (
        _resolve_mp2_density_block_nvir(
            naux=naux,
            nocc=nocc,
            nvir=nvir,
            ntarget=ntarget,
            dtype=local_coeff.dtype,
            mf_max_memory_mb=getattr(mf, "max_memory", 256.0),
            configured_memory_mb=domain_selections.thresholds.mp2_block_memory_mb,
            configured_block_nvir=domain_selections.thresholds.mp2_block_nvir,
        )
    )
    h5_path = os.path.join(lov_scratch_dir, "local_lov.h5")
    local_max_memory = getattr(
        mf.with_df, "max_memory", getattr(mf, "max_memory", 256.0)
    )
    # Keep the custom-VJP order: six differentiable inputs, then four fixed
    # controls (nocc, scratch path, DF memory, virtual block width).
    density_args = (
        local_mol, auxmol, local_coeff,
        domain.occupied_energy, domain.virtual_energy, domain.target_projection,
        nocc, h5_path, local_max_memory, block_nvir,
    )
    return _profile.density_call(
        _strong_domain_mp2_density_h5, density_args,
        fragment_index=fragment_index, coeff=local_coeff, naux=naux,
        nocc=nocc, nvir=nvir, ntarget=ntarget, block_nvir=block_nvir,
        block_mode=block_mode, workspace_target_mb=workspace_target_mb,
        h5_path=h5_path,
    )


def _row_gram_keep_numpy(matrix, threshold):
    """Choose ascending Gram-eigenvector labels from a host-side SVD rank."""

    matrix = numpy.asarray(jax.device_get(matrix))
    if matrix.ndim != 2:
        raise ValueError("fragment projection must be rank two")
    if matrix.shape[0] == 0:
        return numpy.zeros((0,), dtype=numpy.int32)
    # Determine numerical rank from singular values, rather than comparing
    # eigenvalues of M M^H with threshold**2.  Exact null eigenvalues of the
    # Gram matrix acquire O(eps) roundoff, which is much larger than a typical
    # squared singular-value cutoff (e.g. 1e-20 for THRESH_INTERNAL=1e-10).
    # The differentiable rebuild below still uses the equivalent Hermitian
    # eigenproblem; its ascending retained labels are simply the final `rank`
    # columns.
    singular_values = scipy_linalg.svdvals(matrix, check_finite=False)
    rank = int(numpy.count_nonzero(singular_values > float(threshold)))
    return numpy.arange(
        matrix.shape[0] - rank, matrix.shape[0], dtype=numpy.int32
    )


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


def _domain_density_in_active_spaces(
    domain_data, domain_selections, fragment_index, domain, density,
):
    """Embed ED densities in the global active-HF occupied/virtual bases."""

    fragment_selection = domain_selections.fragments[int(fragment_index)]
    ao_indices = fragment_selection.extended_ao_indices
    overlap_to_domain = domain_data.s1e[:, ao_indices]
    # C_active^H S[:, ED] C_ED maps ED MOs into global active-HF coordinates.
    occupied_map = (
        domain_data.occupied_coeff.T.conj()
        @ overlap_to_domain
        @ domain.occupied_coeff
    )
    virtual_map = (
        domain_data.virtual_coeff.T.conj()
        @ overlap_to_domain
        @ domain.virtual_coeff
    )
    dmoo = occupied_map @ density.occupied @ occupied_map.T.conj()
    dmvv = virtual_map @ density.virtual @ virtual_map.T.conj()
    return _hermitize(dmoo), _hermitize(dmvv)


def _internal_projection_matrices(domain_data, fragment_index, *, lo_type="iao"):
    """Target overlaps with global active-HF occupied and virtual orbitals."""

    target_data = domain_data.fragment_occupied_data[int(fragment_index)]
    occ_projection = target_data.iao_occ_overlap
    if lo_type == "boys":
        # Occupied Boys targets cannot acquire internal virtual directions
        # from numerical occupied/virtual overlap leakage.
        return occ_projection, np.zeros(
            (0, domain_data.virtual_coeff.shape[1]), dtype=occ_projection.dtype
        )
    vir_projection = (
        target_data.iao_coeff.T.conj() @ domain_data.s1e @ domain_data.virtual_coeff
    )
    return occ_projection, vir_projection


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


def _reference_fragment_selection(
    domain_data,
    domain_selections,
    fragment_index,
    density_occupied_active,
    density_virtual_active,
    *,
    thresh_occ,
    thresh_vir,
    internal_rank_threshold,
):
    """Choose reference labels for target spans, then external-density LNOs."""

    occ_projection, vir_projection = _internal_projection_matrices(
        domain_data, fragment_index, lo_type=domain_selections.lo_type
    )
    if domain_selections.lo_type == "boys":
        # A Boys fragment is one occupied target, independent of the IAO cutoff.
        occ_internal_keep = numpy.asarray([0], dtype=numpy.int32)
    else:
        occ_internal_keep = _row_gram_keep_numpy(
            occ_projection, internal_rank_threshold
        )
    vir_internal_keep = _row_gram_keep_numpy(
        vir_projection, internal_rank_threshold
    )
    occ_internal = _fixed_row_space(occ_projection, occ_internal_keep)
    vir_internal = _fixed_row_space(vir_projection, vir_internal_keep)

    full_occ = float(thresh_occ) <= 0.0
    full_vir = float(thresh_vir) <= 0.0
    occ_lno_keep = _density_keep_numpy(
        density_occupied_active, occ_internal, thresh_occ, full_occ
    )
    vir_lno_keep = _density_keep_numpy(
        density_virtual_active, vir_internal, thresh_vir, full_vir
    )
    return FragmentLISSelection(
        fragment_index=int(fragment_index),
        internal_occ_keep=occ_internal_keep,
        internal_vir_keep=vir_internal_keep,
        occupied_lno_keep=occ_lno_keep,
        virtual_lno_keep=vir_lno_keep,
        full_occupied_space=full_occ,
        full_virtual_space=full_vir,
    )


def build_fragment_lis_selection(
    mf,
    mp2_static,
    fragment_index,
    *,
    common=None,
    domain=None,
    thresh_occ=1e-4,
    thresh_vir=1e-5,
    internal_rank_threshold=LIS_INTERNAL_RANK_THRESHOLD,
):
    """Select fixed internal and external-LNO ranks at a reference geometry.

    ``mp2_static`` supplies fixed domain selections; ``common`` optionally
    supplies the current :class:`DomainData`. Internal IAO directions satisfy
    ``singular_value > internal_rank_threshold``. External LNOs satisfy
    ``abs(density_eigenvalue) > thresh_occ/vir``; a zero LNO threshold selects
    the full corresponding active-HF manifold.

    ``domain`` may be supplied by a caller that constructs ED orbital frames
    separately from the target-conditioned MP2-density calculation.  This is
    the boundary used by the MPI driver: its root rank owns all discrete domain
    construction, while independent ranks evaluate this fragment-local
    operation.  Supplying a domain does not change the serial equations or
    any retained-rank decision.
    """

    domain_selections = mp2_static
    domain_data = common
    if not isinstance(domain_selections, DomainSelections):
        raise TypeError("mp2_static must be DomainSelections")
    fragment_index = int(fragment_index)
    if fragment_index < 0 or fragment_index >= len(domain_selections.fragments):
        raise IndexError(
            f"fragment_index={fragment_index} is outside "
            f"[0, {len(domain_selections.fragments)})"
        )
    for name, value in (
        ("thresh_occ", thresh_occ),
        ("thresh_vir", thresh_vir),
        ("internal_rank_threshold", internal_rank_threshold),
    ):
        if float(value) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if domain_data is None:
        domain_data = rebuild_domain_data(mf, domain_selections)
    if not isinstance(domain_data, DomainData):
        raise TypeError("common must be DomainData")
    if domain is None:
        domain = build_strong_ed_domain(
            domain_data, domain_selections, fragment_index
        )
    if not isinstance(domain, StrongDomain):
        raise TypeError("domain must be StrongDomain")

    with tempfile.TemporaryDirectory(
        prefix=f"pyscfad-lov-frag{fragment_index}-",
        dir=pyscf_lib.param.TMPDIR,
    ) as lov_scratch_dir:
        density = strong_domain_mp2_density(
            mf,
            domain,
            domain_selections,
            fragment_index,
            lov_scratch_dir=lov_scratch_dir,
        )
        dmoo_active, dmvv_active = _domain_density_in_active_spaces(
            domain_data, domain_selections, fragment_index, domain, density
        )
        selection = _reference_fragment_selection(
            domain_data,
            domain_selections,
            fragment_index,
            dmoo_active,
            dmvv_active,
            thresh_occ=thresh_occ,
            thresh_vir=thresh_vir,
            internal_rank_threshold=internal_rank_threshold,
        )
        # Reference selection has no reverse pass; finish pending work before
        # removing its privately owned forward scratch.
        jax.block_until_ready((density, dmoo_active, dmvv_active))
    return selection


def build_lis_selections(
    mf,
    mp2_static,
    *,
    common=None,
    thresh_occ=1e-4,
    thresh_vir=1e-5,
    internal_rank_threshold=LIS_INTERNAL_RANK_THRESHOLD,
):
    """Select all fragment LIS labels, sharing current global arrays.

    This host-side reference stage must run outside differentiation. The
    returned labels are replayed by :func:`build_fragment_lis`.
    """

    domain_selections = mp2_static
    domain_data = common
    if not isinstance(domain_selections, DomainSelections):
        raise TypeError("mp2_static must be DomainSelections")
    for name, value in (
        ("thresh_occ", thresh_occ),
        ("thresh_vir", thresh_vir),
        ("internal_rank_threshold", internal_rank_threshold),
    ):
        if float(value) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if domain_data is None:
        domain_data = rebuild_domain_data(mf, domain_selections)
    if not isinstance(domain_data, DomainData):
        raise TypeError("common must be DomainData")

    fragments = tuple(
        build_fragment_lis_selection(
            mf,
            domain_selections,
            fragment_index,
            common=domain_data,
            thresh_occ=thresh_occ,
            thresh_vir=thresh_vir,
            internal_rank_threshold=internal_rank_threshold,
        )
        for fragment_index in range(len(domain_selections.fragments))
    )

    return LISSelections(
        mp2_static=domain_selections,
        thresh_occ=float(thresh_occ),
        thresh_vir=float(thresh_vir),
        internal_rank_threshold=float(internal_rank_threshold),
        fragments=fragments,
    )


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


def build_fragment_lis(
    mf,
    common,
    static,
    fragment_index,
    *,
    domain=None,
    density=None,
    lov_scratch_dir=None,
):
    """Rebuild one LIS with fixed labels and differentiable current arrays.

    ``common`` is current :class:`DomainData`; ``static`` is :class:`LISSelections`.
    The ED density is embedded in the global active-HF bases, combined with
    the internal target spans, and completed to the impurity MO layout.
    The LIS ranks and retained LNO labels remain fixed.

    A supplied ``density`` must use the supplied ``domain`` orbital bases.
    Otherwise the caller must provide scratch that remains valid through any
    reverse pass; see :func:`strong_domain_mp2_density`.
    """

    domain_data = common
    lis_selections = static
    if not isinstance(lis_selections, LISSelections):
        raise TypeError("static must be LISSelections")
    domain_selections = lis_selections.mp2_static
    if not isinstance(domain_data, DomainData):
        raise TypeError("common must be DomainData")
    fragment_index = int(fragment_index)
    selection = lis_selections.fragments[fragment_index]
    if selection.fragment_index != fragment_index:
        raise ValueError("fragment selection order is inconsistent")

    # Rebuild the strong ED and embed its MP2 density in global active-HF space.
    if domain is None:
        domain = build_strong_ed_domain(
            domain_data, domain_selections, fragment_index
        )
    if density is None:
        if lov_scratch_dir is None:
            raise ValueError(
                "lov_scratch_dir is required when density is not supplied"
            )
        density = strong_domain_mp2_density(
            mf,
            domain,
            domain_selections,
            fragment_index,
            lov_scratch_dir=lov_scratch_dir,
        )
    dmoo_active, dmvv_active = _domain_density_in_active_spaces(
        domain_data, domain_selections, fragment_index, domain, density
    )

    # Replay reference labels for the target spans and external-density LNOs.
    occ_projection, vir_projection = _internal_projection_matrices(
        domain_data, fragment_index, lo_type=domain_selections.lo_type
    )
    occ_internal = _fixed_row_space(
        occ_projection, selection.internal_occ_keep
    )
    vir_internal = _fixed_row_space(
        vir_projection, selection.internal_vir_keep
    )
    occupied_rotation = _fixed_density_space(
        dmoo_active,
        occ_internal,
        selection.occupied_lno_keep,
        selection.full_occupied_space,
    )
    virtual_rotation = _fixed_density_space(
        dmvv_active,
        vir_internal,
        selection.virtual_lno_keep,
        selection.full_virtual_space,
    )

    # Semicanonicalize the LIS and add frozen occupied/virtual complements.
    full_coeff, frozen, occupied_lis_coeff, virtual_lis_coeff = (
        _assemble_full_mo_layout(
            mf,
            domain_selections,
            domain_data.occupied_coeff,
            domain_data.virtual_coeff,
            domain_data.fock,
            occupied_rotation,
            virtual_rotation,
            lis_selections.internal_rank_threshold,
        )
    )
    target_data = domain_data.fragment_occupied_data[fragment_index]
    occupied_projector = occupied_rotation @ occupied_rotation.T.conj()
    virtual_projector = virtual_rotation @ virtual_rotation.T.conj()
    return FragmentLIS(
        mo_coeff=full_coeff,
        frozen=frozen,
        fragment_occupied_anchor=target_data.occupied_projection,
        fragment_iao_coeff=target_data.iao_coeff,
        active_occupied_coeff=occupied_lis_coeff,
        active_virtual_coeff=virtual_lis_coeff,
        occupied_projector=_hermitize(occupied_projector),
        virtual_projector=_hermitize(virtual_projector),
        density_occupied_ed=density.occupied,
        density_virtual_ed=density.virtual,
        density_occupied_active=dmoo_active,
        density_virtual_active=dmvv_active,
        domain=domain,
        n_internal_occ=int(selection.internal_occ_keep.size),
        n_internal_vir=int(selection.internal_vir_keep.size),
        n_lno_occ=(
            int(domain_data.occupied_coeff.shape[1]
                - selection.internal_occ_keep.size)
            if selection.full_occupied_space
            else int(selection.occupied_lno_keep.size)
        ),
        n_lno_vir=(
            int(domain_data.virtual_coeff.shape[1]
                - selection.internal_vir_keep.size)
            if selection.full_virtual_space
            else int(selection.virtual_lno_keep.size)
        ),
    )
