"""IAO-fragment extended-domain MP2.

This module provides both the eager IAO domain/energy validation path and the
fixed-topology nuclear-gradient entry point.  The mathematical design
separates discrete topology (fragment assignments, atom lists, retained ranks,
and the strong-pair graph) from continuous operations carried out inside a
fixed domain.  :class:`IAOFragmentMP2` freezes only the former and rebuilds
IAOs, PAOs, fragment weights, local spaces, integrals, and weak multipole
energies on the differentiable path.

The validation path presently targets real, restricted closed-shell molecular
references.

The key distinction from the conventional one-LMO construction is that the
occupied projections of different IAO fragments overlap.  Spatial domains are
therefore selected from the *weighted* occupied block ``P_occ A_F``, and MP2
energies are partitioned with the positive-semidefinite weight
``W_F = (A_F^H S C_o)^H (A_F^H S C_o)``.  Normalized occupied range projectors
must not be substituted for these weights.
"""

from dataclasses import dataclass, replace
from functools import reduce
import time

import numpy as np
import scipy.linalg
from pyscf.mp.mp2 import _mo_splitter

from pyscfad.mp import dfmp2
from pyscfad.lno import lno_base
from pyscfad.lno.lno_base import get_iao
from pyscfad.lno.tools import autofrag, map_lo_to_frag

from . import multipole_numpy as static_multipole
from . import pao as dlno_pao
from . import util
from .domain import (
    _compute_av_numpy,
    get_bp_domain,
    get_fragment_bp_domain,
    get_primary_domain,
)
from .fragment_mp2 import (
    build_fragment_occupied_data,
    fragment_pair_energy_from_lov,
)


__all__ = [
    "IAOFragmentMP2",
    "IAOFragmentMP2Thresholds",
    "IAOFragmentTopology",
    "IAOFragmentMP2Timing",
    "IAOFragmentDomainResult",
    "IAOFragmentMP2Result",
    "build_iao_fragment_topology",
    "evaluate_iao_fragment_mp2",
    "kernel",
]


class IAOFragmentMP2:
    """Fixed-topology IAO-fragment MP2 energy and nuclear gradient.

    The public :meth:`value_and_grad` entry point mirrors the progressive
    interface used by :class:`pyscfad.dlno.ccsd.DLNOCCSD`: ``build_mf`` is
    traced once, the discrete ED/pair topology is constructed eagerly at the
    reference geometry, and fragment energies are differentiated one at a
    time.  Only atom/index lists, strong/weak pair classes, and retained-rank
    choices are frozen.  IAOs, PAOs, local semicanonical orbitals, local RI
    factors, strong-pair MP2 energies, and weak multipole energies are rebuilt
    on the differentiable path.

    The lower-level :meth:`correlation_value_and_grad` method returns an SCF
    object cotangent rather than closing the SCF response.  DLNO-CCSD(T) can
    therefore add the same PT2 correction to its existing SCF cotangent and
    perform one final CPHF pullback.
    """

    @staticmethod
    def build_static_topology(mf, **kwargs):
        """Build the reference topology and retain only discrete choices.

        A driver that already owns a saved SCF VJP should call this eager
        helper on a detached ``mf`` (for example through ``stop_trace``), as
        :meth:`value_and_grad` does.  Besides defining the derivative
        boundary, this prevents topology-only DF setup from mutating the SCF
        object whose original pytree structure belongs to the saved VJP.
        """
        from .iao_mp2_grad import build_iao_mp2_static_selections

        reference = build_iao_fragment_topology(mf, **kwargs)
        return build_iao_mp2_static_selections(mf, reference)

    @staticmethod
    def correlation_value_and_grad(
        mf, topology, *, return_details=False
    ):
        """Return ``(E_corr, mf_bar[, details])`` for fixed topology."""
        from .iao_mp2_grad import correlation_value_and_grad

        return correlation_value_and_grad(
            mf, topology, return_details=return_details
        )

    @staticmethod
    def correlation_value_and_grad_with_iao(
        mf, iao_coeff, topology, *, return_details=False
    ):
        """Return correlation cotangents and optional scalar diagnostics."""
        from .iao_mp2_grad import correlation_value_and_grad_with_iao

        return correlation_value_and_grad_with_iao(
            mf,
            iao_coeff,
            topology,
            return_details=return_details,
        )

    @classmethod
    def value_and_grad(
        cls,
        mol,
        *,
        build_mf,
        frag_lolist=None,
        frag_atmlist=None,
        frozen=None,
        thresholds=None,
        pair_energy_model="multipole",
        force_full_domains=False,
        topology=None,
        include_hf=True,
        return_details=False,
    ):
        """Return the fixed-topology local-MP2 energy and nuclear gradient.

        Parameters
        ----------
        mol
            Differentiable molecular object.
        build_mf
            Callable ``mol -> converged density-fitted RHF object``.  It is
            evaluated once under :func:`jax.vjp`.
        frag_lolist, frag_atmlist, frozen, thresholds, pair_energy_model,
        force_full_domains
            Passed to :func:`build_iao_fragment_topology` when ``topology`` is
            not supplied.
        topology
            Optional fixed selections returned by
            :func:`pyscfad.dlno.iao_mp2_grad.build_iao_mp2_static_selections`.
            An energy-only :class:`IAOFragmentTopology` is also accepted and
            converted at its matching reference geometry.  Reusing the fixed
            selections for displaced geometries keeps all domain and pair
            decisions identical.
        include_hf
            Include the RHF reference energy and its response.  If false,
            return only the local-MP2 correlation energy and gradient.
        return_details
            If true, append an
            :class:`pyscfad.dlno.iao_mp2_grad.IAOMP2Decomposition` containing
            the strong/weak split, term counts, ED dimensions, and timings.
            The records contain host scalars only; progressive AD tapes are
            still released one term at a time.

        Notes
        -----
        Domain construction is deliberately outside the AD graph, but domain
        *orbitals* are not.  In particular, the cached numerical weak-pair
        energy in :class:`IAOFragmentTopology` is never used by this method.
        The integral-direct local-RI reverse pass currently targets nuclear
        coordinates only; build ``mol`` with ``trace_exp=False`` and
        ``trace_ctr_coeff=False`` for this interface.

        The integral-direct local-RI reverse pass currently requires a
        positive-definite auxiliary metric for its Cholesky pullback.  The
        forward eigenvalue fallback for a linearly dependent auxiliary basis
        does not yet have a corresponding reverse rule and will raise rather
        than silently return an incomplete gradient.

        The saved SCF response is always built with the standard implicit
        backend.  The experimental ``pyscfad_scf_first_order_custom`` backend
        replays a finite SCF iteration history in its backward pass; for a
        nonstationary local-orbital cotangent, a small residual in that replay
        can be amplified into a large, non-reproducible nuclear gradient.
        Standard implicit response instead differentiates the converged SCF
        fixed point and is required by this interface.
        """
        import jax
        import jax.numpy as jnp

        from pyscfad import config_update
        from pyscfad.ops import stop_trace
        from .iao_mp2_grad import (
            IAOFragmentMP2StaticSelections,
            build_iao_mp2_static_selections,
        )

        if (
            getattr(mol, "exp", None) is not None
            or getattr(mol, "ctr_coeff", None) is not None
        ):
            raise NotImplementedError(
                "IAOFragmentMP2.value_and_grad currently differentiates "
                "nuclear coordinates only; build mol with trace_exp=False "
                "and trace_ctr_coeff=False"
            )

        with (
            config_update("pyscfad_scf_implicit_diff", True),
            config_update("pyscfad_scf_first_order_custom", False),
        ):
            mf, scf_pullback = jax.vjp(build_mf, mol)

        if topology is None:
            def build_static(mf_):
                reference = build_iao_fragment_topology(
                    mf_,
                    frozen=frozen,
                    frag_lolist=frag_lolist,
                    frag_atmlist=frag_atmlist,
                    thresholds=thresholds,
                    pair_energy_model=pair_energy_model,
                    force_full_domains=force_full_domains,
                )
                return build_iao_mp2_static_selections(mf_, reference)

            fixed_topology = stop_trace(build_static)(mf)
        elif isinstance(topology, IAOFragmentTopology):
            fixed_topology = stop_trace(
                lambda mf_: build_iao_mp2_static_selections(mf_, topology)
            )(mf)
        elif isinstance(topology, IAOFragmentMP2StaticSelections):
            fixed_topology = topology
        else:
            raise TypeError(
                "topology must be IAOFragmentTopology or "
                "IAOFragmentMP2StaticSelections"
            )

        corr_result = cls.correlation_value_and_grad(
            mf, fixed_topology, return_details=return_details
        )
        e_corr, mf_bar = corr_result[:2]
        if include_hf:
            e_hf, hf_pullback = jax.vjp(lambda mf_: mf_.e_tot, mf)
            hf_bar, = hf_pullback(
                jnp.ones((), dtype=jnp.asarray(e_hf).dtype)
            )

            def add_cotangent(left, right):
                if left is None:
                    return right
                if right is None:
                    return left
                if (
                    hasattr(left, "dtype")
                    and left.dtype == jax.dtypes.float0
                ):
                    return right
                if (
                    hasattr(right, "dtype")
                    and right.dtype == jax.dtypes.float0
                ):
                    return left
                return left + right

            mf_bar = jax.tree_util.tree_map(
                add_cotangent, mf_bar, hf_bar
            )
            energy = e_hf + e_corr
        else:
            energy = e_corr

        mol_bar, = scf_pullback(mf_bar)
        jax.block_until_ready((energy, mol_bar))
        if return_details:
            return energy, mol_bar, corr_result[2]
        return energy, mol_bar


@dataclass(frozen=True)
class IAOFragmentMP2Thresholds:
    """Thresholds for the staged IAO-fragment domain construction.

    ``bp_occ`` defines the compact occupied BP atom list used to select PAO
    parent centers.  ``bp_primary`` defines the occupied BP list from which a
    primary domain is built for multipole pair screening.  ``bp_ed`` is the
    tighter occupied recovery threshold that defines the AO support of the
    actual extended domain.  Keeping these three roles separate follows the
    logic of the Nagy domain hierarchy and avoids using a large primary domain
    merely to decide which PAOs are candidates.

    ``pair_energy`` is applied to Nagy's OS-based pair-increment convention
    (the ``-8`` prefactor of their Eq. (7)), not to the conventional global
    SCS-MP2 opposite-spin component.
    """

    bp_occ: float = 0.985
    bp_primary: float = 0.999
    bp_ed: float = 0.9998
    bp_pao: float = 0.98
    pao_norm: float = 1e-4
    domain_pao: float = 1e-4
    ed_pao: float = 0.995
    # Cutoff used only when a weighted fragment block is converted into a
    # normalized occupied *span*.  The additive energy weight W_F itself is
    # never thresholded.  A cutoff commensurate with the BP recovery loss
    # prevents tiny interfragment tails from becoming unit occupied modes.
    occupied_weight: float = 1e-4
    metric_rank: float = 1e-10
    # Nagy et al.'s default for the same -8 pair-increment convention.
    # Values near 1e-4 are useful loose diagnostics but can benefit from
    # cancellation once the multipole charge distributions overlap.
    pair_energy: float = 1.5e-5
    near_pair_distance: float = 3.5
    multipole_order: int = 4
    mp2_block_memory_mb: float = 256.0


@dataclass(frozen=True)
class IAOFragmentTopology:
    """Reference topology and continuous data for energy-only ED-MP2.

    This validation container is not itself the eventual stop-gradient
    object: only its discrete index lists, atom lists, masks, and rank choices
    may be frozen for a nuclear derivative.  Geometry-dependent arrays stored
    here must instead be reconstructed inside a differentiable fixed-topology
    evaluation.
    """

    frozen: object
    iao_coeff: np.ndarray
    frag_lolist: tuple
    frag_atmlist: tuple
    occupied_coeff: np.ndarray
    virtual_coeff: np.ndarray
    occupied_energy: np.ndarray
    virtual_energy: np.ndarray
    s1e: np.ndarray
    fock: np.ndarray
    fragment_occupied_data: tuple
    pao_coeff: np.ndarray
    ao2pao_map: np.ndarray
    pao_bp_domain: np.ndarray
    compact_bp_domain: np.ndarray
    primary_bp_domain: np.ndarray
    primary_domain: np.ndarray
    tight_bp_domain: np.ndarray
    pair_energy_model: str
    pair_energy: np.ndarray
    weak_pair_energy: np.ndarray
    forced_strong_mask: np.ndarray
    strong_mask: np.ndarray
    strong_fragments: tuple
    pao_center_domain: np.ndarray
    extended_domain: np.ndarray
    thresholds: IAOFragmentMP2Thresholds


@dataclass(frozen=True)
class IAOFragmentMP2Timing:
    """Post-topology wall-time split for one fragment or their aggregate.

    The aggregate is exactly the sum of the fragment timings.  One-time
    overlap/Fock preparation is charged to the first fragment's ED-orbital
    stage, and final evaluator-owned cache cleanup is charged to the last
    fragment's bookkeeping stage.
    """

    ed_orbital_seconds: float = 0.0
    local_ri_lov_seconds: float = 0.0
    weighted_ed_mp2_seconds: float = 0.0
    weak_bookkeeping_seconds: float = 0.0

    @property
    def total_seconds(self):
        """Sum of the four profiled post-topology stages."""
        return (
            self.ed_orbital_seconds
            + self.local_ri_lov_seconds
            + self.weighted_ed_mp2_seconds
            + self.weak_bookkeeping_seconds
        )


@dataclass(frozen=True)
class IAOFragmentDomainResult:
    """MP2 contribution and dimensions for one fragment ED.

    ``weak_multipole_opposite_spin`` follows the OS-based pair-increment
    convention of Nagy et al., including the pair-orientation prefactor in
    their Eq. (7).  It is not the conventional SCS-MP2 OS component resolved
    from a global canonical energy expression.
    """

    fragment_index: int
    strong_fragments: np.ndarray
    pao_center_atoms: np.ndarray
    extended_atoms: np.ndarray
    n_domain_ao: int
    n_domain_occ: int
    n_domain_vir: int
    target_weight_trace: float
    partner_weight_trace: float
    strong_total: float
    strong_opposite_spin: float
    strong_same_spin: float
    weak_multipole_opposite_spin: float
    timing: IAOFragmentMP2Timing | None = None


@dataclass(frozen=True)
class IAOFragmentMP2Result:
    """Summed IAO-fragment ED-MP2 result.

    ``e_strong_os`` and ``e_strong_ss`` are conventional MP2 spin components.
    ``e_weak_multipole_os`` instead carries Nagy's OS-based distant-pair
    increment convention; it approximates the omitted total distant-pair
    contribution and should not be interpreted as an SCS-OS diagnostic.
    """

    e_corr: float
    e_strong: float
    e_weak_multipole_os: float
    e_strong_os: float
    e_strong_ss: float
    fragments: tuple
    topology: IAOFragmentTopology
    timing: IAOFragmentMP2Timing | None = None


def _as_index_tuple(index_lists):
    return tuple(
        np.asarray(indices, dtype=np.int32).reshape(-1)
        for indices in index_lists
    )


def _union_index_lists(index_lists):
    arrays = [
        np.asarray(indices, dtype=np.int32).reshape(-1)
        for indices in index_lists
        if np.asarray(indices).size
    ]
    if not arrays:
        return np.zeros((0,), dtype=np.int32)
    return reduce(np.union1d, arrays).astype(np.int32, copy=False)


def _active_orbitals(mf, frozen):
    pt = dfmp2.MP2(mf, frozen=frozen)
    eris = pt.ao2mo()
    coeff = np.asarray(eris.mo_coeff)
    energy = np.asarray(eris.mo_energy)
    nocc = int(pt.nocc)
    return (
        pt,
        coeff[:, :nocc],
        coeff[:, nocc:],
        energy[:nocc],
        energy[nocc:],
    )


def _pao_projected_out_coeff(pt):
    """Orbitals complementary to the active virtual space of ``pt``."""
    masks = _mo_splitter(pt)
    coeff = np.asarray(pt.mo_coeff)
    blocks = [coeff[:, masks[index]] for index in (0, 1, 3)]
    blocks = [block for block in blocks if block.shape[1]]
    if not blocks:
        return np.zeros((coeff.shape[0], 0), dtype=coeff.dtype)
    return np.hstack(blocks)


def _validate_fragment_atom_lists(mol, frag_atmlist, nfrag):
    if frag_atmlist is None:
        return tuple([None] * nfrag)
    if len(frag_atmlist) != nfrag:
        raise ValueError("frag_atmlist and frag_lolist must have equal length")
    output = []
    for atoms in frag_atmlist:
        atoms = np.unique(np.asarray(atoms, dtype=np.int32).reshape(-1))
        if np.any((atoms < 0) | (atoms >= mol.natm)):
            raise ValueError("fragment atom index is out of range")
        output.append(atoms)
    return tuple(output)


def _physical_pair_matrix(directed):
    """Convert directed fragment contributions to an unordered-pair score."""
    directed = np.asarray(directed)
    physical = directed + directed.T.conj()
    diagonal = np.diag_indices_from(physical)
    physical[diagonal] = directed[diagonal]
    return np.asarray(physical.real)


def _exact_global_fragment_pairs(pt, occupied_data, max_memory_mb=256.0):
    eris = pt.ao2mo()
    nocc = int(pt.nocc)
    nvir = int(pt.nmo - nocc)
    lov = np.asarray(
        pt.loop_ao2mo(eris.mo_coeff, nocc, with_t2=False)
    ).reshape(-1, nocc, nvir)
    weights = np.stack([data.occupied_weight for data in occupied_data])
    directed = fragment_pair_energy_from_lov(
        lov,
        np.asarray(eris.mo_energy[:nocc]),
        np.asarray(eris.mo_energy[nocc:]),
        weights,
        max_memory_mb=max_memory_mb,
    )
    # Nagy's Eq. (7) uses an OS-based *pair increment*: the -8 prefactor
    # includes both occupied-pair orientations and the equivalent restricted
    # pair-domain bookkeeping.  An unordered standard SCS-OS contribution
    # from the global two-sided weights carries a -2 prefactor.  Multiply the
    # latter by four so the diagnostic exact and multipole screening models
    # use the same threshold convention.  ``directed`` itself remains in the
    # standard MP2 OS/SS convention for energy diagnostics.
    screening_os = 4.0 * directed.opposite_spin
    return directed, _physical_pair_matrix(screening_os)


def _metric_orthonormalize(coeff, overlap, threshold):
    coeff = np.asarray(coeff)
    if coeff.ndim != 2:
        raise ValueError("coefficient block must be rank two")
    if coeff.shape[1] == 0:
        return np.zeros((coeff.shape[0], 0), dtype=coeff.dtype)
    metric = coeff.conj().T @ np.asarray(overlap) @ coeff
    metric = 0.5 * (metric + metric.conj().T)
    eigenvalue, eigenvector = scipy.linalg.eigh(metric)
    keep = eigenvalue > threshold
    if not np.any(keep):
        return np.zeros((coeff.shape[0], 0), dtype=coeff.dtype)
    return coeff @ (
        eigenvector[:, keep] / np.sqrt(eigenvalue[keep])[None, :]
    )


def _normalize_metric_columns(coeff, overlap, threshold):
    coeff = np.asarray(coeff)
    norm2 = np.real(np.einsum(
        "ui,uv,vi->i", coeff.conj(), np.asarray(overlap), coeff,
        optimize=True,
    ))
    keep = norm2 > threshold
    return coeff[:, keep] / np.sqrt(norm2[keep])[None, :], keep


def _semicanonicalize(coeff, fock):
    coeff = np.asarray(coeff)
    if coeff.shape[1] == 0:
        return np.zeros((0,), dtype=float), coeff
    projected = coeff.conj().T @ np.asarray(fock) @ coeff
    projected = 0.5 * (projected + projected.conj().T)
    energy, rotation = scipy.linalg.eigh(projected)
    return np.asarray(energy.real), coeff @ rotation


def _fragment_multipole_modes(
    occupied_weight,
    occupied_energy,
    weight_threshold,
    degeneracy_tolerance=1e-10,
):
    r"""Choose a gauge-invariant spectral representation of ``W_F``.

    The right singular vectors of the raw fragment IAO block are not unique
    when singular values are degenerate.  Using those vectors as individual
    multipole orbitals consequently makes the weak-pair estimate depend on an
    arbitrary rotation of the IAOs inside a fragment, even though
    :math:`W_F=M_F^\dagger M_F` is unchanged.

    Here the retained modes are constructed from ``W_F`` itself.  Within each
    numerically degenerate eigenvalue block, the global occupied Fock matrix
    supplies an invariant secondary diagonalization.  The common block weight
    is used because splittings inside such a block are roundoff artifacts;
    mathematically this leaves the spectral representation of an exactly
    degenerate block unchanged.  An exact simultaneous degeneracy of ``W_F``
    and the projected occupied Fock matrix retains the usual arbitrary basis
    within that common eigenspace.
    """
    occupied_weight = np.asarray(occupied_weight)
    occupied_energy = np.asarray(occupied_energy)
    if occupied_weight.ndim != 2 or (
        occupied_weight.shape[0] != occupied_weight.shape[1]
    ):
        raise ValueError("occupied_weight must be a square matrix")
    nocc = occupied_weight.shape[0]
    if occupied_energy.shape != (nocc,):
        raise ValueError("occupied_energy must have one entry per occupied MO")

    hermitian_weight = 0.5 * (
        occupied_weight + occupied_weight.conj().T
    )
    eigenvalue, eigenvector = scipy.linalg.eigh(hermitian_weight)
    order = np.argsort(eigenvalue)[::-1]
    eigenvalue = np.asarray(eigenvalue[order].real)
    eigenvector = np.asarray(eigenvector[:, order])
    keep = eigenvalue > weight_threshold
    eigenvalue = eigenvalue[keep]
    eigenvector = eigenvector[:, keep]
    if eigenvalue.size == 0:
        return eigenvalue, eigenvector

    mode_blocks = []
    weight_blocks = []
    start = 0
    while start < eigenvalue.size:
        stop = start + 1
        while stop < eigenvalue.size and np.isclose(
            eigenvalue[stop],
            eigenvalue[start],
            rtol=degeneracy_tolerance,
            atol=degeneracy_tolerance,
        ):
            stop += 1

        block = eigenvector[:, start:stop]
        if block.shape[1] > 1:
            projected_fock = block.conj().T @ (
                occupied_energy[:, None] * block
            )
            projected_fock = 0.5 * (
                projected_fock + projected_fock.conj().T
            )
            _, rotation = scipy.linalg.eigh(projected_fock)
            block = block @ rotation

        # Fix the otherwise irrelevant sign/phase to make diagnostics and
        # serialized topology deterministic as well.
        for column in range(block.shape[1]):
            pivot = int(np.argmax(np.abs(block[:, column])))
            value = block[pivot, column]
            if abs(value) > 0:
                block[:, column] *= np.conj(value) / abs(value)

        mode_blocks.append(block)
        weight_blocks.append(np.full(
            block.shape[1], np.mean(eigenvalue[start:stop]), dtype=float
        ))
        start = stop

    return np.concatenate(weight_blocks), np.hstack(mode_blocks)


def _fragment_screen_space(mf, topology, fragment_index):
    """Build weighted occupied modes and a PAO span in one primary domain."""
    mol = mf.mol
    thresholds = topology.thresholds
    data = topology.fragment_occupied_data[fragment_index]
    primary_atoms = np.asarray(
        topology.primary_domain[fragment_index], dtype=np.int32
    )
    primary_bp_atoms = np.asarray(
        topology.primary_bp_domain[fragment_index], dtype=np.int32
    )
    ao_idx = util.ao_index_by_atom(mol, primary_atoms)
    s1e = np.asarray(topology.s1e)
    fock = np.asarray(topology.fock)
    s21 = s1e[ao_idx]
    s22 = s1e[np.ix_(ao_idx, ao_idx)]
    fock22 = fock[np.ix_(ao_idx, ao_idx)]

    # Construct the multipole modes from the invariant weight W_F rather than
    # from the gauge-dependent SVD of the raw fragment IAO columns.  The modes
    # are used only for the multipole approximation; their spectral weights
    # remain explicit and are never replaced by unity.
    weight, occupied_mode = _fragment_multipole_modes(
        data.occupied_weight,
        topology.occupied_energy,
        thresholds.occupied_weight,
    )
    occupied_global = np.asarray(topology.occupied_coeff) @ occupied_mode
    if occupied_global.shape[1] == 0:
        raise RuntimeError(
            f"fragment {fragment_index} has no occupied multipole modes"
        )
    occupied_local = util.project_mo(occupied_global, s21, s22)
    occupied_local, norm_keep = _normalize_metric_columns(
        occupied_local, s22, thresholds.metric_rank
    )
    weight = weight[norm_keep]
    if occupied_local.shape[1] == 0:
        raise RuntimeError(
            f"fragment {fragment_index} has no occupied multipole modes"
        )
    occupied_energy = np.real(np.einsum(
        "ui,uv,vi->i", occupied_local.conj(), fock22, occupied_local,
        optimize=True,
    ))

    # Pair screening uses the primary-domain construction: PAOs centered on
    # the occupied BP_PD atoms are represented in the larger primary AO
    # domain.  The more compact BP_occ center list is reserved for the actual
    # improved ED construction below.
    candidate = dlno_pao.pao_overlap_with_domain(
        mol,
        topology.pao_coeff,
        primary_bp_atoms,
        ao2pao_map=topology.ao2pao_map,
        s1e=s1e,
        ovlp_thr=thresholds.domain_pao,
    )
    if candidate.shape[1]:
        completeness = _compute_av_numpy(
            mol, candidate, s1e=s1e, atmlst=primary_atoms
        )
        candidate = candidate[:, completeness > thresholds.bp_pao]
    if candidate.shape[1]:
        candidate = util.project_mo(candidate, s21, s22)
        occupied_span = _metric_orthonormalize(
            occupied_local, s22, thresholds.metric_rank
        )
        candidate = util.orthogonalize(occupied_span, candidate, s22)
        candidate = _metric_orthonormalize(
            candidate, s22, thresholds.metric_rank
        )
        virtual_energy, candidate = _semicanonicalize(candidate, fock22)
    else:
        candidate = np.zeros((ao_idx.size, 0))
        virtual_energy = np.zeros((0,))
    if candidate.shape[1] == 0:
        return None

    # This record is consumed only while selecting the fixed strong-pair
    # topology.  A future differentiated weak-pair energy must rebuild the
    # same quantities through the JAX implementation in ``dlno.mp2``.
    multipole_data = [
        static_multipole.multipole_orbital_data(
            mol,
            occupied_energy[index],
            occupied_local[:, index],
            virtual_energy,
            candidate,
            primary_atoms,
            thresholds.multipole_order,
        )
        for index in range(weight.size)
    ]
    return {
        "weights": weight,
        "occupied_energy": occupied_energy,
        "occupied_coeff": [occupied_local[:, i] for i in range(weight.size)],
        "virtual_energy": [virtual_energy] * weight.size,
        "virtual_coeff": [candidate] * weight.size,
        "atmlst": [primary_atoms] * weight.size,
        "multipole_data": multipole_data,
    }


def _multipole_fragment_pairs(mf, topology_without_pairs):
    nfrag = len(topology_without_pairs.frag_lolist)
    screen = [
        _fragment_screen_space(mf, topology_without_pairs, fragment)
        for fragment in range(nfrag)
    ]
    pair = np.zeros((nfrag, nfrag), dtype=float)
    forced_strong = np.eye(nfrag, dtype=bool)
    order = topology_without_pairs.thresholds.multipole_order
    near_distance = topology_without_pairs.thresholds.near_pair_distance
    coordinates = np.asarray(mf.mol.atom_coords())
    for left in range(nfrag):
        for right in range(left):
            if screen[left] is None or screen[right] is None:
                forced_strong[left, right] = True
                forced_strong[right, left] = True
                continue
            left_atoms = topology_without_pairs.frag_atmlist[left]
            right_atoms = topology_without_pairs.frag_atmlist[right]
            if left_atoms is None or np.asarray(left_atoms).size == 0:
                left_atoms = topology_without_pairs.compact_bp_domain[left]
            if right_atoms is None or np.asarray(right_atoms).size == 0:
                right_atoms = topology_without_pairs.compact_bp_domain[right]
            left_atoms = np.asarray(left_atoms, dtype=np.int32)
            right_atoms = np.asarray(right_atoms, dtype=np.int32)
            separation = np.linalg.norm(
                coordinates[left_atoms, None, :]
                - coordinates[None, right_atoms, :],
                axis=-1,
            ).min()
            # The asymptotic multipole series is inappropriate for adjacent
            # fragments and becomes singular when two weighted fragment modes
            # share a centroid (notably the two halves of a covalent bond).
            # These pairs are unconditionally strong and never enter the weak
            # correction, so there is no need to evaluate the expansion.
            if separation < near_distance:
                forced_strong[left, right] = True
                forced_strong[right, left] = True
                continue
            lhs = screen[left]
            rhs = screen[right]
            orbital_pair = static_multipole.multipole_pair_energy_cross(
                lhs["multipole_data"], rhs["multipole_data"], order=order
            )
            value = np.sum(
                np.asarray(orbital_pair)
                * lhs["weights"][:, None]
                * rhs["weights"][None, :]
            )
            pair[left, right] = pair[right, left] = float(value)
            if not np.isfinite(value):
                forced_strong[left, right] = True
                forced_strong[right, left] = True
                pair[left, right] = pair[right, left] = 0.0
    return pair, forced_strong


def build_iao_fragment_topology(
    mf,
    iao_coeff=None,
    frag_lolist=None,
    frag_atmlist=None,
    *,
    frozen=None,
    thresholds=None,
    pair_energy_model="multipole",
    force_full_domains=False,
):
    """Build fixed BP/primary/extended topology for IAO fragments.

    ``force_full_domains`` is a validation option.  It puts every atom and
    every fragment in every ED and bypasses pair screening.  The resulting
    energy is an exact regression against canonical DF-MP2 when the PAO and
    metric thresholds retain the complete active occupied and virtual spaces.

    Custom ``iao_coeff`` columns must be partitioned exactly once by
    ``frag_lolist``.  ``frag_atmlist``, when supplied, must contain one valid
    atom-index list per fragment and is used to seed the collective BP growth.
    The ``exact`` pair model forms global DF factors and is intended only as a
    small-system screening diagnostic; production domain construction should
    normally use ``multipole``.
    """
    if thresholds is None:
        thresholds = IAOFragmentMP2Thresholds()
    if not isinstance(thresholds, IAOFragmentMP2Thresholds):
        raise TypeError("thresholds must be IAOFragmentMP2Thresholds")
    if not (
        0.0 <= thresholds.bp_occ
        <= thresholds.bp_primary
        <= thresholds.bp_ed
        <= 1.0
    ):
        raise ValueError(
            "occupied BP thresholds must satisfy "
            "0 <= bp_occ <= bp_primary <= bp_ed <= 1"
        )
    if not 0.0 <= thresholds.bp_pao <= 1.0:
        raise ValueError("bp_pao must lie between zero and one")
    if not 0.0 <= thresholds.ed_pao <= 1.0:
        raise ValueError("ed_pao must lie between zero and one")
    if min(
        thresholds.pao_norm,
        thresholds.domain_pao,
        thresholds.occupied_weight,
        thresholds.metric_rank,
        thresholds.pair_energy,
        thresholds.near_pair_distance,
    ) < 0.0:
        raise ValueError("domain and pair cutoffs must be non-negative")
    if thresholds.mp2_block_memory_mb <= 0.0:
        raise ValueError("mp2_block_memory_mb must be positive")
    if thresholds.multipole_order not in (2, 3, 4):
        raise ValueError("multipole_order must be 2, 3, or 4")
    if getattr(mf, "with_df", None) is None:
        raise ValueError("IAO-fragment MP2 requires a density-fitted SCF object")
    mo_coeff_input = np.asarray(mf.mo_coeff)
    mo_occ_input = np.asarray(mf.mo_occ)
    if np.iscomplexobj(mo_coeff_input):
        raise NotImplementedError(
            "IAO-fragment MP2 currently supports real orbitals only"
        )
    if mo_coeff_input.ndim != 2 or not np.all(
        (np.abs(mo_occ_input) < 1e-12)
        | (np.abs(mo_occ_input - 2.0) < 1e-12)
    ):
        raise NotImplementedError(
            "IAO-fragment MP2 currently supports restricted closed-shell "
            "references only"
        )

    mol = mf.mol
    s1e = np.asarray(mol.intor_symmetric("int1e_ovlp"))
    fock = np.asarray(mf.get_fock())
    pt, occupied, virtual, e_occ, e_vir = _active_orbitals(mf, frozen)
    if occupied.shape[1] == 0:
        raise ValueError("IAO-fragment MP2 requires an active occupied space")
    if virtual.shape[1] == 0:
        raise ValueError("IAO-fragment MP2 requires an active virtual space")
    if iao_coeff is None:
        iao_coeff = np.asarray(get_iao(mol, occupied))
    else:
        iao_coeff = np.asarray(iao_coeff)
    if np.iscomplexobj(iao_coeff):
        raise NotImplementedError(
            "IAO-fragment MP2 currently supports real IAOs only"
        )

    auto_frag_atoms = None
    if frag_lolist is None:
        auto_frag_atoms = autofrag(mol)
        frag_lolist = map_lo_to_frag(
            mol, iao_coeff, auto_frag_atoms, verbose=0
        )
    frag_lolist = _as_index_tuple(frag_lolist)
    if any(indices.size == 0 for indices in frag_lolist):
        raise ValueError("each IAO fragment must contain at least one IAO")
    if frag_atmlist is None and auto_frag_atoms is not None:
        frag_atmlist = auto_frag_atoms
    frag_atmlist = _validate_fragment_atom_lists(
        mol, frag_atmlist, len(frag_lolist)
    )

    occupied_data = build_fragment_occupied_data(
        occupied, iao_coeff, frag_lolist, s1e,
        svd_thr=thresholds.occupied_weight ** 0.5,
    )
    for fragment, data in enumerate(occupied_data):
        weight_trace = float(np.trace(data.occupied_weight).real)
        if weight_trace <= np.finfo(float).eps:
            raise ValueError(
                f"IAO fragment {fragment} has zero active occupied weight; "
                "merge it with a neighboring fragment or revise the IAO "
                "partition"
            )
    weight_sum = sum(data.occupied_weight for data in occupied_data)
    if not np.allclose(
        weight_sum, np.eye(occupied.shape[1]), atol=1e-8, rtol=1e-8
    ):
        raise ValueError(
            "the complete IAO fragment weights do not resolve the active "
            "occupied identity"
        )

    # A PAO is an AO projected into the *active virtual* space.  Besides all
    # occupied orbitals, explicitly project out any user-frozen virtuals.
    # Omitting the latter silently reintroduces frozen virtual MOs in the ED.
    pao_projected_out = _pao_projected_out_coeff(pt)
    pao_coeff, ao2pao_map = dlno_pao.pao(
        mol,
        pao_projected_out,
        s1e=s1e,
        norm_thr=thresholds.pao_norm,
    )
    pao_bp = get_bp_domain(
        mol, pao_coeff, s1e=s1e, bp_thr=thresholds.bp_pao
    )
    occupied_blocks = [data.occupied_projection for data in occupied_data]
    seed_atoms = None
    if all(atoms is not None for atoms in frag_atmlist):
        seed_atoms = frag_atmlist

    compact_bp = get_fragment_bp_domain(
        mol, occupied_blocks, s1e=s1e, bp_thr=thresholds.bp_occ,
        atmlsts=seed_atoms,
    )
    primary_bp = get_fragment_bp_domain(
        mol, occupied_blocks, s1e=s1e, bp_thr=thresholds.bp_primary,
        atmlsts=seed_atoms,
    )
    tight_bp = get_fragment_bp_domain(
        mol, occupied_blocks, s1e=s1e, bp_thr=thresholds.bp_ed,
        atmlsts=seed_atoms,
    )
    primary_domain = get_primary_domain(
        mol, primary_bp, pao_bp, ao2pao_map
    )

    nfrag = len(frag_lolist)
    all_atoms = np.arange(mol.natm, dtype=np.int32)
    if force_full_domains:
        compact_bp = np.empty(nfrag, dtype=object)
        primary_bp = np.empty(nfrag, dtype=object)
        tight_bp = np.empty(nfrag, dtype=object)
        primary_domain = np.empty(nfrag, dtype=object)
        for collection in (compact_bp, primary_bp, tight_bp, primary_domain):
            collection[:] = [all_atoms.copy() for _ in range(nfrag)]

    model = str(pair_energy_model).lower().replace("_", "-")
    valid_models = {
        "all", "all-strong", "exact",
        "multipole", "multipole-os", "os-multipole",
    }
    if model not in valid_models:
        raise ValueError(f"unknown pair_energy_model: {pair_energy_model}")

    # Construct a temporary topology because multipole screening consumes the
    # primary-domain spaces but precedes the strong graph and ED unions.
    placeholder_pair = np.zeros((nfrag, nfrag), dtype=float)
    placeholder_mask = np.eye(nfrag, dtype=bool)
    placeholder_strong = tuple(
        np.asarray([fragment], dtype=np.int32) for fragment in range(nfrag)
    )
    placeholder_domains = np.empty(nfrag, dtype=object)
    placeholder_domains[:] = [np.asarray(domain) for domain in tight_bp]
    base = IAOFragmentTopology(
        frozen=frozen,
        iao_coeff=iao_coeff,
        frag_lolist=frag_lolist,
        frag_atmlist=frag_atmlist,
        occupied_coeff=occupied,
        virtual_coeff=virtual,
        occupied_energy=e_occ,
        virtual_energy=e_vir,
        s1e=s1e,
        fock=fock,
        fragment_occupied_data=occupied_data,
        pao_coeff=pao_coeff,
        ao2pao_map=ao2pao_map,
        pao_bp_domain=pao_bp,
        compact_bp_domain=compact_bp,
        primary_bp_domain=primary_bp,
        primary_domain=primary_domain,
        tight_bp_domain=tight_bp,
        pair_energy_model=model,
        pair_energy=placeholder_pair,
        weak_pair_energy=placeholder_pair,
        forced_strong_mask=placeholder_mask,
        strong_mask=placeholder_mask,
        strong_fragments=placeholder_strong,
        pao_center_domain=compact_bp,
        extended_domain=placeholder_domains,
        thresholds=thresholds,
    )

    if (
        force_full_domains
        or thresholds.pair_energy <= 0.0
        or model in ("all", "all-strong")
    ):
        pair_energy = np.zeros((nfrag, nfrag), dtype=float)
        weak_pair_energy = np.zeros((nfrag, nfrag), dtype=float)
        forced_strong = np.ones((nfrag, nfrag), dtype=bool)
        strong_mask = np.ones((nfrag, nfrag), dtype=bool)
    elif model == "exact":
        _, pair_energy = _exact_global_fragment_pairs(
            pt,
            occupied_data,
            max_memory_mb=thresholds.mp2_block_memory_mb,
        )
        weak_pair_energy, forced_strong = _multipole_fragment_pairs(mf, base)
        strong_mask = (
            np.abs(pair_energy) > thresholds.pair_energy
        ) | forced_strong
    elif model in ("multipole", "multipole-os", "os-multipole"):
        pair_energy, forced_strong = _multipole_fragment_pairs(mf, base)
        weak_pair_energy = pair_energy
        strong_mask = (
            np.abs(pair_energy) > thresholds.pair_energy
        ) | forced_strong
    strong_mask = np.asarray(strong_mask | strong_mask.T, dtype=bool)
    np.fill_diagonal(strong_mask, True)
    if thresholds.pair_energy <= 0:
        strong_mask[:] = True
    strong_fragments = tuple(
        np.where(strong_mask[fragment])[0].astype(np.int32)
        for fragment in range(nfrag)
    )
    pao_center_domain = np.empty(nfrag, dtype=object)
    extended_domain = np.empty(nfrag, dtype=object)
    for fragment, partners in enumerate(strong_fragments):
        pao_center_domain[fragment] = _union_index_lists(
            [compact_bp[partner] for partner in partners]
        )
        extended_domain[fragment] = _union_index_lists(
            [tight_bp[partner] for partner in partners]
        )

    return IAOFragmentTopology(
        frozen=frozen,
        iao_coeff=iao_coeff,
        frag_lolist=frag_lolist,
        frag_atmlist=frag_atmlist,
        occupied_coeff=occupied,
        virtual_coeff=virtual,
        occupied_energy=e_occ,
        virtual_energy=e_vir,
        s1e=s1e,
        fock=fock,
        fragment_occupied_data=occupied_data,
        pao_coeff=pao_coeff,
        ao2pao_map=ao2pao_map,
        pao_bp_domain=pao_bp,
        compact_bp_domain=compact_bp,
        primary_bp_domain=primary_bp,
        primary_domain=primary_domain,
        tight_bp_domain=tight_bp,
        pair_energy_model=model,
        pair_energy=np.asarray(pair_energy),
        weak_pair_energy=np.asarray(weak_pair_energy),
        forced_strong_mask=np.asarray(forced_strong),
        strong_mask=strong_mask,
        strong_fragments=strong_fragments,
        pao_center_domain=pao_center_domain,
        extended_domain=extended_domain,
        thresholds=thresholds,
    )


def _build_fragment_domain_orbitals(
    mf, topology, fragment_index, *, s1e=None, fock=None
):
    mol = mf.mol
    thresholds = topology.thresholds
    if s1e is None:
        s1e = np.asarray(mol.intor_symmetric("int1e_ovlp"))
    else:
        s1e = np.asarray(s1e)
    if fock is None:
        fock = np.asarray(mf.get_fock())
    else:
        fock = np.asarray(fock)
    extended_atoms = np.asarray(
        topology.extended_domain[fragment_index], dtype=np.int32
    )
    center_atoms = np.asarray(
        topology.pao_center_domain[fragment_index], dtype=np.int32
    )
    ao_idx = util.ao_index_by_atom(mol, extended_atoms)
    s21 = s1e[ao_idx]
    s22 = s1e[np.ix_(ao_idx, ao_idx)]
    fock22 = fock[np.ix_(ao_idx, ao_idx)]

    partners = topology.strong_fragments[fragment_index]
    # The vertically stacked thin factors M_G satisfy
    # M_S^H M_S = sum_G W_G.  Its economy SVD therefore gives the occupied
    # union without constructing or diagonalizing an nocc x nocc weight.
    partner_overlap = np.vstack([
        topology.fragment_occupied_data[partner].iao_occ_overlap
        for partner in partners
    ])
    _, singular_value, vh = scipy.linalg.svd(
        partner_overlap, full_matrices=False
    )
    keep = singular_value**2 > thresholds.occupied_weight
    if not np.any(keep):
        keep[np.argmax(singular_value)] = True
    occupied_candidate = topology.occupied_coeff @ vh.conj().T[:, keep]
    occupied_local = util.project_mo(occupied_candidate, s21, s22)
    occupied_local = _metric_orthonormalize(
        occupied_local, s22, thresholds.metric_rank
    )
    occupied_energy, occupied_local = _semicanonicalize(
        occupied_local, fock22
    )
    if occupied_local.shape[1] == 0:
        raise RuntimeError(f"fragment {fragment_index} ED has no occupied space")

    virtual_candidate = dlno_pao.pao_overlap_with_domain(
        mol,
        topology.pao_coeff,
        extended_atoms,
        p_domain=center_atoms,
        ao2pao_map=topology.ao2pao_map,
        s1e=s1e,
        ovlp_thr=thresholds.domain_pao,
    )
    if virtual_candidate.shape[1]:
        completeness = _compute_av_numpy(
            mol, virtual_candidate, s1e=s1e, atmlst=extended_atoms
        )
        virtual_candidate = virtual_candidate[
            :, completeness > thresholds.ed_pao
        ]
    if virtual_candidate.shape[1]:
        virtual_local = util.project_mo(virtual_candidate, s21, s22)
        virtual_local = util.orthogonalize(
            occupied_local, virtual_local, s22
        )
        virtual_local = _metric_orthonormalize(
            virtual_local, s22, thresholds.metric_rank
        )
        virtual_energy, virtual_local = _semicanonicalize(
            virtual_local, fock22
        )
    else:
        virtual_local = np.zeros((ao_idx.size, 0), dtype=occupied_local.dtype)
        virtual_energy = np.zeros((0,), dtype=float)
    if virtual_local.shape[1] == 0:
        raise RuntimeError(f"fragment {fragment_index} ED has no virtual space")

    target_iao = topology.fragment_occupied_data[fragment_index].iao_coeff
    partner_iao = np.hstack([
        topology.fragment_occupied_data[partner].iao_coeff
        for partner in partners
    ])
    target_projection = target_iao.conj().T @ s1e[:, ao_idx] @ occupied_local
    partner_projection = (
        partner_iao.conj().T @ s1e[:, ao_idx] @ occupied_local
    )
    target_weight = target_projection.conj().T @ target_projection
    partner_weight = partner_projection.conj().T @ partner_projection

    return {
        "ao_idx": ao_idx,
        "extended_atoms": extended_atoms,
        "center_atoms": center_atoms,
        "occupied_coeff": occupied_local,
        "virtual_coeff": virtual_local,
        "occupied_energy": occupied_energy,
        "virtual_energy": virtual_energy,
        "target_projection": target_projection,
        "target_weight": target_weight,
        "partner_weight": partner_weight,
    }


def _domain_lov(mf, domain):
    nocc = domain["occupied_coeff"].shape[1]
    local_coeff = np.hstack([
        domain["occupied_coeff"], domain["virtual_coeff"]
    ])
    return np.asarray(lno_base.get_local_Lov(
        mf,
        local_coeff,
        nocc,
        domain["extended_atoms"],
        integral_direct=True,
    ))


def evaluate_iao_fragment_mp2(mf, topology):
    """Evaluate exact strong-pair ED MP2 plus weak multipole OS-MP2.

    This energy-validation routine uses the reference continuous quantities
    cached in ``topology``.  It is therefore not yet a nuclear-gradient entry
    point; see :class:`IAOFragmentTopology` for the required fixed-topology
    rebuild boundary.
    """
    if not isinstance(topology, IAOFragmentTopology):
        raise TypeError("topology must be IAOFragmentTopology")

    setup_start = time.perf_counter()
    fragment_results = []
    strong_total = 0.0
    strong_os = 0.0
    strong_ss = 0.0
    weak_os = 0.0
    # This energy-validation evaluator intentionally consumes the continuous
    # reference arrays already stored with the topology.  Rebuilding the
    # global overlap and especially the DF Fock matrix here duplicated a
    # whole-molecule J/K calculation after topology construction and polluted
    # the post-SCF ED timing.  A future gradient entry point must reconstruct
    # these quantities inside its differentiable fixed-topology boundary.
    s1e = np.asarray(topology.s1e)
    fock = np.asarray(topology.fock)
    setup_seconds = time.perf_counter() - setup_start
    for fragment_index in range(len(topology.frag_lolist)):
        stage_start = time.perf_counter()
        domain = _build_fragment_domain_orbitals(
            mf,
            topology,
            fragment_index,
            s1e=s1e,
            fock=fock,
        )
        ed_orbital_seconds = time.perf_counter() - stage_start
        if fragment_index == 0:
            ed_orbital_seconds += setup_seconds

        stage_start = time.perf_counter()
        lov = _domain_lov(mf, domain)
        local_ri_lov_seconds = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        energy = fragment_pair_energy_from_lov(
            lov,
            domain["occupied_energy"],
            domain["virtual_energy"],
            domain["target_weight"],
            domain["partner_weight"],
            target_factor=domain["target_projection"],
            max_memory_mb=topology.thresholds.mp2_block_memory_mb,
        )
        weighted_ed_mp2_seconds = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        fragment_strong = float(np.real(energy.total))
        fragment_os = float(np.real(energy.opposite_spin))
        fragment_ss = float(np.real(energy.same_spin))

        weak_mask = ~topology.strong_mask[fragment_index]
        # ``pair_energy`` is a symmetric unordered-pair multipole estimate.
        # Half of each row assigns one half to either endpoint, so summing all
        # fragment rows counts every weak pair once.
        fragment_weak = 0.5 * float(np.sum(
            topology.weak_pair_energy[fragment_index, weak_mask]
        ))
        strong_total += fragment_strong
        strong_os += fragment_os
        strong_ss += fragment_ss
        weak_os += fragment_weak
        weak_bookkeeping_seconds = time.perf_counter() - stage_start
        fragment_results.append(IAOFragmentDomainResult(
            fragment_index=fragment_index,
            strong_fragments=np.asarray(
                topology.strong_fragments[fragment_index]
            ),
            pao_center_atoms=domain["center_atoms"],
            extended_atoms=domain["extended_atoms"],
            n_domain_ao=int(domain["ao_idx"].size),
            n_domain_occ=int(domain["occupied_coeff"].shape[1]),
            n_domain_vir=int(domain["virtual_coeff"].shape[1]),
            target_weight_trace=float(np.trace(
                domain["target_weight"]
            ).real),
            partner_weight_trace=float(np.trace(
                domain["partner_weight"]
            ).real),
            strong_total=fragment_strong,
            strong_opposite_spin=fragment_os,
            strong_same_spin=fragment_ss,
            weak_multipole_opposite_spin=fragment_weak,
            timing=IAOFragmentMP2Timing(
                ed_orbital_seconds=ed_orbital_seconds,
                local_ri_lov_seconds=local_ri_lov_seconds,
                weighted_ed_mp2_seconds=weighted_ed_mp2_seconds,
                weak_bookkeeping_seconds=weak_bookkeeping_seconds,
            ),
        ))

    aggregate_timing = IAOFragmentMP2Timing(
        ed_orbital_seconds=sum(
            fragment.timing.ed_orbital_seconds
            for fragment in fragment_results
        ),
        local_ri_lov_seconds=sum(
            fragment.timing.local_ri_lov_seconds
            for fragment in fragment_results
        ),
        weighted_ed_mp2_seconds=sum(
            fragment.timing.weighted_ed_mp2_seconds
            for fragment in fragment_results
        ),
        weak_bookkeeping_seconds=sum(
            fragment.timing.weak_bookkeeping_seconds
            for fragment in fragment_results
        ),
    )

    return IAOFragmentMP2Result(
        e_corr=strong_total + weak_os,
        e_strong=strong_total,
        e_weak_multipole_os=weak_os,
        e_strong_os=strong_os,
        e_strong_ss=strong_ss,
        fragments=tuple(fragment_results),
        topology=topology,
        timing=aggregate_timing,
    )


def kernel(mf, **kwargs):
    """Build IAO-fragment topology and evaluate the ED-MP2 energy."""
    topology = build_iao_fragment_topology(mf, **kwargs)
    return evaluate_iao_fragment_mp2(mf, topology)
