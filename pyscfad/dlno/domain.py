"""Atom-domain topology and pair screening for IAO or Boys targets.

The reference calculation follows four stages: prepare target orbitals and
PAOs, build BP/primary domains, screen pairs, and union the strong partners'
atom lists into extended domains. Only the final stage creates DomainTopology.
Differentiable reconstruction of the selected spaces belongs to dlno_base."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
import numbers

import numpy
from pyscf.mp.mp2 import _mo_splitter
import scipy.linalg

from pyscfad.lno import df as lno_df
from pyscfad.lno.tools import autofrag, map_lo_to_frag
from pyscfad.mp import dfmp2

from . import multipole_numpy as static_multipole, pao as dlno_pao, tools
from .fragment_mp2 import build_fragment_occupied_data, fragment_pair_energy_from_lov
from .targets import build_targets, resolve_target_options, validate_boys_target_groups
from ._selection import _reference_occupied_space


__all__ = [
    "DLNOThresholds",
    "DomainTopology",
    "build_domain_topology",
]


def get_bp_domain(mol, mos, s1e=None, bp_thr=0.999,
                  q_thr=None, atmlst=None):
    """BP domains based on partial Mulliken charges.
    """
    if s1e is None:
        s1e = mol.intor_symmetric('int1e_ovlp')
    if q_thr is None:
        q_thr = min(0.05, 5*(1-bp_thr))
    if atmlst is None:
        atmlst = numpy.arange(mol.natm)

    mos = numpy.asarray(mos)
    if mos.ndim == 1:
        mos = mos.reshape(-1,1)
    assert mos.ndim == 2
    nao, nmo = mos.shape
    # TODO project MOs onto smaller basis
    assert nao == mol.nao

    rr = atom_distance(mol, atmlst)
    aoslices = mol.aoslice_by_atom()[:,2:]
    bp_atmlst = []

    s1e = numpy.asarray(s1e)
    # Gross orbital populations for all orbitals at once. This matches the
    # per-orbital expression sum_v C_ui S_uv C_vi used below, but avoids
    # rebuilding an AO x AO outer product for every orbital.
    gop = mos * (s1e @ mos)
    q_by_atom = numpy.abs(numpy.asarray([
        numpy.sum(gop[slice(*aoslices[a])], axis=0) for a in atmlst
    ]))
    sorted_atom_idx = numpy.argsort(rr, axis=1)

    for i in range(nmo):
        orbi = mos[:, i]
        q = q_by_atom[:, i]

        _atms = atmlst[q > q_thr]
        av = _compute_av_numpy(mol, orbi, s1e, _atms)

        if av < bp_thr:
            center_id = int(numpy.argsort(-q)[0])
            _sorted_atm_idx = sorted_atom_idx[center_id]

            for iatm in _sorted_atm_idx[1:]:
                a = atmlst[iatm]
                if a not in _atms:
                    _atms = numpy.append(_atms, a)
                    av = _compute_av_numpy(mol, orbi, s1e, _atms)
                    if av >= bp_thr:
                        break
        bp_atmlst.append(_atms)

    return tools.list_to_array(bp_atmlst)


def _fragment_trace_completeness_numpy(mol, occupied_block, s1e=None,
                                       atmlst=None):
    """Fraction of a fragment occupied block recovered on an AO domain.

    For an AO-coefficient block ``X``, the recovered population is

    ``Tr[X^H S[:,D] S[D,D]^-1 S[D,:] X] / Tr[X^H S X]``.

    Unlike applying the scalar BP criterion column by column, this quantity is
    invariant to a unitary rotation among the columns of ``X``.  The columns do
    not need to be normalized or mutually orthogonal; consequently weak
    occupied components of an IAO fragment retain their physical weights.
    """
    if s1e is None:
        s1e = mol.intor_symmetric('int1e_ovlp')
    if atmlst is None:
        atmlst = numpy.arange(mol.natm)

    occupied_block = numpy.asarray(occupied_block)
    if occupied_block.ndim == 1:
        occupied_block = occupied_block.reshape(-1, 1)
    if occupied_block.ndim != 2 or occupied_block.shape[0] != mol.nao:
        raise ValueError('occupied_block must have shape (mol.nao, nocc_block).')
    if occupied_block.shape[1] == 0:
        raise ValueError('occupied_block must contain at least one column.')

    s1e = numpy.asarray(s1e)
    sx = s1e @ occupied_block
    total = numpy.real(numpy.sum(occupied_block.conj() * sx))
    scale = max(1.0, float(numpy.linalg.norm(occupied_block))**2)
    if not numpy.isfinite(total) or total <= numpy.finfo(float).eps * scale:
        raise ValueError('occupied_block has a vanishing or invalid S norm.')

    atmlst = numpy.asarray(atmlst, dtype=numpy.int32).ravel()
    if atmlst.size == 0:
        return 0.0
    ao_idx = tools.ao_index_by_atom(mol, atmlst)
    v = sx[ao_idx]
    recovered_coeff = numpy.linalg.solve(s1e[numpy.ix_(ao_idx, ao_idx)], v)
    recovered = numpy.real(numpy.sum(v.conj() * recovered_coeff))
    return float(recovered / total)


def get_fragment_bp_domain(mol, occupied_blocks, s1e=None, bp_thr=0.999,
                           q_thr=None, atmlsts=None):
    """BP atom domains for occupied components of IAO fragments.

    Parameters
    ----------
    occupied_blocks : sequence of arrays
        AO-coefficient blocks ``X_F = P_occ A_F``, one for each fragment.
        The blocks need not be normalized or mutually orthogonal.
    atmlsts : sequence of atom-index arrays, optional
        Fixed seed atoms for each fragment.  When omitted, atoms whose
        rotation-invariant aggregate Mulliken population exceeds ``q_thr`` are
        used as seeds.

    Notes
    -----
    Starting from the seed set, the domain is enlarged one atom at a time.  The
    next atom minimizes its distance to the current domain; aggregate fragment
    population and atom index provide deterministic tie breaking.  Selection
    stops when the trace-recovered fragment population reaches ``bp_thr``.
    """
    if s1e is None:
        s1e = mol.intor_symmetric('int1e_ovlp')
    if q_thr is None:
        q_thr = min(0.05, 5 * (1 - bp_thr))
    if not 0 <= bp_thr <= 1:
        raise ValueError('bp_thr must lie between zero and one.')

    # A single dense rank-2 block is a useful shorthand for one fragment.  A
    # rank-3 array and an object/list sequence continue to mean many blocks.
    if isinstance(occupied_blocks, numpy.ndarray) and occupied_blocks.ndim == 2:
        occupied_blocks = [occupied_blocks]
    else:
        occupied_blocks = list(occupied_blocks)

    nfrag = len(occupied_blocks)
    if atmlsts is None:
        atmlsts = [None] * nfrag
    else:
        atmlsts = list(atmlsts)
        if len(atmlsts) != nfrag:
            raise ValueError('atmlsts must have one seed atom list per block.')

    s1e = numpy.asarray(s1e)
    rr = atom_distance(mol)
    aoslices = mol.aoslice_by_atom()[:, 2:]
    all_atoms = numpy.arange(mol.natm, dtype=numpy.int32)
    fragment_domains = []

    for occupied_block, seed_atoms in zip(occupied_blocks, atmlsts):
        occupied_block = numpy.asarray(occupied_block)
        if occupied_block.ndim == 1:
            occupied_block = occupied_block.reshape(-1, 1)
        if occupied_block.ndim != 2 or occupied_block.shape[0] != mol.nao:
            raise ValueError(
                'Each occupied block must have shape (mol.nao, nocc_block).'
            )
        if occupied_block.shape[1] == 0:
            raise ValueError('Occupied fragment blocks may not be empty.')

        sx = s1e @ occupied_block
        # Summing over the complete column block before taking the magnitude
        # makes this population invariant under X_F -> X_F U_F.
        q = numpy.abs(numpy.asarray([
            numpy.sum(
                occupied_block[slice(*aoslices[a])].conj()
                * sx[slice(*aoslices[a])]
            )
            for a in all_atoms
        ]))

        if seed_atoms is None:
            selected = all_atoms[q > q_thr]
        else:
            selected = numpy.unique(numpy.asarray(seed_atoms, dtype=numpy.int32).ravel())
            if numpy.any((selected < 0) | (selected >= mol.natm)):
                raise ValueError('Fragment seed atom index is out of range.')
        if selected.size == 0:
            selected = numpy.asarray([int(numpy.argmax(q))], dtype=numpy.int32)

        completeness = _fragment_trace_completeness_numpy(
            mol, occupied_block, s1e=s1e, atmlst=selected
        )
        while completeness < bp_thr and selected.size < mol.natm:
            remaining = numpy.setdiff1d(all_atoms, selected, assume_unique=True)
            distance_to_domain = numpy.min(rr[numpy.ix_(remaining, selected)], axis=1)
            # numpy.lexsort uses the last key as primary: distance first, then
            # larger population, and finally atom index.
            order = numpy.lexsort((remaining, -q[remaining], distance_to_domain))
            selected = numpy.sort(numpy.append(selected, remaining[order[0]]))
            completeness = _fragment_trace_completeness_numpy(
                mol, occupied_block, s1e=s1e, atmlst=selected
            )

        fragment_domains.append(selected)

    return tools.list_to_array(fragment_domains)


def _compute_av_numpy(mol, mo, s1e=None, atmlst=None):
    """NumPy BP value for non-differentiable domain topology selection."""
    if s1e is None:
        s1e = mol.intor_symmetric('int1e_ovlp')
    if atmlst is None:
        atmlst = numpy.arange(mol.natm)

    mo = numpy.asarray(mo)
    ao_idx = tools.ao_index_by_atom(mol, atmlst)
    s1e = numpy.asarray(s1e)
    v = s1e[ao_idx] @ mo
    a = numpy.linalg.solve(s1e[numpy.ix_(ao_idx, ao_idx)], v)
    return numpy.sum(a * v, axis=0)


def get_primary_domain(mol, lmo_bp_domain, pao_bp_domain, ao2pao_map=None):
    """Extend LMO BP domain by PAO BP domains.
    """
    if ao2pao_map is None:
        ao2pao_map = numpy.arange(mol.nao)

    nocc = len(lmo_bp_domain)
    aoslices = mol.aoslice_by_atom()[:,2:]
    pd_atmlst = []

    for i in range(nocc):
        _atms = numpy.empty((0,), dtype=numpy.int32)
        for a in lmo_bp_domain[i]:
            _tmp = ao2pao_map[slice(*aoslices[a])]
            pao_idx = _tmp[_tmp >= 0]
            _atms = numpy.union1d(_atms, reduce(numpy.union1d, pao_bp_domain[pao_idx]))
        pd_atmlst.append(numpy.union1d(lmo_bp_domain[i], _atms))

    return tools.list_to_array(pd_atmlst)


def atom_distance(mol, atmlst=None):
    """Atomic distance array
    """
    if atmlst is None:
        atmlst = numpy.arange(mol.natm)
    coords = numpy.asarray(mol.atom_coords())[numpy.asarray(atmlst)].reshape(-1,3)
    diff = coords[:,None,:] - coords[None,:,:]
    return numpy.linalg.norm(diff, axis=-1)


@dataclass(frozen=True)
class DLNOThresholds:
    """Thresholds for IAO-fragment or Boys-orbital domain construction.

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

    The LIS MP2-density path always stores local ``Lov`` in the caller's
    rank-private HDF5 scratch.  Its virtual block width is automatic by default.
    ``mp2_block_nvir`` is an advanced exact-width override and
    ``mp2_block_memory_mb`` is an optional workspace-target override; neither
    setting is a hard process-memory cap.
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
    # The MP2 selection-density block width is automatic unless an advanced
    # workspace target or an exact width is requested explicitly.
    mp2_block_memory_mb: float | None = None
    mp2_block_nvir: int | None = None

    def __post_init__(self):
        if (
            self.mp2_block_memory_mb is not None
            and self.mp2_block_memory_mb <= 0.0
        ):
            raise ValueError("mp2_block_memory_mb must be positive")
        if (
            self.mp2_block_nvir is not None
            and (
                not isinstance(self.mp2_block_nvir, numbers.Integral)
                or isinstance(self.mp2_block_nvir, bool)
                or self.mp2_block_nvir <= 0
            )
        ):
            raise ValueError("mp2_block_nvir must be a positive integer")


@dataclass(frozen=True)
class DomainTopology:
    """Completed reference domains, pair graph, and orbital data.

    Coefficients use the global AO basis; occupied/virtual energies belong to
    the active HF spaces. BP, primary, and extended domains are atom-index
    arrays in fragment order. Pair matrices use that same fragment order.

    This record supports reference energies and discrete selection in
    _selection.build_domain_selections. Geometry-dependent arrays are rebuilt
    by dlno_base for derivatives; the whole record must not be stop-gradiented.
    ``iao_coeff`` is the historical field name for either target basis.
    """

    frozen: object
    iao_coeff: numpy.ndarray
    frag_lolist: tuple
    frag_atmlist: tuple
    occupied_coeff: numpy.ndarray
    virtual_coeff: numpy.ndarray
    occupied_energy: numpy.ndarray
    virtual_energy: numpy.ndarray
    s1e: numpy.ndarray
    fock: numpy.ndarray
    fragment_occupied_data: tuple
    pao_coeff: numpy.ndarray
    ao2pao_map: numpy.ndarray
    pao_bp_domain: numpy.ndarray
    compact_bp_domain: numpy.ndarray
    primary_bp_domain: numpy.ndarray
    primary_domain: numpy.ndarray
    tight_bp_domain: numpy.ndarray
    pair_energy_model: str
    pair_energy: numpy.ndarray
    weak_pair_energy: numpy.ndarray
    forced_strong_mask: numpy.ndarray
    strong_mask: numpy.ndarray
    strong_fragments: tuple
    pao_center_domain: numpy.ndarray
    extended_domain: numpy.ndarray
    thresholds: DLNOThresholds
    lo_type: str = "iao"
    lo_kwargs: dict | None = None

    @property
    def target_coeff(self):
        """AO target coefficients (historically stored as ``iao_coeff``)."""
        return self.iao_coeff

    @property
    def strong_lmo_indices(self):
        """Boys column indices, distinct from strong target positions."""
        if self.lo_type != "boys":
            raise ValueError("strong_lmo_indices is defined only for Boys targets")
        return tuple(numpy.asarray([self.frag_lolist[p][0] for p in partners],
                                dtype=numpy.int32)
                     for partners in self.strong_fragments)


@dataclass(frozen=True)
class _PrimaryDomains:
    """Reference inputs available before pair screening.

    All fields are also carried by the completed DomainTopology. Keeping the
    screening input separate lets the final topology be assembled once, after
    the pair graph and extended domains are known. No arrays are copied here.
    """

    occupied_coeff: numpy.ndarray
    occupied_energy: numpy.ndarray
    s1e: numpy.ndarray
    fock: numpy.ndarray
    fragment_occupied_data: tuple
    frag_atmlist: tuple
    pao_coeff: numpy.ndarray
    ao2pao_map: numpy.ndarray
    compact_bp_domain: numpy.ndarray
    primary_bp_domain: numpy.ndarray
    primary_domain: numpy.ndarray
    thresholds: DLNOThresholds
    lo_type: str


def _as_index_tuple(index_lists):
    return tuple(
        numpy.asarray(indices, dtype=numpy.int32).reshape(-1)
        for indices in index_lists
    )


def _union_index_lists(index_lists):
    arrays = [
        numpy.asarray(indices, dtype=numpy.int32).reshape(-1)
        for indices in index_lists
        if numpy.asarray(indices).size
    ]
    if not arrays:
        return numpy.zeros((0,), dtype=numpy.int32)
    return reduce(numpy.union1d, arrays).astype(numpy.int32, copy=False)


def _active_orbitals(mf, frozen):
    pt = dfmp2.MP2(mf, frozen=frozen)
    eris = pt.ao2mo()
    coeff = numpy.asarray(eris.mo_coeff)
    energy = numpy.asarray(eris.mo_energy)
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
    coeff = numpy.asarray(pt.mo_coeff)
    blocks = [coeff[:, masks[index]] for index in (0, 1, 3)]
    blocks = [block for block in blocks if block.shape[1]]
    if not blocks:
        return numpy.zeros((coeff.shape[0], 0), dtype=coeff.dtype)
    return numpy.hstack(blocks)


def _validate_fragment_atom_lists(mol, frag_atmlist, nfrag):
    if frag_atmlist is None:
        return tuple([None] * nfrag)
    if len(frag_atmlist) != nfrag:
        raise ValueError("frag_atmlist and frag_lolist must have equal length")
    output = []
    for atoms in frag_atmlist:
        atoms = numpy.unique(numpy.asarray(atoms, dtype=numpy.int32).reshape(-1))
        if numpy.any((atoms < 0) | (atoms >= mol.natm)):
            raise ValueError("fragment atom index is out of range")
        output.append(atoms)
    return tuple(output)


def _physical_pair_matrix(directed):
    """Convert directed fragment contributions to an unordered-pair score."""
    directed = numpy.asarray(directed)
    physical = directed + directed.T.conj()
    diagonal = numpy.diag_indices_from(physical)
    physical[diagonal] = directed[diagonal]
    return numpy.asarray(physical.real)


def _exact_global_fragment_pairs(pt, occupied_data, max_memory_mb=256.0):
    eris = pt.ao2mo()
    nocc = int(pt.nocc)
    nvir = int(pt.nmo - nocc)
    lov = numpy.asarray(
        pt.loop_ao2mo(eris.mo_coeff, nocc, with_t2=False)
    ).reshape(-1, nocc, nvir)
    weights = numpy.stack([data.occupied_weight for data in occupied_data])
    directed = fragment_pair_energy_from_lov(
        lov,
        numpy.asarray(eris.mo_energy[:nocc]),
        numpy.asarray(eris.mo_energy[nocc:]),
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
    coeff = numpy.asarray(coeff)
    if coeff.ndim != 2:
        raise ValueError("coefficient block must be rank two")
    if coeff.shape[1] == 0:
        return numpy.zeros((coeff.shape[0], 0), dtype=coeff.dtype)
    metric = coeff.conj().T @ numpy.asarray(overlap) @ coeff
    metric = 0.5 * (metric + metric.conj().T)
    eigenvalue, eigenvector = scipy.linalg.eigh(metric)
    keep = eigenvalue > threshold
    if not numpy.any(keep):
        return numpy.zeros((coeff.shape[0], 0), dtype=coeff.dtype)
    return coeff @ (
        eigenvector[:, keep] / numpy.sqrt(eigenvalue[keep])[None, :]
    )


def _normalize_metric_columns(coeff, overlap, threshold):
    coeff = numpy.asarray(coeff)
    norm2 = numpy.real(numpy.einsum(
        "ui,uv,vi->i", coeff.conj(), numpy.asarray(overlap), coeff,
        optimize=True,
    ))
    keep = norm2 > threshold
    return coeff[:, keep] / numpy.sqrt(norm2[keep])[None, :], keep


def _semicanonicalize(coeff, fock):
    coeff = numpy.asarray(coeff)
    if coeff.shape[1] == 0:
        return numpy.zeros((0,), dtype=float), coeff
    projected = coeff.conj().T @ numpy.asarray(fock) @ coeff
    projected = 0.5 * (projected + projected.conj().T)
    energy, rotation = scipy.linalg.eigh(projected)
    return numpy.asarray(energy.real), coeff @ rotation


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
    occupied_weight = numpy.asarray(occupied_weight)
    occupied_energy = numpy.asarray(occupied_energy)
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
    order = numpy.argsort(eigenvalue)[::-1]
    eigenvalue = numpy.asarray(eigenvalue[order].real)
    eigenvector = numpy.asarray(eigenvector[:, order])
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
        while stop < eigenvalue.size and numpy.isclose(
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
            pivot = int(numpy.argmax(numpy.abs(block[:, column])))
            value = block[pivot, column]
            if abs(value) > 0:
                block[:, column] *= numpy.conj(value) / abs(value)

        mode_blocks.append(block)
        weight_blocks.append(numpy.full(
            block.shape[1], numpy.mean(eigenvalue[start:stop]), dtype=float
        ))
        start = stop

    return numpy.concatenate(weight_blocks), numpy.hstack(mode_blocks)


def _fragment_screen_space(mf, primary, fragment_index):
    """Build weighted occupied modes and a PAO span in one primary domain.

    ``primary`` supplies the pre-screening fields in _PrimaryDomains; a
    completed DomainTopology has the same fields for reference diagnostics.
    Returned coefficients use the primary domain's AO rows, not global AOs.
    """
    mol = mf.mol
    thresholds = primary.thresholds
    data = primary.fragment_occupied_data[fragment_index]
    primary_atoms = numpy.asarray(
        primary.primary_domain[fragment_index], dtype=numpy.int32
    )
    primary_bp_atoms = numpy.asarray(
        primary.primary_bp_domain[fragment_index], dtype=numpy.int32
    )
    ao_idx = tools.ao_index_by_atom(mol, primary_atoms)
    s1e = numpy.asarray(primary.s1e)
    fock = numpy.asarray(primary.fock)
    s21 = s1e[ao_idx]
    s22 = s1e[numpy.ix_(ao_idx, ao_idx)]
    fock22 = fock[numpy.ix_(ao_idx, ao_idx)]

    # Boys preserves the actual localized orbital with unit screening weight.
    # IAO uses invariant spectral modes of W_F with explicit weights.
    if primary.lo_type == "boys":
        weight = numpy.ones(1)
        occupied_global = numpy.asarray(data.iao_coeff)
    else:
        weight, occupied_mode = _fragment_multipole_modes(
            data.occupied_weight,
            primary.occupied_energy,
            thresholds.occupied_weight,
        )
        occupied_global = numpy.asarray(primary.occupied_coeff) @ occupied_mode
    if occupied_global.shape[1] == 0:
        raise RuntimeError(
            f"fragment {fragment_index} has no occupied multipole modes"
        )
    occupied_local = tools.project_mo(occupied_global, s21, s22)
    occupied_local, norm_keep = _normalize_metric_columns(
        occupied_local, s22, thresholds.metric_rank
    )
    weight = weight[norm_keep]
    if occupied_local.shape[1] == 0:
        raise RuntimeError(
            f"fragment {fragment_index} has no occupied multipole modes"
        )
    occupied_energy = numpy.real(numpy.einsum(
        "ui,uv,vi->i", occupied_local.conj(), fock22, occupied_local,
        optimize=True,
    ))

    # Pair screening uses the primary-domain construction: PAOs centered on
    # the occupied BP_PD atoms are represented in the larger primary AO
    # domain.  The more compact BP_occ center list is reserved for the actual
    # improved ED construction below.
    candidate = dlno_pao.pao_overlap_with_domain(
        mol,
        primary.pao_coeff,
        primary_bp_atoms,
        ao2pao_map=primary.ao2pao_map,
        s1e=s1e,
        ovlp_thr=thresholds.domain_pao,
    )
    if candidate.shape[1]:
        completeness = _compute_av_numpy(
            mol, candidate, s1e=s1e, atmlst=primary_atoms
        )
        candidate = candidate[:, completeness > thresholds.bp_pao]
    if candidate.shape[1]:
        candidate = tools.project_mo(candidate, s21, s22)
        occupied_span = _metric_orthonormalize(
            occupied_local, s22, thresholds.metric_rank
        )
        candidate = tools.orthogonalize(occupied_span, candidate, s22)
        candidate = _metric_orthonormalize(
            candidate, s22, thresholds.metric_rank
        )
        virtual_energy, candidate = _semicanonicalize(candidate, fock22)
    else:
        candidate = numpy.zeros((ao_idx.size, 0))
        virtual_energy = numpy.zeros((0,))
    if candidate.shape[1] == 0:
        return None

    if primary.lo_type == "boys" and (
        not numpy.all(numpy.isfinite(virtual_energy))
        or not numpy.all(numpy.isfinite(occupied_energy))
        or numpy.any(virtual_energy[:, None] <= occupied_energy[None, :])
    ):
        raise RuntimeError(
            f"Boys target {fragment_index} has invalid screening denominators"
        )

    # Screening uses concrete moments. Weak-pair gradients rebuild these
    # moments in dlno_base.build_weak_multipole_screen.
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


def _multipole_fragment_pairs(mf, primary):
    nfrag = len(primary.fragment_occupied_data)
    screen = [
        _fragment_screen_space(mf, primary, fragment)
        for fragment in range(nfrag)
    ]
    pair = numpy.zeros((nfrag, nfrag), dtype=float)
    forced_strong = numpy.eye(nfrag, dtype=bool)
    order = primary.thresholds.multipole_order
    near_distance = primary.thresholds.near_pair_distance
    coordinates = numpy.asarray(mf.mol.atom_coords())
    for left in range(nfrag):
        for right in range(left):
            if screen[left] is None or screen[right] is None:
                forced_strong[left, right] = True
                forced_strong[right, left] = True
                continue
            left_atoms = primary.frag_atmlist[left]
            right_atoms = primary.frag_atmlist[right]
            if left_atoms is None or numpy.asarray(left_atoms).size == 0:
                left_atoms = primary.compact_bp_domain[left]
            if right_atoms is None or numpy.asarray(right_atoms).size == 0:
                right_atoms = primary.compact_bp_domain[right]
            left_atoms = numpy.asarray(left_atoms, dtype=numpy.int32)
            right_atoms = numpy.asarray(right_atoms, dtype=numpy.int32)
            separation = numpy.linalg.norm(
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
            value = numpy.sum(
                numpy.asarray(orbital_pair)
                * lhs["weights"][:, None]
                * rhs["weights"][None, :]
            )
            pair[left, right] = pair[right, left] = float(value)
            if not numpy.isfinite(value):
                if primary.lo_type == "boys":
                    raise RuntimeError(
                        "nonfinite Boys screening pair estimate "
                        f"for targets {left}, {right}"
                    )
                forced_strong[left, right] = True
                forced_strong[right, left] = True
                pair[left, right] = pair[right, left] = 0.0
    return pair, forced_strong


def _check_domain_inputs(mf, thresholds):
    """Validate the RHF reference and domain cutoffs before construction."""
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
    if thresholds.multipole_order not in (2, 3, 4):
        raise ValueError("multipole_order must be 2, 3, or 4")
    if getattr(mf, "with_df", None) is None:
        raise ValueError("DLNO requires a density-fitted SCF object")
    mo_coeff_input = numpy.asarray(mf.mo_coeff)
    mo_occ_input = numpy.asarray(mf.mo_occ)
    if numpy.iscomplexobj(mo_coeff_input):
        raise NotImplementedError(
            "DLNO currently supports real orbitals only"
        )
    if mo_coeff_input.ndim != 2 or not numpy.all(
        (numpy.abs(mo_occ_input) < 1e-12)
        | (numpy.abs(mo_occ_input - 2.0) < 1e-12)
    ):
        raise NotImplementedError(
            "DLNO currently supports restricted closed-shell "
            "references only"
        )


def _build_fragment_targets(mol, occupied, target_coeff, frag_lolist,
                            frag_atmlist, s1e, thresholds, lo_type, lo_kwargs):
    """Build targets, their fragment partition, and additive occupied weights."""
    if target_coeff is None:
        target_coeff = numpy.asarray(build_targets(
            mol, occupied, lo_type=lo_type, lo_kwargs=lo_kwargs
        ))
    else:
        target_coeff = numpy.asarray(target_coeff)
    if numpy.iscomplexobj(target_coeff):
        raise NotImplementedError(
            "DLNO currently supports real target orbitals only"
        )

    auto_frag_atoms = None
    if lo_type == "boys":
        if target_coeff.shape != occupied.shape or not numpy.allclose(
            target_coeff.T @ s1e @ target_coeff,
            numpy.eye(occupied.shape[1]), atol=1e-8, rtol=1e-8,
        ):
            raise ValueError(
                "Boys target columns must be an orthonormal active occupied frame"
            )
        frag_lolist = validate_boys_target_groups(frag_lolist, occupied.shape[1])
    elif frag_lolist is None:
        auto_frag_atoms = autofrag(mol)
        frag_lolist = map_lo_to_frag(
            mol, target_coeff, auto_frag_atoms, verbose=0
        )
    frag_lolist = _as_index_tuple(frag_lolist)
    if any(indices.size == 0 for indices in frag_lolist):
        raise ValueError("each fragment must contain at least one target orbital")
    if frag_atmlist is None and auto_frag_atoms is not None:
        frag_atmlist = auto_frag_atoms
    frag_atmlist = _validate_fragment_atom_lists(
        mol, frag_atmlist, len(frag_lolist)
    )

    occupied_data = build_fragment_occupied_data(
        occupied, target_coeff, frag_lolist, s1e,
        svd_thr=thresholds.occupied_weight ** 0.5,
    )
    for fragment, data in enumerate(occupied_data):
        weight_trace = float(numpy.trace(data.occupied_weight).real)
        if weight_trace <= numpy.finfo(float).eps:
            raise ValueError(
                f"fragment {fragment} has zero active occupied weight; "
                "merge it with a neighboring fragment or revise the target "
                "partition"
            )
    weight_sum = sum(data.occupied_weight for data in occupied_data)
    if not numpy.allclose(
        weight_sum, numpy.eye(occupied.shape[1]), atol=1e-8, rtol=1e-8
    ):
        raise ValueError(
            "the complete fragment weights do not resolve the active "
            "occupied identity"
        )

    return target_coeff, frag_lolist, frag_atmlist, occupied_data


def _occupied_bp_domains(mol, target_coeff, frag_lolist, occupied_data,
                         frag_atmlist, s1e, thresholds, lo_type):
    """Compact, primary-screening, and tight-ED occupied BP atom lists.

    Boys applies scalar completeness to each actual LMO. IAO applies trace
    completeness to the raw occupied fragment block, preserving its weights.
    The three levels use the same policy and differ only in recovery cutoff.
    """
    cutoffs = (thresholds.bp_occ, thresholds.bp_primary, thresholds.bp_ed)
    if lo_type == "boys":
        # Fragment order can differ from the underlying target-column order.
        coeff = target_coeff[:, [indices[0] for indices in frag_lolist]]
        return tuple(get_bp_domain(mol, coeff, s1e=s1e, bp_thr=cutoff)
                     for cutoff in cutoffs)

    occupied_blocks = [data.occupied_projection for data in occupied_data]
    seed_atoms = frag_atmlist if all(a is not None for a in frag_atmlist) else None
    return tuple(get_fragment_bp_domain(
        mol, occupied_blocks, s1e=s1e, bp_thr=cutoff, atmlsts=seed_atoms,
    ) for cutoff in cutoffs)


def _screen_pairs(mf, pt, primary, model, force_full_domains):
    """Pair estimates and the symmetric strong graph in fragment order."""
    thresholds = primary.thresholds
    nfrag = len(primary.fragment_occupied_data)
    if (
        force_full_domains
        or thresholds.pair_energy <= 0.0
        or model in ("all", "all-strong")
    ):
        pair_energy = numpy.zeros((nfrag, nfrag), dtype=float)
        weak_pair_energy = numpy.zeros((nfrag, nfrag), dtype=float)
        forced_strong = numpy.ones((nfrag, nfrag), dtype=bool)
        strong_mask = numpy.ones((nfrag, nfrag), dtype=bool)
        return pair_energy, weak_pair_energy, forced_strong, strong_mask

    if model == "exact":
        _, pair_energy = _exact_global_fragment_pairs(
            pt, primary.fragment_occupied_data,
            max_memory_mb=(256.0 if thresholds.mp2_block_memory_mb is None
                           else thresholds.mp2_block_memory_mb),
        )
    # The weak correction always uses multipoles, including the diagnostic
    # route that classifies pairs using exact global MP2 estimates.
    weak_pair_energy, forced_strong = _multipole_fragment_pairs(mf, primary)
    if model != "exact":
        pair_energy = weak_pair_energy
    strong_mask = (numpy.abs(pair_energy) > thresholds.pair_energy) | forced_strong
    strong_mask = numpy.asarray(strong_mask | strong_mask.T, dtype=bool)
    numpy.fill_diagonal(strong_mask, True)
    return pair_energy, weak_pair_energy, forced_strong, strong_mask


def build_domain_topology(
    mf,
    iao_coeff=None,
    frag_lolist=None,
    frag_atmlist=None,
    *,
    frozen=None,
    thresholds=None,
    pair_energy_model="multipole",
    force_full_domains=False,
    lo_type="iao",
    lo_kwargs=None,
):
    """Build fixed BP/primary/extended topology for IAO or Boys targets.

    ``lo_type="boys"`` localizes the active occupied block and assigns one
    target per LMO. ``lo_kwargs`` is passed to the existing differentiable
    localizer. Boys atom supports are generated by scalar BP domains; atom
    fragment overrides and grouped orbital targets are rejected.

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
    lo_type, lo_kwargs = resolve_target_options(lo_type, lo_kwargs)
    if lo_type == "boys" and frag_atmlist is not None:
        raise ValueError("Boys targets do not accept molecular frag_atmlist overrides")
    if thresholds is None:
        thresholds = DLNOThresholds()
    if not isinstance(thresholds, DLNOThresholds):
        raise TypeError("thresholds must be DLNOThresholds")
    _check_domain_inputs(mf, thresholds)
    model = str(pair_energy_model).lower().replace("_", "-")
    valid_models = {
        "all", "all-strong", "exact",
        "multipole", "multipole-os", "os-multipole",
    }
    if model not in valid_models:
        raise ValueError(f"unknown pair_energy_model: {pair_energy_model}")

    # 1. Reference orbitals, target fragments, and active-virtual PAOs.
    mol = mf.mol
    s1e = numpy.asarray(mol.intor_symmetric("int1e_ovlp"))
    fock = numpy.asarray(mf.get_fock())
    pt, occupied, virtual, e_occ, e_vir = _active_orbitals(mf, frozen)
    if occupied.shape[1] == 0:
        raise ValueError("DLNO requires an active occupied space")
    if virtual.shape[1] == 0:
        raise ValueError("DLNO requires an active virtual space")
    target_coeff, frag_lolist, frag_atmlist, occupied_data = _build_fragment_targets(
        mol, occupied, iao_coeff, frag_lolist, frag_atmlist, s1e,
        thresholds, lo_type, lo_kwargs,
    )
    # Project out all occupied and user-frozen virtual MOs.
    pao_coeff, ao2pao_map = dlno_pao.pao(
        mol, _pao_projected_out_coeff(pt), s1e=s1e,
        norm_thr=thresholds.pao_norm,
    )

    # 2. BP hierarchy and primary AO domains for multipole screening.
    pao_bp = get_bp_domain(mol, pao_coeff, s1e=s1e, bp_thr=thresholds.bp_pao)
    compact_bp, primary_bp, tight_bp = _occupied_bp_domains(
        mol, target_coeff, frag_lolist, occupied_data, frag_atmlist,
        s1e, thresholds, lo_type,
    )
    primary_domain = get_primary_domain(mol, primary_bp, pao_bp, ao2pao_map)
    nfrag = len(frag_lolist)
    if force_full_domains:
        all_atoms = numpy.arange(mol.natm, dtype=numpy.int32)
        for domains in (compact_bp, primary_bp, tight_bp, primary_domain):
            domains[:] = [all_atoms.copy() for _ in range(nfrag)]
    primary = _PrimaryDomains(
        occupied_coeff=occupied,
        occupied_energy=e_occ,
        s1e=s1e,
        fock=fock,
        fragment_occupied_data=occupied_data,
        frag_atmlist=frag_atmlist,
        pao_coeff=pao_coeff,
        ao2pao_map=ao2pao_map,
        compact_bp_domain=compact_bp,
        primary_bp_domain=primary_bp,
        primary_domain=primary_domain,
        thresholds=thresholds,
        lo_type=lo_type,
    )

    # 3. Pair screening. No EDs or placeholder pair results are needed here.
    pair_energy, weak_pair_energy, forced_strong, strong_mask = _screen_pairs(
        mf, pt, primary, model, force_full_domains,
    )

    # 4. Union strong partners' compact PAO centers and tight AO supports.
    strong_fragments = tuple(
        numpy.where(strong_mask[fragment])[0].astype(numpy.int32)
        for fragment in range(nfrag)
    )
    pao_center_domain = numpy.empty(nfrag, dtype=object)
    extended_domain = numpy.empty(nfrag, dtype=object)
    for fragment, partners in enumerate(strong_fragments):
        pao_center_domain[fragment] = _union_index_lists(
            [compact_bp[partner] for partner in partners]
        )
        extended_domain[fragment] = _union_index_lists(
            [tight_bp[partner] for partner in partners]
        )

    return DomainTopology(
        **vars(primary),
        frozen=frozen,
        iao_coeff=target_coeff,
        frag_lolist=frag_lolist,
        virtual_coeff=virtual,
        virtual_energy=e_vir,
        pao_bp_domain=pao_bp,
        tight_bp_domain=tight_bp,
        pair_energy_model=model,
        pair_energy=numpy.asarray(pair_energy),
        weak_pair_energy=numpy.asarray(weak_pair_energy),
        forced_strong_mask=numpy.asarray(forced_strong),
        strong_mask=strong_mask,
        strong_fragments=strong_fragments,
        pao_center_domain=pao_center_domain,
        extended_domain=extended_domain,
        lo_kwargs=lo_kwargs,
    )


def _build_fragment_domain_orbitals(
    mf, topology, fragment_index, *, s1e=None, fock=None
):
    mol = mf.mol
    thresholds = topology.thresholds
    if s1e is None:
        s1e = numpy.asarray(mol.intor_symmetric("int1e_ovlp"))
    else:
        s1e = numpy.asarray(s1e)
    if fock is None:
        fock = numpy.asarray(mf.get_fock())
    else:
        fock = numpy.asarray(fock)
    extended_atoms = numpy.asarray(
        topology.extended_domain[fragment_index], dtype=numpy.int32
    )
    center_atoms = numpy.asarray(
        topology.pao_center_domain[fragment_index], dtype=numpy.int32
    )
    ao_idx = tools.ao_index_by_atom(mol, extended_atoms)
    s21 = s1e[ao_idx]
    s22 = s1e[numpy.ix_(ao_idx, ao_idx)]
    fock22 = fock[numpy.ix_(ao_idx, ao_idx)]

    partners = topology.strong_fragments[fragment_index]
    occupied_energy, occupied_local, _, _ = _reference_occupied_space(
        topology, fragment_index, s21, s22, fock22
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
        virtual_local = tools.project_mo(virtual_candidate, s21, s22)
        virtual_local = tools.orthogonalize(
            occupied_local, virtual_local, s22
        )
        virtual_local = _metric_orthonormalize(
            virtual_local, s22, thresholds.metric_rank
        )
        virtual_energy, virtual_local = _semicanonicalize(
            virtual_local, fock22
        )
    else:
        virtual_local = numpy.zeros((ao_idx.size, 0), dtype=occupied_local.dtype)
        virtual_energy = numpy.zeros((0,), dtype=float)
    if virtual_local.shape[1] == 0:
        raise RuntimeError(f"fragment {fragment_index} ED has no virtual space")

    target_coeff = topology.fragment_occupied_data[fragment_index].iao_coeff
    partner_coeff = numpy.hstack([
        topology.fragment_occupied_data[partner].iao_coeff
        for partner in partners
    ])
    target_projection = target_coeff.conj().T @ s1e[:, ao_idx] @ occupied_local
    partner_projection = (
        partner_coeff.conj().T @ s1e[:, ao_idx] @ occupied_local
    )
    target_weight = target_projection.conj().T @ target_projection
    partner_weight = partner_projection.conj().T @ partner_projection
    if topology.lo_type == "boys":
        retained_target_norm = float(numpy.trace(target_weight).real)
        if (not numpy.isfinite(retained_target_norm)
                or retained_target_norm <= thresholds.metric_rank):
            raise RuntimeError(
                f"fragment {fragment_index} ED lost its central Boys target "
                f"(retained norm {retained_target_norm:.3g})"
            )

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
    local_coeff = numpy.hstack([
        domain["occupied_coeff"], domain["virtual_coeff"]
    ])
    return numpy.asarray(lno_df.get_local_Lov(
        mf,
        local_coeff,
        nocc,
        domain["extended_atoms"],
        integral_direct=True,
    ))
