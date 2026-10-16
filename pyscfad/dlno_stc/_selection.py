"""Concrete Boys selections owned by STC.

BP hierarchy and PAO rank bookkeeping are copied from dlno/domain.py and
dlno/_selection.py. Only concrete labels live here; no occupied weights do.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
import numbers

import jax
import numpy
import scipy.linalg as scipy_linalg
from pyscf.mp.mp2 import _mo_splitter
from pyscf.dft.rks import KohnShamDFT
from pyscfad.mp import dfmp2
from pyscfad.lib._compilation_cache import configure_compilation_cache
from pyscfad.lno import tools
from pyscfad.lo.boys import boys
from pyscfad.scf.hf import RHF

_PAO_ORTH_THRESHOLD = 1e-6


@dataclass(frozen=True)
class BoysDomainThresholds:
    """Boys BP/PAO and pair-screen cutoffs, retaining the native defaults."""

    bp_occ: float = 0.985
    bp_primary: float = 0.999
    bp_ed: float = 0.9998
    bp_pao: float = 0.98
    pao_norm: float = 1e-4
    domain_pao: float = 1e-4
    ed_pao: float = 0.995
    metric_rank: float = 1e-10
    pair_energy: float = 1.5e-5
    near_pair_distance: float = 3.5
    multipole_order: int = 4

    def __post_init__(self):
        values = (self.bp_occ, self.bp_primary, self.bp_ed, self.bp_pao,
                  self.pao_norm, self.domain_pao, self.ed_pao, self.metric_rank,
                  self.pair_energy, self.near_pair_distance)
        if not numpy.isfinite(values).all():
            raise ValueError("Boys thresholds must be finite")
        if not 0 <= self.bp_occ <= self.bp_primary <= self.bp_ed <= 1:
            raise ValueError("occupied BP thresholds require 0 <= bp_occ <= bp_primary <= bp_ed <= 1")
        if not 0 <= self.bp_pao <= 1 or not 0 <= self.ed_pao <= 1:
            raise ValueError("BP/ED PAO completeness cutoffs must lie between zero and one")
        if min(self.pao_norm, self.domain_pao, self.metric_rank,
               self.pair_energy, self.near_pair_distance) < 0:
            raise ValueError("domain and pair cutoffs must be non-negative")
        if self.multipole_order not in (2, 3, 4):
            raise ValueError("multipole_order must be 2, 3, or 4")


@dataclass(frozen=True)
class FixedPAOSubspaceSelection:
    parent_columns: numpy.ndarray
    support_ao_indices: numpy.ndarray
    canonical_keep: numpy.ndarray
    overlap_keep: numpy.ndarray
    completeness_keep: numpy.ndarray
    metric_keep: numpy.ndarray


@dataclass(frozen=True)
class FragmentDomainSelection:
    fragment_index: int
    iao_indices: numpy.ndarray
    fragment_atoms: numpy.ndarray | None
    strong_fragments: numpy.ndarray
    extended_atoms: numpy.ndarray
    extended_ao_indices: numpy.ndarray
    pao_center_atoms: numpy.ndarray
    strong_occ_metric_keep: numpy.ndarray
    strong_virtual: FixedPAOSubspaceSelection | None
    primary_atoms: numpy.ndarray
    primary_ao_indices: numpy.ndarray
    primary_bp_atoms: numpy.ndarray
    weak_occ_norm_keep: numpy.ndarray
    weak_occ_span_metric_keep: numpy.ndarray
    weak_virtual: FixedPAOSubspaceSelection | None
    target_index: int

    @property
    def has_weak_screen(self):
        return self.weak_virtual is not None


@dataclass(frozen=True)
class DomainSelections:
    frozen: object
    thresholds: BoysDomainThresholds
    active_occ_indices: numpy.ndarray
    active_vir_indices: numpy.ndarray
    pao_projected_out_indices: numpy.ndarray
    pao_parent_ao_indices: numpy.ndarray
    ao2pao_map: numpy.ndarray
    frag_lolist: tuple[numpy.ndarray, ...]
    frag_atmlist: tuple[numpy.ndarray | None, ...]
    strong_mask: numpy.ndarray
    fragments: tuple[FragmentDomainSelection, ...]
    lo_type: str = "boys"
    lo_kwargs: dict | None = None
    target_reference_coeff: object = None
    target_reference_coords: object = None
    compact_bp_domain: tuple[numpy.ndarray, ...] = ()
    primary_bp_domain: tuple[numpy.ndarray, ...] = ()
    tight_bp_domain: tuple[numpy.ndarray, ...] = ()
    pao_bp_domain: tuple[numpy.ndarray, ...] = ()
    primary_domain: tuple[numpy.ndarray, ...] = ()
    pair_energy_model: str = "multipole"
    pair_energy: object = None
    weak_pair_energy: object = None
    forced_strong_mask: object = None


@dataclass(frozen=True)
class DomainGroup:
    """One saved preparation and its distinct local selected-target slots."""

    fragment_indices: tuple[int, ...]
    representative_index: int
    target_indices: tuple[int, ...]


def _grouping_indices(values, upper, label):
    """Check concrete integer bounds before any narrowing conversion."""
    values = numpy.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError(f"invalid {label}: expected one-dimensional integer indices")
    indices = tuple(int(value) for value in values)
    if (len(set(indices)) != len(indices)
            or any(value < 0 or (upper is not None and value >= upper)
                   for value in indices)):
        raise ValueError(f"invalid {label}: duplicate or out-of-range indices")
    return indices


def _grouping_target_slot(value, nocc, label):
    if (not isinstance(value, numbers.Integral)
            or isinstance(value, (bool, numpy.bool_)) or not 0 <= value < nocc):
        raise ValueError(f"invalid {label}: local target slot is out of range")
    return int(value)


def _strong_preparation_key(static, fragment, index, labels):
    """Use exactly the saved strong replay frame; exclude weak/target fields.

    The Boys frame projects all ordered partners, applies metric Cholesky,
    and replays the retained PAO subspace selections.
    None of those continuous operations uses the central singleton target.
    ``extended_atoms`` also determines the local auxiliary support in DF.
    Reference orbitals, frozen spaces and DF setup are shared by ``static``.
    """
    partners = _grouping_indices(fragment.strong_fragments, len(labels),
                                 f"Boys domain {index} ordered strong partners")
    slot = _grouping_target_slot(fragment.target_index, len(partners),
                                 f"Boys domain {index}")
    candidates = tuple(labels[partner] for partner in partners)
    if partners[slot] != index or candidates[slot] != labels[index]:
        raise ValueError(f"Boys domain {index} must retain its unique local target label")
    keep = _grouping_indices(fragment.strong_occ_metric_keep, len(partners),
                             f"Boys domain {index} occupied rank")
    if keep != tuple(range(len(partners))):
        raise RuntimeError(f"Boys domain {index} loses occupied target/partner labels; rebuild domains with tighter support")
    nao = len(static.ao2pao_map)
    natm = (None if static.target_reference_coords is None
            else numpy.shape(static.target_reference_coords)[0])
    supports = (
        _grouping_indices(fragment.extended_atoms, natm, "extended atoms"),
        _grouping_indices(fragment.extended_ao_indices, nao, "extended AO support"),
        _grouping_indices(fragment.pao_center_atoms, natm, "PAO center atoms"),
    )
    selection = fragment.strong_virtual
    if not isinstance(selection, FixedPAOSubspaceSelection):
        raise ValueError(f"Boys domain {index} must retain its strong PAO selection")
    parent = _grouping_indices(selection.parent_columns,
                               len(static.pao_parent_ao_indices), "PAO parents")
    support = _grouping_indices(selection.support_ao_indices, nao, "PAO AO support")
    canonical = _grouping_indices(selection.canonical_keep, len(parent), "canonical PAO keep")
    overlap = _grouping_indices(selection.overlap_keep, len(canonical), "overlap PAO keep")
    completeness = _grouping_indices(selection.completeness_keep, len(overlap), "completeness PAO keep")
    metric = _grouping_indices(selection.metric_keep, len(completeness), "metric PAO keep")
    return (partners, candidates, keep, supports,
            (parent, support, canonical, overlap, completeness, metric)), candidates, slot


def build_domain_groups(static, *, merge_identical=True) -> tuple[DomainGroup, ...]:
    """Group only identical ordered strong preparations, in first-seen order.

    Original singleton selections and weak-pair topology remain unchanged.
    Target slots are resolved in each representative's candidate-label map;
    equality of supports alone never permits a different replay frame.
    """
    if not isinstance(static, DomainSelections) or static.lo_type != "boys":
        raise TypeError("static must be STC Boys DomainSelections")
    singletons = _validate_boys_target_groups(static.frag_lolist, len(static.active_occ_indices))
    if len(static.fragments) != len(singletons):
        raise ValueError("Boys targets require a complete singleton partition")
    labels = tuple(int(indices[0]) for indices in singletons)
    # A dict preserves first occurrence; its values hold original positions.
    members, candidate_maps = {}, {}
    for index, fragment in enumerate(static.fragments):
        key, candidates, _ = _strong_preparation_key(static, fragment, index, labels)
        if not merge_identical:
            key = (index, key)
        members.setdefault(key, []).append(index)
        candidate_maps.setdefault(key, candidates)
    groups = []
    for key, indices in members.items():
        positions = {label: slot for slot, label in enumerate(candidate_maps[key])}
        targets = tuple(positions[labels[index]] for index in indices)
        if len(set(targets)) != len(targets):
            raise ValueError("grouped Boys targets must map to distinct occupied slots")
        groups.append(DomainGroup(tuple(indices), indices[0], targets))
    return tuple(groups)


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
        candidate_local = _orthogonalize(
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

    return tuple(numpy.asarray(atoms, dtype=numpy.int32) for atoms in bp_atmlst)


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
            _atms = numpy.union1d(_atms, _union_index_lists([pao_bp_domain[index] for index in pao_idx]))
        pd_atmlst.append(numpy.union1d(lmo_bp_domain[i], _atms))

    return tuple(numpy.asarray(atoms, dtype=numpy.int32) for atoms in pd_atmlst)


def atom_distance(mol, atmlst=None):
    """Atomic distance array
    """
    if atmlst is None:
        atmlst = numpy.arange(mol.natm)
    coords = numpy.asarray(mol.atom_coords())[numpy.asarray(atmlst)].reshape(-1,3)
    diff = coords[:,None,:] - coords[None,:,:]
    return numpy.linalg.norm(diff, axis=-1)


def _orthogonalize(occupied, candidate, overlap):
    return candidate - occupied @ (occupied.T.conj() @ overlap @ candidate)


def _union_index_lists(arrays):
    arrays = [numpy.asarray(array, dtype=numpy.int32).reshape(-1) for array in arrays]
    return numpy.unique(numpy.concatenate(arrays)).astype(numpy.int32) if arrays else numpy.zeros(0, dtype=numpy.int32)


def _resolve_boys_options(lo_kwargs):
    if lo_kwargs is None:
        return {}
    if not isinstance(lo_kwargs, Mapping):
        raise TypeError("lo_kwargs must be a mapping or None")
    allowed = {"init_guess", "conv_tol", "conv_tol_grad", "max_cycle", "symmetry", "gmres_options"}
    unknown = set(lo_kwargs) - allowed
    if unknown:
        raise ValueError(f"unsupported boys localization options: {sorted(unknown)}")
    return deepcopy(dict(sorted(lo_kwargs.items())))


def _build_boys_targets(mol, occupied, lo_kwargs):
    return boys(mol, occupied, **_resolve_boys_options(lo_kwargs))


def _validate_boys_target_groups(frag_lolist, nocc):
    message = "Boys targets require a disjoint complete singleton partition of occupied columns"
    if frag_lolist is None:
        return tuple(numpy.array([i], dtype=numpy.int32) for i in range(nocc))
    groups = []
    for indices in frag_lolist:
        values = numpy.asarray(indices)
        if values.ndim != 1 or values.size != 1 or values.dtype.kind not in "iu":
            raise ValueError(message)
        value = int(values[0])
        if not 0 <= value < nocc:
            raise ValueError(message)
        groups.append(numpy.array([value], dtype=numpy.int32))
    if len(groups) != nocc or sorted(int(g[0]) for g in groups) != list(range(nocc)):
        raise ValueError(message)
    return tuple(groups)


def _validate_reference(mf, frozen):
    if not isinstance(mf, RHF) or isinstance(mf, KohnShamDFT):
        raise NotImplementedError("Boys STC requires a restricted Hartree-Fock (RHF) reference")
    if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves((mf.mol, mf.mo_coeff))):
        raise ValueError("build domain selections from concrete data before tracing")
    if not getattr(mf, "converged", False):
        raise ValueError("Boys domain construction requires converged DF-RHF")
    if getattr(mf, "with_df", None) is None:
        raise ValueError("Boys domain construction requires density-fitted RHF")
    coeff, occ, energy = map(_host_array, (mf.mo_coeff, mf.mo_occ, mf.mo_energy))
    if (coeff.ndim != 2 or occ.ndim != 1 or energy.ndim != 1
            or coeff.shape[1] != occ.size or energy.size != occ.size
            or not numpy.all((abs(occ) < 1e-12) | (abs(occ - 2) < 1e-12))
            or getattr(mf.mol, "spin", 0) != 0):
        raise NotImplementedError("Boys STC requires a restricted closed-shell reference")
    if any(numpy.iscomplexobj(value) for value in (coeff, occ, energy)):
        raise NotImplementedError("Boys STC requires real orbitals")
    if not all(numpy.isfinite(value).all() for value in (coeff, occ, energy, _host_array(mf.mol.atom_coords()))):
        raise ValueError("Boys reference inputs must be finite")
    if frozen is None:
        return
    if isinstance(frozen, numbers.Integral) and not isinstance(frozen, (bool, numpy.bool_)):
        if not 0 <= frozen <= numpy.count_nonzero(occ > 0):
            raise ValueError("frozen occupied count is out of range")
        return
    if not isinstance(frozen, (list, tuple, numpy.ndarray)):
        raise ValueError("frozen must be an occupied count or distinct MO indices")
    values = numpy.asarray(frozen)
    if (values.ndim != 1
            or any(not isinstance(value, numbers.Integral)
                   or isinstance(value, (bool, numpy.bool_)) for value in frozen)
            or len(set(values.tolist())) != values.size
            or numpy.any(values < 0) or numpy.any(values >= coeff.shape[1])):
        raise ValueError("frozen must be an occupied count or distinct MO indices")


def build_domain_selections(
    mf, *, frozen, thresholds=None, lo_kwargs=None, target_coeff=None,
    frag_lolist=None, pair_energy_model="multipole", force_full_domains=False,
):
    """Select Boys domains, full occupied labels and retained virtual subspaces.

    Fragment order is independent of global Boys column order. Thresholds,
    ranks and graph are chosen only from concrete reference data.
    """
    from .domain import DomainData, TargetOccupiedData
    from ._pairs import screen_pairs

    thresholds = BoysDomainThresholds() if thresholds is None else thresholds
    if not isinstance(thresholds, BoysDomainThresholds):
        raise TypeError("thresholds must be BoysDomainThresholds")
    _validate_reference(mf, frozen)
    options = _resolve_boys_options(lo_kwargs)
    model = str(pair_energy_model).lower().replace("_", "-")
    if model not in {"all", "all-strong", "exact", "multipole", "multipole-os", "os-multipole"}:
        raise ValueError(f"unknown pair_energy_model: {pair_energy_model}")
    configure_compilation_cache()
    mol = mf.mol
    overlap, fock = map(_host_array, (mol.intor_symmetric("int1e_ovlp"), mf.get_fock()))
    if numpy.iscomplexobj(overlap) or numpy.iscomplexobj(fock):
        raise NotImplementedError("Boys STC requires real overlap and Fock matrices")
    if not numpy.isfinite(overlap).all() or not numpy.isfinite(fock).all():
        raise ValueError("Boys overlap and Fock must be finite")
    active_occ, active_vir, projected_out = _extract_active_indices(mf, frozen)
    occupied, virtual = (_host_array(mf.mo_coeff)[:, indices] for indices in (active_occ, active_vir))
    if not active_occ.size or not active_vir.size:
        raise ValueError("Boys domains require active occupied and virtual spaces")
    targets = _host_array(_build_boys_targets(mol, occupied, options) if target_coeff is None else target_coeff)
    if numpy.iscomplexobj(targets):
        raise NotImplementedError("Boys targets must be real")
    if (targets.shape != occupied.shape or not numpy.isfinite(targets).all()
            or not numpy.allclose(targets.T @ overlap @ targets, numpy.eye(active_occ.size), atol=1e-8, rtol=1e-8)):
        raise ValueError("Boys target columns must be an orthonormal active occupied frame")
    rotation = occupied.T @ overlap @ targets
    residual = targets - occupied @ rotation
    if (not numpy.allclose(rotation.T @ rotation, numpy.eye(active_occ.size), atol=1e-8, rtol=1e-8)
            or numpy.max(abs(residual)) > 1e-8):
        raise ValueError("Boys targets must span the active occupied space")
    groups = _validate_boys_target_groups(frag_lolist, active_occ.size)
    out = _host_array(mf.mo_coeff)[:, projected_out]
    raw_pao = numpy.eye(mol.nao) - out @ (out.T @ overlap)
    norms = numpy.sqrt(numpy.einsum("ui,uv,vi->i", raw_pao, overlap, raw_pao))
    pao_parent = numpy.where(norms > thresholds.pao_norm)[0].astype(numpy.int32)
    pao_coeff = raw_pao[:, pao_parent] / norms[pao_parent]
    ao2pao = numpy.full(mol.nao, -1, dtype=numpy.int32)
    ao2pao[pao_parent] = numpy.arange(pao_parent.size)
    ordered = targets[:, [int(group[0]) for group in groups]]
    compact, primary_bp, tight = (get_bp_domain(mol, ordered, overlap, bp_thr=cutoff)
                                  for cutoff in (thresholds.bp_occ, thresholds.bp_primary, thresholds.bp_ed))
    pao_bp = get_bp_domain(mol, pao_coeff, overlap, bp_thr=thresholds.bp_pao)
    primary = get_primary_domain(mol, primary_bp, pao_bp, ao2pao)
    nfrag = len(groups)
    if force_full_domains:
        compact, primary_bp, tight, primary = [tuple(numpy.arange(mol.natm, dtype=numpy.int32) for _ in groups) for _ in range(4)]
    empty = numpy.zeros(0, dtype=numpy.int32)
    fragments = []
    for i, group in enumerate(groups):
        ao = tools.ao_index_by_atom(mol, primary[i]).astype(numpy.int32)
        s22 = overlap[numpy.ix_(ao, ao)]
        raw_occ = tools.project_mo(targets[:, group], overlap[ao], s22)
        norm2 = numpy.einsum("ui,uv,vi->i", raw_occ, s22, raw_occ)
        norm_keep = numpy.where(norm2 > thresholds.metric_rank)[0].astype(numpy.int32)
        span_keep = empty
        weak_virtual = None
        if norm_keep.size:
            normalized = raw_occ[:, norm_keep] / numpy.sqrt(norm2[norm_keep])
            occ_span, span_keep = _reference_metric_space(normalized, s22, thresholds.metric_rank)
            weak_virtual, weak_coeff = _reference_pao_overlap_selection(
                mol, pao_coeff, ao2pao, overlap, parent_atoms=primary_bp[i], support_atoms=primary_bp[i],
                representation_atoms=primary[i], completeness_threshold=thresholds.bp_pao,
                overlap_threshold=thresholds.domain_pao, occupied_for_projection=occ_span, metric_threshold=thresholds.metric_rank,
            )
            if not weak_coeff.shape[1]:
                weak_virtual = None
        fragments.append(FragmentDomainSelection(i, group.copy(), None, empty, empty, empty, empty, empty,
                        None, primary[i], ao, primary_bp[i], norm_keep, span_keep, weak_virtual, -1))
    common = DomainData(overlap, fock, occupied, virtual,
                        _host_array(mf.mo_energy)[active_occ], _host_array(mf.mo_energy)[active_vir], targets,
                        pao_coeff, tuple(TargetOccupiedData(targets[:, group]) for group in groups))
    static = DomainSelections(frozen, thresholds, active_occ, active_vir, projected_out, pao_parent,
                              ao2pao, groups, (None,) * nfrag, numpy.zeros((nfrag, nfrag), dtype=bool), tuple(fragments),
                              lo_kwargs=options, target_reference_coeff=targets.copy(), target_reference_coords=_host_array(mol.atom_coords()).copy(),
                              compact_bp_domain=compact, primary_bp_domain=primary_bp, tight_bp_domain=tight,
                              pao_bp_domain=pao_bp, primary_domain=primary, pair_energy_model=model)
    pair, weak_pair, forced, strong = screen_pairs(mf, common, static, model=model, force_full_domains=force_full_domains)
    fragments = []
    for i, fragment in enumerate(static.fragments):
        partners = numpy.flatnonzero(strong[i]).astype(numpy.int32)
        if numpy.count_nonzero(partners == i) != 1:
            raise RuntimeError(f"Boys domain {i} must include its target exactly once")
        centers = _union_index_lists([compact[p] for p in partners])
        atoms = _union_index_lists([tight[p] for p in partners])
        ao = tools.ao_index_by_atom(mol, atoms).astype(numpy.int32)
        s22 = overlap[numpy.ix_(ao, ao)]
        candidates = numpy.concatenate([targets[:, groups[p]] for p in partners], axis=1)
        projected = tools.project_mo(candidates, overlap[ao], s22)
        occupied_local, keep = _reference_metric_space(projected, s22, thresholds.metric_rank)
        if keep.size != partners.size:
            raise RuntimeError(f"Boys domain {i} loses occupied target/partner labels; rebuild domains with tighter support")
        selection, coeff = _reference_pao_overlap_selection(
            mol, pao_coeff, ao2pao, overlap, parent_atoms=centers, support_atoms=atoms, representation_atoms=atoms,
            completeness_threshold=thresholds.ed_pao, overlap_threshold=thresholds.domain_pao,
            occupied_for_projection=occupied_local, metric_threshold=thresholds.metric_rank,
        )
        if not coeff.shape[1]:
            raise RuntimeError(f"Boys domain {i} has no retained virtual space; rebuild domains")
        fragments.append(replace(fragment, strong_fragments=partners, extended_atoms=atoms, extended_ao_indices=ao,
                                 pao_center_atoms=centers, strong_occ_metric_keep=keep, strong_virtual=selection,
                                 target_index=int(numpy.flatnonzero(partners == i)[0])))
    static = replace(static, fragments=tuple(fragments), strong_mask=strong, pair_energy=pair,
                     weak_pair_energy=weak_pair, forced_strong_mask=forced)
    return static
