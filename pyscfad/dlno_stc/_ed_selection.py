"""Finalize one planned ED while retaining its live numerical construction."""
from dataclasses import replace

import jax
import numpy as np

from . import domain
from ._primary_response import _metric_space
from ._selection import (EDLocalPAOSelection,
                         _select_virtual_anchor_columns,
                         build_domain_groups)


def _host(value):
    return np.asarray(jax.device_get(jax.lax.stop_gradient(value)))


def select_boys_ed_frame(common, static, index):
    """Select ranks/anchors and construct a frame in one differentiable pass."""
    fragment = static.fragments[index]
    ao, atoms = fragment.extended_ao_indices, fragment.extended_atoms
    partners = fragment.strong_fragments
    overlap = common.s1e[np.ix_(ao, ao)]
    targets = domain.np.concatenate([
        common.fragment_occupied_data[int(p)].iao_coeff for p in partners], axis=1)
    if static.ao_atom_slices is None:
        raise ValueError('pending ED selection requires atom AO slices; rebuild domain plans')
    parent_ao = np.concatenate([np.arange(*static.ao_atom_slices[int(atom)])
                                for atom in fragment.pao_center_atoms])
    parent = static.ao2pao_map[parent_ao]
    parent = parent[parent >= 0].astype(np.int32)
    occupied, projected_pao = domain._project_ed_columns(common, targets, parent, ao)
    occupied, occupied_keep = _metric_space(occupied, overlap, static.thresholds.metric_rank)
    if len(occupied_keep) != len(partners):
        raise RuntimeError(f'Boys domain {index} loses occupied target/partner labels; rebuild domains')
    occupied = occupied * domain._positive_pivot_phases(occupied)
    virtual, (complete, metric) = domain._local_ed_pao_space(
        common, parent, ao, occupied, static.thresholds, projected=projected_pao)
    selection = EDLocalPAOSelection(parent, ao, complete, metric)
    anchors = _select_virtual_anchor_columns(
        _host(virtual), _host(common.s1e[ao]), _host(common.pao_coeff), parent)
    selected = replace(fragment, strong_occ_metric_keep=occupied_keep,
                       strong_virtual=selection, virtual_anchor_columns=anchors)
    virtual, _ = domain._stable_virtual_frame(common, selected, ao, overlap, virtual, anchors)
    virtual = virtual * domain._positive_pivot_phases(virtual)
    return domain.BoysEDFrame(occupied, virtual, ao, atoms, fragment.target_index, anchors), selected


def resolve_group(static, group, selected):
    """Return new immutable labels for this group; keep other plans untouched."""
    fragments = list(static.fragments)
    for index, original in enumerate(fragments):
        same_plan = (original.strong_virtual is None and all(
            np.array_equal(getattr(original, name), getattr(selected, name))
            for name in ('strong_fragments', 'extended_ao_indices', 'pao_center_atoms')))
        if index not in group.fragment_indices and not same_plan:
            continue
        fragments[index] = replace(original, strong_occ_metric_keep=selected.strong_occ_metric_keep,
                                   strong_virtual=selected.strong_virtual,
                                   virtual_anchor_columns=selected.virtual_anchor_columns)
    return replace(static, fragments=tuple(fragments),
                   ed_selection_pending=any(f.strong_virtual is None for f in fragments))


def materialize_domain_selections(mf, static, *, common=None):
    """Explicitly resolve labels for inspection/export, retaining no frames.

    Calculation drivers instead resolve only their current fragment and use
    its frame immediately. This helper is for consumers requesting all ranks.
    """
    common = domain.rebuild_domain_data(mf, static) if common is None else common
    for group in build_domain_groups(static):
        index = group.representative_index
        if static.fragments[index].strong_virtual is None:
            _, selected = select_boys_ed_frame(common, static, index)
            static = resolve_group(static, group, selected)
    return static
