"""Occupied correlation targets shared by DLNO domain and AD paths."""

from collections.abc import Mapping
from copy import deepcopy

import numpy

from pyscfad.lno.lno_base import get_iao
from pyscfad.lo.boys import boys


def resolve_target_options(lo_type="iao", lo_kwargs=None):
    """Validate and copy target-localization options."""
    mode = str(lo_type).lower()
    allowed = {
        "iao": {"minao", "orth"},
        "boys": {
            "init_guess",
            "conv_tol",
            "conv_tol_grad",
            "max_cycle",
            "symmetry",
            "gmres_options",
        },
    }
    if mode not in allowed:
        raise ValueError(f"unsupported lo_type {lo_type!r}; use 'iao' or 'boys'")
    if lo_kwargs is None:
        lo_kwargs = {}
    if not isinstance(lo_kwargs, Mapping):
        raise TypeError("lo_kwargs must be a mapping or None")
    unknown = set(lo_kwargs) - allowed[mode]
    if unknown:
        raise ValueError(
            f"unsupported {mode} localization options: {sorted(unknown)}"
        )
    return mode, deepcopy(dict(sorted(lo_kwargs.items())))


def build_targets(mol, occupied_coeff, *, lo_type="iao", lo_kwargs=None):
    """Build differentiable AO targets from the active occupied reference."""
    mode, options = resolve_target_options(lo_type, lo_kwargs)
    if occupied_coeff.ndim != 2 or occupied_coeff.shape[1] == 0:
        raise ValueError("targets require a nonempty rank-two occupied block")
    if mode == "boys":
        return boys(mol, occupied_coeff, **options)
    return get_iao(mol, occupied_coeff, **options)


def validate_boys_target_groups(frag_lolist, nocc):
    """Return a complete singleton partition in the requested order."""
    message = (
        "Boys targets require a disjoint complete singleton partition "
        "of occupied columns"
    )
    if nocc <= 0:
        raise ValueError(message)
    if frag_lolist is None:
        return tuple(
            numpy.array([index], dtype=numpy.int32)
            for index in range(nocc)
        )

    groups = []
    for indices in frag_lolist:
        values = numpy.asarray(indices)
        if (
            values.ndim != 1
            or values.size != 1
            or values.dtype.kind not in "iu"
        ):
            raise ValueError(message)
        value = int(values[0])
        if not 0 <= value < nocc:
            raise ValueError(message)
        groups.append(numpy.array([value], dtype=numpy.int32))

    if (
        len(groups) != nocc
        or len({int(group[0]) for group in groups}) != nocc
    ):
        raise ValueError(message)
    return tuple(groups)


def semantic_tuple(value):
    """Return a deterministic, hashable representation of nested metadata."""
    if isinstance(value, Mapping):
        return tuple(
            (key, semantic_tuple(item))
            for key, item in sorted(value.items(), key=lambda pair: pair[0])
        )
    if isinstance(value, (list, tuple)):
        return tuple(semantic_tuple(item) for item in value)
    if isinstance(value, numpy.ndarray) or (
        hasattr(value, "shape") and hasattr(value, "dtype")
    ):
        return semantic_tuple(numpy.asarray(value).tolist())
    if isinstance(value, numpy.generic):
        return value.item()
    return value


def canonical_frozen_selection(frozen):
    """Normalize supported PySCF frozen selectors to orbital indices."""
    if frozen is None:
        return ()

    values = numpy.asarray(frozen)
    if values.ndim == 0:
        count = int(values)
        if count < 0:
            raise ValueError("frozen orbital count must be non-negative")
        return tuple(range(count))
    return tuple(sorted(int(index) for index in values.reshape(-1)))


def validate_static_target_options(
    selections,
    *,
    lo_type="iao",
    lo_kwargs=None,
    frag_lolist=None,
    frag_atmlist=None,
    frozen=None,
    selection_name="static_selections",
    nested_attr="mp2_static",
    normalize=semantic_tuple,
):
    """Reject fixed selections that disagree with requested target settings."""
    requested_mode, requested_options = resolve_target_options(
        lo_type, lo_kwargs
    )
    stored = (
        getattr(selections, nested_attr, selections)
        if nested_attr is not None
        else selections
    )
    stored_mode, stored_options = resolve_target_options(
        getattr(stored, "lo_type", "iao"),
        getattr(stored, "lo_kwargs", None),
    )

    if stored_mode != requested_mode:
        raise ValueError(
            f"{selection_name} lo_type mismatch: "
            f"stored {stored_mode!r}, requested {requested_mode!r}"
        )

    stored_semantic = normalize(stored_options)
    requested_semantic = normalize(requested_options)
    if stored_semantic != requested_semantic:
        raise ValueError(
            f"{selection_name} lo_kwargs mismatch: "
            f"stored {stored_semantic!r}, requested {requested_semantic!r}"
        )

    stored_groups = getattr(stored, "frag_lolist", None)
    if requested_mode == "boys" and frag_atmlist is not None:
        raise ValueError(
            "Boys targets do not accept molecular frag_atmlist overrides"
        )

    if frag_lolist is not None:
        groups = (
            validate_boys_target_groups(frag_lolist, len(stored_groups))
            if requested_mode == "boys"
            else frag_lolist
        )
        if normalize(groups) != normalize(stored_groups):
            raise ValueError(f"{selection_name} has incompatible target map")

    if requested_mode != "boys" and frag_atmlist is not None:
        stored_atoms = getattr(stored, "frag_atmlist", None)
        if normalize(frag_atmlist) != normalize(stored_atoms):
            raise ValueError(
                f"{selection_name} has incompatible fragment atom map"
            )

    if (
        frozen is not None
        and canonical_frozen_selection(frozen)
        != canonical_frozen_selection(getattr(stored, "frozen", None))
    ):
        raise ValueError(
            f"{selection_name} frozen selection does not match the request"
        )
    return requested_mode, requested_options
