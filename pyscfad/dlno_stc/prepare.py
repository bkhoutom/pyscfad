"""Project full Fock blocks and fitted OV factors into saved STC frames."""

import jax
import numpy

from pyscfad import numpy as np
from pyscfad.lno import df as lno_df

from .domain import (
    _build_local_strong_ed_domain_and_anchors,
    _validate_virtual_anchor_columns,
    build_local_strong_ed_domain,
    build_stc_domain,
    select_virtual_anchor_columns,
)


def _symmetric_fock_block(fock, coeff):
    """Preserve the real symmetric Fock contract after projection roundoff."""
    block = coeff.T @ fock @ coeff
    # Keep this projection in the AD graph so both transpose paths respond.
    return 0.5 * (block + block.T)


def _project_inputs(mf, fock, occupied_coeff, virtual_coeff, atoms):
    """Use one occupied/virtual frame for both Fock blocks and DF factors."""
    co, cv = occupied_coeff, virtual_coeff
    nocc, nvir = co.shape[1], cv.shape[1]
    coeff = np.concatenate((co, cv), axis=1)
    B = lno_df.get_local_Lov(
        mf, coeff, nocc, atoms, integral_direct=True,
    )
    return {
        "foo": _symmetric_fock_block(fock, co),
        "fvv": _symmetric_fock_block(fock, cv),
        "B": B.reshape((-1, nocc, nvir)),
    }


def prepare_inputs(mf, common, static, fragment_index):
    """Project Fock and fitted OV factors into one full-support Boys frame.

    ``common`` must be rebuilt from the current differentiable ``mf`` using
    ``dlno_base.rebuild_domain_data``. Fixed discrete selections live in
    ``static`` and never replace the continuous orbitals.
    """
    domain = build_stc_domain(mf, common, static, fragment_index)
    return _project_inputs(
        mf, common.fock, domain.occupied_coeff, domain.virtual_coeff,
        domain.extended_atoms,
    )


def _project_finite_domain(mf, common, static, fragment_index, domain):
    fragment = static.fragments[int(fragment_index)]
    ao = numpy.asarray(fragment.extended_ao_indices)
    fock22 = common.fock[numpy.ix_(ao, ao)]
    return _project_inputs(
        mf, fock22, domain.occupied_coeff, domain.virtual_coeff,
        fragment.extended_atoms,
    )


def prepare_finite_export_inputs(mf, common, static, fragment_index):
    """Prepare a finite packet and its PAO anchors with one domain build."""
    domain, anchors = _build_local_strong_ed_domain_and_anchors(
        common, static, fragment_index
    )
    return _project_finite_domain(
        mf, common, static, fragment_index, domain,
    ), anchors


def prepare_finite_inputs(
    mf, common, static, fragment_index, *, virtual_anchor_columns=None,
):
    """Return full local ``foo``, ``fvv``, and ``B`` for one finite ED.

    All three arrays use the same pre-semicanonical, orthonormal occupied and
    virtual frame. The fixed AO/atom/rank selections come from ``static``;
    their continuous coefficients are rebuilt from differentiable ``common``.
    The external solver must use the full Fock matrices when forming its
    amplitudes.
    """
    domain = build_local_strong_ed_domain(
        common, static, fragment_index,
        virtual_anchor_columns=virtual_anchor_columns,
    )
    return _project_finite_domain(
        mf, common, static, fragment_index, domain,
    )


def prepare_weighted_inputs(
    mf, common, static, fragment_index, *, virtual_anchor_columns=None,
):
    """Return five differentiable arrays in one finite local frame.

    The target row and partner weight are the raw projections of the original
    localized orbitals. Select PAO anchors from concrete ``common`` with
    ``select_virtual_anchor_columns`` before constructing a preparation VJP,
    and pass those columns on every traced call. Orbital coefficients and
    projection weights remain differentiable with these selections fixed.
    """
    if virtual_anchor_columns is None and any(
        isinstance(leaf, jax.core.Tracer)
        for leaf in jax.tree_util.tree_leaves(common)
    ):
        raise ValueError(
            "select virtual anchor columns from concrete common data before "
            "tracing weighted preparation"
        )
    domain = build_local_strong_ed_domain(
        common, static, fragment_index,
        virtual_anchor_columns=virtual_anchor_columns,
    )
    inputs = _project_finite_domain(
        mf, common, static, fragment_index, domain,
    )
    inputs["target_projection"] = domain.target_projection
    inputs["partner_weight"] = domain.partner_weight
    return inputs


def prepare_system_inputs(mf, active_occ_indices, active_vir_indices):
    """Project full Fock blocks and reuse global fitted factors in active MOs.

    Concrete active indices are selected before tracing. Attached out-of-core
    factors and their orbital-coefficient response are streamed without
    reconstructing the AO-pair tensor. References without fitted factors
    retain the integral-direct path, which does not build a new DF leaf.
    """
    co = mf.mo_coeff[:, active_occ_indices]
    cv = mf.mo_coeff[:, active_vir_indices]
    get_cderi = getattr(mf.with_df, "_get_cderi_source", None)
    cderi = get_cderi() if get_cderi is not None else mf.with_df._cderi
    if cderi is None:
        return _project_inputs(mf, mf.get_fock(), co, cv,
                               numpy.arange(mf.mol.natm, dtype=numpy.int32))

    nocc, nvir = co.shape[1], cv.shape[1]
    coeff = np.concatenate((co, cv), axis=1)
    B = lno_df.transform_df_to_mo(
        mf, coeff, (0, nocc, nocc, nocc + nvir), atmlst=None,
    )
    fock = mf.get_fock()
    return {
        "foo": _symmetric_fock_block(fock, co),
        "fvv": _symmetric_fock_block(fock, cv),
        "B": B.reshape((-1, nocc, nvir)),
    }
