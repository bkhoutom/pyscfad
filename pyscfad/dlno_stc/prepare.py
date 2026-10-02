"""Project full Fock blocks and fitted OV factors into saved STC frames."""

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


def _project_inputs(mf, fock, occupied_coeff, virtual_coeff, atoms):
    """Use one occupied/virtual frame for both Fock blocks and DF factors."""
    co, cv = occupied_coeff, virtual_coeff
    nocc, nvir = co.shape[1], cv.shape[1]
    coeff = np.concatenate((co, cv), axis=1)
    B = lno_df.get_local_Lov(
        mf, coeff, nocc, atoms, integral_direct=True,
    )
    return {
        "foo": co.T @ fock @ co,
        "fvv": cv.T @ fock @ cv,
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
