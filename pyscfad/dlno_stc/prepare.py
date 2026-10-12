"""Project full Fock blocks and fitted OV factors into saved STC frames."""

import jax
import numpy
from functools import partial

from pyscfad import numpy as np
from pyscfad.lno import df as lno_df

from .domain import build_boys_ed_frame, build_system_local_frame


def _symmetric_fock_block(fock, coeff):
    """Preserve the real symmetric Fock contract after projection roundoff."""
    block = coeff.T @ fock @ coeff
    # Keep this projection in the AD graph so both transpose paths respond.
    return 0.5 * (block + block.T)


def _scf_fock_block(mf, overlap, indices, local_coeff):
    """Project canonical energies while retaining the complete SCF chain rule."""
    canonical = mf.mo_coeff[:, indices]
    rotation = canonical.T @ overlap @ local_coeff
    block = rotation.T @ (mf.mo_energy[indices, None] * rotation)
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
    """Prepare exactly three tensors in one fixed, labelled finite Boys frame.

    ``common`` is rebuilt by the independent Boys ``rebuild_domain_data``.
    The saved selection owns the target's local slot and concrete virtual
    anchors; callers obtain that discrete metadata from ``build_boys_ed_frame``
    rather than placing it in this differentiable tensor packet. Local AO and
    auxiliary supports remain finite through ``get_local_Lov``.
    """
    frame = build_boys_ed_frame(common, static, fragment_index)
    ao = numpy.asarray(frame.extended_ao_indices)
    return _project_inputs(
        mf, common.fock[numpy.ix_(ao, ao)], frame.occupied_coeff,
        frame.virtual_coeff, frame.extended_atoms,
    )


def prepare_boys_group_inputs(mf, common, static, group):
    """Prepare three tensors once in the group's saved representative frame.

    The full occupied partner space is retained. Group membership and target
    slots stay outside the differentiable tensor dictionary.
    """
    return prepare_inputs(mf, common, static, group.representative_index)


def prepare_system_inputs(
    mf, active_occ_indices, active_vir_indices, *, boys_reference=None,
    virtual_anchor_columns=None, fock_from_scf=False,
):
    """Project full Fock blocks and fitted factors into one complete local frame.

    Active occupied Boys orbitals and complete PAO-derived active virtuals use
    fixed reference labels and AO anchors selected before tracing. Their
    current continuous coefficients remain in the preparation graph. Attached
    out-of-core factors and their response are streamed without reconstructing
    the AO-pair tensor; the integral-direct fallback does not build a DF leaf.

    ``fock_from_scf=True`` projects canonical SCF energies into the same local
    frame without rebuilding the AO Fock. It requires converged canonical
    ``mo_coeff`` and matching ``mo_energy``; its VJP must be composed with SCF
    response. Within each active occupied/virtual space, energy gaps must
    exceed the SCF eigensolver's 1e-9 degeneracy cutoff. The high-level driver
    selects the general path otherwise. Leave this false for independent
    Fock/coefficient variations.
    """
    frame = build_system_local_frame(
        mf, active_occ_indices, active_vir_indices, boys_reference=boys_reference,
        virtual_anchor_columns=virtual_anchor_columns,
    )
    return _project_system_frame(
        mf, frame, active_occ_indices, active_vir_indices,
        fock_from_scf=fock_from_scf,
    )


def _project_system_frame(mf, frame, active_occ_indices, active_vir_indices, *,
                          fock_from_scf=False):
    """Project either system frame without changing its DF storage strategy."""
    co, cv = frame.occupied_coeff, frame.virtual_coeff
    get_cderi = getattr(mf.with_df, "_get_cderi_source", None)
    cderi = get_cderi() if get_cderi is not None else mf.with_df._cderi
    nocc, nvir = co.shape[1], cv.shape[1]
    coeff = np.concatenate((co, cv), axis=1)
    if cderi is None:
        B = lno_df.get_local_Lov(
            mf, coeff, nocc, numpy.arange(mf.mol.natm, dtype=numpy.int32),
            integral_direct=True,
        )
    else:
        B = lno_df.transform_df_to_mo(
            mf, coeff, (0, nocc, nocc, nocc + nvir), atmlst=None,
        )
    if fock_from_scf:
        # FC = SC eps holds along the converged SCF solution. Keep eps, C,
        # S, and the Boys/PAO frame in the graph; independent partial VJPs
        # differ from the AO-Fock path until the SCF pullback is applied.
        overlap = mf.get_ovlp()
        foo = _scf_fock_block(mf, overlap, active_occ_indices, co)
        fvv = _scf_fock_block(mf, overlap, active_vir_indices, cv)
    else:
        fock = mf.get_fock()
        foo = _symmetric_fock_block(fock, co)
        fvv = _symmetric_fock_block(fock, cv)
    return {
        "foo": foo,
        "fvv": fvv,
        "B": B.reshape((-1, nocc, nvir)),
    }


@partial(jax.jit, static_argnums=(2, 3))
def _canonical_system_space(coeff, energies, occupied, virtual):
    """Fuse active-column selection and diagonal canonical Fock blocks."""
    active = numpy.asarray(occupied + virtual, dtype=numpy.int32)
    return (coeff[:, active], np.diag(energies[numpy.asarray(occupied)]),
            np.diag(energies[numpy.asarray(virtual)]))


def prepare_canonical_system_inputs(mf, active_occ_indices, active_vir_indices):
    """Prepare a validated canonical SCF frame using orbital-energy response.

    This is a reference path for deterministic frame-invariance checks, not
    production system preparation. Along the canonical SCF solution curve
    C.T F C = diag(eps); off-diagonal projected
    Fock derivatives vanish after the complete orbital/SCF chain rule. General
    orbital frames and independent Fock/coefficient variations must use
    prepare_system_inputs instead.
    """
    occupied = tuple(int(i) for i in active_occ_indices)
    virtual = tuple(int(i) for i in active_vir_indices)
    nocc, nvir = len(occupied), len(virtual)
    coeff, foo, fvv = _canonical_system_space(
        mf.mo_coeff, mf.mo_energy, occupied, virtual)
    get_cderi = getattr(mf.with_df, "_get_cderi_source", None)
    cderi = get_cderi() if get_cderi is not None else mf.with_df._cderi
    if cderi is None:
        B = lno_df.get_local_Lov(
            mf, coeff, nocc, numpy.arange(mf.mol.natm, dtype=numpy.int32),
            integral_direct=True)
    else:
        B = lno_df.transform_df_to_mo(
            mf, coeff, (0, nocc, nocc, nocc + nvir), atmlst=None)
    return {"foo": foo, "fvv": fvv, "B": B.reshape((-1, nocc, nvir))}
