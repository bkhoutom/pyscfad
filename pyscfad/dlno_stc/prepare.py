"""Project full Fock blocks and fitted OV factors into saved STC frames."""

import jax
import numpy
from functools import partial

from pyscfad import numpy as np
from pyscfad.lno import df as lno_df

from .domain import build_boys_ed_frame, build_system_local_frame
from ._profile import response
from ._df_basis import basis_kwargs, can_reuse_scf_df


def _symmetric_fock_block(fock, coeff):
    """Preserve the real symmetric Fock contract after projection roundoff."""
    block = coeff.T @ fock @ coeff
    # Keep this projection in the AD graph so both transpose paths respond.
    return 0.5 * (block + block.T)


def _scf_fock_block(coeff, energies, overlap, indices, local_coeff):
    """Project canonical energies while retaining the complete SCF chain rule."""
    canonical = coeff[:, indices]
    rotation = canonical.T @ overlap @ local_coeff
    block = rotation.T @ (energies[indices, None] * rotation)
    return 0.5 * (block + block.T)


def _project_inputs(mf, fock, occupied_coeff, virtual_coeff, atoms, *, aux_atoms, group=None,
                    correlation_auxbasis=None):
    """Use one occupied/virtual frame for both Fock blocks and DF factors."""
    co, cv = occupied_coeff, virtual_coeff
    nocc, nvir = co.shape[1], cv.shape[1]
    coeff = np.concatenate((co, cv), axis=1)
    B = lno_df.get_local_Lov(
        mf, coeff, nocc, atoms, integral_direct=True, aux_atmlst=aux_atoms,
        **basis_kwargs(correlation_auxbasis),
    )
    foo, fvv = response('stc.domain.input_pullback.local_fock_blocks',
                        lambda f, o, v: (_symmetric_fock_block(f, o),
                                         _symmetric_fock_block(f, v)),
                        fock, co, cv, group=group)
    return {
        "foo": foo,
        "fvv": fvv,
        "B": B.reshape((-1, nocc, nvir)),
    }


def prepare_inputs(mf, common, static, fragment_index, *, return_frame=False,
                   correlation_auxbasis=None):
    """Prepare exactly three tensors in one fixed, labelled finite Boys frame.

    ``common`` is rebuilt by the independent Boys ``rebuild_domain_data``.
    The saved selection owns the target's local slot, retained virtual ranks
    and concrete PAO anchors. That discrete metadata stays outside the
    differentiable tensor packet. Local AO and
    auxiliary supports remain finite through ``get_local_Lov``.

    With ``return_frame=True``, return ``((inputs, (co, cv)), frame)``.
    The coefficient pair is a differentiable output; the full frame is VJP
    auxiliary data carrying the discrete supports and labels. A caller can
    seed both outputs to join downstream frame response with DF/Fock response
    in one preparation pullback.
    """
    frame = build_boys_ed_frame(common, static, fragment_index)
    ao = numpy.asarray(frame.extended_ao_indices)
    inputs = _project_inputs(
        mf, common.fock[numpy.ix_(ao, ao)], frame.occupied_coeff,
        frame.virtual_coeff, frame.extended_atoms, group=fragment_index,
        aux_atoms=static.fragments[fragment_index].pao_center_atoms,
        **basis_kwargs(correlation_auxbasis),
    )
    if return_frame:
        return (inputs, (frame.occupied_coeff, frame.virtual_coeff)), frame
    return inputs


def prepare_boys_group_inputs(mf, common, static, group, *, return_frame=False,
                              correlation_auxbasis=None):
    """Prepare three tensors once in the group's saved representative frame.

    The full occupied partner space is retained. Group membership and target
    slots stay outside the differentiable tensor dictionary.
    """
    return prepare_inputs(mf, common, static, group.representative_index,
                          return_frame=return_frame, **basis_kwargs(correlation_auxbasis))


def prepare_system_inputs(
    mf, active_occ_indices, active_vir_indices, *, boys_reference=None,
    virtual_anchor_columns=None, fock_from_scf=False, correlation_auxbasis=None,
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
        fock_from_scf=fock_from_scf, **basis_kwargs(correlation_auxbasis),
    )


def _project_system_frame(mf, frame, active_occ_indices, active_vir_indices, *,
                          fock_from_scf=False, correlation_auxbasis=None):
    """Project either system frame without changing its DF storage strategy."""
    co, cv = frame.occupied_coeff, frame.virtual_coeff
    nocc, nvir = co.shape[1], cv.shape[1]
    coeff = np.concatenate((co, cv), axis=1)
    if not can_reuse_scf_df(mf, correlation_auxbasis):
        B = lno_df.get_local_Lov(
            mf, coeff, nocc, numpy.arange(mf.mol.natm, dtype=numpy.int32),
            integral_direct=True, **basis_kwargs(correlation_auxbasis),
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
        foo, fvv = response('stc.system.input_pullback.local_fock_blocks',
            lambda c, e, s, o, v: (_scf_fock_block(c, e, s, active_occ_indices, o),
                                   _scf_fock_block(c, e, s, active_vir_indices, v)),
            mf.mo_coeff, mf.mo_energy, overlap, co, cv, mode='scf_energies')
    else:
        fock = response('stc.system.input_pullback.ao_fock', lambda mf: mf.get_fock(), mf)
        foo, fvv = response('stc.system.input_pullback.local_fock_blocks',
            lambda f, o, v: (_symmetric_fock_block(f, o), _symmetric_fock_block(f, v)),
            fock, co, cv, mode='ao_fock')
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


def prepare_canonical_system_inputs(mf, active_occ_indices, active_vir_indices, *,
                                    correlation_auxbasis=None):
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
    if not can_reuse_scf_df(mf, correlation_auxbasis):
        B = lno_df.get_local_Lov(
            mf, coeff, nocc, numpy.arange(mf.mol.natm, dtype=numpy.int32),
            integral_direct=True, **basis_kwargs(correlation_auxbasis))
    else:
        B = lno_df.transform_df_to_mo(
            mf, coeff, (0, nocc, nocc, nocc + nvir), atmlst=None)
    return {"foo": foo, "fvv": fvv, "B": B.reshape((-1, nocc, nvir))}
