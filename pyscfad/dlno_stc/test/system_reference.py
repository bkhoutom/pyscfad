"""Independent full-space preparation oracles, excluded from the library API."""

from functools import partial
from typing import NamedTuple
import jax
import numpy
import scipy.linalg as scipy_linalg
from pyscfad import numpy as np
from pyscfad.dlno_stc._profile import response
from pyscfad.dlno_stc._selection import _host_array, _build_boys_targets, _resolve_boys_options, _validate_virtual_anchor_columns
from pyscfad.dlno_stc.domain import _rebuild_boys_targets, _cholesky_metric_orthonormalize, build_scf_data
from pyscfad.dlno_stc.prepare import _symmetric_fock_block
from pyscfad.dlno_stc._df_basis import basis_kwargs, can_reuse_scf_df
from pyscfad.lno import df as lno_df





class SystemBoysReference(NamedTuple):
    """Fixed initializer and orbital labels for active-occupied Boys replay."""

    target_reference_coeff: object
    target_reference_coords: object
    lo_kwargs: dict | None = None


class SystemLocalFrame(NamedTuple):
    """Complete active spaces, with discrete Boys and AO-anchor metadata."""

    occupied_coeff: object
    virtual_coeff: object
    boys_reference: SystemBoysReference | None
    virtual_anchor_columns: tuple[int, ...]


def _require_concrete_system_selection(mf):
    if any(isinstance(leaf, jax.core.Tracer)
           for leaf in jax.tree_util.tree_leaves((mf.mol, mf.mo_coeff))):
        raise ValueError(
            "select the system Boys reference and virtual anchor columns "
            "from concrete data before tracing preparation"
        )


def select_system_boys_reference(mf, active_occ_indices, *, lo_kwargs=None):
    """Select active-occupied Boys labels using the shared domain setup."""

    _require_concrete_system_selection(mf)
    occupied = mf.mo_coeff[:, active_occ_indices]
    if not occupied.shape[1]:
        return None
    options = _resolve_boys_options(lo_kwargs)
    # Match the Boys-domain benchmark: explicit atomic initialization, the
    # standard derived gradient tolerance, and its localization response solve.
    options = {"init_guess": "atomic", "conv_tol": 1e-10,
               "gmres_options": {"restart": 120, "maxiter": 100}, **options}
    localized = (_build_boys_targets(mf.mol, occupied, options, fock=build_scf_data(mf)[1])
                 if options.get('method') == 'lambda_boys' else
                 _build_boys_targets(mf.mol, occupied, options))
    return SystemBoysReference(
        numpy.array(_host_array(localized), copy=True),
        numpy.array(_host_array(mf.mol.atom_coords()), copy=True),
        options,
    )


def select_system_virtual_anchor_columns(mf, active_vir_indices):
    """Select exactly nvir independent projected AO columns, without domains."""
    _require_concrete_system_selection(mf)
    virtual = mf.mo_coeff[:, active_vir_indices]
    nvir = virtual.shape[1]
    if not nvir:
        return ()
    overlap = mf.mol.intor_symmetric("int1e_ovlp")
    cross = _host_array(virtual.T.conj() @ overlap)
    _, triangular, pivots = scipy_linalg.qr(
        cross, pivoting=True, mode="economic", check_finite=True,
    )
    diagonal = numpy.abs(numpy.diag(triangular))
    if diagonal.size < nvir or numpy.min(diagonal[:nvir]) < 1e-10:
        raise ValueError("system projected AO anchors do not span the active virtual space")
    return tuple(int(column) for column in pivots[:nvir])


def build_system_local_frame(
    mf, active_occ_indices, active_vir_indices, *, boys_reference=None,
    virtual_anchor_columns=None,
):
    """Rebuild one complete Boys/PAO frame with only discrete choices fixed.

    Frozen occupied and virtual columns must already be excluded by the
    supplied active indices. Concrete calls may select their reference and
    anchors; a preparation VJP must receive both selections from its caller.
    Current Boys coefficients, AO projections, overlap and Cholesky remain
    differentiable. No ED support selection or virtual rank truncation occurs.
    """

    occupied = mf.mo_coeff[:, active_occ_indices]
    virtual = mf.mo_coeff[:, active_vir_indices]
    if boys_reference is None and occupied.shape[1]:
        boys_reference = select_system_boys_reference(mf, active_occ_indices)
    if virtual_anchor_columns is None:
        virtual_anchor_columns = select_system_virtual_anchor_columns(mf, active_vir_indices)
    columns = _validate_virtual_anchor_columns(
        virtual_anchor_columns, numpy.arange(mf.mol.nao), virtual.shape[1],
    )
    co = (_rebuild_boys_targets(mf, occupied, boys_reference)
          if occupied.shape[1] else occupied)
    if virtual.shape[1]:
        overlap = response('stc.system.input_pullback.overlap',
                           lambda mol: mol.intor_symmetric("int1e_ovlp"), mf.mol)
        cv = response('stc.system.input_pullback.virtual_frame',
                      lambda v, s: _cholesky_metric_orthonormalize(
                          v @ (v.T.conj() @ s[:, columns]), s), virtual, overlap)
    else:
        cv = virtual
    return SystemLocalFrame(co, cv, boys_reference, columns)




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


def _scf_fock_block(coeff, energies, overlap, indices, local_coeff):
    """Project canonical energies while retaining the complete SCF chain rule."""
    canonical = coeff[:, indices]
    rotation = canonical.T @ overlap @ local_coeff
    block = rotation.T @ (energies[indices, None] * rotation)
    return 0.5 * (block + block.T)




def solve(inputs, metadata, controls, *, scope='system', **options):
    from pyscfad.dlno_stc import backend
    return backend.solve(inputs, dict(metadata, target_indices=tuple(range(inputs['foo'].shape[0]))), controls, **options)
