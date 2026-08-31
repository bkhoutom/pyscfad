r"""IAO-fragment MP2 construction of local interacting spaces.

This module connects the fixed-topology IAO--DLNO--MP2 construction in
``iao_mp2_grad`` to the orbital layout expected by the LNO CCSD(T) impurity
solver.  It deliberately keeps the two uses of MP2 separate:

* the additive MP2 energy uses both the target fragment weight and the
  strong-partner weight; and
* the LNO selection density is conditioned on the target IAO block while the
  second occupied line spans the already selected strong extended domain.

The latter is the direct IAO generalization of the usual ``ie`` LNO density.
If ``p,r`` are semicanonical occupied orbitals of the strong ED, ``a,b`` are
its virtual orbitals, and ``X[I,p]`` is the projection of target-fragment IAO
``I`` into that occupied space, the amplitudes used for LIS construction are

.. math::

   U_{Ir}^{ab} = \sum_p X_{Ip}^*\,
       \frac{(pa|rb)}{\epsilon_p+\epsilon_r-\epsilon_a-\epsilon_b}.

The complete spin-adapted unrelaxed ``oo`` and ``vv`` density blocks are
formed from ``U``.  The strong-partner weight is *not* applied a second time:
its retained range has already selected the ED occupied environment.  This
choice reduces exactly to the conventional one-internal/one-environment LNO
density for orthogonal localized occupied orbitals and remains invariant to
unitary rotations inside an IAO fragment.

The ED orbitals are projected back into the global active HF occupied and
virtual spaces before diagonalizing the density.  Consequently the final LIS
does not mix the HF occupied and virtual manifolds even though the numerical
ED orbitals live in a truncated AO domain.

All discrete ranks and retained natural-orbital labels are selected once by
:func:`build_iao_lis_static_selections`.  The overlap, Fock matrix, IAOs, ED
orbitals, MP2 amplitudes, densities, projectors, and retained LIS orbitals are
rebuilt on the differentiable path by :func:`build_fragment_lis`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import numpy as onp
import scipy.linalg as onp_scipy_linalg

from pyscfad import numpy as np
from pyscfad import scipy
from pyscfad.lno import lno_base
from pyscfad.tools import resource_profile

from .iao_mp2_grad import (
    IAOFragmentMP2ContinuousData,
    IAOFragmentMP2StaticSelections,
    IAOMP2StrongDomain,
    build_strong_ed_domain,
    rebuild_iao_mp2_common,
)


__all__ = [
    "IAO_LIS_INTERNAL_RANK_THRESHOLD",
    "IAOLISFragmentStaticSelection",
    "IAOFragmentLISStaticSelections",
    "IAOMP2Density",
    "IAOFragmentLIS",
    "target_conditioned_mp2_density_from_amplitudes",
    "strong_domain_mp2_density_from_lov",
    "strong_domain_mp2_density",
    "strong_domain_prescreen",
    "build_iao_lis_fragment_static_selection",
    "build_iao_lis_static_selections",
    "build_fragment_lis",
]


# Raw IAOs have algebraically tiny projections onto distant occupied and
# virtual manifolds.  Normalizing those tails as internal LIS orbitals defeats
# locality, even though they are harmless in projector-based ED construction.
# This is a singular-value threshold (not a Gram-eigenvalue threshold).
IAO_LIS_INTERNAL_RANK_THRESHOLD = 1e-6


@dataclass(frozen=True)
class IAOLISFragmentStaticSelection:
    """Fixed rank/label choices for one fragment LIS.

    All index arrays refer to ascending Hermitian-eigenvalue order at the
    corresponding reference eigenproblem.  ``internal_*_keep`` index the
    small IAO-row Gram matrix.  ``*_lno_keep`` index the density projected
    into the complete global active occupied or virtual manifold.
    """

    fragment_index: int
    internal_occ_keep: onp.ndarray
    internal_vir_keep: onp.ndarray
    occupied_lno_keep: onp.ndarray
    virtual_lno_keep: onp.ndarray
    full_occupied_space: bool
    full_virtual_space: bool


@dataclass(frozen=True)
class IAOFragmentLISStaticSelections:
    """Static IAO-MP2 ED topology plus fixed LIS rank selections."""

    mp2_static: IAOFragmentMP2StaticSelections
    thresh_occ: float
    thresh_vir: float
    internal_rank_threshold: float
    fragments: tuple[IAOLISFragmentStaticSelection, ...]


class IAOMP2Density(NamedTuple):
    """Target-conditioned strong-ED MP2 density blocks."""

    occupied: object
    virtual: object


@dataclass(frozen=True)
class IAOFragmentLIS:
    """One rebuilt fragment LIS and the data needed by an impurity solver.

    ``mo_coeff`` contains a complete MO layout ordered as frozen occupied,
    active LIS occupied, active LIS virtual, and frozen virtual blocks in the
    convention expected by :mod:`pyscfad.lno.ccsd`.  ``frozen`` contains the
    corresponding frozen column indices.

    ``fragment_occupied_anchor`` is the occupied projection of the target IAO
    block.  Its overlap with any HF-occupied LIS is identical to that of the
    raw fragment IAOs, while making explicit that the CC energy partition is
    an occupied-space weight.  ``fragment_iao_coeff`` is retained for callers
    that also need the raw IAO block.
    """

    mo_coeff: object
    frozen: onp.ndarray
    fragment_occupied_anchor: object
    fragment_iao_coeff: object
    active_occupied_coeff: object
    active_virtual_coeff: object
    occupied_projector: object
    virtual_projector: object
    density_occupied_ed: object
    density_virtual_ed: object
    density_occupied_active: object
    density_virtual_active: object
    domain: IAOMP2StrongDomain
    n_internal_occ: int
    n_internal_vir: int
    n_lno_occ: int
    n_lno_vir: int

    @property
    def orbfrag(self):
        """Compatibility alias for the complete impurity MO layout."""

        return self.mo_coeff

    @property
    def frzfrag(self):
        """Compatibility alias for the impurity frozen indices."""

        return self.frozen

    @property
    def orbfragloc(self):
        """Raw IAO block used by the historical fragment interface."""

        return self.fragment_iao_coeff


def _hermitize(array):
    return 0.5 * (array + array.T.conj())


def _validate_amplitudes(amplitudes, target_projection):
    amplitudes = np.asarray(amplitudes)
    target_projection = np.asarray(target_projection)
    if amplitudes.ndim != 4:
        raise ValueError("amplitudes must have shape (nocc,nocc,nvir,nvir)")
    nocc, nocc1, nvir, nvir1 = amplitudes.shape
    if nocc1 != nocc or nvir1 != nvir:
        raise ValueError("amplitudes must have shape (nocc,nocc,nvir,nvir)")
    if target_projection.ndim != 2 or target_projection.shape[1] != nocc:
        raise ValueError("target_projection must have shape (ntarget,nocc)")
    return amplitudes, target_projection


def _density_from_target_amplitudes(target_amplitudes):
    """Spin-adapted IE density from ``U[I,r,a,b]``.

    The four contractions are kept in the same form as
    :func:`pyscfad.lno._checkpointed.make_mp2_rdm1_ie`.  Wrapping each target
    slice in ``jax.checkpoint`` makes reverse mode recompute its rank-three
    contractions instead of retaining one set per target IAO.
    """

    target_amplitudes = np.asarray(target_amplitudes)
    if target_amplitudes.ndim != 4:
        raise ValueError(
            "target_amplitudes must have shape (ntarget,nocc,nvir,nvir)"
        )
    ntarget, nocc, nvir, nvir1 = target_amplitudes.shape
    if nvir1 != nvir:
        raise ValueError("the two virtual dimensions must be equal")

    dtype = target_amplitudes.dtype
    dmoo0 = np.zeros((nocc, nocc), dtype=dtype)
    dmvv0 = np.zeros((nvir, nvir), dtype=dtype)

    @jax.checkpoint
    def target_density(amplitude):
        # The established helper uses axes (a,j,b) for one internal target.
        t = amplitude.transpose(1, 0, 2)
        tc = t.conj()

        dmvv = np.dot(t.reshape(nvir, -1), tc.reshape(nvir, -1).T)
        dmvv = dmvv - 0.5 * np.einsum("ajc,cjb->ab", t, tc)
        dmvv = dmvv + np.dot(
            t.reshape(-1, nvir).T,
            tc.reshape(-1, nvir),
        )
        dmvv = dmvv - 0.5 * np.einsum("cja,bjc->ab", t, tc)

        dmoo = np.einsum("aib,ajb->ij", t, tc)
        dmoo = dmoo - 0.5 * np.einsum("aib,bja->ij", t, tc)
        dmoo = dmoo + np.einsum("bia,bja->ij", t, tc)
        dmoo = dmoo - 0.5 * np.einsum("bia,ajb->ij", t, tc)
        return dmoo, dmvv

    def scan_body(carry, amplitude):
        dmoo, dmvv = carry
        term_oo, term_vv = target_density(amplitude)
        return (dmoo + term_oo, dmvv + term_vv), None

    if ntarget == 0:
        return IAOMP2Density(dmoo0, dmvv0)
    (dmoo, dmvv), _ = jax.lax.scan(
        scan_body, (dmoo0, dmvv0), target_amplitudes
    )
    return IAOMP2Density(_hermitize(dmoo), _hermitize(dmvv))


def _density_from_target_amplitude_block(a_block, b_block):
    """Return one contracted-virtual-block contribution to Doo and Dvv."""

    a_block = np.asarray(a_block)
    b_block = np.asarray(b_block)
    if a_block.ndim != 4 or b_block.ndim != 4:
        raise ValueError("amplitude blocks must have rank-4 shapes")
    if a_block.shape != b_block.shape:
        raise ValueError("amplitude blocks must have identical shapes")

    # Put the two full virtual indices first and last, respectively.  The
    # supplied B block has already exchanged those virtual indices.
    a = a_block.transpose(0, 2, 1, 3)
    b = b_block.transpose(0, 2, 1, 3)
    b_first = b.swapaxes(1, 3)

    dmvv = np.dot(
        a.transpose(1, 0, 2, 3).reshape(a.shape[1], -1),
        a.conj().transpose(1, 0, 2, 3).reshape(a.shape[1], -1).T,
    )
    dmvv = dmvv - 0.5 * np.einsum(
        "iajc,icjb->ab", a, b_first.conj()
    )
    dmvv = dmvv + np.dot(
        b_first.reshape(-1, b.shape[1]).T,
        b_first.conj().reshape(-1, b.shape[1]),
    )
    dmvv = dmvv - 0.5 * np.einsum(
        "icja,ibjc->ab", b_first, a.conj()
    )

    dmoo = np.einsum("ixpc,ixqc->pq", a, a.conj())
    dmoo = dmoo - 0.5 * np.einsum(
        "ixpc,icqx->pq", a, b_first.conj()
    )
    dmoo = dmoo + np.einsum(
        "icpx,icqx->pq", b_first, b_first.conj()
    )
    dmoo = dmoo - 0.5 * np.einsum(
        "icpx,ixqc->pq", b_first, a.conj()
    )
    return IAOMP2Density(dmoo, dmvv)


def target_conditioned_mp2_density_from_amplitudes(
    amplitudes,
    target_projection,
):
    """Return the target-conditioned spin-adapted MP2 selection density.

    Parameters
    ----------
    amplitudes
        Semicanonical strong-ED MP2 amplitudes ``T[p,r,a,b]``.  Denominators
        must be applied before the target factor because a general IAO factor
        mixes nondegenerate occupied ED orbitals.
    target_projection
        ``X[I,p]``, the target IAO block projected into the ED occupied frame.

    Returns
    -------
    :class:`IAOMP2Density`
        Occupied and virtual unrelaxed selection-density blocks in the ED
        semicanonical bases.
    """

    amplitudes, target_projection = _validate_amplitudes(
        amplitudes, target_projection
    )
    target_amplitudes = np.einsum(
        "Ip,prab->Irab",
        target_projection.conj(),
        amplitudes,
        optimize=True,
    )
    return _density_from_target_amplitudes(target_amplitudes)


def _mp2_density_virtual_block_size(lov, ntarget, max_memory_mb):
    itemsize = onp.dtype(lov.dtype).itemsize
    nocc = lov.shape[1]
    nvir = lov.shape[2]
    bytes_per_c = itemsize * nocc * nvir * max(2 * ntarget + 6, 1)
    budget = float(max_memory_mb) * 1024.0**2
    block_nvir = int(budget // max(bytes_per_c, 1))
    return max(1, min(nvir, block_nvir))


def strong_domain_mp2_density_from_lov(
    lov,
    occupied_energy,
    virtual_energy,
    target_projection,
    *,
    block_nvir=None,
    max_memory_mb=256.0,
):
    """Build the target-conditioned ED density directly from DF factors.

    The contracted virtual index is processed in fixed-size blocks.  Peak
    target-amplitude storage is ``O(ntarget*nocc*nvir*block_nvir)``.  Nested
    checkpointed scans reconstruct each temporary block during reverse mode
    instead of retaining the complete MP2 or target-amplitude tensor.
    """

    lov = np.asarray(lov)
    occupied_energy = np.asarray(occupied_energy)
    virtual_energy = np.asarray(virtual_energy)
    target_projection = np.asarray(target_projection)
    if lov.ndim != 3:
        raise ValueError("lov must have shape (naux,nocc,nvir)")
    naux, nocc, nvir = lov.shape
    if occupied_energy.shape != (nocc,):
        raise ValueError("occupied_energy must have shape (nocc,)")
    if virtual_energy.shape != (nvir,):
        raise ValueError("virtual_energy must have shape (nvir,)")
    if target_projection.ndim != 2 or target_projection.shape[1] != nocc:
        raise ValueError("target_projection must have shape (ntarget,nocc)")

    if max_memory_mb <= 0:
        raise ValueError("max_memory_mb must be positive")
    ntarget = target_projection.shape[0]
    if block_nvir is None:
        block_nvir = _mp2_density_virtual_block_size(
            lov, ntarget, max_memory_mb
        )
    elif (
        not isinstance(block_nvir, (int, onp.integer))
        or isinstance(block_nvir, bool)
        or block_nvir <= 0
    ):
        raise ValueError("block_nvir must be a positive integer")
    block_nvir = min(int(block_nvir), nvir) if nvir else 1

    dmoo0 = np.zeros((nocc, nocc), dtype=lov.dtype)
    dmvv0 = np.zeros((nvir, nvir), dtype=lov.dtype)
    if ntarget == 0 or nocc == 0 or nvir == 0:
        return IAOMP2Density(dmoo0, dmvv0)

    eia = occupied_energy[:, None] - virtual_energy[None, :]
    nblock = (nvir + block_nvir - 1) // block_nvir
    block_ids = np.arange(nblock, dtype=onp.int32)
    block_offsets = np.arange(block_nvir, dtype=onp.int32)
    occupied_ids = np.arange(nocc, dtype=onp.int32)

    @jax.checkpoint
    def virtual_block_body(carry, block_id):
        dmoo, dmvv = carry
        indices = block_id * block_nvir + block_offsets
        valid = indices < nvir
        safe_indices = np.minimum(indices, nvir - 1)
        lov_c = np.take(lov, safe_indices, axis=2)
        eia_c = np.take(eia, safe_indices, axis=1)
        lov_c = np.where(valid[None, None, :], lov_c, 0)
        a0 = np.zeros((ntarget, nocc, nvir, block_nvir), dtype=lov.dtype)

        @jax.checkpoint
        def occupied_slice(p):
            la = lov[:, p, :]
            eia_p = eia[p]
            target_column = target_projection[:, p]
            integrals = np.einsum(
                "La,Lrc->rac", la, lov_c, optimize=True
            )
            denominator = eia_p[None, :, None] + eia_c[:, None, :]
            denominator = np.where(
                valid[None, None, :],
                denominator,
                np.ones((), dtype=denominator.dtype),
            )
            amplitudes = integrals / denominator
            amplitudes = np.where(
                valid[None, None, :], amplitudes, 0
            )
            a_increment = np.einsum(
                "I,rac->Irac", target_column.conj(), amplitudes,
                optimize=True,
            )
            b_row = np.einsum(
                "Ir,rac->Iac", target_projection.conj(), amplitudes,
                optimize=True,
            )
            return a_increment, b_row

        def occupied_scan_body(a_block, p):
            a_increment, b_row = occupied_slice(p)
            return a_block + a_increment, b_row

        a_block, b_rows = jax.lax.scan(
            jax.checkpoint(occupied_scan_body), a0, occupied_ids
        )
        b_block = b_rows.transpose(1, 0, 2, 3)
        contribution = _density_from_target_amplitude_block(a_block, b_block)
        return (
            dmoo + contribution.occupied,
            dmvv + contribution.virtual,
        ), None

    (dmoo, dmvv), _ = jax.lax.scan(
        jax.checkpoint(virtual_block_body), (dmoo0, dmvv0), block_ids
    )
    return IAOMP2Density(_hermitize(dmoo), _hermitize(dmvv))


def strong_domain_mp2_density(mf, domain, static, fragment_index):
    """Evaluate one fragment's target-conditioned density in its strong ED."""

    if not isinstance(static, IAOFragmentMP2StaticSelections):
        raise TypeError("static must be IAOFragmentMP2StaticSelections")
    if not isinstance(domain, IAOMP2StrongDomain):
        raise TypeError("domain must be IAOMP2StrongDomain")
    fragment_index = int(fragment_index)
    fragment = static.fragments[fragment_index]
    nocc = int(domain.occupied_coeff.shape[1])
    coeff = np.concatenate(
        (domain.occupied_coeff, domain.virtual_coeff), axis=1
    )
    nvir = domain.virtual_coeff.shape[1]
    ntarget = domain.target_projection.shape[0]
    profile_lov = resource_profile.start()
    lov = lno_base.get_local_Lov(
        mf,
        coeff,
        nocc,
        fragment.extended_atoms,
        integral_direct=True,
    ).reshape((-1, nocc, nvir))
    resource_profile.finish(
        "iao_lis.strong_domain_local_lov",
        profile_lov,
        fragment_index=fragment_index,
        coeff_shape=tuple(coeff.shape),
        lov_shape=tuple(lov.shape),
        naux=lov.shape[0],
        nocc=nocc,
        nvir=nvir,
        ntarget=ntarget,
        lov_mib=resource_profile.estimated_array_mib(lov),
        local_direct_block_mb=lno_base._local_direct_int3c_block_mb(),
    )

    naux = lov.shape[0]
    lov_mib = resource_profile.estimated_array_mib(lov)
    profile_details = {
        "fragment_index": fragment_index,
        "coeff_shape": tuple(coeff.shape),
        "lov_shape": tuple(lov.shape),
        "naux": naux,
        "nocc": nocc,
        "nvir": nvir,
        "ntarget": ntarget,
        "lov_mib": lov_mib,
    }

    max_memory_mb = static.thresholds.mp2_block_memory_mb
    block_nvir = _mp2_density_virtual_block_size(
        lov, ntarget, max_memory_mb
    )
    nblock = (nvir + block_nvir - 1) // block_nvir
    itemsize = onp.dtype(lov.dtype).itemsize
    profile_density = resource_profile.start()
    density = strong_domain_mp2_density_from_lov(
        lov,
        domain.occupied_energy,
        domain.virtual_energy,
        domain.target_projection,
        max_memory_mb=max_memory_mb,
    )
    resource_profile.finish(
        "iao_lis.strong_domain_mp2_density",
        profile_density,
        **profile_details,
        block_nvir=block_nvir,
        nblock=nblock,
        max_memory_mb=max_memory_mb,
        full_target_amplitudes_mib=(
            ntarget * nocc * nvir * nvir * itemsize / 1024.0**2
        ),
        block_target_amplitudes_mib=(
            2 * ntarget * nocc * nvir * block_nvir * itemsize / 1024.0**2
        ),
        estimated_block_workspace_mib=(
            itemsize * nocc * nvir * max(2 * ntarget + 6, 1)
            * block_nvir / 1024.0**2
        ),
        occupied_density_shape=tuple(density.occupied.shape),
        virtual_density_shape=tuple(density.virtual.shape),
        density_mib=resource_profile.estimated_array_mib(
            density.occupied, density.virtual
        ),
    )
    return density


def _union_partner_iao_indices(static, fragment_index):
    fragment = static.fragments[int(fragment_index)]
    arrays = [
        onp.asarray(static.frag_lolist[int(partner)], dtype=onp.int32)
        for partner in fragment.strong_fragments
    ]
    if not arrays:
        return onp.zeros((0,), dtype=onp.int32)
    return onp.unique(onp.concatenate(arrays)).astype(onp.int32, copy=False)


def strong_domain_prescreen(common, static, fragment_index, *, domain=None):
    """Return a lightweight legacy-style view of one IAO strong ED.

    This adapter is useful for reporting and transition tests.  The occupied
    and virtual coefficient arrays are expressed in the local AO basis of
    ``extended_primary_domain``.  ``strong_lmo_indices`` are the union of IAO
    column indices belonging to the strong partner *fragments*, not the
    fragment-number array itself.
    """

    if domain is None:
        domain = build_strong_ed_domain(common, static, fragment_index)
    fragment_index = int(fragment_index)
    fragment = static.fragments[fragment_index]
    return {
        "fragment_index": fragment_index,
        "lo_indices": onp.asarray(
            static.frag_lolist[fragment_index], dtype=onp.int32
        ),
        "strong_lmo_indices": _union_partner_iao_indices(
            static, fragment_index
        ),
        "extended_bp_domain": onp.asarray(
            fragment.pao_center_atoms, dtype=onp.int32
        ),
        "extended_primary_domain": onp.asarray(
            fragment.extended_atoms, dtype=onp.int32
        ),
        "occ_prescreen_energies": domain.occupied_energy,
        "occ_prescreen_coeff": domain.occupied_coeff,
        "vir_prescreen_energies": domain.virtual_energy,
        "vir_prescreen_coeff": domain.virtual_coeff,
        "orbfragloc": common.iao_coeff[:, fragment.iao_indices],
    }


def _row_gram_keep_numpy(matrix, threshold):
    matrix = onp.asarray(jax.device_get(matrix))
    if matrix.ndim != 2:
        raise ValueError("fragment projection must be rank two")
    if matrix.shape[0] == 0:
        return onp.zeros((0,), dtype=onp.int32)
    # Determine numerical rank from singular values, rather than comparing
    # eigenvalues of M M^H with threshold**2.  Exact null eigenvalues of the
    # Gram matrix acquire O(eps) roundoff, which is much larger than a typical
    # squared singular-value cutoff (e.g. 1e-20 for THRESH_INTERNAL=1e-10).
    # The differentiable rebuild below still uses the equivalent Hermitian
    # eigenproblem; its ascending retained labels are simply the final `rank`
    # columns.
    singular = onp_scipy_linalg.svdvals(matrix, check_finite=False)
    rank = int(onp.count_nonzero(singular > float(threshold)))
    return onp.arange(
        matrix.shape[0] - rank, matrix.shape[0], dtype=onp.int32
    )


def _fixed_row_space(matrix, keep):
    """Differentiably rebuild a fixed-rank row space in column coordinates."""

    matrix = np.asarray(matrix)
    keep = onp.asarray(keep, dtype=onp.int32)
    if matrix.shape[0] == 0 or keep.size == 0:
        return np.zeros((matrix.shape[1], 0), dtype=matrix.dtype)
    gram = _hermitize(matrix @ matrix.T.conj())
    _, vectors = scipy.linalg.eigh(
        gram, deg_thresh=lno_base.COMPRESS_DEG_THRESH
    )
    candidate = matrix.T.conj() @ vectors[:, keep]
    # The fixed reference rank guarantees a nonsingular retained Gram matrix.
    metric = _hermitize(candidate.T.conj() @ candidate)
    chol = np.linalg.cholesky(metric)
    return np.linalg.solve(chol, candidate.T.conj()).T.conj()


def _domain_density_in_active_spaces(common, static, fragment_index,
                                     domain, density):
    fragment = static.fragments[int(fragment_index)]
    ao_indices = fragment.extended_ao_indices
    overlap_to_domain = common.s1e[:, ao_indices]
    occupied_map = (
        common.occupied_coeff.T.conj()
        @ overlap_to_domain
        @ domain.occupied_coeff
    )
    virtual_map = (
        common.virtual_coeff.T.conj()
        @ overlap_to_domain
        @ domain.virtual_coeff
    )
    dmoo = occupied_map @ density.occupied @ occupied_map.T.conj()
    dmvv = virtual_map @ density.virtual @ virtual_map.T.conj()
    return _hermitize(dmoo), _hermitize(dmvv)


def _internal_projection_matrices(common, fragment_index):
    data = common.fragment_occupied_data[int(fragment_index)]
    occ_projection = data.iao_occ_overlap
    vir_projection = (
        data.iao_coeff.T.conj() @ common.s1e @ common.virtual_coeff
    )
    return occ_projection, vir_projection


def _external_density(density, internal):
    identity = np.eye(density.shape[0], dtype=density.dtype)
    projector = identity - internal @ internal.T.conj()
    return _hermitize(projector @ density @ projector)


def _density_keep_numpy(density, internal, threshold, full_space):
    nspace = int(density.shape[0])
    if full_space:
        return onp.zeros((0,), dtype=onp.int32)
    external = onp.asarray(jax.device_get(_external_density(density, internal)))
    values = onp_scipy_linalg.eigh(
        0.5 * (external + external.T.conj()),
        eigvals_only=True,
        check_finite=False,
    )
    return onp.where(onp.abs(onp.real(values)) > float(threshold))[0].astype(
        onp.int32
    )


def _fixed_density_space(density, internal, keep, full_space):
    nspace = int(density.shape[0])
    if full_space:
        return np.eye(nspace, dtype=density.dtype)
    keep = onp.asarray(keep, dtype=onp.int32)
    if keep.size == 0:
        return internal
    external = _external_density(density, internal)
    _, vectors = scipy.linalg.eigh(
        external, deg_thresh=lno_base.COMPRESS_DEG_THRESH
    )
    lno = vectors[:, keep]
    if internal.shape[1]:
        lno = lno - internal @ (internal.T.conj() @ lno)
    # Numerical projection against the internal space changes only roundoff,
    # but Cholesky normalization makes the rebuilt span explicitly orthonormal.
    metric = _hermitize(lno.T.conj() @ lno)
    chol = np.linalg.cholesky(metric)
    lno = np.linalg.solve(chol, lno.T.conj()).T.conj()
    return np.concatenate((internal, lno), axis=1)


def _reference_fragment_selection(
    common,
    mp2_static,
    fragment_index,
    density_occupied_active,
    density_virtual_active,
    *,
    thresh_occ,
    thresh_vir,
    internal_rank_threshold,
):
    occ_projection, vir_projection = _internal_projection_matrices(
        common, fragment_index
    )
    occ_internal_keep = _row_gram_keep_numpy(
        occ_projection, internal_rank_threshold
    )
    vir_internal_keep = _row_gram_keep_numpy(
        vir_projection, internal_rank_threshold
    )
    occ_internal = _fixed_row_space(occ_projection, occ_internal_keep)
    vir_internal = _fixed_row_space(vir_projection, vir_internal_keep)

    full_occ = float(thresh_occ) <= 0.0
    full_vir = float(thresh_vir) <= 0.0
    occ_lno_keep = _density_keep_numpy(
        density_occupied_active, occ_internal, thresh_occ, full_occ
    )
    vir_lno_keep = _density_keep_numpy(
        density_virtual_active, vir_internal, thresh_vir, full_vir
    )
    return IAOLISFragmentStaticSelection(
        fragment_index=int(fragment_index),
        internal_occ_keep=occ_internal_keep,
        internal_vir_keep=vir_internal_keep,
        occupied_lno_keep=occ_lno_keep,
        virtual_lno_keep=vir_lno_keep,
        full_occupied_space=full_occ,
        full_virtual_space=full_vir,
    )


def build_iao_lis_fragment_static_selection(
    mf,
    mp2_static,
    fragment_index,
    *,
    common=None,
    domain=None,
    thresh_occ=1e-4,
    thresh_vir=1e-5,
    internal_rank_threshold=IAO_LIS_INTERNAL_RANK_THRESHOLD,
):
    """Select the fixed LIS ranks for one fragment.

    ``domain`` may be supplied by a caller that constructs ED orbital frames
    separately from the target-conditioned MP2-density calculation.  This is
    the boundary used by the MPI driver: its root rank owns all discrete domain
    construction, while independent ranks evaluate this fragment-local
    operation.  Supplying a domain does not change the serial equations or
    any retained-rank decision.
    """

    if not isinstance(mp2_static, IAOFragmentMP2StaticSelections):
        raise TypeError("mp2_static must be IAOFragmentMP2StaticSelections")
    fragment_index = int(fragment_index)
    if fragment_index < 0 or fragment_index >= len(mp2_static.fragments):
        raise IndexError(
            f"fragment_index={fragment_index} is outside "
            f"[0, {len(mp2_static.fragments)})"
        )
    for name, value in (
        ("thresh_occ", thresh_occ),
        ("thresh_vir", thresh_vir),
        ("internal_rank_threshold", internal_rank_threshold),
    ):
        if float(value) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if common is None:
        common = rebuild_iao_mp2_common(mf, mp2_static)
    if not isinstance(common, IAOFragmentMP2ContinuousData):
        raise TypeError("common must be IAOFragmentMP2ContinuousData")
    if domain is None:
        domain = build_strong_ed_domain(
            common, mp2_static, fragment_index
        )
    if not isinstance(domain, IAOMP2StrongDomain):
        raise TypeError("domain must be IAOMP2StrongDomain")

    density = strong_domain_mp2_density(
        mf, domain, mp2_static, fragment_index
    )
    dmoo_active, dmvv_active = _domain_density_in_active_spaces(
        common, mp2_static, fragment_index, domain, density
    )
    return _reference_fragment_selection(
        common,
        mp2_static,
        fragment_index,
        dmoo_active,
        dmvv_active,
        thresh_occ=thresh_occ,
        thresh_vir=thresh_vir,
        internal_rank_threshold=internal_rank_threshold,
    )


def build_iao_lis_static_selections(
    mf,
    mp2_static,
    *,
    common=None,
    thresh_occ=1e-4,
    thresh_vir=1e-5,
    internal_rank_threshold=IAO_LIS_INTERNAL_RANK_THRESHOLD,
):
    """Select fixed internal/LNO ranks from a concrete reference geometry."""

    if not isinstance(mp2_static, IAOFragmentMP2StaticSelections):
        raise TypeError("mp2_static must be IAOFragmentMP2StaticSelections")
    for name, value in (
        ("thresh_occ", thresh_occ),
        ("thresh_vir", thresh_vir),
        ("internal_rank_threshold", internal_rank_threshold),
    ):
        if float(value) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if common is None:
        common = rebuild_iao_mp2_common(mf, mp2_static)
    if not isinstance(common, IAOFragmentMP2ContinuousData):
        raise TypeError("common must be IAOFragmentMP2ContinuousData")

    fragments = tuple(
        build_iao_lis_fragment_static_selection(
            mf,
            mp2_static,
            fragment_index,
            common=common,
            thresh_occ=thresh_occ,
            thresh_vir=thresh_vir,
            internal_rank_threshold=internal_rank_threshold,
        )
        for fragment_index in range(len(mp2_static.fragments))
    )

    return IAOFragmentLISStaticSelections(
        mp2_static=mp2_static,
        thresh_occ=float(thresh_occ),
        thresh_vir=float(thresh_vir),
        internal_rank_threshold=float(internal_rank_threshold),
        fragments=fragments,
    )


def _active_complement(selected, threshold):
    """Frozen-gauge complement used only to complete the impurity MO layout."""

    nspace = int(selected.shape[0])
    if selected.shape[1] == nspace:
        return np.zeros((nspace, 0), dtype=selected.dtype)
    identity = np.eye(nspace, dtype=selected.dtype)
    return lno_base._dlno_outside_space(
        identity,
        selected,
        max(float(threshold), 1e-8),
    )


def _semicanonical_space(coeff, fock):
    if coeff.shape[1] == 0:
        return coeff
    return lno_base.semicanonicalize(fock, coeff)[1]


def _assemble_full_mo_layout(
    mf,
    static,
    occupied_coeff,
    virtual_coeff,
    fock,
    occupied_selected,
    virtual_selected,
    rank_threshold,
):
    mo_coeff = np.asarray(mf.mo_coeff)
    mo_occ_host = onp.asarray(jax.device_get(mf.mo_occ))
    nmo = int(mo_occ_host.size)
    all_indices = onp.arange(nmo, dtype=onp.int32)
    occupied_indices = all_indices[mo_occ_host > lno_base.THRESH_OCC]
    virtual_indices = all_indices[mo_occ_host <= lno_base.THRESH_OCC]
    active_occ_indices = onp.asarray(
        static.active_occ_indices, dtype=onp.int32
    )
    active_vir_indices = onp.asarray(
        static.active_vir_indices, dtype=onp.int32
    )
    frozen_occ_indices = onp.setdiff1d(
        occupied_indices, active_occ_indices, assume_unique=False
    )
    frozen_vir_indices = onp.setdiff1d(
        virtual_indices, active_vir_indices, assume_unique=False
    )

    occ_complement = _active_complement(
        occupied_selected, rank_threshold
    )
    vir_complement = _active_complement(
        virtual_selected, rank_threshold
    )
    occupied_active = _semicanonical_space(
        occupied_coeff @ occupied_selected,
        fock,
    )
    virtual_active = _semicanonical_space(
        virtual_coeff @ virtual_selected,
        fock,
    )
    occupied_discarded = (
        occupied_coeff @ occ_complement
    )
    virtual_discarded = (
        virtual_coeff @ vir_complement
    )

    blocks = (
        mo_coeff[:, frozen_occ_indices],
        occupied_discarded,
        occupied_active,
        virtual_active,
        virtual_discarded,
        mo_coeff[:, frozen_vir_indices],
    )
    full_coeff = np.concatenate(blocks, axis=1)
    nocc_total = int(occupied_indices.size)
    n_frozen_occ = int(frozen_occ_indices.size + occ_complement.shape[1])
    n_active_vir = int(virtual_active.shape[1])
    frozen = onp.concatenate((
        onp.arange(n_frozen_occ, dtype=onp.int32),
        onp.arange(
            nocc_total + n_active_vir, nmo, dtype=onp.int32
        ),
    ))
    return full_coeff, frozen, occupied_active, virtual_active


def build_fragment_lis(
    mf,
    common,
    static,
    fragment_index,
    *,
    domain=None,
    density=None,
):
    """Rebuild one fixed-rank IAO-MP2 LIS on the differentiable path."""

    if not isinstance(static, IAOFragmentLISStaticSelections):
        raise TypeError("static must be IAOFragmentLISStaticSelections")
    mp2_static = static.mp2_static
    if not isinstance(common, IAOFragmentMP2ContinuousData):
        raise TypeError("common must be IAOFragmentMP2ContinuousData")
    fragment_index = int(fragment_index)
    selection = static.fragments[fragment_index]
    if selection.fragment_index != fragment_index:
        raise ValueError("fragment selection order is inconsistent")

    if domain is None:
        domain = build_strong_ed_domain(
            common, mp2_static, fragment_index
        )
    if density is None:
        density = strong_domain_mp2_density(
            mf, domain, mp2_static, fragment_index
        )
    dmoo_active, dmvv_active = _domain_density_in_active_spaces(
        common, mp2_static, fragment_index, domain, density
    )

    occ_projection, vir_projection = _internal_projection_matrices(
        common, fragment_index
    )
    occ_internal = _fixed_row_space(
        occ_projection, selection.internal_occ_keep
    )
    vir_internal = _fixed_row_space(
        vir_projection, selection.internal_vir_keep
    )
    occ_selected = _fixed_density_space(
        dmoo_active,
        occ_internal,
        selection.occupied_lno_keep,
        selection.full_occupied_space,
    )
    vir_selected = _fixed_density_space(
        dmvv_active,
        vir_internal,
        selection.virtual_lno_keep,
        selection.full_virtual_space,
    )

    full_coeff, frozen, occupied_active, virtual_active = (
        _assemble_full_mo_layout(
            mf,
            mp2_static,
            common.occupied_coeff,
            common.virtual_coeff,
            common.fock,
            occ_selected,
            vir_selected,
            static.internal_rank_threshold,
        )
    )
    fragment_data = common.fragment_occupied_data[fragment_index]
    occupied_projector = occ_selected @ occ_selected.T.conj()
    virtual_projector = vir_selected @ vir_selected.T.conj()
    return IAOFragmentLIS(
        mo_coeff=full_coeff,
        frozen=frozen,
        fragment_occupied_anchor=fragment_data.occupied_projection,
        fragment_iao_coeff=fragment_data.iao_coeff,
        active_occupied_coeff=occupied_active,
        active_virtual_coeff=virtual_active,
        occupied_projector=_hermitize(occupied_projector),
        virtual_projector=_hermitize(virtual_projector),
        density_occupied_ed=density.occupied,
        density_virtual_ed=density.virtual,
        density_occupied_active=dmoo_active,
        density_virtual_active=dmvv_active,
        domain=domain,
        n_internal_occ=int(selection.internal_occ_keep.size),
        n_internal_vir=int(selection.internal_vir_keep.size),
        n_lno_occ=(
            int(common.occupied_coeff.shape[1]
                - selection.internal_occ_keep.size)
            if selection.full_occupied_space
            else int(selection.occupied_lno_keep.size)
        ),
        n_lno_vir=(
            int(common.virtual_coeff.shape[1]
                - selection.internal_vir_keep.size)
            if selection.full_virtual_space
            else int(selection.virtual_lno_keep.size)
        ),
    )
