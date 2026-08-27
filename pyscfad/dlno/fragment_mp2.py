"""Algebraic primitives for IAO-fragment MP2 energy partitions.

This module deliberately contains no domain-selection logic.  It separates
two objects that play different roles in an IAO-based local correlation
calculation:

* the occupied *span* associated with an IAO fragment, which is useful for
  constructing spatial domains; and
* the positive-semidefinite occupied weight of that fragment, which is
  required for an additive correlation-energy partition.

The routines are NumPy implementations intended first for validating the
fragment algebra on small systems.  A JAX implementation can use the same
identities once the domain construction has been settled.
"""

from dataclasses import dataclass
from functools import partial
import os

import jax
import jax.numpy as jnp
import numpy as np


__all__ = [
    "FragmentOccupiedData",
    "FragmentPairEnergies",
    "FragmentPairPartition",
    "build_fragment_occupied_data",
    "fragment_pair_energy_from_ovov",
    "fragment_pair_energy_from_lov",
    "fragment_pair_energy_from_lov_jax",
    "partition_fragment_pair_energies",
]


@dataclass(frozen=True)
class FragmentOccupiedData:
    r"""Occupied-space data associated with one IAO fragment.

    Attributes
    ----------
    iao_indices
        Indices of the IAOs assigned to the fragment.
    iao_coeff
        Raw AO coefficients :math:`A_F` of the fragment IAOs.
    iao_occ_overlap
        Matrix :math:`M_F=A_F^\dagger S C_o`.
    occupied_projection
        Occupied projection of the raw IAOs,
        :math:`X_F=C_o M_F^\dagger`.
    occupied_weight
        Additive occupied weight :math:`W_F=M_F^\dagger M_F`.
    singular_values
        Singular values of :math:`M_F`.
    right_singular_vectors
        Retained right singular vectors of :math:`M_F`, expressed in the
        input occupied-orbital basis.
    occupied_subspace_coeff
        AO coefficients of the retained normalized occupied span,
        :math:`C_o V_F`.

    Notes
    -----
    ``occupied_weight`` and the projector onto
    ``occupied_subspace_coeff`` are generally different.  The latter must
    not be used to partition a correlation energy when different fragment
    occupied spans overlap.
    """

    iao_indices: np.ndarray
    iao_coeff: np.ndarray
    iao_occ_overlap: np.ndarray
    occupied_projection: np.ndarray
    occupied_weight: np.ndarray
    singular_values: np.ndarray
    right_singular_vectors: np.ndarray
    occupied_subspace_coeff: np.ndarray

    @property
    def rank(self):
        """Numerical rank selected by ``svd_thr``."""

        return self.right_singular_vectors.shape[1]


@dataclass(frozen=True)
class FragmentPairEnergies:
    """Closed-shell MP2 energies resolved by two fragment weights.

    ``opposite_spin`` is the direct MP2 contribution.  ``same_spin`` is
    direct minus exchange, and ``total`` is their sum.  Arrays have shape
    ``(nleft, nright)`` unless one or both weight arguments were supplied as
    a single matrix; singleton fragment axes are then removed.
    """

    total: np.ndarray
    opposite_spin: np.ndarray
    same_spin: np.ndarray

    def summed(self):
        """Return the three components summed over all fragment pairs."""

        return FragmentPairEnergies(
            total=np.sum(self.total),
            opposite_spin=np.sum(self.opposite_spin),
            same_spin=np.sum(self.same_spin),
        )


@dataclass(frozen=True)
class FragmentPairPartition:
    """Exact strong/weak masking of fragment-pair energy arrays."""

    strong: FragmentPairEnergies
    weak: FragmentPairEnergies


def _validate_coefficients(occ_coeff, iao_coeff, s1e):
    occ_coeff = np.asarray(occ_coeff)
    iao_coeff = np.asarray(iao_coeff)
    s1e = np.asarray(s1e)

    if occ_coeff.ndim != 2:
        raise ValueError("occ_coeff must be a rank-2 array")
    if iao_coeff.ndim != 2:
        raise ValueError("iao_coeff must be a rank-2 array")
    if occ_coeff.shape[0] != iao_coeff.shape[0]:
        raise ValueError("occ_coeff and iao_coeff must use the same AO basis")
    nao = occ_coeff.shape[0]
    if s1e.shape != (nao, nao):
        raise ValueError("s1e must have shape (nao, nao)")
    return occ_coeff, iao_coeff, s1e


def _validate_fragment_partition(frag_lolist, niao):
    fragments = []
    for ifrag, indices in enumerate(frag_lolist):
        raw = np.asarray(indices)
        if raw.ndim > 1:
            raise ValueError(
                f"fragment {ifrag} IAO indices must be a one-dimensional array"
            )
        if raw.size and not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(f"fragment {ifrag} IAO indices must be integers")
        idx = raw.astype(np.int64, copy=False).reshape(-1)
        if np.any(idx < 0) or np.any(idx >= niao):
            raise ValueError(f"fragment {ifrag} contains an out-of-range IAO index")
        if np.unique(idx).size != idx.size:
            raise ValueError(f"fragment {ifrag} contains duplicate IAO indices")
        fragments.append(idx)

    if not fragments:
        if niao:
            raise ValueError("frag_lolist does not contain the IAO columns")
        return tuple()

    all_indices = np.concatenate(fragments)
    if np.unique(all_indices).size != all_indices.size:
        raise ValueError("an IAO is assigned to more than one fragment")
    if all_indices.size != niao or not np.array_equal(
        np.sort(all_indices), np.arange(niao)
    ):
        raise ValueError("frag_lolist must assign every IAO exactly once")
    return tuple(fragments)


def build_fragment_occupied_data(
    occ_coeff,
    iao_coeff,
    frag_lolist,
    s1e,
    svd_thr=1e-10,
):
    r"""Construct occupied spans and additive weights for IAO fragments.

    Parameters
    ----------
    occ_coeff : ndarray, shape (nao, nocc)
        Orthonormal occupied-orbital coefficients :math:`C_o`.
    iao_coeff : ndarray, shape (nao, niao)
        Orthonormal IAO coefficients :math:`A`.
    frag_lolist : sequence of one-dimensional integer arrays
        A disjoint, complete partition of the IAO columns.
    s1e : ndarray, shape (nao, nao)
        AO overlap matrix.
    svd_thr : float
        Absolute singular-value threshold used only to define the normalized
        occupied span.  The raw weight :math:`W_F` is never thresholded.

    Returns
    -------
    tuple of :class:`FragmentOccupiedData`

    Notes
    -----
    If the complete IAO set spans the occupied space, then

    .. math::

        \sum_F W_F = I_o,

    even though the normalized occupied spans of different fragments may
    overlap.  This resolution of the identity is the property needed for
    fragment energies to sum to the canonical MP2 energy.
    """

    occ_coeff, iao_coeff, s1e = _validate_coefficients(
        occ_coeff, iao_coeff, s1e
    )
    if svd_thr < 0:
        raise ValueError("svd_thr must be non-negative")
    fragments = _validate_fragment_partition(frag_lolist, iao_coeff.shape[1])

    data = []
    nocc = occ_coeff.shape[1]
    for idx in fragments:
        af = iao_coeff[:, idx]
        mf = af.conj().T @ s1e @ occ_coeff
        xf = occ_coeff @ mf.conj().T
        wf = mf.conj().T @ mf

        if min(mf.shape) == 0:
            sigma = np.zeros((0,), dtype=np.real(mf).dtype)
            right = np.zeros((nocc, 0), dtype=mf.dtype)
        else:
            _, sigma, vh = np.linalg.svd(mf, full_matrices=False)
            keep = sigma > svd_thr
            right = vh.conj().T[:, keep]
        occupied_subspace = occ_coeff @ right

        data.append(
            FragmentOccupiedData(
                iao_indices=idx.copy(),
                iao_coeff=af,
                iao_occ_overlap=mf,
                occupied_projection=xf,
                occupied_weight=wf,
                singular_values=sigma,
                right_singular_vectors=right,
                occupied_subspace_coeff=occupied_subspace,
            )
        )
    return tuple(data)


def _as_weight_stack(weights, nocc, name):
    array = np.asarray(weights)
    was_single = array.ndim == 2
    if was_single:
        array = array[None, :, :]
    if array.ndim != 3 or array.shape[1:] != (nocc, nocc):
        raise ValueError(
            f"{name} must have shape (nocc, nocc) or (nfrag, nocc, nocc)"
        )
    if not np.allclose(array, array.swapaxes(-1, -2).conj(), atol=1e-10):
        raise ValueError(f"{name} must contain Hermitian matrices")
    return array, was_single


def _remove_single_fragment_axes(array, left_single, right_single):
    if left_single and right_single:
        return array[0, 0]
    if left_single:
        return array[0, :]
    if right_single:
        return array[:, 0]
    return array


def fragment_pair_energy_from_ovov(
    ovov,
    e_occ,
    e_vir,
    weights,
    partner_weights=None,
):
    r"""Evaluate an exact two-sided IAO-fragment DF-MP2 partition.

    Parameters
    ----------
    ovov : ndarray, shape (nocc, nvir, nocc, nvir)
        Electron-repulsion integrals :math:`(ia|jb)`, which may have been
        formed from density-fitting factors.
    e_occ, e_vir : ndarray
        Semicanonical occupied and virtual orbital energies.
    weights : ndarray
        One occupied weight or a stack of left-fragment weights.
    partner_weights : ndarray, optional
        One occupied weight or a stack of right-fragment weights.  If
        omitted, ``weights`` is used on both occupied lines.

    Returns
    -------
    :class:`FragmentPairEnergies`
        For left fragment ``F`` and right fragment ``G``, the total entry is

        .. math::

            E_{FG}=\sum_{pqrsab}W^F_{pq}W^G_{rs}t_{pr}^{ab}
            [2(qa|sb)-(qb|sa)].

        Consequently, if :math:`\sum_F W_F=I_o`, summing all entries of the
        returned square matrix gives the canonical closed-shell MP2 energy.

    Notes
    -----
    This diagnostic implementation forms the dense MP2 amplitudes.  It is
    intended for small-system validation, not the eventual large-domain
    production path.
    """

    ovov = np.asarray(ovov)
    e_occ = np.asarray(e_occ)
    e_vir = np.asarray(e_vir)
    if ovov.ndim != 4:
        raise ValueError("ovov must be a rank-4 array")
    nocc, nvir, nocc1, nvir1 = ovov.shape
    if nocc1 != nocc or nvir1 != nvir:
        raise ValueError("ovov must have shape (nocc, nvir, nocc, nvir)")
    if e_occ.shape != (nocc,):
        raise ValueError("e_occ must have shape (nocc,)")
    if e_vir.shape != (nvir,):
        raise ValueError("e_vir must have shape (nvir,)")

    left, left_single = _as_weight_stack(weights, nocc, "weights")
    if partner_weights is None:
        right, right_single = left, left_single
    else:
        right, right_single = _as_weight_stack(
            partner_weights, nocc, "partner_weights"
        )

    eia = e_occ[:, None] - e_vir[None, :]
    denominator = eia[:, None, :, None] + eia[None, :, None, :]
    if np.any(np.abs(denominator) < 1e-14):
        raise ValueError("an MP2 energy denominator is zero")
    integrals = ovov.transpose(0, 2, 1, 3)
    amplitudes = integrals / denominator

    direct = np.einsum(
        "fpq,grs,prab,qasb->fg",
        left,
        right,
        amplitudes,
        ovov,
        optimize=True,
    )
    exchange = np.einsum(
        "fpq,grs,prab,qbsa->fg",
        left,
        right,
        amplitudes,
        ovov,
        optimize=True,
    )
    opposite_spin = direct
    same_spin = direct - exchange
    total = opposite_spin + same_spin

    return FragmentPairEnergies(
        total=_remove_single_fragment_axes(total, left_single, right_single),
        opposite_spin=_remove_single_fragment_axes(
            opposite_spin, left_single, right_single
        ),
        same_spin=_remove_single_fragment_axes(
            same_spin, left_single, right_single
        ),
    )


def _virtual_block_size(lov, nleft, nright, max_memory_mb):
    """Choose a conservative square virtual block for the dense temporaries."""
    if max_memory_mb <= 0:
        raise ValueError("max_memory_mb must be positive")
    nocc = lov.shape[1]
    itemsize = np.dtype(lov.dtype).itemsize
    # Per (a,b) block we hold the ERIs, amplitudes, a denominator, and the
    # occupied-weight transforms, plus comparatively small fragment
    # contractions.  A factor of five leaves room for einsum workspaces.
    bytes_per_virtual_pair = max(5 * nocc * nocc * itemsize, 1)
    budget = max_memory_mb * 1024.0**2
    block = int(np.sqrt(budget / bytes_per_virtual_pair))
    return max(1, min(lov.shape[2], block))


def fragment_pair_energy_from_lov(
    lov,
    e_occ,
    e_vir,
    weights,
    partner_weights=None,
    *,
    target_factor=None,
    block_nvir=None,
    max_memory_mb=256.0,
):
    r"""Memory-bounded two-sided fragment MP2 energy from DF factors.

    Parameters are identical to :func:`fragment_pair_energy_from_ovov`, except
    that ``lov`` has shape ``(naux, nocc, nvir)`` and contains the density-
    fitting factors :math:`L^P_{ia}`.  The virtual indices are processed in
    square blocks, so the largest rank-four temporary is
    ``O(nocc**2 * block_nvir**2)`` rather than
    ``O(nocc**2 * nvir**2)``.

    This routine is the production-sized counterpart of the dense diagnostic
    function.  It retains the same directed fragment-pair and OS/SS
    conventions, including support for stacks of left and right weights.

    For the common single-fragment case, ``target_factor`` may contain a thin
    factor :math:`X` of the left occupied weight,

    .. math::

        W_{\mathrm{left}} = X^\dagger X.

    The factorized path is algebraically identical to the dense-weight path
    but contracts the small target dimension before the occupied partner
    weight.  Stacked weights continue to use the general dense path.
    """
    lov = np.asarray(lov)
    e_occ = np.asarray(e_occ)
    e_vir = np.asarray(e_vir)
    if lov.ndim != 3:
        raise ValueError("lov must have shape (naux, nocc, nvir)")
    _, nocc, nvir = lov.shape
    if e_occ.shape != (nocc,):
        raise ValueError("e_occ must have shape (nocc,)")
    if e_vir.shape != (nvir,):
        raise ValueError("e_vir must have shape (nvir,)")

    left, left_single = _as_weight_stack(weights, nocc, "weights")
    if partner_weights is None:
        right, right_single = left, left_single
    else:
        right, right_single = _as_weight_stack(
            partner_weights, nocc, "partner_weights"
        )

    if target_factor is not None:
        target_factor = np.asarray(target_factor)
        if target_factor.ndim != 2 or target_factor.shape[1] != nocc:
            raise ValueError(
                "target_factor must have shape (ntarget, nocc)"
            )
        if not (left_single and right_single):
            raise ValueError(
                "target_factor is supported only for single left and right "
                "weights"
            )
        factor_weight = target_factor.conj().T @ target_factor
        if not np.allclose(factor_weight, left[0], atol=1e-10, rtol=1e-10):
            raise ValueError(
                "target_factor must satisfy weights = "
                "target_factor^H target_factor"
            )

    if block_nvir is None:
        block_nvir = _virtual_block_size(
            lov, left.shape[0], right.shape[0], max_memory_mb
        )
    elif not isinstance(block_nvir, (int, np.integer)) or block_nvir <= 0:
        raise ValueError("block_nvir must be a positive integer")
    block_nvir = min(int(block_nvir), nvir) if nvir else 1

    dtype_operands = (lov, e_occ, e_vir, left, right)
    if target_factor is not None:
        dtype_operands += (target_factor,)
    result_dtype = np.result_type(*dtype_operands)
    direct = np.zeros((left.shape[0], right.shape[0]), dtype=result_dtype)
    exchange = np.zeros_like(direct)
    eia = e_occ[:, None] - e_vir[None, :]
    single_weight_path = left_single and right_single

    def accumulate_orientation(amplitudes, integrals):
        """Accumulate one ordered pair of virtual blocks.

        ``integrals`` has axes ``(p,a,r,b)``.  Its occupied transpose is the
        exchange integral for the same ordered virtual block,

        ``(q b|s a) = (s a|q b)``.

        Transforming the amplitudes by the two occupied weights therefore
        supplies both the direct and exchange energies from the *same* DF
        reconstruction.  This replaces the former direct and exchange DF
        reconstructions of weighted ``Lov`` factors.
        """
        nonlocal direct, exchange
        eri_prab = integrals.transpose(0, 2, 1, 3)
        if single_weight_path:
            # B[q,s,a,b] = sum_{p,r} Wl[p,q] A[p,r,a,b] Wr[r,s].
            # The two contractions are deliberately explicit: this keeps
            # einsum from considering an outer product of the weight matrices.
            weighted_amplitudes = np.einsum(
                "pq,prab->qrab", left[0], amplitudes, optimize=True
            )
            weighted_amplitudes = np.einsum(
                "qrab,rs->qsab",
                weighted_amplitudes,
                right[0],
                optimize=True,
            )
            direct[0, 0] += np.einsum(
                "qsab,qsab->", weighted_amplitudes, eri_prab,
                optimize=True,
            )
            exchange[0, 0] += np.einsum(
                "qsab,sqab->", weighted_amplitudes, eri_prab,
                optimize=True,
            )
        else:
            # Do not present all fragment and occupied indices to one einsum.
            # NumPy can otherwise select an nfrag**2 * nocc**4 * nvir**2
            # contraction when its preferred four-occupied intermediate is
            # over the einsum memory limit.  Loop over the smaller fragment
            # stack, reduce the virtual and one occupied index immediately,
            # and obtain every fragment on the other side from a rank-2
            # intermediate.
            if left.shape[0] <= right.shape[0]:
                for fragment in range(left.shape[0]):
                    weighted = np.einsum(
                        "pq,prab->qrab",
                        left[fragment],
                        amplitudes,
                        optimize=True,
                    )
                    direct_kernel = np.einsum(
                        "qrab,qsab->rs",
                        weighted,
                        eri_prab,
                        optimize=True,
                    )
                    exchange_kernel = np.einsum(
                        "qrab,sqab->rs",
                        weighted,
                        eri_prab,
                        optimize=True,
                    )
                    direct[fragment] += np.einsum(
                        "grs,rs->g", right, direct_kernel,
                        optimize=True,
                    )
                    exchange[fragment] += np.einsum(
                        "grs,rs->g", right, exchange_kernel,
                        optimize=True,
                    )
            else:
                for fragment in range(right.shape[0]):
                    weighted = np.einsum(
                        "prab,rs->psab",
                        amplitudes,
                        right[fragment],
                        optimize=True,
                    )
                    direct_kernel = np.einsum(
                        "psab,qsab->pq",
                        weighted,
                        eri_prab,
                        optimize=True,
                    )
                    exchange_kernel = np.einsum(
                        "psab,sqab->pq",
                        weighted,
                        eri_prab,
                        optimize=True,
                    )
                    direct[:, fragment] += np.einsum(
                        "fpq,pq->f", left, direct_kernel,
                        optimize=True,
                    )
                    exchange[:, fragment] += np.einsum(
                        "fpq,pq->f", left, exchange_kernel,
                        optimize=True,
                    )

    def target_weighted_amplitudes(amplitudes):
        r"""Apply :math:`X^*` and the dense partner weight to amplitudes."""
        weighted = np.einsum(
            "xp,prab->xrab",
            target_factor.conj(),
            amplitudes,
            optimize=True,
        )
        return np.einsum(
            "xrab,rs->xsab", weighted, right[0], optimize=True
        )

    def accumulate_target_factored_pair(amplitudes_ab, integrals_ab, offdiag):
        r"""Accumulate an upper virtual block and its transposed partner.

        If :math:`G_{AB}^{pr}=(pA|rB)`, then the target-projected exchange
        integrals for orientation ``AB`` are the target-projected direct
        integrals for ``BA`` (with the virtual axes transposed).  Reusing those
        two projections avoids a separate exchange transform in both
        orientations.
        """
        nonlocal direct, exchange
        eri_ab = integrals_ab.transpose(0, 2, 1, 3)
        eri_ba = eri_ab.transpose(1, 0, 3, 2)
        direct_integrals_ab = np.einsum(
            "xq,qsab->xsab", target_factor, eri_ab, optimize=True
        )
        direct_integrals_ba = np.einsum(
            "xq,qsab->xsab", target_factor, eri_ba, optimize=True
        )

        weighted_ab = target_weighted_amplitudes(amplitudes_ab)
        direct[0, 0] += np.einsum(
            "xsab,xsab->", weighted_ab, direct_integrals_ab,
            optimize=True,
        )
        exchange[0, 0] += np.einsum(
            "xsab,xsba->", weighted_ab, direct_integrals_ba,
            optimize=True,
        )
        del weighted_ab

        if offdiag:
            amplitudes_ba = amplitudes_ab.transpose(1, 0, 3, 2)
            weighted_ba = target_weighted_amplitudes(amplitudes_ba)
            direct[0, 0] += np.einsum(
                "xsab,xsab->", weighted_ba, direct_integrals_ba,
                optimize=True,
            )
            exchange[0, 0] += np.einsum(
                "xsab,xsba->", weighted_ba, direct_integrals_ab,
                optimize=True,
            )

    for a0 in range(0, nvir, block_nvir):
        a1 = min(a0 + block_nvir, nvir)
        lov_a = lov[:, :, a0:a1]
        eia_a = eia[:, a0:a1]
        # Only the upper triangle of virtual *blocks* needs an explicit DF
        # reconstruction.  For A != B, (iB|jA) is the occupied/virtual
        # transpose of (iA|jB), and its amplitudes follow by the same view.
        for b0 in range(a0, nvir, block_nvir):
            b1 = min(b0 + block_nvir, nvir)
            lov_b = lov[:, :, b0:b1]
            eia_b = eia[:, b0:b1]

            integrals_ab = np.einsum(
                "Lia,Ljb->iajb", lov_a, lov_b, optimize=True
            )
            denominator = (
                eia_a[:, None, :, None]
                + eia_b[None, :, None, :]
            )
            if np.any(np.abs(denominator) < 1e-14):
                raise ValueError("an MP2 energy denominator is zero")
            amplitudes = integrals_ab.transpose(0, 2, 1, 3) / denominator
            del denominator
            if target_factor is not None:
                accumulate_target_factored_pair(
                    amplitudes, integrals_ab, b0 != a0
                )
            else:
                accumulate_orientation(amplitudes, integrals_ab)

                if b0 != a0:
                    # The lower-triangular block is obtained without another
                    # auxiliary-index contraction:
                    #   G_BA[p,r,b,a] = G_AB[r,p,a,b].
                    integrals_ba = integrals_ab.transpose(2, 3, 0, 1)
                    amplitudes_ba = amplitudes.transpose(1, 0, 3, 2)
                    accumulate_orientation(amplitudes_ba, integrals_ba)

    opposite_spin = direct
    same_spin = direct - exchange
    total = opposite_spin + same_spin
    return FragmentPairEnergies(
        total=_remove_single_fragment_axes(
            total, left_single, right_single
        ),
        opposite_spin=_remove_single_fragment_axes(
            opposite_spin, left_single, right_single
        ),
        same_spin=_remove_single_fragment_axes(
            same_spin, left_single, right_single
        ),
    )


def _fragment_pair_energy_from_lov_jax_ad(
    lov,
    e_occ,
    e_vir,
    target_factor,
    partner_weight,
    *,
    block_nvir=None,
    max_memory_mb=256.0,
):
    r"""Differentiable two-sided fragment MP2 energy from DF factors.

    This is the fixed-topology gradient counterpart of
    :func:`fragment_pair_energy_from_lov` for its common single-target case.
    ``target_factor`` is the (generally rectangular) factor :math:`X_F` with
    :math:`W_F=X_F^\dagger X_F`; ``partner_weight`` is the occupied weight of
    the fixed strong-partner set.  Both are differentiated.

    Virtual indices are processed by a fixed-shape upper-triangular
    :func:`jax.lax.scan`.  Its body is wrapped in :func:`jax.checkpoint`, so
    reverse mode recomputes one local rank-four block and scatter-accumulates
    its cotangent before advancing to the next block.  Only the shared full
    inputs are retained; neither a full :math:`ovov` tensor nor one physical
    ``Lov`` slice per block pair is stored.
    """
    lov = jnp.asarray(lov)
    e_occ = jnp.asarray(e_occ)
    e_vir = jnp.asarray(e_vir)
    target_factor = jnp.asarray(target_factor)
    partner_weight = jnp.asarray(partner_weight)

    if lov.ndim != 3:
        raise ValueError("lov must have shape (naux, nocc, nvir)")
    _, nocc, nvir = lov.shape
    if e_occ.shape != (nocc,):
        raise ValueError("e_occ must have shape (nocc,)")
    if e_vir.shape != (nvir,):
        raise ValueError("e_vir must have shape (nvir,)")
    if target_factor.ndim != 2 or target_factor.shape[1] != nocc:
        raise ValueError("target_factor must have shape (ntarget, nocc)")
    if partner_weight.shape != (nocc, nocc):
        raise ValueError("partner_weight must have shape (nocc, nocc)")

    if block_nvir is None:
        # Shape and dtype are static at trace time, so the NumPy sizing helper
        # is safe here and does not inspect any traced values.
        block_nvir = _virtual_block_size(
            lov, 1, 1, max_memory_mb
        )
    elif not isinstance(block_nvir, (int, np.integer)) or block_nvir <= 0:
        raise ValueError("block_nvir must be a positive integer")
    block_nvir = min(int(block_nvir), nvir) if nvir else 1

    eia = e_occ[:, None] - e_vir[None, :]
    direct = jnp.zeros((), dtype=lov.dtype)
    exchange = jnp.zeros((), dtype=lov.dtype)

    def _block_energy(lov_a, lov_b, eia_a, eia_b, x_target, w_partner,
                      include_transpose, valid_a, valid_b):
        integrals_ab = jnp.einsum(
            "Lia,Ljb->iajb", lov_a, lov_b, optimize=True
        )
        denominator = (
            eia_a[:, None, :, None] + eia_b[None, :, None, :]
        )
        denominator = jnp.where(
            valid_a[None, None, :, None]
            & valid_b[None, None, None, :],
            denominator,
            jnp.ones((), dtype=denominator.dtype),
        )
        amplitudes_ab = (
            integrals_ab.transpose(0, 2, 1, 3) / denominator
        )

        eri_ab = integrals_ab.transpose(0, 2, 1, 3)
        eri_ba = eri_ab.transpose(1, 0, 3, 2)
        direct_integrals_ab = jnp.einsum(
            "xq,qsab->xsab", x_target, eri_ab, optimize=True
        )
        direct_integrals_ba = jnp.einsum(
            "xq,qsab->xsab", x_target, eri_ba, optimize=True
        )

        weighted_ab = jnp.einsum(
            "xp,prab->xrab", x_target.conj(), amplitudes_ab,
            optimize=True,
        )
        weighted_ab = jnp.einsum(
            "xrab,rs->xsab", weighted_ab, w_partner, optimize=True
        )
        block_direct = jnp.einsum(
            "xsab,xsab->", weighted_ab, direct_integrals_ab,
            optimize=True,
        )
        block_exchange = jnp.einsum(
            "xsab,xsba->", weighted_ab, direct_integrals_ba,
            optimize=True,
        )

        def _transpose_energy(_):
            amplitudes_ba = amplitudes_ab.transpose(1, 0, 3, 2)
            weighted_ba = jnp.einsum(
                "xp,prab->xrab", x_target.conj(), amplitudes_ba,
                optimize=True,
            )
            weighted_ba = jnp.einsum(
                "xrab,rs->xsab", weighted_ba, w_partner,
                optimize=True,
            )
            transpose_direct = jnp.einsum(
                "xsab,xsab->", weighted_ba, direct_integrals_ba,
                optimize=True,
            )
            transpose_exchange = jnp.einsum(
                "xsab,xsba->", weighted_ba, direct_integrals_ab,
                optimize=True,
            )
            return transpose_direct, transpose_exchange

        transpose_direct, transpose_exchange = jax.lax.cond(
            include_transpose,
            _transpose_energy,
            lambda _: (
                jnp.zeros((), dtype=block_direct.dtype),
                jnp.zeros((), dtype=block_exchange.dtype),
            ),
            operand=None,
        )
        block_direct = block_direct + transpose_direct
        block_exchange = block_exchange + transpose_exchange
        return block_direct, block_exchange

    if nvir:
        nblock = (nvir + block_nvir - 1) // block_nvir
        block_pairs = jnp.asarray(
            [
                (left_block, right_block)
                for left_block in range(nblock)
                for right_block in range(left_block, nblock)
            ],
            dtype=jnp.int32,
        )
        block_offsets = jnp.arange(block_nvir, dtype=jnp.int32)

        def _gather_virtual_block(array, block, axis):
            """Gather one padded fixed-size block without padding ``array``."""
            indices = block * block_nvir + block_offsets
            valid = indices < nvir
            safe_indices = jnp.minimum(indices, nvir - 1)
            gathered = jnp.take(array, safe_indices, axis=axis)
            mask_shape = [1] * gathered.ndim
            mask_shape[axis] = block_nvir
            return jnp.where(valid.reshape(mask_shape), gathered, 0), valid

        def _scan_block(carry, pair):
            left_block, right_block = pair
            lov_left, valid_left = _gather_virtual_block(
                lov, left_block, 2
            )
            lov_right, valid_right = _gather_virtual_block(
                lov, right_block, 2
            )
            eia_left, _ = _gather_virtual_block(eia, left_block, 1)
            eia_right, _ = _gather_virtual_block(eia, right_block, 1)
            block_direct, block_exchange = _block_energy(
                lov_left,
                lov_right,
                eia_left,
                eia_right,
                target_factor,
                partner_weight,
                left_block != right_block,
                valid_left,
                valid_right,
            )
            return (
                carry[0] + block_direct,
                carry[1] + block_exchange,
            ), None

        # ``scan`` threads one shared Lov/eia cotangent through the reverse
        # loop.  Rematerializing its body recomputes just one fixed-shape pair
        # at a time.  In contrast, an unrolled loop of checkpoints on sliced
        # inputs retains O(nblock) copies of Lov and transposes each slice into
        # a full-shape pad/add cotangent before it can be accumulated.
        (direct, exchange), _ = jax.lax.scan(
            jax.checkpoint(_scan_block),
            (direct, exchange),
            block_pairs,
        )

    opposite_spin = direct.real
    same_spin = (direct - exchange).real
    return FragmentPairEnergies(
        total=opposite_spin + same_spin,
        opposite_spin=opposite_spin,
        same_spin=same_spin,
    )


def _fragment_mp2_orientation_vjp(amplitudes, integrals, target_factor,
                                  partner_weight, direct_bar, exchange_bar):
    """Analytic adjoint of one ordered virtual-block MP2 contribution.

    ``amplitudes`` and ``integrals`` have axes ``(p,r,a,b)``.  This mirrors
    the small-target forward contraction, but explicitly propagates its two
    scalar cotangents so reverse mode does not need to synthesize and replay
    the einsum graph.
    """
    exchanged_integrals = integrals.transpose(1, 0, 3, 2)
    direct_projected = jnp.einsum(
        "xq,qsab->xsab", target_factor, integrals, optimize=True
    )
    exchange_projected = jnp.einsum(
        "xq,qsba->xsba", target_factor, exchanged_integrals,
        optimize=True,
    )
    target_amplitudes = jnp.einsum(
        "xp,prab->xrab", target_factor, amplitudes, optimize=True
    )
    weighted_amplitudes = jnp.einsum(
        "xrab,rs->xsab", target_amplitudes, partner_weight,
        optimize=True,
    )

    weighted_bar = (
        direct_bar * direct_projected
        + exchange_bar * exchange_projected.transpose(0, 1, 3, 2)
    )
    direct_projected_bar = direct_bar * weighted_amplitudes
    exchange_projected_bar = (
        exchange_bar * weighted_amplitudes.transpose(0, 1, 3, 2)
    )

    target_amplitudes_bar = jnp.einsum(
        "xsab,rs->xrab", weighted_bar, partner_weight, optimize=True
    )
    partner_weight_bar = jnp.einsum(
        "xrab,xsab->rs", target_amplitudes, weighted_bar, optimize=True
    )
    amplitudes_bar = jnp.einsum(
        "xp,xrab->prab", target_factor, target_amplitudes_bar,
        optimize=True,
    )

    target_factor_bar = jnp.einsum(
        "xrab,prab->xp", target_amplitudes_bar, amplitudes,
        optimize=True,
    )
    target_factor_bar += jnp.einsum(
        "xsab,qsab->xq", direct_projected_bar, integrals,
        optimize=True,
    )
    target_factor_bar += jnp.einsum(
        "xsba,qsba->xq", exchange_projected_bar,
        exchanged_integrals, optimize=True,
    )

    integrals_bar = jnp.einsum(
        "xq,xsab->qsab", target_factor, direct_projected_bar,
        optimize=True,
    )
    exchanged_bar = jnp.einsum(
        "xq,xsba->qsba", target_factor, exchange_projected_bar,
        optimize=True,
    )
    integrals_bar += exchanged_bar.transpose(1, 0, 3, 2)
    return (
        amplitudes_bar,
        integrals_bar,
        target_factor_bar,
        partner_weight_bar,
    )


def _fragment_mp2_block_vjp(lov_a, lov_b, eia_a, eia_b,
                            target_factor, partner_weight,
                            include_transpose, valid_a, valid_b,
                            direct_bar, exchange_bar):
    """Recompute one DF-MP2 block and return all of its input cotangents."""
    integrals = jnp.einsum(
        "Lia,Ljb->ijab", lov_a, lov_b, optimize=True
    )
    denominator = (
        eia_a[:, None, :, None] + eia_b[None, :, None, :]
    )
    denominator = jnp.where(
        valid_a[None, None, :, None]
        & valid_b[None, None, None, :],
        denominator,
        jnp.ones((), dtype=denominator.dtype),
    )
    amplitudes = integrals / denominator

    bars = _fragment_mp2_orientation_vjp(
        amplitudes,
        integrals,
        target_factor,
        partner_weight,
        direct_bar,
        exchange_bar,
    )
    amplitudes_bar, integrals_bar, target_bar, partner_bar = bars

    def _transpose_bars(_):
        transposed = _fragment_mp2_orientation_vjp(
            amplitudes.transpose(1, 0, 3, 2),
            integrals.transpose(1, 0, 3, 2),
            target_factor,
            partner_weight,
            direct_bar,
            exchange_bar,
        )
        return (
            transposed[0].transpose(1, 0, 3, 2),
            transposed[1].transpose(1, 0, 3, 2),
            transposed[2],
            transposed[3],
        )

    transpose_bars = jax.lax.cond(
        include_transpose,
        _transpose_bars,
        lambda _: (
            jnp.zeros_like(amplitudes_bar),
            jnp.zeros_like(integrals_bar),
            jnp.zeros_like(target_bar),
            jnp.zeros_like(partner_bar),
        ),
        operand=None,
    )
    amplitudes_bar += transpose_bars[0]
    integrals_bar += transpose_bars[1]
    target_bar += transpose_bars[2]
    partner_bar += transpose_bars[3]

    denominator_bar = -amplitudes_bar * integrals / denominator**2
    integrals_bar += amplitudes_bar / denominator
    eia_a_bar = jnp.sum(denominator_bar, axis=(1, 3))
    eia_b_bar = jnp.sum(denominator_bar, axis=(0, 2))
    lov_a_bar = jnp.einsum(
        "ijab,Ljb->Lia", integrals_bar, lov_b, optimize=True
    )
    lov_b_bar = jnp.einsum(
        "ijab,Lia->Ljb", integrals_bar, lov_a, optimize=True
    )
    return (
        lov_a_bar,
        lov_b_bar,
        eia_a_bar,
        eia_b_bar,
        target_bar,
        partner_bar,
    )


def _fragment_pair_energy_components_analytic_bwd(
        lov, e_occ, e_vir, target_factor, partner_weight,
        block_nvir, direct_bar, exchange_bar):
    """Blockwise handwritten VJP of the real two-sided DF-MP2 energy."""
    _, nocc, nvir = lov.shape
    if nvir == 0 or nocc == 0:
        return tuple(jnp.zeros_like(x) for x in (
            lov, e_occ, e_vir, target_factor, partner_weight
        ))

    eia = e_occ[:, None] - e_vir[None, :]
    nblock = (nvir + block_nvir - 1) // block_nvir
    block_pairs = jnp.asarray(
        [
            (left_block, right_block)
            for left_block in range(nblock)
            for right_block in range(left_block, nblock)
        ],
        dtype=jnp.int32,
    )
    block_offsets = jnp.arange(block_nvir, dtype=jnp.int32)

    def _gather(array, block, axis):
        indices = block * block_nvir + block_offsets
        valid = indices < nvir
        safe_indices = jnp.minimum(indices, nvir - 1)
        gathered = jnp.take(array, safe_indices, axis=axis)
        mask_shape = [1] * gathered.ndim
        mask_shape[axis] = block_nvir
        mask = valid.reshape(mask_shape)
        return jnp.where(mask, gathered, 0), valid, safe_indices, mask

    initial = (
        jnp.zeros_like(lov),
        jnp.zeros_like(eia),
        jnp.zeros_like(target_factor),
        jnp.zeros_like(partner_weight),
    )

    def _scan_block(carry, pair):
        left_block, right_block = pair
        lov_a, valid_a, indices_a, mask_a = _gather(
            lov, left_block, 2
        )
        lov_b, valid_b, indices_b, mask_b = _gather(
            lov, right_block, 2
        )
        eia_a, _, _, eia_mask_a = _gather(eia, left_block, 1)
        eia_b, _, _, eia_mask_b = _gather(eia, right_block, 1)
        # Invalid padded denominators are arbitrary constants.  Setting their
        # one-dimensional entries to zero keeps every valid denominator
        # unchanged; the block cotangents are masked before scattering.
        block_bars = _fragment_mp2_block_vjp(
            lov_a,
            lov_b,
            eia_a,
            eia_b,
            target_factor,
            partner_weight,
            left_block != right_block,
            valid_a,
            valid_b,
            direct_bar,
            exchange_bar,
        )
        lov_a_bar = jnp.where(mask_a, block_bars[0], 0)
        lov_b_bar = jnp.where(mask_b, block_bars[1], 0)
        eia_a_bar = jnp.where(eia_mask_a, block_bars[2], 0)
        eia_b_bar = jnp.where(eia_mask_b, block_bars[3], 0)

        lov_bar, eia_bar, target_bar, partner_bar = carry
        block_indices = jnp.concatenate((indices_a, indices_b))
        lov_bar = lov_bar.at[:, :, block_indices].add(
            jnp.concatenate((lov_a_bar, lov_b_bar), axis=2)
        )
        eia_bar = eia_bar.at[:, block_indices].add(
            jnp.concatenate((eia_a_bar, eia_b_bar), axis=1)
        )
        return (
            lov_bar,
            eia_bar,
            target_bar + block_bars[4],
            partner_bar + block_bars[5],
        ), None

    (lov_bar, eia_bar, target_bar, partner_bar), _ = jax.lax.scan(
        _scan_block, initial, block_pairs
    )
    return (
        lov_bar,
        jnp.sum(eia_bar, axis=1),
        -jnp.sum(eia_bar, axis=0),
        target_bar,
        partner_bar,
    )


@partial(jax.custom_vjp, nondiff_argnums=(5,))
def _fragment_pair_energy_components_custom(
        lov, e_occ, e_vir, target_factor, partner_weight, block_nvir):
    result = _fragment_pair_energy_from_lov_jax_ad(
        lov,
        e_occ,
        e_vir,
        target_factor,
        partner_weight,
        block_nvir=block_nvir,
    )
    direct = result.opposite_spin
    exchange = result.opposite_spin - result.same_spin
    return direct, exchange


def _fragment_pair_energy_components_custom_fwd(
        lov, e_occ, e_vir, target_factor, partner_weight, block_nvir):
    output = _fragment_pair_energy_components_custom(
        lov, e_occ, e_vir, target_factor, partner_weight, block_nvir
    )
    return output, (lov, e_occ, e_vir, target_factor, partner_weight)


def _fragment_pair_energy_components_custom_bwd(
        block_nvir, res, output_bars):
    return _fragment_pair_energy_components_analytic_bwd(
        *res, block_nvir, *output_bars
    )


_fragment_pair_energy_components_custom.defvjp(
    _fragment_pair_energy_components_custom_fwd,
    _fragment_pair_energy_components_custom_bwd,
)


def _fragment_mp2_analytic_vjp_enabled():
    value = os.environ.get('PYSCFAD_DLNO_MP2_ANALYTIC_VJP', '1')
    return value.strip().lower() not in ('0', 'false', 'no', 'off')


def fragment_pair_energy_from_lov_jax(
    lov,
    e_occ,
    e_vir,
    target_factor,
    partner_weight,
    *,
    block_nvir=None,
    max_memory_mb=256.0,
):
    """Differentiable blocked DF-MP2 energy with an analytic real VJP.

    The primal and complex-valued fallback are implemented by
    :func:`_fragment_pair_energy_from_lov_jax_ad`.  For real inputs the
    custom rule recomputes and eliminates one upper-triangular virtual block
    at a time while accumulating cotangents for ``lov``, orbital energies,
    the target factor, and the partner weight.
    """
    if getattr(lov, 'ndim', None) != 3:
        return _fragment_pair_energy_from_lov_jax_ad(
            lov,
            e_occ,
            e_vir,
            target_factor,
            partner_weight,
            block_nvir=block_nvir,
            max_memory_mb=max_memory_mb,
        )
    if block_nvir is None:
        block_nvir = _virtual_block_size(
            lov, 1, 1, max_memory_mb
        )
    elif not isinstance(block_nvir, (int, np.integer)) or block_nvir <= 0:
        raise ValueError("block_nvir must be a positive integer")
    nvir = lov.shape[2] if getattr(lov, 'ndim', None) == 3 else 0
    block_nvir = min(int(block_nvir), nvir) if nvir else 1

    operands = (lov, e_occ, e_vir, target_factor, partner_weight)
    use_analytic = (
        _fragment_mp2_analytic_vjp_enabled()
        and not any(jnp.issubdtype(jnp.asarray(x).dtype, jnp.complexfloating)
                    for x in operands)
    )
    if not use_analytic:
        return _fragment_pair_energy_from_lov_jax_ad(
            *operands, block_nvir=block_nvir
        )

    direct, exchange = _fragment_pair_energy_components_custom(
        *operands, block_nvir
    )
    opposite_spin = direct.real
    same_spin = (direct - exchange).real
    return FragmentPairEnergies(
        total=opposite_spin + same_spin,
        opposite_spin=opposite_spin,
        same_spin=same_spin,
    )


def _masked_pair_energies(energies, mask):
    return FragmentPairEnergies(
        total=np.where(mask, energies.total, np.zeros_like(energies.total)),
        opposite_spin=np.where(
            mask,
            energies.opposite_spin,
            np.zeros_like(energies.opposite_spin),
        ),
        same_spin=np.where(
            mask,
            energies.same_spin,
            np.zeros_like(energies.same_spin),
        ),
    )


def partition_fragment_pair_energies(energies, strong_mask):
    """Split exact directed fragment-pair arrays with a Boolean mask.

    The returned arrays satisfy ``strong + weak == energies`` elementwise.
    For a square fragment-pair matrix, a physical pair classification should
    normally use a symmetric mask and mark every diagonal entry as strong.
    No factor of two is introduced here: off-diagonal entries are directed
    contributions, and summing the full matrix already includes both
    directions.
    """

    if not isinstance(energies, FragmentPairEnergies):
        raise TypeError("energies must be a FragmentPairEnergies instance")
    mask = np.asarray(strong_mask, dtype=bool)
    if mask.shape != np.shape(energies.total):
        raise ValueError("strong_mask must have the same shape as the energies")
    strong = _masked_pair_energies(energies, mask)
    weak = _masked_pair_energies(energies, ~mask)
    return FragmentPairPartition(strong=strong, weak=weak)
