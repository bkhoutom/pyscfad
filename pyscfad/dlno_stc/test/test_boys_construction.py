"""Ordered Boys construction: concrete labels and continuous local frames."""

from dataclasses import replace
from copy import copy

import jax
import numpy as np
import pytest

from pyscfad import dft, gto, scf
from pyscfad import dlno_stc
from pyscfad.dlno_stc import domain
from pyscfad.dlno_stc._selection import _validate_boys_target_groups


@pytest.fixture(scope="module")
def reference():
    mol = gto.Mole(atom="O 0 0 0; H 0 .75 .58; H 0 -.75 .58", basis="sto-3g", verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    return mf


def test_public_selection_preserves_permuted_labels_and_retained_virtuals(reference):
    assert hasattr(dlno_stc, "build_domain_selections")
    static = dlno_stc.build_domain_selections(
        reference, frozen=1, frag_lolist=[[3], [1], [0], [2]],
        force_full_domains=True,
    )
    assert type(static).__module__ == "pyscfad.dlno_stc._selection"
    common = domain.rebuild_domain_data(reference, static)
    for index, fragment in enumerate(static.fragments):
        frame = domain.build_boys_ed_frame(common, static, index)
        assert frame.target_index == index
        assert fragment.target_index == index
        assert frame.virtual_coeff.shape[1] == 2
        ao = fragment.extended_ao_indices
        s22 = np.asarray(common.s1e)[np.ix_(ao, ao)]
        candidates = np.asarray(common.iao_coeff)[:, [3, 1, 0, 2]]
        projected = np.linalg.solve(s22, np.asarray(common.s1e)[ao] @ candidates)
        lower = np.linalg.cholesky(projected.T @ s22 @ projected)
        expected = np.linalg.solve(lower, projected.T).T
        pivot = np.argmax(abs(expected), axis=0)
        expected *= np.where(expected[pivot, np.arange(4)] < 0, -1., 1.)
        np.testing.assert_allclose(frame.occupied_coeff, expected, atol=1e-10)
        coeff = np.concatenate((frame.occupied_coeff, frame.virtual_coeff), axis=1)
        np.testing.assert_allclose(coeff.T @ s22 @ coeff, np.eye(6), atol=1e-9)


def test_selection_rejects_invalid_target_partition_and_rank_loss(reference):
    assert hasattr(dlno_stc, "build_domain_selections")
    with pytest.raises(ValueError, match="singleton"):
        dlno_stc.build_domain_selections(reference, frozen=1, frag_lolist=[[0, 1], [2], [3]])
    with pytest.raises(ValueError, match="orthonormal|span"):
        dlno_stc.build_domain_selections(reference, frozen=1, target_coeff=np.ones((7, 4)))
    static = dlno_stc.build_domain_selections(reference, frozen=1, force_full_domains=True)
    common = domain.rebuild_domain_data(reference, static)
    fragment = replace(static.fragments[2], strong_occ_metric_keep=np.arange(3))
    broken = replace(static, fragments=static.fragments[:2] + (fragment,) + static.fragments[3:])
    with pytest.raises(RuntimeError, match="domain 2|fragment 2"):
        domain.build_boys_ed_frame(common, broken, 2)


def test_standalone_system_frame_replays_frozen_virtual_and_continuous_response(reference):
    assert hasattr(domain, "build_system_local_frame")
    occupied, virtual = np.arange(1, 5), np.array([5])
    frame = domain.build_system_local_frame(reference, occupied, virtual)
    expected = domain.build_system_local_frame(reference, occupied, virtual)
    np.testing.assert_allclose(frame.occupied_coeff, expected.occupied_coeff, atol=2e-7)
    np.testing.assert_allclose(frame.virtual_coeff, expected.virtual_coeff, atol=1e-9)
    with pytest.raises(ValueError, match="concrete.*before tracing"):
        jax.vjp(lambda mf: domain.build_system_local_frame(mf, occupied, virtual).virtual_coeff, reference)
    seed = jax.numpy.asarray(np.random.default_rng(4).normal(size=frame.virtual_coeff.shape))
    _, pullback = jax.vjp(lambda mf: domain.build_system_local_frame(
        mf, occupied, virtual, boys_reference=frame.boys_reference,
    ).virtual_coeff, reference)
    bar, = pullback(seed)
    assert np.linalg.norm(np.asarray(bar.mo_coeff)) > 1e-6


def test_selection_accepts_empty_frozen_and_excludes_explicit_virtual(reference):
    empty = dlno_stc.build_domain_selections(reference, frozen=[], force_full_domains=True)
    np.testing.assert_array_equal(empty.active_occ_indices, np.arange(5))
    np.testing.assert_array_equal(empty.active_vir_indices, [5, 6])
    static = dlno_stc.build_domain_selections(reference, frozen=[0, 6], force_full_domains=True)
    np.testing.assert_array_equal(static.active_occ_indices, [1, 2, 3, 4])
    np.testing.assert_array_equal(static.active_vir_indices, [5])
    np.testing.assert_array_equal(static.pao_projected_out_indices, [0, 1, 2, 3, 4, 6])
    common = domain.rebuild_domain_data(reference, static)
    np.testing.assert_allclose(np.asarray(common.pao_coeff).T @ np.asarray(common.s1e)
                               @ np.asarray(reference.mo_coeff[:, [6]]), 0., atol=1e-9)


def test_selection_rejects_complex_occupations_and_targets_outside_active_span(reference):
    current = copy(reference)
    current.mo_occ = np.asarray(current.mo_occ, dtype=complex)
    with pytest.raises(NotImplementedError, match="real"):
        dlno_stc.build_domain_selections(current, frozen=1)
    outside = np.asarray(reference.mo_coeff[:, [1, 2, 3, 5]])
    with pytest.raises(ValueError, match="span"):
        dlno_stc.build_domain_selections(reference, frozen=1, target_coeff=outside)


def test_selection_rejects_converged_dft(reference):
    # density_fit changes the shared _DFHF bases; restore this pre-existing
    # library side effect so the RHF module fixture retains its real type.
    df_type, original_bases = type(reference), type(reference).__bases__
    try:
        current = dft.RKS(reference.mol).density_fit()
        current.conv_tol = 1e-10
        current.kernel()
        assert current.converged
        with pytest.raises(NotImplementedError, match="RHF|Hartree"):
            dlno_stc.build_domain_selections(current, frozen=1)
    finally:
        df_type.__bases__ = original_bases


def test_selection_rejects_complex_fock(reference):
    current = copy(reference)
    fock = np.asarray(reference.get_fock(), dtype=complex)
    current.get_fock = lambda: fock
    with pytest.raises(NotImplementedError, match="real"):
        dlno_stc.build_domain_selections(current, frozen=1)


@pytest.mark.parametrize("labels", [
    [[2**32], [2**32 + 1]],
    np.array([[2**32], [2**32 + 1]], dtype=np.uint64),
    [[-2**32], [-2**32 + 1]],
])
def test_boys_labels_are_range_checked_before_int32_conversion(labels):
    with pytest.raises(ValueError, match="singleton"):
        _validate_boys_target_groups(labels, 2)
