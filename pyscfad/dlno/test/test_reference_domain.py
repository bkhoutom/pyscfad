"""Concrete occupied ED parity for the two intentionally distinct reference frames."""
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg

from pyscfad.dlno import domain


def _problem(mode, metric_rank):
    rng = np.random.default_rng(1702)
    overlap = np.diag(np.linspace(0.8, 1.4, 6))
    rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    orbitals = rotation / np.sqrt(np.diag(overlap))[:, None]
    occupied = orbitals[:, :3]
    fock = overlap @ orbitals @ np.diag([-1.2, -0.8, -0.3, 0.2, 0.6, 1.1]) @ orbitals.T @ overlap
    if mode == 'boys':
        columns = [occupied[:, i:i+1] for i in range(3)]
    else:
        target_rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        targets = orbitals @ target_rotation
        columns = [targets[:, 2*i:2*i+2] for i in range(3)]
    blocks = tuple(SimpleNamespace(iao_coeff=c, iao_occ_overlap=c.T @ overlap @ occupied)
                   for c in columns)
    topology = SimpleNamespace(
        lo_type=mode, occupied_coeff=occupied, fragment_occupied_data=blocks,
        strong_fragments=(np.array([0, 1]),),
        thresholds=SimpleNamespace(occupied_weight=1e-4, metric_rank=metric_rank),
    )
    rows = np.arange(5)
    return topology, overlap[rows], overlap[np.ix_(rows, rows)], fock[np.ix_(rows, rows)]


def _old_reference(topology, s21, s22, fock22, selection_frame):
    partners = topology.strong_fragments[0]
    blocks = topology.fragment_occupied_data
    if topology.lo_type == 'boys':
        candidate = np.concatenate([blocks[i].iao_coeff for i in partners], axis=1)
        union_keep = np.zeros(0, dtype=np.int32)
    else:
        thin = np.vstack([blocks[i].iao_occ_overlap for i in partners])
        if selection_frame:
            weight = thin.T @ thin
            eigenvalues, eigenvectors = scipy.linalg.eigh((weight + weight.T) / 2, check_finite=False)
            union_keep = np.flatnonzero(eigenvalues > topology.thresholds.occupied_weight)
            candidate = topology.occupied_coeff @ eigenvectors[:, union_keep]
        else:
            _, singular, vh = scipy.linalg.svd(thin, full_matrices=False)
            union_keep = np.flatnonzero(singular**2 > topology.thresholds.occupied_weight)
            candidate = topology.occupied_coeff @ vh.T[:, union_keep]
    projected = np.linalg.solve(s22, s21 @ candidate)
    metric = projected.T @ s22 @ projected
    eigenvalues, eigenvectors = scipy.linalg.eigh((metric + metric.T) / 2)
    metric_keep = np.flatnonzero(eigenvalues > topology.thresholds.metric_rank)
    normalized = projected @ (eigenvectors[:, metric_keep] / np.sqrt(eigenvalues[metric_keep]))
    local_fock = normalized.T @ fock22 @ normalized
    energies, rotation = scipy.linalg.eigh((local_fock + local_fock.T) / 2)
    if not selection_frame or normalized.shape[1] > 1:
        normalized = normalized @ rotation
    return energies, normalized, union_keep, metric_keep


@pytest.mark.parametrize('mode', ['iao', 'boys'])
@pytest.mark.parametrize('selection_frame', [False, True])
@pytest.mark.parametrize('metric_rank', [1e-10, 0.98])
def test_shared_reference_occupied_space_preserves_frames_and_rank(mode, selection_frame, metric_rank):
    builder = getattr(domain, '_reference_occupied_space', None)
    assert builder is not None, 'eager and selection paths need the shared concrete occupied builder'
    topology, s21, s22, fock22 = _problem(mode, metric_rank)
    expected = _old_reference(topology, s21, s22, fock22, selection_frame)
    actual = builder(topology, 0, s21, s22, fock22, selection_frame=selection_frame)
    for index in (0, 1):
        np.testing.assert_allclose(actual[index], expected[index], atol=2e-14, rtol=2e-14)
    for index in (2, 3):
        np.testing.assert_array_equal(actual[index], expected[index])
    np.testing.assert_allclose(actual[1].T @ s22 @ actual[1], np.eye(len(actual[0])), atol=2e-14)
