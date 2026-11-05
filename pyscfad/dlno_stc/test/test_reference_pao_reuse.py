"""Concrete PAO selection shares only target-independent preprojection."""
from types import SimpleNamespace

import numpy as np
import pytest

from pyscfad.dlno_stc import _selection as selection


def _inputs():
    rng = np.random.default_rng(912)
    mol = SimpleNamespace(nao=8, aoslice_by_atom=lambda: np.array(
        [[0, 1, 0, 4], [1, 2, 4, 8]]))
    a = rng.normal(size=(8, 8))
    overlap = a.T @ a + np.eye(8)
    pao = rng.normal(size=(8, 8))
    targets = rng.normal(size=(8, 2))
    targets /= np.sqrt(np.einsum('ui,uv,vi->i', targets, overlap, targets))
    args = (mol, pao, np.arange(8), overlap)
    kwargs = dict(parent_atoms=[0], support_atoms=[0], representation_atoms=[0, 1],
                  completeness_threshold=.01, overlap_threshold=.01,
                  metric_threshold=1e-10)
    return args, kwargs, targets


def _assert_same(got, want):
    for name in got[0].__dataclass_fields__:
        np.testing.assert_array_equal(getattr(got[0], name), getattr(want[0], name))
    np.testing.assert_allclose(got[1], want[1], rtol=1e-13, atol=1e-13)


def test_reuses_preprojection_but_keeps_each_occupied_projection(monkeypatch):
    args, kwargs, targets = _inputs()
    expected = [selection._reference_pao_overlap_selection(
        *args, **kwargs, occupied_for_projection=targets[:, i:i+1]) for i in range(2)]
    original = selection._eigh_keep_numpy
    calls = []
    def counted(*args):
        calls.append(args[0].shape)
        return original(*args)
    monkeypatch.setattr(selection, '_eigh_keep_numpy', counted)
    cache = {}
    actual = [selection._reference_pao_overlap_selection(
        *args, **kwargs, occupied_for_projection=targets[:, i:i+1],
        preprojection_cache=cache) for i in range(2)]
    assert len(calls) == 4  # Two shared eigensolves, one metric solve per target.
    assert len(cache) == 1
    for got, want in zip(actual, expected):
        _assert_same(got, want)
    for i, (_, coeff) in enumerate(actual):
        np.testing.assert_allclose(targets[:, i:i+1].T @ args[3] @ coeff, 0, atol=1e-13)
    assert not np.allclose(actual[0][1] @ actual[0][1].T,
                           actual[1][1] @ actual[1][1].T)


@pytest.mark.parametrize('change', [
    dict(parent_atoms=[1]), dict(support_atoms=[1]),
    dict(representation_atoms=[1, 0]), dict(completeness_threshold=.99),
    dict(overlap_threshold=.8),
])
def test_different_support_or_selection_cutoff_cannot_share(change):
    args, kwargs, targets = _inputs()
    cache = {}
    selection._reference_pao_overlap_selection(*args, **kwargs,
        occupied_for_projection=targets[:, :1], preprojection_cache=cache)
    kwargs.update(change)
    # Reorder the target to the representation's AO order, if changed.
    target = targets[[4, 5, 6, 7, 0, 1, 2, 3], :1] if 'representation_atoms' in change else targets[:, :1]
    got = selection._reference_pao_overlap_selection(*args, **kwargs,
        occupied_for_projection=target, preprojection_cache=cache)
    want = selection._reference_pao_overlap_selection(*args, **kwargs,
        occupied_for_projection=target)
    _assert_same(got, want)
    assert len(cache) == 2


def test_empty_space_is_shared_without_metric_eigensolve():
    args, kwargs, targets = _inputs()
    kwargs['completeness_threshold'] = 2.
    cache = {}
    for i in range(2):
        got = selection._reference_pao_overlap_selection(*args, **kwargs,
            occupied_for_projection=targets[:, i:i+1], preprojection_cache=cache)
        assert got[1].shape == (8, 0)
        assert got[0].metric_keep.size == 0
    assert len(cache) == 1
