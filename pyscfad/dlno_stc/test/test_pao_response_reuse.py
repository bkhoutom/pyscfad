"""Shared PAO response must preserve target frames and the full chain rule."""
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
from typing import NamedTuple
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import _pairs, domain
from pyscfad.dlno_stc._selection import FixedPAOSubspaceSelection


class Molecule(NamedTuple):
    coordinates: object

    def atom_coords(self):
        return self.coordinates


class MeanField(NamedTuple):
    mol: object


def _fixture(monkeypatch, changed=None):
    rng = np.random.default_rng(12)
    nao = 8
    root = np.eye(nao) + .06 * rng.normal(size=(nao, nao))
    overlap = root.T @ root
    q, _ = np.linalg.qr(rng.normal(size=(nao, nao)))
    coeff = np.linalg.solve(np.linalg.cholesky(overlap).T, q)
    energies = np.array([-1.7, -1.3, -.9, .6, .9, 1.2, 1.5, 1.8])
    fock = overlap @ coeff @ np.diag(energies) @ coeff.T @ overlap
    pao = coeff[:, 3:] @ rng.normal(size=(5, 4))
    common = domain.DomainData(*map(jnp.asarray, (
        overlap, fock, coeff[:, :3], coeff[:, 3:], energies[:3], energies[3:],
        coeff[:, :3], pao)),
        tuple(domain.TargetOccupiedData(jnp.asarray(coeff[:, i:i+1])) for i in range(3)))
    selection = FixedPAOSubspaceSelection(*map(np.asarray, (
        [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2], [0, 1, 2], [1, 2], [0, 1])))
    fragments = [SimpleNamespace(
        fragment_index=i, weak_virtual=selection,
        primary_ao_indices=np.arange(7), primary_atoms=np.array([0, 1]),
        weak_occ_norm_keep=np.array([0]), weak_occ_span_metric_keep=np.array([0]))
        for i in range(3)]
    if changed is not None:
        name, labels = changed
        if name == 'representation':
            fragments[2].primary_ao_indices = np.asarray(labels)
        else:
            fragments[2].weak_virtual = replace(selection, **{name: np.asarray(labels)})
    static = SimpleNamespace(
        fragments=tuple(fragments), strong_mask=np.eye(3, dtype=bool),
        frag_atmlist=tuple(np.array([i]) for i in range(3)),
        thresholds=SimpleNamespace(metric_rank=1e-10, pair_energy=1e30,
                                   near_pair_distance=0., multipole_order=2))
    mf = MeanField(Molecule(jnp.array([[0., .1, 0.], [5., .2, 0.], [10., 0., .3]])))
    raw = rng.normal(size=(3, 7, 7))
    raw = jnp.asarray((raw + raw.transpose(0, 2, 1)) / 2)

    def moments(mol, order):
        shift = .02 * jnp.sum(mol.coordinates**2)
        return (jnp.eye(7) * (1 + .01 * shift),
                raw * (1 + shift), None, None)

    # Replace only AO integral evaluation: all PAO, screen, endpoint and pair
    # transformations (including their VJPs) remain the production routines.
    monkeypatch.setattr(_pairs, '_fake_mol_with_traced_centers', lambda mol, atoms: mol)
    monkeypatch.setattr(_pairs, '_origin_zero_moments', moments)
    return mf, common, static


def _assert_bars(actual, expected):
    for got, want in zip(jax.tree_util.tree_leaves(actual),
                         jax.tree_util.tree_leaves(expected)):
        np.testing.assert_allclose(got, want, rtol=5e-10, atol=2e-11)


@pytest.mark.parametrize('grouped_ao', [False, True])
def test_identical_paos_share_eigensolvers_and_preserve_full_common_gradient(monkeypatch, grouped_ao):
    mf, common, static = _fixture(monkeypatch)
    expected, expected_bars = _pairs.weak_pair_energy_and_grad(mf, common, static)
    eigensolves = Counter()
    eigh = domain.scipy.linalg.eigh

    def counted(matrix, *args, **kwargs):
        eigensolves[matrix.shape] += 1
        return eigh(matrix, *args, **kwargs)

    monkeypatch.setattr(domain.scipy.linalg, 'eigh', counted)
    prepared = {}
    _pairs.screen_pairs(mf, common, static, prepared_endpoints=prepared)
    # Three endpoints need one canonical PAO solve and one support solve,
    # but retain three independent final semicanonicalizations.
    assert eigensolves == Counter({(4, 4): 1, (3, 3): 1, (2, 2): 3})
    actual, bars = _pairs.weak_pair_energy_and_grad(
        mf, common, static, prepared_endpoints=prepared,
        _grouped_ao_response=grouped_ao)
    np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=1e-12)
    _assert_bars(bars, expected_bars)
    assert prepared == {}
    assert eigensolves == Counter({(4, 4): 1, (3, 3): 1, (2, 2): 3})
    for field in ('pao_coeff', 's1e', 'fock'):
        assert np.linalg.norm(getattr(bars[1], field)) > 1e-9
    assert all(np.linalg.norm(target.iao_coeff) > 1e-9
               for target in bars[1].fragment_occupied_data)


@pytest.mark.parametrize('changed', [
    ('parent_columns', [3, 2, 1, 0]),
    ('support_ao_indices', [1, 2, 3, 4]),
    ('canonical_keep', [1, 2, 3]),
    ('overlap_keep', [2, 1, 0]),
    ('completeness_keep', [0, 2]),
    ('representation', [6, 5, 4, 3, 2, 1, 0]),
])
def test_distinct_saved_pao_labels_and_supports_never_share(monkeypatch, changed):
    mf, common, static = _fixture(monkeypatch, changed)
    expected, expected_bars = _pairs.weak_pair_energy_and_grad(mf, common, static)
    calls = Counter()
    eigh = domain.scipy.linalg.eigh

    def counted(matrix, *args, **kwargs):
        calls[matrix.shape] += 1
        return eigh(matrix, *args, **kwargs)

    monkeypatch.setattr(domain.scipy.linalg, 'eigh', counted)
    prepared = {}
    _pairs.screen_pairs(mf, common, static, prepared_endpoints=prepared)
    assert calls == Counter({(4, 4): 2, (3, 3): 2, (2, 2): 3})
    value, bars = _pairs.weak_pair_energy_and_grad(
        mf, common, static, prepared_endpoints=prepared, _grouped_ao_response=True)
    np.testing.assert_allclose(value, expected, rtol=5e-12, atol=1e-12)
    _assert_bars(bars, expected_bars)


@pytest.mark.parametrize('pairs', [(), ((0, 1),), ((0, 1), (0, 2), (1, 2))])
def test_shared_reverse_runs_once_only_for_used_group_and_releases_tapes(monkeypatch, pairs):
    mf, common, static = _fixture(monkeypatch)
    preprojection = getattr(domain, '_rebuild_selected_pao_preprojection', None)
    assert callable(preprojection), 'PAO preprojection needs a separately reusable response'
    calls = Counter()
    tapes = []
    vjp = jax.vjp

    def counted(fun, *args, **kwargs):
        value, pullback = vjp(fun, *args, **kwargs)
        if getattr(fun, 'func', None) is preprojection:
            calls['forward'] += 1

            def tracked(bar):
                calls['reverse'] += 1
                return pullback(bar)

            tapes.append(weakref.ref(tracked))
            return value, tracked
        return value, pullback

    monkeypatch.setattr(jax, 'vjp', counted)
    prepared = {}
    _pairs.screen_pairs(mf, common, static, prepared_endpoints=prepared)
    assert calls == Counter(forward=1)
    expected, expected_bars = _pairs.weak_pair_energy_and_grad(mf, common, static, pairs=pairs)
    value, bars = _pairs.weak_pair_energy_and_grad(
        mf, common, static, pairs=pairs, prepared_endpoints=prepared,
        _grouped_ao_response=True)
    assert calls == Counter(forward=1, reverse=int(bool(pairs)))
    np.testing.assert_allclose(value, expected, rtol=5e-12, atol=1e-12)
    _assert_bars(bars, expected_bars)
    assert prepared == {}
    assert all(tape() is None for tape in tapes)


@pytest.mark.parametrize('grouped_ao', [False, True])
def test_shared_tape_is_released_before_replaying_the_next_pao_group(monkeypatch, grouped_ao):
    mf, common, static = _fixture(monkeypatch, ('canonical_keep', [1, 2, 3]))
    preprojection = domain._rebuild_selected_pao_preprojection
    tapes = []
    vjp = jax.vjp

    def track(fun, *args, **kwargs):
        value, pullback = vjp(fun, *args, **kwargs)
        if getattr(fun, 'func', None) is preprojection:
            tapes.append(weakref.ref(pullback))
        return value, pullback

    monkeypatch.setattr(jax, 'vjp', track)
    prepared = {}
    _pairs.screen_pairs(mf, common, static, prepared_endpoints=prepared)
    assert len(tapes) == 2
    moments = _pairs._weak_endpoint_from_screen

    def check_release(m, screen, static, fragment_index, **kwargs):
        if fragment_index == 2:
            assert tapes[0]() is None, 'finished PAO group retained its response tape'
        return moments(m, screen, static, fragment_index, **kwargs)

    monkeypatch.setattr(_pairs, '_weak_endpoint_from_screen', check_release)
    _pairs.weak_pair_energy_and_grad(
        mf, common, static, prepared_endpoints=prepared,
        _grouped_ao_response=grouped_ao)
    assert all(tape() is None for tape in tapes)


@pytest.mark.parametrize('grouped_ao', [False, True])
def test_target_metric_ranks_remain_independent_of_shared_preprojection(monkeypatch, grouped_ao):
    mf, common, static = _fixture(monkeypatch)
    # The smallest post-projection metric eigenvalues are 0.7304, 0.7579,
    # and 0.7600: a common threshold leaves ranks one, two, and two.
    static.thresholds.metric_rank = .745
    fragment = static.fragments[0]
    fragment.weak_virtual = replace(fragment.weak_virtual, metric_keep=np.array([1]))
    expected, expected_bars = _pairs.weak_pair_energy_and_grad(mf, common, static)
    calls = Counter()
    eigh = domain.scipy.linalg.eigh

    def counted(matrix, *args, **kwargs):
        calls[matrix.shape] += 1
        return eigh(matrix, *args, **kwargs)

    monkeypatch.setattr(domain.scipy.linalg, 'eigh', counted)
    prepared = {}
    _pairs.screen_pairs(mf, common, static, prepared_endpoints=prepared)
    assert [item[1][0].virtual_coeff.shape[1] for item in prepared.values()] == [1, 2, 2]
    assert calls == Counter({(4, 4): 1, (3, 3): 1, (2, 2): 3})
    value, bars = _pairs.weak_pair_energy_and_grad(
        mf, common, static, prepared_endpoints=prepared,
        _grouped_ao_response=grouped_ao)
    np.testing.assert_allclose(value, expected, rtol=5e-12, atol=1e-12)
    _assert_bars(bars, expected_bars)


@pytest.mark.parametrize('grouped_ao', [False, True])
def test_missing_preparation_uses_legacy_response_beside_shared_group(monkeypatch, grouped_ao):
    mf, common, static = _fixture(monkeypatch)
    expected, expected_bars = _pairs.weak_pair_energy_and_grad(mf, common, static)
    prepared = {}
    _pairs.screen_pairs(mf, common, static, prepared_endpoints=prepared)
    del prepared[1]
    value, bars = _pairs.weak_pair_energy_and_grad(
        mf, common, static, prepared_endpoints=prepared,
        _grouped_ao_response=grouped_ao)
    np.testing.assert_allclose(value, expected, rtol=5e-12, atol=1e-12)
    _assert_bars(bars, expected_bars)
    assert prepared == {}
