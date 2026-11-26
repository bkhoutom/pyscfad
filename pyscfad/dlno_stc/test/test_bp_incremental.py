"""Incremental BP completeness agrees with independent full overlap solves."""
import numpy as np
import pytest

from pyscfad.dlno_stc import _selection as selection


@pytest.mark.parametrize('seed', [7, 18, 29])
def test_block_growth_matches_full_solves(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(12, 12))
    overlap = a @ a.T + np.eye(12) * .1
    coeff = rng.normal(size=12)
    coeff /= np.sqrt(coeff @ overlap @ coeff)
    rhs = overlap @ coeff
    state = selection._BPCompleteness(overlap, rhs)
    factored_sizes = []
    cholesky = selection.scipy_linalg.cholesky
    def counted(block, **kwargs):
        factored_sizes.append(len(block))
        return cholesky(block, **kwargs)
    monkeypatch.setattr(selection.scipy_linalg, 'cholesky', counted)
    chosen = []
    previous = 0.
    for indices in ([4, 5, 6], [0, 1], [8, 9, 10, 11], [2, 3, 7]):
        chosen.extend(indices)
        value = state.append(indices)
        block = overlap[np.ix_(chosen, chosen)]
        expected = rhs[chosen] @ np.linalg.solve(block, rhs[chosen])
        np.testing.assert_allclose(value, expected, atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(state.factor @ state.factor.T, block, atol=2e-12)
        assert value >= previous - 2e-12
        previous = value
    np.testing.assert_allclose(value, 1., atol=2e-12)
    assert factored_sizes == [3, 2, 4, 3]


def test_indefinite_overlap_uses_general_solve():
    overlap = np.array([[1., 2., .1], [2., 1., .2], [.1, .2, 1.]])
    rhs = overlap @ np.array([.2, .3, .4])
    state = selection._BPCompleteness(overlap, rhs)
    state.append([0])
    for indices in ([1], [2]):
        value = state.append(indices)
        assert state.general_solve
        np.testing.assert_allclose(value, state.direct())


def test_singular_overlap_still_raises():
    state = selection._BPCompleteness(np.ones((2, 2)), np.ones(2))
    state.append([0])
    with pytest.raises(np.linalg.LinAlgError):
        state.append([1])


@pytest.mark.parametrize('order', ['population', 'distance'])
@pytest.mark.parametrize('seed_cutoff', [0., .2, 1.])
def test_selected_supports_match_full_solve(order, seed_cutoff, monkeypatch):
    from .test_bp_order import FourAtoms
    rng = np.random.default_rng(82)
    a = rng.normal(size=(4, 4))
    overlap = a @ a.T + np.eye(4) * .2
    orbitals = rng.normal(size=(4, 10))
    orbitals /= np.sqrt(np.einsum('ui,uv,vi->i', orbitals, overlap, orbitals))
    kwargs = dict(bp_thr=.98, q_thr=seed_cutoff, atom_order=order)
    actual = selection.get_bp_domain(FourAtoms(), orbitals, overlap, **kwargs)
    class FullSolve(selection._BPCompleteness):
        def append(self, indices):
            self.indices = np.concatenate((self.indices, np.asarray(indices, dtype=int)))
            return self.direct()
    monkeypatch.setattr(selection, '_BPCompleteness', FullSolve)
    expected = selection.get_bp_domain(FourAtoms(), orbitals, overlap, **kwargs)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)


def test_selection_timers_are_disjoint(boys_water_dimer):
    from pyscfad.tools import resource_profile
    mf, _, reference = boys_water_dimer
    events = []
    with resource_profile.capture_events(events.append):
        actual = selection.build_domain_selections(mf, frozen=reference.frozen)
    phases = ['stc.domain.selection.boys', 'stc.domain.selection.atom_support',
              'stc.domain.selection.primary_spaces']
    selected = [event for event in events if event['phase'] in phases]
    assert [event['phase'] for event in selected] == phases
    assert all(event['wall_s'] >= 0 for event in selected)
    for left, right in zip(selected, selected[1:]):
        assert left['end_wall_s'] <= right['start_wall_s']
    np.testing.assert_array_equal(actual.strong_mask, reference.strong_mask)


from .test_boys_prepare import boys_water_dimer
