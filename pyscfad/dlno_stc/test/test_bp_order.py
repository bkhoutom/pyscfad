"""BP growth reaches completeness using the requested physical ordering."""

import numpy as np
import pytest

from pyscfad.dlno_stc._selection import get_bp_domain


class FourAtoms:
    natm = nao = 4

    def atom_coords(self):
        return np.array([[0., 0., 0.], [1., 0., 0.],
                         [10., 0., 0.], [20., 0., 0.]])

    def aoslice_by_atom(self):
        return np.array([[i, i + 1, i, i + 1] for i in range(4)])


def test_population_growth_prefers_larger_contribution_over_nearby_atom():
    mol = FourAtoms()
    orbital = np.sqrt([.60, .05, .30, .05])
    population, = get_bp_domain(mol, orbital, np.eye(4), bp_thr=.85,
                               q_thr=.4, atom_order="population")
    distance, = get_bp_domain(mol, orbital, np.eye(4), bp_thr=.85, q_thr=.4)
    np.testing.assert_array_equal(population, [0, 2])
    np.testing.assert_array_equal(distance, [0, 1, 2])
    assert sum(orbital[population] ** 2) >= .85


@pytest.mark.parametrize("atom_order", ["population", "distance"])
def test_growth_includes_largest_atom_when_initial_seed_is_empty(atom_order):
    atoms, = get_bp_domain(FourAtoms(), np.sqrt([.60, .05, .30, .05]),
                           np.eye(4), bp_thr=.85, q_thr=1.,
                           atom_order=atom_order)
    assert 0 in atoms
    assert np.sum(np.array([.60, .05, .30, .05])[atoms]) >= .85


def test_growth_rejects_unknown_order():
    with pytest.raises(ValueError, match="atom_order"):
        get_bp_domain(FourAtoms(), np.eye(4), np.eye(4), atom_order="unknown")
