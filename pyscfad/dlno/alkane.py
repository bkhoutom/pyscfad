"""Deterministic, valence-checked linear-alkane geometries for examples."""

from __future__ import annotations

import math

import numpy as np


CC_BOND_ANGSTROM = 1.54
CH_BOND_ANGSTROM = 1.09
CCC_ANGLE_DEG = 112.0


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-14:
        raise ValueError("cannot normalize a zero-length vector")
    return vector / norm


def _perpendicular_frame(axis):
    axis = _unit(axis)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    first = _unit(np.cross(axis, reference))
    return first, _unit(np.cross(axis, first))


def _terminal_hydrogen_directions(carbon_bond):
    axis = _unit(carbon_bond)
    first, second = _perpendicular_frame(axis)
    axial = -1.0 / 3.0
    radial = math.sqrt(1.0 - axial * axial)
    return [
        axial * axis
        + radial * (math.cos(phi) * first + math.sin(phi) * second)
        for phi in (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)
    ]


def _internal_hydrogen_directions(left_bond, right_bond):
    left = _unit(left_bond)
    right = _unit(right_bond)
    cosine = float(np.dot(left, right))
    in_plane = (-1.0 / 3.0) / (1.0 + cosine) * (left + right)
    normal = _unit(np.cross(left, right))
    normal_scale2 = 1.0 - float(np.dot(in_plane, in_plane))
    if normal_scale2 <= 0.0:
        raise ValueError("invalid C-C-C angle for the CH2 construction")
    out_of_plane = math.sqrt(normal_scale2) * normal
    return [in_plane + out_of_plane, in_plane - out_of_plane]


def make_n_alkane(ncarbon):
    """Return a valence-checked planar-zigzag ``C_n H_(2n+2)`` geometry.

    Coordinates are in Angstrom and returned in PySCF's ``[(symbol, xyz)]``
    form.  The construction is deterministic, which also makes the generated
    coordinate hash suitable for checkpoint compatibility checks.
    """
    if ncarbon < 1:
        raise ValueError("ncarbon must be at least one")
    if ncarbon == 1:
        tetrahedron = np.asarray([
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]) / math.sqrt(3.0)
        atoms = [("C", np.zeros(3))]
        atoms.extend(
            ("H", CH_BOND_ANGSTROM * direction)
            for direction in tetrahedron
        )
        return [(symbol, tuple(coordinate)) for symbol, coordinate in atoms]

    half_turn = math.radians(180.0 - CCC_ANGLE_DEG) / 2.0
    carbon = [np.zeros(3)]
    for bond_index in range(ncarbon - 1):
        angle = half_turn if bond_index % 2 == 0 else -half_turn
        direction = np.array([math.cos(angle), math.sin(angle), 0.0])
        carbon.append(carbon[-1] + CC_BOND_ANGSTROM * direction)
    carbon = np.asarray(carbon)

    atoms = [("C", coordinate) for coordinate in carbon]
    for index, coordinate in enumerate(carbon):
        if index == 0:
            directions = _terminal_hydrogen_directions(carbon[1] - coordinate)
        elif index == ncarbon - 1:
            directions = _terminal_hydrogen_directions(
                carbon[-2] - coordinate
            )
        else:
            directions = _internal_hydrogen_directions(
                carbon[index - 1] - coordinate,
                carbon[index + 1] - coordinate,
            )
        atoms.extend(
            ("H", coordinate + CH_BOND_ANGSTROM * direction)
            for direction in directions
        )
    _validate_alkane(atoms, ncarbon)
    return [(symbol, tuple(coordinate)) for symbol, coordinate in atoms]


def _validate_alkane(atoms, ncarbon):
    symbols = [symbol for symbol, _ in atoms]
    coordinates = np.asarray([coordinate for _, coordinate in atoms])
    expected_hydrogen = 2 * ncarbon + 2
    if symbols.count("C") != ncarbon or symbols.count("H") != expected_hydrogen:
        raise ValueError("generated alkane has the wrong molecular formula")
    adjacency = np.zeros((len(atoms), len(atoms)), dtype=bool)
    for left in range(len(atoms)):
        for right in range(left):
            pair = frozenset((symbols[left], symbols[right]))
            distance = np.linalg.norm(coordinates[left] - coordinates[right])
            bonded = (
                (pair == {"C"} and distance < 1.75)
                or (pair == {"C", "H"} and distance < 1.25)
            )
            adjacency[left, right] = adjacency[right, left] = bonded
    expected_valence = np.asarray([
        4 if symbol == "C" else 1 for symbol in symbols
    ])
    if not np.array_equal(adjacency.sum(axis=1), expected_valence):
        raise ValueError("generated alkane does not have single-bond valences")


__all__ = ["make_n_alkane"]
