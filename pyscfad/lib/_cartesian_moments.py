"""Unique Cartesian components of symmetric multiplicative AO operators."""
from functools import lru_cache
from itertools import combinations_with_replacement, product

import numpy as np


@lru_cache(maxsize=4)
def cartesian_components(rank):
    """Return representative flattened indices and the full-to-packed map.

    Ordering is xx, xy, xz, yy, yz, zz at rank two, and lexical sorted
    triples at rank three. Derivative axes are never symmetrized with the
    operator axes.
    """
    if rank not in (0, 1, 2, 3):
        raise ValueError('Cartesian moment rank must be 0, 1, 2, or 3')
    unique = list(combinations_with_replacement(range(3), rank))
    indices = np.asarray([sum(axis * 3**(rank - i - 1)
                             for i, axis in enumerate(axes))
                          for axes in unique], dtype=np.int32)
    lookup = {axes: i for i, axes in enumerate(unique)}
    inverse = np.asarray([lookup[tuple(sorted(axes))]
                          for axes in product(range(3), repeat=rank)], dtype=np.int32)
    indices.flags.writeable = inverse.flags.writeable = False
    return indices, inverse
