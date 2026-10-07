"""Shared defaults for high-level STC-MP2 drivers."""

import numpy

DEFAULT_ENERGY_TOLERANCE = 3e-4  # Hartree (0.3 mHa)


def default_laplace_grid():
    """Return independent arrays for the original six-point fixed grid."""
    roots = numpy.array([0.003431, 0.023534, 0.088984, 0.275603, 0.757121, 1.906218])
    weights = numpy.array([0.009348, 0.035196, 0.107559, 0.293035, 0.729094, 1.690608])
    return roots * 2.6, weights * 2.6
