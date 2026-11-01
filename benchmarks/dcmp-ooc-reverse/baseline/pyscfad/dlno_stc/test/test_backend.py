"""Tiny independent references for the selected/full native backend."""
import importlib

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import pytest

KEYS = ('foo', 'fvv', 'B')


def native():
    # Optional when absent; an installed extension with broken dependencies fails.
    import os
    if importlib.util.find_spec('pyscfad.dlno_stc._stc_mp2') is None:
        if os.environ.get('PYSCFAD_REQUIRE_STC_EXTENSION'):
            pytest.fail('STC C++ extension has not been built')
        pytest.skip('optional STC C++ extension is not built')
    return importlib.import_module('pyscfad.dlno_stc._stc_mp2')


def fixture(repeated=False):
    rng = np.random.default_rng(41)
    foo = np.array([[-.8, .07], [.07, -.5]])
    fvv = np.array([[.3, -.02, .04], [-.02, .6, .01], [.04, .01, .9]])
    if repeated:
        foo = -.6 * np.eye(2)
        fvv = .4 * np.eye(3) + np.diag([0., 1e-13, 0.])
    return dict(zip(KEYS, [foo, fvv, rng.normal(size=(4, 2, 3)) / 5]))
