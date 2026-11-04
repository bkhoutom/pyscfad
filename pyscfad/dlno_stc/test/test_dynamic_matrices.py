"""Dynamic OpenMP teams preserve deterministic W and its complete pullback."""
import ctypes
import numpy as np
from pyscfad.dlno_stc import backend
from pyscfad.dlno_stc.test.test_backend import native
from pyscfad.dlno_stc.test.test_full_kernel import full_fixture
from pyscfad.dlno_stc.test.test_w_delayed_pullback import grid


def test_dynamic_w_matches_fixed_team_energy_matrices_and_pullback():
    native()
    omp = ctypes.CDLL('libgomp.so.1')
    previous = omp.omp_get_dynamic()
    x = full_fixture()
    seeds = dict(energy_bar=.7, occupied_bar=np.eye(2), virtual_bar=np.eye(3)*.4)
    results = []
    try:
        for dynamic in (0, 1):
            omp.omp_set_dynamic(dynamic)
            forward = backend.solve_matrices(x, dict(target_indices=(0,1)),
                dict(grid(), threads=4), matrix_kind='W')
            reverse = backend.pullback_matrices(x, forward, **seeds)
            assert omp.omp_get_dynamic() == dynamic
            results.append((forward,reverse))
        for key in ('energy','occupied','virtual'):
            np.testing.assert_allclose(results[0][0][key],results[1][0][key],rtol=2e-12,atol=2e-13)
        for key in x:
            np.testing.assert_allclose(results[0][1][key],results[1][1][key],rtol=2e-11,atol=2e-12)
    finally:
        omp.omp_set_dynamic(previous)
