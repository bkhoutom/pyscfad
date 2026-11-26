"""Caller-owned OpenMP policy and deterministic/stochastic matrix replay."""
import ctypes
import numpy as np
import pytest
from pyscfad.dlno_stc import backend
from pyscfad.dlno_stc.test.test_backend import native
from pyscfad.dlno_stc.test.test_full_kernel import full_fixture
from pyscfad.dlno_stc.test.test_w_delayed_pullback import grid


@pytest.mark.parametrize('kind', ['W', 'D'])
@pytest.mark.parametrize('legacy', [False, True])
def test_matrix_team_policy_and_caller_restoration(kind, legacy, monkeypatch):
    native()
    if legacy:
        monkeypatch.setattr(backend, '_matrix_extension', lambda *args: native())
    omp = ctypes.CDLL('libgomp.so.1')
    previous = omp.omp_get_dynamic()
    x = full_fixture()
    seeds = dict(energy_bar=.7, occupied_bar=np.eye(2), virtual_bar=np.eye(3)*.4)
    results = []
    try:
        for dynamic in (0, 1):
            omp.omp_set_dynamic(dynamic)
            forward = backend.solve_matrices(x, dict(target_indices=(0,1)),
                dict(grid(), threads=4), matrix_kind=kind)
            assert all(batch['openmp_dynamic'] == bool(dynamic)
                       for batch in forward['diagnostics']['batches'])
            assert omp.omp_get_dynamic() == dynamic
            reverse = backend.pullback_matrices(x, forward, **seeds)
            assert omp.omp_get_dynamic() == dynamic
            results.append((forward,reverse))
        for key in ('energy','occupied','virtual'):
            np.testing.assert_allclose(results[0][0][key],results[1][0][key],rtol=2e-12,atol=2e-13)
        for key in x:
            np.testing.assert_allclose(results[0][1][key],results[1][1][key],rtol=2e-11,atol=2e-12)
    finally:
        omp.omp_set_dynamic(previous)


def test_standalone_mp2_preserves_policy_and_gradients():
    extension = native()
    omp = ctypes.CDLL('libgomp.so.1')
    previous, previous_threads = omp.omp_get_dynamic(), omp.omp_get_max_threads()
    inputs = tuple(full_fixture().values())
    results = []
    try:
        for dynamic in (0, 1):
            omp.omp_set_dynamic(dynamic)
            results.append(extension.solve_selected(*inputs, (0, 1),
                dict(mode='deterministic', threads=4, laplace_roots=[.2, .5],
                     laplace_weights=[.4, .6]), with_grad=True))
            assert omp.omp_get_dynamic() == dynamic
            assert omp.omp_get_max_threads() == previous_threads
        np.testing.assert_allclose(results[0]['energy'], results[1]['energy'],
                                   rtol=2e-12, atol=2e-13)
        for key in ('foo', 'fvv', 'B'):
            np.testing.assert_allclose(results[0]['cotangents'][key],
                                       results[1]['cotangents'][key],
                                       rtol=2e-11, atol=2e-12)
    finally:
        omp.omp_set_dynamic(previous)


@pytest.mark.parametrize('kind', ['W', 'D'])
def test_stochastic_replay_survives_dynamic_policy_change(kind):
    extension = native()
    omp = ctypes.CDLL('libgomp.so.1')
    previous = omp.omp_get_dynamic()
    inputs = tuple(full_fixture().values())
    controls = dict(mode='stochastic', threads=4, laplace_roots=[.2, .5],
                    laplace_weights=[.4, .6], energy_tolerance=.03,
                    system_workload_cutoff=10., matrix_tolerance=.03,
                    pair_pilot_samples=32, min_samples=32, global_seed=74,
                    domain_tau=10., virtual_domain_tau=10.)
    if kind == 'W':
        controls.pop('energy_tolerance')
        controls.pop('system_workload_cutoff')
    seeds = (.7, np.eye(2), .4*np.eye(3))
    try:
        forwards = []
        for dynamic in (0, 1):
            omp.omp_set_dynamic(dynamic)
            forward = extension.solve_matrices(*inputs, (0, 1), controls,
                matrix_kind=kind, aux_offsets=np.array([0, 2, 4]))
            assert forward['diagnostics']['openmp_dynamic'] == bool(dynamic)
            assert omp.omp_get_dynamic() == dynamic
            forwards.append(forward)
        for forward in forwards:
            reverses = []
            for dynamic in (0, 1):
                omp.omp_set_dynamic(dynamic)
                reverses.append(extension.pullback_matrices(
                    *inputs, forward['replay'], *seeds))
                assert omp.omp_get_dynamic() == dynamic
            # Independent forwards may allocate different sample counts from
            # measured pilot costs. Compare policies on the SAME saved replay.
            for key in ('foo', 'fvv', 'B'):
                np.testing.assert_allclose(reverses[0][key], reverses[1][key],
                                           rtol=2e-10, atol=2e-12)
            # The seeded sampled objective is quartic in B with its proposal
            # and draw allocation frozen. This also checks replay vs primal.
            objective = (.7*forward['energy'] + np.trace(forward['occupied'])
                         + .4*np.trace(forward['virtual']))
            np.testing.assert_allclose(np.vdot(reverses[0]['B'], inputs[-1]),
                                       4*objective, rtol=2e-10, atol=2e-12)
    finally:
        omp.omp_set_dynamic(previous)
