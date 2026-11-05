"""Bounded saddle escapes and acceptance of stationary response branches."""
from contextlib import contextmanager

import numpy as np
import pytest
from pyscf import gto
from pyscfad.lo import boys


def test_differentiable_boys_escapes_symmetric_stationary_saddle(monkeypatch):
    # Simultaneously diagonalizable dipoles give an independent exact minimum.
    # At identity the gradient vanishes but mixing columns 0 and 1 lowers cost.
    dipoles = np.zeros((3, 3, 3))
    dipoles[0] = [[0., 1., 0.], [1., 0., 0.], [0., 0., 3.]]

    @contextmanager
    def cache(loc):
        loc._boys_dipole_cache = (dipoles, 0.)
        try:
            yield
        finally:
            del loc._boys_dipole_cache

    monkeypatch.setattr(boys.Boys, '_cache_dipoles', cache)
    mol = gto.M(atom='H 0 0 0', spin=1, verbose=0)
    loc = boys.Boys(mol, np.eye(3))
    with cache(loc):
        np.testing.assert_allclose(loc.get_grad(np.eye(3)), 0., atol=0.)
        assert boys._lowest_curvature(loc, np.eye(3))[0] < -1.
    x, order = boys._boys(None, mol, np.eye(3), init_guess=np.eye(3),
                         conv_tol=1e-12, conv_tol_grad=1e-9)
    u = np.asarray(boys.extract_rotation(x))[:, order]
    with cache(loc):
        assert boys._lowest_curvature(loc, u)[0] > 1.
        assert np.linalg.norm(loc.get_grad(u)) < 1e-8
        np.testing.assert_allclose(loc.cost_function(u), -11., atol=1e-12)


def test_lowest_curvature_detects_saddle_with_positive_diagonal():
    from types import SimpleNamespace
    hessian = np.array([[1., 2.], [2., 1.]])
    loc = SimpleNamespace(gen_g_hop=lambda u: (np.zeros(2), lambda x: hessian @ x,
                                             hessian.diagonal()))
    value, direction = boys._lowest_curvature(loc, None)
    np.testing.assert_allclose(value, -1., atol=1e-14)
    np.testing.assert_allclose(hessian @ direction, value * direction, atol=1e-14)


@pytest.mark.parametrize('minimum', [-20., 0.])
def test_lowest_curvature_resolves_cluster_and_flat_modes(minimum):
    from types import SimpleNamespace
    rng = np.random.default_rng(703)
    rotation, _ = np.linalg.qr(rng.normal(size=(120, 120)))
    spectrum = np.r_[minimum + np.arange(24)*1e-4, np.linspace(1., 10000., 96)]
    hessian = (rotation * spectrum) @ rotation.T
    loc = SimpleNamespace(gen_g_hop=lambda u: (np.zeros(120), lambda x: hessian @ x,
                                             hessian.diagonal()))
    value, direction = boys._lowest_curvature(loc, None)
    np.testing.assert_allclose(value, minimum, atol=1e-8)
    assert np.linalg.norm(hessian @ direction - value*direction) < 1e-7


def test_lowest_curvature_recovers_failed_iterative_solve(monkeypatch):
    from types import SimpleNamespace
    from scipy.sparse import linalg
    hessian = np.array([[1., 2., 0.], [2., 1., 0.], [0., 0., 5.]])
    loc = SimpleNamespace(gen_g_hop=lambda u: (np.zeros(3), lambda x: hessian @ x,
                                             hessian.diagonal()))

    def failed(*args, **kwargs):
        raise linalg.ArpackNoConvergence('failed', np.empty(0), np.empty((3, 0)))

    monkeypatch.setattr(linalg, 'eigsh', failed)
    value, direction = boys._lowest_curvature(loc, None)
    np.testing.assert_allclose(value, -1., atol=1e-14)
    np.testing.assert_allclose(hessian @ direction, value*direction, atol=1e-14)


@pytest.fixture
def persistent_saddle(monkeypatch):
    from io import StringIO
    from types import SimpleNamespace
    from scipy.linalg import expm
    generator = np.array([[0., -1.], [1., 0.]])
    calls = []
    loc = SimpleNamespace(conv_tol_grad=1e-8, conv_tol=1e-10, verbose=0,
                          stdout=StringIO(), unpack_uniq_var=lambda v: v[0]*generator,
                          cost_function=lambda u: -u[1,0]**2,
                          get_grad=lambda u: np.zeros(1))
    monkeypatch.setattr(boys, '_lowest_curvature', lambda loc, u: (-2., np.ones(1)))

    def restarted(loc, **kwargs):
        calls.append(1)
        return expm(.2*len(calls)*generator), np.arange(2)

    monkeypatch.setattr(boys, 'kernel', restarted)
    return loc, calls


def test_at_most_three_escapes_then_accept_stationary_saddle(persistent_saddle):
    loc, calls = persistent_saddle
    with pytest.warns(RuntimeWarning, match='three saddle escapes exhausted'):
        u, order = boys._stabilize_boys(loc, np.eye(2), np.arange(2))
    assert len(calls) == 3
    np.testing.assert_allclose(u[1,0], np.sin(.6))
    np.testing.assert_array_equal(order, [0, 1])


def test_failed_descent_keeps_stationary_solution(persistent_saddle):
    loc, calls = persistent_saddle
    loc.cost_function = lambda u: 0.
    with pytest.warns(RuntimeWarning, match='trial steps did not lower'):
        u, _ = boys._stabilize_boys(loc, np.eye(2), np.arange(2))
    assert not calls
    np.testing.assert_array_equal(u, np.eye(2))


def test_unavailable_curvature_is_not_fatal_for_stationary_frame(persistent_saddle, monkeypatch):
    loc, calls = persistent_saddle

    def unavailable(*args):
        raise boys.BoysCurvatureError('eigenpair residual too large')

    monkeypatch.setattr(boys, '_lowest_curvature', unavailable)
    with pytest.warns(RuntimeWarning, match='curvature check unavailable'):
        u, _ = boys._stabilize_boys(loc, np.eye(2), np.arange(2))
    assert not calls
    np.testing.assert_array_equal(u, np.eye(2))


def test_keep_earlier_stationary_candidate_if_restart_does_not_converge(persistent_saddle):
    loc, calls = persistent_saddle
    loc.get_grad = lambda u: np.array([0. if np.allclose(u, np.eye(2)) else 1e-3])
    with pytest.warns(RuntimeWarning, match='retaining the best stationary'):
        u, _ = boys._stabilize_boys(loc, np.eye(2), np.arange(2))
    assert len(calls) == 3
    np.testing.assert_array_equal(u, np.eye(2))


def test_nonstationarity_is_distinct_from_negative_curvature(persistent_saddle):
    loc, calls = persistent_saddle
    loc.get_grad = lambda u: np.array([1e-3])
    with pytest.raises(RuntimeError, match='No sufficiently stationary Boys solution'):
        boys._stabilize_boys(loc, np.eye(2), np.arange(2))
    assert len(calls) == 3


def test_replay_disables_escapes_without_freezing_response_branch(persistent_saddle):
    loc, calls = persistent_saddle
    with pytest.warns(RuntimeWarning, match='disabled for branch replay'):
        u, order = boys._stabilize_boys(loc, np.eye(2), np.arange(2), max_saddle_escapes=0)
    assert not calls
    np.testing.assert_array_equal(u, np.eye(2))
    np.testing.assert_array_equal(order, [0, 1])


def test_replay_still_rejects_unconverged_stationarity(persistent_saddle):
    loc, calls = persistent_saddle
    loc.get_grad = lambda u: np.array([1e-3])
    with pytest.raises(RuntimeError, match='No sufficiently stationary Boys solution'):
        boys._stabilize_boys(loc, np.eye(2), np.arange(2), max_saddle_escapes=0)
    assert not calls
