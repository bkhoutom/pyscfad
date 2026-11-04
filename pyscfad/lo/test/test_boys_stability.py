"""A stationary localization saddle must not be accepted for differentiation."""
from contextlib import contextmanager

import numpy as np
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
