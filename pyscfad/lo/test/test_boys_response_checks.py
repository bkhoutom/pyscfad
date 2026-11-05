"""Signed response, conditioning, gauge compatibility, and true residuals."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.lo import _boys_response as response


def checked(matrix, *, symmetry=False, base=None):
    if base is None:
        base = lambda action, rhs, **kw: (np.linalg.solve(matrix, rhs), 0)
    return response.checked_solver(base, symmetry=symmetry, tol=1e-10, atol=1e-12)


def test_indefinite_response_keeps_negative_eigenvalues():
    h = np.array([[-2., .4], [.4, 3.]])
    b = np.array([2., 6.])
    x, info = checked(h)(lambda x: h@x, b)
    np.testing.assert_allclose(x, np.linalg.solve(h,b), atol=1e-13)
    assert x[0] < 0 and info == 0


def test_small_nonzero_mode_warns_but_is_not_truncated():
    h = np.diag([-2., 1e-11, 3.])
    b = np.ones(3)
    with pytest.warns(RuntimeWarning, match='Ill-conditioned Boys response'):
        x, _ = checked(h)(lambda x: h@x, b)
    np.testing.assert_allclose(x, [-.5, 1e11, 1/3], rtol=1e-14)


def test_declared_gauge_retains_signed_range_solution():
    h = np.diag([-2., 0., 3.])
    b = np.array([4., 0., 9.])
    with pytest.warns(RuntimeWarning, match='compatible range solution'):
        x, _ = checked(h, symmetry=True)(lambda x: h@x, b)
    np.testing.assert_allclose(x, [-2., 0., 3.], atol=1e-13)
    np.testing.assert_allclose(h@x,b,atol=1e-13)


def test_incompatible_gauge_rhs_is_not_silently_discarded():
    h = np.diag([-2., 0., 3.])
    with pytest.raises(RuntimeError, match='incompatible null-space component'):
        checked(h, symmetry=True)(lambda x: h@x, np.ones(3))


def test_null_mode_is_not_automatically_declared_a_gauge():
    h = np.diag([-2., 0., 3.])
    with pytest.raises(RuntimeError, match='declare symmetry=True'):
        checked(h)(lambda x: h@x, np.array([1.,0.,1.]))


def test_success_status_cannot_hide_a_bad_true_residual():
    h = np.diag([-2., 3.])
    bogus = lambda action, rhs, **kw: (np.zeros_like(rhs), 0)
    with pytest.raises(RuntimeError, match='original equation'):
        checked(h, base=bogus)(lambda x: h@x, np.ones(2))


@pytest.mark.parametrize('compatible', [True, False])
def test_matrix_free_gauge_solve_checks_full_residual(monkeypatch, compatible):
    monkeypatch.setattr(response, '_DENSE_LIMIT', 1)
    h = np.diag([-2., 0., 3.])
    b = np.array([4., 0. if compatible else 1., 9.])
    if compatible:
        x, _ = checked(h, symmetry=True)(lambda x: h@x, b)
        np.testing.assert_allclose(x, [-2.,0.,3.], atol=1e-12)
    else:
        with pytest.raises(RuntimeError, match='original equation'):
            checked(h, symmetry=True)(lambda x: h@x, b)


def test_matrix_free_amplification_exposes_ill_conditioning(monkeypatch):
    monkeypatch.setattr(response, '_DENSE_LIMIT', 1)
    h = np.diag([-2., 1e-11, 3.])
    with pytest.warns(RuntimeWarning, match='condition_lower_bound'):
        x, _ = checked(h)(lambda x: h@x, np.ones(3))
    assert x[1] == 1e11


def test_jitted_signed_response_and_its_derivative():
    h = jnp.array([[-2., .4], [.4, 3.]])
    base = lambda action, rhs, **kw: (jnp.linalg.solve(h, rhs), 0)
    solver = response.checked_solver(base, tol=1e-10, atol=1e-12)
    run = jax.jit(lambda b: solver(lambda x: h@x,b)[0])
    b = jnp.array([2.,6.])
    np.testing.assert_allclose(run(b),np.linalg.solve(h,b),atol=1e-12)
    np.testing.assert_allclose(jax.jacrev(run)(b),np.linalg.inv(h),atol=1e-12)


def test_jitted_bad_residual_is_rejected():
    h = jnp.diag(jnp.array([-2.,3.]))
    base = lambda action, rhs, **kw: (jnp.zeros_like(rhs), 0)
    solver = response.checked_solver(base,tol=1e-10,atol=1e-12)
    run = jax.jit(lambda b: solver(lambda x: h@x,b)[0])
    with pytest.raises(Exception, match='Boys response residual'):
        jax.block_until_ready(run(jnp.ones(2)))


@pytest.mark.parametrize('compatible', [True, False])
def test_jitted_gauge_gmres_checks_original_equation(compatible):
    from functools import partial
    from jax.scipy.sparse.linalg import gmres
    h = jnp.diag(jnp.array([-2., 0., 3.]))
    base = partial(gmres, tol=1e-10, atol=1e-12, solve_method='incremental')
    solver = response.checked_solver(base, symmetry=True, tol=1e-10, atol=1e-12)
    run = jax.jit(lambda b: solver(lambda x: h@x, b)[0])
    b = jnp.array([4., 0. if compatible else 1., 9.])
    if compatible:
        np.testing.assert_allclose(run(b), [-2., 0., 3.], atol=1e-12)
    else:
        with pytest.raises(Exception, match='Boys response'):
            jax.block_until_ready(run(b))


@pytest.mark.parametrize('displacement', [0., .07])
def test_boys_saddle_pullback_matches_independent_finite_difference(displacement):
    """A real Boys stationary branch has a negative mode and finite response."""
    from pyscfad._src.implicit_diff import root_vjp
    from pyscfad.lo import boys
    from pyscfad.tools.linear_solver import gen_gmres, eager_gmres

    # Equal diagonal dipoles in the first two orbitals define an exact saddle:
    # q*cos(2*theta) + sin(2*theta) = 0. The third orbital makes the Hessian
    # indefinite rather than negative definite.
    def condition(x, q):
        dipoles = jnp.zeros((3, 3, 3)).at[0].set(
            jnp.array([[q, 1., 0.], [1., -q, 0.], [0., 0., 3.]]))
        return boys._projected_gradient(x, dipoles)

    q = jnp.asarray(displacement)
    x = jnp.array([-.5*jnp.arctan(q), 0., 0.])
    np.testing.assert_allclose(condition(x, q), 0., atol=1e-13)
    eigenvalues = np.linalg.eigvalsh(jax.jacrev(condition)(x, q))
    assert eigenvalues[0] < -1. and eigenvalues[-1] > 1.
    with eager_gmres():
        solver = response.checked_solver(gen_gmres(tol=1e-10, atol=1e-12),
                                         tol=1e-10, atol=1e-12)
    seed = jax.grad(lambda y: boys.extract_rotation(y)[1, 0])(x)
    derivative = root_vjp(condition, x, (q,), seed, solve=solver)[0]
    h = 1e-5
    observable = lambda q: np.sin(-.5*np.arctan(q))
    finite_difference = (observable(displacement+h) - observable(displacement-h))/(2*h)
    np.testing.assert_allclose(derivative, finite_difference, atol=3e-10, rtol=1e-9)
