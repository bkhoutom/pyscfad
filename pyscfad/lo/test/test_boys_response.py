"""Numerical regression checks for the projected Boys localization response."""
import importlib
from contextlib import nullcontext
from functools import partial, wraps

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad import config, gto, scf

boys = importlib.import_module('pyscfad.lo.boys')


@pytest.fixture(scope='module')
def case():
    previous = config.moleintor_opt
    config.update('pyscfad_moleintor_opt', True)
    mol = gto.Mole(atom='O 0 0 0; H .12 -.74 .57; H -.05 .83 .62',
                   basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = scf.RHF(mol)
    mf.chkfile = None
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    coeff = jnp.asarray(mf.mo_coeff[:, 1:5])
    rng = np.random.default_rng(128)
    x = jnp.asarray(rng.normal(size=6) * .17)
    seed = jnp.asarray(rng.normal(size=6))
    try:
        yield mol, coeff, x, seed
    finally:
        config.update('pyscfad_moleintor_opt', previous)


def test_projected_condition_matches_nonzero_rotation_and_hessian(case):
    assert hasattr(boys, '_projected_opt_cond'), 'projected response is missing'
    mol, coeff, x, seed = case
    expected, baseline = jax.vjp(lambda z: boys._opt_cond(z, mol, coeff), x)
    actual, projected = jax.vjp(lambda z: boys._projected_opt_cond(z, mol, coeff), x)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(projected(seed)[0], baseline(seed)[0],
                               rtol=2e-10, atol=2e-11)


def test_projected_condition_preserves_external_molecule_and_orbital_vjp(case):
    assert hasattr(boys, '_projected_opt_cond'), 'projected response is missing'
    mol, coeff, x, seed = case
    _, baseline = jax.vjp(lambda m, c: boys._opt_cond(x, m, c), mol, coeff)
    _, projected = jax.vjp(lambda m, c: boys._projected_opt_cond(x, m, c), mol, coeff)
    expected, actual = baseline(seed), projected(seed)
    for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_allclose(a, b, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize('variant', [None, 'projected', 'projected_precond'])
def test_complete_localization_pullback_matches_baseline(case, monkeypatch, variant):
    assert hasattr(boys, '_projected_opt_cond'), 'projected response is missing'
    mol, coeff, _, _ = case
    seed = jnp.asarray(np.random.default_rng(27).normal(size=coeff.shape))
    options = dict(init_guess='atomic', conv_tol=1e-12, conv_tol_grad=1e-9,
                   gmres_options=dict(tol=1e-10, atol=1e-12, restart=20, maxiter=100))
    def localize(m, c):
        return boys.boys(m, c, **options)
    monkeypatch.setenv('BOYS_RESPONSE_VARIANT', 'baseline')
    with monkeypatch.context() as uncached:
        uncached.setattr(boys.Boys, '_cache_dipoles', lambda *args, **kwargs: nullcontext())
        expected, baseline = jax.vjp(localize, mol, coeff)
        expected_bars = baseline(seed)
    if variant is None:
        monkeypatch.delenv('BOYS_RESPONSE_VARIANT', raising=False)
    else:
        monkeypatch.setenv('BOYS_RESPONSE_VARIANT', variant)
    actual, projected = jax.vjp(localize, mol, coeff)
    actual_bars = projected(seed)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
    for a, b in zip(jax.tree_util.tree_leaves(actual_bars),
                    jax.tree_util.tree_leaves(expected_bars)):
        np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-9)


def test_preconditioner_is_finite_for_flat_and_negative_diagonal():
    from pyscfad.tools import linear_solver
    assert hasattr(boys, '_projected_preconditioner'), 'preconditioner is missing'
    for dipoles in (jnp.zeros((3, 4, 4)), jnp.ones((3, 4, 4))):
        operator = boys._projected_preconditioner(
            (jnp.zeros(6), dipoles), None, None, solver=linear_solver.pyscfad_gmres)
        actual = operator.matvec(np.arange(1., 7.))
        assert np.all(np.isfinite(actual))
        assert np.all(actual > 0)


def test_preconditioner_reaches_solver_and_true_equation_is_solved(case, monkeypatch):
    assert hasattr(boys, '_projected_preconditioner'), 'preconditioner is missing'
    mol, coeff, _, _ = case
    original = boys.gen_gmres
    records = []
    def instrumented_solver(**options):
        solve = original(**options)
        @wraps(solve)
        def recorded(action, rhs, **kwargs):
            solution, info = solve(action, rhs, **kwargs)
            residual = np.linalg.norm(np.asarray(action(solution)) - np.asarray(rhs))
            records.append((kwargs.get('M'), residual, np.linalg.norm(np.asarray(rhs))))
            return solution, info
        return recorded
    monkeypatch.setattr(boys, 'gen_gmres', instrumented_solver)
    monkeypatch.setenv('BOYS_RESPONSE_VARIANT', 'projected_precond')
    _, pullback = jax.vjp(lambda c: boys.boys(
        mol, c, init_guess='atomic', conv_tol=1e-12, conv_tol_grad=1e-9,
        gmres_options=dict(tol=1e-10, atol=1e-12, restart=20, maxiter=100)), coeff)
    pullback(jnp.asarray(np.random.default_rng(28).normal(size=coeff.shape)))
    assert len(records) == 1
    operator, residual, rhs_norm = records[0]
    assert operator is not None, 'root_vjp must not override the selected preconditioner'
    assert residual <= max(1e-12, 1e-10 * rhs_norm)


@pytest.mark.parametrize('captured', ['default', 'partial', 'wrapped'])
def test_eager_jax_solver_does_not_receive_host_preconditioner(captured):
    from pyscfad.tools import linear_solver
    solver = partial(linear_solver.jax_gmres, tol=1e-6)
    if captured == 'wrapped':
        original = solver
        @wraps(original)
        def solver(*args, **kwargs):
            return original(*args, **kwargs)
    elif captured == 'default':
        solver = None
    previous = config.moleintor_opt
    try:
        config.update('pyscfad_moleintor_opt', False)
        operator = boys._projected_preconditioner(
            (jnp.zeros(6), jnp.zeros((3, 4, 4))), None, None, solver=solver)
    finally:
        config.update('pyscfad_moleintor_opt', previous)
    assert operator is None, 'a concrete JAX solve cannot consume scipy.LinearOperator'


def test_unknown_variant_is_rejected(case, monkeypatch):
    mol, coeff, _, _ = case
    monkeypatch.setenv('BOYS_RESPONSE_VARIANT', 'misspelled')
    with pytest.raises(ValueError, match='BOYS_RESPONSE_VARIANT'):
        boys.boys(mol, coeff)


@pytest.mark.parametrize('requested_tol,requested_atol',
                         [(None, None), (None, 1e-12), (3e-9, 1e-12)])
@pytest.mark.parametrize('optimized_integrals', [False, True])
def test_default_response_solves_with_preconditioner_and_respects_options(
        case, monkeypatch, requested_tol, requested_atol, optimized_integrals):
    """Catch a disabled default, dropped preconditioner, or overwritten tolerance."""
    mol, coeff, _, _ = case
    monkeypatch.delenv('BOYS_RESPONSE_VARIANT', raising=False)
    requested = dict(restart=20, maxiter=100)
    if requested_tol is not None:
        requested['tol'] = requested_tol
    if requested_atol is not None:
        requested['atol'] = requested_atol
    original_options = requested.copy()
    original = boys.gen_gmres
    records = []

    def instrumented_solver(**options):
        solve = original(**options)

        @wraps(solve)
        def recorded(action, rhs, **kwargs):
            solution, info = solve(action, rhs, **kwargs)
            residual = np.linalg.norm(np.asarray(action(solution)) - np.asarray(rhs))
            records.append((options.copy(), kwargs.get('M'), residual,
                            np.linalg.norm(np.asarray(rhs)), info))
            return solution, info
        return recorded

    monkeypatch.setattr(boys, 'gen_gmres', instrumented_solver)
    from pyscfad import config_update
    with config_update('pyscfad_moleintor_opt', optimized_integrals):
        _, pullback = jax.vjp(lambda c: boys.boys(
            mol, c, init_guess='atomic', conv_tol=1e-12, conv_tol_grad=1e-9,
            gmres_options=requested), coeff)
        bars = pullback(jnp.asarray(np.random.default_rng(28).normal(size=coeff.shape)))
    jax.block_until_ready(bars)
    assert requested == original_options, 'caller options must not be mutated'
    assert len(records) == 1
    options, operator, residual, rhs_norm, info = records[0]
    expected_tol = 1e-7 if requested_tol is None else requested_tol
    expected_atol = 1e-7 if requested_atol is None else requested_atol
    assert options.get('tol') == expected_tol
    assert options.get('atol') == expected_atol
    assert operator is not None, 'optimized Boys response must be enabled by default'
    assert info == 0
    assert residual <= max(expected_atol, expected_tol * rhs_norm)
