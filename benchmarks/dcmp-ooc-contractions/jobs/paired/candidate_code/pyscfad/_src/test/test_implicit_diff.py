"""Nested implicit roots must preserve their own root-variable response."""

import jax
import jax.numpy as jnp
import pytest

from pyscfad._src.implicit_diff import (
    _implicit_diff_external_vjp, _implicit_diff_solve_matvec,
    is_implicit_diff_external_vjp, is_implicit_diff_solve_matvec, root_vjp,
)


def _optimality(root, parameter, fixed_root):
    # Analogue of SCF's exact density factors in the external-argument phase.
    density = fixed_root if is_implicit_diff_external_vjp() else root
    return parameter + .5 * density - root


def _scalar_solve(matvec, rhs, **kwargs):
    return rhs / matvec(jnp.ones_like(rhs)), None


def test_nested_root_retains_response_and_restores_parent_phase():
    # root = 2 * parameter, so the independently known derivative is 2.
    args = (jnp.array(3.), jnp.array(6.))
    with _implicit_diff_external_vjp():
        derivative, _ = root_vjp(
            _optimality, jnp.array(6.), args, jnp.array(1.),
            solve=_scalar_solve, nondiff_argnums=(2,),
        )
        assert float(derivative) == 2.
        assert is_implicit_diff_external_vjp()
    assert not is_implicit_diff_external_vjp()


@pytest.mark.parametrize('failure_phase', ('root', 'solve'))
def test_nested_root_restores_parent_phase_on_exception(failure_phase):
    phases = []

    def optimality(root, parameter):
        phases.append(is_implicit_diff_external_vjp())
        if failure_phase == 'root':
            raise RuntimeError('root failure')
        return parameter - root

    def solve(matvec, rhs, **kwargs):
        phases.append(is_implicit_diff_external_vjp())
        raise RuntimeError('solve failure')

    with _implicit_diff_external_vjp():
        with pytest.raises(RuntimeError, match=failure_phase + ' failure'):
            root_vjp(optimality, jnp.array(3.), (jnp.array(3.),),
                     jnp.array(1.), solve=solve)
        assert is_implicit_diff_external_vjp()
    assert not is_implicit_diff_external_vjp()
    assert phases and not any(phases)


@pytest.mark.parametrize('fail_external', (False, True))
def test_nested_external_pullback_clears_parent_solve_phase(fail_external):
    phases = []

    @jax.custom_vjp
    def physical_term(root, parameter):
        return .5 * root + parameter

    def forward(root, parameter):
        return physical_term(root, parameter), None

    def backward(residual, bar):
        # Like DF J/K, the root-only matvec omits unused coordinate bars.
        solution_only = is_implicit_diff_solve_matvec()
        if is_implicit_diff_external_vjp():
            phases.append(solution_only)
            if fail_external:
                raise RuntimeError('external failure')
        return .5 * bar, jnp.zeros_like(bar) if solution_only else bar

    physical_term.defvjp(forward, backward)
    optimality = lambda root, parameter: physical_term(root, parameter) - root

    def differentiate():
        return root_vjp(optimality, jnp.array(6.), (jnp.array(3.),),
                        jnp.array(1.), solve=_scalar_solve)[0]

    with _implicit_diff_solve_matvec():
        if fail_external:
            with pytest.raises(RuntimeError, match='external failure'):
                differentiate()
        else:
            assert float(differentiate()) == 2.
        assert is_implicit_diff_solve_matvec()
        assert not is_implicit_diff_external_vjp()
    assert not is_implicit_diff_solve_matvec()
    assert phases == [False]


def test_fixed_point_aux_preserves_arguments_and_nested_phase():
    from pyscfad._src.implicit_diff import custom_fixed_point
    observed = []

    def fixed_point(root, parameter, fixed_root):
        density = fixed_root if is_implicit_diff_external_vjp() else root
        return parameter + .5 * density, jnp.array(17.)

    def preconditioner(aux, parameter, fixed_root):
        assert not is_implicit_diff_external_vjp()
        observed.append((float(aux), float(parameter), float(fixed_root)))
        return None

    @custom_fixed_point(fixed_point, solve=_scalar_solve, has_aux=True,
                        nondiff_argnums=(2,), use_converged_args={2: 1},
                        optfn_has_aux=True, gen_precond=preconditioner)
    def solver(initial, parameter, fixed_root):
        return 2 * parameter, 2 * parameter

    with _implicit_diff_external_vjp():
        derivative = jax.grad(lambda p: solver(jnp.array(0.), p, None)[0])(jnp.array(3.))
        assert float(derivative) == 2.
        assert is_implicit_diff_external_vjp()
    assert observed == [(17., 3., 6.)]


def test_traced_host_solver_fallback_does_not_build_host_preconditioner():
    from pyscfad.scipy.sparse.linalg import gmres

    def preconditioner(aux, parameter, *, solver):
        raise AssertionError('host preconditioner constructed during traced solve')

    def differentiate(seed):
        return root_vjp(lambda root, p: (p - root, jnp.array(17.)),
                        jnp.array(3.), (jnp.array(3.),), seed,
                        solve=gmres, optfn_has_aux=True,
                        gen_precond=preconditioner, gen_precond_host_only=True)[0]

    assert float(jax.jit(differentiate)(jnp.array(1.))) == 1.


def test_jax_preconditioner_is_preserved_by_traced_implicit_response():
    """A native JAX preconditioner must also run in a limited JIT solve."""
    from functools import partial
    from jax.scipy.sparse.linalg import gmres
    diagonal = jnp.array([1., 10., 100.])
    parameter = jnp.ones(3)
    solver = partial(gmres, restart=1, maxiter=1, tol=1e-12, atol=1e-12)

    def preconditioner(aux, parameter):
        return lambda vector: vector / aux

    def differentiate(seed):
        return root_vjp(lambda root, p: (p - diagonal * root, diagonal),
                        parameter / diagonal, (parameter,), seed, solve=solver,
                        optfn_has_aux=True, gen_precond=preconditioner)[0]

    expected = jnp.array([1., .1, .01])
    assert jnp.allclose(differentiate(jnp.ones(3)), expected, atol=1e-12, rtol=0)
    assert jnp.allclose(jax.jit(differentiate)(jnp.ones(3)), expected, atol=1e-12, rtol=0)
