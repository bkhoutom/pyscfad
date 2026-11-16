"""Saved preparation pullbacks compose all three native cotangents."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscfad.dlno_stc.adjoint import apply_selected_pullback


def prepare(x, y):
    return dict(foo=jnp.array([[x+y, x-y], [x-y, y*y+2.]]),
                fvv=jnp.array([[x*y+3.]]),
                B=jnp.stack((x+2*y, x*y, x-y, y*y)).reshape(2, 2, 1))


@pytest.mark.parametrize('seed', [0., 1., -2.5])
def test_saved_preparation_matches_independent_energy_derivative(seed):
    primals = (jnp.asarray(.7), jnp.asarray(-.2))
    inputs, pullback = jax.vjp(prepare, *primals)
    result = dict(energy=np.float64(0.), energy_standard_error=np.float64(0.),
                  diagnostics={}, cotangents={k: np.asarray(v) for k, v in inputs.items()})
    actual = apply_selected_pullback(pullback, inputs, result, energy_bar=seed)
    expected = jax.grad(lambda x, y: .5*sum(jnp.sum(v*v) for v in prepare(x,y).values()), (0,1))(*primals)
    np.testing.assert_allclose(actual, seed*np.asarray(expected), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('seed', [np.nan, np.inf, [1., 2.]])
def test_non_scalar_or_nonfinite_seeds_are_rejected(seed):
    inputs, pullback = jax.vjp(prepare, jnp.asarray(.7), jnp.asarray(-.2))
    with pytest.raises(ValueError): apply_selected_pullback(pullback, inputs, {}, energy_bar=seed)
