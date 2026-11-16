"""Selected triples and all nine derivatives after undoing occupied framing."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc.smooth_triples import _run_deterministic
from pyscfad.dlno_stc.test.test_triples_stable import fixture, reference


def rotate(values, q):
    axes = ((0,), (0, 1), (0,), (0, 2, 3), (0, 2),
            (0,), (0, 1), (), (0,))
    out = []
    for value, indices in zip(values, axes):
        for axis in indices:
            value = jnp.moveaxis(jnp.tensordot(q.T, value, axes=(1, axis)), 0, axis)
        out.append(value)
    return tuple(out)


@pytest.mark.parametrize('node', [0., .37])
@pytest.mark.parametrize('with_grad', [False, True])
def test_original_occupied_lis_matches_full_six_index_energy_and_all_derivatives(node, with_grad):
    q = jnp.asarray(np.linalg.qr(np.random.default_rng(788).normal(size=(3, 3)))[0])
    old = list(map(jnp.asarray, fixture(selected=2)))
    old[8] = q[:, :2]
    old = tuple(old)
    transformed, transpose = jax.vjp(lambda values: rotate(values, q), old)
    assert float(jnp.max(jnp.abs(transformed[8]-jnp.eye(3)[:, :2]))) < 2e-15
    expected, expected_bars = jax.value_and_grad(reference)(old, node)
    (energy, diagnostics), bars = _run_deterministic(transformed,
        np.array([node]), np.array([1.]), with_grad=with_grad,
        max_memory=2000., rotate=True, trace_exponent=5.)
    assert diagnostics['selected_occupied_frame']
    np.testing.assert_allclose(energy, expected, rtol=3e-11, atol=3e-13)
    if with_grad:
        actual, = transpose(tuple(map(jnp.asarray, bars)))
        for i, (got, want) in enumerate(zip(actual, expected_bars)):
            np.testing.assert_allclose(got, want, rtol=3e-10, atol=3e-12,
                                       err_msg=f'input {i}')
    else:
        assert bars is None
