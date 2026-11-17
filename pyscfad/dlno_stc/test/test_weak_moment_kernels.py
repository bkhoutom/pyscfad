"""Endpoint moments retain their full derivative with bounded JIT shapes."""
import importlib
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import _pairs


def _native():
    name = 'pyscfad.dlno_stc._weak_moments'
    assert importlib.util.find_spec(name) is not None, (
        'weak endpoints need native contractions with fixed translation tiles')
    return importlib.import_module(name)


def _fixture(rows, columns, complex_values=False):
    rng = np.random.default_rng(913 + rows + columns)

    def values(shape):
        value = rng.normal(size=shape) / np.sqrt(rows)
        if complex_values:
            value = value + .3j * rng.normal(size=shape) / np.sqrt(rows)
        return value

    screen = _pairs.WeakScreen(np.array([-.5]), values((rows, 1)),
                              np.linspace(.1, 1., columns), values((rows, columns)))
    # Neither AO legs nor Cartesian components are assumed symmetric in the
    # derivative oracle. These are independent packed operator inputs.
    raw = tuple(values(shape) for shape in
                ((rows, rows), (3, rows, rows), (6, rows, rows), (10, rows, rows)))
    return screen, raw


def _reference(screen, raw, order=4):
    return _pairs._multipole_orbital_data(None, screen, None, order, ao_moments=raw)


def _seeds(value, seed=71):
    rng = np.random.default_rng(seed)

    def one(leaf):
        result = rng.normal(size=leaf.shape)
        if np.iscomplexobj(leaf):
            result = result + 1j * rng.normal(size=leaf.shape)
        return np.asarray(result, dtype=leaf.dtype)

    return jax.tree_util.tree_map(one, value)


def _assert_tree(actual, expected, tolerance=3e-11):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for got, want in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        assert got.shape == want.shape
        assert got.dtype == want.dtype
        np.testing.assert_allclose(got, want, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('rows,columns', [(7, 0), (17, 5), (35, 127), (35, 128), (41, 129)])
@pytest.mark.parametrize('complex_values', [False, True])
def test_native_values_and_all_cotangents_match_nonsymmetric_reference(rows, columns, complex_values):
    native = _native()
    screen, raw = _fixture(rows, columns, complex_values)
    assert native.supports(screen, raw, 4)
    expected, reference_pullback = jax.vjp(_reference, screen, raw)
    actual, pullback = native.linearize(screen, raw, 4)
    _assert_tree(actual, expected)
    _assert_tree(native.endpoint(screen, raw, 4), expected)
    # Repeated application must neither modify seeds nor change the saved tape.
    for number in (71, 82, 71):
        seed = _seeds(expected, number)
        saved_seed = jax.tree_util.tree_map(np.copy, seed)
        _assert_tree(pullback(seed), reference_pullback(seed))
        _assert_tree(seed, saved_seed, tolerance=0.)


def test_native_pullback_matches_directional_finite_difference():
    native = _native()
    screen, raw = _fixture(13, 129)
    endpoint, pullback = native.linearize(screen, raw, 4)
    seed = _seeds(endpoint)
    direction = _seeds((screen, raw), 33)
    bars = pullback(seed)
    analytic = sum(np.sum(b * d) for b, d in zip(
        jax.tree_util.tree_leaves(bars), jax.tree_util.tree_leaves(direction)))

    def value(step):
        changed = jax.tree_util.tree_map(lambda x, dx: x + step * dx, (screen, raw), direction)
        result = native.endpoint(*changed, 4)
        return sum(np.sum(x * w) for x, w in zip(result, seed))

    step = 1e-4
    finite = (value(-2*step) - 8*value(-step) + 8*value(step) - value(2*step)) / (12*step)
    np.testing.assert_allclose(analytic, finite, rtol=2e-8, atol=2e-8)


@pytest.mark.parametrize('unsupported', ['order', 'float32', 'mixed', 'full_cartesian', 'occupied_batch'])
def test_unsupported_inputs_are_left_to_general_jax(unsupported):
    native = _native()
    screen, raw = _fixture(11, 5)
    order = 4
    if unsupported == 'order':
        order = 3
        raw = (*raw[:3], None)
    elif unsupported == 'float32':
        screen, raw = jax.tree_util.tree_map(lambda x: x.astype(np.float32), (screen, raw))
    elif unsupported == 'mixed':
        screen = screen._replace(virtual_coeff=screen.virtual_coeff.astype(np.complex128))
    elif unsupported == 'full_cartesian':
        raw = (*raw[:2], np.zeros((3, 3, 11, 11)), np.zeros((3, 3, 3, 11, 11)))
    else:
        screen = screen._replace(occupied_coeff=np.ones((11, 2)))
    assert not native.supports(screen, raw, order)


def test_traced_inputs_are_left_to_general_jax():
    native = _native()
    screen, raw = _fixture(11, 5)

    def check(value):
        candidate = screen._replace(virtual_coeff=value)
        assert not native.supports(candidate, raw, 4)
        return jnp.sum(value**2)

    actual = jax.grad(check)(jnp.asarray(screen.virtual_coeff))
    np.testing.assert_allclose(actual, 2 * screen.virtual_coeff, rtol=1e-14, atol=1e-14)


def _driver_linearize():
    function = getattr(_pairs, '_linearize_weak_moments', None)
    assert callable(function), 'weak replay must dispatch supported endpoint moments to native BLAS'
    return function


@pytest.mark.parametrize('case', ['order2', 'order3', 'mixed', 'float32', 'full_cartesian'])
def test_driver_fallback_preserves_values_and_all_input_response(case):
    linearize = _driver_linearize()
    screen, raw = _fixture(11, 5)
    order = 4
    if case.startswith('order'):
        order = int(case[-1])
        raw = raw[:order] + (None,) * (4 - order)
    elif case == 'mixed':
        screen = screen._replace(virtual_coeff=screen.virtual_coeff * (1 + .2j))
    elif case == 'float32':
        screen, raw = jax.tree_util.tree_map(lambda x: x.astype(np.float32), (screen, raw))
    else:
        rng = np.random.default_rng(38)
        raw = (*raw[:2], rng.normal(size=(3, 3, 11, 11)), rng.normal(size=(3, 3, 3, 11, 11)))
    expected, reference_pullback = jax.vjp(lambda s, r: _reference(s, r, order), screen, raw)
    actual, pullback = linearize(screen, raw, order)
    _assert_tree(actual, expected)
    _assert_tree(pullback(_seeds(expected)), reference_pullback(_seeds(expected)))


def test_driver_native_pullback_accepts_traced_weights():
    linearize = _driver_linearize()
    screen, raw = _fixture(17, 129)
    endpoint, pullback = linearize(screen, raw, 4)
    _, reference_pullback = jax.vjp(_reference, screen, raw)
    seeds, direction = _seeds(endpoint), _seeds(endpoint, 84)
    actual, tangent = jax.jvp(pullback, (seeds,), (direction,))
    _assert_tree(actual, reference_pullback(seeds))
    _assert_tree(tangent, reference_pullback(direction))


def test_driver_traced_primals_preserve_directional_derivative():
    linearize = _driver_linearize()
    screen, raw = _fixture(13, 7)
    direction = _seeds((screen, raw), 43)
    expected = jax.jvp(_reference, (screen, raw), direction)
    actual = jax.jvp(lambda s, r: linearize(s, r, 4)[0], (screen, raw), direction)
    _assert_tree(actual, expected)


def test_driver_new_physical_sizes_do_not_compile_more_endpoint_kernels(caplog):
    linearize = _driver_linearize()
    crop = getattr(_pairs, '_crop_weak_endpoint_cotangent', None)
    accumulate = getattr(_pairs, '_accumulate_weak_raw_bars', None)
    assert callable(crop) and callable(accumulate)

    def evaluate(rows, columns):
        screen, raw = _fixture(rows, columns)
        value, pullback = linearize(screen, raw, 4)
        seed = _seeds(value)
        padded = (seed[0],) + tuple(np.pad(x, ((0, 0),) * (x.ndim - 1) +
            ((0, (-columns) % 128),)) for x in seed[1:])
        screen_bar, raw_bar = pullback(crop(padded, value))
        accumulator = accumulate(None, raw_bar)
        accumulator = accumulate(accumulator, raw_bar)
        jax.block_until_ready((screen_bar, accumulator))

    with jax.log_compiles(True):
        # Check that compilation logging is observable before asserting its
        # absence for additional physical dimensions.
        jax.jit(lambda x: x + .37)(np.zeros(3)).block_until_ready()
        assert any('Compiling' in r.getMessage() for r in caplog.records)
        evaluate(7, 3)
        caplog.clear()
        for rows, columns in ((19, 127), (37, 128), (71, 129), (133, 259)):
            evaluate(rows, columns)
    assert not [r.getMessage() for r in caplog.records if 'Compiling' in r.getMessage()]


def test_host_raw_accumulation_preserves_traced_linear_response():
    accumulate = getattr(_pairs, '_accumulate_weak_raw_bars', None)
    assert callable(accumulate)
    left = (np.arange(6.).reshape(2, 3), None)
    right = (np.full((2, 3), .3), None)
    tangent = (np.full((2, 3), .7), None)
    actual, derivative = jax.jvp(lambda value: accumulate(left, value), (right,), (tangent,))
    _assert_tree(actual, (left[0] + right[0], None))
    _assert_tree(derivative, tangent)
