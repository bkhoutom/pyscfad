"""Shared deterministic dressing: independent slots and bounded reverse tapes."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import _stc_mp2 as native
from pyscfad.dlno_stc.test._triples_dressing_reference import symmetric_exp
from pyscfad.dlno_stc.test.test_triples_stable import fixture


def independent_dressing(inputs, node, shift):
    """Dress all 26 slots separately, including the selected occupied leg."""
    o = symmetric_exp(node * (inputs[6] - shift * jnp.eye(3)))
    v = symmetric_exp(-node * (inputs[7] - shift * jnp.eye(4)))
    matrices = (None, o, v, inputs[8].T, inputs[8].T @ o)
    bases = [(0, (1, 2)), (1, (1, 1, 2, 2)), (2, (1, 2, 2, 0)),
             (3, (1, 2, 1, 0)), (4, (1, 2, 1, 2)), (5, (1, 2)),
             (1, (1, 1, 2, 0)), (1, (0, 1, 2, 2))]
    slots = list(bases)
    left = ((2, 0), (3, 0), (3, 2), (6, 0), (6, 1), (7, 1))
    right = ((0, 0), (1, 0), (1, 1), (2, 0), (3, 0), (3, 2),
             (4, 0), (4, 2), (5, 0), (6, 0), (6, 1), (7, 1))
    for selected, projections in ((3, left), (4, right)):
        for kind, axis in projections:
            source, transforms = bases[kind]
            transforms = list(transforms)
            transforms[axis] = selected
            slots.append((source, transforms))
    values = []
    for source, transforms in slots:
        value = inputs[source]
        for axis, transform in enumerate(transforms):
            if transform:
                value = jnp.moveaxis(jnp.tensordot(matrices[transform], value,
                                                 axes=(1, axis)), 0, axis)
        values.append(value)
    return tuple(values)


def test_zero_node_needs_no_retained_transform_tape():
    """Identity transforms must not allocate a retained reverse tape."""
    inputs = fixture(selected=2)
    dresser = native.StableTriplesDressing(inputs, full_external=True)
    values = dresser.prepare(0., cache_limit_bytes=2**24)
    assert dresser.cache_bytes == 0
    assert dresser.peak_cache_bytes == 0
    for actual, source in zip(values[:8], (0, 1, 2, 3, 4, 5, 1, 1)):
        np.testing.assert_array_equal(actual, inputs[source])
    for value in values:
        assert not any(np.shares_memory(value, source) for source in inputs)


@pytest.mark.parametrize('node', [0., .37])
@pytest.mark.parametrize('spectrum', ['dense', 'diagonal', 'semicanonical', 'degenerate'])
@pytest.mark.parametrize('cache_limit', [0, 1, 2**24])
def test_independent_slot_bars_match_separately_dressed_reference(node, spectrum, cache_limit):
    """A missing shared-consumer contribution corrupts independent slot bars."""
    inputs = list(fixture(selected=2, degenerate=spectrum == 'degenerate'))
    if spectrum in ('diagonal', 'semicanonical'):
        inputs[7] = np.diag([1.1, .3, 1.6, .7])
    if spectrum == 'semicanonical':
        inputs[6] = np.diag([-.8, -1.2, -.4])
    inputs = tuple(inputs)
    shift = .5 * (np.linalg.eigvalsh(inputs[6]).max()
                  + np.linalg.eigvalsh(inputs[7]).min())
    expected, reverse = jax.vjp(lambda x: independent_dressing(x, node, shift),
                                tuple(map(jnp.asarray, inputs)))
    rng = np.random.default_rng(178)
    cotangents = tuple(rng.normal(size=x.shape) for x in expected)
    saved_cotangents = tuple(x.copy() for x in cotangents)
    saved_inputs = tuple(x.copy() for x in inputs)
    dresser = native.StableTriplesDressing(inputs, full_external=True)
    values = dresser.prepare(node, cache_limit_bytes=cache_limit)
    assert dresser.cache_bytes <= cache_limit
    for actual, want in zip(values, expected):
        np.testing.assert_allclose(actual, want, rtol=2e-12, atol=2e-13)
    # A caller owns its public outputs. Retained prefixes must remain private.
    for value in values:
        value[:] = np.nan
    actual_bars = dresser.pullback(node, cotangents, cache_limit_bytes=cache_limit)
    expected_bars, = reverse(tuple(map(jnp.asarray, cotangents)))
    for actual, want in zip(actual_bars, expected_bars):
        np.testing.assert_allclose(actual, want, rtol=3e-11, atol=3e-12)
    for actual, want in zip(cotangents, saved_cotangents):
        np.testing.assert_array_equal(actual, want)
    for actual, want in zip(inputs, saved_inputs):
        np.testing.assert_array_equal(actual, want)
    assert dresser.cache_bytes == 0
    assert dresser.peak_cache_bytes <= cache_limit
    if node == 0:
        np.testing.assert_array_equal(actual_bars[6], np.zeros_like(inputs[6]))
        np.testing.assert_array_equal(actual_bars[7], np.zeros_like(inputs[7]))


def test_shared_tape_rebuild_and_wrong_node_are_equivalent_to_replay():
    """A tape from another node must be rebuilt within the supplied cap."""
    inputs = fixture(selected=1)
    dresser = native.StableTriplesDressing(inputs, full_external=True)
    rng = np.random.default_rng(981)
    cotangents = tuple(rng.normal(size=x.shape) for x in dresser.prepare(.37))
    expected = dresser.pullback(.37, cotangents)
    dresser.prepare(.19, cache_limit_bytes=2**24)
    actual = dresser.pullback(.37, cotangents, cache_limit_bytes=2**24)
    assert dresser.cache_rebuilds == 1
    assert dresser.cache_bytes == 0
    for actual_bar, expected_bar in zip(actual, expected):
        np.testing.assert_allclose(actual_bar, expected_bar, rtol=2e-12, atol=2e-13)


@pytest.mark.parametrize('node', [0., .37])
@pytest.mark.parametrize('selected', [1, 2, 3])
def test_original_lis_selection_keeps_full_projector_and_fock_pullbacks(node, selected):
    """Slicing U=[I_S,0] must still return its dense independent cotangent."""
    inputs = list(fixture(selected=selected))
    inputs[8] = np.eye(3)[:, :selected]
    inputs = tuple(inputs)
    shift = .5*(np.linalg.eigvalsh(inputs[6]).max()
                + np.linalg.eigvalsh(inputs[7]).min())
    expected, reverse = jax.vjp(lambda x: independent_dressing(x, node, shift),
                                tuple(map(jnp.asarray, inputs)))
    rng = np.random.default_rng(991)
    cotangents = tuple(rng.normal(size=x.shape) for x in expected)
    engine = native.StableTriplesDressing(inputs, full_external=True)
    assert engine.selected_occupied_frame
    actual = engine.prepare(node, cache_limit_bytes=2**24)
    bars = engine.pullback(node, cotangents, cache_limit_bytes=2**24)
    want_bars, = reverse(tuple(map(jnp.asarray, cotangents)))
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=2e-12, atol=2e-13)
    for got, want in zip(bars, want_bars):
        np.testing.assert_allclose(got, want, rtol=3e-11, atol=3e-12)
    if selected < 3:
        assert np.linalg.norm(bars[8][selected:]) > .1


def test_selected_frame_fast_path_rejects_a_physical_off_block_component():
    inputs = list(fixture(selected=1))
    inputs[8] = np.eye(3)[:, :1]
    inputs[8][2, 0] = 1e-10
    assert not native.StableTriplesDressing(tuple(inputs), full_external=True).selected_occupied_frame
