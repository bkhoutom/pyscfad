"""Streaming quadrature and one-permutation VJPs against independent inputs."""
import numpy as np
import pytest
from pyscfad.dlno_stc import _stc_mp2 as native
from pyscfad.dlno_stc.smooth_triples import _run_deterministic, make_solver
from pyscfad.dlno_stc.test.test_triples_stable import fixture


@pytest.mark.parametrize('batch_size', [1, 5, 128])
@pytest.mark.parametrize('selected', [1, 2, 3])
@pytest.mark.parametrize('with_grad', [False, True])
def test_streaming_independent_prepared_tensors(selected, with_grad, batch_size):
    # Deliberately unrelated left/right slots and nonsymmetric integrals/T2:
    # packing and GEMM regrouping may not assume physical tensor identities.
    rng = np.random.default_rng(872)
    dresser = native.StableTriplesDressing(fixture(selected), full_external=True)
    shapes = [x.shape for x in dresser.prepare(.3)]
    values = [tuple(rng.normal(scale=.1, size=shape) for shape in shapes) for _ in range(3)]
    weights = [.7, -.2]
    cache = native.TriplesQuadratureCache(values[0], first=0,
                                         count=selected*6, with_grad=with_grad)
    expected = [cache.step(point, weight) for point, weight in zip(values[1:], weights)]
    left = cache.finish(values[0]) if with_grad else None
    actual = native.triples_quadrature_stream(values[0], values[1:], weights, with_grad=with_grad,
                                              block_batch_size=batch_size)
    np.testing.assert_allclose(actual['point_energies'], [x['energy'] for x in expected],
                               rtol=2e-12, atol=2e-14)
    if with_grad:
        for q, reference in enumerate([left, *expected]):
            for slot, (got, want) in enumerate(zip(actual['cotangents'][q], reference['cotangents'])):
                np.testing.assert_allclose(got, want, rtol=2e-11, atol=2e-13,
                                           err_msg=f'point {q}, slot {slot}')
    assert actual['numerator_cache_peak_bytes'] == 4**3*8*(1+with_grad)*min(batch_size, selected*6)
    if with_grad and batch_size > 1:
        assert actual['grouped_adjoint_max_terms'] > 1
        assert actual['grouped_adjoint_max_reduction'] > 4


def test_point_groups_preserve_full_response():
    inputs = fixture(selected=2)
    nodes, weights = [.1, .4, .7], [.8, -.2, .3]
    results = [_run_deterministic(inputs, nodes, weights, with_grad=True,
                                 max_memory=memory, rotate=True, trace_exponent=5.)
               for memory in (4000., 0.)]
    assert results[0][0][1]['quadrature_point_batches'] == 1
    assert results[1][0][1]['quadrature_point_batches'] == len(nodes)
    np.testing.assert_allclose(results[0][0][0], results[1][0][0], rtol=2e-12, atol=2e-14)
    for got, want in zip(results[0][1], results[1][1]):
        np.testing.assert_allclose(got, want, rtol=2e-11, atol=2e-13)


def larger_occupied_inputs():
    rng = np.random.default_rng(584)
    no, nv = 9, 2
    arrays = tuple(rng.normal(scale=.1, size=s) for s in
                   [(no,nv), (no,no,nv,nv), (no,nv,nv,nv),
                    (no,nv,no,no), (no,nv,no,nv), (no,nv)])
    return arrays + (np.diag(-np.linspace(.3, 1.2, no)), np.diag([.2, 1.]), np.eye(no)[:,:2])


def test_full_domain_cache_is_rejected_and_streaming_is_bounded():
    inputs = larger_occupied_inputs()
    dresser = native.StableTriplesDressing(inputs, full_external=True)
    zero, point = dresser.prepare(0.), dresser.prepare(.2)
    with pytest.raises(ValueError, match='at most 64 blocks'):
        native.TriplesQuadratureCache(zero, first=0, count=90, with_grad=True)
    actual = native.triples_quadrature_stream(zero, [point], [.7], with_grad=True,
                                             workspace_limit_bytes=0)
    assert actual['deterministic_blocks'] == 90
    assert actual['numerator_cache_peak_bytes'] <= 2 * 2**3 * 8
    # A retained-cache oracle is still available in bounded pieces.
    energy = 0.
    for first in range(0, 90, 30):
        cache = native.TriplesQuadratureCache(zero, first=first, count=30)
        energy += cache.step(point, .7)['energy']
    np.testing.assert_allclose(actual['energy'], energy, rtol=2e-12, atol=2e-14)


def test_streaming_requires_updated_extension(monkeypatch):
    monkeypatch.delattr(native, 'TRIPLES_QUADRATURE_STREAMING')
    with pytest.raises(RuntimeError, match='Rebuild.*streaming'):
        make_solver({'mode': 'deterministic'}, ([.3], [1.], {}), with_grad=False)


@pytest.mark.parametrize('batch_size', [0, 129])
def test_invalid_batch_size(batch_size):
    dresser = native.StableTriplesDressing(fixture(2), full_external=True)
    zero = dresser.prepare(0.)
    with pytest.raises(ValueError, match='between 1 and 128'):
        native.triples_quadrature_stream(zero, [zero], [1.], block_batch_size=batch_size)


def test_batched_large_domain_all_cotangents():
    dresser = native.StableTriplesDressing(larger_occupied_inputs(), full_external=True)
    zero, point = dresser.prepare(0.), dresser.prepare(.2)
    actual = native.triples_quadrature_stream(zero, [point], [.7], with_grad=True,
                                             block_batch_size=32)
    assert actual['occupied_block_batches'] == 3
    assert actual['occupied_block_batch_size'] == 32
    expected = [[np.zeros_like(x) for x in zero] for _ in range(2)]
    for first in range(0, 90, 30):
        cache = native.TriplesQuadratureCache(zero, first=first, count=30, with_grad=True)
        right = cache.step(point, .7)
        left = cache.finish(zero)
        for dest, result in zip(expected, [left, right]):
            for acc, value in zip(dest, result['cotangents']):
                acc += value
    for got, want in zip(actual['cotangents'], expected):
        for a, b in zip(got, want):
            np.testing.assert_allclose(a, b, rtol=2e-11, atol=2e-13)


def test_grouped_adjoint_partial_output_tile():
    rng = np.random.default_rng(874)
    no, nv = 2, 17  # nvir**2 straddles the 256-element output tile.
    arrays = tuple(rng.normal(scale=.1, size=s) for s in
                   [(no,nv), (no,no,nv,nv), (no,nv,nv,nv),
                    (no,nv,no,no), (no,nv,no,nv), (no,nv)])
    inputs = arrays + (np.diag([-.3, -1.]), np.diag(np.linspace(.2, 1., nv)), np.eye(no)[:,:1])
    dresser = native.StableTriplesDressing(inputs, full_external=True)
    shapes = [x.shape for x in dresser.prepare(0.)]
    left, right = [tuple(rng.normal(scale=.1, size=s) for s in shapes) for _ in range(2)]
    cache = native.TriplesQuadratureCache(left, first=0, count=3, with_grad=True)
    expected_right = cache.step(right, .7)
    expected_left = cache.finish(left)
    actual = native.triples_quadrature_stream(left, [right], [.7], with_grad=True)
    for result, reference in zip(actual['cotangents'], [expected_left, expected_right]):
        for got, want in zip(result, reference['cotangents']):
            np.testing.assert_allclose(got, want, rtol=2e-11, atol=2e-13)
