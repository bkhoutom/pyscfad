"""Auxiliary SVD truncation, anchored response, and factorization lifetime."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import _aux_compress as aux


def unpack(matrix, nmo):
    rows, cols = np.tril_indices(nmo)
    values = matrix / np.sqrt(np.where(rows == cols, 1., 2.))
    factors = np.zeros((matrix.shape[0], nmo, nmo))
    factors[:, rows, cols] = values
    factors[:, cols, rows] = values
    return factors


def packed(factors):
    rows, cols = np.tril_indices(factors.shape[1])
    return factors[:, rows, cols] * np.sqrt(np.where(rows == cols, 1., 2.))


def example(spectrum, naux=None, nmo=3, seed=8):
    rng = np.random.default_rng(seed)
    naux = len(spectrum) if naux is None else naux
    npair = nmo * (nmo + 1) // 2
    left = np.linalg.qr(rng.normal(size=(naux, len(spectrum))))[0]
    right = np.linalg.qr(rng.normal(size=(npair, len(spectrum))))[0]
    return unpack((left * spectrum) @ right.T, nmo)


@pytest.mark.parametrize('tol', [None, 0, 0., np.float64(0)])
def test_disabled_identity(tol):
    factors = jnp.asarray(example([3., 1., .1]))
    compressor = aux.make_compressor(tol)
    result, pullback = jax.vjp(compressor, factors)
    np.testing.assert_array_equal(result, factors)
    cotangent = np.arange(factors.size, dtype=float).reshape(factors.shape)
    np.testing.assert_array_equal(pullback(cotangent)[0], cotangent)
    assert aux.normalize_tolerance(tol) is None


@pytest.mark.parametrize('tol', [-1., np.nan, np.inf, -np.inf, 1j, '1e-3', [1.], True])
def test_invalid_tolerance(tol):
    with pytest.raises((TypeError, ValueError), match='lis_aux_svd_tol.*tolerance'):
        aux.make_compressor(tol)


def test_weighted_packing_and_reference_eris():
    factors = example([3., 2., .2, .1])
    compressor = aux.make_compressor(.5)
    result = np.asarray(compressor(jnp.asarray(factors)))
    _, singular, right = np.linalg.svd(packed(factors), full_matrices=False)
    reference = unpack(singular[:2, None] * right[:2], 3)
    np.testing.assert_allclose(result, reference, atol=3e-14)
    np.testing.assert_allclose(np.einsum('Lpq,Lrs->pqrs', result, result),
                               np.einsum('Lpq,Lrs->pqrs', reference, reference),
                               atol=3e-14)
    np.testing.assert_allclose(np.linalg.norm(packed(factors)), np.linalg.norm(factors))
    assert compressor.selection.rank == 2
    assert compressor.selection.anchor.shape == (4, 2)
    assert compressor.diagnostics['original_naux'] == 4
    assert compressor.diagnostics['retained_naux'] == 2
    assert compressor.diagnostics['discarded_weight'] == pytest.approx(.05)
    assert compressor.diagnostics['boundary_gap'] == pytest.approx(4. - .04)
    assert all(np.ndim(value) == 0 for value in compressor.diagnostics.values())


def test_full_rank_identity_and_all_discarded():
    factors = jnp.asarray(example([3., 2., 1.]))
    result, pullback = jax.vjp(aux.make_compressor(.1), factors)
    np.testing.assert_array_equal(result, factors)
    cotangent = np.arange(factors.size, dtype=float).reshape(factors.shape)
    np.testing.assert_array_equal(pullback(cotangent)[0], cotangent)
    with pytest.raises(ValueError, match='every auxiliary direction'):
        aux.make_compressor(10.)(factors)


@pytest.mark.parametrize('offset', [0., .025])
@pytest.mark.parametrize('spectrum,naux', [([3., 3., .3, .3], 4),
                                           ([3., 2., 1., .4, .2, .1], 9),
                                           ([3., 2., 1., .4, .2, .1], 6)])
def test_fixed_anchor_directional_derivative(spectrum, naux, offset):
    factors = example(spectrum, naux=naux)
    initial = aux.make_compressor(.8)
    initial(jnp.asarray(factors))
    rng = np.random.default_rng(17)
    # Replay away from the anchor: normalization response must be included.
    factors = factors + offset * rng.normal(size=factors.shape)
    direction = rng.normal(size=factors.shape)
    compressor = aux.make_compressor(.8, selection=initial.selection)
    result, pullback = jax.vjp(compressor, jnp.asarray(factors))
    cotangent = rng.normal(size=result.shape)
    analytic = np.vdot(pullback(cotangent)[0], direction)
    step = 2e-6
    plus = compressor(jnp.asarray(factors + step * direction))
    minus = compressor(jnp.asarray(factors - step * direction))
    numeric = np.vdot(cotangent, (plus - minus) / (2 * step))
    np.testing.assert_allclose(analytic, numeric, atol=2e-8, rtol=2e-7)


def test_auxiliary_basis_rotation_preserves_compressed_eris():
    factors = example([3., 2., .2, .1])
    rotation = np.linalg.qr(np.random.default_rng(22).normal(size=(4, 4)))[0]
    first = aux.make_compressor(.5)(jnp.asarray(factors))
    second = aux.make_compressor(.5)(jnp.asarray(np.einsum('LK,Kpq->Lpq', rotation, factors)))
    np.testing.assert_allclose(np.einsum('Lpq,Lrs->pqrs', first, first),
                               np.einsum('Lpq,Lrs->pqrs', second, second), atol=3e-14)


@pytest.mark.parametrize('tolerance', [.5, 100.])
def test_compression_uses_dense_blas_budget_and_restores_baseline(monkeypatch, tolerance):
    from threadpoolctl import threadpool_info, threadpool_limits

    factors = jnp.asarray(example([3., 2., .2, .1]))
    observed = []
    factorize = np.linalg.svd

    def recording_svd(*args, **kwargs):
        observed.extend((pool['num_threads'], pool.get('threading_layer'))
                        for pool in threadpool_info()
                        if pool['user_api'] == 'blas')
        return factorize(*args, **kwargs)

    monkeypatch.setenv('PYSCFAD_DENSE_BLAS_THREADS', '2')
    monkeypatch.setattr(np.linalg, 'svd', recording_svd)
    with threadpool_limits(limits=1, user_api='blas'):
        if tolerance > 9.:
            with pytest.raises(ValueError, match='every auxiliary direction'):
                aux.make_compressor(tolerance)(factors)
        else:
            aux.make_compressor(tolerance)(factors)
        # Serial BLAS builds cannot change their thread count. Thread-capable
        # pools must still honor the requested budget, and all restore below.
        assert observed and all(count == (1 if layer == 'disabled' else 2)
                                for count, layer in observed)
        assert {pool['num_threads'] for pool in threadpool_info()
                if pool['user_api'] == 'blas'} == {1}


def test_one_forward_svd_and_no_pullback_factorization(monkeypatch):
    factors = jnp.asarray(example([3., 2., .2, .1]))
    original_svd = np.linalg.svd
    calls = []

    def counted_svd(*args, **kwargs):
        calls.append(args[0].shape)
        return original_svd(*args, **kwargs)

    monkeypatch.setattr(np.linalg, 'svd', counted_svd)
    compressor = aux.make_compressor(.5)
    # Match the custom-VJP correction's eager inner fragment response.
    @jax.custom_vjp
    def solve(value):
        return jnp.sum(compressor(value) ** 2)

    def forward(value):
        return jax.vjp(lambda x: jnp.sum(compressor(x) ** 2), value)

    def backward(inner, cotangent):
        return inner(cotangent)

    solve.defvjp(forward, backward)
    value, pullback = jax.vjp(solve, factors)
    assert np.isfinite(value)
    assert calls == [(4, 6)]

    def forbidden(*args, **kwargs):
        raise AssertionError('factorization in pullback')

    monkeypatch.setattr(np.linalg, 'svd', forbidden)
    monkeypatch.setattr(np.linalg, 'eigh', forbidden)
    monkeypatch.setattr(np.linalg, 'cholesky', forbidden)
    monkeypatch.setattr(aux, 'solve_triangular', forbidden)
    gradient = pullback(jnp.asarray(1.))[0]
    assert np.isfinite(gradient).all()
    assert calls == [(4, 6)]


def test_strict_cutoff_and_fixed_rank_replay():
    factors = jnp.asarray(unpack(np.diag([3., 2., 1.]), 2))
    compressor = aux.make_compressor(4.)
    assert compressor(factors).shape == (1, 2, 2)
    # Explicit replay follows the selected rank even when the cutoff changes.
    replay = aux.make_compressor(100., selection=compressor.selection)
    np.testing.assert_allclose(replay(factors), compressor(factors))


def test_fixed_selection_rejects_degenerate_boundary():
    compressor = aux.make_compressor(2.)
    compressor(jnp.asarray(unpack(np.diag([3., 2., 1.]), 2)))
    with pytest.raises(ValueError, match='boundary is degenerate'):
        compressor(jnp.asarray(unpack(np.diag([3., 1., 1.]), 2)))


def test_tall_retaining_all_thin_svd_directions():
    factors = example([3., 2., 1.], naux=5, nmo=2)
    compressor = aux.make_compressor(.1)
    compressed, pullback = jax.vjp(compressor, jnp.asarray(factors))
    np.testing.assert_allclose(np.einsum('Lpq,Lrs->pqrs', compressed, compressed),
                               np.einsum('Lpq,Lrs->pqrs', factors, factors), atol=3e-14)
    rng = np.random.default_rng(41)
    direction = rng.normal(size=factors.shape)
    cotangent = rng.normal(size=compressed.shape)
    analytic = np.vdot(pullback(cotangent)[0], direction)
    step = 2e-6
    numeric = np.vdot(cotangent, (compressor(factors + step * direction)
                                 - compressor(factors - step * direction)) / (2 * step))
    np.testing.assert_allclose(analytic, numeric, atol=2e-8, rtol=2e-7)
