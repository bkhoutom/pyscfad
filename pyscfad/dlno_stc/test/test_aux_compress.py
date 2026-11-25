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
    # Eigenvectors and singular vectors may choose different signs/gauges.
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
    factorize = aux._auxiliary_eigensystem

    def recording_svd(*args, **kwargs):
        observed.extend((pool['num_threads'], pool.get('threading_layer'))
                        for pool in threadpool_info()
                        if pool['user_api'] == 'blas')
        return factorize(*args, **kwargs)

    monkeypatch.setenv('PYSCFAD_DENSE_BLAS_THREADS', '2')
    monkeypatch.setattr(aux, '_auxiliary_eigensystem', recording_svd)
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


def test_one_forward_decomposition_and_no_pullback_factorization(monkeypatch):
    factors = jnp.asarray(example([3., 2., .2, .1]))
    original_svd = aux._auxiliary_eigensystem
    calls = []

    def counted_svd(*args, **kwargs):
        calls.append(args[0].shape)
        return original_svd(*args, **kwargs)

    monkeypatch.setattr(aux, '_auxiliary_eigensystem', counted_svd)
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
    monkeypatch.setattr(aux, 'eigh', forbidden)
    monkeypatch.setattr(aux, '_auxiliary_eigensystem', forbidden)
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


@pytest.mark.parametrize('layout', ['C', 'F', 'strided'])
def test_gram_storage_layout(monkeypatch, layout):
    matrix = np.random.default_rng(73).normal(size=(7, 22))
    matrix = matrix[:, ::2] if layout == 'strided' else np.array(matrix, order=layout)
    original = aux.get_blas_funcs
    calls = []

    def get_blas(name, arrays):
        routine = original(name, arrays)

        def syrk(alpha, operand, **kwargs):
            calls.append((operand.flags.f_contiguous,
                          np.shares_memory(operand, matrix), kwargs['trans']))
            gram = routine(alpha, operand, **kwargs)
            np.testing.assert_allclose(np.tril(gram), np.tril(matrix @ matrix.T),
                                       atol=2e-14, rtol=2e-14)
            return gram

        return syrk

    monkeypatch.setattr(aux, 'get_blas_funcs', get_blas)
    vectors, values, diagnostics = aux._auxiliary_eigensystem(matrix, .01)
    assert diagnostics['method'] == 'gram_eigh'
    assert calls == [(True, layout != 'strided', int(layout == 'C'))]
    assert diagnostics['gram_copy_bytes'] == (matrix.nbytes if layout == 'strided' else 0)
    np.testing.assert_allclose((vectors * values) @ vectors.T, matrix @ matrix.T,
                               atol=5e-14, rtol=5e-14)


def test_gram_matches_svd_with_fixed_anchor():
    factors = example([3., 2., .2, .1])
    matrix = packed(factors)
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    anchor = aux.AuxiliarySelection(2, left[:, :2])
    compressor = aux.make_compressor(.5, selection=anchor)
    compressed = compressor(jnp.asarray(factors))
    np.testing.assert_allclose(compressed, unpack(left[:, :2].T @ matrix, 3), atol=3e-14)
    assert compressor.diagnostics['method'] == 'gram_eigh'
    np.testing.assert_allclose(compressor.diagnostics['boundary_kept_eigenvalue'], singular[1]**2)


def test_gram_falls_back_when_cutoff_is_below_roundoff():
    # The tiny retained direction would be vulnerable to Gram cancellation.
    factors = example([3., 1e-8, 1e-10])
    _, values, diagnostics = aux._auxiliary_eigensystem(packed(factors), 1e-18)
    assert np.count_nonzero(values > 1e-18) == 2
    assert diagnostics['method'] == 'svd_cutoff_fallback'
    # Preserve the existing rejection of unresolved response boundaries.
    with pytest.raises(ValueError, match='boundary is degenerate'):
        aux.make_compressor(1e-18)(jnp.asarray(factors))


def test_gram_fixed_anchor_gradient_matches_svd(monkeypatch):
    factors = jnp.asarray(example([3., 3., .2, .1]))
    anchor = aux.make_compressor(.5)
    anchor(factors)
    cotangent = jnp.asarray(np.random.default_rng(71).normal(size=(2, 3, 3)))
    value, pullback = jax.vjp(aux.make_compressor(.5, selection=anchor.selection), factors)
    gradient = pullback(cotangent)[0]

    def svd(matrix, tol, selection=None):
        left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
        return left, singular**2, {}

    monkeypatch.setattr(aux, '_auxiliary_eigensystem', svd)
    reference, reference_pullback = jax.vjp(
        aux.make_compressor(.5, selection=anchor.selection), factors)
    np.testing.assert_allclose(value, reference, atol=3e-14)
    np.testing.assert_allclose(gradient, reference_pullback(cotangent)[0], atol=1e-12)
