"""Inverse-free triples against a six-index reference and broad spectra."""
import itertools
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscfad.dlno_stc import _stc_mp2 as native
from pyscfad.dlno_stc.test._triples_dressing_reference import symmetric_exp


def transform(tensor, matrices):
    for axis, matrix in enumerate(matrices):
        tensor = jnp.moveaxis(jnp.tensordot(matrix, tensor, axes=(1, axis)), 0, axis)
    return tensor


def reference(inputs, node):
    """Form the raw numerator first; dress its external legs only."""
    t1, t2, gv, go, gg, fov, foo, fvv, u = inputs
    w = jnp.einsum('iabf,kjcf->ijkabc', gv, t2) - jnp.einsum('iajm,mkbc->ijkabc', go, t2)
    v = jnp.einsum('iajb,kc->ijkabc', gg, t1) + jnp.einsum('ijab,kc->ijkabc', t2, fov)
    permutations = list(itertools.permutations(range(3)))
    w = sum(w.transpose((*p, *(i+3 for i in p))) for p in permutations)
    v = sum(v.transpose((*p, *(i+3 for i in p))) for p in permutations)
    z = w + .5*v
    z = (4*z + z.transpose(1,2,0,3,4,5) + z.transpose(2,0,1,3,4,5)
         - 2*z.transpose(2,1,0,3,4,5) - 2*z.transpose(0,2,1,3,4,5)
         - 2*z.transpose(1,0,2,3,4,5))
    o, v = symmetric_exp(node*foo/2), symmetric_exp(-node*fvv/2)
    left = transform(w, (u.T, o, o, v, v, v))
    right = transform(z, (u.T@symmetric_exp(node*foo), o, o, v, v, v))
    return -jnp.sum(left*right)/3


def fixture(selected=2, wide=False, degenerate=False):
    rng = np.random.default_rng(382)
    no, nv = 3, 4
    arrays = tuple(rng.normal(scale=.1, size=shape) for shape in
                   ((no,nv), (no,no,nv,nv), (no,nv,nv,nv),
                    (no,nv,no,no), (no,nv,no,nv), (no,nv)))
    eo = np.array([-11., -.6, -.35]) if wide else np.array([-1.2, -.8, -.4])
    ev = np.array([.23, 1.2, 5., 85.]) if wide else np.array([.3, .7, 1.1, 1.6])
    if degenerate:
        eo[:], ev[:] = -.7, .8
    qo = np.linalg.qr(rng.normal(size=(no,no)))[0]
    qv = np.linalg.qr(rng.normal(size=(nv,nv)))[0]
    return arrays + ((qo*eo)@qo.T, (qv*ev)@qv.T,
                     np.eye(no) if selected==no else rng.normal(size=(no,selected)))


def evaluate(inputs, node, cache=0, samples=0):
    engine = native.StableTriplesDressing(tuple(np.asarray(x) for x in inputs))
    values = engine.prepare(node, cache_limit_bytes=cache)
    result = native.triples_prepared_energy_vjp(values, nsamples=samples, seed=71)
    bars = engine.pullback(node, result['cotangents'], cache_limit_bytes=cache)
    return result, bars


@pytest.mark.parametrize('selected', [1, 2, 3])
@pytest.mark.parametrize('node', [0., .37, 3.17])
@pytest.mark.parametrize('spectrum', ['ordinary', 'degenerate', 'wide'])
def test_energy_and_all_responses_against_six_index_reference(selected, node, spectrum):
    inputs = fixture(selected, wide=spectrum=='wide', degenerate=spectrum=='degenerate')
    expected, expected_bars = jax.value_and_grad(reference)(tuple(map(jnp.asarray, inputs)), node)
    result, bars = evaluate(inputs, node, cache=2**24)
    assert result['deterministic_blocks'] == selected * (3 * 4 // 2)
    assert result['samples'] == 0 and result['standard_error'] == 0.
    assert result['proposal_bytes'] == 0 and result['vjp_grouping_seconds'] == 0.
    np.testing.assert_allclose(result['energy'], expected, rtol=3e-10, atol=3e-13)
    for i, (got, want) in enumerate(zip(bars, expected_bars)):
        assert np.isfinite(got).all()
        np.testing.assert_allclose(got, want, rtol=2e-9, atol=2e-12, err_msg=f'input {i}')


def test_deterministic_prepared_independent_tensor_directions():
    """Check every prepared input, without relying on dressing identities."""
    rng = np.random.default_rng(592)
    engine = native.StableTriplesDressing(fixture(selected=1))
    values = tuple(rng.normal(scale=.1, size=x.shape) for x in engine.prepare(.37))
    result = native.triples_prepared_energy_vjp(values, nsamples=0)
    energy = native.triples_prepared_energy(values, nsamples=0, seed=2026)
    assert result['energy'] == energy['energy']
    assert result['deterministic_workspace_bytes'] > 0
    assert energy['reverse_buffer_bytes'] == 0
    step = 1e-5
    for index, value in enumerate(values):
        direction = rng.normal(size=value.shape)
        plus, minus = list(values), list(values)
        plus[index] = value + step*direction
        minus[index] = value - step*direction
        fd = (native.triples_prepared_energy(tuple(plus), nsamples=0)['energy']
              - native.triples_prepared_energy(tuple(minus), nsamples=0)['energy'])/(2*step)
        np.testing.assert_allclose(np.sum(result['cotangents'][index]*direction), fd,
                                   rtol=2e-8, atol=2e-10, err_msg=f'prepared input {index}')


def test_selected_occupied_partitions_add_to_full_energy_and_response():
    inputs = fixture(selected=3)
    full, full_bars = evaluate(inputs, .37)
    parts = []
    for columns in (slice(0, 1), slice(1, 3)):
        partial = inputs[:-1] + (inputs[-1][:, columns],)
        parts.append(evaluate(partial, .37))
    np.testing.assert_allclose(sum(x[0]['energy'] for x in parts), full['energy'],
                               rtol=2e-12, atol=2e-14)
    for i in range(8):
        np.testing.assert_allclose(sum(x[1][i] for x in parts), full_bars[i],
                                   rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(np.concatenate([x[1][8] for x in parts], axis=1),
                               full_bars[8], rtol=2e-12, atol=2e-14)


def test_smooth_deterministic_requires_rebuilt_extension(monkeypatch):
    from pyscfad.dlno_stc.smooth_triples import make_solver
    monkeypatch.delattr(native, 'TRIPLES_DETERMINISTIC_BLOCKS')
    with pytest.raises(RuntimeError, match='Rebuild the STC native extension'):
        make_solver({'mode': 'deterministic'}, ([.3], [1.], {}), with_grad=False)


def test_smooth_requires_diagonal_virtual_extension(monkeypatch):
    from pyscfad.dlno_stc.smooth_triples import make_solver
    monkeypatch.delattr(native, 'TRIPLES_DIAGONAL_VIRTUAL_DRESSING')
    with pytest.raises(RuntimeError, match='Rebuild the STC native extension'):
        make_solver({'mode': 'deterministic'}, ([.3], [1.], {}), with_grad=False)


@pytest.mark.parametrize('spectrum', ['ordinary', 'degenerate', 'wide'])
@pytest.mark.parametrize('node', [0., .37, 3.17])
@pytest.mark.parametrize('cache', [0, 2**24])
def test_diagonal_virtual_energy_and_full_response(spectrum, node, cache):
    inputs = list(fixture(selected=2))
    # Deliberately unsorted: scalar factors must stay in the tensor's basis.
    values = dict(ordinary=[1.1, .3, 1.6, .7], degenerate=[.8]*4,
                  wide=[5., .23, 85., 1.2])[spectrum]
    inputs[7] = np.diag(values)
    engine = native.StableTriplesDressing(tuple(inputs))
    assert engine.virtual_diagonal_at(node)
    expected, expected_bars = jax.value_and_grad(reference)(tuple(map(jnp.asarray, inputs)), node)
    result = native.triples_prepared_energy_vjp(
        engine.prepare(node, cache_limit_bytes=cache), nsamples=0)
    bars = engine.pullback(node, result['cotangents'])
    np.testing.assert_allclose(result['energy'], expected, rtol=3e-10, atol=3e-13)
    for i, (got, want) in enumerate(zip(bars, expected_bars)):
        np.testing.assert_allclose(got, want, rtol=2e-9, atol=2e-12, err_msg=f'input {i}')
    if node:
        # A diagonal primal must not discard off-diagonal Fock derivatives.
        offdiag = bars[7] - np.diag(np.diag(bars[7]))
        assert np.linalg.norm(offdiag) > 1e-12


@pytest.mark.parametrize('degenerate', [False, True])
def test_diagonal_virtual_fock_finite_difference_crosses_dense_branch(degenerate):
    inputs = list(fixture(selected=1))
    inputs[7] = np.diag([.8]*4 if degenerate else [.3, .7, 1.1, 1.6])
    node = .37
    _, bars = evaluate(tuple(inputs), node)
    direction = np.zeros((4, 4)); direction[0, 1] = direction[1, 0] = 1.
    step = 1e-5
    energies = []
    for sign in (-1, 1):
        displaced = list(inputs)
        displaced[7] = inputs[7] + sign*step*direction
        engine = native.StableTriplesDressing(tuple(displaced))
        assert not engine.virtual_diagonal_at(node)
        energies.append(native.triples_prepared_energy(engine.prepare(node), nsamples=0)['energy'])
    np.testing.assert_allclose(np.sum(bars[7]*direction), (energies[1]-energies[0])/(2*step),
                               rtol=2e-7, atol=1e-11)


def test_roundoff_virtual_residual_is_bounded_per_node():
    inputs = list(fixture(selected=2))
    inputs[7] = np.diag([.3, .7, 1.1, 1.6])
    inputs[7][0, 1] = inputs[7][1, 0] = 1e-14
    engine = native.StableTriplesDressing(tuple(inputs))
    assert engine.virtual_diagonal_at(.37)
    assert not engine.virtual_diagonal_at(3.17)
    expected, expected_bars = jax.value_and_grad(reference)(tuple(map(jnp.asarray, inputs)), .37)
    actual, bars = evaluate(tuple(inputs), .37, cache=2**24)
    np.testing.assert_allclose(actual['energy'], expected, rtol=3e-12, atol=3e-14)
    for got, want in zip(bars, expected_bars):
        np.testing.assert_allclose(got, want, rtol=3e-11, atol=3e-13)
    # Basis choice must not depend on an arbitrary common energy shift.
    shifted = list(inputs)
    shifted[6] = inputs[6] + 1000*np.eye(3)
    shifted[7] = inputs[7] + 1000*np.eye(4)
    shifted_engine = native.StableTriplesDressing(tuple(shifted))
    for node in (.37, 3.17):
        assert shifted_engine.virtual_diagonal_at(node) == engine.virtual_diagonal_at(node)


def test_broad_spectrum_off_diagonal_fock_finite_differences_and_cache():
    inputs = fixture(wide=True)
    result, bars = evaluate(inputs, 3.17)
    cached, cached_bars = evaluate(inputs, 3.17, cache=2**24)
    np.testing.assert_allclose(result['energy'], cached['energy'], atol=1e-15)
    for a, b in zip(bars, cached_bars):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)
    for index in (6, 7):
        direction = np.zeros_like(inputs[index]); direction[0,1] = direction[1,0] = 1.
        h = 1e-5
        plus, minus = list(inputs), list(inputs)
        plus[index] = inputs[index] + h*direction
        minus[index] = inputs[index] - h*direction
        derivative = (evaluate(plus, 3.17)[0]['energy'] - evaluate(minus, 3.17)[0]['energy'])/(2*h)
        np.testing.assert_allclose(np.sum(bars[index]*direction), derivative, rtol=2e-6, atol=1e-11)


def test_sampling_uses_same_stream_for_energy_and_response():
    engine = native.StableTriplesDressing(fixture(wide=True))
    values = engine.prepare(3.17)
    energy = native.triples_prepared_energy(values, nsamples=1025, seed=41)
    response = native.triples_prepared_energy_vjp(values, nsamples=1025, seed=41)
    for key in ('energy', 'standard_error', 'samples'):
        assert energy[key] == response[key]
    assert all(np.isfinite(bar).all() for bar in engine.pullback(3.17, response['cotangents']))


def test_common_energy_shift_and_rebuilt_tape():
    inputs = fixture(wide=True)
    expected, bars = evaluate(inputs, 3.17)
    shifted = list(inputs)
    shifted[6] = inputs[6] + 1000*np.eye(3)
    shifted[7] = inputs[7] + 1000*np.eye(4)
    actual, shifted_bars = evaluate(shifted, 3.17)
    np.testing.assert_allclose(actual['energy'], expected['energy'], rtol=3e-11, atol=1e-14)
    for a, b in zip(bars, shifted_bars):
        np.testing.assert_allclose(a, b, rtol=3e-9, atol=1e-13)
    engine = native.StableTriplesDressing(inputs)
    prepared = engine.prepare(3.17)
    out = native.triples_prepared_energy_vjp(prepared, nsamples=0)
    rebuilt = engine.pullback(3.17, out['cotangents'], cache_limit_bytes=2**24)
    assert engine.cache_rebuilds == 1 and engine.cache_bytes == 0
    for a, b in zip(bars, rebuilt):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


def test_stochastic_energy_and_fock_response_against_exact():
    inputs = fixture(wide=True)
    engine = native.StableTriplesDressing(inputs)
    values = engine.prepare(3.17)
    exact = native.triples_prepared_energy_vjp(values, nsamples=0)
    exact_bars = engine.pullback(3.17, exact['cotangents'])
    draws, errors = [], []
    for seed in range(12):
        sample = native.triples_prepared_energy_vjp(values, nsamples=4096, seed=seed)
        bars = engine.pullback(3.17, sample['cotangents'])
        draws.append([sample['energy'], bars[6][0,1], bars[7][0,1], bars[8][0,0]])
        errors.append(sample['standard_error'])
    draws = np.asarray(draws)
    expected = np.array([exact['energy'], exact_bars[6][0,1], exact_bars[7][0,1], exact_bars[8][0,0]])
    se = draws.std(axis=0, ddof=1)/np.sqrt(len(draws))
    assert np.all(np.abs(draws.mean(axis=0)-expected) < 6*se + 1e-13)
    assert abs(draws[:,0].mean()-expected[0]) < 6*np.linalg.norm(errors)/len(errors)


def test_parallel_deterministic_matches_serial_for_independent_inputs():
    """Slab accumulation must preserve every independently varied cotangent."""
    import ctypes
    runtime = ctypes.CDLL(native.__file__)
    get_threads = runtime.omp_get_max_threads
    get_threads.restype = ctypes.c_int
    set_threads = runtime.omp_set_num_threads
    set_threads.argtypes = [ctypes.c_int]
    original = get_threads()
    rng = np.random.default_rng(809)
    engine = native.StableTriplesDressing(fixture(selected=2))
    values = tuple(rng.normal(scale=.1, size=x.shape) for x in engine.prepare(.37))
    try:
        set_threads(1)
        serial = native.triples_prepared_energy_vjp(values, nsamples=0)
        set_threads(4)
        for _ in range(3):
            parallel = native.triples_prepared_energy_vjp(values, nsamples=0)
            assert parallel['energy'] == serial['energy']
            for got, want in zip(parallel['cotangents'], serial['cotangents']):
                np.testing.assert_allclose(got, want, rtol=2e-12, atol=2e-14)
        assert 1 <= parallel['sampling_threads'] <= 4
    finally:
        set_threads(original)
