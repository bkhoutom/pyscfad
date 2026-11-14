"""Cached-integral Laplace rotations, native numerator, and composed response."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad import lib
from pyscfad.dlno_stc import controls, smooth_triples
from pyscfad.dlno_stc.test._triples_dressing_reference import symmetric_exp
from pyscfad.dlno_stc.test.test_triples import triples_fixture, triples_native


def integrals(bvo, boo, bvv):
    return (jnp.einsum('Pia,Pbc->iabc', bvo, bvv),
            jnp.einsum('Pia,Pjk->iajk', bvo, boo),
            jnp.einsum('Pia,Pjb->iajb', bvo, bvo))


@pytest.mark.parametrize('degenerate', [False, True])
@pytest.mark.parametrize('node', [0., .37, 3.])
@pytest.mark.parametrize('cache_bytes', [0, 2**25])
def test_rotation_and_all_input_cotangents(degenerate, node, cache_bytes):
    rng = np.random.default_rng(293)
    no, nv = 3, 4
    foo = rng.normal(scale=.03, size=(no, no)); foo = (foo+foo.T)/2-np.eye(no)
    fvv = rng.normal(scale=.03, size=(nv, nv)); fvv = (fvv+fvv.T)/2+np.eye(nv)
    if degenerate:
        foo, fvv = -np.eye(no), np.eye(nv)
    inputs = tuple(rng.normal(size=s) for s in [(no, nv), (no, no, nv, nv),
        (no, nv, nv, nv), (no, nv, no, no), (no, nv, no, nv), (no, nv)])
    inputs += (foo, fvv, rng.normal(size=(no, 2)))

    def reference(t1, t2, ovvv, ovoo, ovov, fov, foo, fvv, u):
        o, v = symmetric_exp(foo*node/2), symmetric_exp(-fvv*node/2)
        oi, vi = symmetric_exp(-foo*node/2), symmetric_exp(fvv*node/2)
        return (o @ t1 @ v,
            jnp.einsum('iI,jJ,aA,bB,IJAB->ijab', o, o, v, v, t2),
            jnp.einsum('iI,aA,bB,cC,IABC->iabc', o, v, v, vi, ovvv),
            jnp.einsum('iI,aA,jJ,kK,IAJK->iajk', o, v, o, oi, ovoo),
            jnp.einsum('iI,aA,jJ,bB,IAJB->iajb', o, v, o, v, ovov),
            o @ fov @ v, oi @ u, o @ u)

    expected, pullback = jax.vjp(reference, *map(jnp.asarray, inputs))
    engine = triples_native().TriplesDressing(inputs, integral_inputs=True)
    for got, want in zip(engine.prepare(node, cache_limit_bytes=cache_bytes), expected):
        np.testing.assert_allclose(got, want, rtol=2e-12, atol=2e-12)
    seeds = tuple(rng.normal(size=x.shape) for x in expected)
    for got, want in zip(engine.pullback(node, seeds), pullback(tuple(map(jnp.asarray, seeds)))):
        np.testing.assert_allclose(got, want, rtol=2e-11, atol=5e-11)
    assert engine.cache_bytes == 0
    assert engine.cache_hits == bool(cache_bytes)


def native_inputs():
    x = triples_fixture(grouped=True)
    rng = np.random.default_rng(409)
    values = dict(t1=x['t1'], t2=x['t2'], fov=rng.normal(scale=.03, size=x['t1'].shape),
                  L=rng.normal(size=x['U'].shape), R=rng.normal(size=x['U'].shape))
    values.update(zip(('ovvv', 'ovoo', 'ovov'), map(np.asarray,
        integrals(x['Bvo'], x['Boo'], x['Bvv']))))
    return x, values


def test_native_integrals_against_independent_numerator():
    """Independent tensor directions catch hidden same-space symmetry assumptions."""
    from pyscfad.lno.ccsd_t_slow import _compute_et
    _, values = native_inputs()
    rng = np.random.default_rng(52)
    for name in ('ovvv', 'ovoo', 'ovov'):
        values[name] = rng.normal(scale=.03, size=values[name].shape)

    def reference(x):
        no, nv = x['t1'].shape
        indices = jnp.array([(a, b, c) for a in range(nv)
                             for b in range(nv) for c in range(nv)])
        return _compute_et(x['L'] @ x['R'].T, x['t1'], x['t2'],
            x['ovvv'].transpose(1, 2, 0, 3), x['ovoo'].transpose(1, 0, 2, 3),
            x['ovov'].transpose(1, 3, 0, 2), x['fov'].T,
            -jnp.ones((no, no, no)), jnp.zeros(nv), indices, jnp.full(len(indices), 2.))

    expected, bars = jax.value_and_grad(reference)(jax.tree.map(jnp.asarray, values))
    result = triples_native().triples_asymmetric_integrals_energy_vjp(**values, nsamples=0)
    np.testing.assert_allclose(result['energy'], expected, rtol=2e-12, atol=2e-15)
    for name in values:
        np.testing.assert_allclose(result['cotangents'][name], bars[name],
                                   rtol=2e-12, atol=2e-14, err_msg=name)


@pytest.mark.parametrize('samples', [0, 257])
def test_native_factor_equivalence_and_strided_integrals(samples):
    native = triples_native()
    x, values = native_inputs()
    factors = {k: values[k] for k in ('t1', 't2', 'fov', 'L', 'R')}
    factors.update({k: x[k] for k in ('Bvo', 'Boo', 'Bvv')})
    options = dict(nsamples=samples, seed=19)
    expected = native.triples_asymmetric_energy_vjp(**factors, **options)
    actual = native.triples_asymmetric_integrals_energy_vjp(**values, **options)
    for key in ('energy', 'standard_error', 'samples'):
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-12, atol=1e-15)
    for key in ('t1', 't2', 'fov', 'L', 'R'):
        np.testing.assert_allclose(actual['cotangents'][key], expected['cotangents'][key],
                                   rtol=2e-12, atol=2e-14)
    _, pullback = jax.vjp(integrals, *[jnp.asarray(x[k]) for k in ('Bvo', 'Boo', 'Bvv')])
    bars = pullback(tuple(jnp.asarray(actual['cotangents'][k]) for k in ('ovvv', 'ovoo', 'ovov')))
    for got, name in zip(bars, ('Bvo', 'Boo', 'Bvv')):
        np.testing.assert_allclose(got, expected['cotangents'][name], rtol=2e-12, atol=2e-14)
    strided = {}
    for name, value in values.items():
        storage = np.zeros(tuple(2*n for n in value.shape))
        view = storage[tuple(slice(None, None, -2) for _ in value.shape)]
        view[...] = value
        strided[name] = view
    result = native.triples_asymmetric_integrals_energy_vjp(**strided, **options)
    assert result['energy'] == actual['energy']
    for name in values:
        np.testing.assert_array_equal(result['cotangents'][name], actual['cotangents'][name])


@pytest.mark.parametrize('name', ['ovvv', 'ovoo', 'ovov'])
@pytest.mark.parametrize('invalid', ['dtype', 'nan', 'rank', 'shape'])
def test_native_integral_input_contract(name, invalid):
    _, values = native_inputs()
    if invalid == 'dtype':
        values[name] = values[name].astype(np.float32)
    elif invalid == 'nan':
        values[name] = np.full_like(values[name], np.nan)
    elif invalid == 'rank':
        values[name] = values[name].ravel()
    else:
        values[name] = values[name][:-1]
    with pytest.raises((ValueError, TypeError), match=name):
        triples_native().triples_asymmetric_integrals_energy_vjp(**values, nsamples=2)


@pytest.mark.parametrize('mode', ['deterministic', 'fixed', 'adaptive'])
@pytest.mark.parametrize('whole', [False, True])
@pytest.mark.parametrize('canonical_virtual', [False, True])
def test_solver_energy_and_full_response_match_factor_path(mode, whole, canonical_virtual, monkeypatch):
    native_calls = []
    native = smooth_triples._native

    def counted_native(*args, **kwargs):
        native_calls.append(kwargs['with_grad'])
        return native(*args, **kwargs)

    monkeypatch.setattr(smooth_triples, '_native', counted_native)
    x = jax.tree.map(jnp.asarray, triples_fixture(grouped=True))
    rng = np.random.default_rng(72)
    no, nv = x['t1'].shape
    foo = rng.normal(scale=.02, size=(no, no)); foo = (foo+foo.T)/2+np.diag(x.pop('eocc'))
    fvv = rng.normal(scale=.02, size=(nv, nv)); fvv = (fvv+fvv.T)/2+np.diag(x.pop('evir'))
    if canonical_virtual:
        fvv = np.diag(np.diag(fvv))
    x.update(foo=jnp.asarray(foo), fvv=jnp.asarray(fvv),
             fov=jnp.asarray(rng.normal(scale=.03, size=(no, nv))))
    if whole:
        x['U'] = jnp.eye(no)
    grid = (np.array([.1, .6]), np.array([.2, .8]), {})
    settings = dict(mode='deterministic') if mode == 'deterministic' else dict(
        mode='stochastic', nsamples=256, pilot_samples=64, global_seed=71)
    if mode == 'adaptive':
        settings['target_standard_error'] = 1e-3

    def evaluate(x, transform, with_grad):
        solver = smooth_triples.make_solver(dict(settings, integral_transform=transform),
                                            grid, with_grad=with_grad)
        # Match packed DF symmetry in both paths, including its pullback.
        bvv = lib.unpack_tril(lib.pack_tril(x['Bvv']))
        ovvv, ovoo, ovov = integrals(x['Bvo'], x['Boo'], bvv)
        eris = SimpleNamespace(ovvv=lib.pack_tril(ovvv.reshape(no*nv, nv, nv)).reshape(no, nv, -1),
            ovoo=ovoo, ovov=ovov, fock=jnp.block([[x['foo'], x['fov']], [x['fov'].T, x['fvv']]]))
        # Rotation must work without any B factors in the ERI object.
        if transform == 'df':
            eris.Lov, eris.Loo, eris.Lvv = x['Bvo'], x['Boo'], lib.pack_tril(bvv)
        return solver(None, eris, x['U'].T, t1=x['t1'], t2=x['t2'])

    for with_grad in (False, True):
        if with_grad:
            (expected, reference_diag), reference_bar = jax.value_and_grad(
                lambda x: evaluate(x, 'df', True), has_aux=True)(x)
            (actual, diagnostics), bar = jax.value_and_grad(
                lambda x: evaluate(x, 'rotate', True), has_aux=True)(x)
            for key in x:
                np.testing.assert_allclose(bar[key], reference_bar[key], rtol=2e-10,
                                           atol=2e-13, err_msg=key)
        else:
            expected, reference_diag = evaluate(x, 'df', False)
            actual, diagnostics = evaluate(x, 'rotate', False)
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-14)
        np.testing.assert_allclose(diagnostics['standard_error'], reference_diag['standard_error'],
                                   rtol=2e-11, atol=2e-14)
        assert diagnostics['samples'] == reference_diag['samples']
        assert diagnostics['diagonal_virtual_points'] == (len(grid[0]) if canonical_virtual else 0)
        assert reference_diag['diagonal_virtual_points'] == diagnostics['diagonal_virtual_points']
        assert diagnostics['rotated_integrals']
        assert not reference_diag['rotated_integrals']
        assert diagnostics['stable_dressing'] and reference_diag['stable_dressing']
        for timing in (diagnostics, reference_diag):
            # Independent allocation pilots are energy-only in every mode.
            assert timing['pilot_reverse_seconds'] == 0
    if mode == 'fixed':
        # Two allocation pilots precede two production calls for each path.
        # The energy-only passes use no VJPs; derivative passes differentiate
        # production only, preserving upstream's elimination of pilot work.
        assert native_calls == [False]*8 + [False, False, True, True]*2
    if mode == 'deterministic':
        # Check quadrature-weighted cotangents independently of both reverse
        # implementations, including simultaneous integral and Fock changes.
        direction = jax.tree.map(
            lambda value: jnp.asarray(rng.normal(scale=.01, size=value.shape)), x)
        step = 1e-5
        plus = jax.tree.map(lambda value, delta: value + step*delta, x, direction)
        minus = jax.tree.map(lambda value, delta: value - step*delta, x, direction)
        finite_difference = (evaluate(plus, 'rotate', False)[0]
                             - evaluate(minus, 'rotate', False)[0]) / (2*step)
        response = sum(jnp.sum(bar[key] * direction[key]) for key in x)
        np.testing.assert_allclose(response, finite_difference, rtol=3e-6, atol=2e-10)


@pytest.mark.parametrize('mode', ['deterministic', 'stochastic'])
def test_integral_transform_controls(mode):
    base = dict(mode=mode)
    if mode == 'stochastic':
        base['nsamples'] = 2
    normalize = lambda x: controls.normalize_triples_controls(x, ccsd_t=True)
    assert normalize(dict(base, integral_transform='df')) == normalize(base)
    assert normalize(dict(base, integral_transform='rotate'))['integral_transform'] == 'rotate'
    for invalid in (None, True, 'unknown', []):
        with pytest.raises(ValueError, match='integral_transform'):
            normalize(dict(base, integral_transform=invalid))


def test_rotation_requires_smooth_selection_and_cached_integrals():
    from pyscfad.dlno_stc.ccsd import _allocate_triples_controls
    settings = dict(mode='deterministic', integral_transform='rotate')
    with pytest.raises(ValueError, match='smooth'):
        _allocate_triples_controls(settings, 1)
    solver = smooth_triples.make_solver(settings, (np.array([.3]), np.array([1.]), {}), with_grad=False)
    with pytest.raises(ValueError, match='cached ovvv, ovoo and ovov'):
        solver(None, SimpleNamespace(), np.eye(2), t1=np.zeros((2, 3)), t2=np.zeros((2, 2, 3, 3)))
