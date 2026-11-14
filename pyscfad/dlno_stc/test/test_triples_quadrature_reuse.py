"""One-sided quadrature reuse and fragment-safe cyclic pair contractions."""
import numpy as np
import pytest
from pyscfad.dlno_stc import _stc_mp2 as native
from pyscfad.dlno_stc.test.test_triples_stable import fixture
from pyscfad.dlno_stc.smooth_triples import _run_deterministic


def original(inputs, nodes, weights, with_grad):
    dresser = native.StableTriplesDressing(inputs)
    energy = 0.
    bars = [np.zeros_like(x) for x in inputs]
    for t, w in zip(nodes, weights):
        values = dresser.prepare(t)
        result = (native.triples_prepared_energy_vjp if with_grad else
                  native.triples_prepared_energy)(values, nsamples=0)
        energy += w * result['energy']
        if with_grad:
            part = dresser.pullback(t, tuple(w*b for b in result['cotangents']))
            for a, b in zip(bars, part):
                a += b
    return energy, bars


@pytest.mark.parametrize('selected', [1, 2, 3])
@pytest.mark.parametrize('spectrum', ['dense', 'diagonal', 'wide', 'degenerate'])
@pytest.mark.parametrize('with_grad', [False, True])
def test_reuse_against_original_full_functional(selected, spectrum, with_grad):
    inputs = list(fixture(selected, wide=spectrum=='wide', degenerate=spectrum=='degenerate'))
    if spectrum == 'diagonal':
        inputs[7] = np.diag([1.1, .3, 1.6, .7])
    nodes, weights = [0., .37, 3.17], [.2, .7, .1]
    expected, expected_bars = original(tuple(inputs), nodes, weights, with_grad)
    (actual, diagnostics), bars = _run_deterministic(tuple(inputs), nodes, weights,
        with_grad=with_grad, max_memory=4000., rotate=True, trace_exponent=5.)
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-13)
    assert diagnostics['numerator_cache_batches'] == 1
    assert diagnostics['single_pair_symmetrization']
    if with_grad:
        for i, (a, b) in enumerate(zip(bars, expected_bars)):
            np.testing.assert_allclose(a, b, rtol=2e-10, atol=2e-12, err_msg=f'input {i}')


def test_forced_batching_and_weighted_reverse(monkeypatch):
    inputs = fixture(selected=2)
    nodes, weights = [.1, .4], [.8, -.2]
    monkeypatch.setenv('PYSCFAD_TRIPLES_NUMERATOR_CACHE_MB', '0')
    (actual, diag), bars = _run_deterministic(inputs, nodes, weights,
        with_grad=True, max_memory=4000., rotate=True, trace_exponent=5.)
    expected, expected_bars = original(inputs, nodes, weights, True)
    assert diag['numerator_cache_batches'] == 12
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-13)
    for a, b in zip(bars, expected_bars):
        np.testing.assert_allclose(a, b, rtol=2e-10, atol=2e-12)


def test_cache_range_and_finalization():
    dresser = native.StableTriplesDressing(fixture(selected=1), full_external=True)
    values = dresser.prepare(0.)
    with pytest.raises(ValueError, match='batch'):
        native.TriplesQuadratureCache(values, first=0, count=0)
    cache = native.TriplesQuadratureCache(values, first=0, count=1, with_grad=True)
    cache.step(dresser.prepare(.3), .5)
    cache.finish(values)
    with pytest.raises(ValueError, match='finalized'):
        cache.step(values, 1.)


def test_requires_rebuilt_extension(monkeypatch):
    from pyscfad.dlno_stc.smooth_triples import make_solver
    monkeypatch.delattr(native, 'TRIPLES_QUADRATURE_REUSE')
    with pytest.raises(RuntimeError, match='Rebuild'):
        make_solver({'mode': 'deterministic'}, ([.3], [1.], {}), with_grad=False)
