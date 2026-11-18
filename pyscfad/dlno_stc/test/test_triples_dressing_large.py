"""Stable triples at virtual dimensions that exercise the dense BLAS path."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import _stc_mp2 as native
from pyscfad.dlno_stc.mrcc_triples import run_mrcc
from pyscfad.dlno_stc.test.test_triples_stable import evaluate, reference as asymmetric_reference
from pyscfad.dlno_stc.test.test_mrcc_triples import reference


_INPUT_NAMES = ('t1', 't2', 'gv', 'go', 'gg', 'fov', 'foo', 'fvv', 'u')


def _large_fixture(selected, diagonal=False):
    """Keep the six-index oracle small while crossing the GEMM threshold of 16."""
    rng = np.random.default_rng(704)
    no, nv = 3, 17
    arrays = tuple(rng.normal(scale=.1, size=shape) for shape in
                   ((no, nv), (no, no, nv, nv), (no, nv, nv, nv),
                    (no, nv, no, no), (no, nv, no, nv), (no, nv)))
    qo = np.linalg.qr(rng.normal(size=(no, no)))[0]
    qv = np.linalg.qr(rng.normal(size=(nv, nv)))[0]
    foo = (qo * np.array([-1.2, -.8, -.4])) @ qo.T
    # An unsorted diagonal guards against silently changing the tensor basis.
    ev = np.roll(np.linspace(.2, 1.8, nv), 5)
    fvv = np.diag(ev) if diagonal else (qv * ev) @ qv.T
    u = rng.normal(size=(no, selected))
    return arrays + (foo, fvv, u)


def _assert_energy_and_bars(energy, bars, expected, expected_bars):
    np.testing.assert_allclose(energy, expected, rtol=3e-10, atol=3e-13)
    assert len(bars) == len(expected_bars) == len(_INPUT_NAMES)
    for name, got, want in zip(_INPUT_NAMES, bars, expected_bars):
        assert np.isfinite(got).all(), name
        np.testing.assert_allclose(got, want, rtol=2e-9, atol=2e-12,
                                   err_msg=f'input {name}')


@pytest.mark.parametrize('selected,diagonal', [(2, False), (1, True)],
                         ids=['dense-s2', 'diagonal-s1'])
def test_large_full_external_energy_and_all_responses(monkeypatch, selected, diagonal):
    """Check the shared full-external graph, tape replay, and diagonal response."""
    inputs = _large_fixture(selected, diagonal)
    nodes, weights = (0., .37), (.2, .7)

    def six_index_quadrature(x):
        return sum(weight * reference(x, node) for node, weight in zip(nodes, weights))

    expected, expected_bars = jax.value_and_grad(six_index_quadrature)(
        tuple(map(jnp.asarray, inputs)))
    monkeypatch.setenv('PYSCFAD_TRIPLES_NUMERATOR_CACHE_MB', '16')
    results = []
    for cap in ('0', '16'):
        monkeypatch.setenv('PYSCFAD_TRIPLES_DRESSING_CACHE_MB', cap)
        (energy, diagnostics), bars = run_mrcc(
            inputs, nodes, weights, with_grad=True, max_memory=4000.)
        if cap == '0':
            assert diagnostics['dressing_cache_peak_bytes'] == 0
        else:
            assert 0 < diagnostics['dressing_cache_peak_bytes'] <= 16 * 1024**2
        _assert_energy_and_bars(energy, bars, expected, expected_bars)
        results.append((energy, bars))

    _assert_energy_and_bars(*results[0], *results[1])
    if diagonal:
        # A scalar forward transform still needs the full matrix-exponential VJP.
        offdiag = results[1][1][7] - np.diag(np.diag(results[1][1][7]))
        assert np.linalg.norm(offdiag) > 1e-12
        direction = np.zeros_like(inputs[7])
        direction[0, -1] = direction[-1, 0] = 1.
        step = 1e-5
        energies = []
        for sign in (-1, 1):
            displaced = list(inputs)
            displaced[7] = inputs[7] + sign * step * direction
            assert not native.StableTriplesDressing(tuple(displaced)).virtual_diagonal_at(nodes[1])
            (energy, _), _ = run_mrcc(
                tuple(displaced), nodes, weights, with_grad=False,
                max_memory=4000.)
            energies.append(energy)
        np.testing.assert_allclose(np.sum(results[1][1][7] * direction),
                                   (energies[1] - energies[0]) / (2 * step),
                                   rtol=2e-7, atol=1e-10)


def test_large_half_external_energy_and_all_responses():
    """The stochastic-compatible half dressing must retain its full VJP."""
    inputs = _large_fixture(selected=1)
    node = .37
    expected, expected_bars = jax.value_and_grad(asymmetric_reference)(
        tuple(map(jnp.asarray, inputs)), node)
    result, bars = evaluate(inputs, node, cache=2**24)
    _assert_energy_and_bars(result['energy'], bars, expected, expected_bars)
