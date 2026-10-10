"""Execution-phase profiling must preserve the canonical scan and its AD rules."""
from contextlib import nullcontext

import jax
import numpy
import pytest

from pyscfad import numpy as np
from pyscfad.mp import dfmp2


PROFILE_ENV = 'PYSCFAD_DLNO_RESOURCE_PROFILE'


def _inputs():
    rng = numpy.random.default_rng(46)
    lov = np.asarray(rng.normal(size=(7, 6)))
    energies = np.asarray([-1.2, -0.7, 0.2, 0.6, 1.1])
    return lov, energies


def _objective(lov, energies, with_t2):
    energy, t2 = dfmp2._contract_scan(lov, energies, 2, 3, with_t2)
    if with_t2:
        # Exercise both output cotangents, not only the energy component.
        energy = energy + np.sum(t2 * t2) * 0.03
    return energy


def _phases(capsys):
    return [line.split('phase=')[1].split()[0]
            for line in capsys.readouterr().out.splitlines()
            if 'phase=' in line]


@pytest.mark.parametrize('with_t2', [False, True])
def test_profile_scan_energy_matches_disabled(monkeypatch, capsys, with_t2):
    lov, energies = _inputs()
    monkeypatch.delenv(PROFILE_ENV, raising=False)
    expected = dfmp2._contract_scan(lov, energies, 2, 3, with_t2)
    assert _phases(capsys) == []

    monkeypatch.setenv(PROFILE_ENV, '1')
    actual = dfmp2._contract_scan(lov, energies, 2, 3, with_t2)
    assert _phases(capsys) == ['dfmp2.energy_forward']
    for actual_leaf, expected_leaf in zip(
            jax.tree_util.tree_leaves(actual),
            jax.tree_util.tree_leaves(expected)):
        numpy.testing.assert_allclose(actual_leaf, expected_leaf,
                                      atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('with_t2', [False, True])
def test_profile_scan_vjp_matches_disabled_and_waits_for_results(
        monkeypatch, capsys, with_t2):
    lov, energies = _inputs()
    objective = lambda lov_, energies_: _objective(lov_, energies_, with_t2)
    monkeypatch.delenv(PROFILE_ENV, raising=False)
    expected = jax.value_and_grad(objective, argnums=(0, 1))(lov, energies)
    assert _phases(capsys) == []

    # Record the completed trees to check the profiling boundary synchronizes
    # both the scalar/t2 primal and the Lov/orbital-energy cotangents.
    synchronized = []
    original_wait = jax.block_until_ready

    def wait(value):
        result = original_wait(value)
        synchronized.append(jax.tree_util.tree_leaves(result))
        return result

    monkeypatch.setattr(jax, 'block_until_ready', wait)
    monkeypatch.setenv(PROFILE_ENV, '1')
    with dfmp2.profile_reverse_mode():
        actual = jax.value_and_grad(objective, argnums=(0, 1))(lov, energies)
    assert _phases(capsys) == [
        'dfmp2.energy_forward', 'dfmp2.energy_cotangents'
    ]
    numpy.testing.assert_allclose(actual[0], expected[0], atol=1e-12, rtol=1e-12)
    for actual_bar, expected_bar in zip(actual[1], expected[1]):
        numpy.testing.assert_allclose(actual_bar, expected_bar,
                                      atol=1e-12, rtol=1e-12)
    assert any(any(leaf.shape == () for leaf in leaves)
               for leaves in synchronized)
    assert any([leaf.shape for leaf in leaves] == [lov.shape, energies.shape]
               for leaves in synchronized)
    assert all(not isinstance(leaf, jax.core.Tracer)
               for leaves in synchronized for leaf in leaves)


@pytest.mark.parametrize('reverse_scope', [False, True])
def test_profile_scan_does_not_report_jit_tracing(monkeypatch, capsys, reverse_scope):
    lov, energies = _inputs()
    monkeypatch.delenv(PROFILE_ENV, raising=False)
    expected = jax.value_and_grad(lambda x: _objective(x, energies, False))(lov)
    capsys.readouterr()
    monkeypatch.setenv(PROFILE_ENV, '1')
    with dfmp2.profile_reverse_mode() if reverse_scope else nullcontext():
        actual = jax.jit(jax.value_and_grad(
            lambda x: _objective(x, energies, False)))(lov)
    jax.block_until_ready(actual)
    assert _phases(capsys) == []
    for actual_leaf, expected_leaf in zip(actual, expected):
        numpy.testing.assert_allclose(actual_leaf, expected_leaf,
                                      atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('transform', ['jvp', 'forward_hessian', 'reverse_hessian'])
@pytest.mark.parametrize('reverse_scope', [False, True])
def test_profile_scan_preserves_higher_and_forward_derivatives(
        monkeypatch, capsys, transform, reverse_scope):
    lov, energies = _inputs()
    fn = lambda x: _objective(lov, x, False)
    if transform == 'jvp':
        evaluate = lambda: jax.jvp(fn, (energies,), (np.ones_like(energies),))
    elif transform == 'forward_hessian':
        evaluate = lambda: jax.hessian(fn)(energies)
    else:
        evaluate = lambda: jax.jacrev(jax.grad(fn))(energies)
    monkeypatch.delenv(PROFILE_ENV, raising=False)
    expected = evaluate()
    capsys.readouterr()
    monkeypatch.setenv(PROFILE_ENV, '1')
    with dfmp2.profile_reverse_mode() if reverse_scope else nullcontext():
        actual = evaluate()
    jax.block_until_ready(actual)
    # Nested AD supplies tracers; reporting a host trace as execution is wrong.
    assert _phases(capsys) == []
    for actual_leaf, expected_leaf in zip(jax.tree_util.tree_leaves(actual),
                                         jax.tree_util.tree_leaves(expected)):
        numpy.testing.assert_allclose(actual_leaf, expected_leaf,
                                      atol=1e-11, rtol=1e-11)


def test_profile_call_rejects_staged_closure_and_stops_sampler(monkeypatch, capsys):
    import threading

    lov, _ = _inputs()
    monkeypatch.setenv(PROFILE_ENV, '1')
    monkeypatch.setenv('PYSCFAD_DLNO_RESOURCE_SAMPLE_MS', '1')
    samplers_before = [thread for thread in threading.enumerate()
                       if thread.name == 'dlno-resource-rss']

    # The explicit argument is concrete, but the callback closes over a JIT
    # input.  Its output must not be reported as completed device execution.
    def staged(y):
        return dfmp2._profile_call('test.staged', lambda x: x + y, lov)

    result = jax.jit(staged)(lov)
    numpy.testing.assert_allclose(result, lov * 2)
    assert _phases(capsys) == []
    assert [thread for thread in threading.enumerate()
            if thread.name == 'dlno-resource-rss'] == samplers_before


@pytest.mark.parametrize('with_t2', [False, True])
def test_general_profile_preserves_direct_linearize(monkeypatch, capsys, with_t2):
    lov, energies = _inputs()
    direction = (np.ones_like(lov) * 0.2, np.ones_like(energies) * 0.1)
    fn = lambda lov_, energies_: _objective(lov_, energies_, with_t2)
    monkeypatch.delenv(PROFILE_ENV, raising=False)
    expected, expected_linear_map = jax.linearize(fn, lov, energies)
    expected_tangent = expected_linear_map(*direction)
    capsys.readouterr()

    monkeypatch.setenv(PROFILE_ENV, '1')
    actual, actual_linear_map = jax.linearize(fn, lov, energies)
    actual_tangent = actual_linear_map(*direction)
    numpy.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    numpy.testing.assert_allclose(actual_tangent, expected_tangent,
                                  atol=1e-11, rtol=1e-11)
    assert _phases(capsys) == ['dfmp2.energy_forward']


def test_reverse_profile_context_resets_after_nested_scope_and_error(
        monkeypatch, capsys):
    lov, energies = _inputs()
    fn = lambda x: _objective(lov, x, False)
    monkeypatch.setenv(PROFILE_ENV, '1')
    with dfmp2.profile_reverse_mode():
        with pytest.raises(RuntimeError, match='scope exit'):
            with dfmp2.profile_reverse_mode():
                raise RuntimeError('scope exit')
        jax.grad(fn)(energies)
    assert _phases(capsys) == [
        'dfmp2.energy_forward', 'dfmp2.energy_cotangents',
    ]
    # The inner exception must restore the outer context, and leaving the
    # outer scope must restore the general profiling behavior.
    jax.grad(fn)(energies)
    assert _phases(capsys) == ['dfmp2.energy_forward']
    _, tangent_map = jax.linearize(fn, energies)
    tangent_map(np.ones_like(energies))
    assert _phases(capsys) == ['dfmp2.energy_forward']
