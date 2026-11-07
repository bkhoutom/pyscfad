"""Packed response iterations preserve the equation and CCSD DIIS policies."""
import copy

import jax
import jax.numpy as jnp
import numpy
import pytest

from pyscfad import config_update, gto, scf
from pyscfad.cc import ccsd_lambda, dfccsd
from pyscfad.tools import resource_profile


@pytest.fixture(scope='module')
def response_case():
    with config_update('pyscfad_moleintor_opt', False):
        mol = gto.Mole(atom='H 0 0 0; H 0 0 1.4; H 0 0 3.4; H 0 0 4.8',
                       basis='sto-3g', unit='Bohr', verbose=0)
        mol.build(trace_exp=False, trace_ctr_coeff=False)
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-12
        mf.kernel()
        solver = dfccsd.RCCSD(mf)
        solver.incore_complete = True
        solver.diis_space = 3
        solver.conv_tol = 1e-12
        solver.conv_tol_normt = 1e-10
        eris = solver.ao2mo()
        solver.kernel(eris=eris)
        assert solver.converged
        t1, t2 = solver.t1, solver.t2
        amp = solver.amplitudes_to_vector(t1, t2)
        rng = numpy.random.default_rng(737)
        bars = (.7, jnp.asarray(rng.normal(size=t1.shape) * .01),
                jnp.asarray(rng.normal(size=t2.shape) * .01))

        def update(vector):
            return solver.amplitudes_to_vector(*solver.update_amps(
                *solver.vector_to_amplitudes(vector), eris))

        def objective(vector):
            a, b = solver.vector_to_amplitudes(vector)
            return (bars[0] * solver.energy(a, b, eris)
                    + jnp.vdot(bars[1], a) + jnp.vdot(bars[2], b))

        jacobian = numpy.asarray(jax.jacrev(update)(amp))
        source = numpy.asarray(jax.grad(objective)(amp))
        matrix = numpy.eye(amp.size) - jacobian.T
        expected = numpy.linalg.solve(matrix, source)
        yield solver, eris, bars, matrix, source, expected


@pytest.mark.parametrize('diis,start,energy_gate', [
    (False, 0, 1e9), (True, 0, 1e9), (True, 3, 1e9), (True, 0, 0.),
])
def test_response_matches_dense_adjoint_equation(response_case, diis, start, energy_gate):
    base, eris, bars, matrix, source, expected = response_case
    solver = copy.copy(base)
    solver.diis = diis
    solver.diis_start_cycle = start
    solver.diis_start_energy_diff = energy_gate
    converged, actual, saved = ccsd_lambda.solve_response_lambda(
        solver, eris, solver.t1, solver.t2, *bars,
        max_cycle=150, tol=1e-11, return_intermediates=True)
    assert converged
    numpy.testing.assert_allclose(actual, expected, atol=5e-10, rtol=0)
    assert numpy.linalg.norm(matrix @ actual - source) < 5e-10
    assert 'tau_full' in saved


def test_response_reuses_packing_outside_iterations(response_case, monkeypatch):
    base, eris, bars, *_ = response_case
    solver = copy.copy(base)
    counts = dict(pack=0, unpack=0)
    original_pack = solver.amplitudes_to_vector
    original_unpack = solver.vector_to_amplitudes

    def pack(*args, **kwargs):
        counts['pack'] += 1
        return original_pack(*args, **kwargs)

    def unpack(*args, **kwargs):
        counts['unpack'] += 1
        return original_unpack(*args, **kwargs)

    monkeypatch.setattr(solver, 'amplitudes_to_vector', pack)
    monkeypatch.setattr(solver, 'vector_to_amplitudes', unpack)
    totals = []
    for cycles in (2, 5):
        counts.update(pack=0, unpack=0)
        ccsd_lambda.solve_response_lambda(
            solver, eris, solver.t1, solver.t2, *bars,
            max_cycle=cycles, tol=0., return_intermediates=True)
        totals.append(counts.copy())
    assert totals[0] == totals[1], 'Packing calls must not grow with lambda iterations'
    assert all(count > 0 for count in totals[0].values())


def test_response_keeps_custom_diis_hook(response_case, monkeypatch):
    base, _, _, *_ = response_case
    solver = copy.copy(base)
    vector = solver.amplitudes_to_vector(solver.t1, solver.t2)
    calls = []
    sentinel = object()

    def custom_diis(t1, t2, istep, normt, de, adiis):
        calls.append((istep, normt, de, adiis))
        return t1 + .01, t2 * .9

    monkeypatch.setattr(solver, 'run_diis', custom_diis)
    actual = ccsd_lambda._run_response_diis(solver, vector, 4, .02, sentinel)
    expected = solver.amplitudes_to_vector(solver.t1 + .01, solver.t2 * .9)
    numpy.testing.assert_array_equal(actual, expected)
    assert calls == [(4, .02, 0, sentinel)]


def test_compiled_response_action_uses_runtime_tape(response_case):
    solver, eris, *_ = response_case
    amp = solver.amplitudes_to_vector(solver.t1, solver.t2)

    def update(vector):
        return solver.amplitudes_to_vector(*solver.update_amps(
            *solver.vector_to_amplitudes(vector), eris))

    _, pullback = jax.vjp(update, amp)
    action = ccsd_lambda._compile_response_action(pullback, amp)
    seed = jnp.asarray(numpy.random.default_rng(15).normal(size=amp.shape))
    expected = pullback(seed)[0]
    numpy.testing.assert_allclose(action(pullback, seed), expected,
                                  atol=1e-12, rtol=1e-12)
    numpy.testing.assert_allclose(action(pullback, seed * .3), expected * .3,
                                  atol=1e-12, rtol=1e-12)

    # Replacing the residuals must affect the same executable. Capturing
    # saved tensors as constants would silently return the original action.
    changed = jax.tree.map(
        lambda x: x * .9 if hasattr(x, 'dtype')
        and numpy.issubdtype(x.dtype, numpy.floating) else x,
        pullback)
    changed_expected = changed(seed)[0]
    assert numpy.linalg.norm(changed_expected - expected) > 1e-6
    numpy.testing.assert_allclose(action(changed, seed), changed_expected,
                                  atol=1e-12, rtol=1e-12)


def test_response_reuses_one_compiled_action(response_case, monkeypatch):
    solver, eris, bars, *_ = response_case
    original_compile = ccsd_lambda._compile_response_action
    compilations = []
    applications = []

    def compile_action(*args):
        action = original_compile(*args)
        compilations.append(True)

        def apply(*args):
            applications.append(True)
            return action(*args)

        return apply

    monkeypatch.setattr(ccsd_lambda, '_compile_response_action', compile_action)
    events = []
    with resource_profile.capture_events(events.append):
        ccsd_lambda.solve_response_lambda(
            solver, eris, solver.t1, solver.t2, *bars,
            max_cycle=4, tol=0., return_intermediates=True)
    assert len(compilations) == 1
    assert len(applications) == 4
    phases = [event['phase'] for event in events]
    assert phases.count('ccsd_response.compile_action') == 1
    assert phases.count('ccsd_response.iteration_vjp') == 4


def test_response_action_respects_disable_jit(monkeypatch):
    vector = jnp.arange(3.)
    _, pullback = jax.vjp(lambda x: jnp.sin(x), vector)

    def unexpected_jit(*args, **kwargs):
        pytest.fail('jax.disable_jit() must bypass explicit compilation')

    with jax.disable_jit():
        monkeypatch.setattr(jax, 'jit', unexpected_jit)
        action = ccsd_lambda._compile_response_action(pullback, vector)
        numpy.testing.assert_allclose(action(pullback, vector),
                                      pullback(vector)[0], atol=1e-14)
