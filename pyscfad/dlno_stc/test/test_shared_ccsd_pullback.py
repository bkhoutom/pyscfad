"""Shared preparation response using tiny native packets and a real JAX VJP."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import ccsd
from pyscfad.dlno_stc.domain import BoysEDFrame
from pyscfad.dlno_stc.test.test_backend import native
from pyscfad.dlno_stc.test.test_full_kernel import full_fixture


@pytest.fixture
def group_case(monkeypatch):
    native()
    x = full_fixture()
    controls = ccsd.driver._driver_controls({'mode': 'deterministic', 'threads': 1})
    preparations, packets, replays, baseline_calls, events = [], [], [], [], []
    baseline_targets, replay_energy_bars = [], []

    def frame_coeff(mf, common):
        return (jnp.atleast_2d(1 + .11 * mf + .13 * common),
                jnp.atleast_2d(.7 - .17 * mf + .19 * common))

    def prepare(mf, common):
        co, cv = frame_coeff(mf, common)
        factor = 1 + .07 * mf + .03 * common + .05 * jnp.sum(co * cv)
        return {key: jnp.asarray(value) * factor
                for key, value in x.items()}

    def prepared(state, *args, return_frame=False):
        assert return_frame
        (inputs, coeff), pullback = jax.vjp(
            lambda m, c: (prepare(m, c), frame_coeff(m, c)), state['mf'], state['common'])
        calls = []
        preparations.append(calls)
        def counted(bars):
            calls.append({key: np.array(value) for key, value in bars[0].items()})
            return pullback(bars)
        frame = BoysEDFrame(*coeff, np.array([0]), np.array([0]), 0, (0,))
        return inputs, ccsd.backend.host_inputs(inputs), counted, frame

    def fragment(mf, common, static, index, matrices, group, **options):
        events.append(('correction', index))
        d, v = matrices['occupied'], matrices['virtual']
        frame = options['frame']
        co, cv = frame.occupied_coeff, frame.virtual_coeff
        return ((.013 * (index + 1) * jnp.sum(d*d) + .02 * mf,
                (.7 + mf + .09 * jnp.sum(cv)) * jnp.sum(v*v)
                + common * jnp.sum(d) + .04 * jnp.sum(co * cv),
                .011 * (index + 1) * common * jnp.sum(co**2)), {})

    def retain(bars):
        # Incoming packets are owned by their producer. Read-only arrays catch
        # accidentally using the first native packet as the group accumulator.
        snapshot = {key: value.copy() for key, value in bars.items()}
        for value in bars.values():
            value.flags.writeable = False
        packets.append((bars, snapshot))

    from pyscfad.dlno_stc.backend import solve
    reverse = ccsd.backend.pullback_matrices
    def replay(*args, **kwargs):
        replay_energy_bars.append(kwargs['energy_bar'])
        replays.append(args[1]['metadata']['fragment_id'])
        events.append(('pullback', args[1]['metadata']['fragment_id']))
        bars = reverse(*args, **kwargs)
        retain(bars)
        return bars

    monkeypatch.setattr(ccsd.driver, '_prepare', prepared)
    monkeypatch.setattr(ccsd, '_group_cc_fragments', lambda static, group: static.test_fragments)
    monkeypatch.setattr(ccsd, '_solve_fragment', fragment)
    monkeypatch.setattr(ccsd.backend, 'pullback_matrices', replay)

    def run(with_grad=True, mf=.2, common=-.1,
            matrix_kind='W', matrix_controls=None, vvvv_direct=False,
            baseline_controls=None, baseline_kind='domain', group_targets=(0, 1),
            fragments=((0, (0,)), (1, (1,))),
            cc_solver='deterministic', stc_cc_controls=None):
        matrix = controls if matrix_controls is None else matrix_controls
        baseline = controls if baseline_controls is None else baseline_controls
        group = SimpleNamespace(representative_index=0, fragment_indices=(0, 1),
                                target_indices=group_targets)
        static = SimpleNamespace(mp2_static=SimpleNamespace(), baseline_kind=baseline_kind,
                                 lis_static=SimpleNamespace(matrix_kind=matrix_kind),
                                 test_fragments=fragments, fragments=(True, True))
        state = dict(mf=jnp.asarray(mf), common=jnp.asarray(common), energy=jnp.zeros(()),
                     variance=0., mf_bar=jnp.asarray(.17), common_bar=jnp.asarray(-.23))
        rows = ccsd._run_group(state, static, group, baseline, matrix,
            with_grad=with_grad, verbose_imp=0, ccsd_t=True,

            vvvv_direct=vvvv_direct, cc_solver=cc_solver, stc_cc_controls=stc_cc_controls)
        return state, rows

    return SimpleNamespace(run=run, preparations=preparations, packets=packets,
                           replays=replays, prepare=prepare, fragment=fragment,
                           baseline_calls=baseline_calls, events=events,
                           baseline_targets=baseline_targets, replay_energy_bars=replay_energy_bars,
                           controls=controls, solve=solve)


@pytest.mark.parametrize('mode', ['deterministic', 'stochastic'])
def test_w_energy_and_all_lis_responses_share_one_pullback(group_case, mode):
    case = group_case
    matrix = dict(mode=mode, threads=1)
    if mode == 'stochastic':
        matrix.update(domain_tau=1e-100, virtual_domain_tau=1e-100,
                      pair_pilot_samples=16, min_samples=16, matrix_tolerance=.03)
    options = dict(matrix_controls=matrix)
    state, rows = case.run(**options)
    assert case.baseline_calls == []
    assert case.replays == [0, 1] and case.replay_energy_bars == [1., 1.]
    assert [r.fragment_index for r in rows] == [0, 1]
    assert len(case.preparations) == len(case.preparations[0]) == 1
    x = {k: np.asarray(v) for k, v in case.prepare(.2, -.1).items()}
    oracle = case.solve(x, dict(target_indices=(0, 1)), case.controls, with_grad=False)
    np.testing.assert_allclose(state['baseline_energy'], oracle['energy'], atol=2e-12)
    assert state['variance'] == 0.
    for bars, snapshot in case.packets:
        for key in bars: np.testing.assert_array_equal(bars[key], snapshot[key])
    actual, = case.preparations[0]
    for key in actual:
        np.testing.assert_allclose(actual[key], sum(packet[key] for _, packet in case.packets),
                                   rtol=1e-13, atol=1e-14)
    step = 1e-5
    for name, dm, dc, initial in [('mf_bar', step, 0., .17), ('common_bar', 0., step, -.23)]:
        plus, _ = case.run(**options, with_grad=False, mf=.2+dm, common=-.1+dc)
        minus, _ = case.run(**options, with_grad=False, mf=.2-dm, common=-.1-dc)
        np.testing.assert_allclose(state[name]-initial,
            (plus['energy']-minus['energy'])/(2*step), atol=2e-9, rtol=2e-7)


def test_forward_leaves_accumulated_host_cotangents_untouched(group_case):
    state, rows = group_case.run(with_grad=False)
    assert len(rows) == 2
    assert group_case.preparations == [[]]
    assert group_case.replays == []
    assert float(state['mf_bar']) == .17 and float(state['common_bar']) == -.23


@pytest.mark.parametrize('invalid', ['float32', 'missing_bar', 'extra_bar'])
def test_each_incoming_packet_is_validated_before_accumulation(group_case, monkeypatch, invalid):
    """Float64 accumulation must not hide malformed original input bars."""
    real = ccsd.backend.pullback_matrices
    def corrupt(*args, **kwargs):
        bars = real(*args, **kwargs)
        if invalid == 'float32':
            bars['foo'] = bars['foo'].astype(np.float32)
        elif invalid == 'missing_bar':
            del bars['B']
        else:
            bars['unexpected'] = np.zeros((), dtype=np.float64)
        return bars
    monkeypatch.setattr(ccsd.backend, 'pullback_matrices', corrupt)
    with pytest.raises((TypeError, ValueError)):
        group_case.run()
    assert len(group_case.preparations[0]) == 0


@pytest.mark.parametrize('direct', [False, True])
def test_group_forwards_ladder_mode_to_each_fragment(group_case, monkeypatch, direct):
    """Each physical fragment must honor its owner's requested cache policy."""
    modes = []
    original = ccsd._solve_fragment
    def observed(*args, **options):
        modes.append(options['vvvv_direct'])
        return original(*args, **options)
    monkeypatch.setattr(ccsd, '_solve_fragment', observed)
    group_case.run(with_grad=False, vvvv_direct=direct)
    assert modes == [direct, direct]


@pytest.mark.parametrize('with_grad', [False, True])
@pytest.mark.parametrize('precision', [False, True])
def test_stc_cc_interface_preserves_shared_w_energy_and_response(group_case, monkeypatch,
                                                                with_grad, precision):
    case = group_case
    reference, _ = case.run( with_grad=with_grad)
    options_seen = []
    original = ccsd._solve_fragment
    def stc_fragment(*args, **options):
        options_seen.append(options)
        values, _ = original(*args, **options)
        return values, {'ccsd': {'seed': options['stc_cc_controls']['seed']}}
    monkeypatch.setattr(ccsd, '_solve_fragment', stc_fragment)
    controls = {'seed': 71}
    if precision:
        controls.update(energy_tolerance=1e3, precision_pilot_replicas=2,
                        precision_min_replicas=2, precision_max_replicas=2)
    state, rows = case.run( with_grad=with_grad,
                           cc_solver='stc', stc_cc_controls=controls)
    assert len(options_seen) == (8 if precision else 2)
    assert all(option['cc_solver'] == 'stc' for option in options_seen)
    assert sum(option['with_grad'] for option in options_seen) == (
        (4 if precision else 2) if with_grad else 0)
    assert all(row.ccsd_method == 'stc' for row in rows)
    if precision:
        assert all(row.ccsd_diagnostics['precision']['production_replicas'] == 2 for row in rows)
    else:
        assert all(row.ccsd_diagnostics['seed'] == 71 for row in rows)
    for key in ('energy', 'baseline_energy', 'variance', 'mf_bar', 'common_bar'):
        np.testing.assert_allclose(state[key], reference[key], rtol=2e-12, atol=2e-13)
    assert case.baseline_calls == []
