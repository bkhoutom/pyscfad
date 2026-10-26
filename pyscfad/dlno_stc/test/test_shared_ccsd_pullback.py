"""Shared preparation response using tiny native packets and a real JAX VJP."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import ccsd
from pyscfad.dlno_stc.test.test_backend import native
from pyscfad.dlno_stc.test.test_full_kernel import full_fixture


@pytest.fixture
def group_case(monkeypatch):
    native()
    x = full_fixture()
    controls = ccsd.driver._driver_controls({'mode': 'deterministic', 'threads': 1})
    group = SimpleNamespace(representative_index=0, fragment_indices=(0, 1), target_indices=(0, 1))
    static = SimpleNamespace(mp2_static=SimpleNamespace(), matrix_kind='D')
    preparations, packets, replays = [], [], []

    def prepare(mf, common):
        return {key: jnp.asarray(value) * (1 + .07 * mf + .03 * common)
                for key, value in x.items()}

    def prepared(state, *args):
        inputs, pullback = jax.vjp(prepare, state['mf'], state['common'])
        calls = []
        preparations.append(calls)
        def counted(bars):
            calls.append({key: np.array(value) for key, value in bars.items()})
            return pullback(bars)
        return inputs, ccsd.backend.host_inputs(inputs), counted

    def fragment(mf, common, static, index, matrices, group, **options):
        d, v = matrices['occupied'], matrices['virtual']
        return (.013 * (index + 1) * jnp.sum(d*d) + .02 * mf,
                (.7 + mf) * jnp.sum(v*v) + common * jnp.sum(d),
                .011 * (index + 1) * common)

    def retain(bars):
        # Incoming packets are owned by their producer. Read-only arrays catch
        # accidentally using the first native packet as the group accumulator.
        snapshot = {key: value.copy() for key, value in bars.items()}
        for value in bars.values():
            value.flags.writeable = False
        packets.append((bars, snapshot))

    solve, reverse = ccsd.backend.solve, ccsd.backend.pullback_matrices
    def baseline(*args, **kwargs):
        result = solve(*args, **kwargs)
        if kwargs['with_grad']:
            retain(result['cotangents'])
        return result

    def replay(*args, **kwargs):
        assert kwargs['energy_bar'] == 0.
        replays.append(args[1]['metadata']['fragment_id'])
        bars = reverse(*args, **kwargs)
        retain(bars)
        return bars

    monkeypatch.setattr(ccsd.driver, '_prepare', prepared)
    monkeypatch.setattr(ccsd, '_group_cc_fragments', lambda static, group: ((0, (0,)), (1, (1,))))
    monkeypatch.setattr(ccsd, '_solve_fragment', fragment)
    monkeypatch.setattr(ccsd.backend, 'solve', baseline)
    monkeypatch.setattr(ccsd.backend, 'pullback_matrices', replay)

    def run(include_baseline=True, indices=(0, 1), with_grad=True, mf=.2, common=-.1):
        state = dict(mf=jnp.asarray(mf), common=jnp.asarray(common), energy=jnp.zeros(()),
                     variance=0., mf_bar=jnp.asarray(.17), common_bar=jnp.asarray(-.23))
        rows = ccsd._run_group(state, static, group, controls, controls,
            with_grad=with_grad, verbose_imp=0, ccsd_t=True,
            include_baseline=include_baseline, fragment_indices=indices)
        return state, rows

    return SimpleNamespace(run=run, preparations=preparations, packets=packets,
                           replays=replays, prepare=prepare, fragment=fragment)


@pytest.mark.parametrize('include_baseline,indices,with_grad', [
    (True, (0, 1), True), (True, (), True), (False, (0, 1), True),
    (False, (1,), True), (False, (), True), (True, (0, 1), False),
])
def test_one_saved_preparation_pullback_per_contributing_group(group_case, include_baseline, indices, with_grad):
    case = group_case
    state, rows = case.run(include_baseline, indices, with_grad)
    contributes = include_baseline or bool(indices)
    assert len(case.preparations) == 1
    assert len(case.preparations[0]) == int(with_grad and contributes)
    assert case.replays == (list(indices) if with_grad else [])
    assert [row.fragment_index for row in rows] == list(indices)
    assert all(row.native_host_bar_norm is None for row in rows)
    assert all((row.direct_host_bar_norm > 0.) if with_grad else
               row.direct_host_bar_norm is None for row in rows)
    for bars, snapshot in case.packets:
        for key in bars:
            np.testing.assert_array_equal(bars[key], snapshot[key])
    if not with_grad:
        assert float(state['mf_bar']) == .17 and float(state['common_bar']) == -.23
        return
    if contributes:
        actual, = case.preparations[0]
        for key in actual:
            assert actual[key].dtype == np.float64
            np.testing.assert_allclose(actual[key], sum(packet[key] for _, packet in case.packets),
                                       rtol=1e-13, atol=1e-14)
    # Independent energy evaluations include direct host terms and native D/W
    # response; adding native matrix energy or dropping either path fails FD.
    step = 1e-5
    for name, dm, dc, initial in [('mf_bar', step, 0., .17), ('common_bar', 0., step, -.23)]:
        plus, _ = case.run(include_baseline, indices, False, .2+dm, -.1+dc)
        minus, _ = case.run(include_baseline, indices, False, .2-dm, -.1-dc)
        np.testing.assert_allclose(state[name] - initial, (plus['energy']-minus['energy'])/(2*step),
                                   atol=2e-9, rtol=2e-7)


@pytest.mark.parametrize('source,invalid', [
    ('baseline', 'float32'), ('baseline', 'missing_bar'), ('baseline', 'extra_bar'),
    ('baseline', 'missing_diagnostics'), ('baseline', 'float32_energy'),
    ('matrix', 'float32'), ('matrix', 'missing_bar'), ('matrix', 'extra_bar'),
])
def test_each_incoming_packet_is_validated_before_accumulation(group_case, monkeypatch, source, invalid):
    """Float64 accumulation must not hide malformed original input bars."""
    name = 'solve' if source == 'baseline' else 'pullback_matrices'
    real = getattr(ccsd.backend, name)
    def corrupt(*args, **kwargs):
        packet = real(*args, **kwargs)
        bars = packet['cotangents'] if source == 'baseline' else packet
        if invalid == 'float32':
            bars['foo'] = bars['foo'].astype(np.float32)
        elif invalid == 'missing_bar':
            del bars['B']
        elif invalid == 'extra_bar':
            bars['unexpected'] = np.zeros((), dtype=np.float64)
        elif invalid == 'missing_diagnostics':
            del packet['diagnostics']
        elif invalid == 'float32_energy':
            packet['energy'] = np.float32(packet['energy'])
        return packet
    monkeypatch.setattr(ccsd.backend, name, corrupt)
    with pytest.raises((TypeError, ValueError)):
        group_case.run(include_baseline=source == 'baseline', indices=(0,))
    assert len(group_case.preparations[0]) == 0
