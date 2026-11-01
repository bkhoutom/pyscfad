"""Opt-in response estimates preserve real molecular derivatives and work."""

from collections import Counter
from functools import wraps
import importlib

import jax
import numpy as np
import pytest

from pyscfad import dlno_stc, gto
from pyscfad.dlno_stc.test.test_system import _mf, _controls


@pytest.mark.parametrize('scope', ['domain', 'system'])
def test_response_profile_preserves_energy_gradient_and_work(scope, monkeypatch, capsys):
    boys = importlib.import_module('pyscfad.lo.boys')
    df_vjp = importlib.import_module('pyscfad.df._cderi_vjp')
    counts = Counter()
    original_solver = boys.gen_gmres
    original_reconstruct = df_vjp._stream_nr_e2_from_cderi_source
    from pyscfad.dlno_stc import driver

    def counted(name, fun):
        @wraps(fun)
        def call(*args, **kwargs):
            counts[name] += 1
            return fun(*args, **kwargs)
        return call

    monkeypatch.setattr(boys, '_boys', counted('boys_forward', boys._boys))
    prepare_name = 'prepare_boys_group_inputs' if scope == 'domain' else 'prepare_system_inputs'
    monkeypatch.setattr(driver, prepare_name, counted('prepare', getattr(driver, prepare_name)))

    def solver(*args, **kwargs):
        solve = original_solver(*args, **kwargs)
        def counted(*args, **kwargs):
            counts['boys_solve'] += 1
            return solve(*args, **kwargs)
        return counted

    def reconstruct(*args, **kwargs):
        counts['reconstruct'] += 1
        return original_reconstruct(*args, **kwargs)

    monkeypatch.setattr(boys, 'gen_gmres', solver)
    monkeypatch.setattr(df_vjp, '_stream_nr_e2_from_cderi_source', reconstruct)
    mol = gto.Mole(atom='O 0 0 0; H 0.1 -0.75 0.57; H 0 0.8 0.61',
                   basis='sto-3g', verbose=0, max_memory=3000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mol.incore_anyway = True
    options = {'scope': scope}
    if scope == 'domain':
        from pyscfad.dlno_stc._selection import build_domain_groups
        static = dlno_stc.build_domain_selections(_mf(mol), frozen=1)
        options['static'] = static
        group_count = len(build_domain_groups(static))
    else:
        options['frozen'] = 1
        group_count = 1
    values, work = [], []
    for enabled in ('0', '1'):
        monkeypatch.setenv('PYSCFAD_DLNO_RESOURCE_PROFILE', enabled)
        counts.clear()
        values.append(dlno_stc.value_and_grad(mol, _mf, controls=_controls(), **options))
        jax.block_until_ready(values[-1])
        work.append(counts.copy())
        output = capsys.readouterr().out
        if enabled == '0':
            assert '[DLNO-RESOURCE]' not in output
    np.testing.assert_allclose(values[1][0], values[0][0], atol=2e-10, rtol=0)
    np.testing.assert_allclose(values[1][1].coords, values[0][1].coords,
                               atol=2e-8, rtol=0)
    assert work[0] == work[1]
    assert work[1]['boys_solve'] == 1
    assert work[1]['prepare'] == group_count
    events = [dict(field.split('=', 1) for field in line.split() if '=' in field)
              for line in output.splitlines() if line.startswith('[DLNO-RESOURCE] rank=')]
    phases = Counter(event['phase'] for event in events)
    # Component estimates belong to actual reverse execution, after the native
    # energy/cotangent evaluation, never to preparation tracing.
    native_index = next(i for i, e in enumerate(events) if e['phase'] == f'stc.{scope}.native')
    assert all(i > native_index for i, e in enumerate(events)
               if e['phase'].endswith(('.boys_response_total', '.virtual_frame', '.local_fock_blocks')))
    parent = f'stc.{scope}.common_pullback' if scope == 'domain' else 'stc.system.input_pullback'
    assert phases[parent + '.boys_response_total'] == 1
    assert phases[f'stc.{scope}.input_pullback.virtual_frame'] == group_count
    assert phases[f'stc.{scope}.input_pullback.local_fock_blocks'] == group_count
    assert phases[f'stc.{scope}.scf_response'] == 1
    if scope == 'domain':
        for component in ('pao', 'overlap', 'ao_fock'):
            assert phases[parent + '.' + component] == 1
        assert all('group' in e for e in events if e['phase'].endswith('.virtual_frame'))
    else:
        state, = [e for e in events if e['phase'] == 'stc.system.fock_state']
        assert state['fock_from_scf'] == 'True'
        assert state['degeneracy_fallback'] == 'False'


def test_traced_reverse_has_no_runtime_estimate(monkeypatch, capsys):
    from pyscfad.dlno_stc._profile import response

    monkeypatch.setenv('PYSCFAD_DLNO_RESOURCE_PROFILE', '1')
    derivative = jax.jit(jax.grad(lambda x: response('traced', lambda x: x * x, x)))
    assert float(derivative(3.0)) == 6.0
    assert '[DLNO-RESOURCE]' not in capsys.readouterr().out
