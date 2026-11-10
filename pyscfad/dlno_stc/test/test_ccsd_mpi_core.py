"""Small CC owner-scheduling checks without molecular or native solver work."""
import concurrent.futures
import threading
from types import SimpleNamespace
from typing import NamedTuple

import jax
import numpy
import pytest

from pyscfad import numpy as jnp
from pyscfad.dlno_stc import _mpi_ccsd, ccsd
from pyscfad.dlno_stc._selection import DomainGroup
from pyscfad.dlno_stc.test.test_mpi_core import _World
from pyscfad.lib import _compilation_cache


@pytest.mark.parametrize('with_grad', [False, True])
@pytest.mark.parametrize('clear_jax_caches, clear_failure', [
    (False, None), (True, None), (True, 'initialization'), (True, 'after_group')])
def test_cc_owners_configure_cache_before_setup_and_keep_serial_options(
        monkeypatch, with_grad, clear_jax_caches, clear_failure):
    class MF(NamedTuple):
        mo_coeff: object
        mo_energy: object
        mo_occ: object
        e_tot: object

    class Common(NamedTuple):
        s1e: object

    mf = MF(jnp.ones((1, 1)), jnp.ones(1), jnp.ones(1)*2, jnp.array(-1.))
    common = Common(jnp.ones((1, 1)))
    domains = SimpleNamespace(fragments=(object(), object()))
    static = SimpleNamespace(mp2_static=domains)
    groups = (DomainGroup((0,), 0, (0,)), DomainGroup((1,), 1, (0,)))
    configured, owned, weak, allocated = set(), {}, [], []
    stages = {f'rank-{rank}': [] for rank in range(3)}
    cleared = {f'rank-{rank}': [] for rank in range(3)}
    group_live = {}
    root_bars = None
    zero_cotangents = _mpi_ccsd.zero_term_cotangents
    cache_boundary = ccsd._clear_jax_caches
    triples = {'mode': 'stochastic', 'nsamples': 16}
    stochastic_cc = {'global_seed': 71}

    def clear_boundary(stage, *live_results, **metadata):
        name = threading.current_thread().name
        assert name in configured
        if stage == 'initialization':
            assert not stages[name]
            assert len(live_results) == 2
            assert live_results[0] is mf
            numpy.testing.assert_array_equal(live_results[1].s1e, common.s1e)
            assert not metadata
        else:
            assert stage == 'after_group'
            assert stages[name] == ['initialization', 'group']
            assert len(live_results) == 3
            assert all(actual is expected
                       for actual, expected in zip(live_results, group_live[name]))
            assert metadata == {'frag': owned[name] + 1}
        stages[name].append(stage)
        return cache_boundary(stage, *live_results, **metadata)

    def clear():
        name = threading.current_thread().name
        stage = stages[name][-1]
        cleared[name].append(stage)
        failure_rank = 'rank-2' if clear_failure == 'initialization' else 'rank-1'
        if stage == clear_failure and name == failure_rank:
            raise RuntimeError('synthetic cache cleanup failure')

    def configure_cache():
        configured.add(threading.current_thread().name)

    def initialize_cotangents(*args):
        allocated.append(threading.current_thread().name)
        return zero_cotangents(*args)

    def initialize():
        nonlocal root_bars
        # Root's serial initialization already configures its cache.
        configure_cache()
        root_bars = initialize_cotangents(mf, common) if with_grad else (None, None)
        state = dict(mf=mf, common=common, weak_pairs=(),
                     mf_bar=root_bars[0], common_bar=root_bars[1],
                     cc_fragment_indices=(0, 1), cc_triples_controls=triples,
                     cc_solver_options=dict(cc_solver='stc', stc_cc_controls=stochastic_cc,
                                            lis_aux_svd_tol=1e-6))
        return state, static, {'mode': 'deterministic'}, {'mode': 'deterministic'}

    def build_worker(*args):
        assert threading.current_thread().name in configured, 'CC worker cache was not configured'
        return mf

    def run_group(state, selected, group, baseline, matrix, **options):
        name = threading.current_thread().name
        assert name in configured
        stages[name].append('group')
        if name == 'rank-0':
            assert state['mf_bar'] is root_bars[0]
            assert state['common_bar'] is root_bars[1]
        assert selected is static
        assert options['vvvv_direct'] is True
        assert options['cc_solver'] == 'stc'
        assert options['stc_cc_controls'] == stochastic_cc
        assert options['lis_aux_svd_tol'] == 1e-6
        assert options['triples_controls'] == triples
        assert options['fragment_indices'] == (0, 1)
        assert options['with_grad'] is with_grad
        owned[name] = group.representative_index
        state['energy'] += group.representative_index + 1.
        group_live[name] = (state['energy'], state['mf_bar'], state['common_bar'])
        return [group.representative_index]

    def run_weak(state, *args, **kwargs):
        weak.append(threading.current_thread().name)
        state['energy'] += .5

    monkeypatch.setattr(_compilation_cache, 'configure_compilation_cache', configure_cache)
    monkeypatch.setattr(_mpi_ccsd, 'zero_term_cotangents', initialize_cotangents)
    monkeypatch.setattr(_mpi_ccsd, 'build_domain_groups', lambda *args, **kwargs: groups)
    monkeypatch.setattr(_mpi_ccsd, '_reference_signature', lambda value: {})
    monkeypatch.setattr(_mpi_ccsd, '_build_cc_worker', build_worker)
    monkeypatch.setattr(_mpi_ccsd, '_validate_shared_reference', lambda *args, **kwargs: None)
    monkeypatch.setattr(ccsd, '_run_group', run_group)
    monkeypatch.setattr(ccsd, '_run_weak', run_weak)
    monkeypatch.setenv('PYSCFAD_STC_CLEAR_JAX_CACHES', str(int(clear_jax_caches)))
    monkeypatch.setattr(ccsd, '_clear_jax_caches', clear_boundary)
    monkeypatch.setattr(jax, 'clear_caches', clear)
    monkeypatch.setattr(ccsd, '_finish', lambda state, selected, records, **options:
                        (state['energy'], records))
    world = _World(3)

    def run(rank):
        threading.current_thread().name = f'rank-{rank}'
        try:
            return _mpi_ccsd._run_jobs(
                initialize, lambda canonical: mf, world.communicator(rank),
                with_grad=with_grad, include_hf=False, merge_identical=True,
                verbose_imp=0, ccsd_t=True, return_components=True,
                include_baseline=True, fragment_indices=None, scf_service=None,
                vvvv_direct=True)
        except Exception as exc:
            return exc

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run, range(3)))
    if clear_failure is not None:
        assert all(isinstance(result, RuntimeError) for result in results), results
        assert all('synthetic cache cleanup failure' in str(result) for result in results)
        assert not weak
        if clear_failure == 'initialization':
            assert not owned
            assert all(value == ['initialization'] for value in stages.values())
        else:
            assert owned == {'rank-0': 0, 'rank-1': 1}
        return
    assert not any(isinstance(result, Exception) for result in results), results
    numpy.testing.assert_allclose(results[0][0], 3.5)
    assert results[0][1] == [0, 1]
    assert results[1:] == [None, None]
    assert configured == {'rank-0', 'rank-1', 'rank-2'}
    assert owned == {'rank-0': 0, 'rank-1': 1}
    assert weak == ['rank-0']
    assert sorted(allocated) == (['rank-0', 'rank-1', 'rank-2'] if with_grad else [])
    assert stages == {'rank-0': ['initialization', 'group', 'after_group'],
                      'rank-1': ['initialization', 'group', 'after_group'],
                      'rank-2': ['initialization']}
    assert cleared == ({'rank-0': ['initialization', 'after_group'],
                        'rank-1': ['initialization', 'after_group'],
                        'rank-2': ['initialization']} if clear_jax_caches else
                       {'rank-0': [], 'rank-1': [], 'rank-2': []})
