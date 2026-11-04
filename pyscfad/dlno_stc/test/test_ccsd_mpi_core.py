"""Small CC owner-scheduling checks without molecular or native solver work."""
import concurrent.futures
import threading
from types import SimpleNamespace
from typing import NamedTuple

import numpy
import pytest

from pyscfad import numpy as jnp
from pyscfad.dlno_stc import _mpi_ccsd, ccsd
from pyscfad.dlno_stc._selection import DomainGroup
from pyscfad.dlno_stc.test.test_mpi_core import _World
from pyscfad.lib import _compilation_cache


@pytest.mark.parametrize('with_grad', [False, True])
def test_cc_owners_configure_cache_before_setup_and_keep_serial_options(monkeypatch, with_grad):
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
    root_bars = None
    zero_cotangents = _mpi_ccsd.zero_term_cotangents
    triples = {'mode': 'stochastic', 'nsamples': 16}
    stochastic_cc = {'global_seed': 71}

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
                     cc_solver_options=dict(cc_solver='stc', stc_cc_controls=stochastic_cc))
        return state, static, {'mode': 'deterministic'}, {'mode': 'deterministic'}

    def build_worker(*args):
        assert threading.current_thread().name in configured, 'CC worker cache was not configured'
        return mf

    def run_group(state, selected, group, baseline, matrix, **options):
        name = threading.current_thread().name
        assert name in configured
        if name == 'rank-0':
            assert state['mf_bar'] is root_bars[0]
            assert state['common_bar'] is root_bars[1]
        assert selected is static
        assert options['vvvv_direct'] is True
        assert options['cc_solver'] == 'stc'
        assert options['stc_cc_controls'] == stochastic_cc
        assert options['triples_controls'] == triples
        assert options['fragment_indices'] == (0, 1)
        assert options['with_grad'] is with_grad
        owned[name] = group.representative_index
        state['energy'] += group.representative_index + 1.
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
    assert not any(isinstance(result, Exception) for result in results), results
    numpy.testing.assert_allclose(results[0][0], 3.5)
    assert results[0][1] == [0, 1]
    assert results[1:] == [None, None]
    assert configured == {'rank-0', 'rank-1', 'rank-2'}
    assert owned == {'rank-0': 0, 'rank-1': 1}
    assert weak == ['rank-0']
    assert sorted(allocated) == (['rank-0', 'rank-1', 'rank-2'] if with_grad else [])
