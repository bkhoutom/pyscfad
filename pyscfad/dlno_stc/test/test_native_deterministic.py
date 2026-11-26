"""Deterministic native providers preserve the old numerical contract."""
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from pyscfad.dlno_stc import _native_w, _native_triples
from pyscfad.dlno_stc.test.test_native_w import fixture, old
from pyscfad.dlno_stc.test.test_triples_stable import fixture as triples_fixture


@pytest.mark.parametrize('degenerate', [False, True])
def test_density_and_scalar_mp2(degenerate):
    arrays, controls, seeds = fixture(degenerate)
    with threadpool_limits(1, user_api='blas'):
        ref = old.solve_matrices(*arrays, (0,2), controls, matrix_kind='D')
        expected = old.pullback_matrices(*arrays, ref['replay'], *seeds)
        scalar = old.solve_selected(*arrays, (0,2), controls, with_grad=True)
    got = _native_w.solve_matrices(*arrays, (0,2), controls, matrix_kind='D')
    bars = _native_w.pullback_matrices(*arrays, got['replay'], *seeds)
    for key in ('energy','occupied','virtual'):
        np.testing.assert_allclose(got[key], ref[key], rtol=3e-10, atol=3e-13)
    for key in bars:
        np.testing.assert_allclose(bars[key], expected[key], rtol=3e-10, atol=3e-13)
    got = _native_w.solve_selected(*arrays, (0,2), controls, with_grad=True)
    np.testing.assert_allclose(got['energy'],scalar['energy'], rtol=3e-11,atol=3e-13)
    for key in bars:
        np.testing.assert_allclose(got['cotangents'][key],scalar['cotangents'][key],rtol=3e-10,atol=3e-13)


@pytest.mark.parametrize('node',[0.,.37])
def test_triples_old_new_all_bars(node):
    inputs=triples_fixture(selected=2)
    results=[]
    with threadpool_limits(2, user_api='openmp'), threadpool_limits(1, user_api='blas'):
        for native in (old, _native_triples):
            engine=native.StableTriplesDressing(inputs,full_external=True,mrcc_partition=True)
            values=engine.prepare(node,cache_limit_bytes=100000000)
            result=native.triples_mrcc_point(values,.8,with_grad=True,block_batch_size=4)
            bars=engine.pullback(node,result['cotangents'],cache_limit_bytes=100000000)
            results.append((result['energy'],bars))
    np.testing.assert_allclose(results[0][0],results[1][0],rtol=3e-11,atol=3e-13)
    for a,b in zip(results[0][1],results[1][1]):
        np.testing.assert_allclose(a,b,rtol=3e-10,atol=3e-12)


def test_all_triples_entries_restore_thread_policy():
    import ctypes
    from threadpoolctl import ThreadpoolController,threadpool_info
    runtime=next(p.dynlib for p in ThreadpoolController().lib_controllers if p.user_api=='openmp')
    get=runtime.omp_get_dynamic;get.restype=ctypes.c_int
    set_dynamic=runtime.omp_set_dynamic;set_dynamic.argtypes=[ctypes.c_int]
    original=get()
    try:
        for policy in (0,1):
            set_dynamic(policy)
            with threadpool_limits(2):
                before=threadpool_info()
                inputs=triples_fixture(selected=2)
                engine=_native_triples.StableTriplesDressing(inputs,full_external=True,mrcc_partition=True)
                values=engine.prepare(.37)
                result=_native_triples.triples_mrcc_point(values,with_grad=True,block_batch_size=4)
                engine.pullback(.37,result['cotangents'])
                assert get()==policy
                assert threadpool_info()==before
    finally:set_dynamic(original)
