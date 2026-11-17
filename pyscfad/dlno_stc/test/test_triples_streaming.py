"""MRCC batch caps preserve the energy and all input cotangents."""
import numpy as np
import pytest
from pyscfad.dlno_stc import _stc_mp2 as native
from pyscfad.dlno_stc.mrcc_triples import run_mrcc
from pyscfad.dlno_stc.test.test_triples_stable import fixture


def test_batched_mrcc_matches_full_small_batch(monkeypatch):
    x = fixture(selected=2)
    answers = []
    for cap in ('0', '16'):
        monkeypatch.setenv('PYSCFAD_TRIPLES_NUMERATOR_CACHE_MB', cap)
        result, bars = run_mrcc(x, [.1, .4, .7], [.8, -.2, .3],
            with_grad=True, max_memory=2000., rotate=True)
        answers.append((result[0], bars))
        assert result[1]['occupied_block_batch_size'] <= 128
        if cap == '0':
            assert result[1]['occupied_block_batch_size'] == 1
    for a, b in zip([answers[0][0], *answers[0][1]], [answers[1][0], *answers[1][1]]):
        np.testing.assert_allclose(a, b, rtol=3e-10, atol=3e-12)


@pytest.mark.parametrize('batch', [0, 129])
def test_mrcc_rejects_invalid_batch(batch):
    engine = native.StableTriplesDressing(fixture(2), full_external=True, mrcc_partition=True)
    with pytest.raises(ValueError, match='batch'):
        native.triples_mrcc_point(engine.prepare(.3), block_batch_size=batch)


def test_removed_deterministic_apis():
    assert not hasattr(native, 'triples_quadrature_stream')
    assert not hasattr(native, 'TriplesQuadratureCache')


def larger_occupied_inputs():
    rng = np.random.default_rng(584)
    no, nv = 9, 2
    arrays = tuple(rng.normal(scale=.1, size=s) for s in
                   [(no,nv), (no,no,nv,nv), (no,nv,nv,nv),
                    (no,nv,no,no), (no,nv,no,nv), (no,nv)])
    return arrays + (np.diag(-np.linspace(.3, 1.2, no)), np.diag([.2, 1.]), np.eye(no)[:,:2])
