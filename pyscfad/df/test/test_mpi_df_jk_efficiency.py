"""Serial DF optimizations remain effective at the MPI service boundary."""

import sys
from types import SimpleNamespace

from mpi4py import MPI
import numpy as np
import pytest
from pyscf import lib

from pyscfad.df import _df_jk_opt, mpi_df_jk
from pyscfad.df.df import DF
from pyscfad.df.mpi_df_jk import MPIDFJKExecutor, local_density_vjp, local_jk


def _synthetic_df():
    return SimpleNamespace(
        mol=SimpleNamespace(nao=3, incore_anyway=False),
        _cderi=np.random.default_rng(872).normal(size=(11, 6)),
        blockdim=2, max_memory=1000, verbose=0, stdout=sys.stdout,
    )


def _workspace_df():
    obj = DF(SimpleNamespace(nao=3, incore_anyway=False, max_memory=.1,
                             verbose=0, stdout=sys.stdout))
    obj._cderi = np.random.default_rng(873).normal(size=(11, 6))
    obj.blockdim = 12
    return obj


@pytest.mark.parametrize('workspace,expected_rows', [(None, 4), (.1, 12)])
@pytest.mark.parametrize('service_view', [False, True])
@pytest.mark.parametrize('tagged', [False, True])
def test_workspace_preserves_rank_partition_and_source(
        monkeypatch, workspace, expected_rows, service_view, tagged):
    obj = _workspace_df()
    source = obj._cderi
    saved_source = source.copy()
    if service_view:
        from pyscfad.dlno_stc._mpi_scf import _source_view
        obj = _source_view(obj, lambda: source)
    obj.jk_workspace_memory = workspace
    original_cderi = obj._cderi
    monkeypatch.setattr(lib, 'current_memory', lambda: (20., 0.))
    blocks = []
    iterate = mpi_df_jk._iter_cderi_range

    def observed(dfobj, q0, q1, blksize=None):
        blocks.append((q0, q1, blksize))
        yield from iterate(dfobj, q0, q1, blksize)

    monkeypatch.setattr(mpi_df_jk, '_iter_cderi_range', observed)
    coeff = np.array([[.7], [.2], [-.4]])
    density = coeff @ coeff.T
    tags = dict(mo_coeff=coeff, mo_occ=np.ones(1)) if tagged else {}
    parts = [local_jk(obj, density, rank=rank, size=2, **tags)
             for rank in range(2)]
    vj, vk = np.sum(parts, axis=0)
    factors = lib.unpack_tril(source)
    expected_j = np.einsum('Luv,uv,Lij->ij', factors, density, factors)
    expected_k = np.einsum('Liu,uv,Lvj->ij', factors, density, factors)
    np.testing.assert_allclose(vj, expected_j, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(vk, expected_k, atol=2e-12, rtol=2e-12)
    assert blocks == [(0, 5, expected_rows), (5, 11, expected_rows)]
    assert obj.max_memory == .1
    assert obj._cderi is original_cderi
    assert obj._get_cderi_source() is source
    np.testing.assert_array_equal(source, saved_source)


@pytest.mark.parametrize('workspace', [0, -1, float('nan'), float('inf')])
def test_forward_rejects_invalid_jk_workspace(workspace):
    obj = _workspace_df()
    obj.jk_workspace_memory = workspace
    with pytest.raises(ValueError, match='jk_workspace_memory'):
        local_jk(obj, np.eye(3))


@pytest.mark.parametrize('batch', [False, True])
@pytest.mark.parametrize('kind', ['psd', 'stale', 'indefinite', 'zero'])
def test_forward_uses_validated_thin_factors(monkeypatch, kind, batch):
    obj = _synthetic_df()
    coeff = np.array([[.7], [.2], [-.4]])
    density = coeff @ coeff.T
    if kind == 'indefinite':
        density = np.diag([1., -1., .5])
    elif kind == 'zero':
        density = np.zeros((3, 3))
    if batch:
        density = np.stack((density, .5 * density))
        coeff = np.stack((coeff, np.sqrt(.5) * coeff))
    if kind in ('stale', 'indefinite'):
        density = lib.tag_array(density, mo_coeff=2 * coeff,
                                mo_occ=np.ones(coeff.shape[:-2] + (1,)))

    executor = MPIDFJKExecutor(MPI.COMM_SELF)
    payloads = []
    execute = executor._execute_forward

    def record(dfobj, payload):
        payloads.append(payload)
        return execute(dfobj, payload)

    monkeypatch.setattr(executor, '_execute_forward', record)
    try:
        vj, vk = executor._forward(obj, density, hermi=1, with_j=True,
                                   with_k=True, direct_scf_tol=1e-13)
    finally:
        executor.close_local()

    factors = lib.unpack_tril(obj._cderi)
    dense = np.asarray(density)
    expected_j = np.einsum('Luv,...uv,Lij->...ij', factors, dense, factors)
    expected_k = np.einsum('Liu,...uv,Lvj->...ij', factors, dense, factors)
    np.testing.assert_allclose(vj, expected_j, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(vk, expected_k, atol=2e-12, rtol=2e-12)
    payload, = payloads
    if kind == 'indefinite':
        assert payload['mo_coeff'] is payload['mo_occ'] is None
    else:
        assert payload['mo_coeff'] is not None
        assert payload['mo_coeff'].shape[-1] == 1


@pytest.mark.parametrize('rank,start,stop', [(0, 0, 5), (1, 5, 11)])
def test_density_response_reuses_block_scratch_and_matches_dense(
        monkeypatch, rank, start, stop):
    obj = _synthetic_df()
    occupied = np.asfortranarray([[.7], [.2], [-.4]])
    response = np.asfortranarray([[.1], [-.5], [.3]])
    vk_bar = occupied @ response.T + response @ occupied.T
    vj_bar = -.3 * vk_bar
    buffers = []
    original = _df_jk_opt._df_vk_dm_vjp_lowrank

    def record(*args, **kwargs):
        buffers.append((args[4], args[5]))
        return original(*args, **kwargs)

    monkeypatch.setattr(_df_jk_opt, '_df_vk_dm_vjp_lowrank', record)
    actual = local_density_vjp(obj, (3, 3), vj_bar, vk_bar, rank=rank, size=2,
                               lowrank_factors=(occupied, response))
    factors = lib.unpack_tril(obj._cderi[start:stop])
    expected = (np.einsum('Luv,uv,Lij->ij', factors, vj_bar, factors)
                + np.einsum('Liu,uv,Lvj->ij', factors, vk_bar, factors))
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
    assert len(buffers) == 3
    assert all(left is buffers[0][0] and right is buffers[0][1]
               for left, right in buffers)
    assert buffers[0][0].shape == (2, 3)
