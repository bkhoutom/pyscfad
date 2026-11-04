"""Serial DF optimizations remain effective at the MPI service boundary."""

import sys
from types import SimpleNamespace

from mpi4py import MPI
import numpy as np
import pytest
from pyscf import lib

from pyscfad.df import _df_jk_opt
from pyscfad.df.mpi_df_jk import MPIDFJKExecutor, local_density_vjp


def _synthetic_df():
    return SimpleNamespace(
        mol=SimpleNamespace(nao=3, incore_anyway=False),
        _cderi=np.random.default_rng(872).normal(size=(11, 6)),
        blockdim=2, max_memory=1000, verbose=0, stdout=sys.stdout,
    )


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
