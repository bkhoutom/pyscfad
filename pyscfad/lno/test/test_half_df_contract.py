import numpy as np
import pytest
from threadpoolctl import threadpool_info
from pyscfad.lno._df_half_contract import coefficient_contractions
from pyscfad.lib._threading import blas_threads


@pytest.mark.parametrize('threads', [1, 3])
@pytest.mark.parametrize('layout', ['C', 'F', 'sliced'])
@pytest.mark.parametrize('symmetric', [False, True])
def test_half_coefficient_derivative(monkeypatch, threads, layout, symmetric):
    monkeypatch.setenv('PYSCFAD_DENSE_BLAS_THREADS', str(threads))
    rng = np.random.default_rng(614)
    p, m, o, v = 7, 11, 3, 5
    if symmetric:
        v = o
    # Differentiate the underlying raw AO transform independently of H.
    raw = rng.normal(size=(p, m, m))
    raw = raw + raw.transpose(0, 2, 1)
    coeff = rng.normal(size=(m, o+v+2))
    left = slice(1, 1+o)
    right = left if symmetric else slice(1+o, 1+o+v)
    z = rng.normal(size=(p, o, v))
    if symmetric:
        z = (z + z.transpose(0, 2, 1))/2
    if layout == 'F':
        z = np.asfortranarray(z.reshape(p, -1)).reshape(p, o, v)
    elif layout == 'sliced':
        storage = np.zeros((p*2, o, v*2))
        storage[::2, :, ::2] = z
        z = storage[::2, :, ::2]
    h = raw @ coeff
    a, b = coefficient_contractions(h, z, left, right, symmetric=symmetric)
    expected_a = np.einsum('pml,pkl->mk', h[:, :, right], z)
    expected_b = np.einsum('pmk,pkl->ml', h[:, :, left], z)
    np.testing.assert_allclose(a, expected_a*(2 if symmetric else 1), atol=2e-12)
    np.testing.assert_allclose(b, 0 if symmetric else expected_b, atol=2e-12)
    bar = np.zeros_like(coeff)
    bar[:, left] += a
    if not symmetric:
        bar[:, right] += b
    direction = rng.normal(size=coeff.shape)
    def energy(c):
        return np.einsum('mi,pmn,nj,pij->', c[:, left], raw, c[:, right], z)
    eps = 1e-5
    numeric = (energy(coeff+eps*direction)-energy(coeff-eps*direction))/(2*eps)
    np.testing.assert_allclose(np.sum(bar*direction), numeric, rtol=1e-8, atol=1e-7)


def test_half_thread_scope_restored_on_worker_failure(monkeypatch):
    monkeypatch.setenv('PYSCFAD_DENSE_BLAS_THREADS', '3')
    def counts():
        return [p['num_threads'] for p in threadpool_info() if p['user_api']=='blas']
    def fail(*args, **kwargs):
        assert all(n == 1 for n in counts())
        raise RuntimeError('worker failure')
    with blas_threads(2):
        before = counts()
        monkeypatch.setattr(np, 'matmul', fail)
        with pytest.raises(RuntimeError, match='worker failure'):
            coefficient_contractions(np.ones((4, 5, 6)), np.ones((4, 2, 4)),
                                     slice(0, 2), slice(2, 6))
        assert counts() == before
