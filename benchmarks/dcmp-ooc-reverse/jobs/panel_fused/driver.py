"""Measure fused physical density construction for symmetric J inputs."""
import time

import numpy as np
from pyscf import lib
from scipy.linalg.blas import dgemm
from threadpoolctl import threadpool_info, threadpool_limits

P, NAO, RANK, REPEATS, THREADS = 17, 800, 80, 3, 16
rng = np.random.default_rng(451)
panel = rng.standard_normal((P, NAO, RANK))
S = rng.standard_normal((NAO, RANK))
G = rng.standard_normal((NAO, NAO))
G = (G + G.T) * .5
D = rng.standard_normal((NAO, NAO))
D = (D + D.T) * .5
a = rng.standard_normal(P)
b = rng.standard_normal(P)
# These depend only on the full reverse call, not on its auxiliary block.
GD = np.asfortranarray(np.stack((G.ravel(), D.ravel()), axis=1))
S2T = np.ascontiguousarray(2 * S.T)


def current(with_j, with_k):
    if with_k:
        density = (panel.reshape(-1, RANK) @ S.T).reshape(P, NAO, NAO)
        density += density.swapaxes(1, 2).copy()
    else:
        density = np.zeros((P, NAO, NAO))
    if with_j:
        density += a[:, None, None] * G
        density += b[:, None, None] * D
    density *= 2
    return density, None


def fused(with_j, with_k):
    if with_k:
        density = (panel.reshape(-1, RANK) @ S2T).reshape(P, NAO, NAO)
    else:
        density = np.zeros((P, NAO, NAO))
    shares = None
    if with_j:
        coeff = np.array((a, b), order='F')
        flat_t = density.reshape(P, -1).T
        updated = dgemm(1., GD, coeff, beta=1., c=flat_t, overwrite_c=1)
        shares = np.shares_memory(updated, density)
        # Always consume BLAS's returned buffer, even if it unexpectedly copied.
        density = updated.T.reshape(P, NAO, NAO)
    before = density
    density = lib.hermi_sum(density, axes=(0, 2, 1), inplace=True)
    assert np.shares_memory(before, density)
    return density, shares


def timed(fn, with_j, with_k):
    start = time.perf_counter()
    result = fn(with_j, with_k)
    return result, time.perf_counter() - start


with threadpool_limits(limits=THREADS, user_api='blas'):
    print(f'panel={panel.shape} S={S.shape} repeats={REPEATS} blas_threads={THREADS}', flush=True)
    print('threadpools:', threadpool_info(), flush=True)
    print('GD Fortran contiguous:', GD.flags.f_contiguous, flush=True)
    for label, with_j, with_k in [('J', True, False), ('K', False, True), ('JK', True, True)]:
        # Warm up both paths and OpenMP before timing.
        for fn in (current, fused):
            warm, _ = fn(with_j, with_k)
            del warm
        times = {'current': [], 'fused': []}
        for repeat in range(REPEATS):
            order = (current, fused) if repeat % 2 == 0 else (fused, current)
            results = {}
            for fn in order:
                (density, shares), elapsed = timed(fn, with_j, with_k)
                results[fn.__name__] = density
                times[fn.__name__].append(elapsed)
                if fn is fused:
                    print(f'{label} repeat={repeat+1} BLAS output shares input={shares}', flush=True)
                    assert np.array_equal(density, density.swapaxes(1, 2)), 'fused output not exactly symmetric'
            np.testing.assert_allclose(results['current'], results['fused'], rtol=1e-12, atol=1e-12)
            max_abs = np.max(np.abs(results['current'] - results['fused']))
            print(f'{label} repeat={repeat+1} current={times["current"][-1]:.6f}s '
                  f'fused={times["fused"][-1]:.6f}s max_abs={max_abs:.3e}', flush=True)
        print(f'{label} median current={np.median(times["current"]):.6f}s '
              f'fused={np.median(times["fused"]):.6f}s', flush=True)
print('Numerical agreement and exact fused symmetry passed.', flush=True)
