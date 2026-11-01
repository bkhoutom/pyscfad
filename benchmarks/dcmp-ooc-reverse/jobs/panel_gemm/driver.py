"""Compare occupied-panel density construction on one compute node."""
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits


P, NAO, RANK, REPEATS, THREADS = 17, 800, 80, 3, 16
rng = np.random.default_rng(451)
panel = rng.standard_normal((P, NAO, RANK))
S = rng.standard_normal((NAO, RANK))
G = rng.standard_normal((NAO, NAO))
D = rng.standard_normal((NAO, NAO))
a = rng.standard_normal(P)
b = rng.standard_normal(P)


def core(method):
    if method == 'batched':
        return panel @ S.T
    return (panel.reshape(-1, RANK) @ S.T).reshape(P, NAO, NAO)


def full(method):
    density = core(method)
    density += density.swapaxes(1, 2).copy()
    density += a[:, None, None] * G
    density += b[:, None, None] * D
    density *= 2
    return density


def timed(fn, method):
    start = time.perf_counter()
    result = fn(method)
    return result, time.perf_counter() - start


with threadpool_limits(limits=THREADS, user_api='blas'):
    print(f'panel={panel.shape}, S={S.shape}, repeats={REPEATS}, threads={THREADS}', flush=True)
    print('threadpools:', threadpool_info(), flush=True)
    # Warm each path before comparing separately timed core and full operations.
    for method in ('batched', 'flattened'):
        del_result = full(method)
        del del_result
    for fn in (core, full):
        times = {'batched': [], 'flattened': []}
        for repeat in range(REPEATS):
            # Alternate order to reduce systematic order effects.
            order = ('batched', 'flattened') if repeat % 2 == 0 else ('flattened', 'batched')
            results = {}
            for method in order:
                results[method], elapsed = timed(fn, method)
                times[method].append(elapsed)
            np.testing.assert_allclose(results['batched'], results['flattened'], rtol=1e-12, atol=1e-12)
            max_abs = np.max(np.abs(results['batched'] - results['flattened']))
            print(f'{fn.__name__} repeat={repeat + 1} '
                  f'batched={times["batched"][-1]:.6f}s '
                  f'flattened={times["flattened"][-1]:.6f}s max_abs={max_abs:.3e}', flush=True)
        print(f'{fn.__name__} median: batched={np.median(times["batched"]):.6f}s '
              f'flattened={np.median(times["flattened"]):.6f}s', flush=True)
print('Numerical comparisons passed.', flush=True)
