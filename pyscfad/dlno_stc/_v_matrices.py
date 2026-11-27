"""Direct Coulomb Gram matrices for LIS selection, without MP2 amplitudes.

For real g[i,j,a,b] = (ia|jb), summed over selected targets i:
    Vo[j,k] = sum(i,a,b) g[i,j,a,b] g[i,k,a,b]
    Vv[a,c] = .5 sum(i,j,b) (g[i,j,a,b] g[i,j,c,b]
                             + g[i,j,b,a] g[i,j,b,c]).
No spin factor or exchange term is included. Eigenvalues have units Eh**2.
Auxiliary Gram contractions give quartic dense cost and cubic storage.
Only a fingerprint and small metadata survive forward; reverse rebuilds the
Gram matrices and returns the exact arbitrary-seed DF-factor derivative.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from .controls import normalize_matrix_controls, estimate_matrix_memory, check_solver_memory
from .protocol import matrix_request_fingerprint, validate_full_inputs, validate_matrix_result

matrix_kernel_version = 'direct_coulomb_v_v1'


@dataclass(frozen=True)
class _Replay:
    fingerprint: str


def _sym(a):
    return (a + a.T) * .5


def _target_batches(targets, settings):
    size = settings['target_batch_size'] or len(targets)
    for first in range(0, len(targets), size):
        yield list(targets[first:first + size])


def _grams(b, targets, settings):
    p, no, nv = b.shape
    x = b.reshape(p, no * nv)
    h = x @ x.T
    k = np.zeros_like(h)
    vv = np.zeros((nv, nv))
    for indices in _target_batches(targets, settings):
        y = b[:, indices, :].reshape(p, len(indices) * nv)
        k += y @ y.T
        hy = (h @ y).reshape(p, len(indices), nv)
        vv += .5 * np.einsum('Pia,Pic->ac', y.reshape(hy.shape), hy)
    return x, h, k, vv


def _memory(inputs, targets, settings, with_grad):
    p, no, nv = inputs['B'].shape
    estimate = estimate_matrix_memory(p, no, nv, len(targets), settings,
                                      matrix_kind='V', with_grad=with_grad)
    check_solver_memory(estimate, settings.get('memory_limit_mb'), stage='V matrices')
    return estimate['total_bytes']


def solve(inputs, metadata, controls):
    from .backend import _matrix_targets
    validate_full_inputs(inputs)
    settings = normalize_matrix_controls(controls, 'V')
    targets = _matrix_targets(metadata, inputs['B'].shape[1])
    memory = _memory(inputs, targets, settings, False)
    metadata = dict(fragment_id=metadata.get('fragment_id', 0), target_indices=targets,
                    native_version=matrix_kernel_version)
    started = perf_counter()
    with threadpool_limits(settings['threads'], user_api='blas'):
        b = np.asarray(inputs['B'])
        x, _, k, vv = _grams(b, targets, settings)
        kb = (k @ x).reshape(b.shape)
        vo = _sym(np.einsum('Pja,Pka->jk', b, kb))
        vv = _sym(vv + .5 * np.einsum('Pja,Pjc->ac', b, kb))
    diagnostics = dict(mode='deterministic', matrix_kind='V',
        native_version=matrix_kernel_version, threads=settings['threads'],
        occupied_frobenius_se=0., virtual_frobenius_se=0.,
        occupied_standard_error=np.zeros_like(vo), virtual_standard_error=np.zeros_like(vv),
        occupied_target_met=True, virtual_target_met=True,
        memory_estimate_bytes=memory, production_draws=0,
        native_timings={'contraction_seconds': perf_counter() - started})
    fingerprint = matrix_request_fingerprint(inputs, metadata, settings, None, matrix_kind='V')
    result = dict(occupied=vo, virtual=vv, matrix_kind='V', diagnostics=diagnostics,
                  metadata=metadata, controls=settings, replay=_Replay(fingerprint))
    validate_matrix_result(result, inputs, matrix_kind='V')
    return result


def pullback(inputs, forward, *, energy_bar, occupied_bar, virtual_bar):
    seed = np.asarray(energy_bar)
    if seed.shape != () or not np.isfinite(seed) or seed != 0:
        raise ValueError('V is matrix-only and requires energy_bar=0')
    replay = forward['replay']
    settings = normalize_matrix_controls(forward['controls'], 'V')
    fingerprint = matrix_request_fingerprint(
        inputs, forward['metadata'], settings, None, matrix_kind='V')
    if not isinstance(replay, _Replay) or replay.fingerprint != fingerprint:
        raise ValueError('V matrix replay fingerprint changed')
    seeds = []
    for name, value in (('occupied', occupied_bar), ('virtual', virtual_bar)):
        value = np.asarray(value, dtype=np.float64)
        if value.shape != forward[name].shape or not np.isfinite(value).all():
            raise ValueError(f'invalid V {name} cotangent')
        seeds.append(_sym(value))
    so, sv = seeds
    targets = forward['metadata']['target_indices']
    _memory(inputs, targets, settings, True)
    with threadpool_limits(settings['threads'], user_api='blas'):
        b = np.asarray(inputs['B'])
        p, no, nv = b.shape
        x, h, k, _ = _grams(b, targets, settings)
        ox = np.einsum('jk,Pka->Pja', so, b).reshape(x.shape)
        vx = (b @ sv).reshape(x.shape)
        kbar = _sym(ox @ x.T + .5 * (vx @ x.T))
        bbar = (k @ (2 * ox + vx)).reshape(b.shape)
        hbar = np.zeros_like(h)
        for indices in _target_batches(targets, settings):
            y = b[:, indices, :]
            vy = (y @ sv).reshape(p, len(indices) * nv)
            y = y.reshape(vy.shape)
            hbar += .5 * (vy @ y.T)
            bbar[:, indices, :] += (h @ vy + 2 * kbar @ y).reshape(p, len(indices), nv)
        bbar += ((hbar + hbar.T) @ x).reshape(b.shape)
    return dict(foo=np.zeros_like(inputs['foo']), fvv=np.zeros_like(inputs['fvv']), B=bbar)
