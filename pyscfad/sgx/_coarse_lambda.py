"""GMRES warm start stopped by its Arnoldi residual estimate.

This is deliberately not a convergence-certified linear solver. The final-grid
solver validates its own residual. A true coarse residual is formed only when
another restart block needs it, never to certify the returned warm start.
"""
import time

import numpy as np
from scipy.sparse.linalg import aslinearoperator

from pyscfad.lib import _solver_memory
from pyscfad.lib._threading import fixed_threads


def coarse_lambda_guess(matvec, b, *, tol, atol, M, restart, maxiter, log=None):
    from pyscfad._src.scipy.sparse.linalg import _matvec_to_scipy
    shape = b.shape
    b = np.asarray(b).ravel()
    operator = (_matvec_to_scipy(matvec, b.reshape(shape)) if callable(matvec)
                else aslinearoperator(matvec))
    dtype = np.result_type(b.dtype, operator.dtype)
    if dtype.kind not in 'fc':
        dtype = np.dtype(float)
    b = b.astype(dtype, copy=False)
    n = b.size
    restart = min(n, 20 if restart is None else int(restart))
    maxiter = n * 10 if maxiter is None else int(maxiter)
    if restart <= 0 or maxiter <= 0:
        raise ValueError('coarse GMRES restart and maxiter must be positive')
    if not np.isfinite(tol) or tol < 0 or not np.isfinite(atol or 0.) or (atol or 0.) < 0:
        raise ValueError('coarse GMRES tolerances must be finite and nonnegative')
    precondition = (lambda x: x) if M is None else aslinearoperator(M).matvec
    x = np.zeros(n, dtype=dtype)
    iterations = evaluations = blocks = 0
    started = time.perf_counter()
    reason = 'iteration_limit'
    estimate = 0.
    with fixed_threads(), _solver_memory.reserve_gmres(n, dtype, restart, M is not None):
        norm_b = np.linalg.norm(b)
        threshold = max(float(atol or 0.), tol * norm_b)
        residual = b.copy()
        # Match the usual left-preconditioned GMRES estimate scaling.
        target = (np.linalg.norm(precondition(b)) * min(1., threshold / norm_b)
                  if norm_b else threshold)
        if not np.isfinite(target):
            raise RuntimeError('nonfinite coarse GMRES preconditioner or RHS')
        if norm_b <= threshold:
            reason = 'estimated_tolerance'
        else:
            basis = np.empty((restart + 1, n), dtype=dtype)
            h = np.zeros((restart + 1, restart), dtype=dtype)
            for block in range(maxiter):
                blocks += 1
                r = np.asarray(precondition(residual))
                beta = np.linalg.norm(r)
                if not np.isfinite(beta):
                    raise RuntimeError('nonfinite coarse GMRES residual')
                if beta <= target:
                    estimate, reason = beta, 'estimated_tolerance'
                    break
                basis[0] = r / beta
                h.fill(0.)
                rhs = np.zeros(restart + 1, dtype=dtype)
                rhs[0] = beta
                breakdown = False
                for j in range(restart):
                    w = np.array(precondition(operator.matvec(basis[j])), dtype=dtype, copy=True)
                    evaluations += 1
                    iterations += 1
                    norm_before = np.linalg.norm(w)
                    if not np.isfinite(norm_before):
                        raise RuntimeError('nonfinite coarse GMRES operator result')
                    for k in range(j + 1):
                        h[k, j] = np.vdot(basis[k], w)
                        w -= h[k, j] * basis[k]
                    h[j + 1, j] = np.linalg.norm(w)
                    breakdown = abs(h[j + 1, j]) <= np.finfo(dtype).eps * norm_before
                    if not breakdown:
                        basis[j + 1] = w / h[j + 1, j]
                    small_h = h[:j + 2, :j + 1]
                    y = np.linalg.lstsq(small_h, rhs[:j + 2], rcond=None)[0]
                    estimate = np.linalg.norm(rhs[:j + 2] - small_h @ y)
                    if estimate <= target or breakdown:
                        break
                x += y @ basis[:j + 1]
                if not np.isfinite(x).all():
                    raise RuntimeError('nonfinite coarse GMRES warm start')
                if estimate <= target or breakdown:
                    reason = 'estimated_tolerance' if estimate <= target else 'Arnoldi_breakdown'
                    break
                if block + 1 < maxiter:
                    residual = b - operator.matvec(x)
                    evaluations += 1
    if log is not None:
        log.info('SCF lambda coarse GMRES: iterations=%d restart_cycles=%d '
                 'operator_evaluations=%d wall_seconds=%.3f '
                 'estimated_residual=%.6g final_residual_check=skipped stop=%s',
                 iterations, blocks, evaluations, time.perf_counter() - started,
                 estimate, reason)
    return x.reshape(shape)
