"""Checks on the original, signed Boys response equation."""
from functools import wraps
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from pyscf.lib import logger


_DENSE_LIMIT = 256
_CONDITION_WARNING = 1e10


def _report(residual, rhs_norm, solution_norm, condition, *, tol, atol,
            condition_kind, log=None, nullity=0, method='gmres'):
    residual, rhs_norm, solution_norm, condition = map(
        float, (residual, rhs_norm, solution_norm, condition))
    allowed = max(atol, tol * rhs_norm)
    if not all(np.isfinite(v) for v in (residual, rhs_norm, solution_norm)):
        raise RuntimeError('Boys response contains nonfinite values')
    if residual > allowed * (1 + 1e-6):
        raise RuntimeError(f'Boys response residual {residual:.6g} exceeds {allowed:.6g} '
                           'in the original equation; possible incompatible null-space component')
    amplification = solution_norm / rhs_norm if rhs_norm else 0.
    if log is not None:
        logger.info(log, 'Boys response: method=%s residual=%.6g target=%.6g '
                    'amplification=%.6g %s=%.6g nullity=%s', method, residual,
                    allowed, amplification, condition_kind, condition, nullity)
    if condition >= _CONDITION_WARNING:
        warnings.warn(f'Ill-conditioned Boys response: {condition_kind}={condition:.6g}, '
                      f'amplification={amplification:.6g}, residual={residual:.6g}; '
                      'small nonzero and negative modes were retained', RuntimeWarning, stacklevel=3)


def checked_solver(solver, *, tol, atol, symmetry=False, log=None, maxiter=30, restart=20):
    """Preserve the Hessian signs and check the unpreconditioned residual.

    Small concrete systems get an exact spectral condition diagnostic.
    Larger/traced systems report a condition estimate or norm-based lower
    bound and the actual response amplification; these cannot certify good
    conditioning. Explicit ``symmetry=True`` selects the minimum-norm
    range solution, conditional on compatibility with the *original* equation.
    A numerical null mode is not automatically presumed to be a gauge.
    """
    @wraps(solver)  # Preserve preconditioner/backend discovery through wrappers.
    def solve(action, rhs, **kwargs):
        n = rhs.size
        shape = rhs.shape
        # Fixed local probe, with no effect on stochastic sampling state.
        probe = np.random.default_rng(817).normal(size=shape)
        probe /= np.linalg.norm(probe)
        applied_probe = action(jnp.asarray(probe))
        traced = any(isinstance(v, jax.core.Tracer) for v in
                     jax.tree_util.tree_leaves((rhs, applied_probe)))
        if traced:
            # Starting from zero without a preconditioner keeps Krylov vectors
            # in range(H) for compatible symmetric singular systems.
            if symmetry:
                kwargs = dict(kwargs, M=None, x0=jnp.zeros_like(rhs))
            solution, info = solver(action, rhs, **kwargs)
            residual = jnp.linalg.norm(action(solution) - rhs)
            rhs_norm = jnp.linalg.norm(rhs)
            solution_norm = jnp.linalg.norm(solution)
            lower_bound = jnp.linalg.norm(applied_probe) * solution_norm / jnp.maximum(
                rhs_norm, jnp.finfo(rhs.dtype).tiny)
            jax.debug.callback(
                lambda r, b, x, c: _report(r, b, x, c, tol=tol, atol=atol,
                    condition_kind='condition_lower_bound', log=log, nullity='unchecked',
                    method='traced_gmres'),
                residual, rhs_norm, solution_norm, lower_bound, ordered=True)
            return solution, info

        b = np.asarray(rhs).ravel()
        rhs_norm = np.linalg.norm(b)
        allowed = max(atol, tol * rhs_norm)
        if not np.all(np.isfinite(b)) or not np.all(np.isfinite(applied_probe)):
            raise RuntimeError('Boys response has nonfinite Hessian action or right-hand side')
        nullity = 'unchecked'
        condition = None
        method = 'gmres'

        def matvec(x):
            return np.asarray(action(x.reshape(shape))).ravel()

        if n <= _DENSE_LIMIT:
            matrix = np.column_stack([matvec(v) for v in np.eye(n)])
            if not np.all(np.isfinite(matrix)):
                raise RuntimeError('Boys response Hessian contains nonfinite values')
            values, vectors = np.linalg.eigh(.5 * (matrix + matrix.T))
            scale = np.max(np.abs(values), initial=0.)
            # Only numerical zero at floating-point resolution. This is not
            # the much larger curvature/conditioning warning threshold.
            null_tol = 8 * np.finfo(matrix.dtype).eps * n * scale
            null = np.abs(values) <= null_tol
            nullity = int(np.count_nonzero(null))
            condition = np.inf if nullity else scale / np.min(np.abs(values))
            if nullity and rhs_norm:
                if not symmetry:
                    raise RuntimeError('Boys response has a numerical null space; '
                                       'declare symmetry=True only for known gauge freedom')
                component = vectors[:, null].T @ b
                if np.linalg.norm(component) > allowed:
                    raise RuntimeError('Boys response right-hand side has an incompatible '
                                       f'null-space component {np.linalg.norm(component):.6g} '
                                       f'(tolerance {allowed:.6g}); refusing to discard it')
                retained = ~null
                condition = scale / np.min(np.abs(values[retained])) if np.any(retained) else 1.
                solution = (vectors[:, retained] @
                            ((vectors[:, retained].T @ b) / values[retained])).reshape(shape)
                method, info = 'declared_gauge_range', 0
                warnings.warn(f'Boys response: using the compatible range solution for '
                              f'{nullity} declared gauge null modes', RuntimeWarning, stacklevel=2)
            elif not rhs_norm:
                solution, info = np.zeros(shape), 0
            else:
                solution, info = solver(action, rhs, **kwargs)
        elif symmetry:
            # LSMR from zero returns the minimum-norm range solution without
            # dropping negative curvature or explicitly storing null vectors.
            from scipy.sparse.linalg import LinearOperator, lsmr
            operator = LinearOperator((n, n), matvec=matvec, rmatvec=matvec, dtype=b.dtype)
            result = lsmr(operator, b, atol=min(tol, 1e-12), btol=min(tol, 1e-12),
                          conlim=0., maxiter=maxiter * min(n, restart))
            solution, info = result[0].reshape(shape), result[1]
            method = 'declared_gauge_lsmr'
            # Full residual below, including any incompatible null component.
        else:
            solution, info = solver(action, rhs, **kwargs)
        solution_norm = np.linalg.norm(np.asarray(solution))
        residual = np.linalg.norm(matvec(np.asarray(solution).ravel()) - b)
        if condition is None:
            condition = np.linalg.norm(applied_probe) * solution_norm / rhs_norm if rhs_norm else 0.
            if method == 'declared_gauge_lsmr':
                condition = max(condition, result[6])
                condition_kind = 'lsmr_condition_estimate'
            else:
                condition_kind = 'condition_lower_bound'
        else:
            condition_kind = 'range_spectral_condition' if symmetry and nullity else 'spectral_condition'
        _report(residual, rhs_norm, solution_norm, condition, tol=tol, atol=atol,
                condition_kind=condition_kind, log=log, nullity=nullity, method=method)
        # Acceptance is determined by the checked original equation, including
        # LSMR's alternative successful termination codes.
        return solution, 0
    return solve
