"""Check coarse warm starts against ordinary GMRES and count actual work."""
import io
import re

import numpy as np
import pytest
from pyscf.lib import logger
from scipy.sparse.linalg import LinearOperator, gmres

from pyscfad.lib import _solver_memory
from pyscfad.sgx._coarse_lambda import coarse_lambda_guess


@pytest.mark.parametrize('restart', [3, 20])
@pytest.mark.parametrize('complex_rhs', [False, True])
@pytest.mark.parametrize('preconditioned', [False, True])
def test_estimate_stopping_saves_one_operator_call(restart, complex_rhs, preconditioned):
    rng = np.random.default_rng(81)
    matrix = np.diag(np.linspace(1., 3., 13)) + .01*rng.normal(size=(13, 13))
    rhs = rng.normal(size=13)
    if complex_rhs:
        matrix = matrix + .01j*rng.normal(size=(13, 13))
        rhs = rhs + 1j*rng.normal(size=13)
    precond = np.diag(1 / matrix.diagonal()) if preconditioned else None
    reference_calls, actual_calls, inner = [], [], []
    def reference_op(x):
        reference_calls.append(1)
        return matrix @ x
    reference, info = gmres(LinearOperator(matrix.shape, reference_op, dtype=matrix.dtype),
                            rhs, rtol=1e-8, atol=1e-12, restart=restart, maxiter=100,
                            M=precond, callback=inner.append, callback_type='pr_norm')
    assert info == 0
    def observed(x):
        assert _solver_memory.active_plans()
        actual_calls.append(1)
        return matrix @ x
    output = io.StringIO()
    result = coarse_lambda_guess(observed, rhs, tol=1e-8, atol=1e-12, M=precond,
                                  restart=restart, maxiter=100, log=logger.Logger(output, 4))
    np.testing.assert_allclose(result, reference, rtol=1e-7, atol=1e-10)
    if not preconditioned:
        assert len(actual_calls) == len(reference_calls) - 1
    else:
        # A preconditioned estimate can pass before the true residual does;
        # ordinary GMRES may then need additional work that a warm start skips.
        assert len(actual_calls) <= len(reference_calls) - 1
    reported = re.search(r'iterations=(\d+) restart_cycles=(\d+) operator_evaluations=(\d+)',
                         output.getvalue())
    iterations, blocks, evaluations = map(int, reported.groups())
    assert iterations <= len(inner)
    assert evaluations == len(actual_calls) == iterations + blocks - 1
    assert 'final_residual_check=skipped' in output.getvalue()
    assert not _solver_memory.active_plans()


def test_iteration_limit_returns_useful_guess_without_last_residual():
    calls = []
    matrix = np.diag(np.linspace(1., 4., 9))
    def observed(x):
        calls.append(1)
        return matrix @ x
    value = coarse_lambda_guess(observed, np.ones(9), tol=1e-14, atol=0.,
                                 M=None, restart=2, maxiter=2)
    assert len(calls) == 5  # Four Arnoldi steps, one restart residual.
    assert np.linalg.norm(matrix @ value - 1) < 3


def test_zero_rhs_does_not_evaluate_operator():
    def forbidden(x):
        pytest.fail('zero RHS needs no operator')
    result = coarse_lambda_guess(forbidden, np.zeros((2, 2)), tol=1e-6, atol=1e-12,
                                  M=None, restart=20, maxiter=30)
    np.testing.assert_array_equal(result, np.zeros((2, 2)))


def test_nonfinite_coarse_operator_rejected_and_reservation_released():
    with pytest.raises(RuntimeError, match='nonfinite'):
        coarse_lambda_guess(lambda x: x*np.nan, np.ones(3), tol=1e-6, atol=1e-12,
                            M=None, restart=3, maxiter=30)
    assert not _solver_memory.active_plans()
