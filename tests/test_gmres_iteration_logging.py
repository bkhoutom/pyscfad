"""Count inner steps without changing GMRES work or checkpoint callbacks."""
import io
import re

import numpy as np
import pytest
from pyscf.lib import logger

from pyscfad._src.scipy.sparse.linalg import gmres
from pyscfad.lib.test.test_response_checkpoint import manager, solver
from pyscfad.tools import _response_checkpoint as checkpoint


def counts(output):
    match = re.search(r'iterations=(\d+) restart_cycles=(\d+) operator_evaluations=(\d+)',
                      output.getvalue())
    assert match, output.getvalue()
    return tuple(map(int, match.groups()))


@pytest.mark.parametrize('callback_type', [None, 'x', 'pr_norm', 'legacy'])
@pytest.mark.parametrize('guess', ['zero', 'warm', 'exact', 'zero_rhs'])
def test_logged_counts_and_operator_work(callback_type, guess):
    matrix = np.diag(np.linspace(1., 2., 7))
    rhs = np.zeros(7) if guess == 'zero_rhs' else np.arange(1., 8.)
    x0 = dict(zero=None, warm=np.full(7, .1), exact=rhs / matrix.diagonal(),
              zero_rhs=np.ones(7))[guess]
    inner_steps = []
    reference, _ = gmres(lambda x: matrix @ x, rhs, x0=x0, tol=1e-9, atol=1e-12,
                         restart=3, maxiter=100, callback=inner_steps.append,
                         callback_type='pr_norm')
    runs = []
    for logged in (False, True):
        calls, callbacks = [], []
        output = io.StringIO()
        def matvec(x):
            calls.append(x.copy())
            return matrix @ x
        answer, _ = gmres(matvec, rhs, x0=x0, tol=1e-9, atol=1e-12, restart=3, maxiter=100,
                          callback=None if callback_type is None else callbacks.append,
                          callback_type=callback_type,
                          log=logger.Logger(output, 5) if logged else None)
        np.testing.assert_allclose(answer, reference, atol=1e-12)
        runs.append((calls, callbacks))
        if logged:
            iterations, blocks, evaluations = counts(output)
            assert iterations == len(inner_steps)
            assert evaluations == len(calls)
            if guess in ('zero', 'warm'):
                assert blocks > 1
    # Logging must preserve every evaluation and callback, including x0 checks.
    for before, after in zip(runs[0], runs[1]):
        np.testing.assert_array_equal(before, after)


def test_checkpoint_reports_inner_steps_and_keeps_iterate_callbacks(tmp_path):
    output = io.StringIO()
    matrix = np.diag(np.linspace(1., 4., 9))
    rhs = np.arange(1., 10.)
    expected = []
    gmres(lambda x: matrix @ x, rhs, tol=1e-10, atol=1e-12, restart=2,
          maxiter=200, callback=expected.append, callback_type='pr_norm')
    obj = manager(tmp_path)
    try:
        with checkpoint.scope(obj):
            active = checkpoint.session(solver(log=logger.Logger(output, 5)))
            active.prepare(lambda x: matrix @ x, rhs, {})
            answer = active.solve(lambda x: matrix @ x, rhs, None)
        iterations, blocks, _ = counts(output)
        assert iterations == len(expected)
        assert blocks == obj.load_record('scf-lambda-converged').scalars['cycles']
        np.testing.assert_allclose(matrix @ answer, rhs, atol=1e-9)
    finally:
        obj.close()
