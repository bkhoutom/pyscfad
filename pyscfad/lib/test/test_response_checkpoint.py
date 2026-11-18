"""Interrupt adjoint solves and assembly, retaining only durable work."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import _restart as storage
from pyscfad.tools import _response_checkpoint as checkpoint, linear_solver


class Interrupted(RuntimeError):
    pass


def manager(path, resume=False):
    return storage.RestartManager(path, resume=resume, method='test-scf-response',
                                  scientific_payload={'geometry': 'fixed'})


def solver(**kwargs):
    with linear_solver.eager_gmres():
        return checkpoint.tag_solver(linear_solver.gen_gmres(
            tol=1e-10, atol=1e-12, restart=2, maxiter=200, **kwargs))


def test_interrupted_gmres_uses_latest_iterate_as_x0(tmp_path, monkeypatch):
    from pyscfad._src.scipy.sparse import linalg
    matrix = np.diag(np.linspace(1., 4., 9))
    rhs = np.arange(1., 10.)
    matvec = lambda x: matrix @ x
    obj = manager(tmp_path)
    def interrupt(stage, key, path):
        if stage == 'scf-lambda-iterate':
            raise Interrupted('after durable iterate')
    monkeypatch.setattr(storage, '_CHECKPOINT_EVENT_HOOK', interrupt)
    with checkpoint.scope(obj):
        active = checkpoint.session(solver())
        guess, done = active.prepare(matvec, rhs, {})
        assert guess is None and not done
        with pytest.raises(Interrupted):
            active.solve(matvec, rhs, guess)
    latest = obj.load_record('scf-lambda-iterate').scalars['lambda']
    assert np.linalg.norm(matvec(latest) - rhs) < np.linalg.norm(rhs)
    assert obj.load_record('scf-lambda-converged') is None
    obj.close()
    monkeypatch.setattr(storage, '_CHECKPOINT_EVENT_HOOK', None)
    real_gmres, seen = linalg.gmres, []
    def observed(*args, **kwargs):
        seen.append(kwargs['x0'].copy())
        return real_gmres(*args, **kwargs)
    monkeypatch.setattr(linalg, 'gmres', observed)
    obj = manager(tmp_path, resume=True)
    with checkpoint.scope(obj):
        active = checkpoint.session(solver())
        guess, done = active.prepare(matvec, rhs, {})
        assert not done
        answer = active.solve(matvec, rhs, guess)
    np.testing.assert_array_equal(seen[0], latest)
    np.testing.assert_allclose(answer, np.linalg.solve(matrix, rhs), atol=1e-9)
    assert obj.load_record('scf-lambda-converged').scalars['residual_norm'] < 1e-8
    obj.close()


def test_converged_lambda_survives_external_vjp_failure(tmp_path, monkeypatch):
    from pyscfad._src.implicit_diff import root_vjp, is_implicit_diff_external_vjp
    from pyscfad._src.scipy.sparse import linalg
    fail = [True]
    @jax.custom_vjp
    def physical(root, parameter):
        return 2 * root - parameter
    def forward(root, parameter):
        return physical(root, parameter), None
    def backward(_, bar):
        if fail[0] and is_implicit_diff_external_vjp():
            raise Interrupted('during force assembly')
        return 2 * bar, -bar
    physical.defvjp(forward, backward)
    def optimality(root, parameter):
        return physical(root, parameter), ()
    preconditions = []
    def precond(*args):
        preconditions.append(True)
        return None
    root, parameter = jnp.array([1., 2.]), jnp.array([2., 4.])
    run = lambda: root_vjp(optimality, root, (parameter,), jnp.array([3., 5.]),
                           solve=solver(), optfn_has_aux=True, gen_precond=precond)
    obj = manager(tmp_path)
    with checkpoint.scope(obj), pytest.raises(Interrupted):
        run()
    assert obj.load_record('scf-lambda-converged') is not None
    assert len(preconditions) == 1
    obj.close()
    fail[0] = False
    def forbidden(*args, **kwargs):
        pytest.fail('a converged lambda must skip GMRES and preconditioner setup')
    monkeypatch.setattr(linalg, 'gmres', forbidden)
    preconditions.clear()
    obj = manager(tmp_path, resume=True)
    with checkpoint.scope(obj):
        actual, = run()
    np.testing.assert_allclose(actual, [1.5, 2.5], atol=1e-12)
    assert not preconditions
    obj.close()


@pytest.mark.parametrize('changed', ['rhs', 'operator', 'tolerance'])
def test_saved_solution_is_checked_against_current_equation(tmp_path, changed):
    obj = manager(tmp_path)
    rhs = np.array([1., 2.])
    with checkpoint.scope(obj):
        active = checkpoint.session(solver())
        active.prepare(lambda x: x, rhs, {})
        active.solve(lambda x: x, rhs, None)
        if changed == 'tolerance':
            # Valid at the old 1e-10 tolerance but not at the new 1e-13 one.
            active._save('scf-lambda-converged', rhs + 1e-11)
        new_rhs = rhs + .1 if changed == 'rhs' else rhs
        matvec = (lambda x: 2*x) if changed == 'operator' else (lambda x: x)
        kwargs = {'tol': 1e-13, 'atol': 0.} if changed == 'tolerance' else {}
        guess, done = active.prepare(matvec, new_rhs, kwargs)
        assert guess is not None and not done
        result = active.solve(matvec, new_rhs, guess, **kwargs)
        np.testing.assert_allclose(matvec(result), new_rhs, atol=1e-12)
    obj.close()


def test_interrupted_atomic_write_keeps_previous_lambda(tmp_path, monkeypatch):
    obj = manager(tmp_path)
    active = checkpoint.LambdaCheckpoint(obj, solver())
    active.prepare(lambda x: x, np.ones(3), {})
    active._save('scf-lambda-iterate', np.full(3, .2))
    def crash(temporary, destination):
        raise Interrupted('before rename')
    monkeypatch.setattr(storage, '_POST_SAVE_TEST_HOOK', crash)
    with pytest.raises(Interrupted):
        active._save('scf-lambda-iterate', np.full(3, .7))
    np.testing.assert_array_equal(obj.load_record('scf-lambda-iterate').scalars['lambda'], [.2]*3)
    obj.close()


def test_nonconvergence_keeps_iterate_without_marking_converged(tmp_path):
    obj = manager(tmp_path)
    active = checkpoint.LambdaCheckpoint(obj, solver())
    rhs = np.ones(5)
    matvec = lambda x: np.arange(1., 6.) * x
    active.prepare(matvec, rhs, {})
    with pytest.raises(RuntimeError, match='failed to converge'):
        active.solve(matvec, rhs, None, restart=1, maxiter=1, tol=1e-14, atol=0.)
    assert obj.load_record('scf-lambda-iterate') is not None
    assert obj.load_record('scf-lambda-converged') is None
    obj.close()


def test_completed_contractions_are_reused_and_changed_seeds_miss(tmp_path):
    obj = manager(tmp_path)
    with checkpoint.scope(obj):
        value, reused = checkpoint.contraction('test', np.ones(3), lambda: np.arange(3.))
        assert not reused
    obj.close()
    obj = manager(tmp_path, resume=True)
    def forbidden():
        pytest.fail('completed contraction was repeated')
    with checkpoint.scope(obj):
        answer, reused = checkpoint.contraction('test', np.ones(3), forbidden)
        np.testing.assert_array_equal(answer, value)
        assert reused
        _, reused = checkpoint.contraction('test', np.zeros(3), lambda: np.zeros(3))
        assert not reused
        assert checkpoint.session(linear_solver.gen_gmres()) is None
        assert checkpoint.session(solver(), traced=True) is None
    assert checkpoint.session(solver()) is None
    obj.close()
