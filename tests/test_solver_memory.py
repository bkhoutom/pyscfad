"""Live GMRES reservations follow solver inputs and end with the solve."""
import numpy as np
import pytest

from pyscfad.lib import _solver_memory
from pyscfad._src.scipy.sparse.linalg import gmres


@pytest.mark.parametrize('shape,dtype,restart', [
    ((4, 4), np.float32, 3), ((5, 5), np.float64, 7),
    ((3, 3), np.complex128, 20), ((2, 2), np.float64, None)])
def test_live_reservation_matches_actual_solver(shape, dtype, restart):
    rhs = np.arange(1, np.prod(shape)+1, dtype=np.float64).reshape(shape).astype(dtype)
    diag = np.linspace(1., 2., rhs.size).reshape(shape).astype(dtype)
    observed = []

    def matvec(x):
        observed.append(_solver_memory.active_plans()[0])
        return diag*x

    solution, info = gmres(matvec, rhs, restart=restart, maxiter=100,
                           tol=1e-6, atol=1e-8)
    assert info == 0
    assert np.linalg.norm(diag*solution-rhs) <= 1.1e-6*np.linalg.norm(rhs)
    assert observed
    for plan in observed:
        assert plan['dimension'] == rhs.size
        assert plan['dtype'] == np.dtype(dtype).name
        assert plan['restart'] == min(rhs.size, 20 if restart is None else restart)
        assert plan['vector_bytes'] == rhs.nbytes
        assert plan['basis_bytes'] == (plan['restart']+1)*rhs.nbytes
    assert _solver_memory.reserved_bytes() == 0


def test_nested_reservation_and_exception_cleanup():
    assert _solver_memory.reserved_bytes() == 0
    with _solver_memory.reserve_gmres(23, np.float64, 4) as outer:
        with pytest.raises(RuntimeError, match='matvec failed'):
            def fail(x):
                plans = _solver_memory.active_plans()
                assert len(plans) == 2
                assert _solver_memory.reserved_bytes() == sum(p['reserved_bytes'] for p in plans)
                raise RuntimeError('matvec failed')
            gmres(fail, np.ones((3, 3)), restart=2, atol=0.)
        assert _solver_memory.reserved_bytes() == outer['reserved_bytes']
    assert _solver_memory.active_plans() == ()


def test_preconditioner_and_dtype_affect_reservation():
    plain = _solver_memory.gmres_workspace(100, np.float64, 8)
    precond = _solver_memory.gmres_workspace(100, np.float64, 8, True)
    complex_plan = _solver_memory.gmres_workspace(100, np.complex128, 8)
    assert precond['reserved_bytes']-plain['reserved_bytes'] == plain['vector_bytes']
    assert complex_plan['reserved_bytes'] == 2*plain['reserved_bytes']
