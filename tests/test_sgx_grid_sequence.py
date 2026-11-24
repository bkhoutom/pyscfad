"""Grid changes must preserve final-grid SCF roots and implicit derivatives."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import scf as native_scf

from pyscfad import config_update, gto, scf
from pyscfad.sgx import sgx_fit, _sgx_grad
from pyscfad.tools import linear_solver


def molecule():
    mol = gto.Mole(atom='O 0 0 0; H .1 -1.4 1.1; H 0 1.5 1',
                   unit='Bohr', basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def build(mol, level, sequence, native=False, factor=1.):
    mf = native_scf.RHF(mol.to_pyscf()) if native else scf.RHF(mol)
    mf = sgx_fit(mf, auxbasis='weigend', grid_level=level,
                 grid_sequence=sequence, coarse_conv_factor=factor,
                 df_memory_mb=100, screening='static')
    mf.conv_tol, mf.conv_tol_grad, mf.chkfile = 1e-12, 1e-9, None
    return mf


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('level', [2, 3])
def test_final_grid_scf_root(monkeypatch, native, level):
    mol = molecule()
    reference = build(mol, level, False, native)
    reference.kernel()
    seen = []
    original = _sgx_grad.SGX.get_jk

    def record(obj, *args, **kwargs):
        seen.append(obj.grids.level)
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(_sgx_grad.SGX, 'get_jk', record)
    staged = build(mol, level, True, native)
    iterations = []

    def observe(env):
        mf = env['mf']
        grid = mf.with_df.grids.level if native else mf._sgx_backend.grid_level
        iterations.append((grid, bool(env['scf_conv'])))

    staged.callback = observe
    staged.kernel()
    assert staged.converged
    assert level - 1 in seen and seen[-1] == level
    coarse = [converged for grid, converged in iterations if grid == level - 1]
    fine = [converged for grid, converged in iterations if grid == level]
    assert coarse[-1] and fine[-1]
    assert len(coarse) <= staged.max_cycle and len(fine) <= staged.max_cycle
    np.testing.assert_allclose(staged.e_tot, reference.e_tot, atol=2e-10, rtol=0)
    np.testing.assert_allclose(staged.make_rdm1(), reference.make_rdm1(), atol=2e-8, rtol=0)


@pytest.mark.parametrize('implicit', [True, False])
def test_nonstationary_gradient_and_lambda_grid(monkeypatch, implicit):
    mol = molecule()
    monkeypatch.setattr(scf.hf, 'gen_gmres', partial(
        linear_solver.gen_gmres_with_default_kwargs, tol=1e-11))
    seed = np.random.default_rng(71).normal(size=(mol.nao, mol.nao))
    seen = []
    original = _sgx_grad.density_response

    def record(obj, *args, **kwargs):
        seen.append(obj.grids.level)
        return original(obj, *args, **kwargs)

    def objective(m, sequence):
        mf = build(m, 2, sequence)
        mf.kernel()
        assert mf.converged
        return mf.e_tot + .03 * jnp.sum(seed * mf.make_rdm1())

    with config_update('pyscfad_scf_implicit_diff', implicit):
        reference, grad_ref = jax.value_and_grad(objective)(mol, False)
        monkeypatch.setattr(_sgx_grad, 'density_response', record)
        actual, grad = jax.value_and_grad(objective)(mol, True)
        np.testing.assert_allclose(actual, reference, atol=2e-9, rtol=0)
        np.testing.assert_allclose(grad.coords, grad_ref.coords, atol=2e-7, rtol=0)
        if implicit:
            assert 1 in seen and seen[-1] == 2
        delta = np.zeros_like(mol.coords)
        delta[1, 2] = 1e-4
        plus = mol.set_geom_(np.asarray(mol.coords) + delta, unit='Bohr', inplace=False)
        minus = mol.set_geom_(np.asarray(mol.coords) - delta, unit='Bohr', inplace=False)
        fd = (objective(plus, True) - objective(minus, True)) / 2e-4
        np.testing.assert_allclose(grad.coords[1, 2], fd, atol=5e-7, rtol=0)


@pytest.mark.parametrize('native', [False, True])
def test_failed_coarse_scf_does_not_start_final_iterations(native):
    mf = build(molecule(), 2, True, native)
    mf.max_cycle = 1
    with pytest.raises(RuntimeError, match='coarse-grid SCF did not converge'):
        mf.kernel()
    assert mf.max_cycle == 1


@pytest.mark.parametrize('invalid', [-1, 1.5, 3])
def test_invalid_sequence(invalid):
    with pytest.raises(TypeError, match='grid_sequence'):
        sgx_fit(scf.RHF(molecule()), grid_sequence=invalid)


def test_lambda_checkpoint_validates_final_operator_and_reuses_guess(tmp_path):
    from pyscfad.sgx._grid_sequence import response_backend, converge_coarse_lambda
    from pyscfad.tools import _response_checkpoint as checkpoint
    from pyscfad.dlno_stc._restart import RestartManager
    backend = build(molecule(), 2, True)._sgx_backend
    matrix = np.diag(np.linspace(1., 3., 9))
    rhs = np.arange(1., 10.)
    seen = []
    coarse_trials = []

    def matvec(x):
        coarse = response_backend(backend) is not backend
        seen.append(coarse)
        if coarse:
            coarse_trials.append(x.copy())
        return matrix @ x + (0.4 * x if coarse else 0.)

    solve = checkpoint.tag_solver(partial(
        linear_solver.pyscfad_gmres, tol=1e-11, atol=1e-12,
        warm_start=partial(converge_coarse_lambda, backend)))
    manager = RestartManager(tmp_path, method='test-sgx-grid', scientific_payload={})
    try:
        with checkpoint.scope(manager):
            active = checkpoint.session(solve)
            guess, done = active.prepare(matvec, rhs, {})
            assert guess is None and not done
            result = active.solve(matvec, rhs, guess)
            assert any(seen) and seen[-1] is False
            # One coarse Arnoldi block: only normalized basis vectors are
            # evaluated, with no final evaluation of the approximate solution.
            np.testing.assert_allclose([np.linalg.norm(x) for x in coarse_trials],
                                       1., atol=1e-12, rtol=0)
            np.testing.assert_allclose(matrix @ result, rhs, atol=1e-10, rtol=0)
            seen.clear()
            guess, done = checkpoint.session(solve).prepare(matvec, rhs, {})
            assert done and not any(seen)
            # A partially converged restart also bypasses the coarse stage.
            seen.clear()
            solve(matvec, rhs, x0=result * .9)
            assert not any(seen)
    finally:
        manager.close()


def test_lambda_grid_restored_after_failure():
    from pyscfad.sgx._grid_sequence import response_backend, converge_coarse_lambda
    backend = build(molecule(), 2, True)._sgx_backend

    def fail(x):
        assert response_backend(backend).grid_level == 1
        raise RuntimeError('interrupted coarse solve')

    with pytest.raises(RuntimeError, match='interrupted coarse solve'):
        converge_coarse_lambda(backend, fail, np.ones(3), tol=1e-10, atol=0., M=None, restart=3, maxiter=20)
    assert response_backend(backend) is backend


def test_unconverged_coarse_lambda_is_a_warm_start():
    from pyscfad.sgx._grid_sequence import response_backend, converge_coarse_lambda
    backend = build(molecule(), 2, True)._sgx_backend
    matrix = np.diag(np.arange(1., 10.))
    value = converge_coarse_lambda(backend, lambda x: matrix @ x, np.ones(9),
                                   tol=1e-12, atol=0., M=None, restart=1, maxiter=1)
    assert np.isfinite(value).all()
    assert 0 < np.linalg.norm(matrix @ value - 1) < 3
    assert response_backend(backend) is backend


@pytest.mark.parametrize('factor', [1., 1000.])
def test_mixed_precision_staged_response_matches_final_grid(factor):
    mol = molecule()
    seed = np.random.default_rng(72).normal(size=(mol.nao, mol.nao))

    def objective(m, sequence):
        mf = sgx_fit(scf.RHF(m), auxbasis='weigend', grid_level=2,
                     grid_sequence=sequence, coarse_conv_factor=factor, mixed_precision=True,
                     df_memory_mb=100)
        mf.conv_tol, mf.conv_tol_grad, mf.chkfile = 1e-10, 1e-7, None
        mf.kernel()
        assert mf.converged
        return mf.e_tot + .03 * jnp.sum(seed * mf.make_rdm1())

    with config_update('pyscfad_scf_implicit_diff', True):
        reference, grad_ref = jax.value_and_grad(objective)(mol, False)
        actual, grad = jax.value_and_grad(objective)(mol, True)
    np.testing.assert_allclose(actual, reference, atol=2e-8, rtol=0)
    np.testing.assert_allclose(grad.coords, grad_ref.coords, atol=2e-6, rtol=0)


@pytest.mark.parametrize('native', [False, True])
def test_relaxed_scf_keeps_final_tolerances(native):
    mol = molecule()
    reference = build(mol, 3, False, native)
    reference.kernel()
    staged = build(mol, 3, True, native, factor=1000.)
    seen = {}

    def callback(env):
        mf = env['mf']
        level = mf.with_df.grids.level if native else mf._sgx_backend.grid_level
        seen[level] = (env['conv_tol'], env['conv_tol_grad'])

    staged.callback = callback
    staged.kernel()
    np.testing.assert_allclose(seen[2], (1e-9, 1e-6), rtol=1e-14)
    np.testing.assert_allclose(seen[3], (1e-12, 1e-9), rtol=1e-14)
    assert staged.converged and staged.conv_tol == 1e-12
    np.testing.assert_allclose(staged.e_tot, reference.e_tot, atol=2e-10, rtol=0)
    np.testing.assert_allclose(staged.make_rdm1(), reference.make_rdm1(), atol=2e-8, rtol=0)


def test_relaxed_lambda_keeps_final_tolerances(monkeypatch):
    from pyscfad.sgx._grid_sequence import response_backend, converge_coarse_lambda
    from pyscfad._src.scipy.sparse import linalg
    from pyscfad.sgx import _coarse_lambda
    backend = build(molecule(), 3, True, factor=1000.)._sgx_backend
    matrix = np.diag(np.linspace(1., 3., 13))
    rhs = np.arange(1., 14.)
    seen = []
    original = linalg.scipy_gmres

    def record(*args, **kwargs):
        seen.append((response_backend(backend).grid_level,
                     kwargs.get('rtol', kwargs.get('tol')), kwargs['atol']))
        return original(*args, **kwargs)

    def matvec(x):
        return matrix @ x + (.4*x if response_backend(backend) is not backend else 0.)

    original_coarse = _coarse_lambda.coarse_lambda_guess
    def coarse_record(*args, **kwargs):
        seen.append((response_backend(backend).grid_level, kwargs['tol'], kwargs['atol']))
        return original_coarse(*args, **kwargs)
    monkeypatch.setattr(_coarse_lambda, 'coarse_lambda_guess', coarse_record)
    monkeypatch.setattr(linalg, 'scipy_gmres', record)
    value, _ = linalg.gmres(matvec, rhs, tol=1e-9, atol=1e-12,
                            warm_start=partial(converge_coarse_lambda, backend))
    np.testing.assert_allclose(seen, [(2, 1e-6, 1e-9), (3, 1e-9, 1e-12)], rtol=1e-14)
    assert np.linalg.norm(matrix @ value-rhs) <= 1e-9*np.linalg.norm(rhs)


@pytest.mark.parametrize('factor', [True, 0, .5, float('inf'), float('nan')])
def test_invalid_coarse_factor(factor):
    with pytest.raises(ValueError, match='coarse_conv_factor'):
        sgx_fit(scf.RHF(molecule()), coarse_conv_factor=factor)


@pytest.mark.parametrize('factor', [None, 1., 1000.])
def test_lambda_does_not_repeat_scipy_residual_evaluations(monkeypatch, factor):
    from pyscfad.sgx._grid_sequence import response_backend, converge_coarse_lambda
    import io, re
    from pyscf.lib import logger
    from pyscfad.sgx import _coarse_lambda
    from pyscfad._src.scipy.sparse import linalg
    backend = build(molecule(), 3, factor is not None, factor=factor or 1.)._sgx_backend
    matrix = np.diag(np.linspace(1., 3., 13))
    rhs = np.arange(1., 14.)
    original = linalg.scipy_gmres
    in_scipy = False
    records = []

    def matvec(x):
        # Both the coarse warm start and the fine solve own their operator
        # calls; wrapper-level duplicate residual checks must stay absent.
        assert in_scipy, 'operator evaluated outside a solver'
        records[-1]['matvecs'] += 1
        return matrix @ x + (.4*x if response_backend(backend) is not backend else 0.)

    def solve(*args, **kwargs):
        nonlocal in_scipy
        row = dict(grid=response_backend(backend).grid_level, matvecs=0,
                   iterations=0, nonzero_guess=kwargs.get('x0') is not None)
        records.append(row)

        def callback(residual):
            row['iterations'] += 1

        kwargs.update(callback=callback, callback_type='pr_norm')
        in_scipy = True
        try:
            return original(*args, **kwargs)
        finally:
            in_scipy = False

    original_coarse = _coarse_lambda.coarse_lambda_guess
    def coarse_solve(*args, **kwargs):
        nonlocal in_scipy
        output = io.StringIO()
        row = dict(grid=2, matvecs=0, iterations=0, nonzero_guess=False)
        records.append(row)
        kwargs['log'] = logger.Logger(output, 4)
        in_scipy = True
        try:
            result = original_coarse(*args, **kwargs)
            row['iterations'] = int(re.search(r'iterations=(\d+)', output.getvalue())[1])
            return result
        finally:
            in_scipy = False
    monkeypatch.setattr(_coarse_lambda, 'coarse_lambda_guess', coarse_solve)
    monkeypatch.setattr(linalg, 'scipy_gmres', solve)
    warm_start = None if factor is None else partial(converge_coarse_lambda, backend)
    value, _ = linalg.gmres(matvec, rhs, tol=1e-9, atol=1e-12,
                            restart=20, maxiter=1, warm_start=warm_start)
    assert [row['grid'] for row in records] == ([3] if factor is None else [2, 3])
    for row in records:
        # One block: coarse has only Arnoldi matvecs; fine also has initial
        # and final true residuals when starting from a nonzero guess.
        assert row['matvecs'] == (row['iterations'] + int(row['grid'] == 3)
                                  + int(row['nonzero_guess']))
    assert np.linalg.norm(matrix @ value-rhs) <= 1e-9*np.linalg.norm(rhs)


def test_gmres_rejects_nonfinite_solution_without_extra_matvec(monkeypatch):
    from pyscfad._src.scipy.sparse import linalg
    monkeypatch.setattr(linalg, 'scipy_gmres', lambda *args, **kwargs: (np.full(3, np.nan), 0))
    with pytest.raises(RuntimeError, match='nonfinite solution'):
        linalg.gmres(np.eye(3), np.ones(3))
