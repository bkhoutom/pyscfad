"""Direct DF-J / SGX-K forward, arbitrary response seeds, and SCF response."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import scf as pscf

from pyscfad import config_update, gto, scf
from pyscfad.df.df import DF
from pyscfad.sgx import sgx_fit
from pyscfad.sgx import sgx as impl
from pyscfad.tools import linear_solver


@pytest.fixture
def mol():
    m = gto.Mole(atom='O 0 0 0; H .1 -1.4 1.1; H 0 1.5 1',
                 unit='Bohr', basis='sto-3g', verbose=0)
    m.build(trace_exp=False, trace_ctr_coeff=False)
    return m


def build(mol, **kw):
    mf = sgx_fit(scf.RHF(mol), auxbasis='weigend', grid_level=0,
                 max_memory=100, **kw)
    mf.conv_tol, mf.conv_tol_grad, mf.chkfile = 1e-12, 1e-9, None
    return mf


def moved(mol, step, leaf_only=False):
    coords = np.asarray(mol.coords).copy()
    coords[1, 2] += step
    if not leaf_only:
        return mol.set_geom_(coords, unit='Bohr', inplace=False)
    # For the SGX primitive alone, also catch stale native _env/cache use.
    other = mol.copy()
    other.coords = coords
    return other


@pytest.mark.parametrize('fit', [True, False])
@pytest.mark.parametrize('k', [0, 1])
def test_arbitrary_bilinear_response(mol, fit, k):
    mf = build(mol, fit_ovlp=fit)
    rng = np.random.default_rng(123)
    dm, bar, direction = rng.normal(size=(3, mol.nao, mol.nao))
    fun = lambda m, d: jnp.sum(mf.get_jk(m, d, with_j=k == 0, with_k=k == 1)[k] * bar)
    value, (g, gd) = jax.value_and_grad(fun, (0, 1))(mol, dm)
    h = 1e-4
    fd = (fun(moved(mol, h, True), dm) - fun(moved(mol, -h, True), dm)) / (2*h)
    np.testing.assert_allclose(g.coords[1, 2], fd, atol=2e-7, rtol=1e-6)
    fd_dm = (fun(mol, dm+h*direction) - fun(mol, dm-h*direction)) / (2*h)
    np.testing.assert_allclose(np.sum(gd*direction), fd_dm, atol=2e-9, rtol=1e-9)
    np.testing.assert_allclose(np.sum(g.coords, axis=0), 0, atol=1e-9)
    # A second seed must reuse the tape without a cached-gradient assumption.
    _, pullback = jax.vjp(fun, mol, dm)
    second, _ = pullback(jnp.asarray(-1e-6))
    np.testing.assert_allclose(second.coords, -1e-6*g.coords, atol=2e-13, rtol=2e-7)
    assert np.isfinite(value)


def test_density_response_is_linear_without_geometry_work(mol, monkeypatch):
    mf = build(mol)
    rng = np.random.default_rng(44)
    dm, bar = rng.normal(size=(2, 2, mol.nao, mol.nao))
    dm = impl._sym(dm)
    fun = lambda d: mf.get_jk(dm=d)
    jk, pull = jax.vjp(fun, dm)
    def forbidden(*args, **kwargs):
        raise AssertionError('density response computed geometry derivatives')
    monkeypatch.setattr(impl._dfj, 'geometry_many', forbidden)
    monkeypatch.setattr(impl._sgx_grad, 'geometry_many', forbidden)
    gb, = pull((bar, .3*bar))
    j, k = mf.get_jk(dm=impl._sym(bar))
    np.testing.assert_allclose(gb, j+.3*k, atol=1e-10)
    np.testing.assert_allclose(np.sum(bar*jk[1]), np.sum(dm*k), atol=2e-10)
    j1, k1 = mf.get_jk(dm=dm[0]+dm[1])
    np.testing.assert_allclose(j1, jk[0].sum(0), atol=2e-10)
    np.testing.assert_allclose(k1, jk[1].sum(0), atol=2e-10)


@pytest.mark.parametrize('screening', ['static', 'per_pair'])
@pytest.mark.parametrize('exchange', [False, True])
def test_pullback_with_tiny_response_seeds(mol, screening, exchange):
    mf = build(mol, screening=screening, mixed_precision=screening == 'per_pair')
    dm = pscf.RHF(mol.to_pyscf()).get_init_guess()
    _, pull = jax.vjp(lambda d: mf.get_jk(dm=d, with_j=not exchange, with_k=exchange)[int(exchange)], dm)
    bar = np.random.default_rng(904).normal(size=dm.shape)
    reference, = pull(bar)
    obj = mf._sgx_backend._native
    opt = obj._vjopt
    forward_tol = None if opt is None else opt.direct_scf_tol
    from pyscfad.sgx import _native
    plan = _native.get_plan(obj._pjs_data) if exchange else None
    accuracy = None if plan is None else plan.accuracy
    for scale in (1e-9, 1e-11, 1e-20):
        actual, = pull(scale*bar)
        if exchange and screening == 'per_pair':
            # Rescaling can change FP32 classification. Test approximate
            # homogeneity at FP32 precision; the static cases test FP64.
            bound = 4*np.finfo(np.float32).eps*np.max(np.abs(reference))
            np.testing.assert_allclose(actual/scale, reference, atol=bound, rtol=0)
        else:
            np.testing.assert_allclose(actual/scale, reference, atol=2e-12, rtol=2e-12)
        if opt is not None:
            assert opt.direct_scf_tol == forward_tol
        if plan is not None:
            assert plan.accuracy is accuracy


@pytest.mark.parametrize('implicit', [True, False])
def test_scf_gradient_and_nonstationary_response(mol, monkeypatch, implicit):
    # No global three-index factorization is allowed in either direction.
    def forbidden(*args, **kwargs):
        raise AssertionError('SCF/response built global DF factors')
    monkeypatch.setattr(DF, 'build', forbidden)
    monkeypatch.setattr(scf.hf, 'gen_gmres', partial(
        linear_solver.gen_gmres_with_default_kwargs, tol=1e-11))
    native = sgx_fit(pscf.RHF(mol.to_pyscf()), auxbasis='weigend', grid_level=0,
                     max_memory=100)
    native.conv_tol, native.conv_tol_grad = 1e-12, 1e-9
    native.kernel()
    native_grad = native.nuc_grad_method().kernel()
    rng = np.random.default_rng(12)
    seed = rng.normal(size=(mol.nao, mol.nao))
    def objective(m, extra):
        mf = build(m)
        mf.kernel()
        assert mf.converged
        assert mf.with_df._cderi is None and mf._eri is None
        # Density and Fock terms are nonvariational: this needs a real SCF
        # response, and cannot pass using only the HF energy gradient.
        return (mf.e_tot + extra * (jnp.sum(seed*mf.make_rdm1())
                                    + .1*jnp.sum(seed*mf.get_fock())))
    with config_update('pyscfad_scf_implicit_diff', implicit):
        energy, gradient = jax.value_and_grad(objective)(mol, 0.)
        np.testing.assert_allclose(energy, native.e_tot, atol=2e-10)
        np.testing.assert_allclose(gradient.coords, native_grad, atol=2e-7)
        _, response = jax.value_and_grad(objective)(mol, .03)
        h = 1e-4
        fd = (objective(moved(mol, h), .03)-objective(moved(mol, -h), .03))/(2*h)
        np.testing.assert_allclose(response.coords[1, 2], fd, atol=5e-7)


def test_correlation_df_preserved_and_unsupported_modes_rejected(mol):
    with pytest.raises(ValueError, match='grid_level'):
        sgx_fit(scf.RHF(mol), grid_level=8)
    original = scf.RHF(mol).density_fit(auxbasis='def2-svp-jkfit')
    mf = sgx_fit(original, auxbasis='weigend', grid_level=0)
    assert mf.with_df is original.with_df
    assert not hasattr(original, '_sgx_backend')
    dm = np.eye(mol.nao)
    for kwargs in ({'hermi': 0}, {'omega': .3}):
        with pytest.raises(NotImplementedError):
            mf.get_jk(dm=dm, **kwargs)
    for flag in ('pyscfad_scf_first_order_custom', 'pyscfad_scf_df_precond'):
        with config_update(flag, True), pytest.raises(NotImplementedError):
            mf.kernel()
    other = mol.copy()
    other.exp = np.ones(1)
    with pytest.raises(NotImplementedError, match='coordinate derivatives'):
        build(other)
    assert mf.get_jk(dm=dm, with_j=False, with_k=False) == (None, None)
    native = mf.to_pyscf()
    native.build()
    for x, y in zip(mf.get_jk(dm=dm), native.get_jk(dm=dm)):
        np.testing.assert_allclose(x, y, atol=1e-11)


def test_stc_correlated_force_uses_sgx_response(mol, monkeypatch):
    from pyscfad import dlno_stc
    monkeypatch.setattr(scf.hf, 'gen_gmres', partial(
        linear_solver.gen_gmres_with_default_kwargs, tol=1e-11))
    calls = []
    def reference(m):
        mf = build(m)
        mf.kernel()
        calls.append(mf)
        return mf
    nodes, weights = np.polynomial.legendre.leggauss(64)
    controls = dict(mode='deterministic', laplace_roots=20*(nodes+1),
                    laplace_weights=20*weights)
    energy, gradient = dlno_stc.value_and_grad(
        mol, reference, scope='system', frozen=1, controls=controls, include_hf=True)
    assert len(calls) == 1
    h = 1e-4
    vals = []
    for x in (h, -h):
        mf = reference(moved(mol, x))
        vals.append(mf.e_tot + dlno_stc.kernel(mf, scope='system', frozen=1, controls=controls))
    np.testing.assert_allclose(gradient.coords[1, 2], (vals[0]-vals[1])/(2*h), atol=8e-7)
    assert np.isfinite(energy)


@pytest.mark.parametrize('cart', [True, False])
@pytest.mark.parametrize('fit', [True, False])
def test_direct_bilinear_kernels_match_native_gradients(cart, fit):
    from pyscf.sgx import sgx as native_sgx
    from pyscf.sgx.grad import rhf as native_grad
    from pyscf.df.grad import rhf as native_df_grad
    from pyscfad.sgx import _dfj, _sgx_grad
    m = gto.Mole(atom='O 0 0 0; H .1 -.75 .57; H 0 .8 .61; '
                      'O 0 0 8; H .1 -.75 8.57; H 0 .8 8.61',
                 basis='cc-pvdz', cart=cart, verbose=0)
    m.build(trace_exp=False, trace_ctr_coeff=False)
    rng = np.random.default_rng(107)
    dm, bar = impl._sym(rng.normal(size=(2, m.nao, m.nao)))
    mf = sgx_fit(scf.RHF(m), auxbasis='weigend', grid_level=0,
                 fit_ovlp=fit, max_memory=32)
    actual_jk = mf.get_jk(dm=dm)
    obj = mf._sgx_backend.native(m, mf.direct_scf_tol)
    native = native_sgx.SGX(m.to_pyscf(), auxbasis='weigend')
    native.sgx_tol_energy = native.sgx_tol_potential = None
    native.fit_ovlp, native._grid_response = fit, True
    native.build(level=0)
    for actual, expected in zip(actual_jk, native.get_jk(dm)):
        np.testing.assert_allclose(actual, expected, atol=2e-10)
    expected_k = .5 * (native_grad.get_k_grad_only(native, dm+bar).aux.sum((0, 1))
                       - native_grad.get_k_grad_only(native, dm-bar).aux.sum((0, 1)))
    reference = pscf.RHF(native.mol).density_fit()
    reference.with_df = native
    v = native_df_grad.get_j(reference.nuc_grad_method(), dm=np.stack((dm, bar)))
    expected_j = v.aux[0, 1] + v.aux[1, 0]
    for a, (_, _, p, q) in enumerate(m.aoslice_by_atom()):
        expected_j[a] += 2 * (np.einsum('xij,ij->x', v[0, :, p:q], bar[p:q])
                              + np.einsum('xij,ij->x', v[1, :, p:q], dm[p:q]))
    np.testing.assert_allclose(_dfj.geometry(obj, dm, bar), expected_j, atol=3e-9)
    np.testing.assert_allclose(_sgx_grad.geometry(obj, dm, bar, mf.direct_scf_tol), expected_k, atol=3e-9)


def test_response_reuses_factors_and_respects_metric_tile_cap(mol, monkeypatch):
    from pyscfad.sgx import _dfj, _sgx_grad
    from pyscf.df.grad import rhf as native_df_grad
    from pyscf.sgx.grad import rhf as native_grad
    import scipy.linalg
    mf = build(mol)
    rng = np.random.default_rng(19)
    dm, bar = impl._sym(rng.normal(size=(2, mol.nao, mol.nao)))
    mf.get_jk(dm=dm)
    obj = mf._sgx_backend.native(mol, mf.direct_scf_tol)
    reference_j = _dfj.geometry(obj, dm, bar)
    reference_k = _sgx_grad.geometry(obj, dm, bar, mf.direct_scf_tol)
    def forbidden(*args, **kwargs):
        raise AssertionError('response rebuilt a factorization or used the old gradient')
    monkeypatch.setattr(scipy.linalg, 'lu_factor', forbidden)
    monkeypatch.setattr(scipy.linalg, 'cho_factor', forbidden)
    monkeypatch.setattr(np.linalg, 'solve', forbidden)
    monkeypatch.setattr(native_df_grad, 'get_j', forbidden)
    monkeypatch.setattr(native_grad, 'get_k_grad_only', forbidden)
    intor = obj.auxmol.intor
    allocated = []
    def metric(name, *args, **kwargs):
        assert name == 'int2c2e_ip1', 'response rebuilt the auxiliary metric'
        assert kwargs.get('shls_slice') is not None
        v = intor(name, *args, **kwargs)
        allocated.append(v.nbytes)
        return v
    monkeypatch.setattr(obj.auxmol, 'intor', metric)
    # Deliberately tiny metric-tile allowance; no 20-auxiliary minimum and
    # no three-index panel. Grid workspace retains its independent minimum.
    obj._response_workspace_bytes = 4096
    for factor in (1., -1e-8):
        np.testing.assert_allclose(_dfj.geometry(obj, dm, factor*bar),
                                   factor*reference_j, atol=1e-10)
        np.testing.assert_allclose(_sgx_grad.geometry(obj, dm, factor*bar, mf.direct_scf_tol),
                                   factor*reference_k, atol=1e-10)
    assert allocated and max(allocated) <= 4096


def test_grid_workspace_never_silently_exceeds_minimum(mol):
    mf = build(mol)
    data = mf._sgx_backend.native(mol, mf.direct_scf_tol)._pjs_data
    data._response_workspace_bytes = 1
    with pytest.raises(MemoryError, match='grid block'):
        data.get_loop_data(nset=2, grad=True)


def test_response_thread_count_does_not_change_result(mol):
    from pyscf import lib
    from pyscfad.sgx import _dfj, _sgx_grad
    from threadpoolctl import threadpool_limits
    mf = build(mol)
    rng = np.random.default_rng(71)
    dm, bar = impl._sym(rng.normal(size=(2, mol.nao, mol.nao)))
    mf.get_jk(dm=dm)
    obj = mf._sgx_backend.native(mol, mf.direct_scf_tol)
    values = []
    for threads in (1, 2, 4):
        with threadpool_limits(threads), lib.with_omp_threads(threads):
            values.append((_dfj.geometry(obj, dm, bar),
                           _sgx_grad.geometry(obj, dm, bar, mf.direct_scf_tol)))
    for result in values[1:]:
        np.testing.assert_allclose(result, values[0], atol=2e-10)


@pytest.mark.parametrize('native_batch', [128, 224])
def test_large_grid_plan_keeps_threads_busy_with_available_memory(native_batch):
    from pyscfad.sgx._sgx_grad import _grid_plan
    # Size planning only: no water200 integrals or SCF are run here.
    plan = _grid_plan(11600, 600, 2_000_000, 2, True, 32,
                      500_000_000_000, 100_000_000_000, native_batch=native_batch)
    assert plan['target_grid_jobs'] == 128
    assert 64 <= plan['selected_grid_jobs'] <= 128
    assert plan['block_points'] % native_batch == 0
    assert plan['block_points'] % 224 == 0
    assert plan['block_points'] % 56 == 0
    assert plan['selected_grid_jobs'] == plan['block_points']//native_batch
    assert plan['grid_bytes'] > 256*1024**2
    reserved = plan['grid_bytes'] + plan['fixed_reserve_bytes'] + plan['thread_scratch_bytes']
    assert reserved <= .8*plan['live_headroom_bytes']


@pytest.mark.parametrize('generated', [False, True])
def test_grid_plan_reclaims_released_memory_and_obeys_workspace_cap(mol, monkeypatch, generated):
    from types import SimpleNamespace
    from pyscf import lib
    from pyscf.sgx import sgx_jk
    from pyscfad.sgx import _sgx_grad
    mf = build(mol)
    data = mf._sgx_backend.native(mol, mf.direct_scf_tol)._pjs_data
    data._generated_plan = (SimpleNamespace(grid_batch_points=128,
        scratch_bytes=lambda *a, **k: 0) if generated else None)
    # Exercise the live planner with production-sized dimensions while
    # replacing only PySCF's metadata lookup (no large arrays allocated).
    data._mol = SimpleNamespace(nao=928, natm=48)
    data._grids = SimpleNamespace(weights=range(160000))
    data._response_workspace_bytes = 10_000_000_000
    data._response_total_memory_mb = 12000
    monkeypatch.setattr(sgx_jk.SGXData, 'get_loop_data',
                        lambda *a, **k: (0, None, None, None, 160000))
    monkeypatch.setattr(_sgx_grad, '_system_available_bytes', lambda: 100_000_000_000)
    monkeypatch.setattr(lib, 'num_threads', lambda: 32)
    monkeypatch.setattr(lib, 'current_memory', lambda: (11000., 0.))
    small = data.get_loop_data(nset=2, grad=True)[-1]
    monkeypatch.setattr(lib, 'current_memory', lambda: (1000., 0.))
    large = data.get_loop_data(nset=2, grad=True)[-1]
    assert large > small
    assert data._response_grid_plan['selected_grid_jobs'] >= 64
    from pyscfad.lib import _solver_memory
    with _solver_memory.reserve_gmres(100_000_000, np.float64, 4) as solver:
        limited = data.get_loop_data(nset=2, grad=True)[-1]
        assert limited < large
        assert data._response_grid_plan['solver_reserve_bytes'] == solver['reserved_bytes']
    assert data.get_loop_data(nset=2, grad=True)[-1] == large
    assert data._response_grid_plan['solver_reserve_bytes'] == 0
    data._response_workspace_bytes = 600_000_000
    capped = data.get_loop_data(nset=2, grad=True)[-1]
    assert capped < large
    plan = data._response_grid_plan
    assert plan['grid_bytes'] + plan['fixed_reserve_bytes'] + plan['thread_scratch_bytes'] <= 600_000_000
    assert plan['native_batch_points'] == (128 if generated else 224)
    # PySCF's additional cap need not be aligned to generated-kernel tiles.
    data._response_workspace_bytes = 10_000_000_000
    monkeypatch.setattr(sgx_jk.SGXData, 'get_loop_data',
                        lambda *a, **k: (0, None, None, None, 1420))
    clipped = data.get_loop_data(nset=2, grad=True)[-1]
    assert clipped == (896 if generated else 1344)
    assert data._response_grid_plan['selected_grid_jobs'] == clipped//plan['native_batch_points']
    monkeypatch.setattr(_sgx_grad, '_system_available_bytes', lambda: 1000)
    with pytest.raises(MemoryError, match='grid block'):
        data.get_loop_data(nset=2, grad=True)


def test_generated_grid_plan_preserves_tiles_and_allows_final_tail():
    from pyscfad.sgx._sgx_grad import _grid_plan
    args = (1856, 96, 276912, 2, True, 32)
    plan = _grid_plan(*args, 500_000_000_000, 900_000_000_000, native_batch=128)
    assert plan['block_points'] == 16128
    assert plan['selected_grid_jobs'] == 126
    assert plan['block_alignment_points'] == 896
    # A partial final panel is allowed, but an intermediate boundary must
    # not split either the AO mask or a screened integral tile.
    tail = _grid_plan(100, 3, 13, 1, False, 1, 10**9, 10**9, native_batch=128)
    assert tail['block_points'] == 13
    assert tail['selected_grid_jobs'] == 1
    limited = (plan['fixed_reserve_bytes'] + plan['thread_scratch_bytes']
               + 895*plan['bytes_per_grid'])
    with pytest.raises(MemoryError, match='aligned native grid block'):
        _grid_plan(*args, limited, 900_000_000_000, native_batch=128)
