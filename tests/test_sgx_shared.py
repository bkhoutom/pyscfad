"""Fused SGX/DF-J geometry, exact seed grouping and complete SCF VJPs."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import lib
from pyscf.scf import _vhf
from pyscf.sgx import sgx_jk

from pyscfad import config_update, scf
from pyscfad.dlno_stc import driver
from pyscfad.sgx import _shared, _dfj, _sgx_grad, _native
from pyscfad.sgx.sgx import _sym
from pyscfad.tools import linear_solver
from .test_sgx import mol, build, moved


@pytest.mark.parametrize('fit', [True, False])
def test_joint_bilinear_geometry_matches_finite_difference_and_separate_passes(mol, monkeypatch, fit):
    monkeypatch.setattr(sgx_jk, 'SWITCH_SIZE', 0)
    mf = build(mol, fit_ovlp=fit)
    rng = np.random.default_rng(789)
    ds, bs = _sym(rng.normal(size=(2, 3, mol.nao, mol.nao)))
    seeds = list(zip(ds, bs))
    obj = mf._sgx_backend.native(mol, mf.direct_scf_tol)
    for k, kernel in enumerate((_dfj.geometry_many, _sgx_grad.geometry_many)):
        args = () if k == 0 else (mf.direct_scf_tol,)
        expected = sum(kernel(obj, [seed], *args) for seed in seeds)
        actual = kernel(obj, seeds, *args)
        np.testing.assert_allclose(actual, expected, atol=3e-10)
        def objective(m):
            values = mf.get_jk(m, ds, with_j=k == 0, with_k=k == 1)[k]
            return np.sum(values*bs)
        h = 1e-4
        fd = (objective(moved(mol, h))-objective(moved(mol, -h)))/(2*h)
        np.testing.assert_allclose(actual[1, 2], fd, atol=3e-6, rtol=1e-6)
        np.testing.assert_allclose(actual.sum(axis=0), 0, atol=2e-9)


@pytest.mark.parametrize('coordinate_scale', [1., 1.03])
@pytest.mark.parametrize('joint_outputs', [False, True])
def test_complete_scf_merges_geometry_and_reuses_tape(
        mol, monkeypatch, coordinate_scale, joint_outputs):
    monkeypatch.setattr(sgx_jk, 'SWITCH_SIZE', 0)
    monkeypatch.setattr(scf.hf, 'gen_gmres', partial(
        linear_solver.gen_gmres_with_default_kwargs, tol=1e-11))
    rng = np.random.default_rng(167)
    seed = rng.normal(size=(mol.nao, mol.nao))
    def reference(current):
        mapped = current.copy()
        mapped.coords = current.coords*coordinate_scale
        mf = build(mapped)
        mf.kernel()
        assert mf.converged
        return mf
    def objective(mf):
        return (mf.e_tot + .01*jnp.sum(seed*mf.get_fock())
                + .02*jnp.sum(mf.mo_energy) + .003*jnp.sum(mf.mo_coeff))
    calls, receipts = [], []
    gen = sgx_jk._gen_k_direct
    generated = _native.IntegralPlan.contract_with_grad
    finish = _shared.GeometryScope.finish
    def traced_gen(*args, **kwargs):
        if kwargs.get('grad'):
            calls.append('derivative_integral_traversal')
        return gen(*args, **kwargs)
    def traced_generated(*args, **kwargs):
        calls.append('derivative_integral_traversal')
        return generated(*args, **kwargs)
    def finished(scope, bar):
        result = finish(scope, bar)
        receipts.append(dict(scope.stats))
        return result
    monkeypatch.setattr(sgx_jk, '_gen_k_direct', traced_gen)
    monkeypatch.setattr(_native.IntegralPlan, 'contract_with_grad', traced_generated)
    monkeypatch.setattr(_shared.GeometryScope, 'finish', finished)
    values, work = [], []
    with config_update('pyscfad_scf_implicit_diff', True):
        for enabled in (False, True):
            mf, scf_pull = driver._initialize_scf(
                mol, reference, combine_df=enabled, share_scf_data=joint_outputs)
            data_seed = ((jnp.zeros_like(scf_pull.scf_data[0]), .007*seed)
                         if joint_outputs else None)
            value, pull = jax.vjp(objective, mf)
            calls.clear()
            with driver._global_df_response({'mf': mf}):
                mf_bar, = pull(jnp.ones_like(value))
                mol_bar, = (scf_pull(mf_bar, data_seed) if joint_outputs else scf_pull(mf_bar))
            values.append((value, mol_bar.coords))
            work.append(len(calls))
        np.testing.assert_allclose(values[1][0], values[0][0], atol=1e-12)
        np.testing.assert_allclose(values[1][1], values[0][1], atol=3e-9)
        assert work[0] >= 3 and work[1] == 1
        assert receipts[-1]['consumers'] >= 3
        assert receipts[-1]['j_integral_passes'] == receipts[-1]['k_integral_passes'] == 1
        assert receipts[-1]['retained_seed_bytes'] == 0
        assert receipts[-1]['j_geometry_seconds'] > 0
        assert receipts[-1]['k_geometry_seconds'] > 0
        assert {batch['kind'] for batch in receipts[-1]['geometry_batches']} == {'j', 'k'}
        # A partial reverse must report and clear pending seeds. The same
        # immutable tape then remains usable with a different output seed.
        with pytest.raises(RuntimeError, match='unconsumed SGX geometry'):
            with _shared.producer_scope(mf):
                pull(jnp.ones_like(value))
        assert _shared._scope.get() is None
        with driver._global_df_response({'mf': mf}):
            mf_bar, = pull(jnp.asarray(-.4))
            repeated, = (scf_pull(mf_bar, jax.tree_util.tree_map(lambda x: -.4*x, data_seed))
                         if joint_outputs else scf_pull(mf_bar))
        np.testing.assert_allclose(repeated.coords, -.4*values[0][1], atol=3e-9)
    assert _shared._scope.get() is None


def test_geometry_screening_receipt_is_per_call_without_resetting_plan(mol):
    mf = build(mol, integral_backend='generated', screening='per_pair',
               mixed_precision=True, energy_tol=1e-10, potential_tol=1e-8,
               fp32_multiplier=1e5)
    obj = mf._sgx_backend.native(mol, mf.direct_scf_tol)
    rng = np.random.default_rng(183)
    dm, bar = _sym(rng.normal(size=(2, mol.nao, mol.nao)))
    plan = _native.get_plan(obj._pjs_data)
    before = plan.screening_stats()
    first = _sgx_grad.geometry_many(obj, [(dm, bar)], mf.direct_scf_tol)
    receipt = obj._pjs_data._geometry_stats
    first_counts = receipt['screening_counts']
    assert receipt['screening_counts_available']
    assert sum(first_counts[k] for k in ('skipped_tiles', 'fp32_tiles', 'fp64_tiles')) > 0
    second = _sgx_grad.geometry_many(obj, [(dm, bar)], mf.direct_scf_tol)
    assert obj._pjs_data._geometry_stats['screening_counts'] == first_counts
    assert plan.screening_stats() == {k: before[k]+2*v for k, v in first_counts.items()}
    np.testing.assert_allclose(first, second, atol=2e-13, rtol=2e-13)


def test_scope_groups_only_equal_densities_and_ignores_other_geometry(mol):
    dm = np.eye(mol.nao)
    with _shared.capture_producers():
        mf, _ = jax.vjp(lambda m: build(m), mol)
    with _shared.producer_scope(mf) as scope:
        assert _shared.defer(mf._sgx_backend, mf.mol, dm, dm, dm, 1e-13)
        assert _shared.defer(mf._sgx_backend, mf.mol, dm.copy(), -dm, 2*dm, 1e-13)
        changed = dm.copy()
        changed[0, 0] += 1e-14
        assert _shared.defer(mf._sgx_backend, mf.mol, changed, dm, dm, 1e-13)
        assert len(scope.pending) == 2
        np.testing.assert_array_equal(scope.pending[0]['j'], 0)
        np.testing.assert_array_equal(scope.pending[0]['k'], 3*dm)
        assert not _shared.defer(mf._sgx_backend, moved(mol, .01), dm, dm, dm, 1e-13)
        bar = jax.tree_util.tree_map(np.zeros_like, mf.mol)
        actual = scope.finish(bar)
    assert np.isfinite(actual.coords).all()
    assert scope.stats['j_integral_passes'] == scope.stats['k_integral_passes'] == 1


def test_forward_and_geometry_kernels_disable_dynamic_teams(mol, monkeypatch):
    omp = _vhf.libcvhf
    previous = omp.omp_get_dynamic()
    seen = []
    original = sgx_jk._gen_k_direct
    original_generated = _native.IntegralPlan.contract_with_grad
    original_j = _dfj._contract
    def check():
        seen.append((lib.num_threads(), omp.omp_get_dynamic()))
    def gen(*args, **kwargs):
        fn = original(*args, **kwargs)
        def call(*a, **k):
            check()
            return fn(*a, **k)
        return call
    def j(*args, **kwargs):
        check()
        return original_j(*args, **kwargs)
    def generated(*args, **kwargs):
        check()
        return original_generated(*args, **kwargs)
    monkeypatch.setattr(sgx_jk, '_gen_k_direct', gen)
    monkeypatch.setattr(_native.IntegralPlan, 'contract_with_grad', generated)
    monkeypatch.setattr(_dfj, '_contract', j)
    try:
        omp.omp_set_dynamic(1)
        with lib.with_omp_threads(2):
            mf = build(mol)
            fun = lambda m: sum(jnp.sum(x) for x in mf.get_jk(m, np.eye(m.nao)))
            jax.value_and_grad(fun)(mol)
            assert seen and set(seen) == {(2, 0)}
            assert omp.omp_get_dynamic() == 1
    finally:
        omp.omp_set_dynamic(previous)
