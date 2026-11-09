"""Mixed SGX accuracy, screening, and approximate response validation."""
import numpy as np
import pytest
from pyscf import gto, scf
from scipy.special import hyp1f1

from pyscfad.sgx import _native, _sgx_grad, sgx_fit


@pytest.fixture(scope='module', autouse=True)
def mixed_library():
    dll = _native.library()
    if dll is None or dll.SGX_generated_version() < 3:
        pytest.skip('libsgx_ad version 3 required')
    return dll


def test_float_boys(mixed_library):
    rng = np.random.default_rng(723)
    for n in range(10):
        for size in (1, 7, 127, 128):
            x = rng.uniform(0., 65., size)
            x[::7] = np.geomspace(64., 1e300, len(x[::7]))
            out = np.empty((n+1, size), dtype=np.float32)
            assert mixed_library.SGX_generated_boys_float(n, size, _native._ptr(x), _native._ptr(out)) == 0
            orders = np.arange(n+1)[:, None]
            expected = hyp1f1(orders+.5, orders+1.5, -x)/(2*orders+1)
            np.testing.assert_allclose(out, expected, atol=3e-44, rtol=3e-6)
            assert np.isfinite(out).all()


@pytest.mark.parametrize('same', [False, True])
@pytest.mark.parametrize('depth', [1, 2])
def test_float_s_through_g(same, depth):
    basis = ([[l, [.9, 1.]] for l in range(5)] if depth == 1 else
             [[l, [.9, .6, -.2], [.2, .3, .7]] for l in range(5)])
    mol = gto.M(atom='He 0 0 0; He .4 -.6 1.2', unit='Bohr',
                basis={'He': basis}, verbose=0)
    plan = _native.IntegralPlan(mol, np.ones((mol.nbas, mol.nbas)), 0., 10**9)
    coords = np.random.default_rng(21).normal(size=(131, 3))*3
    coords[:3] = [[0, 0, 0], [.4, -.6, 1.2], [200., 50., -100.]]
    for la in range(5):
        for lb in range(5):
            ia, ib = la, lb+(0 if same else 5)
            val = plan.shell_pair(ia, ib, coords, precision='float')
            fused, grad = plan.shell_pair_with_grad(ia, ib, coords, precision='float')
            expected = mol.intor('int1e_grids_cart', grids=coords,
                shls_slice=(ia, ia+1, ib, ib+1)).transpose(1, 2, 0)
            left = mol.intor('int1e_grids_ip_cart', grids=coords,
                shls_slice=(ia, ia+1, ib, ib+1)).transpose(0, 2, 3, 1)
            right = mol.intor('int1e_grids_ip_cart', grids=coords,
                shls_slice=(ib, ib+1, ia, ia+1)).transpose(0, 3, 2, 1)
            np.testing.assert_allclose(val, expected, atol=2e-6, rtol=2e-5)
            np.testing.assert_allclose(fused, expected, atol=2e-6, rtol=2e-5)
            np.testing.assert_allclose(grad, [left, right], atol=5e-6, rtol=3e-5)


def test_float_derivative_cancellation_regression():
    # The cc-pVTZ H--O d--d counterexample has only one primitive per shell.
    # Contracted HRR loses accuracy here even without primitive cancellation;
    # keep FP32 derivative VRR while independently selecting forward HRR.
    mol = gto.M(atom=[('He1', (0., 0., 0.)),
                      ('He2', (1.6222140912069192, -.030649837644750703,
                               .8145791678190131))], unit='Bohr',
                basis={'He1': [[2, [2.314, 1.]]],
                       'He2': [[2, [1.057, 1.]]]}, verbose=0)
    plan = _native.IntegralPlan(mol, np.ones((2, 2)), 0., 10**8)
    coords = np.random.default_rng(934).normal(size=(17, 3))*2
    value, gradient = plan.shell_pair_with_grad(1, 0, coords, precision='float')
    expected = mol.intor('int1e_grids_cart', grids=coords,
                        shls_slice=(1, 2, 0, 1)).transpose(1, 2, 0)
    left = mol.intor('int1e_grids_ip_cart', grids=coords,
                    shls_slice=(1, 2, 0, 1)).transpose(0, 2, 3, 1)
    right = mol.intor('int1e_grids_ip_cart', grids=coords,
                     shls_slice=(0, 1, 1, 2)).transpose(0, 3, 2, 1)
    np.testing.assert_allclose(value, expected, atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(gradient, [left, right], atol=5e-6, rtol=3e-5)
    np.testing.assert_allclose(plan.shell_pair(1, 0, coords, precision='float'),
                               expected, atol=2e-6, rtol=2e-5)


def test_float_g_accumulation_and_signed_weights():
    mol = gto.M(atom='He 0 0 0; He 0 0 3', basis='cc-pvdz', verbose=0)
    plan = _native.IntegralPlan(mol, np.ones((mol.nbas, mol.nbas)), 0., 10**8)
    rng = np.random.default_rng(27)
    coords = rng.normal(size=(137, 3))
    f = rng.normal(size=(3, mol.nao, len(coords)))
    weights = rng.uniform(-1, 1, len(coords))
    ao_l1 = np.sum(np.abs(mol.eval_gto('GTOval', coords)), axis=1)
    reference = plan.contract_with_grad(mol, coords, f)
    # Force every ordinary test pair into FP32 to test the actual float G/dG
    # path, independently of whether a physical density selects those pairs.
    plan.configure_accuracy(mixed=True, energy_tol=1e5, potential_tol=1e5,
                            fp32_multiplier=1e10, ngrids=len(coords))
    actual = plan.contract_with_grad(mol, coords, f, weights, ao_l1=ao_l1)
    assert plan.screening_stats()['fp64_tiles'] == 0
    assert plan.screening_stats()['fp32_tiles'] > 0
    for a, b in zip(actual, reference):
        np.testing.assert_allclose(a, b, atol=4e-6, rtol=2e-5)


@pytest.mark.parametrize('la,lb,depth,same,promote', [
    (3, 3, 4, False, True), (4, 4, 2, False, True),
    (3, 2, 8, False, True), (4, 3, 4, True, True),
    (3, 3, 4, True, False), (2, 2, 8, True, False),
    (3, 3, 1, False, False), (1, 1, 4, False, False),
])
def test_derivative_cross_precision_dispatch(la, lb, depth, same, promote):
    # One retained shell pair isolates executed-precision counters. Signed
    # general contractions exercise the exponent weights in derivative HRR.
    def shell(l):
        return [[l, *[[float(e), (-1.)**k*.2, .1+k*.02]
                       for k, e in enumerate(np.geomspace(.2, 2., depth))]]]
    mol = gto.M(atom=[('He1', (0., 0., 0.)),
                      ('He2', (0., 0., 0.) if same else (.4, -.6, 1.2))],
                unit='Bohr', basis={'He1': shell(la), 'He2': shell(lb)}, verbose=0)
    plan = _native.IntegralPlan(mol, np.array([[0., 1.], [1., 0.]]), 0., 10**8)
    rng = np.random.default_rng(918)
    coords = rng.normal(size=(7, 3))
    fg = rng.normal(size=(2, mol.nao, len(coords)))
    weights = rng.uniform(.1, 1., len(coords))
    ao_l1 = np.sum(np.abs(mol.eval_gto('GTOval', coords)), axis=1)
    reference = plan.contract_with_grad(mol, coords, fg)
    settings = dict(mixed=True, fp32_multiplier=1e5, ngrids=len(coords),
                    screening_mode='per_pair')
    # Large thresholds make this pair FP32-eligible without skipping it.
    plan.configure_accuracy(**settings, energy_tol=1e50, potential_tol=1e50)
    plan.reset_stats()
    actual = plan.contract_with_grad(mol, coords, fg, weights, ao_l1=ao_l1)
    stats = plan.screening_stats()
    assert stats['skipped_tiles'] == 0
    assert stats['fp64_tiles'] == int(promote)
    assert stats['fp32_tiles'] == int(not promote)
    for a, b in zip(actual, reference):
        np.testing.assert_allclose(a, b, atol=2e-12 if promote else 5e-6,
                                   rtol=2e-12 if promote else 3e-5)
    # Forward selection remains FP32 for the same eligible pair.
    plan.reset_stats()
    plan.contract(mol, coords, fg, weights, ao_l1=ao_l1)
    assert plan.screening_stats()['fp32_tiles'] == 1
    assert plan.screening_stats()['fp64_tiles'] == 0
    # Precision-required work cannot be demoted, even for low angular momenta.
    plan.configure_accuracy(**settings, energy_tol=1e-100, potential_tol=1e-100)
    plan.reset_stats()
    actual = plan.contract_with_grad(mol, coords, fg, weights, ao_l1=ao_l1)
    assert plan.screening_stats()['fp64_tiles'] == 1
    for a, b in zip(actual, reference):
        np.testing.assert_allclose(a, b, atol=2e-12, rtol=2e-12)
    # Skipping precedes performance promotion and stays unchanged.
    plan.configure_accuracy(**settings, screening=True,
                            energy_tol=1e50, potential_tol=1e50)
    plan.reset_stats()
    value, gradient = plan.contract_with_grad(mol, coords, fg, weights, ao_l1=ao_l1)
    assert plan.screening_stats()['skipped_tiles'] == 1
    assert not np.any(value) and not np.any(gradient)


@pytest.mark.parametrize('mixed', [False, True])
@pytest.mark.parametrize('mode', ['cumulative', 'per_pair'])
def test_adaptive_force_and_signed_response(mixed, mode):
    mol = gto.M(atom='O 0 0 0; H 0 -.7 .5; H 0 .7 .5; '
                    'O 0 0 4; H 0 -.7 4.5; H 0 .7 4.5',
                basis='cc-pvdz', verbose=0, max_memory=2000)
    mf = sgx_fit(scf.RHF(mol), grid_level=0, max_memory=500, integral_backend='generated')
    mf.build()
    obj = mf.with_df
    dm = scf.RHF(mol).get_init_guess()
    bar = np.random.default_rng(72).normal(size=dm.shape)/mol.nao
    bar = (bar+bar.T)*.5
    reference_k = mf.get_k(dm=np.stack([dm, bar]))
    plan = _native.get_plan(obj._pjs_data)
    reference_gradient = _sgx_grad.geometry(obj, dm, bar, 1e-13)
    settings = dict(screening=True, mixed=mixed, fp32_multiplier=1e7,
                    ngrids=len(obj.grids.weights))
    if mode == 'per_pair':
        settings.update(screening_mode=mode, energy_tol=1e-10, potential_tol=1e-8,
                        average_weight=float(np.mean(np.abs(obj.grids.weights))),
                        fp32_multiplier=1e5)
    plan.configure_accuracy(**settings)
    actual_k = mf.get_k(dm=np.stack([dm, bar]))
    plan.reset_stats()
    gradient = _sgx_grad.geometry(obj, dm, bar, 1e-13)
    np.testing.assert_allclose(actual_k, reference_k, atol=1e-7, rtol=0)
    np.testing.assert_allclose(gradient, reference_gradient, atol=1e-12, rtol=0)
    assert plan.screening_stats()['skipped_tiles'] > 0
    assert not mixed or plan.screening_stats()['fp32_tiles'] > 0
    if mixed:
        # Precision tuning must never change which contributions are skipped.
        skipped = plan.screening_stats()['skipped_tiles']
        plan.configure_accuracy(**dict(settings, fp32_multiplier=1e9))
        plan.reset_stats()
        _sgx_grad.geometry(obj, dm, bar, 1e-13)
        assert plan.screening_stats()['skipped_tiles'] == skipped
        plan.configure_accuracy(**settings)
    # Classification and FP32 rounding may change with a signed trial vector.
    combined = mf.get_k(dm=dm+bar)
    np.testing.assert_allclose(combined, actual_k.sum(axis=0), atol=1e-7, rtol=0)
    adjoint_error = abs(np.sum(bar*actual_k[0])-np.sum(dm*actual_k[1]))
    assert adjoint_error < 1e-7
    np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=1e-12)


@pytest.mark.parametrize('mode,multiplier,gradient_atol,mixed', [
    ('adaptive', 1e5, 1e-12, True),
    # Mixed density pullbacks retain the nominal FP32 classification but
    # disable density-dependent skipping for tiny Krylov solution vectors.
    # This nonstationary density/Fock objective has a much larger response
    # than physical forces. Bound its mixed error by the requested solver
    # tolerance, and check FP64-only using the same per-pair skip policy.
    ('per_pair', 1e5, 1e-11, True),
    ('per_pair', 1e4, 1e-12, True),
    ('per_pair', 1e5, 1e-12, False),
])
def test_adaptive_scf_implicit_response(monkeypatch, mode, multiplier, gradient_atol, mixed):
    import jax
    import jax.numpy as jnp
    from functools import partial
    from pyscfad import gto as adgto, scf as adscf, config_update
    from pyscfad.tools import linear_solver
    monkeypatch.setattr(adscf.hf, 'gen_gmres', partial(
        linear_solver.gen_gmres_with_default_kwargs, tol=1e-11))
    mol = adgto.Mole(atom='O 0 0 0; H .1 -1.4 1.1; H 0 1.5 1',
                    unit='Bohr', basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    seed = np.random.default_rng(12).normal(size=(mol.nao, mol.nao))
    def objective(m, adaptive):
        # Forward and response use the same public precision setting.
        mf = sgx_fit(adscf.RHF(m), auxbasis='weigend', grid_level=0,
                     max_memory=100, integral_backend='generated',
                     screening=mode if adaptive else 'static',
                     mixed_precision=adaptive and mixed,
                     fp32_multiplier=multiplier)
        mf.conv_tol, mf.conv_tol_grad, mf.chkfile = 1e-13, 1e-10, None
        mf.kernel()
        assert mf.converged
        return mf.e_tot + .03*(jnp.sum(seed*mf.make_rdm1())+.1*jnp.sum(seed*mf.get_fock()))
    with config_update('pyscfad_scf_implicit_diff', True):
        ref, gref = jax.value_and_grad(objective)(mol, False)
        value, gradient = jax.value_and_grad(objective)(mol, True)
        np.testing.assert_allclose(value, ref, atol=1e-8, rtol=0)
        np.testing.assert_allclose(gradient.coords, gref.coords, atol=gradient_atol, rtol=0)
        h = 2e-4
        coords = np.asarray(mol.coords).copy()
        delta = np.zeros_like(coords);delta[1, 2] = h
        plus = mol.set_geom_(coords+delta, unit='Bohr', inplace=False)
        minus = mol.set_geom_(coords-delta, unit='Bohr', inplace=False)
        fd = (objective(plus, True)-objective(minus, True))/(2*h)
        np.testing.assert_allclose(gradient.coords[1, 2], fd, atol=1e-6, rtol=0)


def test_per_pair_independent_of_total_grid_count():
    """Replicating identical tiles must not tighten local classification."""
    mol = gto.M(atom='He 0 0 0; He 0 0 3', basis='cc-pvdz', verbose=0)
    plan = _native.IntegralPlan(mol, np.ones((mol.nbas, mol.nbas)), 0., 10**8)
    rng = np.random.default_rng(381)
    coords = rng.normal(size=(128, 3))
    f = rng.normal(size=(1, mol.nao, 128))*np.geomspace(1e-10, 1., mol.nao)[None, :, None]
    weights = rng.uniform(.1, 1., 128)
    ao = np.sum(np.abs(mol.eval_gto('GTOval', coords)), axis=1)
    def run(repeat):
        plan.configure_accuracy(screening=True, mixed=True, screening_mode='per_pair',
            energy_tol=1e-6, potential_tol=1e-4, fp32_multiplier=1e3,
            ngrids=128*repeat, average_weight=float(np.mean(weights)))
        plan.reset_stats()
        value = plan.contract(mol, np.tile(coords, (repeat, 1)),
            np.tile(f, (1, 1, repeat)), np.tile(weights, repeat), ao_l1=np.tile(ao, repeat))
        return value, plan.screening_stats()
    single, counts = run(1)
    triple, tripled = run(3)
    np.testing.assert_array_equal(triple, np.tile(single, (1, 1, 3)))
    assert all(tripled[k] == 3*v for k, v in counts.items())
    assert counts['skipped_tiles'] > 0
    assert counts['fp32_tiles'] > 0
    assert counts['fp64_tiles'] > 0


@pytest.mark.parametrize('mixed', [False, True])
def test_response_follows_precision_setting_and_restores_screening(mixed, monkeypatch):
    mol = gto.M(atom='He 0 0 0; He 0 0 3', basis='cc-pvdz', verbose=0)
    # Force FP32 classification to prove response honors the public flag,
    # while the equally generous skipping thresholds must never drop tiles.
    settings = dict(grid_level=0, max_memory=500, integral_backend='generated',
                    screening='per_pair', energy_tol=1e5, potential_tol=1e5,
                    fp32_multiplier=1e5)
    mf = sgx_fit(scf.RHF(mol), mixed_precision=mixed, **settings)
    mf.build()
    dm = scf.RHF(mol).get_init_guess()
    mf.get_k(dm=dm)
    obj = mf.with_df
    plan = _native.get_plan(obj._pjs_data)
    accuracy = plan.accuracy
    plan.reset_stats()
    actual = _sgx_grad.density_response(obj, dm, mf.direct_scf_tol)
    counts = plan.screening_stats().copy()
    assert counts['skipped_tiles'] == 0
    assert (counts['fp32_tiles'] > 0) == mixed
    assert plan.accuracy is accuracy
    # Reference calculations use a separate FP64-only backend, without a
    # response-only override on the mixed object.
    fp64 = sgx_fit(scf.RHF(mol), mixed_precision=False, **settings)
    fp64.build()
    reference = _sgx_grad.density_response(fp64.with_df, dm, fp64.direct_scf_tol)
    assert plan.screening_stats() == counts
    np.testing.assert_allclose(actual, reference, atol=3e-6, rtol=3e-6)
    assert plan.accuracy is accuracy
    def failed(*a, **k):
        if mixed:
            assert plan.accuracy['mixed']
            assert not plan.accuracy['screening']
        else:
            assert plan.accuracy is None
        raise RuntimeError('test response failure')
    monkeypatch.setattr(_sgx_grad, 'get_k', failed)
    with pytest.raises(RuntimeError, match='test response failure'):
        _sgx_grad.density_response(obj, dm, mf.direct_scf_tol)
    assert plan.accuracy is accuracy
