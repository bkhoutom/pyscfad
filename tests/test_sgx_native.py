"""Independent numerical checks of the portable generated SGX engine."""
from types import SimpleNamespace
import numpy as np
import pytest
from pyscf import gto, scf
from scipy.special import hyp1f1

from pyscfad.sgx import _native, sgx_fit


@pytest.fixture(scope='module')
def dll():
    result = _native.library()
    if result is None:
        pytest.skip('libsgx_ad has not been built')
    return result


def test_boys_all_orders(dll):
    # Table midpoints, the asymptotic boundary, and mixed/far-field tiles.
    samples = [np.linspace(0., 64., 128),
               np.arange(128)/8 + 1/16,
               np.array([0., 1e-14, .01, 63.99999, 64., 64.00001, 1e3, 1e8]),
               np.geomspace(64., 1e8, 128)]
    for x in samples:
        for n in range(10 if dll.SGX_generated_version() >= 2 else 9):
            out = np.empty((n+1, len(x)))
            assert dll.SGX_generated_boys(n, len(x), _native._ptr(x), _native._ptr(out)) == 0
            orders = np.arange(n+1)[:, None]
            expected = hyp1f1(orders+.5, orders+1.5, -x)/(2*orders+1)
            np.testing.assert_allclose(out, expected, rtol=8e-14, atol=2e-16)
    x = np.array([-1.])
    out = np.empty(9)
    assert dll.SGX_generated_boys(8, 1, _native._ptr(x), _native._ptr(out)) != 0


def test_boys_mixed_lanes_and_tails(dll):
    # Packing/scattering must preserve the original grid order, including
    # singleton far lanes, short SIMD tails and a near lane at exactly zero.
    if dll.SGX_generated_version() < 2:
        pytest.skip('libsgx_ad version 2 required for order nine')
    rng = np.random.default_rng(241)
    for size in (1, 2, 3, 7, 31, 127, 128):
        x = rng.uniform(0., 64., size)
        for far in (np.array([size-1]), np.arange(0, size, 2)):
            mixed = x.copy()
            mixed[0] = 0.
            mixed[far] = np.geomspace(64., 1e8, len(far))
            rng.shuffle(mixed)
            out = np.empty((10, size))
            assert dll.SGX_generated_boys(9, size, _native._ptr(mixed), _native._ptr(out)) == 0
            n = np.arange(10)[:, None]
            expected = hyp1f1(n+.5, n+1.5, -mixed)/(2*n+1)
            np.testing.assert_allclose(out, expected, rtol=8e-14, atol=2e-16)
    # Clipping must precede integer table indexing; mixed tiles must not
    # underflow all orders by seeding a downward recurrence at huge t.
    x = np.array([0., 1e300, np.nextafter(64., 0.), 64., np.nextafter(64., np.inf)])
    out = np.empty((10, len(x)))
    assert dll.SGX_generated_boys(9, len(x), _native._ptr(x), _native._ptr(out)) == 0
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out[0, 1], .5*np.sqrt(np.pi/x[1]), rtol=2e-15, atol=0.)
    np.testing.assert_allclose(out[:, 0], 1/(2*np.arange(10)+1), rtol=2e-15)


@pytest.fixture(scope='module', params=[1, 2])
def all_shells(dll, request):
    # Exercise both the uncontracted crossover and signed general contractions.
    basis = ([[l, [.9, 1.]] for l in range(5)] if request.param == 1 else
             [[l, [.9, .6, -.2], [.2, .3, .7]] for l in range(5)])
    mol = gto.M(atom='He 0 0 0; He .4 -.6 1.2', unit='Bohr',
                basis={'He': basis}, verbose=0)
    plan = _native.IntegralPlan(mol, np.ones((mol.nbas, mol.nbas)), 0., 1_000_000_000)
    return mol, plan


@pytest.mark.parametrize('algorithm', [0, 1, 2])
@pytest.mark.parametrize('same_center', [False, True])
def test_all_shell_pairs(all_shells, algorithm, same_center):
    mol, plan = all_shells
    rng = np.random.default_rng(902)
    coords = rng.normal(size=(131, 3)) * 3
    coords[:3] = [[0., 0., 0.], [.4, -.6, 1.2], [200., 50., -100.]]
    for la in range(5):
        for lb in range(5):
            ia, ib = la, lb + (0 if same_center else 5)
            actual = plan.shell_pair(ia, ib, coords, algorithm=algorithm)
            expected = mol.intor('int1e_grids_cart', grids=coords,
                                 shls_slice=(ia, ia+1, ib, ib+1)).transpose(1, 2, 0)
            np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=2e-11,
                                       err_msg=f'{la},{lb}; algorithm={algorithm}')


@pytest.mark.parametrize('cart', [False, True])
def test_fused_contraction(all_shells, cart):
    source, _ = all_shells
    mol = source.copy()
    mol.cart = cart
    plan = _native.IntegralPlan(mol, np.ones((mol.nbas, mol.nbas)), 0., 1_000_000_000)
    rng = np.random.default_rng(712)
    coords = rng.normal(size=(137, 3))
    fg = rng.normal(size=(3, mol.nao, len(coords)))
    ints = mol.intor('int1e_grids', grids=coords)
    expected = np.einsum('gij,xjg->xig', ints, fg)
    actual = plan.contract(mol, coords, fg)
    np.testing.assert_allclose(actual, expected, atol=4e-12, rtol=1e-11)


@pytest.mark.parametrize('cart', [False, True])
def test_full_k_and_linear_operator(dll, cart):
    mol = gto.M(atom='O 0 0 0; H 0 -.7 .5; H 0 .7 .5', basis='cc-pvdz',
                cart=cart, verbose=0, max_memory=2000)
    mf = sgx_fit(scf.RHF(mol), grid_level=0, max_memory=500,
                 integral_backend='generated')
    mf.build()
    rng = np.random.default_rng(71)
    dm = rng.normal(size=(2, mol.nao, mol.nao))
    dm = (dm + dm.transpose(0, 2, 1))*.5
    actual = mf.get_k(dm=dm)
    data = mf.with_df._pjs_data
    assert data._generated_plan is not None
    data._integral_backend = 'libcint'
    expected = mf.get_k(dm=dm)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-11)
    data._integral_backend = 'generated'
    combined = mf.get_k(dm=dm.sum(axis=0))
    np.testing.assert_allclose(combined, actual.sum(axis=0), atol=2e-12)
    np.testing.assert_allclose(np.sum(dm[0]*actual[1]), np.sum(dm[1]*actual[0]), atol=2e-11)


def test_unsupported_shell_and_memory_limit(dll, all_shells):
    mol = gto.M(atom='He', basis={'He': [[5, [1., 1.]]]}, verbose=0)
    with pytest.raises(NotImplementedError):
        _native.IntegralPlan(mol, np.ones((1, 1)), 0., 1_000_000)
    mol, _ = all_shells
    with pytest.raises(MemoryError, match='allowance'):
        _native.IntegralPlan(mol, np.ones((mol.nbas, mol.nbas)), 0., 1)


def test_auto_fallback_and_explicit_requirement(monkeypatch):
    monkeypatch.setattr(_native, 'library', lambda: None)
    data = SimpleNamespace(_integral_backend='auto', use_dm_screening=False,
                           max_memory=2000, mol=None, _mbar_ij=None, _itol=1e-13)
    assert _native.get_plan(data) is None
    assert 'libsgx_ad' in data._generated_fallback_reason
    del data._generated_plan
    data._integral_backend = 'generated'
    with pytest.raises(ImportError, match='libsgx_ad'):
        _native.get_plan(data)


@pytest.mark.parametrize('same_center', [False, True])
def test_all_shell_derivatives(all_shells, same_center):
    mol, plan = all_shells
    if not plan.has_derivatives:
        pytest.skip('libsgx_ad version 2 required')
    coords = np.random.default_rng(391).normal(size=(131, 3))*2
    coords[:3] = [[0., 0., 0.], [.4, -.6, 1.2], [200., 50., -100.]]
    for la in range(5):
        for lb in range(5):
            ia, ib = la, lb + (0 if same_center else 5)
            value, derivative = plan.shell_pair_with_grad(ia, ib, coords)
            expected = mol.intor('int1e_grids_cart', grids=coords,
                                 shls_slice=(ia, ia+1, ib, ib+1)).transpose(1, 2, 0)
            left = mol.intor('int1e_grids_ip_cart', grids=coords,
                             shls_slice=(ia, ia+1, ib, ib+1)).transpose(0, 2, 3, 1)
            right = mol.intor('int1e_grids_ip_cart', grids=coords,
                              shls_slice=(ib, ib+1, ia, ia+1)).transpose(0, 3, 2, 1)
            np.testing.assert_allclose(value, expected, rtol=2e-11, atol=3e-13)
            np.testing.assert_allclose(derivative, np.stack([left, right]),
                                       rtol=2e-11, atol=6e-13, err_msg=f'{la},{lb}')


@pytest.mark.parametrize('cart', [False, True])
def test_fused_value_derivative_contraction(all_shells, cart):
    source, initial = all_shells
    if not initial.has_derivatives:
        pytest.skip('libsgx_ad version 2 required')
    mol = source.copy()
    mol.cart = cart
    plan = _native.IntegralPlan(mol, np.ones((mol.nbas, mol.nbas)), 0., 1_000_000_000)
    rng = np.random.default_rng(34)
    coords = rng.normal(size=(137, 3))
    fg = rng.normal(size=(3, mol.nao, len(coords)))
    value, derivative = plan.contract_with_grad(mol, coords, fg)
    expected = np.einsum('gij,xjg->xig', mol.intor('int1e_grids', grids=coords), fg)
    expected_deriv = np.einsum('kgij,xjg->xkig', mol.intor('int1e_grids_ip', grids=coords), fg)
    np.testing.assert_allclose(value, expected, atol=4e-12, rtol=1e-11)
    np.testing.assert_allclose(derivative, expected_deriv, atol=5e-12, rtol=2e-11)


@pytest.mark.parametrize('same_center', [False, True])
def test_deeply_contracted_derivatives(dll, same_center):
    # Deep contractions exercise the low-angular-momentum derivative HRR
    # crossover, with primitive-specific exponent weights and signed columns.
    basis = [[l, *[[float(e), (-1.)**k*.2, .1+k*.02]
                    for k, e in enumerate(np.geomspace(.2, 2., 8))]]
             for l in range(5)]
    mol = gto.M(atom='He 0 0 0; He .4 -.6 1.2', unit='Bohr',
                basis={'He': basis}, verbose=0)
    plan = _native.IntegralPlan(mol, np.ones((10, 10)), 0., 10**9)
    coords = np.random.default_rng(731).normal(size=(7, 3))*2
    for a in range(5):
        for b in range(5):
            b = b if same_center else b+5
            value, derivative = plan.shell_pair_with_grad(a, b, coords)
            expected = mol.intor('int1e_grids_cart', grids=coords,
                shls_slice=(a, a+1, b, b+1)).transpose(1, 2, 0)
            left = mol.intor('int1e_grids_ip_cart', grids=coords,
                shls_slice=(a, a+1, b, b+1)).transpose(0, 2, 3, 1)
            right = mol.intor('int1e_grids_ip_cart', grids=coords,
                shls_slice=(b, b+1, a, a+1)).transpose(0, 3, 2, 1)
            np.testing.assert_allclose(value, expected, atol=6e-13, rtol=2e-11)
            np.testing.assert_allclose(derivative, [left, right],
                                       atol=6e-13, rtol=2e-11)


@pytest.mark.parametrize('same_center', [False, True])
def test_independent_center_and_grid_finite_differences(all_shells, same_center):
    original, initial = all_shells
    if not initial.has_derivatives:
        pytest.skip('libsgx_ad version 2 required')
    positions = original.atom_coords()
    if same_center:
        positions[1] = positions[0]
    mol = original.set_geom_(positions, unit='Bohr', inplace=False)
    plan = _native.IntegralPlan(mol, np.ones((mol.nbas, mol.nbas)), 0., 1_000_000_000)
    coords = np.random.default_rng(362).normal(size=(5, 3))
    # Independent g centers require raised h intermediates even when A == B.
    _, derivative = plan.shell_pair_with_grad(4, 9, coords)
    step = 1e-5
    def integrals(m, grid):
        return m.intor('int1e_grids_cart', grids=grid,
                       shls_slice=(4, 5, 9, 10)).transpose(1, 2, 0)
    for k in range(3):
        for center in range(2):
            delta = np.zeros_like(positions)
            delta[center, k] = step
            plus = mol.set_geom_(positions+delta, unit='Bohr', inplace=False)
            minus = mol.set_geom_(positions-delta, unit='Bohr', inplace=False)
            fd = (integrals(plus, coords)-integrals(minus, coords))/(2*step)
            np.testing.assert_allclose(-derivative[center, k], fd, atol=3e-9, rtol=3e-7)
        delta = np.zeros_like(coords)
        delta[:, k] = step
        fd_grid = (integrals(mol, coords+delta)-integrals(mol, coords-delta))/(2*step)
        np.testing.assert_allclose(derivative[:, k].sum(axis=0), fd_grid, atol=3e-9, rtol=3e-7)


def test_geometry_uses_generated_derivatives(dll, monkeypatch):
    from pyscf.sgx import sgx_jk
    from pyscfad.sgx import _sgx_grad
    if dll.SGX_generated_version() < 2:
        pytest.skip('libsgx_ad version 2 required')
    mol = gto.M(atom='O 0 0 0; H 0 -.7 .5; H 0 .7 .5', basis='cc-pvdz',
                verbose=0, max_memory=2000)
    mf = sgx_fit(scf.RHF(mol), grid_level=0, max_memory=500, integral_backend='generated')
    mf.build()
    obj = mf.with_df
    obj._build_pjs(1e-13)
    plan = _native.get_plan(obj._pjs_data)
    seeds = np.random.default_rng(318).normal(size=(2, mol.nao, mol.nao))
    dm, bar = (seeds+seeds.transpose(0, 2, 1))*.5
    plan.has_derivatives = False
    expected = _sgx_grad.geometry(obj, dm, bar, 1e-13)
    plan.has_derivatives = True
    def forbidden(*args, **kwargs):
        raise AssertionError('Generated geometry called the libcint integral driver')
    monkeypatch.setattr(sgx_jk, '_gen_k_direct', forbidden)
    actual = _sgx_grad.geometry(obj, dm, bar, 1e-13)
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-10)
    np.testing.assert_allclose(actual.sum(axis=0), 0., atol=2e-10)
