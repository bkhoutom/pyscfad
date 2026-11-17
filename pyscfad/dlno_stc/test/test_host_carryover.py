"""Independent carryover checks for concrete host primary and moment response."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import _weak_numpy as spaces
from pyscfad.dlno_stc._pairs import WeakScreen
from pyscfad.dlno_stc._selection import BoysDomainThresholds
from pyscfad.dlno_stc.test import _weak_reference as reference
from pyscfad.lib import _multipole_numpy as moments


class Common(NamedTuple):
    s1e: object
    fock: object
    pao_coeff: object


def assert_tree(actual, expected, *, atol=3e-10, rtol=3e-10):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert got.shape == want.shape
        np.testing.assert_allclose(got, want, atol=atol, rtol=rtol)


@pytest.fixture(autouse=True)
def release_oracle_executables():
    # The mathematical oracles compile distinct shapes; production is host-only.
    yield
    jax.clear_caches()


@pytest.mark.parametrize('rows,columns,deficient', [
    (7, 1, False), (19, 5, False), (35, 17, True), (67, 35, True),
])
def test_target_mixed_sizes_and_rank_loss_keep_all_input_bars(rows, columns, deficient):
    rng = np.random.default_rng(717)
    size = rows + 3
    ao = np.r_[0, rng.permutation(np.arange(1, size))[:rows-1]].astype(np.int32)
    root = np.eye(size) + .01 * rng.normal(size=(size, size))
    overlap = root.T @ root
    common = Common(overlap, np.diag(np.r_[-3., np.arange(1., size)]), np.zeros((size, 0)))
    candidate = rng.normal(size=(rows, columns))
    candidate[0] = 0.
    if deficient:
        candidate[:, -1] = .7 * candidate[:, 0]
    rhs = overlap[:, :1]
    options = dict(ao=ao, thresholds=BoysDomainThresholds(), index=4)
    actual, pullback, labels = spaces.target(common, candidate, rhs, **options)
    expected, reverse, expected_labels = jax.vjp(
        lambda *args: reference._select_target(*args, **options),
        common, candidate, rhs, has_aux=True)
    assert_tree(labels, expected_labels, atol=0., rtol=0.)
    assert actual.virtual_coeff.shape == (rows, columns - int(deficient))
    np.testing.assert_allclose(actual.occupied_coeff, expected.occupied_coeff, atol=2e-12)
    np.testing.assert_allclose(actual.virtual_energy, expected.virtual_energy, atol=2e-10)
    weight = rng.normal(size=(rows, rows))
    occupied_weight = rng.normal(size=(rows, 1))

    def observable(screen):
        cv = screen.virtual_coeff
        density = cv @ cv.T
        weighted = (cv * screen.virtual_energy[None, :]) @ cv.T
        return (jnp.sum(weight * (density + .01 * weighted))
                + jnp.sum(occupied_weight * screen.occupied_coeff)
                + jnp.sum(screen.occupied_energy**2))

    np.testing.assert_allclose(observable(actual), observable(expected), atol=2e-10)
    # Projector seeds avoid depending on arbitrary LAPACK eigenvector signs.
    actual_bars = pullback(jax.grad(observable)(actual))
    expected_bars = reverse(jax.grad(observable)(expected))
    assert_tree(actual_bars, expected_bars, atol=2e-8, rtol=2e-7)
    assert all(np.linalg.norm(x) > 1e-6 for x in
               (actual_bars[0].s1e, actual_bars[0].fock, *actual_bars[1:]))


@pytest.mark.parametrize('empty', ['occupied', 'virtual'])
def test_zero_target_rank_does_not_create_an_endpoint(empty):
    common = Common(np.eye(7), np.diag(np.arange(7.) - 2), np.zeros((7, 0)))
    candidate, rhs = np.eye(7)[:, 1:4], np.eye(7)[:, :1]
    if empty == 'occupied':
        rhs *= 0.
    else:
        candidate *= 0.
    options = dict(ao=np.arange(7), thresholds=BoysDomainThresholds(), index=0)
    actual, pullback, labels = spaces.target(common, candidate, rhs, **options)
    expected, expected_labels = reference._select_target(common, candidate, rhs, **options)
    assert actual is expected is None
    assert pullback is None
    assert_tree(labels, expected_labels, atol=0., rtol=0.)


@pytest.mark.parametrize('different_support,deficient', [(False, False), (True, False), (True, True)])
def test_preprojection_preserves_independent_product_and_overlap_bars(different_support, deficient):
    rng = np.random.default_rng(301)
    n = 19
    root = np.eye(n) + .005 * rng.normal(size=(n, n))
    overlap = root.T @ root
    pao = rng.normal(size=(n, n-8)) / np.sqrt(n)
    if deficient:
        pao[:, 0] *= 1e-5
    common = Common(overlap, np.eye(n), pao)
    product = overlap @ pao
    products = pao.T @ product, product
    support = np.arange(n-6)
    ao = np.arange(n-2) if different_support else support[::-1].copy()
    if not different_support:
        support = ao.copy()
    options = dict(columns=rng.permutation(n-8), support=support, ao=ao,
                   thresholds=BoysDomainThresholds(domain_pao=.01, bp_pao=.05))
    actual, pullback, labels = spaces.preprojection(common, products, **options)
    expected, reverse, expected_labels = jax.vjp(
        lambda *args: reference._select_preprojection(*args, **options),
        common, products, has_aux=True)
    assert_tree(labels, expected_labels, atol=0., rtol=0.)
    np.testing.assert_allclose(actual @ actual.T, expected @ expected.T, atol=2e-10)
    weight = rng.normal(size=(len(ao), len(ao)))
    seed = lambda value: (weight + weight.T) @ value
    assert_tree(pullback(seed(actual)), reverse(seed(expected)), atol=3e-8, rtol=2e-7)


def test_degenerate_projection_preserves_completeness_labels_and_subspace():
    rng = np.random.default_rng(442)
    q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    theta = rng.uniform(0, 2*np.pi)
    r = np.array([[0, 0, 1], [np.cos(theta), np.sin(theta), 0],
                  [-np.sin(theta), np.cos(theta), 0]])
    values = np.array([.97, .97, .8])
    virtual = np.concatenate((q @ np.diag(np.sqrt(values)), r @ np.diag(np.sqrt(1-values))))
    projector = virtual @ virtual.T
    pao = projector[:, :3] / np.sqrt(np.diag(projector)[:3])
    common = Common(np.eye(6), np.eye(6), pao)
    products = pao.T @ pao, pao
    options = dict(columns=np.arange(3), support=np.arange(3), ao=np.arange(5),
                   thresholds=BoysDomainThresholds())
    actual, _, labels = spaces.preprojection(common, products, **options)
    expected, expected_labels = reference._select_preprojection(common, products, **options)
    assert_tree(labels, expected_labels, atol=0., rtol=0.)
    np.testing.assert_allclose(actual @ actual.T, expected @ expected.T, atol=2e-11)


def test_degenerate_fock_keeps_selected_denominator_response():
    rng = np.random.default_rng(818)
    candidate = np.zeros((7, 3))
    candidate[1:4] = rng.normal(size=(3, 3))
    common = Common(np.eye(7), np.diag([-2., 1, 1, 1, 3, 4, 5]), np.zeros((7, 0)))
    rhs = np.eye(7)[:, :1]
    options = dict(ao=np.arange(7), thresholds=BoysDomainThresholds(), index=0)
    actual, pullback, labels = spaces.target(common, candidate, rhs, **options)
    expected, reverse, expected_labels = jax.vjp(
        lambda *args: reference._select_target(*args, **options),
        common, candidate, rhs, has_aux=True)
    assert_tree(labels, expected_labels, atol=0., rtol=0.)
    dipole = rng.normal(size=7)

    def energy(screen):
        return jnp.sum((dipole @ screen.virtual_coeff)**2 /
                       (screen.virtual_energy - screen.occupied_energy[0] + 2))

    np.testing.assert_allclose(energy(actual), energy(expected), atol=2e-11)
    assert_tree(pullback(jax.grad(energy)(actual)), reverse(jax.grad(energy)(expected)), atol=2e-11)


def test_prepare_primary_degenerate_fock_keeps_upstream_preprojection_response(monkeypatch):
    from pyscfad.dlno_stc import _pairs, _primary_response as primary, domain

    rng = np.random.default_rng(118)
    pao = rng.normal(size=(7, 3))
    pao[0] = 0.
    occupied = np.eye(7)[:, :1]
    common = domain.DomainData(np.eye(7), np.diag([-2., 1, 1, 1, 1, 1, 1]),
        occupied, np.eye(7)[:, 1:], np.array([-2.]), np.ones(6), occupied, pao,
        (domain.TargetOccupiedData(occupied),))
    columns, ao, thresholds = np.arange(3), np.arange(7), BoysDomainThresholds()
    monkeypatch.setattr(primary, '_pao_columns_on_atoms', lambda *args: columns)
    monkeypatch.setattr(primary.tools, 'ao_index_by_atom', lambda *args: ao)

    def oracle(current):
        products = reference._weak_screen_products(current)
        candidate, _ = reference._select_preprojection(current, products.pao,
            columns=columns, support=ao, ao=ao, thresholds=thresholds)
        screen, _ = reference._select_target(current, candidate, products.occupied[0],
            ao=ao, thresholds=thresholds, index=0)
        return screen

    expected, reverse = jax.vjp(oracle, common)
    *_, (actual, response) = primary.prepare_primary(
        common, spaces.products(common), None, None, thresholds, 0, (0,), (0,), {})
    dipole = rng.normal(size=7)

    def energy(screen):
        return jnp.sum((dipole @ screen.virtual_coeff)**2 /
                       (screen.virtual_energy - screen.occupied_energy[0] + 2))

    actual_bar = _pairs._apply_weak_screen_pullback(response, jax.grad(energy)(actual),
        {response.preprojection_pullback: (1, None)},
        {response.products_pullback.pullback: (1, None)})
    expected_bar, = reverse(jax.grad(energy)(expected))
    np.testing.assert_allclose(energy(actual), energy(expected), atol=2e-11)
    assert_tree(actual_bar, expected_bar, atol=2e-10)
    assert np.linalg.norm(actual_bar.fock) > 1e-6
    assert np.linalg.norm(actual_bar.pao_coeff) > 1e-6


def endpoint_case(rows, columns):
    rng = np.random.default_rng(39)
    screen = WeakScreen(np.array([-.7]), .1 * rng.normal(size=(rows, 1)),
                        np.linspace(.2, 1.4, columns), .1 * rng.normal(size=(rows, columns)))
    raw = tuple(.1 * rng.normal(size=shape) for shape in
                ((rows, rows), (3, rows, rows), (6, rows, rows), (10, rows, rows)))
    return screen, raw


def endpoint_reference(screen, raw):
    center, mu, theta, omega = reference.transition_moments(
        screen.occupied_coeff, screen.virtual_coeff, *raw, 4)
    return center[0], mu[0], screen.virtual_energy-screen.occupied_energy[0], theta[0], omega[0]


def seeds(tree, seed=71):
    rng = np.random.default_rng(seed)
    return jax.tree.map(lambda x: rng.normal(size=x.shape), tree)


@pytest.mark.parametrize('rows,columns', [(7, 0), (17, 5), (35, 127), (35, 128), (41, 129)])
def test_endpoint_nonsymmetric_all_bars_and_repeated_seed_immutability(rows, columns):
    screen, raw = endpoint_case(rows, columns)
    saved_inputs = jax.tree.map(np.copy, (screen, raw))
    expected, reverse = jax.vjp(endpoint_reference, screen, raw)
    actual, pullback = moments.endpoint(screen, raw, 4, with_grad=True)
    assert_tree(actual, expected)
    assert_tree(moments.endpoint(screen, raw, 4), expected)
    for number in (71, 82, 71):
        seed = seeds(expected, number)
        saved_seed = jax.tree.map(np.copy, seed)
        assert_tree(pullback(seed), reverse(seed))
        assert_tree(seed, saved_seed, atol=0., rtol=0.)
        assert_tree((screen, raw), saved_inputs, atol=0., rtol=0.)


def test_endpoint_all_input_directional_finite_difference():
    arguments = endpoint_case(13, 129)
    endpoint, pullback = moments.endpoint(*arguments, 4, with_grad=True)
    seed, direction = seeds(endpoint), seeds(arguments, 33)
    bars = pullback(seed)
    analytic = sum(np.sum(b*d) for b, d in zip(jax.tree.leaves(bars), jax.tree.leaves(direction)))

    def value(step):
        changed = jax.tree.map(lambda x, dx: x + step*dx, arguments, direction)
        result = moments.endpoint(*changed, 4)
        return sum(np.sum(x*w) for x, w in zip(result, seed))

    step = 1e-4
    finite = (value(-2*step) - 8*value(-step) + 8*value(step) - value(2*step)) / (12*step)
    assert abs(analytic) > 1e-6
    np.testing.assert_allclose(analytic, finite, rtol=2e-8, atol=2e-8)


def test_new_host_endpoint_sizes_do_not_compile_jax_kernels(caplog):
    def evaluate(rows, columns):
        screen, raw = endpoint_case(rows, columns)
        value, pullback = moments.endpoint(screen, raw, 4, with_grad=True)
        bars = pullback(seeds(value))
        assert all(np.isfinite(x).all() for x in jax.tree.leaves(bars))

    with jax.log_compiles(True):
        jax.jit(lambda x: x + .37)(np.zeros(3)).block_until_ready()
        assert any('Compiling' in record.getMessage() for record in caplog.records)
        evaluate(7, 3)
        caplog.clear()
        for rows, columns in ((19, 127), (37, 128), (71, 129), (133, 259)):
            evaluate(rows, columns)
    assert not [record.getMessage() for record in caplog.records if 'Compiling' in record.getMessage()]


@pytest.mark.parametrize('grouped', [False, True])
def test_molecular_mixed_supports_keep_all_screen_and_coordinate_bars(grouped):
    from itertools import combinations_with_replacement
    from types import SimpleNamespace
    from pyscfad import gto, scf
    from pyscfad.dlno_stc import _pairs
    from pyscfad.dlno_stc.adjoint import add_cotangent

    mol = gto.Mole(atom='O .11 -.23 .07; H .84 .15 .62; H -.47 .71 -.31',
                   basis='cc-pvdz', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = scf.RHF(mol)
    supports = ((0, 1, 2), (0, 1, 2), (2, 0))
    static = SimpleNamespace(
        fragments=tuple(SimpleNamespace(primary_atoms=np.asarray(a)) for a in supports),
        thresholds=SimpleNamespace(multipole_order=4))
    local_templates = tuple(_pairs._fake_mol_with_traced_centers(mol, atoms) for atoms in supports)
    screens = tuple(endpoint_case(local.nao, width)[0]
                    for local, width in zip(local_templates, (3, 129, 5)))

    names = ('int1e_ovlp', 'int1e_r', 'int1e_rr', 'int1e_rrr')
    components = tuple(np.array([sum(axis * 3**(rank-k-1) for k, axis in enumerate(axes))
                                for axes in combinations_with_replacement(range(3), rank)])
                       for rank in range(4))
    raw_values = []
    for local in local_templates:
        with local.with_common_origin((0., 0., 0.)):
            raw = [local.intor(name).reshape(3**rank, local.nao, local.nao)[components[rank]]
                   for rank, name in enumerate(names)]
        raw_values.append((raw[0][0], *raw[1:]))
    expected, reverse = jax.vjp(lambda current_screens, raw:
        tuple(endpoint_reference(s, r) for s, r in zip(current_screens, raw)),
        screens, tuple(raw_values))
    endpoint_seeds = tuple(seeds(value, 42+i) for i, value in enumerate(expected))
    screen_bars, raw_bars = reverse(endpoint_seeds)
    expected_mf = spaces.zeros(mf)
    # Independent dense bra-plus-ket contraction avoids assuming symmetric
    # endpoint seeds or reusing the production shell contraction.
    for atoms, local, raw_bar in zip(supports, local_templates, raw_bars):
        with local.with_common_origin((0., 0., 0.)):
            for rank, (name, weight) in enumerate(zip(names, raw_bar)):
                weight = np.asarray(weight).reshape(-1, local.nao, local.nao)
                bra = -np.asarray(local.intor(name + '_dr10')).reshape(3, 3**rank, local.nao, local.nao)
                ket = -np.asarray(local.intor(name + '_dr01')).reshape(3**rank, 3, local.nao, local.nao).transpose(1, 0, 2, 3)
                bra, ket = bra[:, components[rank]], ket[:, components[rank]]
                for atom, (_, _, p0, p1) in zip(atoms, local.aoslice_by_atom()):
                    expected_mf.mol.coords[atom] += np.einsum('puv,xpuv->x', weight[:, p0:p1], bra[:, :, p0:p1])
                    expected_mf.mol.coords[atom] += np.einsum('puv,xpuv->x', weight[:, :, p0:p1], ket[:, :, :, p0:p1])
    expected_bars = expected_mf, screen_bars
    linearizations = {}
    for index, screen in enumerate(screens):
        def screen_pullback(bar, index=index):
            return (tuple(bar if i == index else spaces.zeros(value)
                          for i, value in enumerate(screens)),)
        linearizations[index] = screen, screen_pullback
    originals = dict(enumerate(expected))
    bars = spaces.zeros((mf, screens))
    if grouped:
        bars = _pairs._grouped_weak_pullback(mf, static, dict(originals), dict(originals),
            linearizations, dict(enumerate(endpoint_seeds)), bars, {}, {})
    else:
        for index in range(len(screens)):
            value, pullback = _pairs._replay_weak_endpoint(mf, linearizations[index], static, index)
            assert_tree(value, expected[index], atol=2e-10)
            bars = jax.tree.map(add_cotangent, bars, pullback(endpoint_seeds[index]))
    assert_tree(bars, expected_bars, atol=2e-9, rtol=2e-10)
    assert np.linalg.norm(bars[0].mol.coords) > 1e-6
    for screen_bar in bars[1]:
        assert all(np.linalg.norm(leaf) > 1e-6 for leaf in screen_bar)
