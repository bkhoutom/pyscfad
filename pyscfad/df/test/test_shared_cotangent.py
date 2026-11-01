"""Experimental resident accumulation preserves explicit public VJP contracts."""
import importlib
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad import df, gto
from pyscfad.ao2mo import _ao2mo_opt
from pyscfad.df import _df_jk_opt, _int3c_cross_opt, addons


def registry():
    name = 'pyscfad.lib._cotangent_buffer'
    assert importlib.util.find_spec(name) is not None, 'scoped cotangent registry missing'
    return importlib.import_module(name)


def unpack(packed, nao):
    i, j = np.tril_indices(nao)
    b = jnp.zeros((packed.shape[0], nao, nao), dtype=packed.dtype)
    return b.at[:, i, j].set(packed).at[:, j, i].set(packed)


def ao2mo_reference(packed, coeff, orbs_slice=(0, 2, 1, 3)):
    k0, k1, l0, l1 = orbs_slice
    return jnp.einsum('Lpq,pi,qj->Lij', unpack(packed, coeff.shape[0]),
                      coeff[:, k0:k1], coeff[:, l0:l1]).reshape(packed.shape[0], -1)


def jk_reference(packed, dm):
    b = unpack(packed, dm.shape[-1])
    return (jnp.einsum('Lij,ij,Lpq->pq', b, dm, b),
            jnp.einsum('Lpi,ij,Ljq->pq', b, dm, b))


def assert_trees_close(actual, expected):
    for got, want in zip(jax.tree_util.tree_leaves(actual),
                         jax.tree_util.tree_leaves(expected), strict=True):
        np.testing.assert_allclose(got, want, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize('orbs_slice', [(0, 2, 1, 3), (0, 3, 1, 2)])
def test_native_ao2mo_accumulates_into_supplied_buffer(orbs_slice):
    # Missing beta=1/additive native semantics loses the nonzero initial bar.
    assert hasattr(_ao2mo_opt, 'nr_e2_vjp'), 'additive AO2MO VJP interface missing'
    rng = np.random.default_rng(180)
    packed, coeff = rng.normal(size=(9, 3)), rng.normal(size=(2, 3))
    k0, k1, l0, l1 = orbs_slice
    seed = rng.normal(size=(9, (k1-k0)*(l1-l0)))
    _, pullback = jax.vjp(lambda b, c: ao2mo_reference(b, c, orbs_slice), packed, coeff)
    expected_eri, expected_coeff = pullback(seed)
    initial = rng.normal(size=packed.shape)
    target = initial.copy()
    eri_bar, coeff_bar = _ao2mo_opt.nr_e2_vjp(
        packed, coeff, orbs_slice, seed, aosym='s2', eri_bar=target)
    assert eri_bar is target
    np.testing.assert_allclose(target, initial + expected_eri, atol=2e-11)
    np.testing.assert_allclose(coeff_bar, expected_coeff, atol=2e-11)


def test_reverse_scope_isolated_and_reports_unconsumed_buffers(monkeypatch):
    cb = registry()
    monkeypatch.setenv('PYSCFAD_DF_SHARED_COTANGENT', '1')
    x = np.arange(6., dtype=np.float64).reshape(2, 3)
    cb.register(x)
    assert cb.get(x) is None
    assert not cb.registered(x)
    with pytest.raises(RuntimeError, match='unconsumed'):
        with cb.reverse_scope():
            cb.get(x)[:] = 7
    with cb.reverse_scope():
        np.testing.assert_array_equal(cb.get(x), np.zeros_like(x))
        with pytest.raises(RuntimeError, match='nested'):
            with cb.reverse_scope():
                pass
        np.testing.assert_array_equal(cb.pop(x), np.zeros_like(x))
    with pytest.raises(ValueError, match='sentinel'):
        with cb.reverse_scope():
            cb.get(x)[:] = 13
            raise ValueError('sentinel')
    with cb.reverse_scope():
        np.testing.assert_array_equal(cb.get(x), np.zeros_like(x))
        cb.pop(x)


def test_default_registry_does_not_capture_bars(monkeypatch):
    cb = registry()
    monkeypatch.delenv('PYSCFAD_DF_SHARED_COTANGENT', raising=False)
    x = np.ones((2, 3))
    cb.register(x)
    with cb.reverse_scope():
        assert cb.get(x) is None
        assert not cb.registered(x)


def test_registered_standalone_consumer_returns_explicit_bar(monkeypatch):
    cb = registry()
    monkeypatch.setenv('PYSCFAD_DF_SHARED_COTANGENT', '1')
    rng = np.random.default_rng(181)
    packed, coeff = rng.normal(size=(9, 3)), rng.normal(size=(2, 3))
    cb.register(packed)
    fn = lambda b, c: _ao2mo_opt.nr_e2(b, c, (0, 2, 1, 3), aosym='s2')
    _, pullback = jax.vjp(fn, packed, coeff)
    _, reference = jax.vjp(ao2mo_reference, packed, coeff)
    for scale in (1., 2., 0., -.5):
        seed = np.full((9, 4), scale)
        assert_trees_close(pullback(seed), reference(seed))
    assert cb.get(packed) is None


@pytest.mark.parametrize('positive', [False, True])
def test_jk_adds_into_resident_buffer_without_explicit_factor_bar(monkeypatch, positive):
    cb = registry()
    monkeypatch.setenv('PYSCFAD_DF_SHARED_COTANGENT', '1')
    mol = gto.Mole(atom='H 0 0 0; H 0 0 .74', basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    obj = df.DF(mol)
    rng = np.random.default_rng(182)
    obj._cderi = rng.normal(size=(9, 3))
    obj.blockdim = 4
    x = rng.normal(size=(2, 1))
    dm = x @ x.T if positive else np.array([[1., .2], [.2, -1.]])
    seeds = tuple(rng.normal(size=(2, 2)) for _ in range(2))
    _, reference = jax.vjp(jk_reference, obj._cderi, dm)
    expected_eri, expected_dm = reference(seeds)
    cb.register(obj._cderi)
    with cb.reverse_scope():
        target = cb.get(obj._cderi)
        target[:] = .3
        obj_bar, dm_bar = _df_jk_opt.get_jk_bwd(
            1, True, True, 1e-13, (obj, dm), seeds)
        assert obj_bar._cderi is None, 'resident consumer returned duplicate factor cotangent'
        np.testing.assert_allclose(dm_bar, expected_dm, atol=2e-11)
        np.testing.assert_allclose(cb.pop(obj._cderi), .3 + expected_eri, atol=2e-11)


@pytest.mark.parametrize('explicit_weight', [0., .07])
def test_shared_producer_multiple_consumers_repeated_scaled_and_zero_seeds(monkeypatch, explicit_weight):
    cb = registry()
    mol = gto.Mole(atom='H 0 0 0; H 0 0 .74', basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    auxmol = addons.make_auxmol(mol, 'weigend')
    low = jnp.linalg.cholesky(auxmol.intor('int2c2e', hermi=1))
    ranges = ((0, mol.nbas, 0),)
    dm = np.array([[1.1, .2], [.2, .7]])
    coeff = np.array([[1., .2, -.1], [.3, 1., .4]])
    jseed, kseed = np.array([[.2, .3], [.5, .7]]), np.array([[.9, -.1], [.4, .2]])
    def objective(m, a, l):
        packed = _int3c_cross_opt.int3c2e_cholesky(m, a, l, 'int3c2e', ranges)
        obj = df.DF(m)
        obj.auxmol = a
        obj._cderi = packed
        j, k = _df_jk_opt.get_jk(obj, dm)
        transformed = _ao2mo_opt.nr_e2(packed, coeff, (0, 2, 1, 3), aosym='s2')
        # Ordinary JAX contribution must be added to the in-place consumers.
        return jnp.sum(j * jseed) + jnp.sum(k * kseed) + jnp.sum(transformed) + explicit_weight*jnp.sum(packed**2)
    monkeypatch.setenv('PYSCFAD_DF_SHARED_COTANGENT', '0')
    expected_value, baseline = jax.vjp(objective, mol, auxmol, low)
    monkeypatch.setenv('PYSCFAD_DF_SHARED_COTANGENT', '1')
    value, shared = jax.vjp(objective, mol, auxmol, low)
    np.testing.assert_allclose(value, expected_value, atol=1e-12)
    _, second = jax.vjp(objective, mol, auxmol, low)
    for scale, pullback in zip((1., 2., -.5, 0., 1.),
                               (shared, second, shared, second, shared)):
        with cb.reverse_scope() as scope:
            actual = pullback(jnp.asarray(scale))
        assert scope.buffers_created == 1
        assert scope.producer_pops == 1
        assert_trees_close(actual, baseline(jnp.asarray(scale)))
    # A live producer tape cannot change normal, unscoped public VJP behavior.
    assert_trees_close(shared(jnp.asarray(1.)), baseline(jnp.asarray(1.)))


def test_implicit_vjp_suppresses_only_registered_untouched_zeros(monkeypatch):
    from pyscfad._src import implicit_diff
    assert hasattr(implicit_diff, '_call_vjp'), 'implicit cotangent zero suppression missing'
    cb = registry()
    monkeypatch.setenv('PYSCFAD_DF_SHARED_COTANGENT', '1')
    rng = np.random.default_rng(184)
    packed, coeff = rng.normal(size=(9, 3)), rng.normal(size=(2, 3))
    cb.register(packed)
    def fn(b, c):
        return _ao2mo_opt.nr_e2(b, c, (0, 2, 1, 3), aosym='s2')
    seed = rng.normal(size=(9, 4))
    _, expected = jax.vjp(ao2mo_reference, packed, coeff)
    expected_eri, expected_coeff = expected(seed)
    _, pullback = jax.vjp(fn, packed, coeff)
    with cb.reverse_scope():
        eri_bar, coeff_bar = implicit_diff._call_vjp(pullback, (packed, coeff), seed)
        assert eri_bar is None
        np.testing.assert_allclose(coeff_bar, expected_coeff, atol=2e-11)
        np.testing.assert_allclose(cb.pop(packed), expected_eri, atol=2e-11)
    # An ordinary JAX consumer has an explicit factor bar that must survive.
    _, mixed = jax.vjp(lambda b, c: fn(b, c) + jnp.sum(b), packed, coeff)
    with cb.reverse_scope():
        eri_bar, _ = implicit_diff._call_vjp(mixed, (packed, coeff), seed)
        np.testing.assert_allclose(eri_bar, np.full_like(packed, np.sum(seed)), atol=2e-11)
        np.testing.assert_allclose(cb.pop(packed), expected_eri, atol=2e-11)


def test_nested_root_matvec_does_not_accumulate_unused_factor_bar(monkeypatch):
    from pyscfad._src.implicit_diff import custom_root, root_vjp
    cb = registry()
    monkeypatch.setenv('PYSCFAD_DF_SHARED_COTANGENT', '1')
    packed = np.arange(9., dtype=np.float64).reshape(3, 3) / 10
    coeff = np.array([[1., .2, -.1], [.3, 1., .4]])
    cb.register(packed)
    def term(b):
        return jnp.sum(_ao2mo_opt.nr_e2(b, coeff, (0, 2, 1, 3), aosym='s2'))
    def solve(matvec, rhs, **kwargs):
        return rhs / matvec(jnp.ones_like(rhs)), None
    def inner_opt(root, outer, b):
        return .2 * root + .1 * outer + term(b) - root
    @custom_root(inner_opt, solve=solve)
    def inner(initial, outer, b):
        return (.1 * outer + term(b)) / .8
    def outer_opt(root, b):
        return .3 * root + .1 * inner(jnp.array(0.), root, b) - root
    root = 2 * term(packed) / 11
    expected = jax.grad(lambda b: 2 * jnp.sum(ao2mo_reference(b, coeff)) / 11)(packed)
    with cb.reverse_scope():
        explicit, = root_vjp(outer_opt, root, (packed,), jnp.array(1.), solve=solve)
        assert explicit is None
        np.testing.assert_allclose(cb.pop(packed), expected, atol=2e-11)


def test_registry_rejects_different_layout_or_owner_aliases(monkeypatch):
    cb = registry()
    monkeypatch.setenv('PYSCFAD_DF_SHARED_COTANGENT', '1')
    x = np.arange(9., dtype=np.float64).reshape(3, 3)
    cb.register(x)
    aliases = [x.T, x.view(np.int64), x[:, :2], x.copy(),
               np.ctypeslib.as_array(x.ctypes.data_as(__import__('ctypes').POINTER(
                   __import__('ctypes').c_double)), shape=(3, 3))]
    with cb.reverse_scope():
        for alias in aliases:
            assert cb.get(alias) is None
        assert cb.registered(x.view())
        cb.get(x)[:] = 2
        np.testing.assert_array_equal(cb.pop(x.view()), np.full_like(x, 2))


@pytest.mark.parametrize('bad', ['dtype', 'readonly', 'layout', 'alias', 'shape'])
def test_native_ao2mo_rejects_unsafe_output_buffer(bad):
    packed = np.ones((9, 3))
    coeff = np.ones((2, 3))
    target = np.zeros_like(packed)
    if bad == 'dtype':
        target = target.astype(np.float32)
    elif bad == 'readonly':
        target.flags.writeable = False
    elif bad == 'layout':
        target = np.asfortranarray(target)
    elif bad == 'alias':
        target = packed
    elif bad == 'shape':
        target = np.zeros((8, 3))
    with pytest.raises(ValueError, match='separate writable'):
        _ao2mo_opt.nr_e2_vjp(packed, coeff, (0, 2, 1, 3), np.ones((9, 4)),
                            aosym='s2', eri_bar=target)


def test_implicit_adapter_preserves_jitted_explicit_cotangents(monkeypatch):
    from pyscfad._src.implicit_diff import _call_vjp
    cb = registry()
    monkeypatch.setenv('PYSCFAD_DF_SHARED_COTANGENT', '1')
    x = np.arange(6., dtype=np.float64).reshape(2, 3) / 10
    y = np.ones_like(x) * 2
    cb.register(x)
    _, pullback = jax.vjp(jax.jit(lambda a, b: jnp.sum(jnp.sin(a) * b)), x, y)
    for seed in (1., -.4, 0., 1.):
        with cb.reverse_scope() as scope:
            actual = _call_vjp(pullback, (x, y), jnp.array(seed))
        assert_trees_close(actual, (seed * np.cos(x) * y, seed * np.sin(x)))
        assert scope.consumer_uses == scope.buffers_created == 0


def test_density_only_and_traced_jk_do_not_route_registered_factors(monkeypatch):
    from pyscfad._src.implicit_diff import _implicit_diff_solve_matvec
    cb = registry()
    monkeypatch.setenv('PYSCFAD_DF_SHARED_COTANGENT', '1')
    mol = gto.Mole(atom='H 0 0 0; H 0 0 .74', basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    obj = df.DF(mol)
    obj._cderi = np.arange(9., dtype=np.float64).reshape(3, 3) / 10
    dm = np.array([[1.1, .2], [.2, .7]])
    seeds = (np.ones((2, 2)), np.ones((2, 2)))
    cb.register(obj._cderi)
    _, reference = jax.vjp(jk_reference, obj._cderi, dm)
    expected_eri, expected_dm = reference(seeds)
    with cb.reverse_scope() as scope, _implicit_diff_solve_matvec():
        obj_bar, dm_bar = _df_jk_opt.get_jk_bwd(1, True, True, 1e-13, (obj, dm), seeds)
        assert obj_bar is None
        np.testing.assert_allclose(dm_bar, expected_dm, atol=2e-11)
    assert scope.consumer_uses == 0
    def traced(jseed, kseed):
        return _df_jk_opt.get_jk_bwd(1, True, True, 1e-13, (obj, dm), (jseed, kseed))
    with cb.reverse_scope() as scope:
        obj_bar, dm_bar = jax.jit(traced)(*seeds)
    np.testing.assert_allclose(obj_bar._cderi, expected_eri, atol=2e-11)
    np.testing.assert_allclose(dm_bar, expected_dm, atol=2e-11)
    assert scope.consumer_uses == 0


def test_outcore_placeholder_keeps_explicit_geometry_inside_scope(tmp_path, monkeypatch):
    from pyscfad import config_update
    cb = registry()
    monkeypatch.setenv('PYSCFAD_DF_SHARED_COTANGENT', '1')
    mol = gto.Mole(atom='H 0 0 0; H 0 0 .74', basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    obj = df.DF(mol, auxbasis='weigend', incore=False)
    obj._cderi_to_save = str(tmp_path / 'shared-scope-ooc.h5')
    with config_update('pyscfad_moleintor_opt', True):
        obj.build()
    dm = np.array([[1.1, .2], [.2, .7]])
    def objective(m, a):
        view = df.DF(m, auxbasis='weigend')
        view.auxmol = a
        view.attach_outcore_cderi(obj._cderi_to_save)
        j, k = _df_jk_opt.get_jk(view, dm)
        return jnp.sum(j) + .3 * jnp.sum(k)
    with config_update('pyscfad_moleintor_opt', True):
        expected = jax.grad(objective, argnums=(0, 1))(mol, obj.auxmol)
        with cb.reverse_scope() as scope:
            actual = jax.grad(objective, argnums=(0, 1))(mol, obj.auxmol)
    assert_trees_close(actual, expected)
    assert scope.consumer_uses == scope.buffers_created == 0
